"""BED Contrastive Learning Training Module.

Contrastive learning training for CModel using weighted InfoNCE loss
where weights correspond to binned giggle similarity scores.

Architecture:
    BED Files (intervals) + HyenaDNA embeddings (zarr)
            ↓
        CModel (PerceiverIO-based)
            ↓
        BED-level embedding
            ↓
        Weighted InfoNCE Loss (edge types as weights)

Training uses bf16 precision and data parallel sharding across all GPUs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from numpy.typing import NDArray
from tqdm import tqdm

from giggleml.data.similarity_matrix import SimilarityMatrix
from giggleml.models.cmodel import CModel, create_cmodel
from giggleml.train.contrastive_data_loader import (
    ContrastiveBatch,
    ContrastiveDataLoader,
)
from giggleml.train.similarity_graph.similarity_graph import SimilarityGraph
from giggleml.utils.equinox import save_checkpoint, to_bf16, to_f32
from giggleml.utils.file_utils import Pathish
from giggleml.utils.terminal_plot import TerminalLossPlotter

# === Sharding Utilities ===


def create_device_mesh() -> Mesh:
    """Create a 1D device mesh across all available devices."""
    devices = jax.devices()
    return Mesh(np.array(devices), axis_names=("batch",))


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Create sharding spec for replicated data (model params)."""
    return NamedSharding(mesh, P())


def batch_sharding(mesh: Mesh) -> NamedSharding:
    """Create sharding spec for batch-sharded data."""
    return NamedSharding(mesh, P("batch"))


def shard_model(model: CModel, sharding: NamedSharding) -> CModel:
    """Shard model arrays while preserving non-array leaves (functions, static fields)."""
    arrays, non_arrays = eqx.partition(model, eqx.is_array)
    arrays = jax.device_put(arrays, sharding)
    return eqx.combine(arrays, non_arrays)


# === Batch Padding Utilities ===


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pad_to_multiple(n: int, divisor: int) -> int:
    """Round n up to the nearest multiple of divisor."""
    return ((n + divisor - 1) // divisor) * divisor


def pad_batch(
    embeddings_list: Sequence[NDArray[np.generic]],
    intervals_list: Sequence[NDArray[np.generic]],
    num_devices: int = 1,
) -> tuple[
    NDArray[np.generic],
    NDArray[np.int32],
    NDArray[np.bool_],
]:
    """Pad variable-length BED data to next power-of-two length for batched processing.

    Takes numpy arrays (host memory) and returns numpy arrays. Caller should use
    jax.device_put() to transfer to device with appropriate sharding - this allows
    a single host->device transfer without intermediate GPU allocations.

    Padding to powers of two enables better JIT cache reuse and allows XLA to
    apply more automatic optimizations. Batch dimension is padded to be divisible
    by num_devices for sharding.

    Args:
        embeddings_list: List of numpy embedding arrays, each (n_i, edim).
        intervals_list: List of numpy interval arrays, each (n_i, 3).
        num_devices: Number of devices for batch sharding. Batch size will be
            padded to a multiple of this.

    Returns:
        Tuple of numpy arrays (caller transfers to device):
        - Padded embeddings: (batch, pad_len, edim) where pad_len is next power of 2
        - Padded intervals: (batch, pad_len, 3)
        - Mask: (batch, pad_len, 2) where True = masked (padded position)
    """
    real_batch_size = len(embeddings_list)
    batch_size = _pad_to_multiple(real_batch_size, num_devices)
    max_len = max(emb.shape[0] for emb in embeddings_list)
    pad_len = _next_power_of_two(max_len)
    edim = embeddings_list[0].shape[1]

    # Build padded arrays in numpy (CPU) - caller transfers to device
    padded_emb = np.zeros((batch_size, pad_len, edim), dtype=embeddings_list[0].dtype)
    padded_ivs = np.zeros((batch_size, pad_len, 3), dtype=np.int32)
    # Mask: True = masked/padded, starts all True
    mask = np.ones((batch_size, pad_len, 2), dtype=np.bool_)

    # Fill in actual data and unmask valid positions
    for i, (emb, ivs) in enumerate(zip(embeddings_list, intervals_list)):
        n = emb.shape[0]
        padded_emb[i, :n] = emb
        padded_ivs[i, :n] = ivs
        mask[i, :n] = False

    return padded_emb, padded_ivs, mask


def pad_adjacency(
    adjacency: NDArray[np.int32],
    num_devices: int = 1,
) -> NDArray[np.int32]:
    """Pad adjacency matrix batch dimension to be divisible by num_devices.

    Returns numpy array - caller transfers to device via device_put.
    Padded entries are set to 0 (no edge), so they contribute 0 weight to loss.
    """
    real_batch = adjacency.shape[0]
    padded_batch = _pad_to_multiple(real_batch, num_devices)
    if padded_batch == real_batch:
        return adjacency
    padded = np.zeros((padded_batch, padded_batch), dtype=np.int32)
    padded[:real_batch, :real_batch] = adjacency
    return padded


def apply_input_dropout(
    mask: Bool[Array, "batch pad_len 2"],
    p_seq: float,
    p_interval: float,
    p_both: float,
    key: PRNGKeyArray,
) -> Bool[Array, "batch pad_len 2"]:
    """Apply random input dropout to non-padded positions.

    Input dropout is a data augmentation strategy that randomly masks inputs,
    forcing the model to learn robust representations. Three independent dropout
    modes can be combined:

    - p_seq: Probability of masking only the sequence embedding (interval preserved).
        The seq embedding is replaced with a learned mask embedding.
    - p_interval: Probability of masking only the interval encoding (seq preserved).
        The interval encoding is replaced with a learned mask embedding.
    - p_both: Probability of masking both seq and interval (position excluded).
        The position is removed from cross-attention entirely.

    These are applied independently per position. Positions already masked (padding)
    remain masked. The probabilities are applied in order: first p_both, then p_seq
    and p_interval on remaining positions.

    Args:
        mask: Current mask of shape (batch, pad_len, 2). True = masked.
            Padding positions have both columns True.
        p_seq: Probability of masking only sequence embedding.
        p_interval: Probability of masking only interval encoding.
        p_both: Probability of masking both (excluding position).
        key: JAX PRNG key for reproducibility.

    Returns:
        Updated mask with random dropout applied to non-padded positions.
    """
    if p_seq == 0.0 and p_interval == 0.0 and p_both == 0.0:
        return mask

    batch, pad_len, _ = mask.shape
    key_both, key_seq, key_interval = jax.random.split(key, 3)

    # Identify non-padded positions (both columns False = valid position)
    is_padding = mask[:, :, 0] & mask[:, :, 1]  # (batch, pad_len)
    is_valid = ~is_padding

    # Generate random values for dropout decisions
    rand_both = jax.random.uniform(key_both, (batch, pad_len))
    rand_seq = jax.random.uniform(key_seq, (batch, pad_len))
    rand_interval = jax.random.uniform(key_interval, (batch, pad_len))

    # Apply p_both: mask both columns for these positions
    drop_both = is_valid & (rand_both < p_both)

    # Apply p_seq and p_interval to positions not dropped by p_both
    remaining = is_valid & ~drop_both
    drop_seq_only = remaining & (rand_seq < p_seq)
    drop_interval_only = remaining & (rand_interval < p_interval)

    # Build new mask
    # Start with original mask
    new_seq_mask = mask[:, :, 0]
    new_interval_mask = mask[:, :, 1]

    # Apply drop_both: set both columns to True
    new_seq_mask = new_seq_mask | drop_both
    new_interval_mask = new_interval_mask | drop_both

    # Apply drop_seq_only: set only seq column to True
    new_seq_mask = new_seq_mask | drop_seq_only

    # Apply drop_interval_only: set only interval column to True
    new_interval_mask = new_interval_mask | drop_interval_only

    return jnp.stack([new_seq_mask, new_interval_mask], axis=-1)


# === Configuration ===


@dataclass
class ContrastiveTrainingConfig:
    """Configuration for contrastive training of CModel on BED files.

    Attributes:
        seq_dim: HyenaDNA embedding dimension (128 for tiny, 256 for others).
        latent_dim: Latent space dimension in CModel.
        num_latents: Number of latent vectors in CModel.
        shared_per_stack: Number of unique encoder blocks per weight sharing unit.
        num_stacks: Number of times to repeat the shared blocks.
        num_heads: Number of attention heads.
        output_dim: Embedding space dimension for contrastive loss.

        learning_rate: Peak learning rate after warmup.
        weight_decay: AdamW weight decay.
        warmup_steps: Number of warmup steps for LR schedule.
        total_steps: Total training steps.
        temperature: Temperature for softmax in InfoNCE loss.

        bin_thresholds: Thresholds defining similarity bins. Values below t[0] have
            no edge. A value is in bin i if t[i] <= value < t[i+1].
        bin_weights: Weights per bin (length must match thresholds). Lowest bin
            often has weight 0.0 (treating weakly-similar pairs as negatives).

        num_anchors: Number of anchor nodes to sample per batch.
        neighbors_per_anchor: Number of neighbors to sample per anchor.
        max_intervals: Maximum intervals per BED file. Files exceeding this are
            randomly downsampled per-batch. None disables capping.

        embedding_dir: Directory containing zarr arrays of HyenaDNA embeddings.
        bed_dir: Directory containing .bed.gz files.
    """

    # Model
    seq_dim: int = 128
    latent_dim: int = 512
    num_latents: int = 1024
    shared_per_stack: int = 2
    num_stacks: int = 3
    num_heads: int = 8
    output_dim: int = 128
    dropout_rate: float = 0.1
    cross_attn_chunk_size: int = 1024

    # Training
    peak_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 100_000
    temperature: float = 0.07

    # Input dropout (data augmentation)
    # Probabilities for masking seq embeddings and/or interval encodings per position.
    # When both are masked, the position is excluded from cross-attention.
    # When only one is masked, it's replaced with a learned mask embedding.
    input_dropout_seq: float = 0.0  # P(mask seq embedding)
    input_dropout_interval: float = 0.0  # P(mask interval encoding)
    input_dropout_both: float = 0.0  # P(mask both, excluding position)

    # Similarity graph binning
    bin_thresholds: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
    bin_weights: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)

    # Batch (community subgraph)
    num_anchors: int = 16
    neighbors_per_anchor: int = 4
    max_intervals: int | None = None

    # Data paths
    embedding_dir: Pathish = field(default_factory=lambda: Path("."))
    bed_dir: Pathish = field(default_factory=lambda: Path("."))
    memmap_dir: Pathish | None = None


# === Loss Function ===


def weighted_infonce_loss(
    embeddings: Float[Array, "batch edim"],
    adjacency: Int[Array, "batch batch"],
    edge_type_weights: Float[Array, "num_types"],
    temperature: float = 0.07,
) -> Float[Array, ""]:
    """Compute weighted InfoNCE loss for contrastive learning.

    Pairs with positive edge weights are treated as positives, pairs with
    weight 0.0 (including lowest bin) are effectively negatives.

    Args:
        embeddings: L2-normalized embeddings of shape (batch, edim).
        adjacency: Adjacency matrix where entry [i, j] is edge type + 1 (0 = no edge).
        edge_type_weights: Weight for each edge type (indexed by edge_type - 1).
        temperature: Temperature for softmax scaling.

    Returns:
        Scalar weighted InfoNCE loss.
    """
    # L2 normalize for cosine similarity
    embeddings = embeddings / (
        jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8
    )

    # Pairwise similarities / temperature
    logits = einx.dot("i d, j d -> i j", embeddings, embeddings) / temperature

    # Self-mask (exclude diagonal)
    n = embeddings.shape[0]
    self_mask = ~jnp.eye(n, dtype=bool)

    # Denominator: log-sum-exp over all non-self pairs
    masked_logits = jnp.where(self_mask, logits, -jnp.inf)
    log_denom = jax.scipy.special.logsumexp(masked_logits, axis=-1)

    # Weight lookup: adjacency k -> weights[k-1], adjacency 0 -> 0.0
    # Pairs with weight 0.0 (including lowest bin) are effectively negatives
    weights = jnp.where(
        adjacency > 0,
        edge_type_weights[adjacency - 1],
        0.0,
    )

    # Per-pair loss weighted by edge type
    per_pair_loss = -(logits - log_denom[:, None])
    weighted_loss = (weights * per_pair_loss * self_mask).sum()

    return weighted_loss / (weights.sum() + 1e-8)


# === Training ===


def _subset_similarity_matrix(
    similarity_matrix: SimilarityMatrix,
    indices: list[int],
) -> NDArray[np.floating]:
    """Extract a submatrix from similarity matrix for given indices.

    Args:
        similarity_matrix: Full pairwise similarity matrix.
        indices: Node indices to include in subset.

    Returns:
        Submatrix of shape (len(indices), len(indices)).
    """
    arr = similarity_matrix.array.astype(np.float32)
    return arr[np.ix_(indices, indices)]


def create_optimizer(config: ContrastiveTrainingConfig) -> optax.GradientTransformation:
    """Create optimizer with warmup cosine decay schedule.

    Args:
        config: Training configuration.

    Returns:
        Optax gradient transformation (optimizer).
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.peak_learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps - config.warmup_steps,
        end_value=config.peak_learning_rate * 0.01,
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay),
    )


def _single_forward(
    model: CModel,
    embeddings: Float[Array, "max_len edim"],
    intervals: Int[Array, "max_len 3"],
    mask: Bool[Array, "max_len 2"],
    key: PRNGKeyArray,
) -> Float[Array, "output_dim"]:
    """Forward pass for a single BED file with checkpointing."""

    @jax.checkpoint
    def forward(
        emb: Float[Array, "max_len edim"],
        ivs: Int[Array, "max_len 3"],
        m: Bool[Array, "max_len 2"],
        k: PRNGKeyArray,
    ) -> Float[Array, "output_dim"]:
        return model(emb, ivs, mask=m, key=k)

    return forward(embeddings, intervals, mask, key)


def _batched_forward(
    model: CModel,
    embeddings: Float[Array, "batch max_len edim"],
    intervals: Int[Array, "batch max_len 3"],
    mask: Bool[Array, "batch max_len 2"],
    keys: PRNGKeyArray,
) -> Float[Array, "batch output_dim"]:
    """Batched forward pass using vmap - processes all BED files in parallel."""
    return jax.vmap(lambda emb, ivs, m, k: _single_forward(model, emb, ivs, m, k))(
        embeddings, intervals, mask, keys
    )


@eqx.filter_jit
def train_step(
    model: CModel,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    padded_embeddings: Float[Array, "batch max_len edim"],
    padded_intervals: Int[Array, "batch max_len 3"],
    mask: Bool[Array, "batch max_len 2"],
    adjacency: Int[Array, "batch batch"],
    edge_type_weights: Float[Array, "num_types"],
    temperature: float,
    key: PRNGKeyArray,
    input_dropout_seq: float = 0.0,
    input_dropout_interval: float = 0.0,
    input_dropout_both: float = 0.0,
) -> tuple[CModel, optax.OptState, Float[Array, ""]]:
    """Perform a data-parallel training step with gradient checkpointing.

    When inputs are sharded across devices (via device_put with batch_sharding),
    computation is automatically distributed. Each device processes its portion
    of the batch, then embeddings are all-gathered for pairwise loss computation.

    Args:
        model: CModel to train (bf16 weights, replicated across devices).
        opt_state: Optimizer state (replicated).
        optimizer: Optax optimizer.
        padded_embeddings: Padded embeddings (batch, max_len, edim), sharded on batch.
        padded_intervals: Padded intervals (batch, max_len, 3), sharded on batch.
        mask: Padding mask (batch, max_len, 2), sharded on batch. True = masked.
        adjacency: Adjacency matrix (batch, batch), replicated.
        edge_type_weights: Edge type weights, replicated.
        temperature: InfoNCE temperature.
        key: PRNG key for dropout.

    Returns:
        Tuple of (updated_model, updated_opt_state, loss).
    """
    batch_size = padded_embeddings.shape[0]
    key, input_dropout_key = jax.random.split(key)
    dropout_keys = jax.random.split(key, batch_size)

    # Apply input dropout inside JIT to avoid memory leaks from traced ops outside JIT
    mask = apply_input_dropout(
        mask, input_dropout_seq, input_dropout_interval, input_dropout_both, input_dropout_key
    )

    def loss_fn(model: CModel) -> Float[Array, ""]:
        # Forward pass - vmap over batch, sharding distributes across devices
        batch_embeddings = _batched_forward(
            model, padded_embeddings, padded_intervals, mask, dropout_keys
        )
        # Cast to f32 for stable loss computation
        batch_embeddings_f32 = batch_embeddings.astype(jnp.float32)
        return weighted_infonce_loss(
            batch_embeddings_f32, adjacency, edge_type_weights, temperature
        )

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)  # pyright: ignore[reportArgumentType]
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


def train_step_unpadded(
    model: CModel,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    embeddings_batch: Sequence[NDArray[np.generic]],
    intervals_batch: Sequence[NDArray[np.generic]],
    adjacency: Int[Array, "batch batch"],
    edge_type_weights: Float[Array, "num_types"],
    temperature: float,
    key: PRNGKeyArray,
) -> tuple[CModel, optax.OptState, Float[Array, ""]]:
    """Convenience wrapper that pads inputs before calling train_step.

    For testing or when inputs aren't pre-padded.
    """
    padded_emb, padded_ivs, mask = pad_batch(embeddings_batch, intervals_batch)
    return train_step(
        model,
        opt_state,
        optimizer,
        jnp.asarray(padded_emb),
        jnp.asarray(padded_ivs),
        jnp.asarray(mask),
        adjacency,
        edge_type_weights,
        temperature,
        key,
    )


@eqx.filter_jit
def val_step(
    model: CModel,
    padded_embeddings: Float[Array, "batch max_len edim"],
    padded_intervals: Int[Array, "batch max_len 3"],
    mask: Bool[Array, "batch max_len 2"],
    adjacency: Int[Array, "batch batch"],
    edge_type_weights: Float[Array, "num_types"],
    temperature: float,
) -> Float[Array, ""]:
    """Compute validation loss without gradients or dropout.

    Args:
        model: CModel (in inference mode, no dropout).
        padded_embeddings: Padded embeddings (batch, max_len, edim).
        padded_intervals: Padded intervals (batch, max_len, 3).
        mask: Padding mask (batch, max_len, 2). True = masked.
        adjacency: Adjacency matrix (batch, batch).
        edge_type_weights: Edge type weights.
        temperature: InfoNCE temperature.

    Returns:
        Scalar validation loss.
    """
    batch_size = padded_embeddings.shape[0]
    # No dropout during validation - pass None keys
    dummy_keys = jax.random.split(jax.random.key(0), batch_size)

    # Forward pass with inference=True (no dropout)
    model_inf = eqx.nn.inference_mode(model)
    batch_embeddings = _batched_forward(
        model_inf, padded_embeddings, padded_intervals, mask, dummy_keys
    )
    batch_embeddings_f32 = batch_embeddings.astype(jnp.float32)
    return weighted_infonce_loss(
        batch_embeddings_f32, adjacency, edge_type_weights, temperature
    )


def _extract_batch_data(
    batch: ContrastiveBatch,
) -> tuple[list[NDArray[np.generic]], list[NDArray[np.generic]]]:
    """Extract embeddings and intervals lists from batch.

    Returns numpy arrays (host memory) which are converted to JAX in pad_batch.
    """
    embeddings = [bd.embeddings for bd in batch.bed_data]
    intervals = [bd.intervals for bd in batch.bed_data]
    return embeddings, intervals


def train(
    config: ContrastiveTrainingConfig,
    similarity_matrix: SimilarityMatrix,
    bed_names: list[str],
    key: PRNGKeyArray,
    *,
    train_indices: list[int] | None = None,
    val_indices: list[int] | None = None,
    log_every: int = 100,
    val_every: int = 500,
    checkpoint_every: int | None = None,
    checkpoint_dir: Path | None = None,
    plot_loss: bool = False,
) -> CModel:
    """Train CModel with contrastive learning using bf16 and data parallelism.

    Uses bfloat16 precision for forward/backward passes and replicates the model
    across all available devices for data parallel training.

    Args:
        config: Training configuration.
        similarity_matrix: Pairwise similarity matrix between BED files.
        bed_names: Ordered list of BED file names matching graph node indices.
            Should not include suffix.
        key: JAX PRNG key.
        train_indices: Indices into bed_names for training set. If None, uses all.
        val_indices: Indices into bed_names for validation set. If None, skips validation.
        log_every: Log loss every N steps.
        val_every: Compute validation loss every N steps (requires val_indices).
        checkpoint_every: Save checkpoint every N steps (None to disable).
        checkpoint_dir: Directory to save checkpoints (required if checkpoint_every is set).
        plot_loss: If True, display live loss plot in terminal using plotext.

    Returns:
        Trained CModel (in f32).
    """
    key, model_key, data_key, train_key, val_key = jax.random.split(key, 5)

    # Set up device mesh for data parallelism
    mesh = create_device_mesh()
    replicate = replicated_sharding(mesh)
    num_devices = len(jax.devices())
    print(f"Training with {num_devices} device(s), bf16 precision")

    # Handle train/val split
    # bed_names must be sorted to match memmap ordering
    bed_names = sorted(bed_names)
    if train_indices is None:
        train_indices = list(range(len(bed_names)))

    if val_indices is not None:
        print(f"Train/val split: {len(train_indices)} train, {len(val_indices)} val")

    # Build similarity graph from train subset
    train_sim = _subset_similarity_matrix(similarity_matrix, train_indices)
    graph = SimilarityGraph(train_sim, list(config.bin_thresholds))

    # Create model with interval encoding mode
    model = create_cmodel(
        seq_dim=config.seq_dim,
        latent_dim=config.latent_dim,
        num_latents=config.num_latents,
        shared_per_stack=config.shared_per_stack,
        num_stacks=config.num_stacks,
        num_heads=config.num_heads,
        output_dim=config.output_dim,
        dropout_rate=config.dropout_rate,
        pooling="decode",
        cross_attn_chunk_size=config.cross_attn_chunk_size,
        cross_attn_checkpoint=True,
        key=model_key,
    )

    # Filter for only floating-point arrays (inexact arrays)
    trainable_parts = eqx.filter(model, eqx.is_inexact_array)
    trainable_params = sum(x.size for x in jax.tree_util.tree_leaves(trainable_parts))
    print(f"Built CModel, Trainable parameters: {trainable_params:,}")

    # Convert model to bf16 and replicate across devices
    model = to_bf16(model)
    model = shard_model(model, replicate)

    # Create optimizer (operates on bf16 params)
    optimizer = create_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    opt_state = jax.device_put(opt_state, replicate)

    # Edge type weights (replicated)
    edge_type_weights = jnp.array(config.bin_weights, dtype=jnp.float32)
    edge_type_weights = jax.device_put(edge_type_weights, replicate)

    # Create training data loader
    # Pass full bed_names with index_map for memmap compatibility
    train_loader = ContrastiveDataLoader(
        graph=graph,
        bed_names=bed_names,
        embedding_dir=config.embedding_dir,
        bed_dir=config.bed_dir,
        num_anchors=config.num_anchors,
        neighbors_per_anchor=config.neighbors_per_anchor,
        max_intervals=config.max_intervals,
        preload=True,
        memmap_dir=config.memmap_dir,
        index_map=train_indices,
    )

    # Create validation data loader if val_indices provided
    val_loader: ContrastiveDataLoader | None = None
    if val_indices is not None:
        val_sim = _subset_similarity_matrix(similarity_matrix, val_indices)
        val_graph = SimilarityGraph(val_sim, list(config.bin_thresholds))
        val_loader = ContrastiveDataLoader(
            graph=val_graph,
            bed_names=bed_names,
            embedding_dir=config.embedding_dir,
            bed_dir=config.bed_dir,
            num_anchors=config.num_anchors,
            neighbors_per_anchor=config.neighbors_per_anchor,
            max_intervals=config.max_intervals,
            preload=True,
            memmap_dir=config.memmap_dir,
            index_map=val_indices,
        )

    # Sharding specs for batched data
    batch_shard = batch_sharding(mesh)

    # Initialize loss plotter if enabled
    plotter: TerminalLossPlotter | None = None
    if plot_loss:
        plotter = TerminalLossPlotter(title="Contrastive Training Loss")

    # Training loop
    batch_iter = train_loader.iter_batches(data_key)
    val_batch_iter = val_loader.iter_batches(val_key) if val_loader else None

    for step in tqdm(range(config.total_steps), desc="Training"):
        train_key, step_key = jax.random.split(train_key)
        batch = next(batch_iter)
        embeddings_batch, intervals_batch = _extract_batch_data(batch)

        # Pad to uniform length and shard across devices
        padded_emb, padded_ivs, mask = pad_batch(
            embeddings_batch, intervals_batch, num_devices
        )
        padded_emb = jax.device_put(padded_emb, batch_shard)
        padded_ivs = jax.device_put(padded_ivs, batch_shard)
        mask = jax.device_put(mask, batch_shard)

        # Adjacency stays replicated (needed for full pairwise loss)
        # Pad to match batch dimension (padded entries have 0 edge weight)
        adjacency = pad_adjacency(batch.adjacency, num_devices)
        adjacency = jax.device_put(adjacency, replicate)

        model, opt_state, loss = train_step(
            model,
            opt_state,
            optimizer,
            padded_emb,
            padded_ivs,
            mask,
            adjacency,
            edge_type_weights,
            config.temperature,
            step_key,
            config.input_dropout_seq,
            config.input_dropout_interval,
            config.input_dropout_both,
        )

        if step % log_every == 0:
            tqdm.write(f"Step {step}: train_loss = {float(loss):.4f}")
            if plotter is not None:
                plotter.add_train_loss(step, float(loss))
                plotter.plot()

        # Validation
        if val_batch_iter is not None and step % val_every == 0:
            val_batch = next(val_batch_iter)
            val_emb, val_ivs = _extract_batch_data(val_batch)
            padded_val_emb, padded_val_ivs, val_mask = pad_batch(
                val_emb, val_ivs, num_devices
            )
            padded_val_emb = jax.device_put(padded_val_emb, batch_shard)
            padded_val_ivs = jax.device_put(padded_val_ivs, batch_shard)
            val_mask = jax.device_put(val_mask, batch_shard)
            val_adjacency = pad_adjacency(val_batch.adjacency, num_devices)
            val_adjacency = jax.device_put(val_adjacency, replicate)

            val_loss = val_step(
                model,
                padded_val_emb,
                padded_val_ivs,
                val_mask,
                val_adjacency,
                edge_type_weights,
                config.temperature,
            )
            tqdm.write(f"Step {step}: val_loss = {float(val_loss):.4f}")
            if plotter is not None:
                plotter.add_val_loss(step, float(val_loss))
                plotter.plot()

        if checkpoint_every and checkpoint_dir and (step + 1) % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"model_step_{step + 1}.eqx"
            # Convert back to f32 for checkpointing
            save_checkpoint(to_f32(model), ckpt_path)
            tqdm.write(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint (in f32)
    if checkpoint_dir:
        final_path = checkpoint_dir / f"model_step_{config.total_steps}.eqx"
        if not final_path.exists():
            save_checkpoint(to_f32(model), final_path)
            tqdm.write(f"Saved final checkpoint: {final_path}")

    # Return model in f32
    return to_f32(model)
