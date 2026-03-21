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

    # Training
    peak_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 100_000
    temperature: float = 0.07

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
    dropout_keys = jax.random.split(key, batch_size)

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
    log_every: int = 100,
    checkpoint_every: int | None = None,
    checkpoint_dir: Path | None = None,
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
        log_every: Log loss every N steps.
        checkpoint_every: Save checkpoint every N steps (None to disable).
        checkpoint_dir: Directory to save checkpoints (required if checkpoint_every is set).

    Returns:
        Trained CModel (in f32).
    """
    key, model_key, data_key, train_key = jax.random.split(key, 4)

    # Set up device mesh for data parallelism
    mesh = create_device_mesh()
    replicate = replicated_sharding(mesh)
    num_devices = len(jax.devices())
    print(f"Training with {num_devices} device(s), bf16 precision")

    # Build similarity graph from matrix + config thresholds
    graph = SimilarityGraph(
        similarity_matrix.array.astype(np.float32),
        list(config.bin_thresholds),
    )

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

    # Create data loader
    data_loader = ContrastiveDataLoader(
        graph=graph,
        bed_names=bed_names,
        embedding_dir=config.embedding_dir,
        bed_dir=config.bed_dir,
        num_anchors=config.num_anchors,
        neighbors_per_anchor=config.neighbors_per_anchor,
        max_intervals=config.max_intervals,
        preload=True,
        memmap_dir=config.memmap_dir,
    )

    # Sharding specs for batched data
    batch_shard = batch_sharding(mesh)

    # Training loop
    batch_iter = data_loader.iter_batches(data_key)

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
        )

        if step % log_every == 0:
            tqdm.write(f"Step {step}: loss = {float(loss):.4f}")

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
