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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray
from tqdm import tqdm

from giggleml.data.similarity_matrix import SimilarityMatrix
from giggleml.models.cmodel import CModel, create_cmodel
from giggleml.train.contrastive_data_loader import (
    ContrastiveBatch,
    ContrastiveDataLoader,
)
from giggleml.train.similarity_graph.similarity_graph import SimilarityGraph
from giggleml.utils.file_utils import Pathish

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

    # Data paths
    embedding_dir: Pathish = field(default_factory=lambda: Path("."))
    bed_dir: Pathish = field(default_factory=lambda: Path("."))


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


@eqx.filter_jit
def train_step(
    model: CModel,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    embeddings_batch: list[Float[Array, "n edim"]],
    intervals_batch: list[Int[Array, "n 3"]],
    adjacency: Int[Array, "batch batch"],
    edge_type_weights: Float[Array, "num_types"],
    temperature: float,
    key: PRNGKeyArray,
) -> tuple[CModel, optax.OptState, Float[Array, ""]]:
    """Perform a single training step.

    Args:
        model: CModel to train.
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        embeddings_batch: List of embedding arrays for each BED file in batch.
        intervals_batch: List of interval arrays for each BED file in batch.
        adjacency: Adjacency matrix with edge types.
        edge_type_weights: Weight for each edge type.
        temperature: InfoNCE temperature.
        key: PRNG key for dropout.

    Returns:
        Tuple of (updated_model, updated_opt_state, loss).
    """
    # Split keys for each sample in batch
    batch_size = len(embeddings_batch)
    dropout_keys = jax.random.split(key, batch_size)

    def loss_fn(model: CModel) -> Float[Array, ""]:
        # Process each BED file through CModel with dropout
        batch_embeddings = jnp.stack(
            [
                model(emb, ivs, key=k)
                for emb, ivs, k in zip(embeddings_batch, intervals_batch, dropout_keys)
            ]
        )
        return weighted_infonce_loss(
            batch_embeddings, adjacency, edge_type_weights, temperature
        )

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)  # pyright: ignore[reportArgumentType]
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


def _extract_batch_data(
    batch: ContrastiveBatch,
) -> tuple[list[Float[Array, "n edim"]], list[Int[Array, "n 3"]]]:
    """Extract embeddings and intervals lists from batch (for JIT compatibility)."""
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
    """Train CModel with contrastive learning.

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
        Trained CModel.
    """
    key, model_key, data_key, train_key = jax.random.split(key, 4)

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

    # Create optimizer
    optimizer = create_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Edge type weights from config
    edge_type_weights = jnp.array(config.bin_weights, dtype=jnp.float32)

    # Create data loader
    data_loader = ContrastiveDataLoader(
        graph=graph,
        bed_names=bed_names,
        embedding_dir=config.embedding_dir,
        bed_dir=config.bed_dir,
        num_anchors=config.num_anchors,
        neighbors_per_anchor=config.neighbors_per_anchor,
        preload=False,
    )

    # Training loop
    batch_iter = data_loader.iter_batches(data_key)

    for step in tqdm(range(config.total_steps), desc="Training"):
        train_key, step_key = jax.random.split(train_key)
        batch = next(batch_iter)
        embeddings_batch, intervals_batch = _extract_batch_data(batch)

        model, opt_state, loss = train_step(
            model,
            opt_state,
            optimizer,
            embeddings_batch,
            intervals_batch,
            batch.adjacency,
            edge_type_weights,
            config.temperature,
            step_key,
        )

        if step % log_every == 0:
            tqdm.write(f"Step {step}: loss = {float(loss):.4f}")

        if checkpoint_every and checkpoint_dir and (step + 1) % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"model_step_{step + 1}.eqx"
            save_checkpoint(model, ckpt_path)
            tqdm.write(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    if checkpoint_dir:
        final_path = checkpoint_dir / f"model_step_{config.total_steps}.eqx"
        if not final_path.exists():
            save_checkpoint(model, final_path)
            tqdm.write(f"Saved final checkpoint: {final_path}")

    return model


# === Checkpointing ===


def save_checkpoint(model: CModel, path: Path) -> None:
    """Save model checkpoint to disk.

    Args:
        model: CModel to save.
        path: Path to save checkpoint.
    """
    eqx.tree_serialise_leaves(path, model)


def load_checkpoint(path: Path, model_template: CModel) -> CModel:
    """Load model checkpoint from disk.

    Args:
        path: Path to checkpoint file.
        model_template: Model with same structure as saved model (for deserialization).

    Returns:
        Loaded CModel.
    """
    return eqx.tree_deserialise_leaves(path, model_template)
