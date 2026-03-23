"""Inference utilities for CModel.

Provides efficient batched inference with proper memory management.
Shared utilities for both training and inference.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from numpy.typing import NDArray

from giggleml.models.cmodel import CModel
from giggleml.train.contrastive_data_loader import BedFileData
from giggleml.utils.equinox import batch_sharding, create_device_mesh

# === Batch Padding Utilities ===


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def pad_to_multiple(n: int, divisor: int) -> int:
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
    batch_size = pad_to_multiple(real_batch_size, num_devices)
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


# === Forward Pass Utilities ===


@eqx.filter_jit
def batched_forward(
    model: CModel,
    embeddings: Float[Array, "batch max_len edim"],
    intervals: Int[Array, "batch max_len 3"],
    mask: Bool[Array, "batch max_len 2"],
    keys: PRNGKeyArray,
) -> Float[Array, "batch output_dim"]:
    """Batched forward pass using vmap.

    Processes all items in the batch in parallel via jax.vmap.

    Args:
        model: CModel to apply.
        embeddings: Padded embeddings (batch, max_len, edim).
        intervals: Padded intervals (batch, max_len, 3).
        mask: Padding mask (batch, max_len, 2). True = masked.
        keys: PRNG keys for each item in batch.

    Returns:
        Batch embeddings of shape (batch, output_dim).
    """

    def forward_one(
        emb: Float[Array, "max_len edim"],
        ivs: Int[Array, "max_len 3"],
        m: Bool[Array, "max_len 2"],
        k: PRNGKeyArray,
    ) -> Float[Array, "output_dim"]:
        return model(emb, ivs, mask=m, key=k)

    return jax.vmap(forward_one)(embeddings, intervals, mask, keys)


# === High-Level Inference API ===


def embed_batch(
    model: CModel,
    embeddings_list: Sequence[NDArray[np.generic]],
    intervals_list: Sequence[NDArray[np.generic]],
    shard: jax.NamedSharding | None = None,
    num_devices: int | None = None,
) -> Float[Array, "batch output_dim"]:
    """Embed a batch of BED files. Handles padding internally.

    Runs inference on a batch of BED files, padding to uniform length
    and applying the model in parallel via vmap.

    Args:
        model: CModel in inference mode (should have dropout disabled).
        embeddings_list: List of embedding arrays, each (n_i, seq_dim).
        intervals_list: List of interval arrays, each (n_i, 3).

    Returns:
        Batch embeddings of shape (batch_size, output_dim).
    """

    real_batch_size = len(embeddings_list)
    num_devices = len(jax.devices()) if num_devices is None else num_devices

    # Pad to uniform length with device alignment
    padded_emb, padded_ivs, mask = pad_batch(
        embeddings_list, intervals_list, num_devices
    )

    if shard is None:
        # Apply sharding to distribute the batch across GPUs
        mesh = create_device_mesh()
        shard = batch_sharding(mesh)

    padded_emb_jax = jax.device_put(jnp.asarray(padded_emb), shard)
    padded_ivs_jax = jax.device_put(jnp.asarray(padded_ivs), shard)
    mask_jax = jax.device_put(jnp.asarray(mask), shard)

    batch_size = padded_emb_jax.shape[0]
    dummy_keys = jax.random.split(jax.random.key(0), batch_size)

    embeddings = batched_forward(
        model, padded_emb_jax, padded_ivs_jax, mask_jax, dummy_keys
    )

    return embeddings[:real_batch_size]


def embed_dataset(
    model: CModel,
    bed_data: list[BedFileData],
    batch_size: int = 64,
) -> Float[Array, "n_files output_dim"]:
    """Embed entire dataset efficiently in batches.

    Processes all BED files in the dataset in batches for memory efficiency.

    Args:
        model: CModel (will be put in inference mode).
        bed_data: List of BedFileData objects.
        batch_size: Number of files to process per batch.

    Returns:
        Embeddings for all files, shape (n_files, output_dim).
    """
    model = eqx.nn.inference_mode(model)

    mesh = create_device_mesh()
    shard = batch_sharding(mesh)

    all_embeddings = []
    n_files = len(bed_data)

    for start in range(0, n_files, batch_size):
        end = min(start + batch_size, n_files)
        batch_bed_data = bed_data[start:end]

        # Extract embeddings and intervals
        embeddings_list = [bd.embeddings for bd in batch_bed_data]
        intervals_list = [bd.intervals for bd in batch_bed_data]

        # Embed batch
        batch_embeddings = embed_batch(
            model,
            embeddings_list,
            intervals_list,
            shard=shard,
        )
        all_embeddings.append(batch_embeddings)

    # Concatenate all batches
    return jnp.concatenate(all_embeddings, axis=0)
