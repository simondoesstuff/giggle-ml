from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from giggleml.models.cmodel import CModel
from giggleml.utils.file_utils import Pathish


def to_bf16[T](tree: T) -> T:
    """Convert all float32 arrays in a T to bfloat16."""

    def convert(x: Array) -> Array:
        if eqx.is_inexact_array(x) and x.dtype == jnp.float32:
            return x.astype(jnp.bfloat16)
        return x

    return jax.tree.map(convert, tree, is_leaf=eqx.is_array)


def to_f32[T](tree: T) -> T:
    """Convert all bfloat16 arrays in a T to float32."""

    def convert(x: Array) -> Array:
        if eqx.is_inexact_array(x) and x.dtype == jnp.bfloat16:
            return x.astype(jnp.float32)
        return x

    return jax.tree.map(convert, tree, is_leaf=eqx.is_array)


# === Checkpointing ===


def save_checkpoint(model: object, path: Pathish) -> None:
    """Save model checkpoint to disk.

    Args:
        model: T to save.
        path: Path to save checkpoint.
    """
    eqx.tree_serialise_leaves(Path(path), model)


def load_checkpoint[T](path: Pathish, model_template: T) -> T:
    """Load model checkpoint from disk.

    Args:
        path: Path to checkpoint file.
        model_template: Model with same structure as saved model (for deserialization).

    Returns:
        Loaded T.
    """
    return eqx.tree_deserialise_leaves(Path(path), model_template)


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
