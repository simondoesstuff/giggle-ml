from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

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
