from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray

from giggleml.utils.file_utils import Pathish


@dataclass
class TrainState[T]:
    """Complete training state for resumable training.

    Attributes:
        step: Current training step (0-indexed, represents completed steps).
        model: The model being trained (in f32 for serialization).
        opt_state: Optimizer state.
        train_key: PRNG key for training randomness (dropout, etc).
        data_key: PRNG key for data iteration.
    """

    step: int
    model: T
    opt_state: optax.OptState
    train_key: PRNGKeyArray
    data_key: PRNGKeyArray


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


def save_train_state[T](state: TrainState[T], directory: Pathish) -> None:
    """Save complete training state for resumable training.

    Creates a directory containing:
    - model.eqx: Model weights (f32)
    - opt_state.eqx: Optimizer state
    - metadata.json: Step count and PRNG keys

    Args:
        state: Training state to save.
        directory: Directory to save state files.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Save model (in f32)
    model_f32 = to_f32(state.model)
    eqx.tree_serialise_leaves(dir_path / "model.eqx", model_f32)

    # Save optimizer state
    eqx.tree_serialise_leaves(dir_path / "opt_state.eqx", state.opt_state)

    # Save metadata (step and keys)
    # Convert keys to serializable format (raw data as list)
    metadata: dict[str, Any] = {
        "step": state.step,
        "train_key": jax.random.key_data(state.train_key).tolist(),
        "data_key": jax.random.key_data(state.data_key).tolist(),
    }
    with open(dir_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_train_state[T](
    directory: Pathish,
    model_template: T,
    opt_state_template: optax.OptState,
) -> TrainState[T]:
    """Load complete training state for resuming training.

    Args:
        directory: Directory containing state files.
        model_template: Model with same structure as saved model.
        opt_state_template: Optimizer state with same structure as saved state.

    Returns:
        Loaded TrainState ready for resuming training.
    """
    dir_path = Path(directory)

    # Load model
    model = eqx.tree_deserialise_leaves(dir_path / "model.eqx", model_template)

    # Load optimizer state
    opt_state = eqx.tree_deserialise_leaves(
        dir_path / "opt_state.eqx", opt_state_template
    )

    # Load metadata
    with open(dir_path / "metadata.json") as f:
        metadata = json.load(f)

    step = metadata["step"]
    train_key = jax.random.wrap_key_data(
        jnp.array(metadata["train_key"], dtype=jnp.uint32)
    )
    data_key = jax.random.wrap_key_data(
        jnp.array(metadata["data_key"], dtype=jnp.uint32)
    )

    return TrainState(
        step=step,
        model=model,
        opt_state=opt_state,
        train_key=train_key,
        data_key=data_key,
    )


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


def shard_model[T](model: T, sharding: NamedSharding) -> T:
    """Shard model arrays while preserving non-array leaves (functions, static fields)."""
    arrays, non_arrays = eqx.partition(model, eqx.is_array)
    arrays = jax.device_put(arrays, sharding)
    return eqx.combine(arrays, non_arrays)
