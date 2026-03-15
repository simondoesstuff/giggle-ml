"""Data-parallel inference utilities for Equinox models.

Provides functions for distributing inference across multiple JAX devices using
SPMD (Single Program Multiple Data) parallelism via JAX's sharding APIs.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Generic, TypeVar

import equinox as eqx
import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

T = TypeVar("T")


class CallableModule(eqx.Module, ABC, Generic[T]):
    """Equinox modules that are callable with a generic output type."""

    @abstractmethod
    def __call__(self, *args: Array) -> T: ...


@eqx.filter_jit
def global_parallel_forward(
    model: CallableModule[T], batch_tuple: tuple[Array, ...]
) -> T:
    """Execute a batched forward pass with automatic vectorization.

    Defined at module level (outside any class) to avoid JIT recompilation issues
    that occur when methods capture different `self` references.

    Args:
        model: An Equinox module to apply to each batch element.
        batch_tuple: Tuple of arrays where each array's first axis is the batch
            dimension. All arrays must have the same batch size.

    Returns:
        Model outputs stacked along the batch dimension.

    Note:
        `eqx.filter_vmap` intelligently handles Equinox's PyTree structure:
        it preserves model parameters (non-array leaves) while vectorizing
        over axis 0 of all array inputs.
    """
    return eqx.filter_vmap(model)(*batch_tuple)


def generic_data_parallel(
    model: CallableModule[T], data_iterator: Iterable[tuple[Array, ...]]
) -> list[T]:
    """Run inference with data parallelism across all available JAX devices.

    Creates a 1D mesh over all devices and shards each batch along the "batch"
    axis, distributing work evenly. Each device processes a slice of the batch
    in parallel.

    Args:
        model: An Equinox module (must be callable). The model is replicated
            across all devices.
        data_iterator: An iterable yielding tuples of arrays. Each tuple
            represents one batch of inputs to the model. Arrays in the tuple
            should have shape (batch_size, ...).

    Returns:
        List of outputs, one per batch from the iterator. Each output has the
        same structure as `model(*batch_tuple)` would return.

    Example:
        >>> model = MyEquinoxModel(key=jax.random.key(0))
        >>> data = [(x_batch, y_batch) for x_batch, y_batch in dataloader]
        >>> outputs = generic_data_parallel(model, data)
    """
    devices = jax.devices()
    print(f"Using {len(devices)} devices: {str(devices)}")
    mesh = Mesh(devices, axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch"))

    results: list[T] = []

    with mesh:
        for batch_tuple in data_iterator:
            # Shard inputs across devices along the batch axis
            sharded_batch = tuple(jax.device_put(x, data_sharding) for x in batch_tuple)

            # Run the JIT-compiled, vmapped forward pass
            output = global_parallel_forward(model, sharded_batch)
            results.append(output)

    return results
