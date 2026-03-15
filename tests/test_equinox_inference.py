"""Tests for giggleml.inference.equinox_inference module."""

from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from giggleml.inference.equinox_inference import (
    CallableModule,
    generic_data_parallel,
    global_parallel_forward,
)


class SimpleModel(CallableModule[Float[Array, "dim"]]):
    """A simple model that doubles the input."""

    weight: Float[Array, "dim dim"]

    def __init__(self, dim: int, key: jax.Array):
        self.weight = jax.random.normal(key, (dim, dim))

    @override
    def __call__(self, x: Array) -> Float[Array, "dim"]:
        return x @ self.weight


class IdentityModel(CallableModule[Float[Array, "..."]]):
    """A model that returns its input unchanged."""

    @override
    def __call__(self, x: Array) -> Float[Array, "..."]:
        return x


class SumModel(CallableModule[Float[Array, ""]]):
    """A model that sums all elements."""

    @override
    def __call__(self, x: Array) -> Float[Array, ""]:
        return jnp.sum(x)


class MultiInputModel(CallableModule[Float[Array, "dim"]]):
    """A model that takes multiple inputs."""

    @override
    def __call__(self, x: Array, y: Array) -> Float[Array, "dim"]:
        return x + y


class TestCallableModule:
    """Tests for CallableModule abstract class."""

    def test_simple_model_callable(self):
        key = jax.random.key(0)
        model = SimpleModel(dim=4, key=key)
        x = jnp.ones(4)
        result = model(x)
        assert result.shape == (4,)

    def test_identity_model(self):
        model = IdentityModel()
        x = jnp.array([1.0, 2.0, 3.0])
        result = model(x)
        assert jnp.array_equal(result, x)

    def test_sum_model(self):
        model = SumModel()
        x = jnp.array([1.0, 2.0, 3.0])
        result = model(x)
        assert result == 6.0

    def test_multi_input_model(self):
        model = MultiInputModel()
        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        result = model(x, y)
        assert jnp.array_equal(result, jnp.array([4.0, 6.0]))


class TestGlobalParallelForward:
    """Tests for global_parallel_forward function."""

    def test_simple_batch(self):
        model = IdentityModel()
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = global_parallel_forward(model, (batch,))
        assert jnp.array_equal(result, batch)

    def test_batch_dimension(self):
        model = IdentityModel()
        batch = jnp.ones((5, 3))  # 5 samples, 3 features
        result = global_parallel_forward(model, (batch,))
        assert result.shape == (5, 3)

    def test_sum_model_batched(self):
        model = SumModel()
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # 2 samples
        result = global_parallel_forward(model, (batch,))
        # Each row should be summed
        assert jnp.array_equal(result, jnp.array([3.0, 7.0]))

    def test_linear_model_batched(self):
        key = jax.random.key(42)
        model = SimpleModel(dim=4, key=key)
        batch = jnp.ones((3, 4))  # 3 samples, 4 features
        result = global_parallel_forward(model, (batch,))
        assert result.shape == (3, 4)

    def test_multi_input_batched(self):
        model = MultiInputModel()
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = global_parallel_forward(model, (x, y))
        expected = jnp.array([[6.0, 8.0], [10.0, 12.0]])
        assert jnp.array_equal(result, expected)

    def test_jit_compilation(self):
        model = IdentityModel()
        batch = jnp.ones((2, 3))

        # First call compiles
        result1 = global_parallel_forward(model, (batch,))
        # Second call uses cached compilation
        result2 = global_parallel_forward(model, (batch,))

        assert jnp.array_equal(result1, result2)


class TestGenericDataParallel:
    """Tests for generic_data_parallel function.

    These tests run on CPU with a single device.
    """

    def test_single_batch(self, capsys):
        model = IdentityModel()
        data = [(jnp.array([[1.0, 2.0], [3.0, 4.0]]),)]

        results = generic_data_parallel(model, data)

        assert len(results) == 1
        assert jnp.array_equal(results[0], data[0][0])

    def test_multiple_batches(self, capsys):
        model = IdentityModel()
        batch1 = jnp.array([[1.0, 2.0]])
        batch2 = jnp.array([[3.0, 4.0]])
        data = [(batch1,), (batch2,)]

        results = generic_data_parallel(model, data)

        assert len(results) == 2
        assert jnp.array_equal(results[0], batch1)
        assert jnp.array_equal(results[1], batch2)

    def test_empty_iterator(self, capsys):
        model = IdentityModel()
        data: list[tuple[Array, ...]] = []

        results = generic_data_parallel(model, data)

        assert results == []

    def test_prints_device_info(self, capsys):
        model = IdentityModel()
        data = [(jnp.ones((1, 2)),)]

        generic_data_parallel(model, data)

        captured = capsys.readouterr()
        assert "Using" in captured.out
        assert "device" in captured.out

    def test_with_generator_input(self, capsys):
        model = IdentityModel()

        def gen_data():
            yield (jnp.array([[1.0]]),)
            yield (jnp.array([[2.0]]),)

        results = generic_data_parallel(model, gen_data())

        assert len(results) == 2

    def test_sum_model_parallel(self, capsys):
        model = SumModel()
        data = [
            (jnp.array([[1.0, 2.0], [3.0, 4.0]]),),
            (jnp.array([[5.0, 5.0]]),),
        ]

        results = generic_data_parallel(model, data)

        assert len(results) == 2
        assert jnp.array_equal(results[0], jnp.array([3.0, 7.0]))
        assert jnp.array_equal(results[1], jnp.array([10.0]))

    def test_multi_input_parallel(self, capsys):
        model = MultiInputModel()
        x1 = jnp.array([[1.0, 2.0]])
        y1 = jnp.array([[3.0, 4.0]])
        data = [(x1, y1)]

        results = generic_data_parallel(model, data)

        expected = jnp.array([[4.0, 6.0]])
        assert jnp.array_equal(results[0], expected)
