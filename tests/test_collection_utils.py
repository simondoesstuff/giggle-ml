"""Tests for giggleml.utils.collection_utils module."""

from giggleml.utils.collection_utils import as_list, take_key


class TestTakeKey:
    """Tests for take_key function."""

    def test_extracts_key_from_dicts(self):
        items = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = list(take_key("a", items))
        assert result == [1, 3]

    def test_different_key(self):
        items = [{"x": "foo", "y": "bar"}, {"x": "baz", "y": "qux"}]
        result = list(take_key("y", items))
        assert result == ["bar", "qux"]

    def test_empty_iterable(self):
        result = list(take_key("a", []))
        assert result == []

    def test_single_item(self):
        items = [{"key": 42}]
        result = list(take_key("key", items))
        assert result == [42]

    def test_is_lazy_generator(self):
        items = [{"a": 1}, {"a": 2}]
        gen = take_key("a", items)
        # Should be a generator, not a list
        assert hasattr(gen, "__next__")

    def test_with_generator_input(self):
        def gen_dicts():
            yield {"v": 10}
            yield {"v": 20}

        result = list(take_key("v", gen_dicts()))
        assert result == [10, 20]


class TestAsList:
    """Tests for as_list decorator."""

    def test_converts_generator_to_list(self):
        @as_list
        def gen_nums():
            yield 1
            yield 2
            yield 3

        result = gen_nums()
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_with_arguments(self):
        @as_list
        def repeat(val: str, times: int):
            for _ in range(times):
                yield val

        result = repeat("x", 3)
        assert result == ["x", "x", "x"]

    def test_empty_generator(self):
        @as_list
        def empty():
            return
            yield  # Makes it a generator

        result = empty()
        assert result == []

    def test_preserves_function_name(self):
        @as_list
        def my_generator():
            yield 1

        assert my_generator.__name__ == "my_generator"

    def test_with_kwargs(self):
        @as_list
        def gen_range(start: int = 0, end: int = 3):
            for i in range(start, end):
                yield i

        assert gen_range() == [0, 1, 2]
        assert gen_range(start=5, end=8) == [5, 6, 7]

    def test_forces_evaluation(self):
        side_effects: list[int] = []

        @as_list
        def with_side_effect():
            for i in range(3):
                side_effects.append(i)
                yield i

        # Before calling, no side effects
        assert side_effects == []

        # After calling, all side effects happen immediately
        result = with_side_effect()
        assert side_effects == [0, 1, 2]
        assert result == [0, 1, 2]
