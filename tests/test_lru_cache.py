"""Tests for giggleml.utils.lru_cache module."""

import pytest

from giggleml.utils.lru_cache import LRUCache, lru_cache


class TestLRUCache:
    """Tests for LRUCache class."""

    def test_init_default_limit(self):
        cache: LRUCache[str, int] = LRUCache()
        assert cache._limit == 128

    def test_init_custom_limit(self):
        cache: LRUCache[str, int] = LRUCache(limit=10)
        assert cache._limit == 10

    def test_get_calls_default_on_miss(self):
        cache: LRUCache[str, int] = LRUCache()
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return 42

        result = cache.get("key", factory)
        assert result == 42
        assert call_count == 1

    def test_get_returns_cached_on_hit(self):
        cache: LRUCache[str, int] = LRUCache()
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return 42

        cache.get("key", factory)
        result = cache.get("key", factory)
        assert result == 42
        assert call_count == 1  # Factory only called once

    def test_get_evicts_oldest_when_over_limit(self):
        cache: LRUCache[str, int] = LRUCache(limit=2)
        cache.get("a", lambda: 1)
        cache.get("b", lambda: 2)
        cache.get("c", lambda: 3)  # Should evict "a"

        # "a" should be evicted, so factory is called again
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return 100

        result = cache.get("a", factory)
        assert result == 100
        assert call_count == 1

    def test_get_updates_lru_order(self):
        cache: LRUCache[str, int] = LRUCache(limit=2)
        cache.get("a", lambda: 1)
        cache.get("b", lambda: 2)
        cache.get("a", lambda: 999)  # Access "a", making "b" oldest
        cache.get("c", lambda: 3)  # Should evict "b", not "a"

        # "a" should still be cached with original value
        result = cache.get("a", lambda: 999)
        assert result == 1

    def test_pop_returns_value(self):
        cache: LRUCache[str, int] = LRUCache()
        cache.get("key", lambda: 42)
        result = cache.pop("key")
        assert result == 42

    def test_pop_removes_from_cache(self):
        cache: LRUCache[str, int] = LRUCache()
        cache.get("key", lambda: 42)
        cache.pop("key")

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return 100

        result = cache.get("key", factory)
        assert result == 100
        assert call_count == 1

    def test_pop_raises_on_missing_key(self):
        cache: LRUCache[str, int] = LRUCache()
        with pytest.raises(RuntimeError, match="missing key"):
            cache.pop("nonexistent")

    def test_add_inserts_value(self):
        cache: LRUCache[str, int] = LRUCache()
        cache.add("key", 42)

        # Factory should not be called since key exists
        result = cache.get("key", lambda: 999)
        assert result == 42

    def test_add_updates_existing_value(self):
        cache: LRUCache[str, int] = LRUCache()
        cache.add("key", 42)
        cache.add("key", 100)

        result = cache.get("key", lambda: 999)
        assert result == 100

    def test_add_evicts_when_over_limit(self):
        cache: LRUCache[str, int] = LRUCache(limit=2)
        cache.add("a", 1)
        cache.add("b", 2)
        cache.add("c", 3)  # Should evict "a"

        # "a" was evicted, so factory returns new value
        result = cache.get("a", lambda: 100)
        assert result == 100

    def test_add_updates_lru_order(self):
        cache: LRUCache[str, int] = LRUCache(limit=2)
        cache.add("a", 1)
        cache.add("b", 2)
        cache.add("a", 10)  # Update "a", making "b" oldest
        cache.add("c", 3)  # Should evict "b"

        # "a" should still be cached
        result = cache.get("a", lambda: 999)
        assert result == 10


class TestLruCacheDecorator:
    """Tests for lru_cache decorator."""

    def test_decorator_without_args(self):
        call_count = 0

        @lru_cache
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert func(5) == 10
        assert func(5) == 10
        assert call_count == 1

    def test_decorator_with_args(self):
        call_count = 0

        @lru_cache(max_size=2)
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert func(1) == 2
        assert func(2) == 4
        assert func(3) == 6  # Evicts 1
        assert func(1) == 2  # Recomputes
        assert call_count == 4

    def test_decorator_preserves_different_args(self):
        call_count = 0

        @lru_cache
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert func(1) == 2
        assert func(2) == 4
        assert call_count == 2

    def test_decorator_with_typed(self):
        call_count = 0

        @lru_cache(typed=True)
        def func(x: int | float) -> float:
            nonlocal call_count
            call_count += 1
            return float(x) * 2

        assert func(1) == 2.0
        assert func(1.0) == 2.0
        assert call_count == 2  # Different types = different cache entries

    def test_decorator_with_multiple_args(self):
        call_count = 0

        @lru_cache
        def func(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + y

        assert func(1, 2) == 3
        assert func(1, 2) == 3
        assert func(2, 1) == 3
        assert call_count == 2  # Different arg order = different cache key
