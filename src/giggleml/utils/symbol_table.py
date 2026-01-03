from functools import cached_property


class SymbolTable[T]:
    def __init__(self, base: list[T]) -> None:
        self.base: list[T] = base

    @cached_property
    def _inverse(self) -> dict[T, int]:
        return {v: i for i, v in enumerate(self.base)}

    def __getitem__(self, index: int) -> T:
        return self.base[index]

    def index(self, item: T) -> int:
        return self._inverse[item]
