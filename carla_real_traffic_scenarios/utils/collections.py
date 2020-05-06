import typing
from abc import abstractmethod
from typing import TypeVar, Callable, List

from more_itertools import windowed
from typing_extensions import Protocol

T = TypeVar("T")

# For comparable generic see https://github.com/python/typing/issues/59#issuecomment-353878355
C = typing.TypeVar("C", bound="Comparable")


class Comparable(Protocol):
    @abstractmethod
    def __eq__(self, other: typing.Any) -> bool:
        pass

    @abstractmethod
    def __lt__(self: C, other: C) -> bool:
        pass

    def __gt__(self: C, other: C) -> bool:
        return (not self < other) and self != other

    def __le__(self: C, other: C) -> bool:
        return self < other or self == other

    def __ge__(self: C, other: C) -> bool:
        return (not self < other)


def smallest_by(elements: List[T], key_fn: Callable[[T], C]) -> T:
    assert len(elements) > 0

    e_for_smallest_key = elements[0]
    smallest_key = key_fn(e_for_smallest_key)

    for e in elements[1:]:
        key = key_fn(e)
        if key < smallest_key:
            smallest_key = key
            e_for_smallest_key = e

    return e_for_smallest_key


def find_first_matching(elements: List[T], predicate: Callable[[T], bool]) -> T:
    for e in elements:
        if predicate(e):
            return e
    else:
        raise Exception("Element not present in collection")