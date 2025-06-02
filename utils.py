from typing import TypeVar, Generator, Callable, Iterable, Any, Self, Iterator, Optional
from collections.abc import MutableMapping

T = TypeVar('T')
U = TypeVar('U')

"""
Class for 2D discrete (integer) coordinate based on a tuple.
"""
class Coord2D(tuple):
    def __new__(cls, x, y):
        if not (isinstance(x, int) and isinstance(y, int)):
            raise TypeError("Coordinates x and y must be integers.")
        return super().__new__(cls, (x, y))

    @property
    def x(self) -> int:
        return self[0]

    @property
    def y(self) -> int:
        return self[1]

    def __add__(self, other : Self) -> Self:
        if not isinstance(other, Coord2D):
            return NotImplemented
        return Coord2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other : Self) -> Self:
        if not isinstance(other, Coord2D):
            return NotImplemented
        return Coord2D(self.x - other.x, self.y - other.y)

    def __neg__(self) -> int:
        return Coord2D(-self.x, -self.y)

    def __abs__(self) -> int:
        return abs(self.x) + abs(self.y)

    def __repr__(self) -> str:
        return f"(x = {self.x}, y = {self.y})"

"""
A bidirectional map, allowing efficient lookup in both directions, with optional default factories (like defaultdict).
Supports standard dictionary access ('bimap[a] = b') and reverse access via 'bimap.inv[b] = a'.
Updates, deletions, and insertions automatically keep both directions in sync.

Args:
- initial_dict: optional dictionary to initialize the forward map; reverse is inferred.
- default_factory: optional callable that produces default values for missing forward keys.
- inv_default_factory: optional callable for inverse (reverse) lookup default values.
"""
class BiMap(MutableMapping[T, U]):
    def __init__(self, initial_dict: Optional[dict[T, U]] = None, default_factory: Optional[Callable[[], U]] = None, inv_default_factory: Optional[Callable[[], T]] = None):
        self._fwd: dict[T, U] = {}
        self._rev: dict[U, T] = {}
        self._default_factory = default_factory
        self._inv_default_factory = inv_default_factory
        self.inv = self.InverseMap(self)

        if initial_dict:
            for a, b in initial_dict.items():
                self[a] = b

    def __setitem__(self, a: T, b: U) -> None:
        if a in self._fwd:
            old_b = self._fwd[a]
            del self._rev[old_b]
        if b in self._rev:
            old_a = self._rev[b]
            del self._fwd[old_a]
        self._fwd[a] = b
        self._rev[b] = a

    def __getitem__(self, a: T) -> U:
        if a in self._fwd:
            return self._fwd[a]
        if self._default_factory is not None:
            b = self._default_factory()
            self[a] = b
            return b
        raise KeyError(a)

    def __delitem__(self, a: T) -> None:
        b = self._fwd.pop(a)
        del self._rev[b]

    def __iter__(self) -> Iterator[T]:
        return iter(self._fwd)

    def __len__(self) -> int:
        return len(self._fwd)

    class InverseMap(MutableMapping[U, T]):
        def __init__(self, outer: 'BiMap[T, U]'):
            self._outer = outer

        def __setitem__(self, b: U, a: T) -> None:
            self._outer[a] = b

        def __getitem__(self, b: U) -> T:
            if b in self._outer._rev:
                return self._outer._rev[b]
            if self._outer._inv_default_factory is not None:
                a = self._outer._inv_default_factory()
                self._outer[a] = b
                return a
            raise KeyError(b)

        def __delitem__(self, b: U) -> None:
            a = self._outer._rev.pop(b)
            del self._outer._fwd[a]

        def __iter__(self) -> Iterator[U]:
            return iter(self._outer._rev)

        def __len__(self) -> int:
            return len(self._outer._rev)

"""
Compute the manhattan distance betwenn two points 'pt1' and 'pt2' in an n-dimensional lattice.
"""
def manhattan(pt1 : tuple[int, ...], pt2 : tuple[int, ...]) -> int:
    if len(pt1) != len(pt2):
        raise Exception("The two points must have the same dimensionality.")
    
    return sum(abs(coord1 - coord2) for coord1, coord2 in zip(pt1, pt2))

"""
Iterate over all coordinates in a rectangular sub-lattice diagonally.
Starts from (x_beg, y_beg) [included] and proceeds over minor diagonals
towards (x_end, y_end) [included/excluded depending on 'end_included'].

Note: this coincides with an enumeration by increasing manhattan distance from (x_beg, y_beg).
"""
def iter_major_diagonals(x_beg : int, y_beg : int, x_end : int, y_end : int, end_included : bool = False) -> Generator[Coord2D, None, None]:
    x_sign = 1 if x_beg < x_end else -1
    y_sign = 1 if y_beg < y_end else -1
    if end_included:
        x_end += x_sign
        y_end += y_sign
    width = abs(x_end - x_beg)
    height = abs(y_beg - y_end)
    for i in range(height):
        for j in range(min(i + 1, width)):
            yield Coord2D(x_beg + x_sign*j, y_beg + y_sign*(i - j))
    for j in range(1, width):
        for i in range(min(width - j, height)):
            yield Coord2D(x_beg + x_sign*(j + i), y_end - y_sign*(1 + i))

"""
Performs a foldL of the provided 'iterable', starting from 'initial', and return
every intermediate result, in order. Intermediate results include 'initial'.

The provided folding function shall take two arguments, of which the first will
be used for the accumulator.
"""
def scan_left(func : Callable[[T, U], T], iterable : Iterable[U], initial : T) -> list[T]:
    result = [initial]
    acc = initial
    for item in iterable:
        acc = func(acc, item)
        result.append(acc)
    return result