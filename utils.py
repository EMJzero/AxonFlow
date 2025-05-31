from typing import TypeVar, Generator, Callable, Iterable, Any

T = TypeVar('T')
U = TypeVar('U')

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
def iter_major_diagonals(x_beg : int, y_beg : int, x_end : int, y_end : int, end_included : bool = False) -> Generator[tuple[int, int], None, None]:
    x_sign = 1 if x_beg < x_end else -1
    y_sign = 1 if y_beg < y_end else -1
    if end_included:
        x_end += x_sign
        y_end += y_sign
    width = abs(x_end - x_beg)
    height = abs(y_beg - y_end)
    for i in range(height):
        for j in range(min(i + 1, width)):
            yield (x_beg + x_sign*j, y_beg + y_sign*(i - j))
    for j in range(1, width):
        for i in range(min(width - j, height)):
            yield (x_beg + x_sign*(j + i), y_end - y_sign*(1 + i))

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