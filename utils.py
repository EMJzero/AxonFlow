from typing import Generic, TypeVar, Generator, Callable, Iterable, Self, Iterator, Optional
from collections.abc import MutableMapping
from collections import defaultdict

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

    def __str__(self) -> str:
        return "{" + ", ".join(f"{k} <-> {v}" for k, v in self._fwd.items()) + "}"

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
Disjoint set (Union-Find) implementation, credit to: Ivan Lazarevic (https://github.com/ivanbgd/Disjoint-Sets-Data-Structure/)
Each element is part of a set represented by a unique root element.

When both union by rank heuristic and path compression heuristic are used, the average running time of each operation is nearly constant.
Uses Trees, Union by Rank Heuristic, and Path Compression Heuristic.
Uses 1-based indexing of arrays.
Storage style: structure of arrays.

Arguments:
- elements: optional iterable of initial elements, each forming its own singleton set.
"""
class DisjointSet(Generic[T]):
    def __init__(self, elements: Optional[Iterable[T]] = None):
        self.parent: dict[T, T] = {}
        self.rank: dict[T, int] = {}
        if elements:
            for elem in elements:
                self.makeSet(elem)

    """
    Adds a new element as a singleton set.
    If the element already exists in the structure, this does nothing.

    Arguments:
    - x: the element to be added as its own set.
    """
    def makeSet(self, x: T) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    """
    Finds the representative (root) of the set containing x.
    Applies path compression to flatten the tree structure, optimizing future queries.

    Arguments:
    - x: The element whose set representative is to be found.

    Returns: the representative element of the set containing x.
    """
    def find(self, x: T) -> T:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    """
    Merges the sets containing x and y.
    Uses the union by rank heuristic to keep trees shallow for performance.

    Arguments:
    - x: an element in the first set.
    - y: an element in the second set.
    """
    def union(self, x: T, y: T) -> None:
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return  # Already in the same set

        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1
    
    """
    Yields each disjoint set as a generator of its elements.

    Returns: a generator of generators, where each inner generator yields elements of a single set.
    """
    def __iter__(self) -> Iterable[Iterable[T]]:
        sets: defaultdict[T, list[T]] = defaultdict(list)
        for x in self.parent:
            root = self.find(x)
            sets[root].append(x)

        for group in sets.values():
            yield (elem for elem in group)
    
    """
    Enable the use of "in" to check if an elements belongs to any disjont set.
    """
    def __contains__(self, x: T) -> bool:
        return x in self.parent

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

"""
Given multiple dictionaries potentially sharing the same key and having values
for which the '+' operation is defined, this returns the dictionary having all
keys of the originals and for values the sum of the values with the same key
among the original dictionaries.
"""
def dict_sum(*dicts : tuple[dict[T, U], ...]) -> dict[T, U]:
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)

"""
Given N buckets all of size M and a set 'num' of positive integers, this algorithm
checks if there exists a way to distribute the elements of the set among buckets
such that the sum of the integers in each bucket is <=M.
"""
def can_distribute(nums : list[int], N : int, M : int) -> bool:
    if sum(nums) > N * M:
        return False  # Quick fail: total sum exceeds total capacity

    nums.sort(reverse=True)  # Start with biggest numbers to prune faster
    buckets = [0] * N

    def backtrack(index):
        if index == len(nums):
            return True
        num = nums[index]
        for i in range(N):
            if buckets[i] + num <= M:
                buckets[i] += num
                if backtrack(index + 1):
                    return True
                buckets[i] -= num
            # Optimization: If a number doesn't fit in an empty bucket, don't try placing it in other empty buckets (to avoid symmetry)
            if buckets[i] == 0:
                break
        return False

    return backtrack(0)

"""
Given N buckets all of size M and a list of sets of integers 'sets_list', this algorithm
checks if there exists a way to distribute the sets of the list among buckets such that
the count [or sum] of integers in each bucket is <=M and the number of sets added to a
bucket does not exceed K. However, each bucket is itself a set, as such, duplicate
elements do not count towards M.

NOTE: stupidly slow.

Args:
- sets_list: list of sets to distribute among buckets
- N: number of buckets
- if sum_not_count == False:
    - M: maximum number of distinct elements per bucket
  else:
    - M: maximum sum of distinct elements per bucket
- K: maximum number of sets per bucket
"""
def can_distribute_sets(sets_list : list[set[int]], N : int, M : int, K : int, sum_not_count : bool = False) -> bool:
    # Convert each set into a bitmask
    def set_to_bitmask(s):
        mask = 0
        for x in s:
            mask |= 1 << x
        return mask

    def bitcount(x):
        return bin(x).count('1')
    
    def bitvalue(x):
        value = 0
        for i, b in enumerate(bin(x)[:1:-1]):
            if b == 1:
                value += i
        return value

    bitsets = sorted([set_to_bitmask(s) for s in sets_list], key=bitcount, reverse=True)

    # Quick fail: total unique elements must fit
    total_union = 0
    for b in bitsets:
        total_union |= b
    if bitcount(total_union) > N * M or len(bitsets) > N * K:
        return False

    # Each bucket: (bitmask of elements, count of sets)
    initial_state = [(0, 0) for _ in range(N)]
    stack = [(0, initial_state)]

    while stack:
        index, buckets = stack.pop()
        if index == len(bitsets):
            return True  # All sets placed

        current = bitsets[index]

        for i in range(N):
            bits, count = buckets[i]
            if count >= K:
                continue
            merged = bits | current
            if (not sum_not_count and bitcount(merged) <= M) or (sum_not_count and bitvalue(merged) <= M):
                new_buckets = list(buckets)
                new_buckets[i] = (merged, count + 1)
                stack.append((index + 1, new_buckets))
            if count == 0:
                break  # symmetry breaking

    return False

"""
Generate 'num_points' approximately equi-spaced points on a 2D lattice based on Manhattan distance.

Parameters:
- width: the width of the lattice (number of columns). Must be > 0.
- height: the height of the lattice (number of rows). Must be > 0.
- num_points: the number of equi-spaced points to return. Must be ≤ width * height.

Returns: a list of Coord2D instances representing approximately equi-spaced points.
"""
def get_equispaced_lattice_points(width: int, height: int, num_points: int) -> list[Coord2D]:
    if width <= 0 or height <= 0:
        raise Exception("Width and height must be positive integers.")
    
    total_points = width * height
    if num_points > total_points:
        raise Exception(f"Cannot select {num_points} points from a lattice of size {width}x{height} ({total_points} points).")

    # flatten all possible lattice points in row-major order
    all_points = [Coord2D(x, y) for y in range(height) for x in range(width)]
    
    if num_points == 1:
        # choose the center point
        center_x = width // 2
        center_y = height // 2
        return [Coord2D(center_x, center_y)]

    # choose the points approximately equidistant in the list in terms of manhattan spread
    # distribute indices as evenly as possible
    step = total_points / num_points
    selected_points = []
    for i in range(num_points):
        index = int(round(i * step))
        if index >= total_points:
            index = total_points - 1
        selected_points.append(all_points[index])

    # deduplicate in rare edge rounding cases
    return list(dict.fromkeys(selected_points))