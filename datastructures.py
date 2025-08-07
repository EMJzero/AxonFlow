from __future__ import annotations

from typing import Generic, TypeVar, Generator, Callable, Iterable, Self, Iterator, Optional
from collections.abc import MutableMapping
from sortedcontainers import SortedList
from collections import defaultdict
import hashlib
import random
import xxhash
import heapq
import math

from datastructures import *
from settings import *
from prints import *

T = TypeVar('T')
U = TypeVar('U')

"""
Class for 2D discrete (integer) coordinate based on a tuple.
"""
class Coord2D(tuple):
    def __new__(cls, x : int, y : int):
        if isinstance(x, tuple) and y is None: # accept tuple (x, y) for pickling or normal x, y
            x, y = x
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
    
    # tells pickle how to reconstruct this object
    def __reduce__(self):
        return (self.__class__, (self.x, self.y))

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
        self.size = 0
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
        self.size += 1
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

        self.size -= 1
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

    def __len__(self) -> int:
        return self.size

"""
Struct for one entry in an LSH data structure.
"""
class LSHEntry:
    def __init__(self, weighted_set : dict[T, float], signature : list[int], merge_count : int = 1):
        self.weighted_set = weighted_set
        self.signature = signature
        self.merge_count = merge_count

"""
Weighted MinHash LSH (locality sensitive hanshing)-based indexing infrastructure.

Based on:
- https://github.com/ekzhu/datasketch/blob/master/datasketch/lsh.py

Arguments:
- num_perm: number of independent hash functions (permutations) used to generate one MinHash signature.
            => higher values reduce both false positives and true positives.
- num_bands: number of slices (bands) the signature is divided into for LSH bucketing. Must divide 'num_perm' exactly.
            => tune w.r.t. the data, higher values make collisions less likely but more accurate.
"""
class WeightedMinHashLSH(Generic[T]):
    def __init__(self, num_perm : int = 128, num_bands : int = 32):
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.band_size = num_perm // num_bands
        assert self.num_perm % self.num_bands == 0, f"The given 'num_perm' ({num_perm}) must be divisible by 'num_bands' ({num_bands})."

        self.index : list[dict[int, set[int]]] = [defaultdict(set) for _ in range(self.num_bands)] # usage: index[band_idx][hash] -> set of set_ids
        self.data : dict[int, LSHEntry] = {} # usage: data[set_id] -> entry
        self.id_counter = 0

        # Random parameters for ICWS
        self._r = [random.gammavariate(2.0, 1.0) for _ in range(self.num_perm)]
        self._c = [random.gammavariate(2.0, 1.0) for _ in range(self.num_perm)]
        self._beta = [random.uniform(0, 1) for _ in range(self.num_perm)]

    """
    Computes the MinHash signature of a weighted set (dict element->weight).
    Return a list of integers of length 'self.num_perm'.
    """
    def _weighted_minhash_signature(self, weighted_set : dict[T, float]) -> list[int]:
        # TODO: devise a good weighted implementation!!!!!
        # UNWEIGHTED: minimum among the hashes of the set's values.
        return [min((hashlib.sha1(str((k, per)).encode()).hexdigest() for k in weighted_set.keys()), default = random.randbytes(20).hex()) for per in range(self.num_perm)]
        # WEIGHTED??: ICWS-based weighted MinHash implementation.
        signature = []
        for i in range(self.num_perm):
            min_val = float('inf')
            min_hash = None
            r = self._r[i]
            c = self._c[i]
            beta = self._beta[i]
            for key, weight in weighted_set.items():
                if weight <= 0:
                    continue
                logw = math.log(weight)
                f = math.floor(logw / r + beta)
                h = math.exp((f - beta) * r) * c
                if h < min_val:
                    min_val = h
                    min_hash = hash(key)
            signature.append(min_hash)
        return signature

    """
    Hash a single band to an integer value.
    """
    def _hash_band(self, band):
        return hashlib.sha1(str(band).encode()).hexdigest()

    """
    Return the original set dictionary for a given ID.
    """
    def get(self, set_id : int) -> LSHEntry:
        if set_id in self.data:
            return self.data[set_id]
        else:
            raise Exception(f"The provided set ID {set_id} does not exist.")

    """
    Return the IDs present in the datastructure.
    """
    def ids(self) -> Iterable[int]:
        return self.data.keys()

    """
    Insert a new weighted set into the index.
    If no 'set_id' is provided, an internal counter is used.
    The final 'set_id' is returned.
    """
    def insert(self, weighted_set : dict[T, float], set_id : Optional[int] = None, merge_count : int = 1) -> int:
        if set_id is None:
            set_id = self.id_counter
            self.id_counter += 1

        signature = self._weighted_minhash_signature(weighted_set)
        self.data[set_id] = LSHEntry(weighted_set, signature, merge_count)

        for i in range(self.num_bands):
            band = tuple(signature[i*self.band_size:(i + 1)*self.band_size])
            band_hash = self._hash_band(band)
            self.index[i][band_hash].add(set_id)

        return set_id

    """
    Delete a set from the index.
    Returns True if the element was present and thus deleted.
    """
    def delete(self, set_id : int) -> bool:
        if set_id not in self.data:
            return False

        signature = self.data[set_id].signature
        for i in range(self.num_bands):
            band = tuple(signature[i * self.band_size:(i + 1) * self.band_size])
            band_hash = self._hash_band(band)
            self.index[i][band_hash].discard(set_id)

        del self.data[set_id]
        return True

    """
    Merge multiple sets already stored in the structure and insert the merged version.
    The old sets are deleted. Returns new set's ID.
    """
    def merge(self, set_ids : list[int], merged_set_id : int = None) -> int:
        merged = {}
        mcount = 0
        for sid in set_ids:
            if sid not in self.data:
                raise Exception(f"The provided set ID {sid} does not exist.")
            wset = self.data[sid].weighted_set
            mcount += self.data[sid].merge_count
            for k, v in wset.items():
                # WARNING: maybe you should not add weights if the entry was generated from the same hyperedge...
                merged[k] = merged.get(k, 0) + v
        for sid in set_ids:
            self.delete(sid)
        return self.insert(merged, merge_count = mcount, set_id = merged_set_id)

    """
    Query similar sets.
    If 'top_k' is set, it returns up to 'top_k' most similar sets.
    Returns the candidate IDs and optionally the list of their distances.
    """
    def query(self, weighted_set : dict[T, float], top_k : Optional[int] = None) -> tuple[list[int], list[float]]:
        signature = self._weighted_minhash_signature(weighted_set)
        candidates = set()

        for i in range(self.num_bands):
            band = tuple(signature[i * self.band_size:(i + 1) * self.band_size])
            band_hash = self._hash_band(band)
            candidates.update(self.index[i][band_hash])

        scored = []
        for cid in candidates:
            other_set = self.data[cid].weighted_set
            score = self._weighted_jaccard(weighted_set, other_set)
            scored.append((score, cid))

        # If top_k is requested, rank them by similarity
        # TODO: inefficient, builds a new list...
        if top_k is not None:
            scored.sort(reverse=True)
            return [cid for _, cid in scored[:top_k]], [score for score, _ in scored[:top_k]]
        else:
            return [cid for _, cid in scored], [score for score, _ in scored]

    """
    Query similar sets using an existing stored set.
    """
    def query_by_id(self, set_id : int, top_k : Optional[int] = None) -> tuple[list[int], list[float]]:
        if set_id not in self.data:
            raise Exception(f"The provided set ID {set_id} does not exist.")
        target_set = self.data[set_id].weighted_set
        candidates, scores = self.query(target_set, top_k = top_k)
        try:
            idx = candidates.index(set_id)
            candidates.pop(idx)
            scores.pop(idx)
        except ValueError:
            pass
        return candidates, scores

    """
    Compute the weighted Jaccard similarity between two sets.
    """
    def _weighted_jaccard(self, set1 : dict[T, float], set2: dict[T, float]) -> float:
        set1_only = sum(v for k, v in set1.items() if k not in set2)
        set2_only = sum(v for k, v in set2.items() if k not in set1)
        intersection = sum(v + set2[k] for k, v in set1.items() if k in set2)
        union = set1_only + set2_only + intersection
        return intersection / union if union > 0 else 0.0
    
    def __len__(self) -> int:
        return len(self.data)

"""
Like 'WeightedMinHashLSH' but optimized for top-k queries and based on a prefix tree data structure.

Based on:
- http://ilpubs.stanford.edu:8090/678/1/2005-14.pdf
- https://github.com/ekzhu/datasketch/blob/master/datasketch/lshforest.py

Arguments:
- hash_bits: bits to use in reach MinHash and thus in reach prefix of the tree.
             => higher values reduce both false positive and true positives.
"""
class WeightedMinHashLSHForest(Generic[T]):
    def __init__(self, num_perm : int = 256, tree_count : int = 8, hash_bytes : int = 4):
        self.num_perm = num_perm
        self.tree_count = tree_count
        self.hash_bytes = hash_bytes
        assert self.hash_bytes <= 8, f"Maximum supported hash lenght is 16 bytes, {hash_bytes} bytes were requested."
        assert self.num_perm % self.tree_count == 0, f"The given 'num_perm' ({num_perm}) must be divisible by 'tree_count' ({tree_count})."
        self.tree_depth = self.num_perm // self.tree_count
        self.total_hash_bytes = self.hash_bytes*self.tree_depth
        
        self.tables = [defaultdict(list) for _ in range(self.tree_count)] # usage: tables[tree_idx] -> dict[prefix] -> list of idx of entries with that prefix in the tree
        self.data : dict[int, LSHEntry] = {} # usage: data[set_id] -> entry
        # sorted array implementation for the prefix trees
        self.sorted_tables = [[] for _ in range(self.tree_count)] # usage: sorted_tables[tree_idx] -> sorted list of prefixes in the tree
        self.sorted = [True for _ in range(self.tree_count)]
        self.id_counter = 0

    """
    Computes the MinHash signature of a weighted set (dict element->weight).
    Return a list of bytes of length 'self.num_perm'.
    """
    def _weighted_minhash_signature(self, weighted_set : dict[T, float]) -> tuple[bytes, ...]:
        # TODO: devise a good weighted implementation!!!!!
        # UNWEIGHTED: minimum among the hashes of the set's values.
        #return [min((int.from_bytes(hashlib.sha1(str((k, per)).encode()).digest()[:self.hash_bytes], 'big') for k in weighted_set.keys()), default = random.randbytes(20).hex()) for per in range(self.num_perm)]
        #return b''.join(min((hashlib.sha1(str((k, per)).encode()).digest()[:self.hash_bytes] for k in weighted_set.keys()), default = random.randbytes(self.hash_bytes)) for per in range(self.num_perm))
        #return tuple(b''.join(min((hashlib.sha1(str((k, p, t)).encode()).digest()[:self.hash_bytes] for k in weighted_set.keys()), default = random.randbytes(self.hash_bytes)) for p in range(self.tree_depth)) for t in range(self.tree_count))
        
        keys = [str(k) for k in weighted_set.keys()]
        if len(keys) == 0:
            return tuple(random.randbytes(self.total_hash_bytes) for _ in range(self.tree_count))

        result = []
        for t in range(self.tree_count):
            # pre-compute hashes of the right lenght
            # NOTE: we loose a bit of hash independence by using different chunks of the same longer hash for different tree depths...
            key_digests = {}
            for k in keys:
                out = bytearray()
                i = 0
                while len(out) + 8 < self.total_hash_bytes:
                    out.extend(xxhash.xxh128(f"{k}_{i}_{t}").digest())
                    i += 1
                if len(out) < self.total_hash_bytes:
                    out.extend(xxhash.xxh64(f"{k}_{i}_{t}").digest())
                key_digests[k] = bytes(out[:self.total_hash_bytes])

            tree_result = bytearray()
            for p in range(self.tree_depth):
                min_digest = None
                for k in keys:
                    digest = key_digests[k]
                    start = p * self.hash_bytes
                    end = start + self.hash_bytes
                    h = digest[start:end]
                    if min_digest is None or h < min_digest:
                        min_digest = h
                tree_result.extend(min_digest)
            result.append(bytes(tree_result))

        return tuple(result)

    """
    Return the original set dictionary for a given ID.
    """
    def get(self, set_id : int) -> LSHEntry:
        if set_id in self.data:
            return self.data[set_id]
        else:
            raise Exception(f"The provided set ID {set_id} does not exist.")

    """
    Return the IDs present in the datastructure.
    """
    def ids(self) -> Iterable[int]:
        return self.data.keys()

    """
    Insert a new weighted set into the index.
    If no 'set_id' is provided, an internal counter is used.
    The final 'set_id' is returned.
    If 'immediate_sort' is True and trees are already sorted, this inserts the new element
    with a binary search and a cost of 'log n' instead of a constant. Otherwise, a full
    sort will be done upon the first query being requested.
    """
    def insert(self, weighted_set : dict[T, float], set_id : Optional[int] = None, merge_count : int = 1, immediate_sort : bool = False) -> int:
        if set_id is None:
            set_id = self.id_counter
            self.id_counter += 1

        signature = self._weighted_minhash_signature(weighted_set)
        self.data[set_id] = LSHEntry(weighted_set, signature, merge_count)

        for t, prefix, table, sorted_table in zip(range(self.tree_count), signature, self.tables, self.sorted_tables):
            table[prefix].append(set_id)
            if immediate_sort and self.sorted[t]:
                idx = self._binary_search(len(sorted_table), lambda x : sorted_table[x] >= prefix)
                sorted_table.insert(idx, prefix)
            else:
                sorted_table.append(prefix)
                self.sorted[t] = False

        return set_id

    """
    Delete a set from the index.
    Returns True if the element was present and thus deleted.
    """
    def delete(self, set_id : int) -> bool:
        if set_id not in self.data:
            return False

        signature = self.data[set_id].signature
        for t, prefix, table, sorted_table in zip(range(self.tree_count), signature, self.tables, self.sorted_tables):
            if len(table[prefix]) == 1:
                del table[prefix]
            else:
                table[prefix].remove(set_id)
            if self.sorted[t]:
                idx = self._binary_search(len(sorted_table), lambda x : sorted_table[x] >= prefix)
                sorted_table.pop(idx)
                # no need for re-sorting here!
            else:
                sorted_table.remove(prefix)

        del self.data[set_id]
        return True

    """
    Merge multiple sets already stored in the structure and insert the merged version.
    The old sets are deleted. Returns new set's ID.
    """
    def merge(self, set_ids : list[int], merged_set_id : int = None) -> int:
        merged = {}
        mcount = 0
        for sid in set_ids:
            if sid not in self.data:
                raise Exception(f"The provided set ID {sid} does not exist.")
            wset = self.data[sid].weighted_set
            mcount += self.data[sid].merge_count
            for k, v in wset.items():
                # WARNING, options are:
                # 1) adding weights: unfair if the set entry was generated from the same hyperedge, as it would count twice;
                # 2) taking the maximum: underplays the relevance of sharing synapses multiple times, but arguably that shouldn't be a reason to pick which merge to do;
                # 3) computing the average: costly to track.
                merged[k] = max(merged.get(k, 0), v)
        for sid in set_ids:
            self.delete(sid)
        return self.insert(merged, merge_count = mcount, set_id = merged_set_id, immediate_sort = True)

    """
    Sort the prefix trees (lists).
    Must be called before running a query.
    """
    def _sort(self) -> None:
        for t in range(self.tree_count):
            if not self.sorted[t]:
                self.sorted_tables[t].sort()
                self.sorted[t] = True

    """
    Binary search on the range [0, n].
    The callable 'func' is responsible for, given an index, returning True
    if the value at the provided index is >= than the searched value.
    Thus, the access to the searched data structure shall be done in 'func'.
    The searched value shall also be known only inside 'func'.
    """
    def _binary_search(self, n : int, func : Callable[[int], bool]) -> int:
        i, j = 0, n
        while i < j:
            h = int(i + (j - i) / 2)
            if not func(h):
                i = h + 1
            else:
                j = h
        return i

    """
    Support method for 'query'.
    Searches all trees and returns all entries with a prefix equal to signature[:depth].
    This equates to all children of the node signature[:depth] in the prefix trees.
    """
    def _query(self, signature : list[bytes], depth : int) -> Generator[int, None, None]:
        if depth > self.tree_depth or depth <= 0:
            raise ValueError("Depth outside valid range.")
        # generate prefixes of concatenated hash values
        prefixes = map(lambda p : p[:depth], signature)
        for sorted_table, prefix, table in zip(self.sorted_tables, prefixes, self.tables):
            i = self._binary_search(len(sorted_table), lambda x : sorted_table[x][:depth] >= prefix)
            if i < len(sorted_table) and sorted_table[i][:depth] == prefix:
                j = i
                while j < len(sorted_table) and sorted_table[j][:depth] == prefix:
                    for id in table[sorted_table[j]]:
                        yield id
                    j += 1

    """
    Query similar sets.
    Use 'validity_condition' to check if a candidate 'Entry' is a valid
    candidate for fusions with 'weighted_set' wr.r.t. constraints. If
    'count_invalid' is not 0, the number of invalid candidates will be
    divided by 'count_invalid' and then count towards the 'top_k'.
    If 'top_k' is set, it returns up to 'top_k' most similar sets.
    Returns the candidate IDs and optionally the list of their distances.
    """
    def query(self, weighted_set : dict[T, float], validity_condition = Callable[[LSHEntry], bool], top_k : int = 1, count_invalid : int = 0, compute_scores : bool = False) -> tuple[list[int], Optional[list[float]]]:
        self._sort()
        if top_k <= 0:
            raise ValueError("Top-k must be strictly positive.")
        
        signature = self._weighted_minhash_signature(weighted_set)
        candidates = set()
        invalid_candidates = set()
        depth = self.tree_depth
        while depth > 0 and len(candidates) + (len(invalid_candidates)//count_invalid if count_invalid else 0) < top_k:
            for id in self._query(signature, depth):
                if id not in invalid_candidates:
                    if validity_condition(self.data[id]):
                        candidates.add(id)
                    else:
                        invalid_candidates.add(id)
                if len(candidates) >= top_k:
                    break
            depth -= 1
        candidates = list(candidates)
        
        if compute_scores:
            scored = []
            for cid in candidates:
                other_set = self.data[cid].weighted_set
                score = self._weighted_jaccard(weighted_set, other_set)
                scored.append(score)
            return candidates, scored
        else:
            return candidates, None

    """
    Query similar sets using an existing stored set.
    """
    def query_by_id(self, set_id : int, validity_condition = Callable[[LSHEntry], bool], top_k : int = 1, count_invalid : int = 0, compute_scores : bool = False) -> tuple[list[int], list[float]]:
        if set_id not in self.data:
            raise Exception(f"The provided set ID {set_id} does not exist.")
        target_set = self.data[set_id].weighted_set
        candidates, scores = self.query(target_set, validity_condition = validity_condition, top_k = top_k, count_invalid = count_invalid, compute_scores = compute_scores)
        try:
            idx = candidates.index(set_id)
            candidates.pop(idx)
            scores.pop(idx)
        except ValueError:
            pass
        return candidates, scores

    """
    Compute the weighted Jaccard similarity between two sets.
    """
    def _weighted_jaccard(self, set1 : dict[T, float], set2: dict[T, float]) -> float:
        set1_only = sum(v for k, v in set1.items() if k not in set2)
        set2_only = sum(v for k, v in set2.items() if k not in set1)
        intersection = sum(v + set2[k] for k, v in set1.items() if k in set2)
        union = set1_only + set2_only + intersection
        return intersection / union if union > 0 else 0.0
    
    def __contains__(self, key : int) -> bool:
        return key in self.data
    
    def __len__(self) -> int:
        return len(self.data)

"""
Like 'WeightedMinHashLSHForest', but built via permanently sorted lists having O(log(n)) insert and delete complexity.
"""
class WeightedMinHashLSHSortedForest(Generic[T]):
    def __init__(self, num_perm : int = 256, tree_count : int = 8, hash_bytes : int = 4):
        self.num_perm = num_perm
        self.tree_count = tree_count
        self.hash_bytes = hash_bytes
        assert self.hash_bytes <= 8, f"Maximum supported hash lenght is 16 bytes, {hash_bytes} bytes were requested."
        assert self.num_perm % self.tree_count == 0, f"The given 'num_perm' ({num_perm}) must be divisible by 'tree_count' ({tree_count})."
        self.tree_depth = self.num_perm // self.tree_count
        self.total_hash_bytes = self.hash_bytes*self.tree_depth
        
        self.tables = [defaultdict(list) for _ in range(self.tree_count)] # usage: tables[tree_idx] -> dict[prefix] -> list of idx of entries with that prefix in the tree
        self.data : dict[int, LSHEntry] = {} # usage: data[set_id] -> entry
        # sorted array implementation for the prefix trees
        self.sorted_tables = [SortedList() for _ in range(self.tree_count)] # usage: sorted_tables[tree_idx] -> sorted list of prefixes in the tree
        self.id_counter = 0

    """
    Computes the MinHash signature of a weighted set (dict element->weight).
    Return a list of bytes of length 'self.num_perm'.
    """
    def _weighted_minhash_signature(self, weighted_set : dict[T, float]) -> tuple[bytes, ...]:
        # TODO: this is just a sketchy weighted implementation that makes the choice of "min hash" more likely for high-value set entries...
        min_val = min(weighted_set.values())
        max_val = max(weighted_set.values())
        interval_min, interval_max = 1, 10
        normalized_weighted_set = {str(k): interval_min + (interval_max - interval_min) * (v - min_val) / (max_val - min_val) if max_val != min_val else (interval_min + interval_min) / 2 for k, v in weighted_set.items()}
        if len(normalized_weighted_set) == 0:
            # TODO: not a great idea to use random for empty sets, should we use a default that makes them all identical?
            return tuple(random.randbytes(self.total_hash_bytes) for _ in range(self.tree_count))

        result = []
        for t in range(self.tree_count):
            # pre-compute hashes of the right lenght
            # NOTE: we loose a bit of hash independence by using different chunks of the same longer hash for different tree depths...
            key_digests : dict[str, bytes] = {}
            for k in normalized_weighted_set:
                out = bytearray()
                i = 0
                while len(out) + 8 < self.total_hash_bytes:
                    out.extend(xxhash.xxh128(f"{k}_{i}_{t}").digest())
                    i += 1
                if len(out) < self.total_hash_bytes:
                    out.extend(xxhash.xxh64(f"{k}_{i}_{t}").digest())
                key_digests[k] = bytes(out[:self.total_hash_bytes])

            tree_result = bytearray()
            for p in range(self.tree_depth):
                min_digest = None
                min_score = None
                for k in normalized_weighted_set:
                    digest = key_digests[k]
                    start = p * self.hash_bytes
                    end = start + self.hash_bytes
                    h = digest[start:end]
                    s = int.from_bytes(h) / normalized_weighted_set[k]
                    if min_digest is None or s < min_score:
                        min_digest = h
                        min_score = s
                tree_result.extend(min_digest)
            result.append(bytes(tree_result))

        return tuple(result)

    """
    Return the original set dictionary for a given ID.
    """
    def get(self, set_id : int) -> LSHEntry:
        if set_id in self.data:
            return self.data[set_id]
        else:
            raise Exception(f"The provided set ID {set_id} does not exist.")

    """
    Return the IDs present in the datastructure.
    """
    def ids(self) -> Iterable[int]:
        return self.data.keys()

    """
    Insert a new weighted set into the index.
    If no 'set_id' is provided, an internal counter is used.
    The final 'set_id' is returned.
    If 'immediate_sort' is True and trees are already sorted, this inserts the new element
    with a binary search and a cost of 'log n' instead of a constant. Otherwise, a full
    sort will be done upon the first query being requested.
    """
    def insert(self, weighted_set : dict[T, float], set_id : Optional[int] = None, merge_count : int = 1) -> int:
        if set_id is None:
            set_id = self.id_counter
            self.id_counter += 1

        signature = self._weighted_minhash_signature(weighted_set)
        self.data[set_id] = LSHEntry(weighted_set, signature, merge_count)

        for prefix, table, sorted_table in zip(signature, self.tables, self.sorted_tables):
            table[prefix].append(set_id)
            sorted_table.add(prefix)

        return set_id

    """
    Like 'insert', but takes directly an 'LSHEntry'.
    """
    def insertEntry(self, entry : LSHEntry, set_id : Optional[int] = None) -> int:
        if set_id is None:
            set_id = self.id_counter
            self.id_counter += 1

        self.data[set_id] = entry

        for prefix, table, sorted_table in zip(entry.signature, self.tables, self.sorted_tables):
            table[prefix].append(set_id)
            sorted_table.add(prefix)

        return set_id

    """
    Delete a set from the index.
    Returns True if the element was present and thus deleted.
    """
    def delete(self, set_id : int) -> bool:
        if set_id not in self.data:
            return False

        signature = self.data[set_id].signature
        for prefix, table, sorted_table in zip(signature, self.tables, self.sorted_tables):
            if len(table[prefix]) == 1:
                del table[prefix]
            else:
                table[prefix].remove(set_id)
            sorted_table.discard(prefix)

        del self.data[set_id]
        return True

    """
    Merge multiple sets already stored in the structure and insert the merged version.
    The old sets are deleted. Returns new set's ID.
    """
    def merge(self, set_ids : list[int], merged_set_id : int = None) -> int:
        merged = {}
        mcount = 0
        for sid in set_ids:
            if sid not in self.data:
                raise Exception(f"The provided set ID {sid} does not exist.")
            wset = self.data[sid].weighted_set
            mcount += self.data[sid].merge_count
            for k, v in wset.items():
                # WARNING, options are:
                # 1) adding weights: unfair if the set entry was generated from the same hyperedge, as it would count twice;
                # 2) taking the maximum: underplays the relevance of sharing synapses multiple times, but arguably that shouldn't be a reason to pick which merge to do;
                # 3) computing the average: costly to track.
                merged[k] = max(merged.get(k, 0), v)
        for sid in set_ids:
            self.delete(sid)
        return self.insert(merged, merge_count = mcount, set_id = merged_set_id)

    """
    Support method for 'query'.
    Searches all trees and returns all entries with a prefix equal to signature[:depth].
    This equates to all children of the node signature[:depth] in the prefix trees.
    """
    def _query(self, signature : list[bytes], depth : int) -> Generator[int, None, None]:
        if depth > self.tree_depth or depth <= 0:
            raise ValueError("Depth outside valid range.")
        # generate prefixes of concatenated hash values
        upper_prefixes = map(lambda p : p[:depth] + bytes([0xFF] * depth), signature)
        lower_prefixes = map(lambda p : p[:depth] + bytes(depth), signature)
        for sorted_table, lower_prefix, upper_prefix, table in zip(self.sorted_tables, lower_prefixes, upper_prefixes, self.tables):
            for prefix in sorted_table.irange(lower_prefix, upper_prefix):
                for id in table[prefix]:
                    yield id

    """
    Query similar sets.
    Use 'validity_condition' to check if a candidate 'Entry' is a valid
    candidate for fusions with 'weighted_set' wr.r.t. constraints. If
    'count_invalid' is not 0, the number of invalid candidates will be
    divided by 'count_invalid' and then count towards the 'top_k'.
    If 'top_k' is set, it returns up to 'top_k' most similar sets.
    Returns the candidate IDs and optionally the list of their distances.
    """
    def query(self, weighted_set : dict[T, float], validity_condition = Callable[[LSHEntry], bool], top_k : int = 1, count_invalid : int = 0, compute_scores : bool = False) -> tuple[list[int], Optional[list[float]]]:
        if top_k <= 0:
            raise ValueError("Top-k must be strictly positive.")
        
        signature = self._weighted_minhash_signature(weighted_set)
        candidates = set()
        invalid_candidates = set()
        depth = self.tree_depth
        while depth > 0 and len(candidates) + (len(invalid_candidates)//count_invalid if count_invalid else 0) < top_k:
            for id in self._query(signature, depth):
                if id not in invalid_candidates:
                    if validity_condition(self.data[id]):
                        candidates.add(id)
                    else:
                        invalid_candidates.add(id)
                if len(candidates) >= top_k:
                    break
            depth -= 1
        candidates = list(candidates)
        
        if compute_scores:
            scored = []
            for cid in candidates:
                other_set = self.data[cid].weighted_set
                score = self._weighted_jaccard(weighted_set, other_set)
                scored.append(score)
            return candidates, scored
        else:
            return candidates, None

    """
    Query similar sets using an existing stored set.
    """
    def query_by_id(self, set_id : int, validity_condition = Callable[[LSHEntry], bool], top_k : int = 1, count_invalid : int = 0, compute_scores : bool = False) -> tuple[list[int], list[float]]:
        if set_id not in self.data:
            raise Exception(f"The provided set ID {set_id} does not exist.")
        target_set = self.data[set_id].weighted_set
        candidates, scores = self.query(target_set, validity_condition = validity_condition, top_k = top_k, count_invalid = count_invalid, compute_scores = compute_scores)
        try:
            idx = candidates.index(set_id)
            candidates.pop(idx)
            scores.pop(idx)
        except ValueError:
            pass
        return candidates, scores

    """
    Compute the weighted Jaccard similarity between two sets.
    """
    def _weighted_jaccard(self, set1 : dict[T, float], set2: dict[T, float]) -> float:
        set1_only = sum(v for k, v in set1.items() if k not in set2)
        set2_only = sum(v for k, v in set2.items() if k not in set1)
        intersection = sum(v + set2[k] for k, v in set1.items() if k in set2)
        union = set1_only + set2_only + intersection
        return intersection / union if union > 0 else 0.0
    
    def __contains__(self, key : int) -> bool:
        return key in self.data
    
    def __len__(self) -> int:
        return len(self.data)

"""
Priority queue with dictionary-like content addressability.
Complexity:
- insertion: O(log n)
- max extraction: O(log n)
- deletion by key (via lazy removal): O(1)

Arguments:
- get_value: a function that given a 'value' of this data structure returns its corresponding priority.
             Defaults to the identity function.
"""
class AddressableMaxPQ(MutableMapping[T, U]):
    def __init__(self, get_value: Callable[[U], float] = lambda x : x):
        self._heap: list[tuple[float, int, T]] = [] # (-priority, counter, key)
        self._entry_finder: dict[T, U] = {} # key -> value
        self._counter = 0 # unique counter to break ties
        self._get_value = get_value
        self._REMOVED = object() # marker for removed keys

    """
    Insert or update an item.
    """
    def __setitem__(self, key: T, value: U):
        if key in self._entry_finder:
            self.__delitem__(key)
        priority = -self._get_value(value)
        heapq.heappush(self._heap, (priority, self._counter, key))
        self._counter += 1
        self._entry_finder[key] = value

    def __getitem__(self, key: T) -> U:
        return self._entry_finder[key]

    """
    Mark an entry as removed.
    Lazy removal performed during pop/peek.
    """
    def __delitem__(self, key: T):
        if key not in self._entry_finder:
            raise KeyError(key)
        self._entry_finder.pop(key)
        heapq.heappush(self._heap, (float('-inf'), self._counter, self._REMOVED))
        self._counter += 1

    def __contains__(self, key: object) -> bool:
        return key in self._entry_finder

    def __len__(self) -> int:
        return len(self._entry_finder)

    def __iter__(self):
        return iter(self._entry_finder)

    """
    Remove and return the item with the highest priority.
    """
    def popMax(self) -> tuple[T, U]:
        while self._heap:
            _, _, key = heapq.heappop(self._heap)
            if key is not self._REMOVED and key in self._entry_finder:
                value = self._entry_finder.pop(key)
                return key, value
        raise KeyError("Priority queue is empty.")

    """
    Return the highest-priority item without removing it.
    """
    def peekMax(self) -> tuple[T, U]:
        while self._heap:
            _, _, key = self._heap[0]
            if key is not self._REMOVED and key in self._entry_finder:
                return key, self._entry_finder[key]
            heapq.heappop(self._heap) # discard stale entry
        raise KeyError("Priority queue is empty.")