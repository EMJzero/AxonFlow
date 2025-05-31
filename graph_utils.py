from collections import deque
from copy import deepcopy

from snn import *

"""
Checks if the provided hypergraph has a cycle.
Complexity: O(#nodes + #hyperedges)
"""
def isAcyclic(hg : HyperGraph) -> bool:
    in_degree = [0] * hg.nodes
    outgoing = [[] for _ in range(hg.nodes)]

    # Build adjacency list and in-degree count
    for he in hg.hyperedges:
        src = he.source()
        for dst in he.destinations():
            outgoing[src].append(dst)
            in_degree[dst] += 1

    # Kahn's algorithm
    queue = deque(i for i in range(hg.nodes) if in_degree[i] == 0)
    visited_count = 0

    while queue:
        node = queue.popleft()
        visited_count += 1
        for dst in outgoing[node]:
            in_degree[dst] -= 1
            if in_degree[dst] == 0:
                queue.append(dst)

    return visited_count != hg.nodes

"""
Given a directed acyclic HyperGraph, returns a new HyperHraph that has its
nodes ordered topologically: nodes with a lower index will never be reachable
from hyperedges starting from nodes with a higher index.
"""
def acyclicTopologycalOrder(hg : HyperGraph) -> HyperGraph:
    n = hg.nodes
    in_degree = [0] * n
    outgoing = [[] for _ in range(n)]

    for he in hg.hyperedges:
        src = he.source()
        for dst in he.destinations():
            in_degree[dst] += 1
            outgoing[src].append(dst)

    # Kahn's algorithm
    queue = deque(i for i in range(n) if in_degree[i] == 0)
    topo_order = [0] * n
    order_index = 0
    while queue:
        node = queue.popleft()
        topo_order[order_index] = node
        order_index += 1
        for dst in outgoing[node]:
            in_degree[dst] -= 1
            if in_degree[dst] == 0:
                queue.append(dst)
    if order_index != n:
        raise Exception("The hypergraph contains a cycle.")

    new_index = [0] * n
    for new_id, old_id in enumerate(topo_order):
        new_index[old_id] = new_id

    new_hyperedges = []
    for he in hg.hyperedges:
        new_hyperedges.append(HyperEdge(he.source(), tuple(new_index[dst] for dst in he.destinations()), he.spike_frequency))

    return HyperGraph(n, new_hyperedges)

"""
Given a directed HyperGraph, returns a new HyperHraph that has its nodes ordered topologically:
nodes with a lower index will never be reachable from hyperedges starting from nodes with a higher index.
If 'break_cycles' is True, the input graph can have cycles, and the final output will be almost in
topological order, pending ignoring a set of connection (that is also returned). Otherwise having
cycles will raise an exception.

The function returns:
- a new HyperGraph with topologically ordered nodes (or mostly ordered, if the input has cycles).
- the set of masked edges (i.e., (src, dst) pairs that were ignored when computing the ordering,
  ignoring those in the returned hypergraph makes it topologically ordered and acyclic).
"""
def topologycalOrder(hg: HyperGraph, break_cycles: bool = False) -> tuple[HyperGraph, set[tuple[int, int]]]:
    n = hg.nodes
    original_in_degree = [0] * n
    outgoing = [[] for _ in range(n)]

    # Build initial graph structure
    for he in hg.hyperedges:
        src = he.source()
        for dst in he.destinations():
            original_in_degree[dst] += 1
            outgoing[src].append(dst)

    # Try topological sort (Kahn’s algorithm) with optional edge masking
    def try_topo_sort(masked_edges: set[tuple[int, int]]) -> Optional[list[int]]:
        in_degree = original_in_degree.copy()
        for _, dst in masked_edges:
            in_degree[dst] -= 1

        queue = deque(i for i in range(n) if in_degree[i] == 0)
        topo = []
        while queue:
            node = queue.popleft()
            topo.append(node)
            for dst in outgoing[node]:
                if (node, dst) in masked_edges:
                    continue
                in_degree[dst] -= 1
                if in_degree[dst] == 0:
                    queue.append(dst)

        return topo if len(topo) == n else None

    # First attempt: no cycle breaking
    masked_edges: set[tuple[int, int]] = set()
    topo = try_topo_sort(masked_edges)
    if topo is None:
        if not break_cycles:
            raise Exception("The hypergraph contains a cycle.")

        # Gather all connections with their spike frequencies
        all_connections = []
        for he in hg.hyperedges:
            src = he.source()
            for dst in he.destinations():
                all_connections.append(((src, dst), he.spike_frequency))

        # Sort connections by increasing spike frequency (low-cost to remove)
        all_connections.sort(key = lambda x: x[1])

        # Incrementally mask connections until a valid topological order is found
        for (src, dst), _ in all_connections:
            masked_edges.add((src, dst))
            topo = try_topo_sort(masked_edges)
            if topo is not None:
                break

        if topo is None:
            raise Exception("Failed to break cycles to obtain a topological order.")

    # Remap node indices based on topo sort
    new_index = [0] * n
    for new_id, old_id in enumerate(topo):
        new_index[old_id] = new_id

    # Rebuild hyperedges with remapped node indices — ALL original connections are preserved
    new_hyperedges = []
    for he in hg.hyperedges:
        src_old = he.source()
        dsts_old = he.destinations()
        src_new = new_index[src_old]
        dsts_new = tuple(new_index[dst] for dst in dsts_old)
        new_hyperedges.append(HyperEdge(src_new, dsts_new, he.spike_frequency))

    return HyperGraph(n, new_hyperedges), masked_edges

"""
Forcibly makes a directed HyperGraphs acyclic by trying to heuristically
remove the fewest edges and, among them, those with the fewest connections.
Returns a new HyperGraph, preserving the original.
"""
def makeAcyclic(hg: HyperGraph) -> HyperGraph:
    def is_acyclic(hyperedges: list[HyperEdge]) -> bool:
        n = hg.nodes
        in_degree = [0] * n
        for he in hyperedges:
            for dst in he.destinations():
                in_degree[dst] += 1

        queue = deque(i for i in range(n) if in_degree[i] == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for he in hyperedges:
                if he.source() == node:
                    for dst in he.destinations():
                        in_degree[dst] -= 1
                        if in_degree[dst] == 0:
                            queue.append(dst)
        return visited == n

    if is_acyclic(hg.hyperedges):
        return deepcopy(hg)
    else:
        print("WARNING: forcibly making a SNN acyclic is likely to break it.")

    indexed_hyperedges = list(enumerate(hg.hyperedges))
    indexed_hyperedges.sort(key=lambda x: (len(x[1].nodes), x[0]))
    kept_hyperedges = []
    kept_indices = []
    for idx, he in indexed_hyperedges:
        candidate = kept_hyperedges + [he]
        if is_acyclic(candidate):
            kept_hyperedges.append(he)
            kept_indices.append(idx)

    return HyperGraph(hg.nodes, [he.nodes for he in kept_hyperedges], [he.spike_frequency for he in kept_hyperedges])