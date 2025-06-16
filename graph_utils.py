from collections import deque
from copy import deepcopy

from prints import *
from snn import *

"""
Checks if the provided hypergraph has a cycle.
Complexity: O(#nodes + #hyperedges)
"""
def isAcyclic(hg : HyperGraph) -> bool:
    in_degree = [0] * hg.nodes
    outgoing = [[] for _ in range(hg.nodes)]

    # Build adjacency list and in-degree count
    for he in hg:
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

    for he in hg:
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
    for he in hg:
        new_hyperedges.append(HyperEdge(he.source(), tuple(new_index[dst] for dst in he.destinations()), he.spike_frequency))

    return HyperGraph(n, new_hyperedges)

"""
Given a directed HyperGraph, returns a new HyperHraph that has its nodes ordered topologically:
nodes with a lower index will never be reachable from hyperedges starting from nodes with a higher index.
If 'break_cycles' is True, the input graph can have cycles, and the final output will be almost in
topological order, pending ignoring a set of connection (that is also returned). Otherwise having
cycles will raise an exception.

The cycles breaking strategy works by repeatedly:
    1) finding any directed cycle (via DFS on the hypergraph lowered in a graph);
    2) removing exactly the single arc of minimal spike frequency from that cycle;
    3) repeating until no directed cycle remains.

The function returns:
- a new HyperGraph with topologically ordered nodes (or mostly ordered, if the input has cycles).
- the set of masked edges (i.e., (src, dst) pairs that were ignored when computing the ordering,
  ignoring those in the returned hypergraph makes it topologically ordered and acyclic).
"""
from collections import deque
from typing import Optional

@core
def topologicalOrder(hg: HyperGraph, break_cycles: bool = False) -> tuple[HyperGraph, list[tuple[int, int]]]:
    n = hg.nodes
    arc_src, arc_dst, arc_weight = [], [], []
    outgoing_arcs = [[] for _ in range(n)]
    active_arcs = set()

    # Flatten hyperedges into individual arcs
    for he in hg:
        s = he.source()
        for d in he.destinations():
            i = len(arc_src)
            arc_src.append(s)
            arc_dst.append(d)
            arc_weight.append(he.spike_frequency)
            outgoing_arcs[s].append(i)
            active_arcs.add(i)

    masked = []

    # Efficient DFS-based cycle finder, returns one real cycle as list of arc indices
    def find_cycle() -> Optional[list[int]]:
        color = [0] * n  # 0 = unvisited, 1 = visiting, 2 = visited
        parent_arc = [-1] * n

        for start in range(n):
            if color[start] != 0:
                continue

            stack = [(start, iter(outgoing_arcs[start]))]
            color[start] = 1

            while stack:
                u, children = stack[-1]
                try:
                    aid = next(children)
                    if aid not in active_arcs:
                        continue
                    v = arc_dst[aid]
                    if color[v] == 0:
                        parent_arc[v] = aid
                        color[v] = 1
                        stack.append((v, iter(outgoing_arcs[v])))
                    elif color[v] == 1:
                        # Cycle found
                        cycle = [aid]
                        w = u
                        while w != v:
                            pa = parent_arc[w]
                            cycle.append(pa)
                            w = arc_src[pa]
                        return cycle
                except StopIteration:
                    color[u] = 2
                    stack.pop()
        return None

    if break_cycles:
        while (cycle := find_cycle()):
            # Remove the weakest arc from the cycle
            worst = min(cycle, key=lambda i: arc_weight[i])
            active_arcs.remove(worst)
            masked.append((arc_src[worst], arc_dst[worst]))
    else:
        if find_cycle():
            raise Exception("Cycle detected and break_cycles=False")

    # Topological sort (Kahn's algorithm) on remaining active arcs
    indeg = [0] * n
    adj = [[] for _ in range(n)]
    for i in active_arcs:
        u, v = arc_src[i], arc_dst[i]
        indeg[v] += 1
        adj[u].append(v)

    queue = deque(i for i in range(n) if indeg[i] == 0)
    order, new_id = [], [0] * n

    while queue:
        u = queue.popleft()
        new_id[u] = len(order)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    if len(order) < n:
        raise Exception("Topological sort failed (graph not acyclic)")

    # Rebuild hypergraph with permuted node IDs
    new_edges = [HyperEdge(new_id[he.source()], tuple(new_id[d] for d in he.destinations()), he.spike_frequency) for he in hg.hyperedges]

    return HyperGraph(n, new_edges), masked


"""
Same functionality as 'topologicalOrder', but the logic to make the graph acyclic is weaker: it attempts to
remove the minimum-sum of spike_frequency connections, one by one, in ascending order of their spike_frequency.
"""
def topologicalOrderWeak(hg: HyperGraph, break_cycles: bool = False) -> tuple[HyperGraph, list[tuple[int, int]]]:
    n = hg.nodes
    original_in_degree = [0] * n
    outgoing = [[] for _ in range(n)]

    for he in hg:
        src = he.source()
        for dst in he.destinations():
            original_in_degree[dst] += 1
            outgoing[src].append(dst)

    from collections import Counter

    """
    Attempt a Kahn's algorithm on the directed masked hypergraph.
    If a topological order exists, it is returned.
    """
    def try_topo_sort(masked_counts: Counter) -> Optional[list[int]]:
        in_degree = original_in_degree[:]  # make a fresh copy
        for (_, dst), count in masked_counts.items():
            in_degree[dst] -= count

        queue = deque(i for i in range(n) if in_degree[i] == 0)
        topo = []
        rem_masks = masked_counts.copy()
        while queue:
            u = queue.popleft()
            topo.append(u)
            for v in outgoing[u]:
                if rem_masks.get((u, v), 0) > 0:
                    rem_masks[(u, v)] -= 1
                else:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

        return topo if len(topo) == n else None

    masked_counts = Counter() # how many copies of each (src, dst) have been masked so far
    topo = try_topo_sort(masked_counts)

    if topo is None:
        if not break_cycles:
            raise Exception("The hypergraph contains a cycle and break_cycles=False.")

        all_connections : list[tuple[tuple[int, int], float]] = []
        for he in hg:
            src = he.source()
            for dst in he.destinations():
                all_connections.append(((src, dst), he.spike_frequency))

        # Sort by ascending spike_frequency so that we remove the cheapest‐cost arcs first.
        all_connections.sort(key=lambda pair: pair[1])

        # mask them one by one until try_topo_sort succeeds.
        for (src, dst), _ in all_connections:
            masked_counts[(src, dst)] += 1
            topo = try_topo_sort(masked_counts)
            if topo is not None:
                break

        if topo is None:
            raise Exception("Failed to break cycles to obtain a topological order. (This should never happen.)")

    new_index = [0] * n
    for new_id, old_id in enumerate(topo):
        new_index[old_id] = new_id

    new_hyperedges: list[HyperEdge] = []
    for he in hg:
        old_src = he.source()
        old_dsts = he.destinations()
        new_src = new_index[old_src]
        new_dsts = tuple(new_index[dst] for dst in old_dsts)
        new_hyperedges.append(HyperEdge(new_src, new_dsts, he.spike_frequency))

    new_hg = HyperGraph(n, new_hyperedges)

    masked_list: list[tuple[int, int]] = []
    for (s, d), count in masked_counts.items():
        masked_list.extend([(s, d)] * count)

    return new_hg, masked_list

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