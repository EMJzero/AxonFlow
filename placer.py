from collections import deque
from copy import deepcopy

from snn import *

"""
Given a directed acyclic HyperGraph, returns a new HyperHraph that has its
nodes ordered topologically: nodes with a lower index will never be reachable
from hyperedges starting from nodes with a higher index.
"""
def topologycalOrder(hg : HyperGraph) -> HyperGraph:
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
Forcibly makes a directed HyperGraphs acyclic by trying to heuristically
remove the fewest edges and among them, those with the fewest connections.
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
