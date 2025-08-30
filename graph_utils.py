from collections import deque
from typing import Optional
from copy import deepcopy
import math

from datastructures import *
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
Returns True iif the hypergraph is topologically sorted.
That is, for every hyperedge, source <= all destinations.
"""
def isTopologicallySorted(hg : HyperGraph) -> bool:
    for he in hg.hyperedges:
        src = he.source()
        if any(dst < src for dst in he.destinations()):
            return False
    return True

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
@core
def topologicalOrder(hg : HyperGraph, break_cycles : bool = False) -> tuple[HyperGraph, list[tuple[int, int]]]:
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
            worst = min(cycle, key = lambda i : arc_weight[i])
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
Similar to a topological order, ignores cycles by default.
Builds a priority queue of the yet-to-visit nodes and visits them following its order.
Starts from nodes with no inbound hyperedges, if none is present, this fails.
For each visited node, nodes connected to its outbound hyperedges are added to the
queue with their spike frequency as weight, if they are alreay there, the spike
frequency is incremented by the present amount.
Ideally requires no self-cycles to be present.
"""
@core
def feedForwardOrder(hg : HyperGraph) -> HyperGraph:
    # - addressable priority queue of nodes (partitions)
    # => upgrade it to have lazy heapify and fast update/insert
    # - initially contains node 0
    # => better: cheaply find a node from which the whole graph is reachable, start from it
    # => start from nodes with no inbound hyperedges, give them maximum cost? But if they are too many, we disrupt locality...
    # - place it, then put in the queue all nodes [it is connected to] reached from the hyperedge departing from it, ranked by spike frequency
    # - place the next highest spike frequency node, insert / update all those it is connected to in the queue by adding spike frequencies
    timerPrint = getTimerPrinter(Settings.PRINT_INTERVAL)
    pq : AddressableMaxPQ[int, float] = AddressableMaxPQ(default_factory = lambda : 0.0)
    new_id, next_id = [-1] * hg.nodes, 0
    while next_id != hg.nodes:
        min_inbound_count, min_inbound_nodes = math.inf, []
        for n, id in enumerate(new_id):
            if id == -1:
                # NOTE: not guaranteed to exist...
                inbound = hg.getInboundHyperedges(n)
                if len(inbound) == 0:
                    pq[n] = 2**31 - 1
                elif len(inbound) < min_inbound_count:
                    min_inbound_count = len(inbound)
                    min_inbound_nodes = [n]
                elif len(inbound) == min_inbound_count:
                    min_inbound_nodes.append(n)
        
        if len(pq) == 0:
            for n in min_inbound_nodes:
                pq[n] = 2**31 - 1
        
        while len(pq) > 0:
            n = pq.popMax()[0]
            new_id[n] = next_id
            next_id += 1
            for he in hg.getOutboundHyperedges(n):
                for m in he:
                    if new_id[m] == -1:
                        pq[m] += he.spike_frequency
            timerPrint(f"Reordered {next_id}/{hg.nodes} nodes...")
    
    # Rebuild hypergraph with permuted node IDs
    new_edges = [HyperEdge(new_id[he.source()], tuple(new_id[d] for d in he.destinations()), he.spike_frequency) for he in hg.hyperedges]
    return HyperGraph(hg.nodes, new_edges)

"""
Forcibly makes a directed HyperGraphs acyclic by trying to heuristically
remove the fewest edges and, among them, those with the fewest connections.
Returns a new HyperGraph, preserving the original.
"""
def makeAcyclic(hg : HyperGraph) -> HyperGraph:
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