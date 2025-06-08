from typing import Optional

from collections import defaultdict, Counter
import random
import heapq
import math

from utils import *
from snn import *

# NOTE:
# This is an hypergraph partitioning problem, the goal is to minimize the connectivity (or λ − 1) metric,
# that is the total weight (for us the spike frequency) of hyperedges that got cut in the process, counting
# each hyperedge's weight once for each time it got cut.
#
# Techniques:
# - greedy appraoch.
# - multilevel approach.
# - multilevel, multistart, approach with refinement.
# - KaHyPar hierarchical hypergraph partitioning (no good: it does not accept constraints on inbound edges).

"""
Partition a directed hypergraph while minimizing the objective function given by the
connectivity (or λ - 1) metric, that means each hyperedge has a weight and suchweight
is paid once every time the hyperedge is cut, multiple cuts on the same one pay once each.
Constraints:
- each partition cannot exceed a maximum number N of nodes;
- each partition cannot exceed a number M of distinct inbound hyperedges;
- the number of partitions is not known in advance, but a maximum number K of partitions
  cannot be exceeded, while any lower amount that minimizes the objective is acceptable.

Partitions are numbered from 0, the result is a list of one partition index per node of
the hypergraph, assigning it to its partition.
"""
def partitionGreedy(hg: HyperGraph, N: int, M: int, K: int) -> list[int]:
    node_to_edges = defaultdict(list)
    for i, he in enumerate(hg.hyperedges):
        for node in he:
            node_to_edges[node].append(i)

    partitions = []  # Each partition is a dict: {'nodes': set, 'in_edges': set}
    node_to_partition = [-1] * hg.nodes  # Output assignment

    for node in range(hg.nodes):
        best_partition = -1
        best_score = float('inf')
        candidate_edges = node_to_edges[node]

        for pid, p in enumerate(partitions):
            if len(p['nodes']) >= N:
                continue

            new_in_edges = set()
            for ei in candidate_edges:
                he = hg.hyperedges[ei]
                if node in he.destinations() and ei not in p['in_edges']:
                    new_in_edges.add(ei)

            if len(p['in_edges']) + len(new_in_edges) > M:
                continue

            cut_cost = 0
            for ei in candidate_edges:
                he = hg.hyperedges[ei]
                other_parts = set()
                for n in he:
                    if n == node:
                        continue
                    if node_to_partition[n] != -1:
                        other_parts.add(node_to_partition[n])
                if len(other_parts) > 0 and pid not in other_parts:
                    cut_cost += he.spike_frequency

            if cut_cost < best_score:
                best_score = cut_cost
                best_partition = pid

        if best_partition == -1:
            if len(partitions) >= K:
                raise Exception("Exceeded maximum number of partitions K.")
            best_partition = len(partitions)
            partitions.append({'nodes': set(), 'in_edges': set()})

        p = partitions[best_partition]
        p['nodes'].add(node)
        for ei in candidate_edges:
            he = hg.hyperedges[ei]
            if node in he.destinations():
                p['in_edges'].add(ei)
        node_to_partition[node] = best_partition

    return node_to_partition

"""
Merges original nodes by heavy-edge matching until coarse node count <= max_coarse_nodes.
Returns (coarse_graph, coarse_groups), where 'coarse_groups[i]' lists original nodes merged into coarse node 'i'.
"""
def coarsen_hypergraph(hypergraph: HyperGraph, max_coarse_nodes: int, max_inbound_edges: int) -> tuple[HyperGraph, list[list[int]]]:
    parent = list(range(hypergraph.nodes))
    groups = {i: [i] for i in range(hypergraph.nodes)}

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    # Count shared weight between node pairs
    pair_weights: Counter = Counter()
    for edge in hypergraph.hyperedges:
        for i in range(1, len(edge.nodes)):
            for j in range(i+1, len(edge.nodes)):
                pair = tuple(sorted((edge.nodes[i], edge.nodes[j])))
                pair_weights[pair] += edge.spike_frequency

    # Merge strongest pairs
    for (u, v), _ in sorted(pair_weights.items(), key=lambda x: -x[1]):
        ru, rv = find(u), find(v)
        if ru != rv and len(groups[ru]) + len(groups[rv]) <= max_coarse_nodes:
            # check original‐graph inbound‐edges of the would‐be merged group
            merged = groups[ru] + groups[rv]
            in_eids = {
                eid
                for eid, edge in enumerate(hypergraph.hyperedges)
                # only count edges whose source is outside 'merged' and where at least one destination lies inside 'merged'
                if edge.source() not in merged
                   and any(d in merged for d in edge.destinations())
            }
            if len(in_eids) > max_inbound_edges:
                continue
            parent[rv] = ru
            groups[ru].extend(groups[rv])
            del groups[rv]
        if len(groups) <= max_coarse_nodes:
            break

    # Build coarse-node → original-node mapping
    coarse_groups = list(groups.values())
    node_to_coarse = [0] * hypergraph.nodes
    for coarse_idx, orig_list in enumerate(coarse_groups):
        for orig in orig_list:
            node_to_coarse[orig] = coarse_idx

    # Build coarse hyperedges
    seen = set()
    coarse_hes: list[tuple[int, ...]] = []
    coarse_freqs: list[float] = []
    for edge in hypergraph.hyperedges:
        src = node_to_coarse[edge.source()]
        dsts = tuple({
            node_to_coarse[d]
            for d in edge.destinations()
            if node_to_coarse[d] != src
        })
        if not dsts:
            continue
        key = (src,) + dsts
        if key in seen:
            continue
        seen.add(key)
        coarse_hes.append(key)
        coarse_freqs.append(edge.spike_frequency)

    coarse_graph = HyperGraph(len(coarse_groups), coarse_hes, coarse_freqs)
    return coarse_graph, coarse_groups

"""
FM-style (Fiduccia-Mattheyses) refinement on the full hypergraph.
Starts from 'coarse_assignment' and 'coarse_groups', returns a refined list of length 'hypergraph.nodes'.
"""
def refine_partition_FM(hypergraph: HyperGraph, coarse_assignment: list[int], coarse_groups: list[list[int]], max_nodes: int, max_inbound_edges: int) -> list[int]:
    # 1) Initialize original-node → partition
    node_to_partition = [-1] * hypergraph.nodes
    partitions = defaultdict(set)
    inbound_edges = defaultdict(set)

    for c_idx, orig_nodes in enumerate(coarse_groups):
        pid = coarse_assignment[c_idx]
        for u in orig_nodes:
            node_to_partition[u] = pid
            partitions[pid].add(u)

    # 2) Build node -> edges
    node_to_edges = defaultdict(list)
    for eid, edge in enumerate(hypergraph.hyperedges):
        for u in edge:
            node_to_edges[u].append(eid)
    # 2.5) Build true inbound_edges: only edges whose source is outside and at least one destination is inside the partition
    for eid, edge in enumerate(hypergraph.hyperedges):
        src = edge.source()
        for pid in { node_to_partition[d] for d in edge.destinations() }:
            if pid >= 0 and node_to_partition[src] != pid:
                inbound_edges[pid].add(eid)

    # 3) Compute initial gains
    heap: list[tuple[float, int, int]] = []
    moved = set()
    for u in range(hypergraph.nodes):
        cur_pid = node_to_partition[u]
        # neighbors’ partitions
        neighbor_pids = {
            node_to_partition[v]
            for eid in node_to_edges[u]
            for v in hypergraph.hyperedges[eid]
            if node_to_partition[v] != cur_pid
        }
        for tgt_pid in neighbor_pids:
            gain = 0.0
            for eid in node_to_edges[u]:
                edge = hypergraph.hyperedges[eid]
                before = {node_to_partition[v] for v in edge}
                cut_before = len(before) > 1
                after = (before - {cur_pid}) | {tgt_pid}
                cut_after = len(after) > 1
                gain += edge.spike_frequency * (cut_before - cut_after)
            if gain > 0:
                heapq.heappush(heap, (-gain, u, tgt_pid))

    # 4) Apply highest-gain moves
    while heap:
        _, u, tgt_pid = heapq.heappop(heap)
        if u in moved:
            continue
        cur_pid = node_to_partition[u]
        if tgt_pid == cur_pid:
            continue

        # Enforce max_nodes AT EXECUTION time
        if len(partitions[tgt_pid]) + 1 > max_nodes:
            continue

        # Enforce inbound-edges
        extra = {
            eid for eid in node_to_edges[u]
            if u in hypergraph.hyperedges[eid].destinations()
               and hypergraph.hyperedges[eid].source() != tgt_pid
        }
        new_in = inbound_edges[tgt_pid] | extra
        if len(new_in) > max_inbound_edges:
            continue

        # Do the move
        partitions[cur_pid].remove(u)
        partitions[tgt_pid].add(u)
        inbound_edges[tgt_pid] = new_in
        node_to_partition[u] = tgt_pid
        moved.add(u)

    return node_to_partition

"""
Hypergraph partitioning algorithm made of three steps:
1) coarseneing, merges together nodes that have the heaviest connections between them.
2) greedy λ - 1 partitioning of coarsened graph.
3) partitioning refinement through the Fiduccia-Mattheyses algorithm.

Arguments:
- hg: hypergraph to partition.
- max_nodes: maximum number of nodes that each partition can be assigned.
- max_inbound_edges: maximum number of hyperedges that can be inbound to any partition.
- max_partitions: maximum number of allowed partitions.

Returns:
A list with an entry per hypergraph node, that is the node's assigned partition's index.
"""
def partitionGreedyMultilevelRefined(hg: HyperGraph, max_nodes: int, max_inbound_edges: int, max_partitions: int) -> list[int]:
    # Step 1: Coarsen
    coarse_hg, groupings = coarsen_hypergraph(hg, min(max_partitions * max_nodes // 2, hg.nodes // 10), max_inbound_edges)
    # Step 2: Initial Partitioning
    coarse_partition = partitionGreedy(coarse_hg, max_nodes, max_inbound_edges, max_partitions)
    # Step 3: Refinement
    final_partition = refine_partition_FM(hg, coarse_partition, groupings, max_nodes, max_inbound_edges)

    return final_partition

"""
Multistart version of the 'partitionGreedyMultilevelRefined' partitioning algorithm.
"""
def partitionGreedyMultilevelRefinedMultistart(hg: HyperGraph, max_nodes: int, max_inbound_edges: int, max_partitions: int, multistarts: int = 3, seed : Optional[int] = None) -> list[int]:
    rng = random.Random(seed)
    best_assign = []
    best_cut = float('inf')

    for _ in range(multistarts):
        # 1) Coarsen
        coarse_graph, coarse_groups = coarsen_hypergraph(hg, max_nodes, max_inbound_edges)

        # 2) Compute coarse-node weights = group sizes
        coarse_weights = [len(g) for g in coarse_groups]

        # 3) Greedy initial on coarse graph
        indices = list(range(coarse_graph.nodes))
        rng.shuffle(indices)
        coarse_assignment = [-1] * coarse_graph.nodes
        part_weight = []
        part_inbound = []

        # Precompute coarse node -> edge list
        coarse_node_to_edges = defaultdict(list)
        for eid, edge in enumerate(coarse_graph.hyperedges):
            for u in edge:
                coarse_node_to_edges[u].append(eid)

        for u in indices:
            w_u = coarse_weights[u]
            best_pid, best_cost = -1, float('inf')

            for pid, w_sum in enumerate(part_weight):
                if w_sum + w_u > max_nodes:
                    continue
                in_u = {eid for eid in coarse_node_to_edges[u]
                        if u in coarse_graph.hyperedges[eid].destinations()}
                if len(part_inbound[pid] | in_u) > max_inbound_edges:
                    continue

                cut = 0.0
                for eid in coarse_node_to_edges[u]:
                    edge = coarse_graph.hyperedges[eid]
                    assigned = {
                        coarse_assignment[v]
                        for v in edge
                        if coarse_assignment[v] != -1
                    }
                    if assigned and pid not in assigned:
                        cut += edge.spike_frequency
                if cut < best_cost:
                    best_cost, best_pid = cut, pid

            if best_pid == -1:
                if len(part_weight) >= max_partitions:
                    raise RuntimeError("Exceeded max_partitions")
                best_pid = len(part_weight)
                part_weight.append(0)
                part_inbound.append(set())

            # Assign u
            part_weight[best_pid] += w_u
            in_u = {eid for eid in coarse_node_to_edges[u]
                    if u in coarse_graph.hyperedges[eid].destinations()}
            part_inbound[best_pid].update(in_u)
            coarse_assignment[u] = best_pid

        # 4) FM refinement on the full graph
        final_assignment = refine_partition_FM(hg, coarse_assignment, coarse_groups, max_nodes, max_inbound_edges)

        # 5) Evaluate cut quality
        cut_value = sum(edge.spike_frequency for edge in hg.hyperedges if len({final_assignment[v] for v in edge}) > 1)

        if cut_value < best_cut:
            best_cut = cut_value
            best_assign = final_assignment

    return best_assign

"""
Simple sequential partitioning algorithm, assigns nodes to the same partition until a constraint
would be violated, then creates and starts filling the next partition.

Used by the Ouwen Jin paper.
"""
def partitionSequential(hg: HyperGraph, N: int, M: int, K: int) -> list[int]:
    partitioning = []
    inbound_edges = 0
    assigned_nodes = 0
    current_partition = 0
    for node in range(hg.nodes):
        assigned_nodes += 1
        inbound_edges += sum(1 for he in hg.hyperedges if node in he.destinations())
        if assigned_nodes > N or inbound_edges > M:
            assigned_nodes = 0
            inbound_edges = 0
            current_partition += 1
            if current_partition >= K:
                raise Exception("Exceeded maximum number of partitions K.")
        partitioning.append(current_partition)
    return partitioning

"""
Swaps-based partitioning algorithm, start from a random partitioning and swaps

Implementation of the Kernighan-Lin heuristic.
Used by DFSynthesizer.

Arguments:
- hg, N, M, K as in 'partitionGreedy'.
- min_delta: minimum improvement on the total cost that justifies a move and another round.
- 
"""
def swapPartitioner(hg: HyperGraph, N: int, M: int, K: int, min_delta : float = 0.1, multistarts : int = 1) -> list[int]:
    best_partitioning = None
    best_cost = math.inf
    
    for _ in range(multistarts):
        partitioning = [0 for _ in range(hg.nodes)]
        inbound_edges = [0]
        assigned_nodes = [0]
        current_partition = 0
        nodes = list(range(hg.nodes))
        random.shuffle(nodes)
        # same logic as 'partitionSequential', but the order of nodes is randomized
        for node in nodes:
            assigned_nodes[-1] += 1
            inbound_edges[-1] += sum(1 for he in hg.hyperedges if node in he.destinations())
            if assigned_nodes[-1] > N or inbound_edges[-1] > M:
                assigned_nodes.append(0)
                inbound_edges.append(0)
                current_partition += 1
                if current_partition >= K:
                    raise Exception("Exceeded maximum number of partitions K.")
            partitioning[node] = current_partition
        
        def cost(hg : HyperGraph, partitions : list[int]) -> float:
            result = 0
            for he in hg:
                connections = set()
                for n in he.destinations():
                    connections.add(partitions[n])
                result += he.spike_frequency*len(connections)
            return result
        
        delta = math.inf
        current_cost = cost(hg, partitioning)
        while delta > min_delta:
            delta = 0
            for n in range(hg.nodes):
                for m in range(hg.nodes): # here goes the quadratic complexity...
                    part_n, part_m = partitioning[n], partitioning[m]
                    edges_n, edges_m = sum(1 for he in hg.hyperedges if n in he.destinations()), sum(1 for he in hg.hyperedges if m in he.destinations())
                    if part_n != part_m:
                        # it's a swap, assigned nodes will always be fine
                        if inbound_edges[part_n] - edges_n + edges_m <= M and inbound_edges[part_m] - edges_m + edges_n <= M:
                            partitioning[n], partitioning[m] = partitioning[m], partitioning[n]
                            new_cost = cost(hg, partitioning)
                            if new_cost >= current_cost:
                                partitioning[n], partitioning[m] = partitioning[m], partitioning[n]
                            else:
                                delta = current_cost - new_cost
                                current_cost = new_cost
        
        if current_cost < best_cost:
            best_partitioning = partitioning
            best_cost = current_cost
    if best_partitioning == None:
        raise Exception("No partitioning found, something broke.")
    return best_partitioning