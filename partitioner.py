from typing import Optional

from datasketch import MinHash, MinHashLSH, WeightedMinHashGenerator
from collections import defaultdict, Counter
from itertools import combinations
import random
import heapq
import math

from prints import *
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
def coarsen_hypergraph(hg: HyperGraph, max_coarse_nodes: int, max_inbound_edges: int) -> tuple[HyperGraph, list[list[int]]]:
    parent = list(range(hg.nodes))
    groups = {i: [i] for i in range(hg.nodes)}

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    # Compute heavy‐edge weights between destination pairs
    pair_weights = Counter()
    for e in hg.hyperedges:
        for u, v in combinations(e.nodes[1:], 2):
            a, b = (u, v) if u < v else (v, u)
            pair_weights[(a, b)] += e.spike_frequency

    # Merge strongest pairs
    for (u, v), _ in sorted(pair_weights.items(), key=lambda x: -x[1]):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue

        merged = groups[ru] + groups[rv]
        if len(merged) > max_coarse_nodes:
            continue

        merged_set = set(merged)
        # Gather all hyperedges that have any destination in merged_set
        candidate = sum((hg.getInboundHyperedges(n) for n in merged_set), start = [])
        hyperedges = {
            he
            for he in candidate
            if any(d in merged_set for d in he.destinations())
        }

        if len(hyperedges) > max_inbound_edges:
            continue

        # Commit merge
        parent[rv] = ru
        groups[ru].extend(groups[rv])
        del groups[rv]

        if len(groups) <= max_coarse_nodes:
            break

    # Build coarse groups and edges
    coarse_groups = list(groups.values())
    node_to_coarse = [0] * hg.nodes
    for ci, grp in enumerate(coarse_groups):
        for n in grp:
            node_to_coarse[n] = ci

    seen = set()
    coarse_hes, coarse_freqs = [], []
    for e in hg.hyperedges:
        src = node_to_coarse[e.source()]
        dsts = tuple({ node_to_coarse[d] for d in e.destinations() if node_to_coarse[d] != src })
        if not dsts:
            continue
        key = (src,) + dsts
        if key in seen:
            continue
        seen.add(key)
        coarse_hes.append(key)
        coarse_freqs.append(e.spike_frequency)

    return HyperGraph(len(coarse_groups), coarse_hes, coarse_freqs), coarse_groups

"""
FM-style (Fiduccia-Mattheyses) refinement on the full hypergraph.
Starts from 'coarse_assignment' and 'coarse_groups', returns a refined list of length 'hypergraph.nodes'.
"""
def refine_partition_FM(hg: HyperGraph, coarse_assignment: list[int], coarse_groups: list[list[int]], max_nodes: int, max_inbound_edges: int) -> list[int]:
    node_to_part = [-1] * hg.nodes
    parts = defaultdict(set)
    in_edges = defaultdict(set)

    # Initialize from coarse assignment
    for ci, grp in enumerate(coarse_groups):
        pid = coarse_assignment[ci]
        for u in grp:
            node_to_part[u] = pid
            parts[pid].add(u)

    # Build node -> incident hyperedges map
    node_to_edges = defaultdict(list)
    for eid, e in enumerate(hg.hyperedges):
        for u in e:
            node_to_edges[u].append(eid)

    # Initial inbound-edge sets: any hyperedge with at least one destination in the partition
    for eid, e in enumerate(hg.hyperedges):
        for d in e.destinations():
            pid = node_to_part[d]
            if pid >= 0:
                in_edges[pid].add(eid)

    # Build FM gain heap
    heap, moved = [], set()
    for u in range(hg.nodes):
        cur = node_to_part[u]
        neighbor_parts = {
            node_to_part[v]
            for eid in node_to_edges[u]
            for v in hg.hyperedges[eid]
            if node_to_part[v] != cur
        }
        for tgt in neighbor_parts:
            gain = 0.0
            for eid in node_to_edges[u]:
                e = hg.hyperedges[eid]
                before = {node_to_part[v] for v in e}
                after = (before - {cur}) | {tgt}
                gain += e.spike_frequency * ((len(before) > 1) - (len(after) > 1))
            if gain > 0:
                heapq.heappush(heap, (-gain, u, tgt))

    # FM refinement moves
    while heap:
        _, u, tgt = heapq.heappop(heap)
        if u in moved:
            continue
        cur = node_to_part[u]
        if cur == tgt or len(parts[tgt]) + 1 > max_nodes:
            continue

        # Simulate new inbound edges for target
        new_in = set(in_edges[tgt])
        for eid in node_to_edges[u]:
            e = hg.hyperedges[eid]
            if any(d in parts[tgt] or d == u for d in e.destinations()):
                new_in.add(eid)
        if len(new_in) > max_inbound_edges:
            continue

        # Commit move
        parts[cur].remove(u)
        parts[tgt].add(u)
        node_to_part[u] = tgt
        moved.add(u)

        # Recompute inbound edges for both affected partitions
        for pid in (cur, tgt):
            updated = set()
            for n in parts[pid]:
                for eid in node_to_edges[n]:
                    e = hg.hyperedges[eid]
                    if any(d in parts[pid] for d in e.destinations()):
                        updated.add(eid)
            in_edges[pid] = updated

    return node_to_part

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
@core
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
@core
def partitionGreedyMultilevelRefinedMultistart(hg: HyperGraph, max_nodes: int, max_inbound_edges: int, max_partitions: int, multistarts: int = 3, seed : Optional[int] = None) -> list[int]:
    rng = random.Random(seed)
    best_assign, best_cut = [], float('inf')

    for ms in range(multistarts):
        print("Multistart count:", ms)
        # 1) Coarsen
        coarse_hg, coarse_groups = coarsen_hypergraph(hg, max_nodes, max_inbound_edges)

        # 2) Greedy assign on coarse graph
        indices = list(range(coarse_hg.nodes))
        rng.shuffle(indices)
        coarse_assignment = [-1] * coarse_hg.nodes
        part_weight, part_inbound = [], []

        # Precompute coarse node -> edges
        cnode_to_edges = defaultdict(list)
        for eid, e in enumerate(coarse_hg.hyperedges):
            for u in e:
                cnode_to_edges[u].append(eid)

        for u in indices:
            w = len(coarse_groups[u])
            best_pid, best_cost = -1, float('inf')
            for pid, total_w in enumerate(part_weight):
                if total_w + w > max_nodes:
                    continue
                in_u = {eid for eid in cnode_to_edges[u]
                        if u in coarse_hg.hyperedges[eid].destinations()}
                if len(part_inbound[pid] | in_u) > max_inbound_edges:
                    continue

                cut = sum(coarse_hg.hyperedges[eid].spike_frequency
                          for eid in cnode_to_edges[u]
                          if pid not in {coarse_assignment[v] for v in coarse_hg.hyperedges[eid] if coarse_assignment[v] != -1})
                if cut < best_cost:
                    best_cost, best_pid = cut, pid

            if best_pid == -1:
                if len(part_weight) >= max_partitions:
                    raise RuntimeError("Exceeded max_partitions")
                best_pid = len(part_weight)
                part_weight.append(0)
                part_inbound.append(set())

            part_weight[best_pid] += w
            in_u = {eid for eid in cnode_to_edges[u]
                    if u in coarse_hg.hyperedges[eid].destinations()}
            part_inbound[best_pid].update(in_u)
            coarse_assignment[u] = best_pid

        # 3) FM refinement on full graph
        final_assign = refine_partition_FM(
            hg, coarse_assignment, coarse_groups, max_nodes, max_inbound_edges)

        # 4) Evaluate cut
        cut_value = sum(e.spike_frequency
                        for e in hg.hyperedges
                        if len({final_assign[v] for v in e}) > 1)

        if cut_value < best_cut:
            best_cut, best_assign = cut_value, final_assign

    return best_assign

@core
def partitionGreedyMultilevelRefinedV2(hg: HyperGraph, N: int, M: int, K: int) -> list[int]:
    n = hg.nodes
    part = [-1] * n
    node_weights = [1] * n  # Unit weights for original nodes

    # Coarsening phase
    coarse_map = list(range(n))
    coarse_weights = node_weights[:]  # Tracks weights of coarse nodes
    contraction_pairs = []
    active_nodes = n

    def heavy_edge_rating(u: int, v: int) -> float:
        common_hyperedges = set(hg.getTouchingHyperedges(u)) & set(hg.getTouchingHyperedges(v))
        rating = 0.0
        for he in common_hyperedges:
            rating += he.spike_frequency / (len(he.nodes) - 1)
        return rating

    # Coarsening until we have <= K*N nodes
    while active_nodes > K * N:
        ratings = []
        for u in range(n):
            if coarse_map[u] != u:  # Already contracted
                continue
            for he in hg.getTouchingHyperedges(u):
                for v in he.nodes:
                    if u == v or coarse_map[v] != v:
                        continue
                    # Enforce weight constraint during coarsening
                    if coarse_weights[u] + coarse_weights[v] > N:
                        continue
                    rating = heavy_edge_rating(u, v)
                    ratings.append((rating, u, v))
        
        if not ratings:
            break
        
        ratings.sort(reverse=True, key=lambda x: x[0])
        contracted = set()
        new_contractions = []
        
        for _, u, v in ratings:
            if u in contracted or v in contracted:
                continue
            contracted.add(u)
            contracted.add(v)
            coarse_weights[u] += coarse_weights[v]  # Update weight
            coarse_map[v] = u
            active_nodes -= 1
            contraction_pairs.append((v, u))
            new_contractions.append((u, v))
            
            if active_nodes <= K * N:
                break

    # Build coarse hypergraph
    root_nodes = {}
    reverse_map = defaultdict(list)
    for i in range(n):
        root = coarse_map[i]
        while coarse_map[root] != root:
            root = coarse_map[root]
        coarse_map[i] = root
        reverse_map[root].append(i)
    
    coarse_nodes = list(reverse_map.keys())
    coarse_node_count = len(coarse_nodes)
    coarse_node_id = {node: i for i, node in enumerate(coarse_nodes)}
    coarse_hyperedges = []
    coarse_spike_freqs = []
    coarse_node_weights = [coarse_weights[node] for node in coarse_nodes]  # Weights for coarse nodes
    
    # Create hyperedges as flat tuples of integers
    for he in hg.hyperedges:
        coarse_source = coarse_map[he.source()]
        coarse_dests = set()
        for node in he.destinations():
            coarse_dest = coarse_map[node]
            if coarse_dest != coarse_source:  # Skip self-loops
                coarse_dests.add(coarse_dest)
        
        if not coarse_dests:
            continue
            
        edge_nodes = (coarse_node_id[coarse_source],) + tuple(coarse_node_id[d] for d in coarse_dests)
        coarse_hyperedges.append(edge_nodes)
        coarse_spike_freqs.append(he.spike_frequency)
    
    coarse_hypergraph = HyperGraph(coarse_node_count, coarse_hyperedges, coarse_spike_freqs)
    
    # Internal function for coarse partitioning with weight constraints
    def partition_coarse(hg: HyperGraph, node_weights: list[int], M: int, N: int, K: int) -> list[int]:
        n_coarse = hg.nodes
        part_coarse = [-1] * n_coarse
        part_count = 0
        part_total_weights = [0] * K  # Track total weight per partition
        part_nodes = [set() for _ in range(K)]
        
        # Track partition connectivity for hyperedges
        hyperedge_source_part = [-1] * len(hg.hyperedges)
        hyperedge_dest_part = [set() for _ in range(len(hg.hyperedges))]
        
        # Sort coarse nodes by degree (high to low)
        node_degrees = []
        for v in range(n_coarse):
            degree = len(hg.getTouchingHyperedges(v))
            node_degrees.append((degree, v))
        node_degrees.sort(reverse=True)
        
        for _, v in node_degrees:
            best_part = -1
            best_gain = -10**18
            weight_v = node_weights[v]
            
            # Try existing partitions and new partition
            for p in range(part_count + 1):
                if p < part_count and part_total_weights[p] + weight_v > N:
                    continue  # Skip if adding node would exceed weight limit
                
                gain = 0.0
                for he in hg.getTouchingHyperedges(v):
                    he_idx = hg.hyperedges.index(he)
                    if v == he.source():
                        # Source moving to new partition
                        if hyperedge_source_part[he_idx] != p:
                            # Connectivity changes
                            old_connectivity = len(hyperedge_dest_part[he_idx])
                            if hyperedge_source_part[he_idx] != -1:
                                old_connectivity += 1
                            
                            new_connectivity = len(hyperedge_dest_part[he_idx])
                            if p != -1:
                                new_connectivity += 1
                            
                            gain += (old_connectivity - new_connectivity) * he.spike_frequency
                    else:
                        # Destination moving to new partition
                        if p not in hyperedge_dest_part[he_idx]:
                            old_connectivity = len(hyperedge_dest_part[he_idx])
                            if hyperedge_source_part[he_idx] != -1:
                                old_connectivity += 1
                            
                            new_connectivity = old_connectivity + 1
                            if hyperedge_source_part[he_idx] == p:
                                new_connectivity -= 1
                            
                            gain += (old_connectivity - new_connectivity) * he.spike_frequency
                
                if gain > best_gain:
                    best_gain = gain
                    best_part = p
            
            # Create new partition if needed
            if best_part == -1:
                if part_count < K - 1:
                    best_part = part_count
                    part_count += 1
                else:
                    # Find first partition with capacity
                    for p in range(part_count):
                        if part_total_weights[p] + weight_v <= N:
                            best_part = p
                            break
            
            # Assign node to partition
            if best_part == -1:
                # Shouldn't happen, but fallback to first partition
                best_part = 0
                part_total_weights[0] += weight_v  # Might violate constraint, but no choice
            else:
                part_total_weights[best_part] += weight_v
                
            part_coarse[v] = best_part
            part_nodes[best_part].add(v)
            
            # Update hyperedge tracking
            for he in hg.getTouchingHyperedges(v):
                he_idx = hg.hyperedges.index(he)
                if v == he.source():
                    hyperedge_source_part[he_idx] = best_part
                else:
                    hyperedge_dest_part[he_idx].add(best_part)
        
        return part_coarse

    # Partition the coarse hypergraph with weight constraints
    coarse_part = partition_coarse(coarse_hypergraph, coarse_node_weights, M, N, K)
    
    # Map back to original nodes
    for coarse_idx, p in enumerate(coarse_part):
        original_root = coarse_nodes[coarse_idx]
        for node in reverse_map[original_root]:
            part[node] = p

    # Track partition weights for refinement
    partition_weights = [0] * K
    for i, p in enumerate(part):
        partition_weights[p] += 1  # Unit weights

    # Refinement during uncoarsening
    def refine_node(v: int):
        nonlocal partition_weights
        current_part = part[v]
        best_gain = 0
        best_part = current_part
        
        # Get adjacent partitions
        adjacent_parts = set()
        for he in hg.getTouchingHyperedges(v):
            for node in he.nodes:
                if part[node] != current_part:
                    adjacent_parts.add(part[node])
        
        # Calculate current connectivity contribution
        current_metric = 0
        for he in hg.getTouchingHyperedges(v):
            partitions = set()
            for node in he.nodes:
                partitions.add(part[node])
            lambda_e = len(partitions)
            current_metric += (lambda_e - 1) * he.spike_frequency
        
        # Evaluate moves to adjacent partitions
        for target_part in adjacent_parts:
            if target_part == current_part:
                continue
            
            # Skip if move would violate weight constraint
            if partition_weights[target_part] + 1 > N:
                continue
            
            # Simulate move
            new_metric = 0
            for he in hg.getTouchingHyperedges(v):
                partitions = set()
                for node in he.nodes:
                    if node == v:
                        partitions.add(target_part)
                    else:
                        partitions.add(part[node])
                lambda_e = len(partitions)
                new_metric += (lambda_e - 1) * he.spike_frequency
            
            gain = current_metric - new_metric
            if gain > best_gain:
                best_gain = gain
                best_part = target_part
        
        # Apply best move
        if best_part != current_part:
            # Update partition weights
            partition_weights[current_part] -= 1
            partition_weights[best_part] += 1
            part[v] = best_part

    # Refine around contracted nodes
    for v, _ in contraction_pairs:
        refine_node(v)
    
    return part

@core
def partitionGreedyMultilevelRefinedV3(hg: HyperGraph, N: int, M: int, K: int, seed : Optional[int] = None) -> list[int]:
    rng = random.Random(seed)

    #---------------
    # 1) COARSENING
    #---------------
    def coarsen(hg, max_coarse, M):
        parent = list(range(hg.nodes))
        groups = {i: [i] for i in range(hg.nodes)}

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        # build heavy-edge weights on touching hyperedges
        pair_w = Counter()
        for u in range(hg.nodes):
            hes = hg.getOutboundHyperedges(u)
            for e in hes:
                for a, b in combinations(e.destinations(), 2):
                    pair_w[tuple(sorted((a, b)))] += e.spike_frequency

        # group inbound-edge tracking
        group_in = {i: set(e for e in map(lambda e:e, hg.getInboundHyperedges(i))) for i in range(hg.nodes)}

        for (u, v), _ in pair_w.most_common():
            ru, rv = find(u), find(v)
            if ru == rv: continue
            if len(groups[ru]) + len(groups[rv]) > max_coarse: continue

            merged_nodes = set(groups[ru]) | set(groups[rv])
            # gather inbound hyperedges of merged set
            cand = set()
            for n in merged_nodes:
                cand |= {id(e) for e in hg.getInboundHyperedges(n)}
            in_eids = {eid for eid in cand if any(d in merged_nodes for d in hg.hyperedges[eid].destinations())}
            if len(in_eids) > M: 
                continue

            # merge
            parent[rv] = ru
            groups[ru] += groups[rv]
            del groups[rv]
            group_in[ru] = in_eids

            if len(groups) <= max_coarse:
                break

        # build coarse hypergraph
        coarse_groups = list(groups.values())
        node2coarse = [0]*hg.nodes
        for ci, grp in enumerate(coarse_groups):
            for u in grp: node2coarse[u] = ci

        seen = set()
        ch, cf = [], []
        for e in hg.hyperedges:
            src = node2coarse[e.source()]
            dsts = tuple(sorted({node2coarse[d] for d in e.destinations() if node2coarse[d]!=src}))
            if not dsts: continue
            key = (src,)+dsts
            if key in seen: continue
            seen.add(key)
            ch.append(key)
            cf.append(e.spike_frequency)

        return HyperGraph(len(coarse_groups), ch, cf), coarse_groups

    # pick coarsening level
    max_coarse = min(K * N // 2, hg.nodes // 10) or 1
    c_hg, c_groups = coarsen(hg, max_coarse, M)

    #---------------
    # 2) GREEDY INITIAL
    #---------------
    # node->touching edges
    touching = {u: hg.getTouchingHyperedges(u) for u in range(hg.nodes)}

    part_of = [-1]*hg.nodes
    parts = []
    in_bounds = []

    for u in range(hg.nodes):
        best_pid, best_cost = None, float('inf')
        rng.shuffle(parts)
        for pid, nodes in enumerate(parts):
            if len(nodes) >= N: continue

            # inbound edges if u added
            new_in = set(in_bounds[pid])
            for e in hg.getInboundHyperedges(u):
                if any(d in nodes for d in e.destinations()):
                    new_in.add(id(e))
            if len(new_in) > M: continue

            # cost delta
            delta = 0.0
            for e in touching[u]:
                before = {part_of[v] for v in e.nodes() if part_of[v]>=0}
                after = before | {pid}
                delta += e.spike_frequency * ((len(after)-1) - max(len(before)-1,0))
            if delta < best_cost:
                best_cost, best_pid = delta, pid

        if best_pid is None:
            if len(parts) >= K:
                raise RuntimeError("Too many partitions")
            best_pid = len(parts)
            parts.append(set())
            in_bounds.append(set())

        part_of[u] = best_pid
        parts[best_pid].add(u)
        # update inbound
        for e in hg.getInboundHyperedges(u):
            if any(d in parts[best_pid] for d in e.destinations()):
                in_bounds[best_pid].add(id(e))

    #---------------
    # 3) FM REFINEMENT
    #---------------
    # build node->touching edges ids
    eid_of_node = {u: [id(e) for e in touching[u]] for u in range(hg.nodes)}

    heap = []
    in_heap = defaultdict(set)
    for u in range(hg.nodes):
        cur = part_of[u]
        nbr_parts = {part_of[v] for e in touching[u] for v in e.nodes() if part_of[v]!=cur}
        for tgt in nbr_parts:
            gain = 0.0
            for eid in eid_of_node[u]:
                e = hg.hyperedges[eid]
                b = {part_of[v] for v in e.nodes()}
                a = (b - {cur}) | {tgt}
                gain += e.spike_frequency * ((len(b)-1)-(len(a)-1))
            if gain>0:
                heapq.heappush(heap,(-gain,u,tgt))
                in_heap[u].add(tgt)

    moved = set()
    while heap:
        neg,u,tgt = heapq.heappop(heap)
        if u in moved: continue
        cur = part_of[u]
        if cur==tgt or len(parts[tgt])+1 > N: continue

        # simulate inbound
        new_in = set(in_bounds[tgt])
        for e in hg.getInboundHyperedges(u):
            if any(d in parts[tgt] or d==u for d in e.destinations()):
                new_in.add(id(e))
        if len(new_in)>M: continue

        # commit
        parts[cur].remove(u); parts[tgt].add(u)
        part_of[u] = tgt; moved.add(u)

        # recompute inbound for both
        for pid in (cur, tgt):
            ib = set()
            for v in parts[pid]:
                for e in hg.getInboundHyperedges(v):
                    if any(d in parts[pid] for d in e.destinations()):
                        ib.add(id(e))
            in_bounds[pid] = ib

        # re‐enqueue neighbors
        for e in touching[u]:
            for v in e.nodes():
                if v==u: continue
                cur2 = part_of[v]
                nbr2 = {part_of[w] for e2 in touching[v] for w in e2.nodes() if part_of[w]!=cur2}
                for tgt2 in nbr2 - in_heap[v]:
                    gain=0.0
                    for e2 in touching[v]:
                        b={part_of[w] for w in e2.nodes()}
                        a=(b-{cur2})|{tgt2}
                        gain+=e2.spike_frequency*((len(b)-1)-(len(a)-1))
                    if gain>0:
                        heapq.heappush(heap,(-gain,v,tgt2))
                        in_heap[v].add(tgt2)

    return part_of

"""
Run the following map operation on the nodes of the hypergraph:
node -> inbound edges -> inbound edges sources set

Now we have a list of length containing sets. Each set contains integers. This routine fuses some of those sets to reduce their number, starting
from those that are the most similar, where similarity is defined by the number of elements in common between sets. The process has three constraints:
- Each final set can result from the union of at most N sets. In other words, you can't unite more than N original sets in the same final set.
- Each final set can't contain more than M elements.
- In the end you must have <= K sets. If this can't be satisfied, erroring out is acceptable.
This works to partition the hypergraph because the similarity above mirrors the number of hyperedges that would get combined (thus with less cuts)
when two nodes share the same partition.

Arguments:
- hg: hypergraph to partition
- M: max elements per final set
- N: max original sets per final set
- K: target number of clusters
- num_perm: number of MinHash permutations
- threshold: similarity fraction (0.0, 1.0) required for a merge

NOTE: currently the performed map considers only the first hyperedges hop between nodes, by recursively adding to the sets also the sources of edges
      inbound to the nodes already in the sets (where second, third, etc. order nodes added to the sets get a lower weight), one could get better
      similarity metric, leading to better results at the price of O((|nodes|*|hyperedges|)^x) complexity during the map with x = 2, 3, 4, ...
"""
@core
def partitionSetlistMiniHash(hg: HyperGraph, N: int, M: int, K: int, num_perm : int = 128, threshold : float = 0.0) -> list[int]:
    #inbound_edges_sources = map(lambda n : map(lambda he : he[0], hg.getInboundHyperedges(n)), range(hg.nodes))
    # 0) Build the inbound edges's sources sets list
    sets = [{he.source() for he in hg.getInboundHyperedges(n)} for n in range(hg.nodes)]

    # 1) Build an empty MinHashLSH index over clusters:
    lsh = MinHashLSH(threshold = threshold, num_perm = num_perm)

    clusters = {} # cid -> { 'union_set', 'count', 'minhash' }
    assignments = [] # will hold cid for each input set
    next_cid = 0

    for S in sets:
        # 2a) Create MinHash sketch of S
        mh = MinHash(num_perm = num_perm)
        for x in S:
            mh.update(str(x).encode('utf8'))

        # 2b) Get candidate cluster IDs whose sketch ≥ threshold
        cand_ids = lsh.query(mh)

        # 2c) Among candidates, pick the best mergeable cluster
        best_cid, best_jacc = None, 0.0
        for cid in cand_ids:
            cl = clusters[cid]
            if cl['count'] >= N:
                continue

            # Estimate Jaccard via sketch
            j = mh.jaccard(cl['minhash'])
            if j <= best_jacc:
                continue

            # Exact union‐size check via intersection count
            inter = len(cl['union_set'].intersection(S))
            if (len(cl['union_set']) + len(S) - inter) <= M:
                best_cid, best_jacc = cid, j

        # 2d) If no existing cluster fits, start a new one
        if best_cid is None:
            cid = next_cid
            next_cid += 1
            clusters[cid] = {'union_set': set(S), 'count': 1, 'minhash': mh.copy()}
            lsh.insert(cid, mh)
        else:
            # 2e) Merge into the chosen cluster
            cid = best_cid
            cl  = clusters[cid]
            cl['union_set'].update(S)
            cl['count'] += 1
            # update cluster’s MinHash: pointwise min of hashvalues
            for i in range(num_perm):
                cl['minhash'].hashvalues[i] = min(cl['minhash'].hashvalues[i], mh.hashvalues[i])
            # re‐index so its buckets reflect the updated sketch
            lsh.remove(cid)
            lsh.insert(cid, cl['minhash'])

        # record which cluster this set went into
        assignments.append(cid)

    # 3) Enforce the ≤ K clusters requirement
    unique_cids = sorted(clusters.keys())
    if len(unique_cids) > K:
        raise Exception(f"Partitioning could only form {len(unique_cids)} > {K} clusters under the provided N and M constraints.")

    # 4) Remap arbitrary cid values into 0..C–1
    cid_map = {old: new for new, old in enumerate(unique_cids)}
    labels  = [cid_map[c] for c in assignments]
    return labels

"""
Same as 'partitionSetlistMiniHash', but each inbound hyperedge is weighted by its spike frequency.
As a result, sets became dictionaries, and we use the weighted Jaccard distance for similarity.

Extra: the merging threshold progressively decreases from 'threshold_initial' to 'threshold_final'
       following the law $thr(i) = thr_{init} - (thr_{final} - thr_{init}) \cdot \frac{i}{|nodes| - i}$
       to anneal the result.
"""
@core
def partitionSetlistMiniHashWeighted(hg: HyperGraph, N: int, M: int, K: int, num_perm : int = 128, threshold_initial : float = 0.8, threshold_final : float = 0.1) -> list[int]:
    #inbound_edges_sources = map(lambda n : map(lambda he : he[0], hg.getInboundHyperedges(n)), range(hg.nodes))
    # 0) Build the inbound edges's sources sets list
    sets = [{he.source() : he.spike_frequency for he in hg.getInboundHyperedges(n)} for n in range(hg.nodes)]
    
    # Convert to list so we know total length for decay schedule
    if not sets:
        return []

    # 1) Determine universe dimension D
    #    We assume element IDs across all sets range [0, max_elem).
    max_elem = max(e for wdict in sets for e in wdict)
    D = max_elem + 1

    # 2) Build the weighted MinHash generator
    wmg = WeightedMinHashGenerator(D, sample_size = num_perm)
    # 3) Create an LSH index with the smallest threshold
    lsh = MinHashLSH(threshold = threshold_final, num_perm = num_perm)

    clusters = {} # cid -> {'union_keys', 'count', 'minhash'}
    assignments = [] # will store the cluster ID for each input set
    next_cid = 0
    total = len(sets)

    for i, wdict in enumerate(sets):
        # 4) Compute the current, linearly decayed threshold
        frac = i / (total - 1)
        thr  = threshold_initial - (threshold_initial - threshold_final) * frac

        # 5) Sketch this weighted set
        #    Keys of wdict must be ints < D
        mh = wmg.minhash(wdict)

        # 6) Query LSH for any clusters with est. J ≥ threshold_final
        cand_ids = lsh.query(mh)

        # 7) Find the best mergeable cluster (est. J ≥ thr, caps OK)
        best_cid, best_j = None, thr
        for cid in cand_ids:
            cl = clusters[cid]
            if cl['count'] >= N:
                continue

            j = mh.jaccard(cl['minhash'])
            if j < best_j:
                continue

            # Exact check of distinct-element cap M
            inter = len(cl['union_keys'].intersection(wdict))
            if (len(cl['union_keys']) + len(wdict) - inter) <= M:
                best_cid, best_j = cid, j

        # 8) Merge or create new cluster
        if best_cid is None:
            cid = next_cid
            next_cid += 1
            clusters[cid] = {'union_keys': set(wdict), 'count': 1, 'minhash': mh}
            lsh.insert(cid, mh)
        else:
            cid = best_cid
            cl  = clusters[cid]
            cl['union_keys'].update(wdict)
            cl['count'] += 1
            # update the weighted sketch in place
            for p in range(num_perm):
                cl['minhash'].hashvalues[p] = min(cl['minhash'].hashvalues[p], mh.hashvalues[p])
            # refresh LSH buckets so future queries see updated sketch
            lsh.remove(cid)
            lsh.insert(cid, cl['minhash'])

        assignments.append(cid)

    # 9) Enforce ≤ K clusters
    unique_cids = sorted(clusters.keys())
    if len(unique_cids) > K:
        raise Exception(f"Partitioning could only form {len(unique_cids)} > {K} clusters under the provided N and M constraints.")

    # 10) Remap to contiguous labels 0…C−1
    cid_map = {old: new for new, old in enumerate(unique_cids)}
    return [cid_map[c] for c in assignments]

"""
Simple sequential partitioning algorithm, assigns nodes to the same partition until a constraint
would be violated, then creates and starts filling the next partition.

Used by the Ouwen Jin paper.
"""
@core
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
@core
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
        
        cnt = 0
        delta = math.inf
        current_cost = cost(hg, partitioning)
        while delta > min_delta:
            print("Swap iteration:", cnt, "delta:", delta)
            cnt += 1
            delta = 0
            for n in range(hg.nodes):
                for m in range(n): # here goes the quadratic complexity...
                    part_n, part_m = partitioning[n], partitioning[m]
                    edges_n, edges_m = len(hg.getInboundHyperedges(n)), len(hg.getInboundHyperedges(m))
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