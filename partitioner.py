from typing import Optional

from collections import defaultdict, Counter
from datasketch import MinHash, MinHashLSH
from itertools import combinations
import numpy as np
import random
import heapq
import math

from datastructures import *
from settings import *
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
#
# Variables for complexity:
# - n : # nodes
# - e : # hyperedges
# - d : # connections per hyperedge
# - h : # connections per node

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
@core
def partitionGreedy(hg: HyperGraph, N: int, M: int, K: int) -> list[int]:
    partitions = []  # each partition is a dict: {'nodes': set, 'in_edges': set}
    node_to_partition = [-1] * hg.nodes  # partitions assignment

    for node in range(hg.nodes):
        best_partition = -1
        best_score = math.inf
        candidate_edges = hg.getTouchingHyperedges(node)

        for pid, p in enumerate(partitions):
            if len(p['nodes']) >= N:
                continue

            if len(p['in_edges']) + sum(1 for he in hg.getInboundHyperedges(node) if he not in p['in_edges']) > M:
                continue

            cut_cost = 0
            for he in candidate_edges:
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
        for he in candidate_edges:
            if node in he.destinations():
                p['in_edges'].add(he)
        node_to_partition[node] = best_partition

    return node_to_partition

"""
Merges original nodes by heavy-edge matching until coarse nodes count <= target_coarse_nodes or no valid merges remain.
Returns (coarse_graph, coarse_groups), where 'coarse_groups[i]' lists original nodes merged into coarse node 'i'.
Constraints are given by 'max_nodes' and 'max_inbound_edges'.
"""
def coarsen_hypergraph(hg: HyperGraph, target_coarse_nodes: int, max_nodes : int, max_inbound_edges: int) -> tuple[HyperGraph, list[list[int]]]:
    parent = list(range(hg.nodes))
    groups = {i: [i] for i in range(hg.nodes)}

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    # compute heavy‐edge weights between destination pairs
    pair_weights = Counter()
    for he in hg.hyperedges:
        for u, v in combinations(he.nodes[1:], 2):
            a, b = (u, v) if u < v else (v, u)
            pair_weights[(a, b)] += he.spike_frequency

    # merge strongest pairs
    for (u, v), _ in sorted(pair_weights.items(), key = lambda x: -x[1]):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue

        merged = groups[ru] + groups[rv]
        if len(merged) > max_nodes:
            continue

        # constrain the number of hyperedges that have any destination in merged_set
        inbound = set()
        for n in merged:
            inbound.update(hg.getInboundHyperedges(n))
        if len(inbound) > max_inbound_edges:
            continue

        # commit merge
        parent[rv] = ru
        groups[ru].extend(groups[rv])
        del groups[rv]

        if len(groups) <= target_coarse_nodes:
            break

    # build coarse groups and edges
    coarse_groups = list(groups.values())
    node_to_coarse = [0] * hg.nodes
    for ci, grp in enumerate(coarse_groups):
        for n in grp:
            node_to_coarse[n] = ci

    coarse_hes = defaultdict(float) # he-tuple -> spike frequency
    for he in hg.hyperedges:
        src = node_to_coarse[he.source()]
        dsts = tuple({node_to_coarse[d] for d in he.destinations() if node_to_coarse[d] != src})
        if not dsts:
            continue
        key = (src,) + dsts
        coarse_hes[key] += he.spike_frequency

    return HyperGraph(len(coarse_groups), list(coarse_hes.keys()), list(coarse_hes.values())), coarse_groups

"""
FM-style (Fiduccia-Mattheyses) refinement on the full hypergraph. Let 'hg' be the uncoarsened hypergraph.
Starts from 'coarse_assignment' (partition of the coarsened hypergraph) and 'coarse_groups' (uncoarsened
hypergraph nodes per-coarsened group), returns a refined list of length 'hypergraph.nodes'.
"""
def refine_partition_FM(hg: HyperGraph, coarse_assignment: list[int], coarse_groups: list[list[int]], max_nodes: int, max_inbound_edges: int) -> list[int]:
    node_to_part = [-1] * hg.nodes
    parts = defaultdict(set)
    in_edges = defaultdict(set)

    # initialize from coarse assignment
    for ci, grp in enumerate(coarse_groups):
        pid = coarse_assignment[ci]
        for u in grp:
            node_to_part[u] = pid
            parts[pid].add(u)

    # initial inbound-edge sets: any hyperedge with at least one destination in the partition
    for he in hg:
        for d in he.destinations():
            pid = node_to_part[d]
            if pid >= 0:
                in_edges[pid].add(he)

    # build FM gain heap
    heap, moved = [], set()
    for u in range(hg.nodes):
        cur = node_to_part[u]
        neighbor_parts = [node_to_part[v] for he in hg.getTouchingHyperedges(u) for v in he if node_to_part[v] != cur]
        for tgt in neighbor_parts:
            gain = 0.0
            for he in hg.getTouchingHyperedges(u):
                before = {node_to_part[v] for v in he}
                after = (before - {cur}) | {tgt}
                gain += he.spike_frequency * ((len(before) > 1) - (len(after) > 1))
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

        # simulate new inbound edges for target
        new_in = set(in_edges[tgt])
        for he in hg.getTouchingHyperedges(u):
            if any(d in parts[tgt] or d == u for d in he.destinations()):
                new_in.add(he)
        if len(new_in) > max_inbound_edges:
            continue

        # commit move
        parts[cur].remove(u)
        parts[tgt].add(u)
        node_to_part[u] = tgt
        moved.add(u)

        # recompute inbound edges for both affected partitions
        for pid in (cur, tgt):
            updated = set()
            for n in parts[pid]:
                for he in hg.getTouchingHyperedges(n):
                    if any(d in parts[pid] for d in he.destinations()):
                        updated.add(he)
            in_edges[pid] = updated

    return node_to_part

"""
Ensures that the final partition respects constraints, moving nodes if needed.
"""
def enforce_partition_constraints(hg: HyperGraph, partition: list[int], max_nodes: int, max_inbound_edges: int, max_partitions: int) -> list[int]:
    from collections import defaultdict

    node_to_edges = defaultdict(list)
    for eid, he in enumerate(hg.hyperedges):
        for node in he:
            node_to_edges[node].append(eid)

    part_to_nodes = defaultdict(set)
    part_to_edges = defaultdict(set)

    for node, pid in enumerate(partition):
        part_to_nodes[pid].add(node)
        for eid in node_to_edges[node]:
            if node in hg.hyperedges[eid].destinations():
                part_to_edges[pid].add(eid)

    for pid in list(part_to_nodes):
        while len(part_to_nodes[pid]) > max_nodes or len(part_to_edges[pid]) > max_inbound_edges:
            # Identify node whose removal reduces inbound edges the most
            best_node, best_delta = None, -1
            for node in part_to_nodes[pid]:
                delta = 0
                for eid in node_to_edges[node]:
                    if eid not in part_to_edges[pid]:
                        continue
                    he = hg.hyperedges[eid]
                    if all(n == node or partition[n] != pid for n in he.destinations()):
                        delta += 1
                if delta > best_delta:
                    best_node, best_delta = node, delta

            if best_node is None:
                break  # Nothing to remove safely

            # Try to move to an existing or new valid partition
            moved = False
            for cand in range(len(part_to_nodes) + 1):
                if cand == pid:
                    continue
                if cand in part_to_nodes and len(part_to_nodes[cand]) >= max_nodes:
                    continue

                simulated_edges = set(part_to_edges[cand]) if cand in part_to_edges else set()
                for eid in node_to_edges[best_node]:
                    he = hg.hyperedges[eid]
                    if best_node in he.destinations():
                        simulated_edges.add(eid)

                if len(simulated_edges) > max_inbound_edges:
                    continue

                # Create new partition if needed
                if cand == len(part_to_nodes):
                    if cand >= max_partitions:
                        raise Exception("Exceeded maximum number of partitions during constraint enforcement.")
                    part_to_nodes[cand] = set()
                    part_to_edges[cand] = set()

                # Move node
                part_to_nodes[pid].remove(best_node)
                part_to_nodes[cand].add(best_node)
                partition[best_node] = cand

                # Recompute inbound edges only for pid and cand
                for tgt in (pid, cand):
                    edges = set()
                    for n in part_to_nodes[tgt]:
                        for eid in node_to_edges[n]:
                            if n in hg.hyperedges[eid].destinations():
                                edges.add(eid)
                    part_to_edges[tgt] = edges

                moved = True
                break

            if not moved:
                raise Exception("Failed to move node to satisfy constraints. Consider increasing max_partitions.")

    return partition


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
    coarse_hg, groupings = coarsen_hypergraph(hg, min(max_partitions * max_nodes // 2, hg.nodes // 10), max_nodes, max_inbound_edges)
    # Step 2: Initial Partitioning
    coarse_partition = partitionGreedy(coarse_hg, max_nodes, max_inbound_edges, max_partitions)
    # Step 3: Refinement
    refined_partition  = refine_partition_FM(hg, coarse_partition, groupings, max_nodes, max_inbound_edges)
     # Step 4: Post-processing to strictly enforce constraints
    final_partition = enforce_partition_constraints(hg, refined_partition, max_nodes, max_inbound_edges, max_partitions)

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
        coarse_hg, coarse_groups = coarsen_hypergraph(hg, min(max_partitions, hg.nodes // max_nodes), max_nodes, max_inbound_edges)

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
                in_u = {eid for eid in cnode_to_edges[u] if u in coarse_hg.hyperedges[eid].destinations()}
                if len(part_inbound[pid] | in_u) > max_inbound_edges:
                    continue

                cut = sum(coarse_hg.hyperedges[eid].spike_frequency for eid in cnode_to_edges[u] if pid not in {coarse_assignment[v] for v in coarse_hg.hyperedges[eid] if coarse_assignment[v] != -1})
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
        refined_assign = refine_partition_FM(hg, coarse_assignment, coarse_groups, max_nodes, max_inbound_edges)
        # 4) Post-processing to strictly enforce constraints
        final_assign = enforce_partition_constraints(hg, refined_assign, max_nodes, max_inbound_edges, max_partitions)

        # 4) Evaluate cut
        cut_value = sum(e.spike_frequency for e in hg.hyperedges if len({final_assign[v] for v in e}) > 1)

        if cut_value < best_cut:
            best_cut, best_assign = cut_value, final_assign

    return best_assign

"""
True hierarchical partitioning as in hMETIS.
"""
@core
def partitionHMETIS(hg: HyperGraph, max_nodes: int, max_inbound_edges: int, max_partitions: int, multistarts: int = 3, seed : Optional[int] = None) -> list[int]:
    timerPrint = getTimerPrinter(Settings.PRINT_INTERVAL)

    """
    Source: "Multilevel Hypergraph Partitioning: Applications in VLSI Domain" by George Karypis
    => "Edge Coarsening (EC)" technique!
    """
    def coarsen_hypergraph(hg: HyperGraph, target_coarse_nodes: int, max_nodes : int, max_inbound_edges: int, seed : Optional[int] = None) -> tuple[list[tuple[HyperGraph, list[list[int]]]], list[list[int]], list[Counter[int]]]:
        current_hg = hg
        partition_sizes = [1 for _ in range(hg.nodes)] # size of each partition (node) in current_hg
        inbound_he_ids = [Counter(hash(he) for he in hg.getInboundHyperedges(n)) for n in range(hg.nodes)] # inbound hyperedges IDs for each partition (node) in current_hg

        nodes_in_node : list[list[int]]  = [] # tracks at each level of coarsening, how many original nodes ended up inside each present node, in other words, tracks 'partition_sizes'
        result : list[tuple[HyperGraph, list[list[int]]]] = [] # tracks successively coarser hypergraphs and the pair of nodes that were coarsened together

        coarsened = True
        while current_hg.nodes > target_coarse_nodes and coarsened:
            coarsened = False
            coarsenings = []
            # TODO: remove those and update the originals in place by making them dictionaries.
            next_partition_sizes = []
            next_inbound_he_ids = []
            
            # visit verticies in random order, for each iterate all yet unmatched connected nodes
            # => merge the node with the other note it is connected with under the strongest weight
            rng = np.random.default_rng(seed)
            nodes = np.arange(current_hg.nodes)
            rng.shuffle(nodes)
            unused = set(range(current_hg.nodes))
            for n in nodes:
                if n in unused:
                    candidates = defaultdict(lambda : 0)
                    for he in current_hg.getTouchingHyperedges(n):
                        for m in he:
                            if m in unused and m != n:
                                candidates[m] += he.spike_frequency
                    # TODO: inefficient max retrieval -> use a heap? Apparently it got worse, so no.
                    while candidates:
                        best = max(candidates, key = candidates.get)
                        if (new_size := partition_sizes[n] + partition_sizes[best]) < max_nodes and len(new_ids_set := inbound_he_ids[n] + inbound_he_ids[best]) < max_inbound_edges:
                            coarsenings.append((n, best))
                            next_partition_sizes.append(new_size)
                            next_inbound_he_ids.append(new_ids_set)
                            coarsened = True
                            unused.remove(n)
                            unused.remove(best)
                            break
                        candidates.pop(best)
            next_part_idx = 0
            partitions = [-1 for _ in range(current_hg.nodes)]
            for n, m in coarsenings:
                partitions[n] = next_part_idx
                partitions[m] = next_part_idx
                next_part_idx += 1
            for u in unused:
                partitions[u] = next_part_idx
                coarsenings.append((u,))
                next_partition_sizes.append(partition_sizes[u])
                next_inbound_he_ids.append(inbound_he_ids[u])
                next_part_idx += 1
            
            current_hg = current_hg.getPartitionsHypergraph(partitions, keep_self_cycles = True, squish_hyperedges = True)
            partition_sizes = next_partition_sizes
            inbound_he_ids = next_inbound_he_ids

            nodes_in_node.append(partition_sizes)
            result.append((current_hg, coarsenings))
            timerPrint(f"Coarsening: {current_hg.nodes} > {target_coarse_nodes} remaining nodes...")
        
        return result, nodes_in_node, inbound_he_ids

    """
    Source: "Multilevel k-way Hypergraph Partitioning" by George Karypis
    Updates the candidate 'partitioning' in place!
    """
    def greedy_FM_refinement(hg: HyperGraph, partitioning: list[int], partition_sizes : list[int], nodes_in_node : list[int], inbound_he_ids : list[Counter[int]], max_nodes: int, max_inbound_edges: int, seed : Optional[int] = None) -> None:
        rng = np.random.default_rng(seed)
        nodes = np.arange(hg.nodes)
        rng.shuffle(nodes)
        # For each hedge, track how many nodes it connects to per-partition (cost to build: e * d)
        connections_counts = {he : Counter(partitioning[n] for n in he.destinations()) for he in hg}
        for n in nodes:
            my_partition = partitioning[n]
            connectivity_w_partitions = defaultdict(float) # partition -> sum of spike frequency of connections
            for he in hg.getTouchingHyperedges(n):
                for p, c in connections_counts[he].items():
                    if p == my_partition:
                        c -= 1
                    connectivity_w_partitions[p] += c*he.spike_frequency
                # NOTE: the hgraph can have self-cycles, the same node can occur both as source and destination => handle source separately
                if he.source() != n:
                    connectivity_w_partitions[partitioning[he.source()]] += he.spike_frequency
            loss = connectivity_w_partitions.pop(my_partition) if my_partition in connectivity_w_partitions else 0
            while connectivity_w_partitions:
                best_partition = max(connectivity_w_partitions, key = connectivity_w_partitions.get)
                if connectivity_w_partitions[best_partition] - loss < 0:
                    break
                elif partition_sizes[best_partition] + nodes_in_node[n] <= max_nodes and len(best_inbound := inbound_he_ids[best_partition] + (my_inbound := Counter(map(hash, hg.getInboundHyperedges(n))))) <= max_inbound_edges:
                    partitioning[n] = best_partition
                    partition_sizes[best_partition] += nodes_in_node[n]
                    partition_sizes[my_partition] -= nodes_in_node[n]
                    inbound_he_ids[best_partition] = best_inbound
                    inbound_he_ids[my_partition] -= my_inbound
                    for he in hg.getTouchingHyperedges(n):
                        connections_counts[he][my_partition] -= 1
                        connections_counts[he][best_partition] += 1
                    break
                connectivity_w_partitions.pop(best_partition)

    # TODO: multistart!!!!!!!
    # =>=> Add a cost estimation for partitions feature in the model!
    # =>=> Extract a random seed for each start by using the initial seed!
    
    # hierarchically coarsened hypergraphs, from less to most coarsened
    coarsening_levels, nodes_in_node, inbound_he_ids = coarsen_hypergraph(hg, min(max_partitions, hg.nodes // max_nodes), max_nodes, max_inbound_edges, seed)
    if len(coarsening_levels) == 0:
        if hg.nodes > max_partitions:
            raise Exception("Cannot coarsen the hypergraph.")
        else:
            # TODO: do one round of refinement here still
            return [i for i in range(hg.nodes)]
    # NOTE: partitioning = initialPartitionig( ... )
    # => no need since we coarsen up to the point of having the right number of partitions
    partition_sizes = nodes_in_node[-1]
    partitions_count = coarsening_levels[-1][0].nodes
    if partitions_count > max_partitions:
        raise Exception("Cannot coarsen up until reaching a sufficiently low number of partitions.")
    partitioning = [i for i in range(coarsening_levels[-1][0].nodes)]
    for i in range(len(coarsening_levels) - 1, -1, -1):
        cl_hg, cl_coarsenings = coarsening_levels[i]
        greedy_FM_refinement(cl_hg, partitioning, partition_sizes, nodes_in_node[i], inbound_he_ids, max_nodes, max_inbound_edges, seed)
        # undo the coarsening
        new_partitioning = [-1 for _ in range(coarsening_levels[i - 1][0].nodes if i > 0 else hg.nodes)]
        for p, c in zip(partitioning, cl_coarsenings):
            for n in c:
                new_partitioning[n] = p
        partitioning = new_partitioning
        timerPrint(f"Refining: {i + 1}/{len(coarsening_levels)} levels left...")
    greedy_FM_refinement(hg, partitioning, partition_sizes, [1 for _ in range(hg.nodes)], inbound_he_ids, max_nodes, max_inbound_edges, seed)
    force_array_of_contigous_integers(partitioning)
    return partitioning

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
- K: maximum number of final clusters
- num_perm: number of MinHash permutations (128 is fast, 256 is good)
- threshold: similarity fraction (0.0, 1.0) required for a merge

NOTE: currently the performed map considers only the first hyperedges hop between nodes, by recursively adding to the sets also the sources of edges
      inbound to the nodes already in the sets (where second, third, etc. order nodes added to the sets get a lower weight), one could get better
      similarity metric, leading to better results at the price of O((|nodes|*|hyperedges|)^x) complexity during the map with x = 2, 3, 4, ...
"""
@core
def partitionSetlistMiniHash(hg: HyperGraph, N: int, M: int, K: int, num_perm : int = 256, threshold : float = 0.0) -> list[int]:
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
Experimental version of 'partitionSetlistMiniHashTEMP' with weights.
The implementation is extremely inefficient, but its purpose is to show if using weights can improve the result.

Variant: here we don't start with zero clusters, but with each node initially being its own cluster.
"""
@core
def partitionSetlistMiniHashWeights(hg: HyperGraph, N: int, M: int, K: int, threshold : float = 0.0) -> list[int]:
    # TODO: tune my arguments!
    # NOTE: for now (1k nodes), unless num_perm == num_bands it is too unlikely to get a collision...
    #       => these arguments shall dynamically adapt w.r.t. the 'hg' size...
    lhs : WeightedMinHashLSH[int] = WeightedMinHashLSH(num_perm = 8, num_bands = 8)
    
    for n in range(hg.nodes):
        d = dict()
        inbound = hg.getInboundHyperedges(n)
        #average_sf = 0
        for he in inbound:
            src = he.source()
            #average_sf += he.spike_frequency
            if src not in d:
                d[he.source()] = he.spike_frequency
            else:
                d[he.source()] += he.spike_frequency
        # NOTE: having oneself in the sources should push towards two nodes connected by an edge being together,
        #       but this worsens performance since it consumes an inbound edge slot for a weakly shared hyperedge!
        #if inbound:
        #    d[n] = average_sf / len(inbound)
        #else:
        #    d[n] = 0.0
        lhs.insert(d, set_id = n)

    queue = list(lhs.ids())
    assignments = DisjointSet(i for i in range(hg.nodes))
    merged = True

    while merged:
        merged = False
        while queue:
            i = queue.pop(0)
            cluster = lhs.get(i)

            cand_ids, cand_dists = lhs.query_by_id(i)
            
            # pick the best mergeable cluster
            best_cid, best_jacc = None, 0.0
            for cid, dist in zip(cand_ids, cand_dists):
                cl = lhs.get(cid)
                if cl.merge_count + cluster.merge_count > N:
                    continue

                # IDEA: to avoid putting together only the best nodes, punish merges between already large clusters!
                d = dist / max(cluster.merge_count, cl.merge_count)
                if d <= best_jacc:
                    continue

                # exact union‐size check via intersection count
                if len(cl.weighted_set.keys() | cluster.weighted_set.keys()) <= M:
                    best_cid, best_jacc = cid, d

            # merge into the chosen cluster
            if best_cid is not None and best_jacc >= threshold * (len(lhs) / hg.nodes)**2:
                assignments.union(i, best_cid)
                lhs.merge([i, best_cid], merged_set_id = best_cid)
                try:
                    queue.remove(best_cid)
                except:
                    pass
                merged = True
        queue = list(lhs.ids())

    result = [-1 for _ in range(hg.nodes)]
    for i, part in enumerate(assignments):
        for node in part:
            result[node] = i

        # enforce the <= K clusters requirement
        if i >= K:
            raise Exception(f"Partitioning could only form {len(assignments)} > {K} clusters under the provided N and M constraints.")

    return result

"""
Experimental version of 'partitionSetlistMiniHash' with weights.
The LSH implementation is still flawed, but its purpose is to show if using weights can improve the result.

Variant: here we don't start with zero clusters, but with each node initially being its own cluster.
"""
@core
def partitionSetlistMiniHashWeightsForest(hg: HyperGraph, N: int, M: int, K: int, threshold : float = 0.0, top_k : int = 16) -> list[int]:
    # TODO: tune my arguments!
    # 16, 4 is fast and works decently
    # 32, 8 takes double the time, but is akin to a round of FM
    # 64, 16 is slow but beats one round of FM
    # 256, 128 is best for the 8k model
    # higher 'top_k' costs slightly more time for slightly better results (e.g. 2% on both when doubled)
    #lhs : WeightedMinHashLSHForest[int] = WeightedMinHashLSHForest(num_perm = 32, tree_count = 16)
    # IDEA for 'size_multiplier': after 1024*32, increase by 1 every time you multiply by 16 the nodes in the hypergraph
    # TODO: fine tune 'count_invalid'!
    size_multiplier = math.ceil(math.log(max((hg.nodes + 1) / (1024*32), 1), 16)) + 1
    count_invalid = 2
    normalized_weights_range = (1, 16) # (1, 10) is quite good
    lhs : WeightedMinHashLSHSortedForest[int] = WeightedMinHashLSHSortedForest(num_perm = 128*size_multiplier, tree_count = 64*size_multiplier, hash_bytes = 4, normalized_weights_range = normalized_weights_range)
    print(f"Creating LSH forest with: {lhs.num_perm} perms, {lhs.tree_count} trees, {lhs.hash_bytes} hash bytes, {normalized_weights_range} normalized weights range, {top_k} top-k and {count_invalid} count invalid queries.")
    
    timerPrint = getTimerPrinter(Settings.PRINT_INTERVAL)
    
    # fill up LSH
    for n in range(hg.nodes):
        d = dict()
        inbound = hg.getInboundHyperedges(n)
        average_sf = 0
        for he in inbound:
            src = he.source()
            average_sf += he.spike_frequency
            if src not in d:
                d[he.source()] = he.spike_frequency
            else:
                d[he.source()] += he.spike_frequency
        if inbound:
            d[n] = average_sf / len(inbound)
        else:
            #d[n] = 0.001
            # default hyperedge for who has no inbound connections -> helps merge nodes that receive outside input
            # NOTE: node ids must be positive...
            d[-1 & 0xFFFFFFFF] = 1
        lhs.insert(d, set_id = n)
        timerPrint(f"Building LSH forest: {n}/{hg.nodes}...")

    queue = list(lhs.ids())
    assignments = DisjointSet(i for i in range(hg.nodes))
    merged = True

    cluster : LSHEntry = None
    def valid(other_cluster : LSHEntry):
        # Checks:
        # 1) Total count of merged original nodes
        # 2) Exact inbound hyperedges union‐size check via intersection count
        return other_cluster.merge_count + cluster.merge_count <= N and len(other_cluster.weighted_set.keys() | cluster.weighted_set.keys()) <= M

    while merged:
        skip = set()
        merged = False
        unmerged = []
        while queue:
            i = queue.pop(0)
            if i in skip:
                continue
            cluster = lhs.get(i)

            # NOTE: query by entry does not guarantee that the entry itself will not be in the output, filter it later...
            cand_ids, cand_simils = lhs.query(cluster, valid, top_k, count_invalid, True)
            
            # pick the best mergeable cluster
            best_cid, best_jacc = None, 0.0
            for cid, sim in zip(cand_ids, cand_simils):
                #cl = lhs.get(cid)
                # IDEA: to avoid putting together only the best nodes, punish merges between already large clusters!
                #sim = sim / max(cluster.merge_count, cl.merge_count)
                #cost = ((cluster.merge_count + cl.merge_count) / N + (len(cluster.weighted_set.keys() | cl.weighted_set.keys())) / M)
                #sim = sim / cost
                if cid != i and sim > best_jacc:
                    best_cid, best_jacc = cid, sim

            # merge into the chosen cluster
            if best_cid is not None and best_jacc >= threshold * (len(lhs) / hg.nodes)**2:
                assignments.union(i, best_cid)
                lhs.merge([i, best_cid], merged_set_id = best_cid)
                skip.add(best_cid)
                merged = True
            else:
                unmerged.append(i)
        # remove cluster with no valid merge candidates
        # NOTE: no need remove from 'unmerged' clusters that got merged into others,
        #       since their id became that of the other cluster, so they will not be deleted!
        for full_id in unmerged:
            lhs.delete(full_id)
        
        queue = list(lhs.ids())
        timerPrint(f"Merging nodes: {len(assignments)} left...")

    # enforce the <= K clusters requirement
    if len(assignments) > K:
        raise Exception(f"Partitioning could only form {len(assignments)} > {K} clusters under the provided N and M constraints.")
    
    result = [-1 for _ in range(hg.nodes)]
    for i, part in enumerate(assignments):
        for node in part:
            result[node] = i
    return result

"""
Simple sequential partitioning algorithm, assigns nodes to the same partition until a constraint
would be violated, then creates and starts filling the next partition.

Used by the Ouwen Jin paper.
"""
@core
def partitionSequential(hg: HyperGraph, N: int, M: int, K: int) -> list[int]:
    partitioning = []
    inbound_edges = set()
    assigned_nodes = 0
    current_partition = 0
    for node in range(hg.nodes):
        assigned_nodes += 1
        current_inbound = hg.getInboundHyperedges(node)
        inbound_edges.update(current_inbound)
        if assigned_nodes > N or len(inbound_edges) > M:
            assigned_nodes = 1
            inbound_edges.clear()
            inbound_edges.update(current_inbound)
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
        inbound_edges = [Counter()]
        assigned_nodes = [0]
        current_partition = 0
        nodes = list(range(hg.nodes))
        random.shuffle(nodes)
        # initial partitioning: same logic as 'partitionSequential', but the order of nodes is randomized
        for node in nodes:
            current_inbound = hg.getInboundHyperedges(node)
            inbound_edges[-1].update(current_inbound)
            if assigned_nodes[-1] + 1 > N or len(inbound_edges[-1]) > M:
                assigned_nodes.append(0)
                inbound_edges[-1].subtract(current_inbound)
                inbound_edges.append(Counter(current_inbound))
                current_partition += 1
                if current_partition >= K:
                    raise Exception("Exceeded maximum number of partitions K.")
            assigned_nodes[-1] += 1
            partitioning[node] = current_partition
        
        cnt = 0
        delta = math.inf
        current_cost = 0
        # initial cost
        for he in hg:
            connections = set()
            for n in he.destinations():
                connections.add(partitioning[n])
            current_cost += he.spike_frequency*len(connections)
        # improve cost by swapping nodes
        while delta > min_delta:
            print("Swap iteration:", cnt, "delta:", delta)
            cnt += 1
            delta = 0
            for n in range(hg.nodes):
                edges_n = Counter(hg.getInboundHyperedges(n))
                for m in range(n): # here goes the quadratic complexity...
                    part_n, part_m = partitioning[n], partitioning[m]
                    edges_m = Counter(hg.getInboundHyperedges(m))
                    if part_n != part_m:
                        # it's a swap, assigned nodes will always be fine
                        if len(in_n := (inbound_edges[part_n] - edges_n + edges_m)) <= M and len(in_m := (inbound_edges[part_m] - edges_m + edges_n)) <= M:
                            partitioning[n], partitioning[m] = partitioning[m], partitioning[n]
                            new_cost = current_cost
                            # differential cost update
                            for he in hg.getInboundHyperedges(n):
                                src_part = partitioning[he.source()]
                                if src_part != partitioning[n] and src_part == partitioning[m]:
                                    new_cost += he.spike_frequency
                                elif src_part == partitioning[n] and src_part != partitioning[m]:
                                    new_cost -= he.spike_frequency
                            for he in hg.getInboundHyperedges(m):
                                src_part = partitioning[he.source()]
                                if src_part != partitioning[m] and src_part == partitioning[n]:
                                    new_cost += he.spike_frequency
                                elif src_part == partitioning[m] and src_part != partitioning[n]:
                                    new_cost -= he.spike_frequency
                            for he in hg.getOutboundHyperedges(n):
                                for dst in he.destinations():
                                    dst_part = partitioning[dst]
                                    if dst_part != partitioning[n] and dst_part == partitioning[m]:
                                        new_cost += he.spike_frequency
                                    elif dst_part == partitioning[n] and dst_part != partitioning[m]:
                                        new_cost -= he.spike_frequency
                            for he in hg.getOutboundHyperedges(m):
                                for dst in he.destinations():
                                    dst_part = partitioning[dst]
                                    if dst_part != partitioning[m] and dst_part == partitioning[n]:
                                        new_cost += he.spike_frequency
                                    elif dst_part == partitioning[m] and dst_part != partitioning[n]:
                                        new_cost -= he.spike_frequency
                            
                            if new_cost >= current_cost:
                                partitioning[n], partitioning[m] = partitioning[m], partitioning[n]
                            else:
                                delta = current_cost - new_cost
                                current_cost = new_cost
                            
                            inbound_edges[part_n] = in_n
                            inbound_edges[part_m] = in_m
        
        if current_cost < best_cost:
            best_partitioning = partitioning
            best_cost = current_cost
    if best_partitioning == None:
        raise Exception("No partitioning found, something broke.")
    return best_partitioning

"""
Algorithm from "EdgeMap: An Optimized Mapping Toolchain for Spiking Neural Network in Edge Computing" by Jianwei Xue.
"""
@core
def partitionEdgeHiding(hg: HyperGraph, max_nodes: int, max_inbound_edges: int, max_partitions: int) -> list[int]:
    initial_partitions_count = math.ceil(hg.nodes/max_nodes)
    partitions : list[set[int]] = [set() for _ in range(initial_partitions_count)] # usage: partitions[partition_idx] -> set of nodes in partition
    partitions_inbound_sets : list[set[HyperEdge]] = [set() for _ in range(initial_partitions_count)] # usage: partitions_inbound_sets[partition_idx] -> set of hyperedge inbound to that partition

    timerPrint = getTimerPrinter(Settings.PRINT_INTERVAL)

    # prepare to normalize spike frequency
    #spike_frequency_min, spike_frequency_max = math.inf, 0
    #for he in hg:
    #    if he.spike_frequency > spike_frequency_max:
    #        spike_frequency_max = he.spike_frequency
    #    if he.spike_frequency < spike_frequency_min:
    #        spike_frequency_min = he.spike_frequency
    #spike_frequency_span = (spike_frequency_max - spike_frequency_min)*0.01

    # greedy, one-shot
    # NOTE: unless the nodes you connect to have been at least in part seen before you, this works terribly.
    for node in range(hg.nodes):
        timerPrint(f"Working on node: {node}/{hg.nodes}...")
        best_part_idx = -1
        best_delta = -math.inf
        node_inbound_set = hg.getInboundHyperedges(node)
        node_outbound_set = hg.getOutboundHyperedges(node)
        for part_idx, (part, part_inbount_set) in enumerate(zip(partitions, partitions_inbound_sets)):
            # constraints check
            #if len(part) < max_nodes and len(part_inbount_set | node_inbound_set) < max_inbound_edges:
            if len(part) == 0 or len(part) < max_nodes and len(part_inbount_set) + len(node_inbound_set) - sum(1 for he in node_inbound_set if he in part_inbount_set) < max_inbound_edges:
                # how many connections would get "hidden" inside a single partition by this move?
                # in other words: total spike frequency of the transmission that would have src and dst within this core.
                # NOTE: this does NOT actively consider synaptic reuse, but enforce spike resolution within a core, that shall contain both src and dst neuron!
                # NOTE: the penality for larger partitions is "len(part)**2 - (len(part) + 1)**2" that equates "1 - 2*len(part)", this also prevents
                #       merges with partitions with nothing in common when there are instead empty partitions available.
                # >> base version (as in the article)
                delta = sum(he.spike_frequency for he in node_inbound_set if he.source() in part)*100 + sum(he.spike_frequency for he in node_outbound_set for dst in he.destinations() if dst in part)*100 - 1 - 2*len(part)
                # >> bonus for putting nodes with no inbound connections together
                delta += 100 if len(node_inbound_set) == 0 and len(part_inbount_set) == 0 else 0
                # >> looking at second order locality (nodes with the same set of connected nodes, this is synaptic reuse - instead of first order, where you only look at directly connected nodes)
                #delta = sum(...)*100 - 1 - 2*len(part)
                if delta > best_delta:
                    best_part_idx = part_idx
                    best_delta = delta
        if best_part_idx < 0:
            partitions.append({node})
            partitions_inbound_sets.append(set(node_inbound_set))
        else:
            partitions[best_part_idx].add(node)
            partitions_inbound_sets[best_part_idx].update(node_inbound_set)
    # enforce max_partitions constraint
    if len(partitions) > max_partitions:
        raise Exception(f"Partitioning could only form {len(partitions)} > {max_partitions} clusters under the provided constraints.")
    
    result = [-1 for _ in range(hg.nodes)]
    for i, partition in enumerate(partitions):
        for node in partition:
            result[node] = i
    return result

"""
Novel idea derived from 'greedyEdgeHiding'.
We bundle together hyperedges based on them being connected to the same nodes.

General idea:
- order hedges by descending spike frequency
- pick the first hedge, iterate over all its connected nodes and their connected hedges (d*h)
- select the first seen hedge by occurrencies weighted by its spike frequency
- join all their nodes together in the current partition, if they are too many for the
  constraints, start a new partition and continue from there
- pick the next hyperedge by total spike frequency, and repeat
- repeat for the next hedge in queue, omitting those already seen

Complexity bound: O(e*d*h)
"""
@core
def partitionHyperedgeHiding(hg: HyperGraph, max_nodes: int, max_inbound_edges: int, max_partitions: int) -> list[int]:
    partitions : list[int] = [-1 for _ in range(hg.nodes)]
    #sorted_hes = sorted(hg.hyperedges, key = lambda he : he.spike_frequency, reverse = True)
    sorted_hes = sorted(hg.hyperedges, key = lambda he : (he.spike_frequency + 0.000001)*len(he), reverse = True)

    timerPrint = getTimerPrinter(Settings.PRINT_INTERVAL)

    seen_hes = set()
    next_partition_idx = 0
    for he in sorted_hes:
        if he in seen_hes:
            continue
        seen_hes.add(he)
        timerPrint(f"Working on hyperedge: {len(seen_hes)}/{len(sorted_hes)}...")
        
        #ranking : AddressableMaxPQ[HyperEdge, int] = AddressableMaxPQ(lambda he, cnt : (cnt - math.log2(len(he) - cnt + 1))*he.spike_frequency, lambda : 0)
        #ranking : AddressableMaxPQ[HyperEdge, int] = AddressableMaxPQ(lambda he, cnt : cnt*math.log10(he.spike_frequency), lambda : 0)
        ranking : AddressableMaxPQ[HyperEdge, int] = AddressableMaxPQ(lambda he, cnt : cnt/len(he), lambda : 0) # sometimes 'he.spike_frequency*cnt/len(he)' works better...
        nodes_count = 0 # tracks nodes involved in the present partition
        inbound_set = set() # tracks the inbound hyperedges to the present partition
        for node in he:
            if partitions[node] != -1:
                continue
            inbound_set.update(hg.getInboundHyperedges(node))
            if nodes_count == max_nodes or len(inbound_set) > max_inbound_edges:
                inbound_set = set(hg.getInboundHyperedges(node))
                nodes_count = 0
                next_partition_idx += 1
            nodes_count += 1
            partitions[node] = next_partition_idx
            for other_he in hg.getTouchingHyperedges(node): # using 'getInboundHyperedges' works too...
                if other_he not in seen_hes:
                    ranking[other_he] += 1
        
        if nodes_count == 0:
            continue
        
        while ranking:
            # greedy, second-order
            best_he, _ = ranking.popMax()
            seen_hes.add(best_he)
            timerPrint(f"Working on hyperedge: {len(seen_hes)}/{len(sorted_hes)}...")
            for node in best_he:
                if partitions[node] != -1:
                    continue
                inbound_set.update(hg.getInboundHyperedges(node))
                if nodes_count == max_nodes or len(inbound_set) > max_inbound_edges:
                    inbound_set = set(hg.getInboundHyperedges(node))
                    nodes_count = 0
                    next_partition_idx += 1
                nodes_count += 1
                partitions[node] = next_partition_idx
                for other_he in hg.getTouchingHyperedges(node): # using 'getInboundHyperedges' works too...
                    if other_he not in seen_hes:
                        ranking[other_he] += 1
    
    # enforce max_partitions constraint
    if next_partition_idx + 1 > max_partitions:
        raise Exception(f"Partitioning could only form {next_partition_idx} > {max_partitions} clusters under the provided constraints.")
    
    return partitions

# MORE COMPACT FORM (functionally equivalent)
@core
def partitionHyperedgeHidingCompact(hg: HyperGraph, max_nodes: int, max_inbound_edges: int, max_partitions: int) -> list[int]:
    partitions : list[int] = [-1 for _ in range(hg.nodes)]
    # ordering by either only-length or only-spike-frequency seems to work better than to do so by their product...
    #sorted_hes = sorted(hg.hyperedges, key = lambda he : (he.spike_frequency + 0.000001)*len(he), reverse = True)
    sorted_hes = sorted(hg.hyperedges, key = lambda he : he.spike_frequency, reverse = True)
    #sorted_hes = sorted(hg.hyperedges, key = lambda he : len(he), reverse = True)
    hes_length = {he : len(he) for he in hg.hyperedges}

    timerPrint = getTimerPrinter(Settings.PRINT_INTERVAL)

    seen_hes = set()
    next_partition_idx = 0
    sorted_hes_iterator = (he for he in sorted_hes if he not in seen_hes)
    # next hyperedge: the one with the highest overlap ratio
    ranking : AddressableMaxPQ[HyperEdge, int] = AddressableMaxPQ(lambda he, cnt : cnt/hes_length[he], int) # sometimes 'he.spike_frequency*cnt/len(he)' works better...
    nodes_count = 0 # tracks nodes involved in the present partition
    inbound_set = set() # tracks the inbound hyperedges to the present partition
    while True:
        if len(ranking) != 0:
            he, _ = ranking.popMax()
        else:
            he = next(sorted_hes_iterator, None)
            if he == None:
                break
        seen_hes.add(he)
        timerPrint(f"Working on hyperedge: {len(seen_hes)}/{len(sorted_hes)}...")
       
        # next node: the one with least new hyperedges, and then the most common hyperedges
        nodes = {node : set(hg.getInboundHyperedges(node)) for node in he if partitions[node] == -1}
        while len(nodes) > 0:
            if len(inbound_set) == 0:
                # empty inbound set? Pick the node with the largest inbound set (see Loihi Compiler), as to make sure it fits
                best_node, inbound = max(nodes.items(), key = lambda item : len(item[1]))
                #best_node, inbound = min(nodes.items(), key = lambda item : (len(item[1]), sum(ohe.spike_frequency for ohe in hg.getOutboundHyperedges(item[0]))))
            else:
                # this already gives the minimum overlap node, if it can't fit, no other node can, thus we need a new partition
                best_node, inbound = min(nodes.items(), key = lambda item : (len(item[1] - inbound_set), -len(item[1]))) # TODO: in case of tie, break it by total spike frequency
                #best_node, inbound = max(nodes.items(), key = lambda item : len(item[1] & inbound_set)/len(item[1]) if item[1] else 0)
            inbound_set.update(inbound)
            if nodes_count == max_nodes or len(inbound_set) > max_inbound_edges:
                #ranking.clear()
                #for r in tuple(ranking.keys()):
                #    ranking[r] *= 0.2
                if nodes_count == 0:
                    raise Exception(f"Node {best_node} has more inbound hyperedges than the hardware can handle per-core: {len(inbound_set)} > {max_inbound_edges}.")
                inbound_set.clear()
                nodes_count = 0
                next_partition_idx += 1
                continue
            nodes.pop(best_node)
            nodes_count += 1
            partitions[best_node] = next_partition_idx
            for other_he in hg.getTouchingHyperedges(best_node):
                if other_he not in seen_hes:
                    hes_length[other_he] -= 1
                    if hes_length[other_he] == 0:
                        seen_hes.add(other_he)
                        ranking.pop(other_he)
                    else:
                        ranking[other_he] += 1
    
    # enforce max_partitions constraint
    if next_partition_idx + 1 > max_partitions:
        raise Exception(f"Partitioning could only form {next_partition_idx} > {max_partitions} clusters under the provided constraints.")
    
    return partitions

# CONSIDERATION:
# The LSH-based setlist method did a query for the highest overlapping node among all nodes to create pairs, and repeat.
# This hehiding does the query manually, but limits the scope to nodes in the same hyperedge, pushing the problem a second
# query level that selects hyperedges, it then builds partitions sequentially instead of pairing nodes.
# A middle-ground could be all-nodes max-overlap queries with LSH and constructing partitions sequentially?
# However, nodes in the same hyperedge are also very likely to have high overlap, is there really a need to extend
# the query beyond the boundaries of the hyperedge? Likely yes, the question is how much it pays off tho.
#
# Another way to interpret hehiding is that we limit queries for the highest overlap node to only those connected by the
# presented hyperedge. The idea being that co-membership in one hyperedge leads to a high likelihood of co-membership in
# others for most involved nodes. Possible limitation: we force in the same sequence of partitions all nodes from one
# hyperedge before moving to the next one, this is fatal in case the above idea doesn't hold for every node.
#
# Ultimately, the key is selecting the scope of the search for the highest overlap node efficiently.
# Hehiding uses the "tail/spill" of the previous hyperedge to seed the next partition and thus do the scope selection
# w.r.t. other hyperedges overlap with those "tail" nodes. But this is far from optimal...
#
# Once we have a partition, the hehiding logic for filling it up via ranking of other hyperedges and least extra inbound
# connections nodes works well. The issue is how do we seed new partitions.
# Idea: seed them by picking the hyperedge that presents the highest average overlap with others.
# Final result: two continous nested searches, one for the best hyperedge to act as the current scope. The others
# for the highest overlap node within the present partition and hyperedge.
#
# NOTE: a good method must be robust even to the "all spike frequencies are 1.0" case!
#
# Simply put, sequential + feed forward greedy ordering puts down a node once it has been reached by many (w) connections,
# and likely the same connections will be the reason for which the next node will be placed, leading to good overlap!
# This is even more so true when nodes are highly clustered (small-world), so, to improve on it, hehiding must manage
# to immediately spot and exploits groups of hyperedges with high mutual overlap.
#
# Idea, let's try to identify "communities" of densely connected nodes from the hyperedge's perspective.
# Partitions are constrained by distinct inbound hyperedges. So the ideal community definition is:
# "A set of nodes that are covered by mostly the same set of inbound hyperedges."
# That's what we could also call an hyperedge-local cluster. Hence, the job of the heuristic becomes is not only to maximize
# overlap in general, but also to identify groups of hyperedges that strongly co-occur across the same set of nodes, and to
# expand partitions from there.

@core
def partitionHyperedgeHidingOnlyInbound(hg: HyperGraph, max_nodes: int, max_inbound_edges: int, max_partitions: int) -> list[int]:
    partitions : list[int] = [-1 for _ in range(hg.nodes)]
    # ordering by either only-length or only-spike-frequency seems to work better than to do so by their product...
    #sorted_hes = sorted(hg.hyperedges, key = lambda he : (he.spike_frequency + 0.000001)*len(he), reverse = True)
    #sorted_hes = sorted(hg.hyperedges, key = lambda he : he.spike_frequency, reverse = True)
    sorted_hes = sorted(hg.hyperedges, key = lambda he : len(he), reverse = True)
    # counter of each hyperedge's yet-to-assign nodes
    hes_length = {he : len(he) for he in hg.hyperedges}

    timerPrint = getTimerPrinter(Settings.PRINT_INTERVAL)

    seen_hes = set()
    next_partition_idx = 0
    sorted_hes_iterator = (he for he in sorted_hes if he not in seen_hes)
    # next hyperedge: the one with the highest overlap ratio
    ranking : AddressableMaxPQ[HyperEdge, int] = AddressableMaxPQ(lambda he, cnt : he.spike_frequency*cnt/hes_length[he], int) # with or w/out 'he.spike_frequency*cnt/len(he)'?
    #ranking : AddressableMaxPQ[HyperEdge, int] = AddressableMaxPQ(lambda he, cnt : (cnt - hes_length[he])/he.spike_frequency, int) # minimize new nodes
    # likely better:
    #ranking : AddressableMaxPQ[HyperEdge, int] = AddressableMaxPQ(lambda he, cnt : cnt*he.spike_frequency, int) # strictly highest (weighted) overlap
    nodes_count = 0 # tracks nodes involved in the present partition
    inbound_set = set() # tracks the inbound hyperedges to the present partition
    while True:
        if len(ranking) != 0:
            he, _ = ranking.popMax()
        else:
            he = next(sorted_hes_iterator, None)
            if he == None:
                break
        seen_hes.add(he)
        timerPrint(f"Working on hyperedge: {len(seen_hes)}/{len(sorted_hes)}...")
       
        # next node: the one with least new hyperedges, and then the most common hyperedges
        # NOTE: consider destinations only! In particular:
        # - if the source has zero inbound hyperedges, sure, bring it closer to its destinations
        # - otherwise, leave it together with other nodes that have similar inbound sets, since not doing so would
        #   fragment their hyperedges without ever asking their permission (since they would end up with lenght zero!)
        nodes = {node : set(hg.getInboundHyperedges(node)) for node in he.destinations() if partitions[node] == -1}
        if len(hg.getInboundHyperedges(src := he.source())) == 0 and partitions[src] == -1:
            nodes[src] = set()
        while len(nodes) > 0:
            if len(inbound_set) == 0:
                # empty inbound set? Pick the node with the largest inbound set (see Loihi Compiler), as to make sure it fits
                best_node, inbound = max(nodes.items(), key = lambda item : len(item[1]))
                #best_node, inbound = min(nodes.items(), key = lambda item : (len(item[1]), sum(ohe.spike_frequency for ohe in hg.getOutboundHyperedges(item[0]))))
            else:
                # pick the node with the least new inbound hedges, if it can't fit, no other node can, thus we need a new partition;
                # as a tiebreaker, pick the node with the largest inbound set overlap
                best_node, inbound = min(nodes.items(), key = lambda item : (len(item[1] - inbound_set), -len(item[1]))) # TODO: in case of further tie, maybe also break it by total spike frequency?
                ## this gives the maximum relative overlap node, if it can't fit, no other node can, thus we need a new partition
                #best_node, inbound = max(nodes.items(), key = lambda item : len(item[1] & inbound_set)/len(item[1]) if item[1] else 0)
                ## among nodes that fit, pick the one with the most overlapping hyperedges
                #best_node, inbound = max(nodes.items(), key = lambda item : ((new := len(item[1] - inbound_set)) <= max_inbound_edges, len(item[1]) - new, -new))
            inbound_set.update(inbound)
            if nodes_count == max_nodes or len(inbound_set) > max_inbound_edges:
                if nodes_count == 0:
                    raise Exception(f"Node {best_node} has more inbound hyperedges than the hardware can handle per-core: {len(inbound_set)} > {max_inbound_edges}.")
                ranking.clear()
                inbound_set.clear()
                nodes_count = 0
                next_partition_idx += 1
                continue
            nodes.pop(best_node)
            nodes_count += 1
            partitions[best_node] = next_partition_idx
            for other_he in hg.getTouchingHyperedges(best_node):
                if other_he not in seen_hes:
                    hes_length[other_he] -= 1
                    if hes_length[other_he] == 0:
                        seen_hes.add(other_he)
                        ranking.pop(other_he)
                    else:
                        ranking[other_he] += 1
    
    # enforce max_partitions constraint
    if next_partition_idx + 1 > max_partitions:
        raise Exception(f"Partitioning could only form {next_partition_idx} > {max_partitions} clusters under the provided constraints.")
    # S;G
    return partitions

@core
# IDEA: re-pick from the top of the ranking the current hyperedge every time you assign a node
def partitionHyperedgeHidingHeReset(hg: HyperGraph, max_nodes: int, max_inbound_edges: int, max_partitions: int) -> list[int]:
    partitions : list[int] = [-1 for _ in range(hg.nodes)]
    remaining_nodes = hg.nodes # just for logging
    # counter of each hyperedge's yet-to-assign nodes
    hes_length = {he : len(he) for he in hg.hyperedges}
    he_co_occurence = {he : 0 for he in hg.hyperedges}
    for n in range(hg.nodes):
        co_occurrences = len(hg.getInboundHyperedges(n))
        for he in hg.getInboundHyperedges(n):
            he_co_occurence[he] += co_occurrences
    # ordering hyperedges by co-occurrence
    # TODO: replace with most seen hyperedges during a random walk! When empty, redo the random walk on unassigned nodes!
    he_priority = sorted(hg.hyperedges, key = lambda he : he_co_occurence[he])
    del he_co_occurence

    timerPrint = getTimerPrinter(Settings.PRINT_INTERVAL)

    next_partition_idx = 0
    #sorted_hes_iterator = (he for he in sorted_hes if he not in seen_hes)
    # next hyperedge: the one with the highest overlap
    ranking : AddressableMaxPQ[HyperEdge, int] = AddressableMaxPQ(lambda he, cnt : cnt*he.spike_frequency, int) # strictly highest (weighted) overlap
    #ranking : AddressableMaxPQ[HyperEdge, int] = AddressableMaxPQ(lambda he, cnt : he.spike_frequency*cnt/hes_length[he], int) # Jaccard-like (weighted) overlap ratio
    nodes_count = 0 # tracks nodes involved in the present partition
    inbound_set = set() # tracks the inbound hyperedges to the present partition
    candidate_nodes : dict[int, set[HyperEdge]] = {} # current set of candidate nodes seen across visited hyperedges for the present partition
    previous_candidate = None # if a node is skipped because it had too many additions, it is placed here and used anyway if it happens to be chosen twice in a row
    while len(hes_length) > 0:
        if len(ranking) != 0:
            #he, _ = ranking.popMax()
            he, _ = ranking.peekMax()
            if nodes_count == 0:
                ranking.clear()
        else:
            while he_priority[-1] not in hes_length:
                he_priority.pop()
                if len(he_priority) == 0:
                    raise Exception("Something impossible happened. No helpful error message can be provided here.")
            he = he_priority[-1]
        
        timerPrint(f"Remaining nodes to assign: {remaining_nodes}/{hg.nodes}...")
        
        # next node: the one with least new hyperedges, and then the most common hyperedges
        for node in he: # he.destinations():
            if partitions[node] == -1 and node not in candidate_nodes:
                candidate_nodes[node] = set(hg.getInboundHyperedges(node))
        #if len(hg.getInboundHyperedges(src := he.source())) == 0 and partitions[src] == -1 and src not in candidate_nodes:
        #    candidate_nodes[src] = set()
        if len(candidate_nodes) == 0:
            hes_length.pop(he)
            ranking.pop(he)
            continue
        
        if len(inbound_set) == 0:
            # empty inbound set? Pick the node with the smallest inbound set
            best_node, inbound = min(candidate_nodes.items(), key = lambda item : len(item[1]))
        else:
            # pick the node with the least new inbound hedges, if it can't fit, no other node can, thus we need a new partition; as a tiebreaker, pick the node with the largest inbound set overlap
            best_node, inbound = min(candidate_nodes.items(), key = lambda item : (len(item[1] - inbound_set), -len(item[1]))) # TODO: in case of further tie, maybe also break it by total spike frequency?
            #candidate_overlaps = [(node, additions, inbound) for node, inbound in candidate_nodes.items() if len(additions := inbound - inbound_set) <= len(inbound)/2] # skip nodes with more than 50% new hyperedges
            #candidate_overlaps = [(node, inbound - inbound_set, inbound) for node, inbound in candidate_nodes.items()] # skip nodes with more than 25% (12.5% was too aggressive) new hyperedges
            #best_node, additions, inbound = min(candidate_overlaps, key = lambda item : (len(item[1]), -len(item[2])))
            #if (len(additions) > len(inbound)/4 or len(additions) + len(inbound_set) > max_inbound_edges) and previous_candidate != best_node:
            #    # TODO: speedup by retaining 'candidate_overlaps'!
            #    previous_candidate = best_node
            #    continue
            #else: # same candidated picked twice, ignoring pre-check and using it anyway
            #    previous_candidate = None
        inbound_set.update(inbound)
        if nodes_count == max_nodes or len(inbound_set) > max_inbound_edges:
            if nodes_count == 0:
                raise Exception(f"Node {best_node} has more inbound hyperedges than the hardware can handle per-core: {len(inbound_set)} > {max_inbound_edges}.")
            inbound_set.clear()
            candidate_nodes.clear()
            nodes_count = 0
            next_partition_idx += 1
            continue
        candidate_nodes.pop(best_node)
        nodes_count += 1
        partitions[best_node] = next_partition_idx
        remaining_nodes -= 1
        for other_he in hg.getTouchingHyperedges(best_node):
            if other_he in hes_length:
                hes_length[other_he] -= 1
                if hes_length[other_he] == 0:
                    hes_length.pop(other_he)
                    ranking.pop(other_he)
                else:
                    ranking[other_he] += 1
    
    # enforce max_partitions constraint
    if next_partition_idx + 1 > max_partitions:
        raise Exception(f"Partitioning could only form {next_partition_idx} > {max_partitions} clusters under the provided constraints.")
    # S;G
    return partitions