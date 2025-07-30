from typing import Optional

from collections import defaultdict, Counter
from datasketch import MinHash, MinHashLSH
from itertools import combinations
import numpy as np
import random
import heapq
import math

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

    coarse_hes = defaultdict(lambda : 0) # he-tuple -> spike frequency
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
    """
    def coarsen_hypergraph(hg: HyperGraph, target_coarse_nodes: int, max_nodes : int, max_inbound_edges: int, seed : Optional[int] = None) -> tuple[list[tuple[HyperGraph, list[list[int]]]], list[int], list[Counter[int]]]:
        current_hg = hg
        partition_sizes = [1 for _ in range(hg.nodes)] # size of each partition (node) in current_hg
        inbound_he_ids = [Counter(hash(he) for he in hg.getInboundHyperedges(n)) for n in range(hg.nodes)] # inbound hyperedges IDs for each partition (node) in current_hg

        result = []

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
                    # TODO: inefficient max retrieval -> use a heap?
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
            
            current_hg = current_hg.getPartitionsHypergraph(partitions, squish_hyperedges = True)
            partition_sizes = next_partition_sizes
            inbound_he_ids = next_inbound_he_ids

            result.append((current_hg, coarsenings))
            timerPrint(f"Coarsening: {current_hg.nodes} > {target_coarse_nodes}...")
        
        return result, partition_sizes, inbound_he_ids

    """
    Source: "Multilevel k-way Hypergraph Partitioning" by George Karypis
    Updates the candidate 'partitioning' in place!
    """
    def greedy_FM_refinement(hg: HyperGraph, partitioning: list[int], partition_sizes : list[int], inbound_he_ids : list[Counter[int]], max_nodes: int, max_inbound_edges: int, seed : Optional[int] = None) -> None:
        rng = np.random.default_rng(seed)
        nodes = np.arange(hg.nodes)
        rng.shuffle(nodes)
        # TODO: should iterate until no more moves occur? Likely yes, but put a cap on the number of iterations (e.g. 8)...
        for n in nodes:
            connectivity_w_partitions = defaultdict(lambda : 0) # partition -> sum of spike frequency of connections
            for he in hg.getTouchingHyperedges(n):
                for m in he:
                    if m != n:
                        connectivity_w_partitions[partitioning[m]] += he.spike_frequency
            my_partition = partitioning[n]
            loss = connectivity_w_partitions.pop(my_partition) if my_partition in connectivity_w_partitions else 0
            while connectivity_w_partitions:
                best_partition = max(connectivity_w_partitions, key = connectivity_w_partitions.get)
                if connectivity_w_partitions[best_partition] - loss < 0:
                    break
                elif partition_sizes[best_partition] + 1 <= max_nodes and len(best_inbound := inbound_he_ids[best_partition] + (my_inbound := Counter(map(hash, hg.getInboundHyperedges(n))))) <= max_inbound_edges:
                    partitioning[n] = best_partition
                    partition_sizes[best_partition] += 1
                    partition_sizes[my_partition] -= 1
                    inbound_he_ids[best_partition] = best_inbound
                    inbound_he_ids[my_partition] -= my_inbound
                    break
                connectivity_w_partitions.pop(best_partition)

    # TODO: multistart!!!!!!!
    # !!!!!
    # !!!!!
    # =>=> Add a cost estimation for partitions feature in the model!
    # =>=> Extract a random seed for each start by using the initial seed!
    
    # hierarchically coarsened hypergraphs, from less to most coarsened
    coarsening_levels, partition_sizes, inbound_he_ids = coarsen_hypergraph(hg, min(max_partitions, hg.nodes // max_nodes), max_nodes, max_inbound_edges, seed)
    if len(coarsening_levels) == 0:
        if hg.nodes > max_partitions:
            raise Exception("Cannot coarsen the hypergraph.")
        else:
            # TODO: do one round of refinement here still
            return [i for i in range(hg.nodes)]
    # NOTE: partitioning = initialPartitionig( ... )
    # => no need since we coarsen up to the point of having the right number of partitions
    partitions_count = coarsening_levels[-1][0].nodes
    if partitions_count > max_partitions:
        raise Exception("Cannot coarsen up until reaching a sufficiently low number of partitions.")
    partitioning = [i for i in range(coarsening_levels[-1][0].nodes)]
    for i in range(len(coarsening_levels) - 1, -1, -1):
        cl_hg, cl_coarsenings = coarsening_levels[i]
        greedy_FM_refinement(cl_hg, partitioning, partition_sizes, inbound_he_ids, max_nodes, max_inbound_edges, seed)
        # undo the coarsening
        new_partitioning = [-1 for _ in range(coarsening_levels[i - 1][0].nodes if i > 0 else hg.nodes)]
        for p, c in zip(partitioning, cl_coarsenings):
            for n in c:
                new_partitioning[n] = p
        partitioning = new_partitioning
        timerPrint(f"Refining: {i + 1} levels left...")
    greedy_FM_refinement(hg, partitioning, partition_sizes, inbound_he_ids, max_nodes, max_inbound_edges, seed)
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

                # Exact union‐size check via intersection count
                if len(set(cl.weighted_set.keys()) | set(cluster.weighted_set.keys())) <= M:
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
    # higher 'top_k' costs slightly more time for slightly better results (e.g. 2% on both when doubled)
    lhs : WeightedMinHashLSHForest[int] = WeightedMinHashLSHForest(num_perm = 32, tree_count = 16)
    
    timerPrint = getTimerPrinter(Settings.PRINT_INTERVAL)
    
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
        timerPrint(f"Building LSH forest: {n}/{hg.nodes}...")

    queue = list(lhs.ids())
    assignments = DisjointSet(i for i in range(hg.nodes))
    merged = True

    cluster : WeightedMinHashLSHForest.Entry = None
    def valid(other_cluster : WeightedMinHashLSHForest.Entry):
        # Checks:
        # 1) Total count of merged original nodes
        # 2) Exact inbound hyperedges union‐size check via intersection count
        return other_cluster.merge_count + cluster.merge_count <= N and len(set(other_cluster.weighted_set.keys()) | set(cluster.weighted_set.keys())) <= M

    while merged:
        merged = False
        unmerged = []
        while queue:
            i = queue.pop(0)
            cluster = lhs.get(i)

            # TODO: fine tune "count_invalid ="!
            cand_ids, cand_simils = lhs.query_by_id(i, valid, top_k, 2, True)
            
            # pick the best mergeable cluster
            best_cid, best_jacc = None, 0.0
            for cid, sim in zip(cand_ids, cand_simils):
                cl = lhs.get(cid)
                # IDEA: to avoid putting together only the best nodes, punish merges between already large clusters!
                s = sim / max(cluster.merge_count, cl.merge_count)
                if s > best_jacc:
                    best_cid, best_jacc = cid, s

            # merge into the chosen cluster
            if best_cid is not None and best_jacc >= threshold * (len(lhs) / hg.nodes)**2:
                assignments.union(i, best_cid)
                lhs.merge([i, best_cid], merged_set_id = best_cid)
                try:
                    queue.remove(best_cid)
                except:
                    pass
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
        # same logic as 'partitionSequential', but the order of nodes is randomized
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
                edges_n = Counter(hg.getInboundHyperedges(n))
                for m in range(n): # here goes the quadratic complexity...
                    part_n, part_m = partitioning[n], partitioning[m]
                    edges_m = Counter(hg.getInboundHyperedges(m))
                    if part_n != part_m:
                        # it's a swap, assigned nodes will always be fine
                        if len(inbound_edges[part_n] - edges_n + edges_m) <= M and len(inbound_edges[part_m] - edges_m + edges_n) <= M:
                            partitioning[n], partitioning[m] = partitioning[m], partitioning[n]
                            new_cost = cost(hg, partitioning)
                            if new_cost >= current_cost:
                                partitioning[n], partitioning[m] = partitioning[m], partitioning[n]
                            else:
                                delta = current_cost - new_cost
                                current_cost = new_cost
                            inbound_edges[part_n].subtract(edges_n) ; inbound_edges[part_n].update(edges_m)
                            inbound_edges[part_m].subtract(edges_m) ; inbound_edges[part_m].update(edges_n)
        
        if current_cost < best_cost:
            best_partitioning = partitioning
            best_cost = current_cost
    if best_partitioning == None:
        raise Exception("No partitioning found, something broke.")
    return best_partitioning