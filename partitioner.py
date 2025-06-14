from typing import Optional

from collections import defaultdict, Counter
from datasketch import MinHash, MinHashLSH
from itertools import combinations
import hashlib
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
@core
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
    coarse_hg, groupings = coarsen_hypergraph(hg, min(max_partitions * max_nodes // 2, hg.nodes // 10), max_inbound_edges)
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
Experimental version of 'partitionSetlistMiniHash' with weights.
The implementation is extremely inefficient, but its purpose is to show if using weights can improve the result.
"""
@core
def partitionSetlistMiniHashWeightsTEMP(hg: HyperGraph, N: int, M: int, K: int, num_perm : int = 256, threshold : float = 0.0) -> list[int]:
    sets : list[dict[int, float]] = []
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
        #       but this worsens performance since it consumes an inbound edge slow for a weakly shared hyperedge!
        #if inbound:
        #    d[n] = average_sf / len(inbound)
        #else:
        #    d[n] = 0.0
        sets.append(d)

    # TODO: setup fast query infrastructure
    # lhs = ...

    # EXPERIMENTAL VERSION WITH A SLOW DISTANCE CALCULATION
    def distance(set1 : dict[int, float], set2 : dict[int, float]) -> float:
        return sum(v + set2[k] for k, v in set1.items() if k in set2)
    
    clusters = {} # cid -> { 'union_set', 'count', 'minhash' (unused for now) }
    assignments = [] # will hold cid for each input set
    next_cid = 0

    for i, s in enumerate(sets):
        # TODO: prepare weighted MinHash
        # TODO: pull candidates from the query infrastructure
        cand_ids = [cid for cid in clusters.keys()]
        
        # pick the best mergeable cluster
        best_cid, best_jacc = None, 0.0
        for cid in cand_ids:
            cl = clusters[cid]
            if cl['count'] >= N:
                continue

            # estimate weighted Jaccard distance
            j = distance(s, cl['union_set']) # use cl['minhash'] eventually here!
            if j <= best_jacc:
                continue

            # Exact union‐size check via intersection count
            if len(set(cl['union_set'].keys()) | set(s.keys())) <= M:
                best_cid, best_jacc = cid, j

        # no existing cluster fits, start a new one
        if best_cid is None or best_jacc < threshold / i:
            cid = next_cid
            next_cid += 1
            clusters[cid] = {'union_set': s.copy(), 'count': 1, 'minhash': None}
            # TODO: add new cluster to query infrastructure
            #lsh.insert(cid, mh)
        else:
            # merge into the chosen cluster
            cid = best_cid
            cl = clusters[cid]
            cl['union_set'] |= s
            cl['count'] += 1
            # TODO: update cluster’s MinHash: pointwise min of hashvalues
            #for i in range(num_perm):
            #    cl['minhash'].hashvalues[i] = min(cl['minhash'].hashvalues[i], mh.hashvalues[i])
            # TODO: re‐index query infrastruture so its buckets reflect the updated sketch
            #lsh.remove(cid)
            #lsh.insert(cid, cl['minhash'])

        # record which cluster this set went into
        assignments.append(cid)

    # enforce the <= K clusters requirement
    unique_cids = sorted(clusters.keys())
    if len(unique_cids) > K:
        raise Exception(f"Partitioning could only form {len(unique_cids)} > {K} clusters under the provided N and M constraints.")

    # remap arbitrary cid values into 0..C–1
    cid_map = {old: new for new, old in enumerate(unique_cids)}
    labels  = [cid_map[c] for c in assignments]
    return labels

"""
Experimental version of 'partitionSetlistMiniHashTEMP' with weights.
The implementation is extremely inefficient, but its purpose is to show if using weights can improve the result.

Variant: here we don't start with zero clusters, but with each node initially being its own cluster.
"""
@core
def partitionSetlistMiniHashWeightsTEMPVAR(hg: HyperGraph, N: int, M: int, K: int, num_perm : int = 256, threshold : float = 2.0) -> list[int]:
    class Cluster:
        def __init__(self, nodes_set : dict[int, float]):
            self.nodes_set = nodes_set
            self.count = 1
            #self.minhash = ...
    
    sets : dict[Cluster] = dict()
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
        #       but this worsens performance since it consumes an inbound edge slow for a weakly shared hyperedge!
        #if inbound:
        #    d[n] = average_sf / len(inbound)
        #else:
        #    d[n] = 0.0
        sets[n] = Cluster(d)

    # TODO: setup fast query infrastructure
    # TODO: prefill it with all elements of 'sets'
    # lhs = ...

    # EXPERIMENTAL VERSION WITH A SLOW DISTANCE CALCULATION
    def distance(set1 : dict[int, float], set2 : dict[int, float]) -> float:
        return sum(v + set2[k] for k, v in set1.items() if k in set2)
    
    queue = list(sets.keys())
    assignments = [-1 for _ in range(hg.nodes)] # will hold cid for each input set
    next_cid = 0
    merged = True

    while merged:
        merged = False
        while queue:
            i = queue.pop(0)
            cluster = sets[i]

            # TODO: prepare weighted MinHash
            # TODO: pull candidates from the query infrastructure
            cand_ids = [cid for cid in sets.keys() if cid != i]
            
            # pick the best mergeable cluster
            best_cid, best_jacc = None, 0.0
            for cid in cand_ids:
                cl = sets[cid]
                if cl.count >= N:
                    continue

                # estimate weighted Jaccard distance
                j = distance(cluster.nodes_set, cl.nodes_set) # use cl['minhash'] eventually here!
                if j <= best_jacc:
                    continue

                # Exact union‐size check via intersection count
                if len(set(cl.nodes_set.keys()) | set(cluster.nodes_set.keys())) <= M:
                    best_cid, best_jacc = cid, j

            # merge into the chosen cluster
            if best_cid == None and best_jacc >= threshold * (len(sets) / hg.nodes)**2:
                if assignments[best_cid] == -1:
                    assignments[best_cid] = next_cid
                    next_cid += 1
                assignments[i] = assignments[best_cid]
                cl = sets[best_cid]
                cl.nodes_set |= cluster.nodes_set
                cl.count += cluster.count
                sets.pop(i)
                try:
                    queue.remove(best_cid)
                except:
                    pass
                # TODO: update cluster’s MinHash: pointwise min of hashvalues
                # TODO: re‐index query infrastruture so its buckets reflect the updated sketch
                merged = True
        queue = list(sets.keys())

    for i in range(len(assignments)):
        if assignments[i] == -1:
            assignments[i] = next_cid
            next_cid += 1

    # enforce the <= K clusters requirement
    if next_cid > K:
        raise Exception(f"Partitioning could only form {next_cid} > {K} clusters under the provided N and M constraints.")

    return assignments

"""
Experimental version of 'partitionSetlistMiniHashTEMPVAR'.
The implementation is stupidly inefficient, but its purpose is to show if picking always the best pair yields good results.

Variant: here we simply iterate over all pairs instead of just all nodes (n -> n^2 complexity).
"""
@core
def partitionSetlistMiniHashWeightsTEMPEXH(hg: HyperGraph, N: int, M: int, K: int, num_perm : int = 256, threshold : float = 0.0) -> list[int]:
    class Cluster:
        def __init__(self, nodes_set : dict[int, float]):
            self.nodes_set = nodes_set
            self.count = 1
            #self.minhash = ...
    
    sets : dict[int, Cluster] = dict()
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
        #       but this worsens performance since it consumes an inbound edge slow for a weakly shared hyperedge!
        #if inbound:
        #    d[n] = average_sf / len(inbound)
        #else:
        #    d[n] = 0.0
        sets[n] = Cluster(d)

    # TODO: setup fast query infrastructure
    # TODO: prefill it with all elements of 'sets'
    # lhs = ...

    # EXPERIMENTAL VERSION WITH A SLOW DISTANCE CALCULATION
    def distance(set1 : dict[int, float], set2 : dict[int, float]) -> float:
        set1_only = sum(v for k, v in set1.items() if k not in set2)
        set2_only = sum(v for k, v in set2.items() if k not in set1)
        intersection = sum(v + set2[k] for k, v in set1.items() if k in set2)
        union = set1_only + set2_only + intersection
        return intersection / union if union > 0 else 0.0
    
    assignments = DisjointSet(i for i in range(hg.nodes))
    merged = True

    while merged:
        merged = False
        best_cid1, best_cid2, best_jacc = None, None, 0.0
        keys = list(sets.keys())
        Nskips, Mskips = 0, 0
        for _i in range(len(keys)):
            i = keys[_i]
            cluster1 = sets[i]
            if cluster1.count >= N:
                    Nskips += _i + 1
                    continue
            
            for _j in range(_i):
                j = keys[_j]
                cluster2 = sets[j]
                if cluster2.count + cluster1.count > N:
                    Nskips += 1
                    continue

                # ISSUE: too many nodes remain unclustered!
                #        This occurs because

                # estimate weighted Jaccard distance
                #d = distance(cluster1.nodes_set, cluster2.nodes_set) # use cl['minhash'] eventually here!
                #d = distance(cluster1.nodes_set, cluster2.nodes_set) / min(cluster1.count, cluster2.count)
                # IDEA: to avoid putting together only the best nodes, punish merges between already large clusters!
                d = distance(cluster1.nodes_set, cluster2.nodes_set) / max(cluster1.count, cluster2.count)
                if d <= best_jacc:
                    continue

                # Exact union‐size check via intersection count
                if len(set(cluster1.nodes_set.keys()) | set(cluster2.nodes_set.keys())) <= M:
                    best_cid1, best_cid2, best_jacc = i, j, d
                else:
                    Mskips += 1

        print(best_cid1, best_jacc, Nskips, Mskips, len(sets))
        # merge into the chosen cluster
        if best_cid1 != None and best_jacc >= threshold * (len(sets) / hg.nodes)**2:
            assignments.union(best_cid1, best_cid2)
            cluster1 = sets[best_cid1]
            cluster2 = sets[best_cid2]
            #cluster1.nodes_set |= cluster2.nodes_set
            cluster1.nodes_set = dict_sum(cluster1.nodes_set, cluster2.nodes_set)
            cluster1.count += cluster2.count
            sets.pop(best_cid2)
            # TODO: update cluster’s MinHash: pointwise min of hashvalues
            # TODO: re‐index query infrastruture so its buckets reflect the updated sketch
            merged = True

    result = [-1 for _ in range(hg.nodes)]
    for i, part in enumerate(assignments):
        for node in part:
            result[node] = i

        # enforce the <= K clusters requirement
        if i >= K:
            raise Exception(f"Partitioning could only form {i + 1} > {K} clusters under the provided N and M constraints.")

    return result

"""
Same as 'partitionSetlistMiniHash', but each inbound hyperedge is weighted by its spike frequency.
As a result, sets became dictionaries, and we use the weighted Jaccard distance for similarity.

Extra args:
- s: sketch size (# samples)
- b: number of LSH bands (sketch rows per band = r = s//b)
"""
@core
def partitionSetlistMiniHashWeighted(hg: HyperGraph, N: int, M: int, K: int, s: int = 128, b: int = 16) -> list[int]:
    # build the inbound edges's sources sets list
    dicts : list[dict[int, float]] = []
    #largest_dict_size = 0
    for n in range(hg.nodes):
        d = {}
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
            d[n] = 0.0
        #largest_dict_size = max(largest_dict_size, len(s))
        dicts.append(d)

    if s % b != 0:
        raise ValueError("Sketch size s must be divisible by band count b")
    r = s // b
    def _rand_uniform(key: int, seed: int) -> float:
        h = hashlib.md5(f"{key}-{seed}".encode()).digest()
        v = int.from_bytes(h[:8], 'big')
        return (v + 1) / (2**64 + 1)
    def cws_sketch(d: dict[int, float]) -> list[int]:
        sketch = [None] * s
        mins = [math.inf] * s
        for key, w in d.items():
            if w <= 0: continue
            for j in range(s):
                u = _rand_uniform(key, j)
                y = -math.log(u) / w
                if y < mins[j]:
                    mins[j] = y
                    sketch[j] = key
        return sketch

    # LSH buckets
    band_buckets = [defaultdict(set) for _ in range(b)]
    clusters = {}  # cid -> meta
    assignments = [-1] * len(dicts)
    next_cid = 0

    # --- streaming insertion ---
    for i, d in enumerate(dicts):
        d_total = sum(d.values())
        sk = cws_sketch(d)
        # LSH query
        cands = set()
        for band in range(b):
            sig = tuple(sk[band*r:(band+1)*r])
            cands |= band_buckets[band].get(sig, set())
        # pick best
        best_cid, best_score = None, 0.0
        for cid in cands:
            c = clusters[cid]
            if c['count'] + 1 > N: continue
            if len(c['weights'] | d.keys()) > M: continue
            # exact weighted Jaccard
            inter = sum(min(c['weights'].get(k,0.0), w) for k,w in d.items())
            union = c['total_weight'] + d_total - inter
            score = inter/union if union>0 else 0.0
            if score > best_score:
                best_score, best_cid = score, cid
        if best_cid is not None:
            # merge
            c = clusters[best_cid]
            for k,w in d.items(): c['weights'][k] = c['weights'].get(k,0.0)+w
            c['total_weight'] += d_total; c['count']+=1; c['elems']=len(c['weights'])
            # update LSH
            old_sk = c['sketch']
            for band in range(b): band_buckets[band][tuple(old_sk[band*r:(band+1)*r])].remove(best_cid)
            new_sk = cws_sketch(c['weights']); c['sketch']=new_sk
            for band in range(b): band_buckets[band][tuple(new_sk[band*r:(band+1)*r])].add(best_cid)
            assignments[i] = best_cid
        else:
            # new cluster
            if next_cid >= K:
                # no room: break to refinement
                break
            cid = next_cid; next_cid+=1
            clusters[cid] = {
                'weights': dict(d), 'total_weight': d_total,
                'count':1, 'elems':len(d), 'sketch':sk
            }
            for band in range(b): band_buckets[band][tuple(sk[band*r:(band+1)*r])].add(cid)
            assignments[i] = cid

    # --- hierarchical refinement to reach ≤K clusters ---
    # build list of current clusters
    cids = list(clusters.keys())
    # greedy merge best pair until len(cids) <= K
    while len(cids) > K:
        best_pair, best_score = None, -1.0
        # find best merge candidate among current clusters
        for i in range(len(cids)):
            for j in range(i+1, len(cids)):
                a, b_cid = cids[i], cids[j]
                A, B = clusters[a], clusters[b_cid]
                # capacity checks
                if A['count']+B['count']>N or len(A['weights']|B['weights'])>M: continue
                # exact weighted Jaccard of centroids
                inter = sum(min(A['weights'][k], B['weights'][k]) for k in A['weights'] if k in B['weights'])
                union = A['total_weight']+B['total_weight']-inter
                score = inter/union if union>0 else 0.0
                if score>best_score:
                    best_score, best_pair = score, (a, b_cid)
        if best_pair is None:
            break  # no valid merges
        # perform merge
        a, b_cid = best_pair
        A, B = clusters[a], clusters[b_cid]
        for k,w in B['weights'].items(): A['weights'][k] = A['weights'].get(k,0.0)+w
        A['total_weight'] += B['total_weight']; A['count']+=B['count']; A['elems']=len(A['weights'])
        # remove cluster b_cid
        del clusters[b_cid]
        cids.remove(b_cid)
    # remap assignments
    cid_map = {old:new for new,old in enumerate(cids)}
    final = []
    for idx in assignments:
        if idx in cid_map:
            final.append(cid_map[idx])
        else:
            # unassigned (in break), assign to nearest
            # simple fallback: 0
            final.append(0)
    return final


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