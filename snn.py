from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Optional, Self, Union

from collections import defaultdict
from array import array
import networkx as nx
import numpy as np
import random
import struct

from prints import *

"""
Directed hyperedge connecting nodes inside an hypergraph.
It has a single source node and one or more destinations.
It is associated with a 'spike_frequency', that is, a weight.
"""
class HyperEdge(Iterable):
    # the first node is the source for the hyperedge
    nodes : tuple[int, ...]
    spike_frequency : float
    _id : int
    
    def __init__(self, source : int, destinations : tuple[int, ...], spike_frequency : float):
        if len(destinations) == 0:
            raise Exception("Hyperedges can't have zero destinations.")
        # TODO: raise an exception if a node occurs more than once in destinations
        self.nodes = (source,) + destinations
        self.spike_frequency = spike_frequency
        self._id = hash(self.nodes + (self.spike_frequency, random.random()))
    
    """
    Returns this hyperedge's source node.
    """
    def source(self) -> int:
        return self.nodes[0]
    
    """
    Returns this hyperedge's destination nodes.
    """
    def destinations(self) -> tuple[int, ...]:
        return self.nodes[1:]
    
    """
    Returns True iif 'other' has the same source and destinations as this hyperedge.
    """
    def sameNodes(self, other : Self) -> bool:
        other_destinations = other.nodes[1:]
        return self.nodes[0] == other.nodes[0] and len(self.nodes) == len(other.nodes) and all(node in other_destinations for node in self.nodes[1:])
    
    """
    Returns an hash that is identical between any two hyperedges with the same source and set of destinations.
    Ignores: spike frequency, order of destinations.
    """
    def sameNodesHash(self) -> int:
        # TODO: make me lazy?
        return hash((self.nodes[0], hash(frozenset(self.nodes[1:]))))
    
    """
    Updates the hyperedge's spike frequency
    """
    # WARNING: also updates the hyperedge's id (and hash).
    # NOTE: no need to update the id, it is already random anyway...
    def updateSpikeFrequency(self, new_spike_frequency : int) -> None:
        self.spike_frequency = new_spike_frequency
        #self._id = hash(self.nodes + (self.spike_frequency, random.random()))
    
    def __iter__(self) -> Iterator[int]:
        return iter(self.nodes)
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __eq__(self, other : Self) -> bool:
        return self.nodes == other.nodes and self.spike_frequency == other.spike_frequency
    
    def __str__(self) -> str:
        return self.nodes.__str__()[:-1] + f", sf = {self.spike_frequency:.1e})"
    
    def __hash__(self):
        return self._id
    
    """
    Number connections in the hyperedge, that is (|nodes| - 1).
    In other words, the number of plain edges the hyperedge corresponds to.
    """
    def connections(self) -> int:
        return len(self.nodes) - 1

"""
Directed hypergraph to model a SNN.
Each edge has one source neuron (node) and represents an axon going into many neurons (nodes) with a synapse for each.
Edges are weighted by the frequency with which they are traversed by a spike.

Arguments:
- nodes: number of nodes in the hypergraph.
- hyperedges: list of ready-to-use Hyperedges or list of tuples representing hyperedges with the first entry being the source.
- spike_frequencies: if 'hyperedges' is a list of tuples, this shall indicate for each tuple, in order, its spike frequency.
- no_checks: if True, hyperedges must be a list of Hyperedges and all checks on arguments are disabled to speedup construction.
"""
class HyperGraph(Iterable):
    # each node is identified by an index in [0, nodes)
    nodes : int
    hyperedges : list[HyperEdge]
    
    # pointers to hyperedges leaving a node
    _outbound : list[tuple[HyperEdge, ...]]
    # pointers to hyperedges entering a node
    _inbound : list[tuple[HyperEdge, ...]]
    
    def __init__(self, nodes : int, hyperedges : list[Union[HyperEdge, tuple[int, ...]]], spike_frequencies : Optional[list[float]] = None, no_checks : bool = False):
        self.nodes = nodes
        if len(hyperedges) == 0:
            self.hyperedges = []
        elif no_checks or all(isinstance(he, HyperEdge) for he in hyperedges):
            self.hyperedges = hyperedges # Be wary, there's no copy here!
        elif all(isinstance(he, tuple) and len(he) >= 2 for he in hyperedges) and spike_frequencies and len(spike_frequencies) == len(hyperedges):
            self.hyperedges = [HyperEdge(he[0], he[1:], spike_frequencies[i]) for i, he in enumerate(hyperedges)]
        else:
            raise Exception(("Failed to build hypergraph. Hyperedges shall be provided either as an empty list, a list of HyperEdge instances, or a list of tuples of at least two entries each."
                             "In the latter case, spike_frequencies must also be a list of the same lenght, while the first entry in each tuple specifies the source node for that hyperedge."))
        if not no_checks and any(node < 0 or node >= nodes for he in self.hyperedges for node in he):
            raise Exception("Invalid hyperedges, all node indices must be in the range [0, nodes).")
        
        # pay the overhead here to build faster access structures
        self._outbound = [[] for _ in range(self.nodes)]
        self._inbound = [[] for _ in range(self.nodes)]
        for he in self.hyperedges:
            self._outbound[he.source()].append(he)
            for d in he.destinations():
                self._inbound[d].append(he)
        map(tuple, self._outbound)
        map(tuple, self._inbound)

    """
    Generate a random hypergraph with 'n' nodes, where each node is the source
    of exactly one hyperedge. The number of destinations for each hyperedge is
    sampled from a Gaussian distribution with mean 'c' and standard deviation 'd'.
    A seed can be provided for repeatability.
    
    Each destination node is selected uniformly at random (excluding the source node).
    Spike frequencies are sampled uniformly from 'spike_frequency_range'.
    """
    @classmethod
    @core
    def generate_random(cls, n: int, c: float, d: float, spike_frequency_range: tuple[float, float] = (0.1, 1.0), seed : Optional[int] = None) -> Self:
        hyperedges = []
        spike_frequencies = []

        rng = np.random.default_rng(seed)
        all_nodes = np.arange(n, dtype = np.int32)
        num_dests = np.clip(rng.normal(loc = c, scale = d, size = n).astype(np.int32), 0, n - 1)
        for source in range(n):
            nd = num_dests[source]
            if nd == 0:
                continue
            candidates = np.delete(all_nodes, source)
            destinations = rng.choice(candidates, size = nd, replace = False)
            hyperedges.append((source, *destinations))
            spike_frequencies.append(rng.uniform(*spike_frequency_range))
        return cls(n, hyperedges, spike_frequencies)

    """
    Lower the HyperGraph into a directed Graph.
    If 'collaps_overlapping_edges' is True, resulting edges with the same source and destination will
    be merged and their spike frequencies added together.
    Complexity: O(e*h) where 'h' are the mean hedges per node.
    """
    def toGraph(self, collapse_overlapping_edges : bool = False) -> Graph:
        if not collapse_overlapping_edges:
            return Graph(self.nodes, [Edge(he.source(), dst, he.spike_frequency) for he in self.hyperedges for dst in he.destinations()])
        else:
            edges : dict[tuple[int, int], Edge] = {}
            for he in self.hyperedges:
                for dst in he.destinations():
                    src = he.source()
                    key = (src, dst)
                    if key in edges:
                        edges[(src, dst)].spike_frequency += he.spike_frequency
                    else:
                        edges[(src, dst)] = Edge(src, dst, he.spike_frequency)
            return Graph(self.nodes, edges.values())

    """
    Returns the hypegraph that arises between partitions of the present hypergraph,
    in which nodes are the partitions and only hyperedges between partitions are kept.
    Complexity: O(e*h) where 'h' are the mean hedges per node.
    
    Args:
    - partitions: list of partitions indices, one per node in the graph, in order.
                  It assigns to each node its partition. Partitions must be indexed
                  sequentially from 0, as they become the new hypergraph's nodes.
    - keep_self_cycles: if True, hyperedges entirely contained in a partition do not
                        disappear, instead are kept as a self-edge from the partition
                        to itself, preserving the total spike frequency.
    - squish_hyperedges: if True, collapses identical post-partitioning hyperedges in
                         a single one. This is more efficient than calling separately
                         the 'squishHyperedges' function.
    """
    def getPartitionsHypergraph(self, partitions : list[int], keep_self_cycles : bool = False, squish_hyperedges : bool = False) -> Self:
        if len(partitions) != self.nodes:
            raise Exception("Each node must be assigned to a partition.")
        new_nodes = len(set(partitions))
        if any(i not in partitions for i in range(0, new_nodes)):
            raise Exception("Partitions must be incrementally indexed from 0 onward.")
        
        # compute new hyperedges
        # HP: impossible to have zero destinations
        if not squish_hyperedges:
            new_hyperedges : list[HyperEdge] = []
            for he in self.hyperedges:
                affected_partitions = set(partitions[node] for node in he.destinations())
                src_part = partitions[he.source()]
                if not keep_self_cycles and src_part in affected_partitions:
                    affected_partitions.remove(src_part)
                if len(affected_partitions) > 1:
                    new_hyperedges.append(HyperEdge(src_part, tuple(affected_partitions), he.spike_frequency))
        else:
            new_hyperedges : dict[int, HyperEdge] = {}
            for he in self.hyperedges:
                affected_partitions = set(partitions[node] for node in he.destinations())
                src_part = partitions[he.source()]
                if not keep_self_cycles and src_part in affected_partitions:
                    affected_partitions.remove(src_part)
                if len(affected_partitions) > 1:
                    samenodes_hash = hash((src_part, hash(frozenset(affected_partitions))))
                    if samenodes_hash not in new_hyperedges:
                        new_hyperedges[samenodes_hash] = HyperEdge(src_part, tuple(affected_partitions), he.spike_frequency)
                    else:
                        new_hyperedges[samenodes_hash].updateSpikeFrequency(new_hyperedges[samenodes_hash].spike_frequency + he.spike_frequency)
            new_hyperedges = list(new_hyperedges.values())

        return HyperGraph(new_nodes, new_hyperedges)
    
    """
    Given a node's index, returns the list of hyperedges outbound from that node.
    Throws an exception if the node's index is invalid.
    """
    def getOutboundHyperedges(self, node : int) -> tuple[HyperEdge]:
        if node < 0 or node >= self.nodes:
            raise Exception("Invalid node.")
        return self._outbound[node]
    
    """
    Given a node's index, returns the list of hyperedges inbound for that node.
    Throws an exception if the node's index is invalid.
    """
    def getInboundHyperedges(self, node : int) -> tuple[HyperEdge, ...]:
        if node < 0 or node >= self.nodes:
            raise Exception("Invalid node.")
        return self._inbound[node]
    
    """
    Given a node's index, returns the list of hyperedges touching that node.
    Throws an exception if the node's index is invalid.
    """
    def getTouchingHyperedges(self, node : int) -> tuple[HyperEdge, ...]:
        if node < 0 or node >= self.nodes:
            raise Exception("Invalid node.")
        return self._outbound[node] + self._inbound[node]
    
    """
    Add an extra nodes to the graph.
    """
    def addNodes(self, amount : int = 1) -> None:
        if amount < 0:
            raise Exception("The amount of nodes to add must be positive.")
        self.nodes += amount
    
    """
    Adds an HyperEdge to the HyperGraph.
    If the HyperEdge uses node indices that are not valid, an exception is thrown.
    If 'add_missing_nodes' is True, using a node index beyond those existing in the
    HyperGraph will result in all nodes up to, and including, that one, being created.
    Complexity: O(h) where 'h' are the mean hedges per node.
    """
    def addHyperedge(self, hyperedge : HyperEdge, add_missing_nodes : bool = False) -> None:
        if any(node < 0 for node in hyperedge):
            raise Exception("Negative node index in the provided hyperedge.")
        if not add_missing_nodes and any(node >= self.nodes for node in hyperedge):
            raise Exception("Out of bounds node index in the provided hyperedge.")
        else:
            self.nodes = max(self.nodes, max(hyperedge))
        self.hyperedges.append(hyperedge)
        self._outbound[hyperedge.source()] += (hyperedge,)
        for node in hyperedge.destinations():
            self._inbound[node] += (hyperedge,)
    
    """
    Adds a sequence of HyperEdges to the HyperGraph.
    DANGER: hyperedges are NOT checked for validity, ensure that all their nodes
    are valid hypergraph nodes for this HyperGraph instance!
    Complexity: O(h) where 'h' are the mean hedges per node.
    """
    def addHyperedges(self, hyperedges : Iterable[HyperEdge]) -> None:
        for he in hyperedges:
            self.hyperedges.append(he)
            self._outbound[he.source()] += (he,)
        for node in he.destinations():
            self._inbound[node] += (he,)
    
    """
    Any pair of HyperEdges that share the same source and destinations are fused in
    a single new HyperEdge having for spike frequency the sum of the originals'.
    Complexity: O(n*h^2) where 'h' are the mean hedges per node.
    """
    @core
    def squishHyperedges(self) -> None:
        to_delete = set()
        # NOTE: this exploits the fact that an hyperedge has a single source
        for src in range(self.nodes):
            hedges = self._outbound[src]
            disjoint_set = {} # usage: disjoint_set[idx] -> parent_idx (where 'parent' is an identical he that accumulated the spike frequency)
            for he_idx1 in range(1, len(hedges)):
                # search for an he identical to the one in he_idx1
                for he_idx2 in range(he_idx1):
                    if self.hyperedges[he_idx1].sameNodes(self.hyperedges[he_idx2]):
                        # at this point, he_idx1 is never in disjoint_set
                        while he_idx2 in disjoint_set:
                            he_idx2 = disjoint_set[he_idx2]
                        disjoint_set[he_idx1] = he_idx2
                        self.hyperedges[he_idx2].spike_frequency += self.hyperedges[he_idx1].spike_frequency
                        break
            self._outbound[src] = tuple(he for he in self._outbound[src] if he not in disjoint_set)
            to_delete.update(disjoint_set)
        self.hyperedges = [he for he in self.hyperedges if he not in to_delete]
        for dst in range(self.nodes):
            self._inbound[dst] = tuple(he for he in self._inbound[dst] if he not in to_delete)
        
        # NOTE: general case variant (when the one-source HP does not hold)
        #seen = defaultdict(list) # usage: seen[hedge_xor] -> list of distinct hedges with the same xor
        #to_delete = set()
        #for he in self.hyperedges:
        #    # needs: from functools import reduce
        #    he_xor = reduce(lambda x, y : x ^ y, he) ^ 0xff51afd7ed558ccd # seed
        #    if he_xor in seen:
        #        found = False
        #        for seen_he in seen[he_xor]:
        #            if he.sameNodes(seen_he):
        #                found = True
        #                to_delete.add(he)
        #                break
        #        if not found:
        #            seen[he_xor].append(he)
        #    else:
        #        seen[he_xor].append(he)
        #self.hyperedges = [he for he in self.hyperedges if he not in to_delete]
        #for node in range(self.nodes):
        #    self._inbound[node] = tuple(he for he in self._inbound[node] if he not in to_delete)
        #    self._outbound[node] = tuple(he for he in self._outbound[node] if he not in to_delete)
    
    """
    Removes and returns the 'fraction*100'% of hyperedges with the lowest spike frequency.
    To reinstate removed hyperedges, use 'addHyperedges'.
    This method can remove part of an hyperedge, as such, after reinstating hyperedges,
    it is recommended to call 'squishHyperedges'.
    Also returns the count of removed connections and removed spike frequency.
    Complexity: O(e*d+n*h).
    """
    @core
    def removeHyperEdgesFraction(self, fraction : float) -> tuple[list[HyperEdge], int, float]:
        self.hyperedges.sort(key = lambda he : he.spike_frequency)
        
        total_conn = sum(he.connections() for he in self.hyperedges)
        target_conn = int(total_conn * fraction)
        removed = set()
        removed_sf = 0
        removed_conn = 0
        kept = []
        for he in self.hyperedges:
            if removed_conn + he.connections() <= target_conn:
                removed.add(he)
                connections = he.connections()
                removed_sf += he.spike_frequency*connections
                removed_conn += connections
            elif removed_conn < target_conn:
                edges_to_remove = target_conn - removed_conn
                removed.add(HyperEdge(he.source(), he.destinations()[:edges_to_remove], spike_frequency = he.spike_frequency))
                kept.append(HyperEdge(he.source(), he.destinations()[edges_to_remove:], spike_frequency = he.spike_frequency))
                removed_sf += he.spike_frequency*edges_to_remove
                removed_conn += edges_to_remove
            else:
                kept.append(he)
        self.hyperedges = kept
        for node in range(self.nodes):
            self._inbound[node] = tuple(he for he in self._inbound[node] if he not in removed)
            self._outbound[node] = tuple(he for he in self._outbound[node] if he not in removed)
        return removed, removed_conn, removed_sf
    
    """
    Same as 'removeEdgesFraction' but 'fraction' indicates the fraction of total
    spike frequency to remove, instead of the fraction of hyperedges.
    """
    @core
    def removeSpikeFrequencyFraction(self, fraction : float) -> tuple[set[HyperEdge], int, float]:
        self.hyperedges.sort(key = lambda he : he.spike_frequency)
        
        total_sf = self.totalSpikeFrequency()
        target_sf = total_sf * fraction
        removed = set()
        removed_sf = 0
        removed_conn = 0
        kept = []
        for he in self.hyperedges:
            if removed_sf + he.spike_frequency*he.connections() <= target_sf:
                removed.add(he)
                connections = he.connections()
                removed_sf += he.spike_frequency*connections
                removed_conn += connections
            elif removed_sf + he.spike_frequency <= target_sf:
                edges_to_remove = int((target_sf - removed_sf) // he.spike_frequency)
                removed.add(HyperEdge(he.source(), he.destinations()[:edges_to_remove], spike_frequency = he.spike_frequency))
                kept.append(HyperEdge(he.source(), he.destinations()[edges_to_remove:], spike_frequency = he.spike_frequency))
                removed_sf += he.spike_frequency*edges_to_remove
                removed_conn += edges_to_remove
            else:
                kept.append(he)
        self.hyperedges = kept
        for node in range(self.nodes):
            self._inbound[node] = tuple(he for he in self._inbound[node] if he not in removed)
            self._outbound[node] = tuple(he for he in self._outbound[node] if he not in removed)
        return removed, removed_conn, removed_sf
    
    """
    Returns the total spike frequency on the hypergraph's connections.
    Each hyperedge is treated as (connected_nodes - 1) connections.
    Complexity: O(e).
    """
    def totalSpikeFrequency(self) -> float:
        result = 0
        for he in self.hyperedges:
            result += he.spike_frequency*he.connections()
        return result
    
    def __iter__(self) -> Iterator[HyperEdge]:
        return iter(self.hyperedges)
    
    def __str__(self) -> str:
        result = '['
        for he in self.hyperedges:
            result += he.__str__() + ', '
        return result[:-2] + ']'
    
    """
    Returns the number of connections in the hypergraph.
    That is, the total nodes touched across hyperedges.
    """
    def totalConnections(self) -> int:
        return sum(len(he) for he in self.hyperedges)
    
    """
    Returns a summary of the hypergraph's statistics.
    """
    def getStatistics(self) -> dict[str, float]:
        connections = self.totalConnections()
        return {
            'nodes_count': self.nodes,
            'edges_count': len(self.hyperedges),
            'nodes_per_edge_mean': connections/len(self.hyperedges),
            'edges_per_node_mean': connections/self.nodes,
            'outbound_edges_per_node_mean': sum(len(node_hes) for node_hes in self._outbound)/self.nodes,
            'inbound_edges_per_node_mean': sum(len(node_hes) for node_hes in self._inbound)/self.nodes,
            'spike_frequency_mean': self.totalSpikeFrequency()/len(self.hyperedges), # this is per connection, divide by the avg. number of connections per hyperedge to get the avg. spike frequency per hyperedge
        }
    
    """
    Save the present hypergraph to 'path'.
    """
    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            f.write(struct.pack('<II', self.nodes, len(self.hyperedges)))
            for he in self.hyperedges:
                src = he.source()
                dsts = he.destinations()
                freq = he.spike_frequency
                f.write(struct.pack('<I', len(dsts)))
                f.write(struct.pack('<I', src))
                f.write(array('I', dsts).tobytes())
                f.write(struct.pack('<f', freq))

    """
    Load and return an hypergraph from 'path'.
    """
    @classmethod
    def load(cls, path: str) -> Self:
        with open(path, 'rb') as f:
            nodes, num_edges = struct.unpack('<II', f.read(8))
            hyperedges = []
            for _ in range(num_edges):
                num_dsts = struct.unpack('<I', f.read(4))[0]
                src = struct.unpack('<I', f.read(4))[0]
                dsts = array('I')
                dsts.frombytes(f.read(4 * num_dsts))
                freq = struct.unpack('<f', f.read(4))[0]
                hyperedges.append(HyperEdge(src, tuple(dsts), freq))
        return cls(nodes, hyperedges, no_checks = True)

class Edge(HyperEdge):
    def __init__(self, source : int, destination : int, spike_frequency : float):
        self.nodes = (source, destination)
        self.spike_frequency = spike_frequency
        self._id = hash(self.nodes + (self.spike_frequency, random.random()))
    
    @classmethod
    def fromHyperEdge(cls, hyperedge : HyperEdge) -> Self:
        if len(hyperedge.nodes) > 2:
            raise Exception("Can't lower an HyperEdge with more than one destination to an Edge.")
        hyperedge.__class__ = cls
        return hyperedge
    
    def destination(self) -> int:
        return self.nodes[1]

"""
Directed graph to model a SNN.
Edges are weighted by the frequency with which they are traversed by a spike.
"""
class Graph(HyperGraph):
    edges : list[Edge]
    def __init__(self, nodes : int, edges : list[Union[Edge, HyperEdge, tuple[int, int]]], spike_frequencies : Optional[list[float]] = None):
        self.nodes = nodes
        if len(edges) == 0:
            self.hyperedges = []
        elif all(isinstance(he, Edge) for he in edges):
            self.hyperedges = edges # Be wary, there's no copy here!
        elif all(isinstance(he, HyperEdge) and len(he.nodes) == 2 for he in edges):
            self.hyperedges = list(map(Edge.fromHyperEdge, edges)) # Be wary, there's no copy here!
        elif all(isinstance(he, tuple) and len(he) == 2 for he in edges) and spike_frequencies and len(spike_frequencies) == len(edges):
            self.hyperedges = [Edge(he[0], he[1:], spike_frequencies[i]) for i, he in enumerate(edges)]
        else:
            raise Exception("""Failed to build graph. Edges shall be provided either as an empty list, a list of Edges, a list of HyperEdges, or a list of tuples of exactly two entries each.
                               In the latter case, spike_frequencies must also be a list of the same lenght, while the first entry in each tuple specifies the source node for that edge.""")
        if any(node < 0 or node >= nodes for he in self.hyperedges for node in he):
            raise Exception("Invalid edges, all node indices must be in the range [0, nodes).")
        self.edges = self.hyperedges
    
    """
    Convert the directed graph in NetworkX format.
    Directionality can be disable with the 'directed' flag.
    """
    def toNxGraph(self, directed : bool = True) -> Union[nx.DiGraph, nx.Graph]:
        g = nx.DiGraph() if directed else nx.Graph()
        g.add_nodes_from(range(self.nodes))
        for e in self.edges:
            src, dst = e.source(), e.destination()
            if g.has_edge(src, dst):
                g[src][dst]['spike_frequency'] += e.spike_frequency
            else:
                g.add_edge(src, dst, spike_frequency = e.spike_frequency)
        return g