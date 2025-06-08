from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Optional, Self, Union

import networkx as nx
import random

"""
Directed hyperedge connecting nodes inside an hypergraph.
It has a single source node and one or more destinations.
It is associated with a 'spike_frequency', that is, a weight.
"""
class HyperEdge(Iterable):
    # the first node is the source for the hyperedge
    nodes : tuple[int, ...]
    spike_frequency : float
    
    def __init__(self, source : int, destinations : tuple[int, ...], spike_frequency : float):
        # TODO: raise an exception if len(destionations) == 0
        self.nodes = (source,) + destinations
        self.spike_frequency = spike_frequency
    
    def source(self) -> int:
        return self.nodes[0]
    
    def destinations(self) -> tuple[int, ...]:
        return self.nodes[1:]
    
    def sameNodes(self, other : Self) -> bool:
        other_destinations = other.nodes[1:]
        return self.nodes[0] == other.nodes[0] and len(self.nodes) == len(other.nodes) and all(node in other_destinations for node in self.nodes[1:])
    
    def __iter__(self) -> Iterator[int]:
        return iter(self.nodes)
    
    def __eq__(self, other : Self) -> bool:
        return self.nodes == other.nodes and self.spike_frequency == other.spike_frequency
    
    def __str__(self) -> str:
        return self.nodes.__str__()[:-1] + f", sf = {self.spike_frequency:.1e})"
    
    """
    Number connections in the hyperedge, that is (|nodes| - 1).
    In other words, the number of plain edges the hyperedge corresponds to.
    """
    def connections(self) -> int:
        return len(self.nodes) - 1

"""
Link between two hypergraphs.
The owning hypergraph's nodes listed in 'nodes' are linked to the another 'target_hypergraph'.
Usually, links are born from torn hyperedges during partitioning.

Attributes:
- target_hypergraph: the other hypergraph to which the nodes are linked.
- linked_nodes: dictionary, keys are node indices, values are the weight of each node's link,
                represented by a spike frequency.
"""
class HyperGraphLink(Iterable):
    target_hypergraph : HyperGraph
    linked_nodes : dict[int, float]
    
    def __init__(self, target_hypergraph : HyperGraph, linked_nodes : dict[int, float]):
        self.target_hypergraph = target_hypergraph
        self.linked_nodes = linked_nodes
    
    def __iter__(self) -> Iterator[int]:
        return iter(self.linked_nodes)

"""
Directed hypergraph to model a SNN.
Each edge has one source neuron (node) and represents an axon going into many neurons (nodes) with a synapse for each.
Edges are weighted by the frequency with which they are traversed by a spike.

Attributes:
- nodes: number of verticies in the hypergraph, node indices must be in [0, nodes).
- hyperedges: list of hyperedges between node of the hypergraph.
- interhypergraph_links: extends the hypergraph by linking some of its nodes with other hypergraphs.
"""
class HyperGraph(Iterable):
    # each node is identified by an index in [0, nodes)
    nodes : int
    hyperedges : list[HyperEdge]
    
    interhypergraph_links : list[HyperGraphLink]
    
    # pointers to hyperedges leaving a node
    _outbound : list[list[HyperEdge]]
    # pointers to hyperedges entering a node
    _inbound : list[list[HyperEdge]]
    
    def __init__(self, nodes : int, hyperedges : list[Union[HyperEdge, tuple[int, ...]]], spike_frequencies : Optional[list[float]] = None, interhypergraph_links : list[HyperGraphLink] = None):
        self.nodes = nodes
        if len(hyperedges) == 0:
            self.hyperedges = []
        elif all(isinstance(he, HyperEdge) for he in hyperedges):
            self.hyperedges = hyperedges # Be wary, there's no copy here!
        elif all(isinstance(he, tuple) and len(he) >= 2 for he in hyperedges) and spike_frequencies and len(spike_frequencies) == len(hyperedges):
            self.hyperedges = [HyperEdge(he[0], he[1:], spike_frequencies[i]) for i, he in enumerate(hyperedges)]
        else:
            raise Exception("""Failed to build hypergraph. Hyperedges shall be provided either as an empty list, a list of HyperEdge instances, or a list of tuples of at least two entries each.
                               In the latter case, spike_frequencies must also be a list of the same lenght, while the first entry in each tuple specifies the source node for that hyperedge.""")
        if any(node < 0 or node >= nodes for he in self.hyperedges for node in he):
            raise Exception("Invalid hyperedges, all node indices must be in the range [0, nodes).")
        
        self.interhypergraph_links = [] if interhypergraph_links is None else interhypergraph_links
        
        # pay the overhead here to build faster access structures
        self._outbound = [[he for he in self.hyperedges if he.source() == node] for node in range(self.nodes)]
        self._inbound = [[he for he in self.hyperedges if node in he.destinations()] for node in range(self.nodes)]

    """
    Generate a random hypergraph with 'n' nodes, where each node is the source
    of exactly one hyperedge. The number of destinations for each hyperedge is
    sampled from a Gaussian distribution with mean 'c' and standard deviation 'd'.
    A seed can be provided for repeatability.
    
    Each destination node is selected uniformly at random (excluding the source node).
    Spike frequencies are sampled uniformly from 'spike_frequency_range'.
    """
    @classmethod
    def generate_random(cls, n: int, c: float, d: float, spike_frequency_range: tuple[float, float] = (0.1, 1.0), seed : Optional[int] = None) -> Self:
        hyperedges = []
        spike_frequencies = []

        random.seed(seed)
        for source in range(n):
            num_dest = max(0, min(n - 1, int(random.normalvariate(mu = c, sigma = d))))
            if num_dest > 0:
                candidates = [i for i in range(n) if i != source]
                destinations = tuple(random.sample(candidates, num_dest)) if num_dest > 0 else tuple()
                spike_freq = random.uniform(*spike_frequency_range)
                hyperedges.append((source,) + destinations)
                spike_frequencies.append(spike_freq)
        return cls(n, hyperedges, spike_frequencies)

    """
    Lower the HyperGraph into a directed Graph.
    If 'collaps_overlapping_edges' is True, resulting edges with the same source and destination will
    be merged and their spike frequencies added together.
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
    
    Args:
    - partitions: list of partitions indices, one per node in the graph, in order.
                  It assigns to each node its partition.
    - keep_self_cycles: if True, hyperedges entirely contained in a partition do not
                        disappear, instead are kept as a self-edge from the partition
                        to itself, preserving the total spike frequency.
    """
    def getPartitionsHypergraph(self, partitions : list[int], keep_self_cycles : bool = False) -> Self:
        if len(partitions) != self.nodes:
            raise Exception("Each node must be assigned to a partition.")
        new_nodes = len(set(partitions))
        if any(i not in partitions for i in range(0, new_nodes)):
            raise Exception("Partitions must be incrementally indexed from 0 onward.")
        
        new_hyperedges = []
        self_cycles = {}
        for he in self.hyperedges:
            affected_partitions = tuple(set([partitions[node] for node in he])) # deduplicate
            if len(affected_partitions) > 1:
                new_hyperedges.append(HyperEdge(affected_partitions[0], affected_partitions[1:], he.spike_frequency))
            elif keep_self_cycles:
                if affected_partitions[0] not in self_cycles:
                    self_cycles[affected_partitions[0]] = he.spike_frequency
                else:
                    self_cycles[affected_partitions[0]] += he.spike_frequency
        for partition, spike_frequency in self_cycles.items():
            new_hyperedges.append(HyperEdge(partition, (partition,), spike_frequency))
        return HyperGraph(new_nodes, new_hyperedges)
    
    """
    Returns an hypergraph for each partition of the present hypergraph.
    Each returned hypergraph presents a interhypergraph link to all other partitions,
    as to allow edges between partitions to be preserved and accounted for.
    
    NOTE: while 'getPartitionsHypergraph' returns an hypergraph where each node is a
          partition of the original, where we return one by one the hypergraphs born
          inside each partition.
    
    Args:
    Same as 'getPartitionsHypergraph'.
    """
    def getPartitionedHypergraphs(self, partitions : list[int]) -> list[Self]:
        if len(partitions) != self.nodes:
            raise Exception("Each node must be assigned to a partition.")
        part_idxs = set(partitions)
        if any(i not in partitions for i in range(0, len(part_idxs))):
            raise Exception("Partitions must be incrementally indexed from 0 onward.")
        
        hypergraphs : list[HyperGraph] = []
        nodes_map = [0 for _ in partitions] # map from old hypergraph node index (list idx) to new hypergraph node index (list content)
        for part_idx in part_idxs:
            node_idx_counter = 0
            for old_node, part in enumerate(partitions):
                if part == part_idx:
                    nodes_map[old_node] = node_idx_counter
                    node_idx_counter += 1
            hyperedges = [HyperEdge(nodes_map[he.source()], [nodes_map[d] for d in he.destinations() if partitions[d] == part_idx], he.spike_frequency) for he in self.hyperedges if partitions[he.source()] == part_idx and any(partitions[d] == part_idx for d in he.destinations())]
            hypergraphs.append(HyperGraph(len(nodes_map), hyperedges))
        
        for part_idx in part_idxs:
            for other_part_idx in part_idx:
                if part_idx != other_part_idx:
                    linked_nodes = {}
                    for he in self.hyperedges:
                        if any(partitions[n] == other_part_idx for n in he):
                            for n in he:
                                if partitions[n] == part_idx:
                                    if n in linked_nodes:
                                        linked_nodes[n] += he.spike_frequency
                                    else:
                                        linked_nodes[n] = he.spike_frequency
                    link = HyperGraphLink(hypergraphs[other_part_idx], link)
                    hypergraphs[part_idx].interhypergraph_links.append(link)
        
        return hypergraphs
    
    """
    Given a node's index, returns the list of hyperedges outbound from that node.
    Throws an exception if the node's index is invalid.
    """
    def getOutboundHyperedges(self, node : int) -> list[HyperEdge]:
        if node < 0 or node >= self.nodes:
            raise Exception("Invalid node.")
        return self._outbound[node]
    
    """
    Given a node's index, returns the list of hyperedges inbound for that node.
    Throws an exception if the node's index is invalid.
    """
    def getInboundHyperedges(self, node : int) -> list[HyperEdge]:
        if node < 0 or node >= self.nodes:
            raise Exception("Invalid node.")
        return self._inbound[node]
    
    """
    Given a node's index, returns the list of hyperedges touching that node.
    Throws an exception if the node's index is invalid.
    """
    def getTouchingHyperedges(self, node : int) -> list[HyperEdge]:
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
    """
    def addHyperedge(self, hyperedge : HyperEdge, add_missing_nodes : bool = False) -> None:
        if any(node < 0 for node in hyperedge):
            raise Exception("Negative node index in the provided hyperedge.")
        if not add_missing_nodes and any(node >= self.nodes for node in hyperedge):
            raise Exception("Out of bounds node index in the provided hyperedge.")
        else:
            self.nodes = max(self.nodes, max(hyperedge))
        self.hyperedges.append(hyperedge)
        self._outbound[hyperedge.source()].append(hyperedge)
        for node in hyperedge.destinations():
            self._inbound[node].append(hyperedge)
    
    """
    Any pair of HyperEdges that share the same source and destinations are fused in
    a single new HyperEdge having for spike frequency the sum of the originals'.
    """
    def squishHyperedges(self) -> None:
        to_delete = {} # keys will be deleted because they are identical to their value
        for he_idx1 in range(len(self.hyperedges)):
            for he_idx2 in range(he_idx1):
                if he_idx1 != he_idx2 and self.hyperedges[he_idx1].sameNodes(self.hyperedges[he_idx2]):
                    # at this point, he_idx1 should never be in to_delete yet
                    while he_idx2 in to_delete:
                        he_idx2 = to_delete[he_idx2]
                    to_delete[he_idx1] = he_idx2
                    self.hyperedges[he_idx2].spike_frequency += self.hyperedges[he_idx1].spike_frequency
                    break
        for he_idx in list(to_delete.keys())[::-1]:
            self.hyperedges.pop(he_idx)
    
    """
    Returns the total spike frequency on the hypergraph's connections.
    Each hyperedge is treated as (connected_nodes - 1) connections.
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

class Edge(HyperEdge):
    def __init__(self, source : int, destination : int, spike_frequency : float):
        self.nodes = (source, destination)
        self.spike_frequency = spike_frequency
    
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