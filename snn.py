from collections.abc import Iterable, Iterator
from typing import Any, Optional, Self

import random

class HyperEdge(Iterable):
    # the first node is the source for the hyperedge
    nodes : tuple[int, ...]
    spike_frequency : float
    
    def __init__(self, source : int, destinations : tuple[int, ...], spike_frequency : float):
        self.nodes = (source,) + destinations
        self.spike_frequency = spike_frequency
    
    def source(self) -> int:
        return self.nodes[0]
    
    def destinations(self) -> tuple[int, ...]:
        return self.nodes[1:]
    
    def __iter__(self) -> Iterator[int]:
        return iter(self.nodes)
    
    def __str__(self) -> str:
        return self.nodes.__str__()[:-1] + f", sf = {self.spike_frequency:.1e})"

    
    """
    Number connections in the hyperedge, that is (|nodes| - 1).
    In other words, the number of plain edges the hyperedge corresponds to.
    """
    def connections(self) -> int:
        return len(self.nodes) - 1

class HyperGraph:
    # each node is identified by an index in [0, nodes)
    nodes : int
    hyperedges : list[HyperEdge]
    
    def __init__(self, nodes : int, hyperedges : list[Any], spike_frequencies : Optional[list[float]] = None):
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
        #used_sources = set()
        #for he in self.hyperedges:
        #    if (src := he.source()) not in used_sources:
        #        used_sources.add(src)
        #    else:
        #        raise Exception("Each node can act a source for at most one hyperedge.")

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
    Returns the hypegraph that arises between partitions of the present hypergraph,
    in which nodes are the partitions and only hyperedges between partitions are kept.
    
    Args:
    - partitions: list of partitions indices, one per node in the graph, in order.
                  It assigns to each node its partition.
    """
    def getPartitionsHypergraph(self, partitions : list[int]) -> Self:
        if len(partitions) != self.nodes:
            raise Exception("Each node must be assigned to a partition.")
        new_nodes = len(set(partitions))
        if any(i not in partitions for i in range(0, new_nodes)):
            raise Exception("Partitions must be incrementally indexed from 0 onward.")
        
        new_hyperedges = []
        for he in self.hyperedges:
            affected_partitions = tuple(dict.fromkeys([partitions[node] for node in he])) # deduplicate
            if len(affected_partitions) > 1:
                new_hyperedges.append(HyperEdge(affected_partitions[0], affected_partitions[1:], he.spike_frequency))
        return HyperGraph(new_nodes, new_hyperedges)
    
    """
    Returns the total spike frequency on the hypergraph's connections.
    Each hyperedge is treated as (connected_nodes - 1) connections.
    """
    def totalSpikeFrequency(self) -> float:
        result = 0
        for he in self.hyperedges:
            result += he.spike_frequency*he.connections()
        return result
    
    def __str__(self) -> str:
        result = '['
        for he in self.hyperedges:
            result += he.__str__() + ', '
        return result[:-2] + ']'

class Graph(HyperGraph):
    def __init__(self, nodes : int, edges : list[Any], spike_frequencies : Optional[list[float]] = None):
        self.nodes = nodes
        if len(edges) == 0:
            self.hyperedges = []
        elif all(isinstance(he, HyperEdge) and len(he.nodes) == 2 for he in edges):
            self.hyperedges = edges # Be wary, there's no copy here!
        elif all(isinstance(he, tuple) and len(he) == 2 for he in edges) and spike_frequencies and len(spike_frequencies) == len(edges):
            self.hyperedges = [HyperEdge(he[0], he[1:], spike_frequencies[i]) for i, he in enumerate(edges)]
        else:
            raise Exception("""Failed to build hypergraph. Edges shall be provided either as an empty list, a list of HyperEdge instances, or a list of tuples of exactly two entries each.
                               In the latter case, spike_frequencies must also be a list of the same lenght, while the first entry in each tuple specifies the source node for that edge.""")
        if any(node < 0 or node >= nodes for he in self.hyperedges for node in he):
            raise Exception("Invalid edges, all node indices must be in the range [0, nodes).")
        used_sources = set()
        for he in self.hyperedges:
            if (src := he.source()) not in used_sources:
                used_sources.add(src)
            else:
                raise Exception("Each node can act a source for at most one edge.")