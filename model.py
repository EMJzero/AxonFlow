from collections import Counter

from utils import *
from snn import *

# Partial credit to: "Mapping Very Large Scale Spiking Neuron Network to Neuromorphic Hardware" by Ouwen Jin et al, ASPLOS 2023

class HardwareModel:
    # CONSTRAINTS:
    neurons_per_core : int
    # synapses are shared across all neurons in a core, each neuron can have a different
    # weight for a synapse, but the incoming axon is the same for all neurons.
    synapses_per_core : int
    cores_per_chip_x : int
    cores_per_chip_y : int
    chips_per_system_x : int
    chips_per_system_y: int
    
    # COSTS:
    # energy required for a core's router to route a spike [pJ]
    energy_per_routing : float
    # energy requried for a spike to transit over a write between two cores [pJ]
    energy_per_wire : float
    # clock cycles required for a core's router to route a spike [cc or ns]
    latency_per_routing : float
    # clock cycles requried for a spike to transit over a write between two cores [cc or ns]
    latency_per_wire : float
    
    def __init__(self, neurons_per_core : int,
                 synapses_per_core : int,
                 cores_per_chip_x : int,
                 cores_per_chip_y : int,
                 chips_per_system_x : int,
                 chips_per_system_y : int,
                 energy_per_routing : float,
                 energy_per_wire : float,
                 latency_per_routing : float,
                 latency_per_wire : float):
        assert chips_per_system_x == 1 and chips_per_system_y == 1, "Functionality not yet implemented, ensure that 'chips_per_system_x' and 'chips_per_system_y' are 1."
        self.neurons_per_core = neurons_per_core
        self.synapses_per_core = synapses_per_core
        self.cores_per_chip_x = cores_per_chip_x
        self.cores_per_chip_y = cores_per_chip_y
        self.chips_per_system_x = chips_per_system_x
        self.chips_per_system_y = chips_per_system_y
        self.energy_per_routing = energy_per_routing
        self.energy_per_wire = energy_per_wire
        self.latency_per_routing = latency_per_routing
        self.latency_per_wire = latency_per_wire
    
    def coresCount(self) -> int:
        return (self.cores_per_chip_x*self.cores_per_chip_y)*self.chips_per_system_x*self.chips_per_system_y
    
    def coresAlongX(self) -> int:
        return self.cores_per_chip_x*self.chips_per_system_x
    
    def coresAlongY(self) -> int:
        return self.cores_per_chip_y*self.chips_per_system_y
    
    """
    verifies that the SNN's hypergraph partition is valid w.r.t. hardware constraints.
    
    Args:
    - partitions: list of partitions indices, one per neuron, in order.
                  It assigns to each node its partition.
    """
    def checkPartitionValidity(self, snn: HyperGraph, partitions : list[int]) -> bool:
        if len(partitions) != snn.nodes:
            raise Exception("Each neuron must be assigned to a partition.")
        partitions_counter = Counter(partitions)
        partitions_count = len(partitions_counter)
        if partitions_count > self.coresCount():
            return False # more partitions than cores
        neurons_per_partition = partitions_counter.values()
        if any(npc > self.neurons_per_core for npc in neurons_per_partition):
            return False # more neurons per partition than a core can store
        
        synapses_per_partition = [0 for _ in range(partitions_count)]
        if any(i not in partitions for i in range(0, partitions_count)):
            raise Exception("Partitions must be incrementally indexed from 0 onward.")
        for he in snn.hyperedges:
            already_seen = set() # each hyperedge enters only once in each partition, it is then multicasted as needed inside it
            for neuron in he.destinations():
                partition = partitions[neuron]
                if partition not in already_seen:
                    synapses_per_partition[partition] += 1
                    already_seen.add(partition)
        if any(spp > self.synapses_per_core for spp in synapses_per_partition):
            return False # more inbound synapses per partition than a core can handle
        return True
    
    """
    Checks if a placement for a partitioned SNN is valid.
    Let the placement be a pair of X and Y coordinates for each node in the hypergraph.
    The coordinates assigne the specific partition to a core in on the hardware.
    """
    def checkPlacementValidity(self, part_snn : HyperGraph, placement : list[tuple[int, int]]) -> bool:
        if len(placement) != part_snn.nodes:
            raise Exception("Each partition must be assigned to a core.")
        seen_cores = set()
        for core in placement:
            if core[0] < 0 or core[0] >= self.coresAlongX() or core[1] < 0 or core[1] >= self.coresAlongY():
                return False # a core's coordinates are out of the hardware's range
            if core not in seen_cores:
                seen_cores.add(core)
            else:
                return False # a core is used more than once
        return True
    
    """
    Given a placement for a partitioned SNN, estimates its energy consumption.
    Let the placement be a pair of X and Y coordinates for each node in the hypergraph.
    """
    def placementEnergyConsumption(self, part_snn : HyperGraph, placement : list[tuple[int, int]]) -> float:
        result = 0
        for he in part_snn.hyperedges:
            src = he.source()
            for dst in he.destinations():
                manhattan_distance = manhattan(placement[src], placement[dst])
                result += he.spike_frequency*((manhattan_distance + 1)*self.energy_per_routing + manhattan_distance*self.energy_per_wire)
        return result
    
    """
    Given a placement for a partitioned SNN, estimates its average latency.
    Let the placement be a pair of X and Y coordinates for each node in the hypergraph.
    """
    def placementAverageLatency(self, part_snn : HyperGraph, placement : list[tuple[int, int]]) -> float:
        result = 0
        tot_spike_frequency = 0
        for he in part_snn.hyperedges:
            src = he.source()
            tot_spike_frequency += he.spike_frequency*he.connections()
            for dst in he.destinations():
                manhattan_distance = manhattan(placement[src], placement[dst])
                result += he.spike_frequency*((manhattan_distance + 1)*self.latency_per_routing + manhattan_distance*self.latency_per_wire)
        return result / tot_spike_frequency
    
    """
    Given a placement for a partitioned SNN, estimates its maximum latency.
    Let the placement be a pair of X and Y coordinates for each node in the hypergraph.
    """
    def placementMaximumLatency(self, part_snn : HyperGraph, placement : list[tuple[int, int]]) -> float:
        result = 0
        for he in part_snn.hyperedges:
            src = he.source()
            for dst in he.destinations():
                manhattan_distance = manhattan(placement[src], placement[dst])
                result = max((manhattan_distance + 1)*self.latency_per_routing + manhattan_distance*self.latency_per_wire, result)
        return result
    
    """
    Given a placement for a partitioned SNN, estimates its average congestion.
    Let the placement be a pair of X and Y coordinates for each node in the hypergraph.
    """
    def placementAverageCongestion(self, part_snn : HyperGraph, placement : list[tuple[int, int]]) -> float:
        result = 0
        # Note: calculation refactored as the "average probability of spike transit".
        for he in part_snn.hyperedges:
            src_core = placement[he.source()]
            for dst in he.destinations():
                dst_core = placement[dst]
                result += he.spike_frequency*sum(map(sum, self.expectedSpikeTransitProbability(src_core[0], src_core[1], dst_core[0], dst_core[1])))
        return result / self.coresCount()
    
    """
    Given a placement for a partitioned SNN, estimates its maximum congestion.
    Let the placement be a pair of X and Y coordinates for each node in the hypergraph.
    """
    def placementMaximumCongestion(self, part_snn : HyperGraph, placement : list[tuple[int, int]]) -> float:
        congestion_matrix = [[0 for _ in range(self.coresAlongY())] for _ in range(self.coresAlongX())]
        for he in part_snn.hyperedges:
            src_core = placement[he.source()]
            for dst in he.destinations():
                dst_core = placement[dst]
                transit_prob_matrix = self.expectedSpikeTransitProbability(src_core[0], src_core[1], dst_core[0], dst_core[1])
                x_base = min(dst_core[0], src_core[0])
                y_base = min(dst_core[1], src_core[1])
                for x in range(abs(dst_core[0] - src_core[0])):
                    for y in range(abs(dst_core[1] - src_core[1])):
                        congestion_matrix[x_base + x][y_base + y] += he.spike_frequency*transit_prob_matrix[x][y]
        # TODO: upgrade this to also return the average congestion, since we are at it...
        return max(map(max, congestion_matrix))
    
    """
    Computes the matrix of probabilities of a spike to pass through core a '(x, y)' when
    going from cores '(x_src, y_src)' to core '(x_dst, y_dst)'.
    The element '(x, y)' in the matrix matches to '(min(x_src, x_dst) + x, min(y_src, y_dst) + y)' on the original lattice.
    """
    # TODO: precompute or cache those matrices by size (transpose+rotate and return to account for direction)...
    def expectedSpikeTransitProbability(self, x_src : int, y_src : int, x_dst : int, y_dst : int) -> list[list[float]]:
        matrix = [[0 for _ in range(abs(y_dst - y_src) + 1)] for _ in range(abs(x_dst - x_src) + 1)]
        matrix[0][0] = 1
        x_src, x_dst = x_src - (m := min(x_src, x_dst)), x_dst - m
        y_src, y_dst = y_src - (m := min(y_src, y_dst)), y_dst - m
        x_sign = 1 if x_src == 0 else -1
        y_sign = 1 if y_src == 0 else -1
        # TODO: could simplify 'iter_major_diagonals' to take signs (directions), width, and height directly
        for x, y in iter_major_diagonals(x_src, y_src, x_dst, y_dst, end_included = True):
            if x == x_dst and y != y_dst:
                matrix[x][y + y_sign] += matrix[x][y]
            elif x != x_dst and y == y_dst:
                matrix[x + x_sign][y] += matrix[x][y]
            elif x != x_dst and y != y_dst:
                matrix[x + x_sign][y] += matrix[x][y]/2
                matrix[x][y + y_sign] += matrix[x][y]/2
        return matrix


# Library of existing neuromorphic systems:

# Source: table 2 in "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning", referring to data at 0.75V.
loihi = HardwareModel(
    neurons_per_core = -256,
    synapses_per_core = min(2**14, 4096),
    cores_per_chip_x = 16,
    cores_per_chip_y = 8,
    chips_per_system_x = 1,
    chips_per_system_y = 1,
    energy_per_routing = 1.7,
    energy_per_wire = 3.5,
    latency_per_routing = 2.1,
    latency_per_wire = 5.3
)