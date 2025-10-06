from typing import Union

from collections import Counter
import numpy as np

from partitioner import partitionSequential
from datastructures import Coord2D
from utils import *
from snn import *

# Partial credit to: "Mapping Very Large Scale Spiking Neuron Network to Neuromorphic Hardware" by Ouwen Jin et al, ASPLOS 2023

"""
Neuromorphic hardware model class.
Given a placed SNN, checks the placement's validity and returns its performance metrics.
"""
class HardwareModel:
    # CONSTRAINTS:
    # how many neurons a core can store and process.
    neurons_per_core : int
    # synapses are shared across all neurons in a core, each neuron can have a different
    # weight for a synapse, but the incoming axon is the same for all neurons.
    # => this is the number of "max. inbound axons per core" of "max. synapses per neuron".
    synapses_per_core : int
    cores_per_chip_x : int
    cores_per_chip_y : int
    chips_per_system_x : int
    chips_per_system_y: int
    # TODO: two hardware constraints are missing, that Loihi has:
    # 1) maximum number of outbound hyperedge branches per core (counting each hyperedge once per destination core)
    # 2) maximum number of true synapses per core (in TrueNorth, this is just neurons*axons, so no need, but in Loihi it is less, 2**14)
    
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
        assert neurons_per_core > 0 and synapses_per_core > 0 and cores_per_chip_x > 0 and cores_per_chip_y > 0 and chips_per_system_x > 0 and chips_per_system_y > 0, "All hardware specifications must be > 0."
        assert energy_per_routing >= 0 and energy_per_wire >= 0 and latency_per_routing >= 0 and latency_per_wire >= 0, "All hardware costs must be >= 0."
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
    
    def coresPerChipCount(self) -> int:
        return self.cores_per_chip_x*self.cores_per_chip_y
    
    def chipsCount(self) -> int:
        return self.chips_per_system_x*self.chips_per_system_y
    
    def coresAlongX(self) -> int:
        return self.cores_per_chip_x*self.chips_per_system_x
    
    def coresAlongY(self) -> int:
        return self.cores_per_chip_y*self.chips_per_system_y
    
    """
    Empirically verifies if a given SNN could theoretically fit on the hardware w.r.t.
    two constraints: the space available for nodes, and that for edges.
    
    Can give false negatives. Never gives false positives.
    """
    def checkSnnFit(self, snn : HyperGraph, already_partitioned : bool = False, verbose : bool = False) -> bool:
        if already_partitioned:
            if snn.nodes > self.coresCount():
                if verbose:
                    print("SNN CAN'T FIT ON THE HW: more neuron clusters than the HW cores")
                return False
            if any(len(snn.getInboundHyperedges(n)) > self.synapses_per_core for n in range(snn.nodes)):
                if verbose:
                    print("SNN CAN'T FIT ON THE HW: more inbound synapses on a neuron cluster than the HW can handle")
                return False
            return True
        if snn.nodes > self.coresCount()*self.neurons_per_core:
            if verbose:
                print("SNN CAN'T FIT ON THE HW: more neurons than the HW can house")
            return False # more neurons than the HW can house
        if any(len(snn.getInboundHyperedges(n)) > self.synapses_per_core for n in range(snn.nodes)):
            if verbose:
                print("SNN CAN'T FIT ON THE HW: more inbound synapses on a single neuron than the HW can handle")
            return False # more inbound synapses on a single neuron than the HW can handle
        #if not can_distribute_sets_heuristic([{he.source() for he in snn.getInboundHyperedges(n)} for n in range(snn.nodes)], self.coresCount(), self.synapses_per_core, self.neurons_per_core):
        #    print("no valid way to split neurons (and their synapses) among cores")
        #    return False # no valid way to split neurons (and their synapses) among cores
        try:
            partitionSequential(snn, self.neurons_per_core, self.synapses_per_core, self.coresCount())
        except:
            if verbose:
                print("SNN WON'T LIKELY FIT ON THE HW: no valid way to split neurons (and their synapses) among cores")
            return False # no valid way to split neurons (and their synapses) among cores
        return True
    
    """
    Verifies that the SNN's hypergraph partition is valid w.r.t. hardware constraints.
    
    Args:
    - partitions: list of partitions indices, one per neuron, in order.
                  It assigns to each node its partition.
    """
    def checkPartitionValidity(self, snn: HyperGraph, partitions : list[int], verbose : bool = False) -> bool:
        if len(partitions) != snn.nodes:
            raise Exception("Each neuron must be assigned to a partition.")
        partitions_counter = Counter(partitions)
        partitions_count = len(partitions_counter)
        if partitions_count > self.coresCount():
            if verbose:
                print("INVALID PARTITIONING: more partitions than cores")
            return False # more partitions than cores
        neurons_per_partition = partitions_counter.values()
        if any(npc > self.neurons_per_core for npc in neurons_per_partition):
            if verbose:
                print("INVALID PARTITIONING: more neurons per partition than a core can store")
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
            if verbose:
                print("INVALID PARTITIONING: more inbound synapses per partition than a core can handle", synapses_per_partition)
            return False # more inbound synapses per partition than a core can handle
        return True
    
    """
    Checks if a placement for a partitioned SNN is valid.
    Let the placement be a pair of X and Y coordinates for each node in the hypergraph.
    The coordinates assigne the specific partition to a core in on the hardware.
    """
    def checkPlacementValidity(self, part_snn : HyperGraph, placement : list[Coord2D], verbose : bool = False) -> bool:
        if len(placement) != part_snn.nodes:
            raise Exception("Each partition must be assigned to a core.")
        seen_cores = set()
        for core in placement:
            if core[0] < 0 or core[0] >= self.coresAlongX() or core[1] < 0 or core[1] >= self.coresAlongY():
                if verbose:
                    print("INVALID PLACEMENT: a core's coordinates are out of the hardware's range")
                return False
            if core not in seen_cores:
                seen_cores.add(core)
            else:
                if verbose:
                    print("INVALID PLACEMENT: a core is used more than once")
                return False
        return True
    
    """
    Given a partitioning for a SNN, quantify its usage of synaptic reuse.
    The metric is defined as the total number of individual inbound connections
    (synapses) per partition over the number of distinct inbound hedges (axons).
    """
    def synapticReuse(self, snn : HyperGraph, partitions : list[int]) -> dict[str, float]:
        partitions_count = max(partitions) + 1
        synapses_count_per_partition = np.zeros(partitions_count, dtype = np.int32)
        axons_count_per_partition = np.zeros(partitions_count, dtype = np.int32)
        for he in snn.hyperedges:
            already_seen = set()
            for neuron in he.destinations():
                partition = partitions[neuron]
                if partition not in already_seen:
                    axons_count_per_partition[partition] += 1
                    already_seen.add(partition)
                synapses_count_per_partition[partition] += 1
        reuse = np.divide(
            synapses_count_per_partition,
            axons_count_per_partition,
            out = np.zeros_like(synapses_count_per_partition, dtype = np.float64),
            where = axons_count_per_partition != 0
        )
        return {"ar_mean": reuse.mean(), "geo_mean": np.exp(np.log(reuse[reuse > 0]).sum()/partitions_count)}
    
    #"""
    #Like 'synapticReuse', but each connection is weighted by its spike frequency.
    #"""
    #def synapticReuseWeighted(self, snn : HyperGraph, partitions : list[int]) -> float:
    #    partitions_count = max(partitions) + 1
    #    synapses_count_per_partition = np.zeros(partitions_count, dtype = np.float32)
    #    axons_count_per_partition = np.zeros(partitions_count, dtype = np.float32)
    #    for he in snn.hyperedges:
    #        already_seen = set()
    #        for neuron in he.destinations():
    #            partition = partitions[neuron]
    #            if partition not in already_seen:
    #                axons_count_per_partition[partition] += he.spike_frequency
    #                already_seen.add(partition)
    #            synapses_count_per_partition[partition] += he.spike_frequency
    #    reuse = np.divide(
    #        synapses_count_per_partition,
    #        axons_count_per_partition,
    #        out = np.zeros_like(synapses_count_per_partition, dtype = np.float32),
    #        where = axons_count_per_partition != 0
    #    )
    #    # WARNING: the mean should not be "divide by instances", but should be "divide by total weight", do it manually!
    #    return (reuse.mean(), np.exp(np.mean(np.log(reuse))), reuse.max(), reuse.min(), reuse)
    
    """
    Given a placement for a partitioned SNN, quantify its connections locality.
    The metric is defined as the average number of core coordinates enclosed by
    the convex hull defined around the cores connected by each hedge.
    """
    def connectionsLocality(self, part_snn : HyperGraph, placement : list[Coord2D]) -> dict[str, float]:
        ar_mean, geo_mean, ar_mean_weighted, geo_mean_weighted = 0.0, 0.0, 0.0, 0.0
        weights_sum = 0.0
        for he in part_snn:
            traversed_cores = intersection_with_convex_hull([placement[p] for p in he], self.coresAlongX(), self.coresAlongY())
            ar_mean += traversed_cores
            geo_mean += math.log(traversed_cores)
            ar_mean_weighted += traversed_cores*he.spike_frequency
            geo_mean_weighted += math.log(traversed_cores)*he.spike_frequency
            weights_sum += he.spike_frequency
        ar_mean /= len(part_snn.hyperedges)
        geo_mean = math.exp(geo_mean/len(part_snn.hyperedges))
        ar_mean_weighted /= weights_sum
        geo_mean_weighted = math.exp(geo_mean_weighted/weights_sum)
        return {"ar_mean": ar_mean, "geo_mean": geo_mean, "ar_mean_weighted": ar_mean_weighted, "geo_mean_weighted": geo_mean_weighted}
    
    """
    Given a placement for a partitioned SNN, estimates its energy consumption.
    Let the placement be a pair of X and Y coordinates for each node in the hypergraph.
    """
    def placementEnergyConsumption(self, part_snn : HyperGraph, placement : list[Coord2D]) -> float:
        result = 0.0
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
    def placementAverageLatency(self, part_snn : HyperGraph, placement : list[Coord2D]) -> float:
        result = 0.0
        tot_spike_frequency = 0.0
        for he in part_snn.hyperedges:
            src = he.source()
            tot_spike_frequency += he.spike_frequency*he.connections()
            for dst in he.destinations():
                manhattan_distance = manhattan(placement[src], placement[dst])
                result += he.spike_frequency*((manhattan_distance + 1)*self.latency_per_routing + manhattan_distance*self.latency_per_wire)
        return result / tot_spike_frequency if tot_spike_frequency > 0 else 0
    
    """
    Given a placement for a partitioned SNN, estimates its maximum latency.
    Let the placement be a pair of X and Y coordinates for each node in the hypergraph.
    """
    def placementMaximumLatency(self, part_snn : HyperGraph, placement : list[Coord2D]) -> float:
        result = 0.0
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
    def placementAverageCongestion(self, part_snn : HyperGraph, placement : list[Coord2D]) -> float:
        result = 0.0
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
    def placementMaximumCongestion(self, part_snn : HyperGraph, placement : list[Coord2D]) -> float:
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
    
    """
    Summarizes all model metrics for a given placement.
    """
    def getAllMetrics(self, part_snn : HyperGraph, placement : list[Coord2D]) -> dict[str, float]:
        valid = self.checkPlacementValidity(part_snn, placement)
        if valid:
            return {
                'valid': valid,
                'energy': self.placementEnergyConsumption(part_snn, placement),
                'avg_latency': self.placementAverageLatency(part_snn, placement),
                'max_latency': self.placementMaximumLatency(part_snn, placement),
                'avg_congestion': self.placementAverageCongestion(part_snn, placement),
                'max_congestion': self.placementMaximumCongestion(part_snn, placement),
                'connections_locality': self.connectionsLocality(part_snn, placement)
            }
        else:
            return {'valid': valid, 'energy': None, 'avg_latency': None, 'max_latency': None, 'avg_congestion': None, 'max_congestion': None, 'connections_locality': None}
    
    """
    Returns a compound cost metric that is the product of energy, latency, and congestion, all of which shall be minimized.
    """
    def getCompoundMetric(self, part_snn : HyperGraph, placement : list[Coord2D], include_congestion : bool = False) -> float:
        energy = 0
        latency = 0
        tot_spike_frequency = 0
        congestion = 0
        for he in part_snn.hyperedges:
            tot_spike_frequency += he.spike_frequency*he.connections()
            src_core = placement[he.source()]
            for dst in he.destinations():
                dst_core = placement[dst]
                manhattan_distance = manhattan(src_core, dst_core)
                energy += he.spike_frequency*((manhattan_distance + 1)*self.energy_per_routing + manhattan_distance*self.energy_per_wire)
                latency += he.spike_frequency*((manhattan_distance + 1)*self.latency_per_routing + manhattan_distance*self.latency_per_wire)
                if include_congestion:
                    congestion += he.spike_frequency*sum(map(sum, self.expectedSpikeTransitProbability(src_core[0], src_core[1], dst_core[0], dst_core[1])))
        return energy * (latency / tot_spike_frequency if tot_spike_frequency > 0 else 0) * (congestion / self.coresCount() if include_congestion else 1)
    
    """
    Returns the "force", aka the reduction in the hardware's potential energy (defined as a proxy for the hardware's energy
    and latency), that would derive from moving the 'placement' for the provided 'node' in any of 'directions'.
    One force for each direction is returned, even if the destination is outside the hardware's bounds.
    An invalid node index silently results in zero force in all directions.
    """
    def getForces(self, part_snn : HyperGraph, placement : Union[list[Coord2D], dict[int, Coord2D]], node : int, directions : tuple[Coord2D, ...] = (Coord2D(1, 0), Coord2D(0, 1), Coord2D(-1, 0), Coord2D(0, -1)), potential_func : Callable[[Coord2D], float] = lambda c : max(abs(c), 1)) -> dict[Coord2D, float]:
        # ISSUE: the original version used as 'potential_func' just 'abs', without 'max(1, ...)', but that meant that you ignored the potential
        # energy caused by the node already occupying 'node_placement + d', and that is a problem if such a node is heavily connected!
        if node < 0 or node >= part_snn.nodes:
            return {d : 0.0 for d in directions}
        node_placement = placement[node]
        node_placement_plus_d = {d : node_placement + d for d in directions}
        base_potential = 0.0
        alt_potentials = {d : 0.0 for d in directions}
        for he in part_snn.getInboundHyperedges(node):
            src_placement = placement[he.source()]
            base_potential += potential_func(node_placement - src_placement)*he.spike_frequency
            for d in directions:
                alt_potentials[d] += potential_func(node_placement_plus_d[d] - src_placement)*he.spike_frequency
        # ISSUE: the original version depended only on inbound, not outbound connections (forces were not symmetric)
        for he in part_snn.getOutboundHyperedges(node):
            for dst in he.destinations():
                dst_placement = placement[dst]
                base_potential += potential_func(node_placement - dst_placement)*he.spike_frequency
                for d in directions:
                    alt_potentials[d] += potential_func(node_placement_plus_d[d] - dst_placement)*he.spike_frequency
        return {d : base_potential - alt_potentials[d] for d in directions}


# Library of existing neuromorphic systems:

# TODO: replace "synapses_per_core" with "axons_per_core", and create the separate concept of "synapses_per_core"!!

# Source: table 2 in "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning", referring to data at 0.75V.
loihi = HardwareModel(
    neurons_per_core = 1024,
    synapses_per_core = 4096,
    cores_per_chip_x = 16,
    cores_per_chip_y = 8,
    chips_per_system_x = 1,
    chips_per_system_y = 1,
    energy_per_routing = 1.7,
    energy_per_wire = 3.5,
    latency_per_routing = 2.1,
    latency_per_wire = 5.3
)
loihi_large = HardwareModel(
    neurons_per_core = 1024,
    synapses_per_core = 4096,
    cores_per_chip_x = 64,
    cores_per_chip_y = 64,
    chips_per_system_x = 1,
    chips_per_system_y = 1,
    energy_per_routing = 1.7,
    energy_per_wire = 3.5,
    latency_per_routing = 2.1,
    latency_per_wire = 5.3
)
# Test configuration to see if the mapper can handle well synaptic reuse.
loihi_reuse_test = HardwareModel(
    neurons_per_core = 1024*1024,
    synapses_per_core = 4096,
    cores_per_chip_x = 64,
    cores_per_chip_y = 64,
    chips_per_system_x = 1,
    chips_per_system_y = 1,
    energy_per_routing = 1.7,
    energy_per_wire = 3.5,
    latency_per_routing = 2.1,
    latency_per_wire = 5.3
)

# Source: tables 2 and 3 in "Mapping Very Large Scale Spiking Neuron Network to Neuromorphic Hardware".
# => It is essentially 4x w.r.t. base Loihi.
loihi_jin_84 = HardwareModel(
    neurons_per_core = 4096,
    synapses_per_core = 1024*64,
    cores_per_chip_x = 84,
    cores_per_chip_y = 84,
    chips_per_system_x = 1,
    chips_per_system_y = 1,
    energy_per_routing = 1.0,
    energy_per_wire = 0.1,
    latency_per_routing = 1.0,
    latency_per_wire = 0.01
)
loihi_jin_1024 = HardwareModel(
    neurons_per_core = 4096,
    synapses_per_core = 1024*64,
    cores_per_chip_x = 1024,
    cores_per_chip_y = 1024,
    chips_per_system_x = 1,
    chips_per_system_y = 1,
    energy_per_routing = 1.0,
    energy_per_wire = 0.1,
    latency_per_routing = 1.0,
    latency_per_wire = 0.01
)

# Source: section V.A in "TrueNorth: Design and Tool Flow of a 65 mW 1 Million Neuron Programmable Neurosynaptic Chip".
truenorth = HardwareModel(
    neurons_per_core = 256,
    synapses_per_core = 256,
    cores_per_chip_x = 64,
    cores_per_chip_y = 64,
    chips_per_system_x = 1,
    chips_per_system_y = 1,
    energy_per_routing = 1.7, # unknown (this is from Loihi)
    energy_per_wire = 3.5, # unknown (this is from Loihi)
    latency_per_routing = 2.1, # unknown (this is from Loihi)
    latency_per_wire = 5.3 # unknown (this is from Loihi)
)