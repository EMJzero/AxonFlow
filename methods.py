from typing import Optional
import json
import os

from graph_utils import *
from partitioner import *
from placer import *
from snn import *

# EXPERIMENTS:

class Result:
    name : str
    # HYPERGRAPH
    graph_nodes : int
    graph_edges : int
    graph_cost : float
    # HARDWARE
    hw_npc : int
    hw_spc : int
    hw_cpc : int
    # PARTITIONING
    part_valid : bool
    part_cost : float # lower is better
    part_count : int
    part_synaptic_reuse: float
    # INITIAL PLACEMENT
    init_valid : bool
    init_energy : float
    init_avg_lat : float
    init_max_lat : float
    init_avg_cong : float
    init_max_cong : float
    init_connections_locality: float
    # FINAL PLACEMENT
    plac_valid : bool
    plac_energy : float
    plac_avg_lat : float
    plac_max_lat : float
    plac_avg_cong : float
    plac_max_cong : float
    plac_connections_locality: float
    # EXTRA
    note : str
    # TIME
    part_time : int
    time : int
    approx_time : int

    def __init__(self, name : str):
        self.name = name

    def setGraph(self, nodes : int, edges : int, total_spike_frequency : float) -> None:
        self.graph_nodes = nodes
        self.graph_edges = edges
        self.graph_cost = total_spike_frequency

    def setHw(self, npc : int, spc : int, cpc : int) -> None:
        self.hw_npc = npc
        self.hw_spc = spc
        self.hw_cpc = cpc

    def setPart(self, valid : bool, cost : float, count : int, part_synaptic_reuse : float) -> None:
        self.part_valid = valid
        self.part_cost = cost
        self.part_count = count
        self.part_synaptic_reuse = part_synaptic_reuse

    def setInitPlac(self, valid : bool, energy : float, avg_latency : float, max_latency : float, avg_congestion : float, max_congestion : float, connections_locality : float) -> None:
        self.init_valid = valid
        self.init_energy = energy
        self.init_avg_lat = avg_latency
        self.init_max_lat = max_latency
        self.init_avg_cong = avg_congestion
        self.init_max_cong = max_congestion
        self.init_connections_locality = connections_locality

    def setPlac(self, valid : bool, energy : float, avg_latency : float, max_latency : float, avg_congestion : float, max_congestion : float, connections_locality : float) -> None:
        self.plac_valid = valid
        self.plac_energy = energy
        self.plac_avg_lat = avg_latency
        self.plac_max_lat = max_latency
        self.plac_avg_cong = avg_congestion
        self.plac_max_cong = max_congestion
        self.plac_connections_locality = connections_locality

    def setNote(self, note : str) -> None:
        self.note = note

    def startTime(self) -> None:
        self._start_time = time.time()

    def partTime(self) -> None:
        self.part_time = time.time() - self._start_time

    def endTime(self) -> None:
        self.time = time.time() - self._start_time
        del self._start_time

    def setApproxTime(self, time : int) -> None:
        self.approx_time = time

    def __str__(self) -> str:
        return str(self.__dict__)

    """
    Given a JSON file (if it does not exist, it is created), append
    to its list of entries the present class as a new dictionary.
    """
    # TODO: it is costly to read the file every time...
    def toFile(self, filename : str) -> None:
        if os.path.exists(filename):
            with open(filename, 'r', encoding = 'utf-8') as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        raise ValueError("JSON file does not contain a list of dictionaries.")
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(self.__dict__)
        with open(filename, 'w', encoding = 'utf-8') as f:
            json.dump(data, f, indent = 4)


# FULL METHODS:

# NOTE: partitioning is repeated each time to have multiple execution time measurements...

def run_sequential_topo_hilbert_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    if not isTopologicallySorted(hg):
        hg = feedForwardOrder(hg)
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    if isTopologicallySorted(part_snn):
        topological_order = part_snn
    else:
        topological_order, _ = topologicalOrder(part_snn, break_cycles = True)
    plac = hilbertPlacement(topological_order.nodes, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(part_snn, plac))
    plac = forceDirectedRefinement(part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_sequential_hilbert_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    if not isTopologicallySorted(hg):
        hg = feedForwardOrder(hg)
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = hilbertPlacement(ordered_part_snn.nodes, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    plac = forceDirectedRefinement(ordered_part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_sequential_hilbert_ps(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    if not isTopologicallySorted(hg):
        hg = feedForwardOrder(hg)
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = hilbertPlacement(ordered_part_snn.nodes, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    plac = particleSwarmPlacement(ordered_part_snn, hw, num_iterations = 20, initial_layout = plac)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_sequential_spectral_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    if not isTopologicallySorted(hg):
        hg = feedForwardOrder(hg)
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    plac = spectralPlacementScipy(part_snn, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(part_snn, plac))
    plac = forceDirectedRefinement(part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_sequential_spectral_ps(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    if not isTopologicallySorted(hg):
        hg = feedForwardOrder(hg)
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    plac = spectralPlacementScipy(part_snn, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(part_snn, plac))
    plac = particleSwarmPlacement(part_snn, hw, num_iterations = 20, initial_layout = plac)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_sequential_truenorth(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    if not isTopologicallySorted(hg):
        hg = feedForwardOrder(hg)
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = trueNorthPlacement(ordered_part_snn, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_hehiding_hilbert_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionHyperedgeHidingOnlyInbound(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = hilbertPlacement(ordered_part_snn.nodes, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    plac = forceDirectedRefinement(ordered_part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_hehiding_hilbert_ps(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionHyperedgeHidingOnlyInbound(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = hilbertPlacement(ordered_part_snn.nodes, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    plac = particleSwarmPlacement(ordered_part_snn, hw, num_iterations = 20, initial_layout = plac)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_hehiding_spectral_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionHyperedgeHidingOnlyInbound(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    plac = spectralPlacementScipy(part_snn, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(part_snn, plac))
    plac = forceDirectedRefinement(part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_hehiding_spectral_ps(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionHyperedgeHidingOnlyInbound(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    plac = spectralPlacementScipy(part_snn, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(part_snn, plac))
    plac = particleSwarmPlacement(part_snn, hw, num_iterations = 20, initial_layout = plac)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_hehiding_truenorth(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionHyperedgeHidingOnlyInbound(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = trueNorthPlacement(ordered_part_snn, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_swap_particleswarm(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = swapPartitioner(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    plac = particleSwarmPlacement(part_snn, hw, num_iterations = 20)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_multistart_truenorth(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionGreedyMultilevelRefinedMultistart(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = trueNorthPlacement(ordered_part_snn, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_multistart_spectral_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionGreedyMultilevelRefinedMultistart(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    plac = spectralPlacementScipy(part_snn, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(part_snn, plac))
    plac = forceDirectedRefinement(part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_setlist_hilbert_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionSetlistMiniHashWeightsForest(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = hilbertPlacement(ordered_part_snn.nodes, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    plac = forceDirectedRefinement(ordered_part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_setlist_hilbert_ps(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionSetlistMiniHashWeightsForest(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = hilbertPlacement(ordered_part_snn.nodes, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    plac = particleSwarmPlacement(ordered_part_snn, hw, num_iterations = 20, initial_layout = plac)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_setlist_spectral_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionSetlistMiniHashWeightsForest(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    plac = spectralPlacementScipy(part_snn, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(part_snn, plac))
    plac = forceDirectedRefinement(part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_setlist_spectral_ps(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionSetlistMiniHashWeightsForest(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    plac = spectralPlacementScipy(part_snn, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(part_snn, plac))
    plac = particleSwarmPlacement(part_snn, hw, num_iterations = 20, initial_layout = plac)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_setlist_truenorth(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionSetlistMiniHashWeightsForest(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = trueNorthPlacement(ordered_part_snn, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_hmetis_hilbert_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionHMETIS(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = hilbertPlacement(ordered_part_snn.nodes, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    plac = forceDirectedRefinement(ordered_part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_hmetis_hilbert_ps(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionHMETIS(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = hilbertPlacement(ordered_part_snn.nodes, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    plac = particleSwarmPlacement(ordered_part_snn, hw, num_iterations = 20, initial_layout = plac)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_hmetis_spectral_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionHMETIS(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    plac = spectralPlacementScipy(part_snn, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(part_snn, plac))
    plac = forceDirectedRefinement(part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_hmetis_spectral_ps(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionHMETIS(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    plac = spectralPlacementScipy(part_snn, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(part_snn, plac))
    plac = particleSwarmPlacement(part_snn, hw, num_iterations = 20, initial_layout = plac)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_hmetis_truenorth(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionHMETIS(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part, squish_hyperedges = True)
    res.partTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    del hg
    ordered_part_snn = feedForwardOrder(part_snn)
    plac = trueNorthPlacement(ordered_part_snn, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res


# PARTITIONING:

def run_unordered_sequential(name : str, hg : HyperGraph, hw : HardwareModel, seed : int, save : Optional[str] = None) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    res.endTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    if save:
        part_snn.save(os.path.join(save, name))
    return res

def run_sequential(name : str, hg : HyperGraph, hw : HardwareModel, seed : int, save : Optional[str] = None) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    if not isTopologicallySorted(hg):
        hg = feedForwardOrder(hg)
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    res.endTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    if save:
        part_snn.save(os.path.join(save, name))
    return res

def run_edgehiding(name : str, hg : HyperGraph, hw : HardwareModel, seed : int, save : Optional[str] = None) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionEdgeHiding(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    res.endTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    if save:
        part_snn.save(os.path.join(save, name))
    return res

def run_hehiding(name : str, hg : HyperGraph, hw : HardwareModel, seed : int, save : Optional[str] = None) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionHyperedgeHidingOnlyInbound(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    res.endTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    if save:
        part_snn.save(os.path.join(save, name))
    return res

def run_swap(name : str, hg : HyperGraph, hw : HardwareModel, seed : int, save : Optional[str] = None) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = swapPartitioner(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    res.endTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    if save:
        part_snn.save(os.path.join(save, name))
    return res

def run_multistart(name : str, hg : HyperGraph, hw : HardwareModel, seed : int, save : Optional[str] = None) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionGreedyMultilevelRefinedMultistart(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part)
    res.endTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    if save:
        part_snn.save(os.path.join(save, name))
    return res

def run_setlist(name : str, hg : HyperGraph, hw : HardwareModel, seed : int, save : Optional[str] = None) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionSetlistMiniHashWeightsForest(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    res.endTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    if save:
        part_snn.save(os.path.join(save, name))
    return res

def run_hmetis(name : str, hg : HyperGraph, hw : HardwareModel, seed : int, save : Optional[str] = None) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionHMETIS(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part)
    res.endTime()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency(), max(part) + 1, hw.synapticReuse(hg, part))
    if save:
        part_snn.save(os.path.join(save, name))
    return res


# PLACEMENT:

def run_hilbert_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    ordered_part_snn = feedForwardOrder(hg)
    plac = hilbertPlacement(ordered_part_snn.nodes, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    plac = forceDirectedRefinement(ordered_part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_hilbert_ps(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    ordered_part_snn = feedForwardOrder(hg)
    plac = hilbertPlacement(ordered_part_snn.nodes, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    plac = particleSwarmPlacement(ordered_part_snn, hw, num_iterations = 20, initial_layout = plac)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res

def run_spectral_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    plac = spectralPlacementScipy(hg, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(hg, plac))
    plac = forceDirectedRefinement(hg, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(hg, plac))
    return res

def run_spectral_ps(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    plac = spectralPlacementScipy(hg, hw.coresAlongX(), hw.coresAlongY())
    res.setInitPlac(**hw.getAllMetrics(hg, plac))
    plac = particleSwarmPlacement(hg, hw, num_iterations = 20, initial_layout = plac)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(hg, plac))
    return res

def run_truenorth(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    ordered_part_snn = feedForwardOrder(hg)
    plac = trueNorthPlacement(ordered_part_snn, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(ordered_part_snn, plac))
    return res