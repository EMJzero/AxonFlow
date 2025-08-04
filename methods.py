import json
import os

from graph_utils import *
from partitioner import *
from placer import *
from snn import *

# EXPERIMENTS:

class Result:
    name : str
    graph_nodes : int
    graph_edges : int
    graph_cost : float
    hw_npc : int
    hw_spc : int
    hw_cpc : int
    part_valid : bool
    part_cost : float # lower is better
    plac_valid : bool
    plac_energy : float
    plac_avg_lat : float
    plac_max_lat : float
    plac_avg_cong : float
    plac_max_cong : float
    note : str
    time : int

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

    def setPart(self, valid : bool, cost : float) -> None:
        self.part_valid = valid
        self.part_cost = cost
    
    def setPlac(self, valid : bool, energy : float, avg_latency : float, max_latency : float, avg_congestion : float, max_congestion : float) -> None:
        self.plac_valid = valid
        self.plac_energy = energy
        self.plac_avg_lat = avg_latency
        self.plac_max_lat = max_latency
        self.plac_avg_cong = avg_congestion
        self.plac_max_cong = max_congestion

    def setNote(self, note : str) -> None:
        self.note = note

    def startTime(self):
        self._start_time = time.time()

    def endTime(self):
        self.time = time.time() - self._start_time
        del self._start_time

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

def run_sequential_topo_hilbert_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    if isTopologicallySorted(part_snn):
        topological_order = part_snn
    else:
        topological_order, _ = topologicalOrder(part_snn, break_cycles = True)
    plac = hilbertPlacement(topological_order.nodes, hw.coresAlongX(), hw.coresAlongY()) 
    plac = forceDirectedRefinement(part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_sequential_hilbert_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    topological_order = feedForwardOrder(part_snn)
    plac = hilbertPlacement(topological_order.nodes, hw.coresAlongX(), hw.coresAlongY()) 
    plac = forceDirectedRefinement(part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_swap_particleswarm(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = swapPartitioner(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
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
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    topological_order = feedForwardOrder(part_snn)
    plac = trueNorthPlacement(topological_order, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(topological_order, plac))
    return res

def run_sequential_truenorth(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    topological_order = feedForwardOrder(part_snn)
    plac = trueNorthPlacement(topological_order, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(topological_order, plac))
    return res

def run_multistart_spectral_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionGreedyMultilevelRefinedMultistart(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    plac = spectralPlacement(part_snn.toGraph().toNxGraph(), hw.coresAlongX(), hw.coresAlongY())
    plac = forceDirectedRefinement(part_snn, plac, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_setlist_spectral_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionSetlistMiniHashWeightsForest(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    plac = spectralPlacement(part_snn.toGraph().toNxGraph(), hw.coresAlongX(), hw.coresAlongY())
    plac = forceDirectedRefinement(part_snn, plac, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_hmetis_hilbert_ps(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionHMETIS(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    topological_order = feedForwardOrder(part_snn)
    plac = hilbertPlacement(topological_order.nodes, hw.coresAlongX(), hw.coresAlongY()) 
    plac = particleSwarmPlacement(part_snn, hw, num_iterations = 20, initial_layout = plac)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_hmetis_spectral_fd(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionHMETIS(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    plac = spectralPlacement(part_snn.toGraph().toNxGraph(), hw.coresAlongX(), hw.coresAlongY())
    plac = forceDirectedRefinement(part_snn, plac, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res


# PARTITIONING:

def run_sequential(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    res.endTime()
    return res

def run_swap(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = swapPartitioner(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    res.endTime()
    return res

def run_multistart(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionGreedyMultilevelRefinedMultistart(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    res.endTime()
    return res

def run_setlist(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionSetlistMiniHashWeightsForest(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    res.endTime()
    return res

def run_hmetis(name : str, hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result(name)
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionHMETIS(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    res.endTime()
    return res