from typing import TypeVar, Union, Any
from types import FrameType

import traceback
import signal
import code
import json
import time
import sys
import os

from graph_utils import *
from partitioner import *
from load_store import *
from settings import *
from placer import *
from prints import *
from utils import *
from model import *
from snn import *

# CLI MANAGEMENT

in_interactive_mode = False

def signal_handler(signal: int, frame: Optional[FrameType]) -> None:
    global in_interactive_mode
    if in_interactive_mode:
        print('EXITING...')
        sys.exit(0)
    else:
        print('\nHANDLING TERMINATION...\n')
        kill_all_children()
        stack = traceback.format_stack(frame)
        print('------------ stack -----------')
        print(''.join(stack[:-1])[:-1])
        print('------------------------------')
        #stop_threads()
        time.sleep(0.2)
        print('\nTERMINATION RECEIVED - SWITCHING TO INTERACTIVE MODE\n[type "exit()" or press "ctrl+c" again to terminate the program]\n')
        in_interactive_mode = True
        code.interact(local=globals())
        in_interactive_mode = False

T = TypeVar('T')

"""
Searchs and removes flags from 'sys.argv'.
If 'with_value' is False, the return values is either True or False depending on the presence or absence of the option.
If 'with_value' is True, the return value is the value assigned with the option, if present, otherwise it is False if
the option is not present and None if no valid argument was provided.
Optionally, 'value_type' can be used to parse the desired value when 'with_value' is True.
Optionally, use 'flags_tag' to override the flags marker if not using '-'.
"""
def args_match_and_remove(flags: Union[str, list[str]], with_value: bool = False, value_type: type[T] = str, flags_tag : str = '-') -> Union[bool, T, None]:
    if isinstance(flags, str):
        flags = [flags]
    for flag in flags:
        try:
            idx = sys.argv.index(flag)
            sys.argv.pop(idx)

            if with_value:
                if idx >= len(sys.argv) or sys.argv[idx].startswith(flags_tag):
                    return None  # flag present, value is missing or looks like another flag
                try:
                    value = value_type(sys.argv[idx])
                    sys.argv.pop(idx)
                    return value
                except Exception:
                    return None  # flag present, value couldn't be parsed
            else:
                return True  # flag present, no value expected
        except ValueError:
            continue
    return False  # no matching of the flags found

def parse_options() -> dict[str, Any]:
    options = {
        "help": args_match_and_remove(["-h", "--help"]),
        "interactive": args_match_and_remove(["-i", "--interactive"]),
        "output": args_match_and_remove(["-o", "--output"], with_value = True),
        "quiet": args_match_and_remove(["-q", "--quiet"]),
    }
    return options

def help_options() -> None:
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-i --interactive\tOnce exploration has finished, instead of terminating the program, enter Python's interactive mode.")
    print("-o --output <file>\tName of the '.json' file where to write results.")
    print("-q, --quiet\t\tDisable verbose logging of optimization functions.")

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
        self._start_time = time.perf_counter()

    def endTime(self):
        self.time = time.perf_counter() - self._start_time
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

def run_sequential_hilbert_fd(hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result("sequential-hilbert-fd")
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionSequential(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, keep_self_cycles = True)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    topological_order, masked_edges = topologicalOrder(part_snn, break_cycles = True)
    plac = hilbertPlacement(topological_order.nodes, hw.coresAlongX(), hw.coresAlongY()) 
    plac = forceDirectedRefinement(part_snn, plac, hw, fixes = False)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_swap_particleswarm(hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result("swap-particleswarm")
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = swapPartitioner(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, keep_self_cycles = True)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    plac = particleSwarmPlacement(part_snn, hw, num_iterations = 20)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_multistart_truenorth(hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result("multistart-truenorth")
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionGreedyMultilevelRefinedMultistart(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part, keep_self_cycles = True)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    topological_order, masked_edges = topologicalOrder(part_snn, break_cycles = True)
    plac = trueNorthPlacement(topological_order, masked_edges, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(topological_order, plac))
    return res

def run_multistart_spectral_fd(hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result("multistart-spectral-fd")
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionGreedyMultilevelRefinedMultistart(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part, keep_self_cycles = True)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    plac = spectralPlacement(part_snn.toGraph().toNxGraph(), hw.coresAlongX(), hw.coresAlongY())
    plac = forceDirectedRefinement(part_snn, plac, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_setlist_spectral_fd(hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result("setlist-spectral-fd")
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.startTime()
    part = partitionSetlistMiniHashWeightsForest(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    part_snn = hg.getPartitionsHypergraph(part, keep_self_cycles = True)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    plac = spectralPlacement(part_snn.toGraph().toNxGraph(), hw.coresAlongX(), hw.coresAlongY())
    plac = forceDirectedRefinement(part_snn, plac, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_hmetis_hilbert_ps(hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result("hmetis-hilbert-ps")
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionHMETIS(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part, keep_self_cycles = True)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    topological_order, masked_edges = topologicalOrder(part_snn, break_cycles = True)
    plac = hilbertPlacement(topological_order.nodes, hw.coresAlongX(), hw.coresAlongY()) 
    plac = particleSwarmPlacement(part_snn, hw, num_iterations = 20, initial_layout = plac)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res

def run_hmetis_spectral_fd(hg : HyperGraph, hw : HardwareModel, seed : int) -> Result:
    res = Result("hmetis-spectral-fd")
    res.setGraph(hg.nodes, hg.totalConnections(), hg.totalSpikeFrequency())
    res.setHw(hw.neurons_per_core, hw.synapses_per_core, hw.coresCount())
    res.startTime()
    part = partitionHMETIS(hg, hw.neurons_per_core, hw.synapses_per_core, hw.coresCount(), seed = seed)
    part_snn = hg.getPartitionsHypergraph(part, keep_self_cycles = True)
    part_snn.squishHyperedges()
    res.setPart(hw.checkPartitionValidity(hg, part), part_snn.totalSpikeFrequency())
    plac = spectralPlacement(part_snn.toGraph().toNxGraph(), hw.coresAlongX(), hw.coresAlongY())
    plac = forceDirectedRefinement(part_snn, plac, hw)
    res.endTime()
    res.setPlac(**hw.getAllMetrics(part_snn, plac))
    return res


if __name__ == "__main__":
    if os.name != "posix":
        print("WARNING: this program was developed for a UNIX-like environment, expect bugs (especially with signals and multiprocessing) on other systems.")

    signal.signal(signal.SIGINT, signal_handler)

    options = parse_options()

    if options["quiet"]:
        Settings.VERBOSE = False

    Settings.CORE_TIMEOUT = 3600

    if options["help"]:
        print("------------ HELP ------------")
        help_options()
        print("------------------------------")
        sys.exit(0)

    if not options["output"]:
        options["output"] = "scalability_results.json"
        print(f"WARNING: missing '-o' option, defaulting to '{options["output"]}'.")
    elif not options["output"].endswith(".json"):
        options["output"] += ".json"
        print(f"WARNING: the output file was missing the '.json' extension, it has updated to '{options["output"]}'.")

    if Settings.MULTIPROCESSING:
        multiprocessing.current_process().name = '0'

    # MAIN CODE:
    try:
        seed = 192 #79
        sizes = [
            {"nodes_count": 256, "nodes_per_edge_mean": 4, "nodes_per_edge_variation": 2,
             "neurons_per_core": 16, "synapses_per_core" : 64, "cores_per_chip_1d": 64},
            {"nodes_count": 512, "nodes_per_edge_mean": 6, "nodes_per_edge_variation": 3,
             "neurons_per_core": 24, "synapses_per_core" : 96, "cores_per_chip_1d": 64},
            {"nodes_count": 1024, "nodes_per_edge_mean": 8, "nodes_per_edge_variation": 4,
             "neurons_per_core": 32, "synapses_per_core" : 128, "cores_per_chip_1d": 64},
            {"nodes_count": 1024*2, "nodes_per_edge_mean": 12, "nodes_per_edge_variation": 6,
             "neurons_per_core": 64, "synapses_per_core" : 256, "cores_per_chip_1d": 64},
            {"nodes_count": 1024*4, "nodes_per_edge_mean": 24, "nodes_per_edge_variation": 12,
             "neurons_per_core": 96, "synapses_per_core" : 384, "cores_per_chip_1d": 64},
            {"nodes_count": 1024*8, "nodes_per_edge_mean": 32, "nodes_per_edge_variation": 16,
             "neurons_per_core": 96, "synapses_per_core" : 384, "cores_per_chip_1d": 64},
            {"nodes_count": 1024*16, "nodes_per_edge_mean": 48, "nodes_per_edge_variation": 24,
             "neurons_per_core": 128, "synapses_per_core" : 512, "cores_per_chip_1d": 64},
            {"nodes_count": 1024*32, "nodes_per_edge_mean": 64, "nodes_per_edge_variation": 32,
             "neurons_per_core": 192, "synapses_per_core" : 768, "cores_per_chip_1d": 64},
            {"nodes_count": 1024*64, "nodes_per_edge_mean": 96, "nodes_per_edge_variation": 48,
             "neurons_per_core": 256, "synapses_per_core" : 1024, "cores_per_chip_1d": 64}
        ]
        methods = [
            run_sequential_hilbert_fd,
            run_swap_particleswarm,
            run_multistart_truenorth,
            run_multistart_spectral_fd,
            run_setlist_spectral_fd,
            run_hmetis_hilbert_ps,
            run_hmetis_spectral_fd
        ]
        
        for size in sizes:
            print("\n------------------------------")
            print("Working on configuration:")
            prettyPrintDict(size, 1)
            hardware = HardwareModel(
                neurons_per_core = size["neurons_per_core"],
                synapses_per_core = size["synapses_per_core"],
                cores_per_chip_x = size["cores_per_chip_1d"],
                cores_per_chip_y = size["cores_per_chip_1d"],
                chips_per_system_x = 1,
                chips_per_system_y = 1,
                energy_per_routing = 1.0,
                energy_per_wire = 0.1,
                latency_per_routing = 1.0,
                latency_per_wire = 0.1
            )
            hypergraph = HyperGraph.generate_random(size["nodes_count"], size["nodes_per_edge_mean"], size["nodes_per_edge_variation"], seed = seed)
            #acyclic_snn = makeAcyclic(snn)
            if not hardware.checkSnnFit(hypergraph, verbose = True):
                print("WARNING: the generated SNN may not fit on the given HW, change either's configuration or the seed.")

            workers : list[Worker] = []
            for method in methods:
                workers.append(Worker(method, hypergraph, hardware, seed))
            
            for worker in workers:
                try:
                    res : Result = worker.get()
                except Exception as e:
                    res = Result("N/A")
                    res.setNote("Failed. Exception: " + str(e))
                res.toFile(options["output"])
                print("\n---------------")
                prettyPrintDict(res.__dict__)
        
    except Exception:
        print(traceback.format_exc())

    kill_all_children()

    if options["interactive"]:
        print("\n------ interactive mode ------")
        in_interactive_mode = True
        code.interact(local = globals())
        in_interactive_mode = False