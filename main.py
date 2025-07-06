from typing import TypeVar, Union, Any
from types import FrameType

import traceback
import signal
import code
import time
import sys
import os

from graph_utils import *
from partitioner import *
from load_store import *
from settings import *
from placer import *
from prints import *
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
        "load": args_match_and_remove(["-l", "--load"], with_value = True),
        "save": args_match_and_remove(["-s", "--save"], with_value = True),
        "reload": args_match_and_remove(["-r", "--reload"], with_value = True),
        "quiet": args_match_and_remove(["-q", "--quiet"]),
    }
    return options

def help_options() -> None:
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-i --interactive\tOnce exploration has finished, instead of terminating the program, enter Python's interactive mode.")
    print(("-l, --load <?path>\tLoads a true SNN graph instead of randomly generating one. If omitted, the default path is './snn_models/simple_cnn'.\n"
           "\t\t\tThe given path is concatenated with '_0.npz', '_input.npz', '.graphml', these are the three files expected to be found."))
    print(("-s, --save <path>\tSaves the used SNN graph efficiently in 'path' after having built it. Recommended extension: '.hgr'."))
    print(("-r, --reload <path>\tReloads a previously saved (--save) SNN graph from 'path'. This takes priority on --load."))
    print("-q, --quiet\t\tDisable verbose logging of optimization functions.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    options = parse_options()

    if options["help"]:
        print("------------ HELP ------------")
        help_options()
        print("------------------------------")
        sys.exit(0)

    if options["quiet"]:
        Settings.VERBOSE = False

    # MAIN CODE:
    try:
        seed = 192 #79
        if options["reload"]:
            print("\n------- reloading graph ------")
            path = options["reload"]
            if not os.path.exists(path):
                raise Exception(f"The provided path does not exist: {path}")
            print("Reloading model from:", path)
            snn = HyperGraph.load(path)
            print(f"Nodes count: {snn.nodes}\nEdges: {len(snn.hyperedges)}\nMean nodes per edge: {sum(he.connections() for he in snn)/len(snn.hyperedges)}\nSeed: {seed}")
        elif options["load"] or options["load"] is None:
            print("\n-------- loading graph -------")
            path = options["load"] if options["load"] else "./snn_models/simple_cnn"
            #snn = loadSNNGraphML(f"{path}.graphml")
            snn = loadSNNcomposite(f"{path}_0.npz", f"{path}_input.npz", f"{path}.graphml")
            print(f"Nodes count: {snn.nodes}\nEdges: {len(snn.hyperedges)}\nMean nodes per edge: {sum(he.connections() for he in snn)/len(snn.hyperedges)}\nSeed: {seed}")
        else:
            print("\n------ generating graph ------")
            nodes_count = 1024
            nodes_per_edge_mean, nodes_per_edge_variation = 8, 4
            print(f"Nodes count: {nodes_count}\nNodes per edge mean: {nodes_per_edge_mean}\nNodes per edge variation: {nodes_per_edge_variation}\nSeed: {seed}")
            snn = HyperGraph.generate_random(nodes_count, nodes_per_edge_mean, nodes_per_edge_variation, seed = seed)
            #acyclic_snn = makeAcyclic(snn)
        if options["save"]:
            path = os.path.abspath(options["save"])
            os.makedirs(os.path.dirname(path), exist_ok = True)
            print("Saving model to:", path)
            snn.save(path)
            print("Saved, file size:", fileSizeString(os.path.getsize(path)))
        
        print("\n------- hardware model -------")
        hardware = HardwareModel(
            neurons_per_core = 256,
            synapses_per_core = 1024,
            cores_per_chip_x = 64,
            cores_per_chip_y = 64,
            chips_per_system_x = 1,
            chips_per_system_y = 1,
            energy_per_routing = 1.0,
            energy_per_wire = 0.1,
            latency_per_routing = 1.0,
            latency_per_wire = 0.1
        )
        print((f"Neurons per core: {hardware.neurons_per_core}\tSynapses per core: {hardware.synapses_per_core}\n"
               f"Cores along x: {hardware.cores_per_chip_x}\tCores along y: {hardware.cores_per_chip_y}\n"
               f"Chips along x: {hardware.chips_per_system_x}\tChips along y: {hardware.chips_per_system_y}\n"
               f"Routing energy: {hardware.energy_per_routing}\tWire energy: {hardware.energy_per_wire}\n"
               f"Routing latency: {hardware.latency_per_routing}\tWire latency: {hardware.latency_per_wire}"))

        print("\n---- checking feasibility ----")
        if not hardware.checkSnnFit(snn, verbose = True):
            print("WARNING: the generated SNN may not fit on the given HW, change either's configuration or the seed.")
        else:
            print("Passed!")
        print("\n-------- partitioning --------")
        partitioning_multilevel_multistart_refined = partitionGreedyMultilevelRefinedMultistart(snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount(), seed = seed) # NEW IDEA!
        partitioning_setlist = partitionSetlistMiniHashWeights(snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount()) # NEW IDEA!
        partitioning_greedy = partitionGreedy(snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount()) # A piece of a new idea.
        partitioning_sequential = partitionSequential(snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount()) # Ouwen Jin's paper.
        partitioning_swap = swapPartitioner(snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount()) # DFSynthesizer's paper.
        partitioning_hmetis = partitionHMETIS(snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount(), seed = seed) # Ouwen Jin's paper.
        part_snn_mmr = snn.getPartitionsHypergraph(partitioning_multilevel_multistart_refined, keep_self_cycles = True)
        part_snn_setlist = snn.getPartitionsHypergraph(partitioning_setlist, keep_self_cycles = True)
        part_snn_greedy = snn.getPartitionsHypergraph(partitioning_greedy, keep_self_cycles = True)
        part_snn_seq = snn.getPartitionsHypergraph(partitioning_sequential, keep_self_cycles = True)
        part_snn_swap = snn.getPartitionsHypergraph(partitioning_swap, keep_self_cycles = True)
        part_snn_hmetis = snn.getPartitionsHypergraph(partitioning_hmetis, keep_self_cycles = True)
        part_snn_mmr.squishHyperedges()
        part_snn_setlist.squishHyperedges()
        part_snn_greedy.squishHyperedges()
        part_snn_seq.squishHyperedges()
        part_snn_swap.squishHyperedges()
        part_snn_hmetis.squishHyperedges()
        topological_order_mmr, masked_edges_mmr = topologicalOrder(part_snn_mmr, break_cycles = True) # Setup Locality for TrueNorth's placement algorithm.
        topological_order_seq, masked_edges_seq = topologicalOrder(part_snn_seq, break_cycles = True) # Setup Locality as in Ouwen Jin's paper.
        print("Metrics multilevel multistart refined partitioning:")
        prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_multilevel_multistart_refined), 'tot_hyperedges_spike_frequency': part_snn_mmr.totalSpikeFrequency()}, 1)
        print("Metrics setlist minihash partitioning:")
        prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_setlist), 'tot_hyperedges_spike_frequency': part_snn_setlist.totalSpikeFrequency()}, 1)
        print("Metrics greedy partitioning:")
        prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_greedy), 'tot_hyperedges_spike_frequency': part_snn_greedy.totalSpikeFrequency()}, 1)
        print("Metrics sequential partitioning:")
        prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_sequential), 'tot_hyperedges_spike_frequency': part_snn_seq.totalSpikeFrequency()}, 1)
        print("Metrics swap partitioning:")
        prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_swap), 'tot_hyperedges_spike_frequency': part_snn_swap.totalSpikeFrequency()}, 1)
        print("Metrics hMETIS partitioning:")
        prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_hmetis), 'tot_hyperedges_spike_frequency': part_snn_hmetis.totalSpikeFrequency()}, 1)

        print("\n----------- layout -----------")
        # These are complete approaches, novel or from previous works
        spectral_placement = spectralPlacement(part_snn_mmr.toGraph().toNxGraph(), hardware.coresAlongX(), hardware.coresAlongY()) # NEW IDEA!
        hsc_placement = hilbertPlacement(topological_order_seq.nodes, hardware.coresAlongX(), hardware.coresAlongY()) # Ouwen Jin's paper.
        pso_placement = particleSwarmPlacement(part_snn_swap, hardware, num_iterations = 20) # This is the full approach from DFSynthesizer's paper.
        truenorth_placement = trueNorthPlacement(topological_order_mmr, masked_edges_mmr, hardware) # TrueNorth's placement algorithm
        metrics_spectral = hardware.getAllMetrics(part_snn_mmr, spectral_placement)
        metrics_hsc = hardware.getAllMetrics(topological_order_seq, hsc_placement)
        metrics_pso = hardware.getAllMetrics(part_snn_swap, pso_placement)
        metrics_truenorth = hardware.getAllMetrics(topological_order_mmr, truenorth_placement)
        print("Metrics spectral layout (canon version - multilevel multistart refined partitioning):")
        prettyPrintDict(metrics_spectral, 1)
        print("Metrics HSC layout (canon version - sequential partitioning):")
        prettyPrintDict(metrics_hsc, 1)
        print("Metrics PSO layout (canon version - swap partitioning):")
        prettyPrintDict(metrics_pso, 1)
        print("Metrics TrueNorth layout (not-so-much canon version - multilevel multistart refined partitioning):")
        prettyPrintDict(metrics_truenorth, 1)
        # These are crossbreeds obtained by mixing placement and partitioning algorithms
        spectral_placement_variant = spectralPlacement(part_snn_seq.toGraph().toNxGraph(), hardware.coresAlongX(), hardware.coresAlongY())
        hsc_placement_variant = hilbertPlacement(topological_order_mmr.nodes, hardware.coresAlongX(), hardware.coresAlongY())
        metrics_spectral_variant = hardware.getAllMetrics(part_snn_seq, spectral_placement_variant)
        metrics_hsc_variant = hardware.getAllMetrics(topological_order_mmr, hsc_placement_variant)
        print("Metrics spectral layout (crossbreed - sequential partitioning):")
        prettyPrintDict(metrics_spectral_variant, 1)
        print("Metrics HSC layout (crossbreed - multilevel multistart refined partitioning):")
        prettyPrintDict(metrics_hsc_variant, 1)
        # these are the complete approaches plus FD algorithm
        spectral_placement_fd = forceDirectedRefinement(part_snn_mmr, spectral_placement, hardware) # 1/2 NEW IDEA!
        hsc_placement_fd = forceDirectedRefinement(topological_order_seq, hsc_placement, hardware) # This is the full approach from Ouwen Jin's paper.
        metrics_spectral_fd = hardware.getAllMetrics(part_snn_mmr, spectral_placement_fd)
        metrics_hsc_fd = hardware.getAllMetrics(topological_order_seq, hsc_placement_fd)
        print("Metrics spectral layout (canon version - refined with force-directed algorithm):")
        prettyPrintDict(metrics_spectral_fd, 1)
        print("Metrics HSC layout (canon version - refined with force-directed algorithm):")
        prettyPrintDict(metrics_hsc_fd, 1)
    except Exception:
        print(traceback.format_exc())

    if options["interactive"]:
        print("\n------ interactive mode ------")
        in_interactive_mode = True
        code.interact(local=globals())
        in_interactive_mode = False