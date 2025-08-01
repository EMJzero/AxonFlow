from typing import TypeVar, Any, Optional
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

def parse_options() -> dict[str, Any]:
    options = {
        "help": args_match_and_remove(["-h", "--help"]),
        "interactive": args_match_and_remove(["-i", "--interactive"]),
        "load": args_match_and_remove(["-l", "--load"], with_value = True),
        "save": args_match_and_remove(["-s", "--save"], with_value = True),
        "reload": args_match_and_remove(["-r", "--reload"], with_value = True),
        "fraction": args_match_and_remove(["-f", "--fraction"], with_value = True, value_type = float),
        "quiet": args_match_and_remove(["-q", "--quiet"]),
    }
    return options

def help_options() -> None:
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-i --interactive\tOnce exploration has finished, instead of terminating the program, enter Python's interactive mode.")
    print(("-l, --load <?path>\tLoads a true SNN graph instead of randomly generating one. If omitted, the default path is './snn_models/simple_cnn'.\n"
           "\t\t\tThe given path is concatenated with '_0.npz', '_input.npz', '.graphml', these are the three files expected to be found."))
    print("-s, --save <path>\tSaves the used SNN graph efficiently in 'path' after having built it. Recommended extension: '.hgr'.")
    print("-r, --reload <path>\tReloads a previously saved (--save) SNN graph from 'path'. This takes priority on --load.")
    print("-f, --fraction <num>\tFraction of the lowest-spike-frequency hyperedges to ignore (still count for costs), let it be a number in [0, 1].")
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
    else:
        #print("----------^----V----^---------")
        #print("--------- Axon . Flow --------")
        #print("------------- ~~~ ------------\n")
        print("--~~~~~~~~~~~~----------------")
        print("------~~~ Axon ~ Flow ~~~-----")
        print("----------------~~~~~~~~~~~~--\n")

    if os.name != "posix":
        print("WARNING: this program was developed for a UNIX-like environment, expect bugs (especially with signals and multiprocessing) on other systems.")

    if Settings.MULTIPROCESSING:
        multiprocessing.current_process().name = '0'

    # MAIN CODE:
    try:
        seed = 192 #79
        print("seed:", seed)
        
        if options["reload"]:
            print("\n------- reloading graph ------")
            path = options["reload"]
            if not os.path.exists(path):
                raise Exception(f"The provided path does not exist: {path}")
            print("Reloading model from:", path)
            snn = HyperGraph.load(path)
            snn_stats = snn.getStatistics()
        elif options["load"] or options["load"] is None:
            print("\n-------- loading graph -------")
            path = options["load"] if options["load"] else "./snn_models/simple_cnn"
            #snn = loadSNNGraphML(f"{path}.graphml")
            snn = loadSNNcomposite(f"{path}_0.npz", f"{path}_input.npz", f"{path}.graphml")
            snn_stats = snn.getStatistics()
        else:
            print("\n------ generating graph ------")
            nodes_count = 1024
            nodes_per_edge_mean, nodes_per_edge_variation = 8, 4
            snn = HyperGraph.generate_random(nodes_count, nodes_per_edge_mean, nodes_per_edge_variation, seed = seed)
            snn_stats = snn.getStatistics()
            snn_stats["nodes_per_edge_mean"] = nodes_per_edge_mean
            snn_stats["nodes_per_edge_variation"] = nodes_per_edge_variation
            #acyclic_snn = makeAcyclic(snn)
        prettyPrintDict(snn.getStatistics(), formatter = lambda v : f"{v:.3f}")
        
        if options["save"]:
            print("\n-------- saving graph --------")
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
        
        def line_by_line():
            print("\n-------- partitioning --------")
            # > remove lowest-spike-frequency hyperedges
            if options["fraction"]:
                # SOLUTION: each partitioning algorithm should handle the removal of hyperedges internally!
                print("DANGER: known issue, partitioning w/out some hyperedges can result in severe constraints violations once those are added back!!!")
                fraction = options["fraction"]
                total_connections = sum(he.connections() for he in snn)
                removed_hes, removed_connections, removed_spike_frequency = snn.removeHyperEdgesFraction(fraction)
                total_spike_frequency = snn.totalSpikeFrequency()
                print(f"Ignored lowest {100*fraction:.1f}% of hyperedges by spike frequency:")
                prettyPrintDict({"ignored count": f"{removed_connections}/{total_connections}",
                                 "ignored total spike frequency": f"{removed_spike_frequency:.3f}/{total_spike_frequency:.3f} ({100*removed_spike_frequency/total_spike_frequency:.3g}%)"}, 1)
            # > partition
            partitioning_multilevel_multistart_refined = Worker(partitionGreedyMultilevelRefinedMultistart, snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount(), seed = seed) # NEW IDEA!
            partitioning_setlist = Worker(partitionSetlistMiniHashWeightsForest, snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount()) # NEW IDEA!
            partitioning_greedy = Worker(partitionGreedy, snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount()) # A piece of a new idea.
            partitioning_sequential = Worker(partitionSequential, snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount()) # Ouwen Jin's paper.
            #partitioning_swap = Worker(swapPartitioner, snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount()) # DFSynthesizer's paper.
            partitioning_hmetis = Worker(partitionHMETIS, snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount(), seed = seed) # Ouwen Jin's paper.
            partitioning_multilevel_multistart_refined = partitioning_multilevel_multistart_refined.get()
            partitioning_setlist = partitioning_setlist.get()
            partitioning_greedy = partitioning_greedy.get()
            partitioning_sequential = partitioning_sequential.get()
            #partitioning_swap = partitioning_swap.get()
            partitioning_hmetis = partitioning_hmetis.get()
            # > reinstate lowest-spike-frequency hyperedges
            if options["fraction"]:
                snn.addHyperedges(removed_hes)
            # > compute partitioned hypergraphs, remove redundant hyperedges, topologically order their nodes
            part_snn_mmr = snn.getPartitionsHypergraph(partitioning_multilevel_multistart_refined, keep_self_cycles = True)
            part_snn_setlist = snn.getPartitionsHypergraph(partitioning_setlist, keep_self_cycles = True)
            part_snn_greedy = snn.getPartitionsHypergraph(partitioning_greedy, keep_self_cycles = True)
            part_snn_seq = snn.getPartitionsHypergraph(partitioning_sequential, keep_self_cycles = True)
            #part_snn_swap = snn.getPartitionsHypergraph(partitioning_swap, keep_self_cycles = True)
            part_snn_hmetis = snn.getPartitionsHypergraph(partitioning_hmetis, keep_self_cycles = True)
            part_snn_mmr.squishHyperedges()
            part_snn_setlist.squishHyperedges()
            part_snn_greedy.squishHyperedges()
            part_snn_seq.squishHyperedges()
            #part_snn_swap.squishHyperedges()
            part_snn_hmetis.squishHyperedges()
            topological_order_mmr, masked_edges_mmr = topologicalOrder(part_snn_mmr, break_cycles = True) # Setup Locality for TrueNorth's placement algorithm.
            topological_order_seq, masked_edges_seq = topologicalOrder(part_snn_seq, break_cycles = True) # Setup Locality as in Ouwen Jin's paper.
            # > print metrics
            print("Metrics multilevel multistart refined partitioning:")
            prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_multilevel_multistart_refined), 'tot_hyperedges_spike_frequency': part_snn_mmr.totalSpikeFrequency()}, 1)
            print("Metrics setlist minihash partitioning:")
            prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_setlist), 'tot_hyperedges_spike_frequency': part_snn_setlist.totalSpikeFrequency()}, 1)
            print("Metrics greedy partitioning:")
            prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_greedy), 'tot_hyperedges_spike_frequency': part_snn_greedy.totalSpikeFrequency()}, 1)
            print("Metrics sequential partitioning:")
            prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_sequential), 'tot_hyperedges_spike_frequency': part_snn_seq.totalSpikeFrequency()}, 1)
            #print("Metrics swap partitioning:")
            #prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_swap), 'tot_hyperedges_spike_frequency': part_snn_swap.totalSpikeFrequency()}, 1)
            print("Metrics hMETIS partitioning:")
            prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_hmetis), 'tot_hyperedges_spike_frequency': part_snn_hmetis.totalSpikeFrequency()}, 1)

            print("\n----------- layout -----------")
            # TODO: should we exclude lowest-spike-frequency hyperedges here too?
            # These are complete approaches, novel or from previous works
            # > placement
            spectral_placement = Worker(spectralPlacement, part_snn_mmr.toGraph().toNxGraph(), hardware.coresAlongX(), hardware.coresAlongY()) # NEW IDEA!
            hsc_placement = Worker(hilbertPlacement, topological_order_seq.nodes, hardware.coresAlongX(), hardware.coresAlongY()) # Ouwen Jin's paper.
            #pso_placement = Worker(particleSwarmPlacement, part_snn_swap, hardware, num_iterations = 20) # This is the full approach from DFSynthesizer's paper.
            truenorth_placement = Worker(trueNorthPlacement, topological_order_mmr, masked_edges_mmr, hardware) # TrueNorth's placement algorithm
            spectral_placement = spectral_placement.get()
            hsc_placement = hsc_placement.get()
            #pso_placement = pso_placement.get()
            truenorth_placement = truenorth_placement.get()
            # > print metrics
            metrics_spectral = hardware.getAllMetrics(part_snn_mmr, spectral_placement)
            metrics_hsc = hardware.getAllMetrics(topological_order_seq, hsc_placement)
            #metrics_pso = hardware.getAllMetrics(part_snn_swap, pso_placement)
            metrics_truenorth = hardware.getAllMetrics(topological_order_mmr, truenorth_placement)
            print("Metrics spectral layout (canon version - multilevel multistart refined partitioning):")
            prettyPrintDict(metrics_spectral, 1)
            print("Metrics HSC layout (canon version - sequential partitioning):")
            prettyPrintDict(metrics_hsc, 1)
            #print("Metrics PSO layout (canon version - swap partitioning):")
            #prettyPrintDict(metrics_pso, 1)
            print("Metrics TrueNorth layout (not-so-much canon version - multilevel multistart refined partitioning):")
            prettyPrintDict(metrics_truenorth, 1)
            # These are crossbreeds obtained by mixing placement and partitioning algorithms
            # > placement
            spectral_placement_variant = Worker(spectralPlacement, part_snn_seq.toGraph().toNxGraph(), hardware.coresAlongX(), hardware.coresAlongY())
            hsc_placement_variant = Worker(hilbertPlacement, topological_order_mmr.nodes, hardware.coresAlongX(), hardware.coresAlongY())
            spectral_placement_variant = spectral_placement_variant.get()
            hsc_placement_variant = hsc_placement_variant.get()
            # > print metrics
            metrics_spectral_variant = hardware.getAllMetrics(part_snn_seq, spectral_placement_variant)
            metrics_hsc_variant = hardware.getAllMetrics(topological_order_mmr, hsc_placement_variant)
            print("Metrics spectral layout (crossbreed - sequential partitioning):")
            prettyPrintDict(metrics_spectral_variant, 1)
            print("Metrics HSC layout (crossbreed - multilevel multistart refined partitioning):")
            prettyPrintDict(metrics_hsc_variant, 1)
            # These are the complete approaches plus FD algorithm
            # > placement
            spectral_placement_fd = Worker(forceDirectedRefinement, part_snn_mmr, spectral_placement, hardware) # 1/2 NEW IDEA!
            hsc_placement_fd = Worker(forceDirectedRefinement, topological_order_seq, hsc_placement, hardware, fixes = False) # This is the full approach from Ouwen Jin's paper.
            spectral_placement_fd = spectral_placement_fd.get()
            hsc_placement_fd = hsc_placement_fd.get()
            # > print metrics
            metrics_spectral_fd = hardware.getAllMetrics(part_snn_mmr, spectral_placement_fd)
            metrics_hsc_fd = hardware.getAllMetrics(topological_order_seq, hsc_placement_fd)
            print("Metrics spectral layout (canon version - refined with force-directed algorithm):")
            prettyPrintDict(metrics_spectral_fd, 1)
            print("Metrics HSC layout (canon version - refined with force-directed algorithm):")
            prettyPrintDict(metrics_hsc_fd, 1)
        # run the above function while skipping lines that give exceptions...
        line_by_line = make_swallowing_wrapper(line_by_line)
        line_by_line()
    except Exception:
        print(traceback.format_exc())

    kill_all_children()

    if options["interactive"]:
        print("\n------ interactive mode ------")
        in_interactive_mode = True
        code.interact(local = globals())
        in_interactive_mode = False