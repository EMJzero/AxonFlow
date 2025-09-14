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
from methods import *
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
        code.interact(local = globals())
        in_interactive_mode = False

T = TypeVar('T')

def parse_options() -> dict[str, Any]:
    options = {
        "help": args_match_and_remove(["-h", "--help"]),
        "interactive": args_match_and_remove(["-i", "--interactive"]),
        "load": args_match_and_remove(["-l", "--load"], with_value = True),
        "save": args_match_and_remove(["-s", "--save"], with_value = True),
        "save-map": args_match_and_remove(["-sm", "--save-map"], with_value = True),
        "reload": args_match_and_remove(["-r", "--reload"], with_value = True),
        "reload-map": args_match_and_remove(["-rm", "--reload-map"], with_value = True),
        "output": args_match_and_remove(["-o", "--output"], with_value = True),
        #"partitioning": args_match_and_remove(["-p", "--partitioning"]),
        "placement": args_match_and_remove(["-pp", "--placement"]),
        "fraction": args_match_and_remove(["-f", "--fraction"], with_value = True, value_type = float),
        "quiet": args_match_and_remove(["-q", "--quiet"]),
    }
    return options

def help_options() -> None:
    print("This main aims to reload previously computed partitionings and/or placements, that were saved through '--save-map'.")
    print("After reloading them, it recomputes their statistics (except execution time) and optionally ('-p', '-pp') reruns part of the mapping flow.")
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-i, --interactive\tOnce exploration has finished, instead of terminating the program, enter Python's interactive mode.")
    print(("-l, --load <?path>\tLoads a true SNN graph instead of randomly generating one. If omitted, the default path is './snn_models/simple_cnn'.\n"
           "\t\t\tThe given path is concatenated with '_0.npz', '_input.npz', '.graphml', these are the three files expected to be found."))
    print("-s, --save <path>\tSaves the input SNN graph efficiently in 'path' after having built it. Recommended extension: '.hgr'.")
    print("-sm, --save-map <dir>\tIf new placements are computed ('-pp'), saves the placement ('.plac') lists in 'dir'. Each in a file named after the methods.")
    print("-r, --reload <path>\tReloads a previously saved (--save) SNN graph from 'path'. This takes priority on --load.")
    print("-rm, --reload-map <dir>\tReloads previously saved (--save-map) partitions and/or placement lists from 'dir'.")
    print("-o, --output <file>\tName of the '.json' file where to write results.")
    #print("-p, --partitioning\tReruns (does not reload) the partitioning algorithms part, uses reloaded placement (takes priority over '-pp').")
    print("-pp, --placement\tReruns (does not reload) the placement algorithms part, uses reloaded partitioning (give a normal SNN as input).")
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
        print("┌────┬────┬────────┬────┬────┐")
        print("│ ┌──┘    └──┐  ┌──┘    └──┐ │")
        print("└─┘   AXON   └──┘   FLOW   └─┘\n")

    Settings.CORE_TIMEOUT = 3600*100

    if os.name != "posix":
        print("WARNING: this program was developed for a UNIX-like environment, expect bugs (especially with signals and multiprocessing) on other systems.")

    if options["save-map"]:
        if not os.path.isdir(options["save-map"]):
            os.mkdir(options["save-map"])
            print(f"WARNING: directory '{options["save-map"]}' did not exist, it has now been created.")
        elif len(os.listdir(options["save-map"])) != 0:
            print(f"WARNING: directory '{options["save-map"]}' is not empty, some files may get overwritten.")
    if options["reload-map"] and not os.path.isdir(options["reload-map"]):
        print(f"ERROR: directory '{options["reload-map"]}' does not exist, can't reload previous partitioning and/or placement lists.")
        sys.exit(0)

    if not options["output"]:
        options["output"] = "targeted_results.json"
        print(f"WARNING: missing '-o' option, defaulting to '{options["output"]}'.")
    elif not options["output"].endswith(".json"):
        options["output"] += ".json"
        print(f"WARNING: the output file was missing the '.json' extension, it has updated to '{options["output"]}'.")

    if Settings.MULTIPROCESSING:
        multiprocessing.current_process().name = '0'

    print("Start time (GMT):", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    # MAIN CODE:
    try:
        seed = 192 #79
        print("Seed:", seed)
        save = options["save-map"]
        if save: print("Saving partitioning and placement lists in:", save)
        
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
            nodes_count = 1024 #512
            nodes_per_edge_mean, nodes_per_edge_variation = 6, 4 #8, 4
            snn = HyperGraph.generate_random(nodes_count, nodes_per_edge_mean, nodes_per_edge_variation, seed = seed)
            #snn = HyperGraph.generate_reservoir_random(n = 5000, mean_fanout = 200, space_dim = 2, locality_sigma = 0.35, long_range_fraction = 0.05, spike_rate_median = 1.0, spike_rate_cv = 4.0, seed = seed)
            snn_stats = snn.getStatistics()
            snn_stats["nodes_per_edge_mean"] = nodes_per_edge_mean
            snn_stats["nodes_per_edge_variation"] = nodes_per_edge_variation
            #acyclic_snn = makeAcyclic(snn)
        prettyPrintDict(snn_stats, formatter = lambda v : f"{v:.3f}")
        
        if options["save"]:
            print("\n-------- saving graph --------")
            path = os.path.abspath(options["save"])
            os.makedirs(os.path.dirname(path), exist_ok = True)
            print("Saving model to:", path)
            snn.save(path)
            print("Saved, file size:", fileSizeString(os.path.getsize(path)))
        
        print("\n------- hardware model -------")
        hardware = loihi_jin_84
        print((f"Neurons per core: {hardware.neurons_per_core}\tSynapses per core: {hardware.synapses_per_core}\n"
               f"Cores along x: {hardware.cores_per_chip_x}\tCores along y: {hardware.cores_per_chip_y}\n"
               f"Chips along x: {hardware.chips_per_system_x}\tChips along y: {hardware.chips_per_system_y}\n"
               f"Routing energy: {hardware.energy_per_routing}\tWire energy: {hardware.energy_per_wire}\n"
               f"Routing latency: {hardware.latency_per_routing}\tWire latency: {hardware.latency_per_wire}"))
        
        methods : list[str] = [
            #"sequential-topo-hilbert-fd",
            "sequential-hilbert-fd",
            "sequential-hilbert-ps",
            "sequential-spectral-fd",
            "sequential-spectral-ps",
            "sequential-truenorth",
            "hehiding-hilbert-fd",
            "hehiding-hilbert-ps",
            "hehiding-spectral-fd",
            "hehiding-spectral-ps",
            "hehiding-truenorth",
            ###"swap-particleswarm",
            ###"multistart-truenorth",
            ###"multistart-spectral-fd",
            ##"setlist-hilbert-fd",
            ##"setlist-hilbert-ps",
            ##"setlist-spectral-fd",
            ##"setlist-spectral-ps",
            ##"setlist-truenorth",
            "hmetis-hilbert-fd",
            "hmetis-hilbert-ps",
            "hmetis-spectral-fd",
            "hmetis-spectral-ps",
            "hmetis-truenorth"
        ]
        placement_methods : dict[str, Callable[[str, HyperGraph, HardwareModel, int], Result]] = {
            "hilbert-fd": run_hilbert_fd,
            "hilbert-ps": run_hilbert_ps,
            "spectral-fd": run_spectral_fd,
            "spectral-ps": run_spectral_ps,
            "truenorth": run_truenorth
        }
        if options["placement"]:
            print("\n------ experiment setup ------")
            print("Methods to test:")
            prettyPrintIterable(placement_methods.keys(), 3, left_aligned = True)
        
        print("\n---- checking feasibility ----")
        if not hardware.checkSnnFit(snn, already_partitioned = options["placement"], verbose = True):
            print("WARNING: the generated SNN may not fit on the given HW, change either's configuration or the seed.")
        else:
            print("Passed!")
        
        workers : dict[str, Worker] = {}
        temp_result : dict[str, Result] = {}
        for name in methods:
            res = Result(name)
            res.setHw(hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount())
            res.setGraph(snn.nodes, snn.totalConnections(), snn.totalSpikeFrequency())
            part_path = os.path.join(options["reload-map"], name + ".part")
            if not os.path.isfile(part_path):
                print("\n---------------")
                print(f"Skipping method '{name}', since there is not file '{part_path}'...")
                continue
            part = load_list(part_path)
            part_snn = snn.getPartitionsHypergraph(part, squish_hyperedges = True)
            res.setPart(hardware.checkPartitionValidity(snn, part), part_snn.totalSpikeFrequency(), max(part) + 1, hardware.synapticReuse(snn, part))
            plac_path = os.path.join(options["reload-map"], name + ".plac")
            if not options["placement"]:
                if not os.path.isfile(plac_path):
                    print(f"Skipping method '{name}', since there is not file '{plac_path}'...")
                    continue
                plac = load_coords(plac_path)
                res.setPlac(**hardware.getAllMetrics(part_snn, plac))
                res.toFile(options["output"])
                print("\n---------------")
                prettyPrintDict(res.__dict__)
            else:
                method = placement_methods[name.split('-', 1)[-1]]
                workers[name] = Worker(method, name, part_snn, hardware, seed, save)
                if Settings.MULTIPROCESSING:
                    print(f"Process {workers[name].getPid()} started for {name}...")
                temp_result[name] = res
            
        while len(workers) > 0:
            for name, worker in list(workers.items()):
                try:
                    outcome, res = worker.try_get()
                except Exception as e:
                    outcome = True
                    res = Result(name)
                    res.setApproxTime(time.time() - worker._start_time)
                    res.setNote("Failed. Exception: " + str(e))
                if outcome:
                    res.setPart(temp_result[name].part_valid, temp_result[name].part_cost, temp_result[name].part_count, temp_result[name].part_synaptic_reuse)
                    res.setNote("Execution time is for placement only...")
                    res.toFile(options["output"])
                    print("\n---------------")
                    prettyPrintDict(res.__dict__)
                    del workers[name]
        
    except Exception:
        print(traceback.format_exc())

    kill_all_children()

    if options["interactive"]:
        print("\n------ interactive mode ------")
        in_interactive_mode = True
        code.interact(local = globals())
        in_interactive_mode = False