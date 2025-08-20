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
        "output": args_match_and_remove(["-o", "--output"], with_value = True),
        "partitioning": args_match_and_remove(["-p", "--partitioning"]),
        "fraction": args_match_and_remove(["-f", "--fraction"], with_value = True, value_type = float),
        "quiet": args_match_and_remove(["-q", "--quiet"]),
    }
    return options

def help_options() -> None:
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-i, --interactive\tOnce exploration has finished, instead of terminating the program, enter Python's interactive mode.")
    print(("-l, --load <?path>\tLoads a true SNN graph instead of randomly generating one. If omitted, the default path is './snn_models/simple_cnn'.\n"
           "\t\t\tThe given path is concatenated with '_0.npz', '_input.npz', '.graphml', these are the three files expected to be found."))
    print("-s, --save <path>\tSaves the used SNN graph efficiently in 'path' after having built it. Recommended extension: '.hgr'.")
    print("-r, --reload <path>\tReloads a previously saved (--save) SNN graph from 'path'. This takes priority on --load.")
    print("-o, --output <file>\tName of the '.json' file where to write results.")
    print("-p, --partitioning\tOnly runs the partitioning algorithms part, skips placement.")
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
        print("┌─┐   AXON   ┌──┐   FLOW   ┌─┐")
        print("│ └──┐    ┌──┘  └──┐    ┌──┘ │")
        print("└────┴────┴────────┴────┴────┘\n")

    Settings.CORE_TIMEOUT = 3600*100

    if os.name != "posix":
        print("WARNING: this program was developed for a UNIX-like environment, expect bugs (especially with signals and multiprocessing) on other systems.")

    if not options["output"]:
        options["output"] = "targeted_results.json"
        print(f"WARNING: missing '-o' option, defaulting to '{options["output"]}'.")
    elif not options["output"].endswith(".json"):
        options["output"] += ".json"
        print(f"WARNING: the output file was missing the '.json' extension, it has updated to '{options["output"]}'.")

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
            nodes_count = 1024 #512
            nodes_per_edge_mean, nodes_per_edge_variation = 6, 4 #8, 4
            snn = HyperGraph.generate_random(nodes_count, nodes_per_edge_mean, nodes_per_edge_variation, seed = seed)
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
        #hardware = HardwareModel(
        #    neurons_per_core = 32, #8
        #    synapses_per_core = 128, #32
        #    cores_per_chip_x = 64,
        #    cores_per_chip_y = 64,
        #    chips_per_system_x = 1,
        #    chips_per_system_y = 1,
        #    energy_per_routing = 1.0,
        #    energy_per_wire = 0.1,
        #    latency_per_routing = 1.0,
        #    latency_per_wire = 0.1
        #)
        hardware = loihi_jin_84
        print((f"Neurons per core: {hardware.neurons_per_core}\tSynapses per core: {hardware.synapses_per_core}\n"
               f"Cores along x: {hardware.cores_per_chip_x}\tCores along y: {hardware.cores_per_chip_y}\n"
               f"Chips along x: {hardware.chips_per_system_x}\tChips along y: {hardware.chips_per_system_y}\n"
               f"Routing energy: {hardware.energy_per_routing}\tWire energy: {hardware.energy_per_wire}\n"
               f"Routing latency: {hardware.latency_per_routing}\tWire latency: {hardware.latency_per_wire}"))
        
        print("\n------ experiment setup ------")
        methods : dict[str, Callable[[str, HyperGraph, HardwareModel, int], Result]] = {
            #"sequential-topo-hilbert-fd": run_sequential_topo_hilbert_fd,
            "sequential-hilbert-fd": run_sequential_hilbert_fd,
            "sequential-hilbert-ps": run_sequential_hilbert_ps,
            "sequential-spectral-fd": run_sequential_spectral_fd,
            "sequential-spectral-ps": run_sequential_spectral_ps,
            "sequential-truenorth": run_sequential_truenorth,
            "hehiding-hilbert-fd": run_hehiding_hilbert_fd,
            "hehiding-hilbert-ps": run_hehiding_hilbert_ps,
            "hehiding-spectral-fd": run_hehiding_spectral_fd,
            "hehiding-spectral-ps": run_hehiding_spectral_ps,
            "hehiding-truenorth": run_hehiding_truenorth,
            #"swap-particleswarm": run_swap_particleswarm,
            #"multistart-truenorth": run_multistart_truenorth,
            #"multistart-spectral-fd": run_multistart_spectral_fd,
            "setlist-hilbert-fd": run_setlist_hilbert_fd,
            "setlist-hilbert-ps": run_setlist_hilbert_ps,
            "setlist-spectral-fd": run_setlist_spectral_fd,
            "setlist-spectral-ps": run_setlist_spectral_ps,
            "setlist-truenorth": run_setlist_truenorth,
            "hmetis-hilbert-fd": run_hmetis_hilbert_fd,
            "hmetis-hilbert-ps": run_hmetis_hilbert_ps,
            "hmetis-spectral-fd": run_hmetis_spectral_fd,
            "hmetis-spectral-ps": run_hmetis_spectral_ps,
            "hmetis-truenorth": run_hmetis_truenorth
        } if not options["partitioning"] else {
            #"unordered-sequential": run_unordered_sequential,
            "sequential": run_sequential,
            "edgehiding": run_edgehiding,
            "hehiding": run_hehiding,
            #"swap": run_swap,
            #"multistart": run_multistart,
            "setlist": run_setlist,
            "hmetis": run_hmetis,
        }
        print("Methods to test:")
        prettyPrintIterable(methods.keys(), 3, left_aligned = True)
        
        print("\n---- checking feasibility ----")
        if not hardware.checkSnnFit(snn, verbose = True):
            print("WARNING: the generated SNN may not fit on the given HW, change either's configuration or the seed.")
        else:
            print("Passed!")
        
        workers : dict[str, Worker] = {}
        for name, method in methods.items():
            workers[name] = Worker(method, name, snn, hardware, seed)
            if Settings.MULTIPROCESSING:
                print(f"Process {workers[name].getPid()} started for {name}...")
            
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