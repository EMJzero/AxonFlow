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
        "output": args_match_and_remove(["-o", "--output"], with_value = True),
        "partitioning": args_match_and_remove(["-p", "--partitioning"]),
        "quiet": args_match_and_remove(["-q", "--quiet"]),
    }
    return options

def help_options() -> None:
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-i --interactive\tOnce exploration has finished, instead of terminating the program, enter Python's interactive mode.")
    print("-o --output <file>\tName of the '.json' file where to write results.")
    print("-p --partitioning\tOnly runs the partitioning algorithms part, skips placement.")
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
        print("---Axon---  /\\__/\\  ----------")
        print("---------- ( o .o ) ----------")
        print("----------  >  ^ <  ---Flow---\n")

    Settings.CORE_TIMEOUT = 3600*8

    if os.name != "posix":
        print("WARNING: this program was developed for a UNIX-like environment, expect bugs (especially with signals and multiprocessing) on other systems.")

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
        print("seed:", seed)
        
        sizes : dict[dict[str, int]] = {
            # LOGIC:
            # - nodes_count: *2
            # - nodes_per_edge_mean: +2, +2, +4, +4, +8, +8, ... then *8
            # - nodes_per_edge_variation: +1, +1, +2, +2, +4, +4, ... then *4
            # - neurons_per_core: +16, +16, +16, +32, +32, +64, +64, ...
            # - ALTERNATIVE neurons_per_core: +16, +16, +16, +32, +32, +32, +64, +64, +64, ...
            # - synapses_per_core: +128, +128, +256, +256, +512, +512 ... then *2 after 1024*32 (included), then *4 after 1024*128, ...
            "256":
               {"nodes_count": 256, "nodes_per_edge_mean": 4*8, "nodes_per_edge_variation": 2*4,
               "neurons_per_core": 16, "synapses_per_core" : 256, "cores_per_chip_1d": 64},
            "512":
                {"nodes_count": 512, "nodes_per_edge_mean": 6*8, "nodes_per_edge_variation": 3*4,
                "neurons_per_core": 24, "synapses_per_core" : 384, "cores_per_chip_1d": 64},
            "1024":
                {"nodes_count": 1024, "nodes_per_edge_mean": 8*8, "nodes_per_edge_variation": 4*4,
                "neurons_per_core": 32, "synapses_per_core" : 512, "cores_per_chip_1d": 64},
            f"{1024*2}":
                {"nodes_count": 1024*2, "nodes_per_edge_mean": 12*8, "nodes_per_edge_variation": 6*4,
                "neurons_per_core": 64, "synapses_per_core" : 768, "cores_per_chip_1d": 64},
            f"{1024*4}":
                {"nodes_count": 1024*4, "nodes_per_edge_mean": 16*8, "nodes_per_edge_variation": 8*4,
                "neurons_per_core": 96, "synapses_per_core" : 1024, "cores_per_chip_1d": 64},
            f"{1024*8}":
                {"nodes_count": 1024*8, "nodes_per_edge_mean": 24*8, "nodes_per_edge_variation": 12*4,
                "neurons_per_core": 128, "synapses_per_core" : 1536, "cores_per_chip_1d": 64},
            f"{1024*16}":
                {"nodes_count": 1024*16, "nodes_per_edge_mean": 32*8, "nodes_per_edge_variation": 16*4,
                "neurons_per_core": 192, "synapses_per_core" : 2048, "cores_per_chip_1d": 64},
            f"{1024*32}":
                {"nodes_count": 1024*32, "nodes_per_edge_mean": 48*8, "nodes_per_edge_variation": 24*4,
                "neurons_per_core": 256, "synapses_per_core" : 3072*2, "cores_per_chip_1d": 64},
            f"{1024*64}":
                {"nodes_count": 1024*64, "nodes_per_edge_mean": 64*8, "nodes_per_edge_variation": 32*4,
                "neurons_per_core": 384, "synapses_per_core" : 4096*2, "cores_per_chip_1d": 64},
            f"{1024*128}":
                {"nodes_count": 1024*128, "nodes_per_edge_mean": 96*8, "nodes_per_edge_variation": 48*4,
                "neurons_per_core": 512, "synapses_per_core" : 6144*4, "cores_per_chip_1d": 64},
            f"{1024*256}":
                {"nodes_count": 1024*256, "nodes_per_edge_mean": 128*8, "nodes_per_edge_variation": 64*4,
                "neurons_per_core": 768, "synapses_per_core" : 8192*4, "cores_per_chip_1d": 96},
            f"{1024*512}":
                {"nodes_count": 1024*512, "nodes_per_edge_mean": 192*8, "nodes_per_edge_variation": 96*4,
                "neurons_per_core": 1024, "synapses_per_core" : 12288*8, "cores_per_chip_1d": 96}
        }
        methods : dict[str, Callable[[str, HyperGraph, HardwareModel, int], Result]] = {
            # TODO: add missing cases, hilbert with fd, spectral with ps, setlist and hmetis with truenorth, etc...
            #"sequential-topo-hilbert-fd": run_sequential_topo_hilbert_fd,
            "sequential-hilbert-fd": run_sequential_hilbert_fd,
            "sequential-truenorth": run_sequential_truenorth,
            #"swap-particleswarm": run_swap_particleswarm,
            #"multistart-truenorth": run_multistart_truenorth,
            #"multistart-spectral-fd": run_multistart_spectral_fd,
            "setlist-spectral-fd": run_setlist_spectral_fd,
            "setlist-hilbert-ps": run_setlist_hilbert_ps,
            "hmetis-hilbert-ps": run_hmetis_hilbert_ps,
            "hmetis-spectral-fd": run_hmetis_spectral_fd
        } if not options["partitioning"] else {
            "unordered-sequential": run_unordered_sequential,
            "sequential": run_sequential,
            #"swap": run_swap,
            #"multistart": run_multistart,
            "setlist": run_setlist,
            "hmetis": run_hmetis,
        }
        
        print("\n------ experiment setup ------")
        print("Random graph sizes to test:")
        prettyPrintIterable(sizes.keys(), 6, left_aligned = True)
        print("Methods to test:")
        prettyPrintIterable(methods.keys(), 3, left_aligned = True)
        
        for experiment, size in sizes.items():
            print("\n------------------------------")
            print("Preparing configuration:")
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
            hypergraph = HyperGraph.generate_random(size["nodes_count"], size["nodes_per_edge_mean"], size["nodes_per_edge_variation"], spike_frequency_range = (0.1, 1000), seed = seed)
            #acyclic_snn = makeAcyclic(snn)
            if not hardware.checkSnnFit(hypergraph, verbose = True):
                print(f"WARNING: the generated SNN of experiment '{experiment}' may not fit on the given HW, change either's configuration or the seed.")
            
            workers : dict[str, Worker] = {}
            for name, method in methods.items():
                full_name = experiment + '-' + name
                workers[full_name] = Worker(method, full_name, hypergraph, hardware, seed)
                if Settings.MULTIPROCESSING:
                    print(f"Process {workers[full_name].getPid()} started for {full_name}...")
            
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