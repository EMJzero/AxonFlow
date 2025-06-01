from typing import TypeVar, Union, Any
from types import FrameType

import traceback
import signal
import code
import time
import sys

from graph_utils import *
from partitioner import *
from placer import *
from model import *
from print import *
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
Searchs and removes flags from sys.argv.
If 'with_value' is False, the return values is either True or False depending on the presence or absence of the option.
If 'with_value' is True, the return value is the value assigned with the option, if present, otherwise it is False if
the option is not present and None if no valid argument was provided.
Optionally, 'value_type' can be used to parse the desired value when 'with_value' is True.
"""
def args_match_and_remove(flag : str, with_value : bool = False, value_type : type[T] = str) -> Union[bool, T]:
    try:
        idx = sys.argv.index(flag)
        sys.argv.pop(idx)
        if with_value:
            try:
                value = value_type(sys.argv[idx])
                sys.argv.pop(idx)
                return value
            except:
                return None
        else:
            return True
    except:
        return False

def parse_options() -> dict[str, Any]:
    options = {
        "help": args_match_and_remove("-h") or args_match_and_remove("--help"),
        "interactive": args_match_and_remove("-i") or args_match_and_remove("--interactive"),
    }
    return options

def help_options() -> None:
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-i --interactive\tOnce exploration has finished, instead of terminating the program, enter Python's interactive mode.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    options = parse_options()

    if options["help"]:
        print("------------ HELP ------------")
        help_options()
        print("------------------------------")
        sys.exit(0)


    # MAIN CODE:
    seed = 79
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
    snn = HyperGraph.generate_random(1024, 16, 4, seed = seed)
    #acyclic_snn = makeAcyclic(snn)

    try:
        print("\n-------- partitioning --------")
        partitioning_multilevel_multistart_refined = partitionGreedyMultilevelRefinedMultistart(snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount(), seed = seed)
        partitioning_greedy = partitionGreedy(snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount())
        partitioning_sequential = partitionSequential(snn, hardware.neurons_per_core, hardware.synapses_per_core, hardware.coresCount())
        part_snn_mmr = snn.getPartitionsHypergraph(partitioning_multilevel_multistart_refined, keep_self_cycles = True)
        part_snn_greedy = snn.getPartitionsHypergraph(partitioning_greedy, keep_self_cycles = True)
        part_snn_seq = snn.getPartitionsHypergraph(partitioning_sequential, keep_self_cycles = True)
        print("Metrics multilevel multistart refined partitioning:")
        prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_multilevel_multistart_refined), 'tot_hyperedges_spike_frequency': part_snn_mmr.totalSpikeFrequency()}, 1)
        print("Metrics greedy partitioning:")
        prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_greedy), 'tot_hyperedges_spike_frequency': part_snn_greedy.totalSpikeFrequency()}, 1)
        print("Metrics sequential partitioning:")
        prettyPrintDict({'valid': hardware.checkPartitionValidity(snn, partitioning_sequential), 'tot_hyperedges_spike_frequency': part_snn_seq.totalSpikeFrequency()}, 1)

        print("\n----------- layout -----------")
        # These are complete approaches, novel or from previous works
        spectral_placement = spectralPlacement(part_snn_mmr.toGraph().toNxGraph(), hardware.coresAlongX(), hardware.coresAlongY())
        topological_order, _ = topologicalOrder(part_snn_seq, break_cycles = True) # this is the full approach from Ouwen Jin's paper.
        hsc_placement = hilbertPlacement(topological_order.nodes, hardware.coresAlongX(), hardware.coresAlongY())
        metrics_spectral = hardware.getAllMetrics(part_snn_mmr, spectral_placement)
        metrics_hsc = hardware.getAllMetrics(topological_order, hsc_placement)
        print("Metrics spectral layout (canon version - multilevel multistart refined partitioning):")
        prettyPrintDict(metrics_spectral, 1)
        print("Metrics HSC layout (canon version - sequential partitioning):")
        prettyPrintDict(metrics_hsc, 1)
        # These are crossbreeds obtained by mixing placement and partitioning algorithms
        spectral_placement_variant = spectralPlacement(part_snn_seq.toGraph().toNxGraph(), hardware.coresAlongX(), hardware.coresAlongY())
        topological_order_variant, _ = topologicalOrder(part_snn_mmr, break_cycles = True)
        hsc_placement_variant = hilbertPlacement(topological_order_variant.nodes, hardware.coresAlongX(), hardware.coresAlongY())
        metrics_spectral_variant = hardware.getAllMetrics(part_snn_seq, spectral_placement_variant)
        metrics_hsc_variant = hardware.getAllMetrics(topological_order_variant, hsc_placement_variant)
        print("Metrics spectral layout (crossbreed - sequential partitioning):")
        prettyPrintDict(metrics_spectral_variant, 1)
        print("Metrics HSC layout (crossbreed - multilevel multistart refined partitioning):")
        prettyPrintDict(metrics_hsc_variant, 1)
    except Exception:
        print(traceback.format_exc())

    if options["interactive"]:
        print("\n------ interactive mode ------")
        in_interactive_mode = True
        code.interact(local=globals())
        in_interactive_mode = False