from typing import TypeVar, Union, Any
from types import FrameType

import traceback
import signal
import code
import time
import sys

from partitioner import *
from placer import *
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
    snn = HyperGraph.generate_random(160, 6, 4, seed = 79)
    partitioning = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4]
    placement = [(x, y) for x in range(4) for y in range(4)]
    spectral_placement = spectralPlacement(snn.toGraph().toNxGraph(), 32, 32)
    hsc_placement = hilbertPlacement(snn.nodes, 32, 32)
    topologycal_order = topologycalOrder(makeAcyclic(snn))
    print("This does nothing, use interactive mode for now...")

    if options["interactive"]:
        print("\n------ interactive mode ------")
        in_interactive_mode = True
        code.interact(local=globals())
        in_interactive_mode = False