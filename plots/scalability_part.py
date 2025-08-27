from typing import TypeVar, Any, Optional
from types import FrameType

import matplotlib.pyplot as plt
import matplotlib
import traceback
import time
import code
import json
import sys
import os

# FIX IMPORTS: add to 'path' the absolute path to the parent directory of this script
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from utils import *

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
        "path": args_match_and_remove(["-p", "--path"], with_value = True),
        "save": args_match_and_remove(["-s", "--save"], with_value = True),
        "quiet": args_match_and_remove(["-q", "--quiet"]),
    }
    return options

def help_options() -> None:
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-i --interactive\tOnce exploration has finished, instead of terminating the program, enter Python's interactive mode.")
    print(("-p, --path <path>\tPath to the '.json' file containing the output of 'scalability_main.py' (possibly ran with the '-p' option)."))
    print(("-s, --save <name>\tSaves the produced plot with the given name, instead of showing it. There is automatic file overwrite prevention.\n"
        "The default extension is '.png', add an extension to <name> to override the file type, supported ones are '.pdf', '.eps', '.svg', '.png'."))
    print("-q, --quiet\t\tDisable verbose logging of optimization functions.")


# MATPLOTLIB SETTINGS:

SUPPORTED_EXTENSIONS = ['.pdf', '.eps', '.svg', '.png']
DPI = 300 #800
SAVE_NOT_SHOW = True

FONTSIZE = 15

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}

matplotlib.rc('font', **font)


# MAIN:

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

    try:
        # Load JSON data
        path = options["path"]
        if not path:
            print(f"No filepath provided, option '-p' or '--path' is mandatory.")
            sys.exit(0)
        if not os.path.exists(path):
            raise Exception(f"The provided path does not exist: {path}")
        elif not os.path.isfile(path):
            raise Exception(f"The provided path is not a file: {path}")
        elif path.split('.')[-1] != "json":
            print(f"WARNING: the '{path}' file does not have the '.json' extension. Are you sure it is a report from 'scalability_main.py'?")
        with open(path, "r") as f:
            data = json.load(f)

        # Organize data
        entries_by_size = defaultdict(dict)  # {size: {technique: entry}}
        techniques = set()
        sizes = set()

        for entry in data:
            name = entry['name']
            try:
                size_str, technique = name.split("-", 1)
                size = int(size_str)
            except ValueError:
                print(f"WARNING: invalid name format \"{name}\"")
                continue

            if "note" in entry:
                print(f"Failed entry '{name}', note content:\n\t{entry['note']}")
                continue

            techniques.add(technique)
            sizes.add(size)
            entries_by_size[size][technique] = entry

            if not entry.get("part_valid", True):
                print(f"WARNING: invalid partitioning for \"{name}\"")

        # Sort sizes
        sorted_sizes = sorted(sizes)
        x_labels = []
        x_indices = list(range(len(sorted_sizes)))

        # Prepare plotting data per technique
        part_costs = defaultdict(list)
        times = defaultdict(list)

        for size in sorted_sizes:
            size_entries = entries_by_size[size]
            nodes = size_entries[next(iter(size_entries))]['graph_nodes']
            edges = size_entries[next(iter(size_entries))]['graph_edges']
            x_labels.append(f"{nodes}\n({edges})")

            for technique in techniques:
                entry = size_entries.get(technique)
                if entry:
                    part_costs[technique].append(entry["part_cost"])
                    times[technique].append(entry["time"])
                else:
                    part_costs[technique].append(None)
                    times[technique].append(None)

        # Optional: normalize w.r.t. the best partitioning
        best_costs = [min([part_costs[technique][i] for technique in techniques if part_costs[technique][i] != None], default = 0) for i in range(len(x_indices))]
        for technique in techniques:
            part_costs[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None else None, zip(part_costs[technique], best_costs)))

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 8), sharex = True)

        # Partitioning cost plot
        for technique in sorted(techniques):
            y = part_costs[technique]
            ax1.plot(x_indices, y, marker = 'o', label = technique)

        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(x_labels, rotation = 45)
        ax1.set_xlabel("Problem size\n(nodes / edges)")
        #ax1.set_yscale('log', base = 10)
        #ax1.set_ylabel("Partition Cost")
        ax1.set_ylabel("Partition Cost (normalized w.r.t. lowest)")
        ax1.set_title("Partition Cost vs Problem Size")
        ax1.legend()
        ax1.grid(True)

        # Time plot
        for technique in sorted(techniques):
            y = times[technique]
            ax2.plot(x_indices, y, marker = 'o', label = technique)

        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(x_labels, rotation = 45)
        ax2.set_xlabel("Problem size\n(nodes / edges)")
        ax2.set_yscale('log', base = 10)
        ax2.set_ylabel("Time [s]")
        ax2.set_title("Execution Time vs Problem Size")
        ax2.legend()
        ax2.grid(True)
        
        # Show the plot
        plt.tight_layout()
        if options["save"]:
            filename = options["save"]
            if not any(filename.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                print("WARNING: the provided filename was missing a valid extension, adding '.png' by default.")
                filename += '.png'
            idx = 1
            while os.path.isfile(filename):
                fn1, fn2 = filename.split('.', 1)
                filename = fn1 + f"_{idx}" + fn2
                idx += 1
            plt.savefig(filename, dpi = DPI)
        else:
            plt.show()
    except Exception:
        print(traceback.format_exc())

    if options["interactive"]:
        print("\n------ interactive mode ------")
        in_interactive_mode = True
        code.interact(local = globals())
        in_interactive_mode = False