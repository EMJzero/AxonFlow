from typing import TypeVar, Any, Optional
from types import FrameType

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import traceback
import time
import code
import json
import time
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
        "dir": args_match_and_remove(["-d", "--dir"], with_value = True),
        "save": args_match_and_remove(["-s", "--save"], with_value = True),
        "quiet": args_match_and_remove(["-q", "--quiet"]),
    }
    return options

def help_options() -> None:
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-i --interactive\tOnce exploration has finished, instead of terminating the program, enter Python's interactive mode.")
    print(("-d, --dir <path>\tPath to the directory (folder) containing one or more '.json' files, each being the output of a run of 'targeted_main.py'."
        "Only files immediately inside the directory (no nested directories) and with the '.json' extension will be considered."))
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
        path = options["dir"]
        if not path:
            print(f"No directory path provided, option '-d' or '--dir' is mandatory.")
            sys.exit(0)
        if not os.path.exists(path):
            raise Exception(f"The provided path does not exist: {path}")
        elif not os.path.isdir(path):
            raise Exception(f"The provided path is not a directory: {path}")

        # Organize data
        files = []
        for f in os.listdir(path):
            if f.endswith(".json"):
                files.append(f)
                print("Added file:", f)
            else:
                print("Skipped file:", f)
        files.sort()
        techniques = set()
        file_data = []  # will hold tuples: (num_nodes, filename, technique_entries)
        x_labels = []
        x_indices = list(range(len(files)))

        # Metrics to collect
        energy = defaultdict(list)
        latency = defaultdict(list)
        times = defaultdict(list)

        # Read files
        for file in files:
            file_path = os.path.join(path, file)

            with open(file_path, "r") as f:
                data = json.load(f)

            graph_nodes = None
            technique_entries = {}

            for entry in data:
                name = entry["name"]

                if "note" in entry:
                    print(f"Failed entry '{file}' -> '{name}', note content:\n\t{entry['note']}")
                    continue

                technique = name
                techniques.add(technique)
                technique_entries[technique] = entry

                if graph_nodes is None and "graph_nodes" in entry:
                    graph_nodes = entry["graph_nodes"]

                if not entry.get("part_valid", True):
                    print(f"WARNING: Invalid partitioning for {file} -> {name}")
                if not entry.get("plac_valid", True):
                    print(f"WARNING: Invalid placement for {file} -> {name}")

            if graph_nodes is not None:
                file_data.append((graph_nodes, file, technique_entries))
            else:
                print(f"WARNING: Could not determine graph_nodes for {file}")

        file_data.sort(key = lambda x : x[0])  # Sort by number of nodes

        # Extract data in increasing graph size order
        for idx, (graph_nodes, file, technique_entries) in enumerate(file_data):
            label = f"{os.path.splitext(file)[0]}\n({graph_nodes})"
            x_labels.append(label)

            for technique in techniques:
                entry = technique_entries.get(technique)
                if entry:
                    energy[technique].append(entry.get("plac_energy", None))
                    latency[technique].append(entry.get("plac_avg_lat", None))
                    times[technique].append(entry.get("time", None))
                else:
                    energy[technique].append(None)
                    latency[technique].append(None)
                    times[technique].append(None)

        # Optional: normalize w.r.t. the best partitioning
        best_energy = [min([energy[technique][i] for technique in techniques if energy[technique][i] != None], default = 0) for i in range(len(x_indices))]
        for technique in techniques:
            energy[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None else None, zip(energy[technique], best_energy)))
        best_latency = [min([latency[technique][i] for technique in techniques if latency[technique][i] != None], default = 0) for i in range(len(x_indices))]
        for technique in techniques:
            latency[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None else None, zip(latency[technique], best_latency)))

        # Plotting
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), sharex=True)

        # Energy plot
        for technique in sorted(techniques):
            ax1.plot(x_indices, energy[technique], marker = 'o', label = technique)
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(x_labels, rotation = 45, ha = "right")
        ax1.set_xlabel("Input Graph (nodes)")
        ax1.set_ylabel("Placement Energy (normalized w.r.t. lowest)")
        ax1.set_title("Energy vs Input Graph")
        ax1.legend()
        ax1.grid(True)

        # Latency plot
        for technique in sorted(techniques):
            ax2.plot(x_indices, latency[technique], marker = 'o', label = technique)
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(x_labels, rotation = 45, ha = "right")
        ax2.set_xlabel("Input Graph (nodes)")
        ax2.set_ylabel("Avg Latency (normalized w.r.t. lowest)")
        ax2.set_title("Latency vs Input Graph")
        ax2.legend()
        ax2.grid(True)

        # Time plot
        for technique in sorted(techniques):
            ax3.plot(x_indices, times[technique], marker = 'o', label = technique)
        ax3.set_xticks(x_indices)
        ax3.set_xticklabels(x_labels, rotation = 45, ha = "right")
        ax3.set_xlabel("Input Graph (nodes)")
        ax3.set_ylabel("Execution Time (s)")
        ax3.set_title("Time vs Input Graph")
        ax3.legend()
        ax3.grid(True)
        
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