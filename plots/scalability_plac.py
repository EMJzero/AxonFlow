from typing import TypeVar, Any, Optional
from functools import reduce
from types import FrameType

import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib
import traceback
import time
import code
import json
import time
import math
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
    print(("-p, --path <path>\tPath to the '.json' file containing the output of 'scalability_main.py'."))
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
        entries_by_size = defaultdict(dict)
        techniques = set()
        sizes = set()

        for entry in data:
            name = entry['name']
            try:
                size_str, *technique_parts = name.split("-")
                size = int(size_str)
                technique = "-".join(technique_parts)
            except ValueError:
                print(f"Invalid name format: {name}")
                continue

            if "note" in entry:
                print(f"Failed entry '{name}', note content:\n\t{entry['note']}")
                continue

            techniques.add(technique)
            sizes.add(size)
            entries_by_size[size][technique] = entry

            if not entry.get("part_valid", True):
                print(f"WARNING: Invalid partitioning for {name}")
            if not entry.get("plac_valid", True):
                print(f"WARNING: Invalid placement for {name}")

        # Prepare X axis
        sorted_sizes = sorted(sizes)
        x_labels = [str(s) for s in sorted_sizes]
        x_indices = list(range(len(sorted_sizes)))

        # Prepare metric containers
        energy : dict[str, list[float]] = defaultdict(list)
        latency : dict[str, list[float]] = defaultdict(list)
        congestion : dict[str, list[float]] = defaultdict(list)
        times : dict[str, list[float]] = defaultdict(list)

        for size in sorted_sizes:
            size_entries = entries_by_size[size]
            for technique in techniques:
                entry = size_entries.get(technique)
                if entry:
                    energy[technique].append(entry.get("plac_energy", None))
                    latency[technique].append(entry.get("plac_avg_lat", None))
                    congestion[technique].append(entry.get("plac_avg_cong", None))
                    times[technique].append(entry.get("time", None))
                else:
                    energy[technique].append(None)
                    latency[technique].append(None)
                    congestion[technique].append(None)
                    times[technique].append(None)

        # Optional: normalize w.r.t. the best partitioning
        energy_delay_product : dict[str, list[float]] = {}
        best_energy_delay_product = [min([energy[technique][i]*latency[technique][i] for technique in techniques if energy[technique][i] != None and latency[technique][i] != None], default = 0) for i in range(len(x_indices))]
        for technique in techniques:
            energy_delay_product[technique] = list(map(lambda c : (c[0] * c[1]) / c[2] if c[0] != None and c[1] != None else None, zip(energy[technique], latency[technique], best_energy_delay_product)))
        best_energy = [min([energy[technique][i] for technique in techniques if energy[technique][i] != None], default = 0) for i in range(len(x_indices))]
        for technique in techniques:
            energy[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None else None, zip(energy[technique], best_energy)))
        best_latency = [min([latency[technique][i] for technique in techniques if latency[technique][i] != None], default = 0) for i in range(len(x_indices))]
        for technique in techniques:
            latency[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None else None, zip(latency[technique], best_latency)))
        best_congestion = [min([congestion[technique][i] for technique in techniques if congestion[technique][i] != None], default = 0) for i in range(len(x_indices))]
        for technique in techniques:
            congestion[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None else None, zip(congestion[technique], best_congestion)))

        # Optional: keep only the best placement for each partitioning technique
        best_techniques = defaultdict(set) # best_technique[part_tech] -> set of techniques that are the best for at least one experiment size
        for technique, edp in energy_delay_product.items():
            partitioning_technique = technique.split('-', 1)[0]
            for i in range(len(sizes)):
                if all(edp[i] is not None and energy_delay_product[other_techinque][i] > edp[i] for other_techinque in best_techniques[partitioning_technique]):
                    best_techniques[partitioning_technique].add(technique)
        techniques = reduce(lambda s1, s2 : s1 | s2, best_techniques.values())
        energy = {k : v for k, v in energy.items() if k in techniques}
        latency = {k : v for k, v in latency.items() if k in techniques}
        congestion = {k : v for k, v in congestion.items() if k in techniques}
        times = {k : v for k, v in times.items() if k in techniques}
        energy_delay_product = {k : v for k, v in energy_delay_product.items() if k in techniques}

        # Plotting
        #fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (18, 6), sharex = True)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12, 12), sharex = True, tight_layout = True)

        # Energy plot
        def energy_plot(ax : matplotlib.axes.Axes):
            for technique in sorted(techniques):
                ax.plot(x_indices, energy[technique], marker = 'o', label = technique)
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel("Problem Size (nodes)")
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Placement Energy (normalized w.r.t. lowest)")
            ax.set_title("Energy vs Problem Size")
            #ax.legend()
            ax.grid(True)

        # Latency plot
        def latency_plot(ax : matplotlib.axes.Axes):
            for technique in sorted(techniques):
                ax.plot(x_indices, latency[technique], marker = 'o', label = technique)
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel("Problem Size (nodes)")
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Avg Latency (normalized w.r.t. lowest)")
            ax.set_title("Latency vs Problem Size")
            #ax.legend()
            ax.grid(True)

        # Congestion plot
        def congestion_plot(ax : matplotlib.axes.Axes):
            for technique in sorted(techniques):
                ax.plot(x_indices, congestion[technique], marker = 'o', label = technique)
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel("Problem Size (nodes)")
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Avg. congestion (normalized w.r.t. lowest)")
            ax.set_title("Congestion vs Problem Size")
            #ax.legend()
            ax.grid(True)

        # TODO: does it even make sense to look at this? It is not like, the longer you run, the more you consume here...
        # Energy x Delay Product plot
        def edp_plot(ax : matplotlib.axes.Axes):
            for technique in sorted(techniques):
                ax.plot(x_indices, energy_delay_product[technique], marker = 'o', label = technique)
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel("Problem Size (nodes)")
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Placement Energy x Latency (normalized w.r.t. lowest)")
            ax.set_title("Energy-Delay Product vs Problem Size")
            #ax.legend()
            ax.grid(True)

        # Time plot
        def time_plot(ax : matplotlib.axes.Axes):
            for technique in sorted(techniques):
                ax.plot(x_indices, times[technique], marker = 'o', label = technique)
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel("Problem Size (nodes)")
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Time [s]")
            ax.set_title("Execution Time vs Problem Size")
            #ax.legend()
            ax.grid(True)
        
        energy_plot(ax1)
        latency_plot(ax2)
        edp_plot(ax3)
        time_plot(ax4)
        
        max_legend_rows = 2
        # HP: all axis have the same entries!
        handles, labels = ax1.get_legend_handles_labels()
        ncols = math.ceil(len(labels) / max_legend_rows)
        fig.legend(handles, labels, loc = 'lower center', ncol = ncols)
        
        # Show the plot
        plt.tight_layout(rect = [0, 0.05, 1, 1]) # TODO: comment me or use "gridspec" for a better scaling of plots!
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