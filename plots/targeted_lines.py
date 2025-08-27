from typing import TypeVar, Any, Optional
from types import FrameType

import matplotlib.pyplot as plt
from functools import reduce
from itertools import cycle
import numpy as np
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
                if os.path.isdir(os.path.join(path, f)):
                    print("Skipped folder:", f)
                else:
                    print("Skipped file:", f)
        files.sort()
        techniques = set()
        file_data = []  # will hold tuples: (num_nodes, filename, technique_entries)
        x_labels = []
        x_indices = list(range(len(files)))

        # Metrics to collect
        energy : dict[str, list[Optional[float]]] = defaultdict(list)
        latency : dict[str, list[Optional[float]]] = defaultdict(list)
        congestion : dict[str, list[Optional[float]]] = defaultdict(list)
        times : dict[str, list[Optional[float]]] = defaultdict(list)

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
                    congestion[technique].append(entry.get("plac_avg_cong", None))
                    times[technique].append(entry.get("time", None))
                else:
                    energy[technique].append(None)
                    latency[technique].append(None)
                    congestion[technique].append(None)
                    times[technique].append(None)

        # Optional: normalize w.r.t. the best partitioning
        energy_delay_product : dict[str, list[Optional[float]]] = {}
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

        # Optional: keep only the best placement by EDP for each partitioning technique
        omit_techniques = {"setlist"}
        best_techniques = defaultdict(set) # best_technique[part_tech] -> set of techniques that are the best for at least one experiment size
        for technique in sorted(energy_delay_product.keys()):
            edp = energy_delay_product[technique]
            partitioning_technique = technique.split('-', 1)[0]
            if partitioning_technique in omit_techniques:
                continue
            for i in range(len(file_data)):
                if all(edp[i] is None or energy_delay_product[other_techinque][i] is None or energy_delay_product[other_techinque][i] > edp[i] for other_techinque in best_techniques[partitioning_technique]) and not all(e is None for e in edp):
                    best_techniques[partitioning_technique].add(technique)
        techniques = reduce(lambda s1, s2 : s1 | s2, best_techniques.values())
        energy = {k : v for k, v in energy.items() if k in techniques}
        latency = {k : v for k, v in latency.items() if k in techniques}
        congestion = {k : v for k, v in congestion.items() if k in techniques}
        times = {k : v for k, v in times.items() if k in techniques}
        energy_delay_product = {k : v for k, v in energy_delay_product.items() if k in techniques}

        # Plotting
        #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 6), sharex = True, tight_layout = True)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12, 12), sharex = True, tight_layout = True)

        # Assign markers to partitioning techniques
        possible_markers = cycle(['o', 'v', '^', 's', 'p', '*', 'p', 'X', 'D'])
        possible_linestyles = cycle(['-', ':', '--', '-.'])
        style = {}
        for technique in sorted(techniques):
            part_technique = technique.split('-', 1)[0]
            if part_technique not in style:
                style[part_technique] = {"marker": next(possible_markers), "linestyle": next(possible_linestyles), "markersize": 8, "alpha": 0.9}

        # Decide the y-axis bounds by ignoring outliers (lower sigma is more brutal)
        def set_bounds(ax : matplotlib.axes.Axes, sigma : float = 0.5, margin : float = 0.2):
            lines = ax.lines
            data_by_x = {}
            for line in lines:
                xdata, ydata = line.get_xdata(), line.get_ydata()
                for xi, yi in zip(xdata, ydata):
                    if yi is None or np.isnan(yi):
                        continue
                    data_by_x.setdefault(xi, []).append(yi)

            valid_y = []
            for xi, values in data_by_x.items():
                values = np.array(values, dtype = float)
                if len(values) == 1:
                    valid_y.extend(values)
                else:
                    mean = values.mean()
                    std = values.std()
                    keep = values[np.abs(values - mean) <= sigma * std]
                    valid_y.extend(keep)

            valid_y = np.array(valid_y)
            ymin, ymax = valid_y.min(), valid_y.max()
            yrange = ymax - ymin
            if yrange == 0:
                ymin, ymax = ymin - 1, ymax + 1
            else:
                ymin -= margin * yrange
                ymax += margin * yrange
            if ax.get_yscale() == "log":
                ymin = 1.0 - margin #max(ymin, np.min(valid_y[valid_y > 0]) * 0.9)
                ymax = max(ymax, ymin * 1.1)
            ax.set_ylim(ymin, ymax)

        # Energy plot
        def energy_plot(ax : matplotlib.axes.Axes):
            for technique in sorted(techniques):
                ax.plot(x_indices, energy[technique], **style[technique.split('-', 1)[0]], label = technique)
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel("Problem Size (nodes)")
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Placement Energy (normalized w.r.t. lowest)")
            ax.set_title("Energy vs Problem Size")
            #ax.legend()
            ax.grid(True)
            set_bounds(ax)

        # Latency plot
        def latency_plot(ax : matplotlib.axes.Axes):
            for technique in sorted(techniques):
                ax.plot(x_indices, latency[technique], **style[technique.split('-', 1)[0]], label = technique)
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel("Problem Size (nodes)")
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Avg Latency (normalized w.r.t. lowest)")
            ax.set_title("Latency vs Problem Size")
            #ax.legend()
            ax.grid(True)
            set_bounds(ax)

        # Congestion plot
        def congestion_plot(ax : matplotlib.axes.Axes):
            for technique in sorted(techniques):
                ax.plot(x_indices, congestion[technique], **style[technique.split('-', 1)[0]], label = technique)
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel("Problem Size (nodes)")
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Avg. congestion (normalized w.r.t. lowest)")
            ax.set_title("Congestion vs Problem Size")
            #ax.legend()
            ax.grid(True)
            set_bounds(ax)

        # TODO: does it even make sense to look at this? It is not like, the longer you run, the more you consume here...
        # Energy x Delay Product plot
        def edp_plot(ax : matplotlib.axes.Axes):
            for technique in sorted(techniques):
                ax.plot(x_indices, energy_delay_product[technique], **style[technique.split('-', 1)[0]], label = technique)
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel("Problem Size (nodes)")
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Placement Energy x Latency (normalized w.r.t. lowest)")
            ax.set_title("Energy-Delay Product vs Problem Size")
            #ax.legend()
            ax.grid(True)
            set_bounds(ax)

        # Time plot
        def time_plot(ax : matplotlib.axes.Axes):
            for technique in sorted(techniques):
                ax.plot(x_indices, times[technique], **style[technique.split('-', 1)[0]], label = technique)
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
        
        max_legend_rows = 3
        # HP: all axis have the same entries!
        handles, labels = ax1.get_legend_handles_labels()
        ncols = math.ceil(len(labels) / max_legend_rows)
        fig.legend(handles, labels, loc = 'lower center', ncol = ncols)
        
        # Show the plot
        plt.tight_layout(rect = [0, 0.065, 1, 1]) # TODO: comment me or use "gridspec" for a better scaling of plots!
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