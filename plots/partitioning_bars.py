from typing import TypeVar, Any, Optional
from types import FrameType

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
import matplotlib.legend_handler
import matplotlib.patches
import matplotlib.axes
import matplotlib

from collections import defaultdict
from functools import reduce
from itertools import cycle
import numpy as np
import traceback
import time
import math
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
        code.interact(local = globals())
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

def rename_label(label : str) -> str:
    if label == "sequential": return "ordered sequential"
    if label == "unordered": return "unordered sequential"
    if label == "edgehiding": return "edgemap"
    if label == "hmetis": return "hierarchical"
    if label == "hehiding": return "h-edge overlap"
    print("Could not rename label:", label)
    return label


# MATPLOTLIB SETTINGS:

SUPPORTED_EXTENSIONS = ['.pdf', '.eps', '.svg', '.png']
DPI = 300 #800
SAVE_NOT_SHOW = True

FONTSIZE = 13
BAR_WIDTH = 0.15 #0.18

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
            if f.startswith("_"):
                print("Ignored file (starts with '_'):", f)
            elif f.endswith(".json"):
                files.append(f)
                print("Added file:", f)
            else:
                if os.path.isdir(os.path.join(path, f)):
                    print("Skipped folder:", f)
                else:
                    print("Skipped file:", f)
        files.sort()
        techniques = set()
        file_data = [] # will hold tuples: (num_nodes, filename, technique_entries)
        x_labels = []
        x_indices = list(range(len(files)))

        # Metrics to collect
        connectivity : dict[str, list[Optional[float]]] = defaultdict(list)
        part_times : dict[str, list[Optional[float]]] = defaultdict(list)
        part_count : dict[str, list[Optional[float]]] = defaultdict(list)

        # Optional: specify partitioning techniques to omit
        omit_part_techniques = {"setlist"}
        # Optional: specify techniques to put first in the order, others will follow in descending alphabetical order
        forceful_order = ["sequential", "unordered", "edgehiding"]

        # Read files
        for file in files:
            file_path = os.path.join(path, file)

            with open(file_path, "r") as f:
                print("Parsing:", file_path)
                data : list[dict[str, float]] = json.load(f)

            graph_nodes = None
            technique_entries = {}

            for entry in data:
                name = entry["name"]
                technique = name.split("-", 1)[0]
                if name in technique_entries:
                    print(f"Already seen technique '{technique}', skipping '{name}'...")
                    continue

                if "note" in entry:
                    print(f"Failed entry '{file}' -> '{name}', note content:\n\t{entry['note']}")
                    continue

                techniques.add(technique)
                technique_entries[technique] = entry

                if graph_nodes is None and "graph_nodes" in entry:
                    graph_nodes = entry["graph_nodes"]
                    #graph_nodes = entry["graph_edges"]

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
            label = f"{os.path.splitext(file)[0]}" # .replace('_', '-') #\n({graph_nodes})"
            x_labels.append(label)

            for technique in techniques:
                entry = technique_entries.get(technique)
                if entry:
                    connectivity[technique].append(entry.get("part_cost", None))
                    part_times[technique].append(entry.get("part_time", None))
                    part_count[technique].append(entry.get("part_count", None))
                else:
                    connectivity[technique].append(None)
                    part_times[technique].append(None)
                    part_count[technique].append(None)
        
        # Optional: normalize w.r.t. the best partitioning
        best_connectivity = [min([connectivity[technique][i] for technique in techniques if connectivity[technique][i] != None], default = 0) for i in range(len(x_indices))]
        for technique in techniques:
            connectivity[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None else None, zip(connectivity[technique], best_connectivity)))

        print("Omitted techniques:", ', '.join(omit_part_techniques))
        techniques = techniques - omit_part_techniques
        connectivity = {k : v for k, v in connectivity.items() if k in techniques}
        part_times = {k : v for k, v in part_times.items() if k in techniques}
        part_count = {k : v for k, v in part_count.items() if k in techniques}

        # Replace 'None' with zero
        for technique in techniques:
            connectivity[technique] = list(map(lambda x : x if x != None else math.nan, connectivity[technique]))
            part_times[technique] = list(map(lambda x : x if x != None else math.nan, part_times[technique]))
            part_count[technique] = list(map(lambda x : x if x != None else math.nan, part_count[technique]))

        # Prepare for bar-plot
        index = np.arange(len(x_labels))
        offset = (len(techniques) - 1)/2
        techniques = sorted(techniques, reverse = True) # descending order on the word before the first '-', then ascending order as a tiebreak.
        for fo in forceful_order[::-1]:
            if fo in techniques:
                techniques.remove(fo)
                techniques.insert(0, fo)

        # Plotting (note: 25.6 = 2560 pixel)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (25.6, 7), sharex = True, tight_layout = True)

        # Assign style to partitioning techniques
        possible_colors = [
                "#6C8EBF", # BLUE
                #"#48617A", # DARK-BLUE
                "#336699", # DARKER-BLUE
                #"#FFB700", # YELLOW # alts: D79B00
                "#B38000", # DARK YELLOW
                #"#FF6978", # PINK
                #"#A8516E", # DARK PINK
                #"#82B366", # GREEN
                "#169E1B", # DARK-GREEN
                #"#2F762F", # DARKER-GREEN
                #"#EB6050", # RED # alts: cc3300, e63900, ec3c00, ff531a, ff3c2d, f03c2d, ea382a, ea3b2e, e7473a, e9493d, e94e3d, eb5847
                "#8E2B25", # DARKER RED
                "#C2E812", # LIME
                #"#768E0B", # DARK LIME
            ]
        #possible_hatches = cycle(['', '/', '\\', 'x', '.']) #['', '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
        possible_markers = cycle(['o', 'v', '^', 's', 'p', '*', 'p', 'X', 'D'])
        possible_linestyles = cycle(['-', ':', '--', '-.'])
        part_techniques_to_hatch = defaultdict(lambda : '', hehiding = '/')
        style = {}
        line_style = {}
        #prev_part_technique, ongoing_color, ongoing_hatch, ongoing_linestyle = None, cycle(possible_colors), next(possible_hatches), next(possible_linestyles)
        prev_part_technique, ongoing_color, ongoing_hatch, ongoing_linestyle = None, cycle(possible_colors), '', next(possible_linestyles)
        for technique in techniques:
            ongoing_linestyle = next(possible_linestyles)
            color = next(ongoing_color)
            style[technique] = {"color": color, "hatch": part_techniques_to_hatch[technique], "edgecolor": "white", "alpha": 1.0}
            line_style[technique] = {"color": color, "marker" : next(possible_markers), "linestyle" : ongoing_linestyle, "markersize" : 8, "alpha" : 1.0}

        # Decide the y-axis bounds by ignoring outliers (lower sigma is more brutal)
        def set_bounds(ax : matplotlib.axes.Axes, data : Optional[dict[str, float]], shapes : Optional[list[matplotlib.patches.Patch]], sigma : float = 0.5, margin : float = 0.2):
            if data:
                data_by_x = defaultdict(list)
                for technique in techniques:
                    for xi, value in zip(x_labels, data[technique]):
                        if value is None or np.isnan(value):
                            continue
                        data_by_x[xi].append(value)

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
                if len(valid_y) > 0:
                    ymin, ymax = valid_y.min(), valid_y.max()
                    yrange = ymax - ymin
                    if yrange == 0:
                        ymin, ymax = ymin - 1, ymax + 1
                    else:
                        ymin -= margin * yrange
                        ymax += margin * yrange
                    if ax.get_yscale() == "log":
                        ymin = 1.0 - margin / 2 #max(ymin, np.min(valid_y[valid_y > 0]) * 0.9)
                        ymax = max(ymax, ymin * 1.1)
                    ax.set_ylim(ymin, ymax)
            
            if shapes:
                min_x, max_x = math.inf, 0
                for shape in shapes:
                    min_x = min(min_x, shape.get_x())
                    max_x = max(max_x, shape.get_x())
                ax.set_xlim(min_x - BAR_WIDTH, max_x + 2*BAR_WIDTH)

        # Sets the y-scale for bar plots to be in percentage
        def format_y_bars(ax : matplotlib.axes.Axes):
            ax.set_yscale("log", base = 10)
            ax.grid(axis = 'y', which = 'both')
            ax.yaxis.set_major_locator(LogLocator(base = 10.0, subs = "all", numticks = 10))
            ax.yaxis.set_minor_locator(LogLocator(base = 10.0, subs = [1.0, 2.0, 5.0], numticks = 10))
            #formatter = FuncFormatter(lambda v, _: f"{v*100:.0f}%" if v > 0 else "")
            formatter = FuncFormatter(lambda v, _: f"{v:.1f}" if v > 0 else "")
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(formatter)

        # Partitioned Hypergraph Connectivity plot
        def conn_plot(ax : matplotlib.axes.Axes):
            rects = []
            for j, technique in enumerate(techniques):
                rects += ax.bar(index + (j - offset) * BAR_WIDTH, connectivity[technique], BAR_WIDTH, label = technique, **style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel("SNN (least → most nodes)")
            ax.set_ylabel("Partitioning Connectivity (normalized w.r.t. lowest)")
            ax.set_title("Connectivity vs Problem Size")
            #ax.legend()
            format_y_bars(ax)
            set_bounds(ax, connectivity, rects)

        # Partitions Count plot
        def count_plot(ax : matplotlib.axes.Axes):
            rects = []
            for j, technique in enumerate(techniques):
                rects += ax.bar(index + (j - offset) * BAR_WIDTH, part_count[technique], BAR_WIDTH, label = technique, **style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel("SNN (least → most nodes)")
            ax.set_ylabel("Partitions Count")
            ax.set_title("Partitions Count vs Problem Size")
            #ax.legend()
            ax.set_yscale("log", base = 10)
            ax.grid(axis = 'y', which = 'both')
            set_bounds(ax, part_count, rects)

        # Partitioning Time plot
        def part_time_plot(ax : matplotlib.axes.Axes):
            for technique in techniques:
                ax.plot(x_indices, part_times[technique], label = technique, **line_style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel("SNN (least → most nodes)")
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Time [s]")
            ax.set_title("Partitioning Time vs Problem Size")
            #ax.legend()
            ax.grid(True)
        
        conn_plot(ax1)
        count_plot(ax2)
        part_time_plot(ax3)
            
        max_legend_rows = 1
        # HP: all axis have the same entries!
        handles, labels = ax1.get_legend_handles_labels()
        # Rename labels
        labels = list(map(rename_label, labels))
        # UNLESS: you use lines for time, instead of bars
        handles_lines, _ = ax3.get_legend_handles_labels()
        combined_handles = list(zip(handles, handles_lines))
        ncols = math.ceil(len(labels) / max_legend_rows)
        fig.legend(combined_handles, labels, loc = 'lower center', ncol = ncols, handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(ndivide = None)}, handlelength = 5.0) # handlelength = 4.0
        
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