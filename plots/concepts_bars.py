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
    result = ""
    for piece in label.split("-"):
        if piece == "sequential": result += "ordered sequential"
        elif label == "unordered": result += "unordered sequential"
        elif label == "edgehiding": result += "edgemap"
        elif piece == "hmetis": result += "hierarchical"
        elif piece == "hehiding": result += "h-edge overlap"
        elif piece == "hilbert": result += "hilbert"
        elif piece == "spectral": result += "spectral"
        elif piece == "truenorth": result += "minimum distance"
        elif piece == "fd": result += "force-directed"
        elif piece == "ps": result += "particle swarm"
        else:
            result += piece
            print("Could not fully rename label:", label, "-> technique note recognized:", piece)
        result += "\n+ "
    return result[:-3]


# MATPLOTLIB SETTINGS:

SUPPORTED_EXTENSIONS = ['.pdf', '.eps', '.svg', '.png']
DPI = 300 #800
SAVE_NOT_SHOW = True

FONTSIZE = 15 # was 13
BAR_WIDTH_PART = 0.15
BAR_WIDTH_PLAC = 0.10

# default font size
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}

matplotlib.rc('font', **font)

# specific element sizes
# autoscalable options: xx-small (0.58x), x-small (0.69x), small (0.83x), medium (1.0x), large (1.2x), x-large (1.44x), xx-large (1.73x)
plt.rcParams.update({
    #'axes.titlesize': 15.6, # default: FONTSIZE*1.2
    #'axes.labelsize': 13, # default: FONTSIZE*1
    'xtick.labelsize': 13, # default: FONTSIZE*1
    'ytick.labelsize': 13, # default: FONTSIZE*1
    #'legend.fontsize': 13, # default: FONTSIZE*1
    #'figure.titlesize': 15.6 # default: FONTSIZE*1.2
})


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
        file_data = []  # will hold tuples: (num_nodes, filename, technique_entries)
        x_labels = []

        # Metrics to collect
        synaptic_reuse_mean : dict[str, list[Optional[float]]] = defaultdict(list)
        synaptic_reuse_geomean : dict[str, list[Optional[float]]] = defaultdict(list)
        connections_locality_mean : dict[str, list[Optional[float]]] = defaultdict(list)
        connections_locality_geomean : dict[str, list[Optional[float]]] = defaultdict(list)
        init_connections_locality_mean : dict[str, list[Optional[float]]] = defaultdict(list)
        init_connections_locality_geomean : dict[str, list[Optional[float]]] = defaultdict(list)
        energy_delay_product : dict[str, list[Optional[float]]] = defaultdict(list)

        # Optional: specify techniques to omit
        #omit_techniques = {"hmetis-truenorth", "hmetis-spectral-ps", "hehiding-hilbert-ps", "hehiding-spectral-ps", "hehiding-truenorth", "sequential-hilbert-ps"}
        omit_techniques = {"sequential-hilbert-ps", "sequential-spectral-ps", "hmetis-hilbert-ps", "hmetis-spectral-ps", "hehiding-hilbert-ps", "hehiding-spectral-ps"}
        # Optional: specify techniques that must be kept
        #must_keep_techniques = {"sequential-truenorth", "hmetis-hilbert-ps", "edgehiding", "unordered"}
        must_keep_techniques = {"sequential-truenorth", "edgehiding", "unordered"}
        # Optional: specify partitioning techniques to omit
        omit_part_techniques = {"setlist"}
        # Optinal: specify placement techniques to omit
        omit_plac_techniques = {"unordered", "edgehiding"}
        # Optional: disable "shades" for initial layout
        no_initial_layout = True
        # Optional: specify techniques to put first in the order, others will follow in descending alphabetical order
        forceful_order = ["sequential", "unordered", "edgehiding"]
        # Optional: divide the connections locality by the partitions count
        norm_by_part_count = False
        # Must: decide x-axis SNNs sort order, options are "nodes", "connections"/"edges"
        x_axis_order = "connections"

        # Read files
        for file in files:
            file_path = os.path.join(path, file)

            with open(file_path, "r") as f:
                print("Parsing:", file_path)
                data : list[dict[str, float]] = json.load(f)

            graph_size = None
            technique_entries = {}

            for entry in data:
                name = entry["name"]
                if name in omit_techniques:
                    print(f"Omitting technique '{name}'...")
                    continue

                if "note" in entry:
                    print(f"Failed entry '{file}' -> '{name}', note content:\n\t{entry['note']}")
                    continue

                technique = name
                techniques.add(technique)
                technique_entries[technique] = entry

                if graph_size is None and "graph_nodes" in entry and "graph_edges" in entry :
                    if x_axis_order == "nodes":
                        graph_size = entry["graph_nodes"]
                    else:
                        graph_size = entry["graph_edges"]

                if not entry.get("part_valid", True):
                    print(f"WARNING: Invalid partitioning for {file} -> {name}")
                if not entry.get("plac_valid", True):
                    print(f"WARNING: Invalid placement for {file} -> {name}")

            if graph_size is not None:
                file_data.append((graph_size, file, technique_entries))
            else:
                print(f"WARNING: Could not determine graph size (edges, nodes) for {file}")

        file_data.sort(key = lambda x : x[0]) # Sort by number of nodes

        # Extract data in increasing graph size order
        max_part_count = []
        for idx, (_, file, technique_entries) in enumerate(file_data):
            label = f"{os.path.splitext(file)[0]}"
            x_labels.append(label)
            max_part_count.append(1)

            for technique in techniques:
                entry = technique_entries.get(technique)
                if entry:
                    synaptic_reuse_mean[technique].append(entry.get("part_synaptic_reuse", None))
                    if isinstance(synaptic_reuse_mean[technique][-1], dict):
                        synaptic_reuse_geomean[technique].append(synaptic_reuse_mean[technique][-1]["geo_mean"])
                        synaptic_reuse_mean[technique][-1] = synaptic_reuse_mean[technique][-1]["ar_mean"]
                    else:
                        synaptic_reuse_geomean[technique].append(synaptic_reuse_mean[technique][-1])
                    connections_locality_mean[technique].append(entry.get("plac_connections_locality", None))
                    if isinstance(connections_locality_mean[technique][-1], dict):
                        connections_locality_geomean[technique].append(connections_locality_mean[technique][-1]["geo_mean_weighted"]/(entry["part_count"] if norm_by_part_count else 1))
                        connections_locality_mean[technique][-1] = connections_locality_mean[technique][-1]["ar_mean_weighted"]/(entry["part_count"] if norm_by_part_count else 1)
                    else:
                        connections_locality_geomean[technique].append(connections_locality_mean[technique][-1])
                    init_connections_locality_mean[technique].append(entry.get("init_connections_locality", None) if not no_initial_layout else None)
                    if isinstance(init_connections_locality_mean[technique][-1], dict):
                        init_connections_locality_geomean[technique].append(init_connections_locality_mean[technique][-1]["geo_mean"]/(entry["part_count"] if norm_by_part_count else 1))
                        init_connections_locality_mean[technique][-1] = init_connections_locality_mean[technique][-1]["ar_mean"]/(entry["part_count"] if norm_by_part_count else 1)
                    else:
                        init_connections_locality_geomean[technique].append(init_connections_locality_mean[technique][-1])
                    en, lat = entry.get("plac_energy", None), entry.get("plac_avg_lat", None)
                    energy_delay_product[technique].append(en * lat if en != None and lat != None else None)
                    if norm_by_part_count and (entry["part_count"] > max_part_count[-1]):
                        max_part_count[-1] = entry["part_count"]
                else:
                    synaptic_reuse_mean[technique].append(None)
                    synaptic_reuse_geomean[technique].append(None)
                    connections_locality_mean[technique].append(None)
                    connections_locality_geomean[technique].append(None)
                    init_connections_locality_mean[technique].append(None)
                    init_connections_locality_geomean[technique].append(None)
                    energy_delay_product[technique].append(None)
        
        x_indices = list(range(len(x_labels)))
        
        # Optional: normalize w.r.t. the best partitioning
        #best_synaptic_reuse_geomean = [max([synaptic_reuse_geomean[technique][i] for technique in techniques if synaptic_reuse_geomean[technique][i] != None], default = 0) for i in range(len(x_indices))]
        #best_connections_locality_geomean = [min([connections_locality_geomean[technique][i] for technique in techniques if connections_locality_geomean[technique][i] != None], default = 0) for i in range(len(x_indices))]
        best_edp = [min([energy_delay_product[technique][i] for technique in techniques if energy_delay_product[technique][i] != None], default = 0) for i in range(len(x_indices))]
        for technique in techniques:
            #synaptic_reuse_mean[technique] = list(map(lambda c : c[0] / c[1] if c[0] else None, zip(synaptic_reuse_mean[technique], best_synaptic_reuse_geomean)))
            #synaptic_reuse_geomean[technique] = list(map(lambda c : c[0] / c[1] if c[0] else None, zip(synaptic_reuse_geomean[technique], best_synaptic_reuse_geomean)))
            #connections_locality_mean[technique] = list(map(lambda c : c[0] / c[1] if c[0] else None, zip(connections_locality_mean[technique], best_connections_locality_geomean)))
            #connections_locality_geomean[technique] = list(map(lambda c : c[0] / c[1] if c[0] else None, zip(connections_locality_geomean[technique], best_connections_locality_geomean)))
            energy_delay_product[technique] = list(map(lambda c : c[0] / c[1] if c[0] else None, zip(energy_delay_product[technique], best_edp)))
            # still normalize them w.r.t. the eventual best for their metric since they will be in the same plot
            #init_connections_locality_mean[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None else None, zip(init_connections_locality_mean[technique], best_connections_locality_geomean)))
            #init_connections_locality_geomean[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None else None, zip(init_connections_locality_geomean[technique], best_connections_locality_geomean)))
            if norm_by_part_count:
                connections_locality_mean[technique] = list(map(lambda c : c[0] * c[1] if c[0] != None else None, zip(connections_locality_mean[technique], max_part_count)))
                connections_locality_geomean[technique] = list(map(lambda c : c[0] * c[1] if c[0] != None else None, zip(connections_locality_geomean[technique], max_part_count)))
                init_connections_locality_mean[technique] = list(map(lambda c : c[0] * c[1] if c[0] != None else None, zip(init_connections_locality_mean[technique], max_part_count)))
                init_connections_locality_geomean[technique] = list(map(lambda c : c[0] * c[1] if c[0] != None else None, zip(init_connections_locality_geomean[technique], max_part_count)))

        # Optional: keep only the best placement by EDP for each partitioning technique
        best_techniques = defaultdict(set) # best_technique[part_tech] -> set of techniques that are the best for at least one experiment size
        for technique in sorted(energy_delay_product.keys()):
            edp = energy_delay_product[technique]
            partitioning_technique = technique.split('-', 1)[0]
            if partitioning_technique in omit_part_techniques or all(e is None for e in edp):
                continue
            # I am not None where someone else is or I am better than them at least once -> keep me!
            if all(any(edp[i] is not None and (energy_delay_product[other_technique][i] is None or energy_delay_product[other_technique][i] > edp[i]) for i in range(len(file_data))) for other_technique in best_techniques[partitioning_technique]):
                best_techniques[partitioning_technique].add(technique)
            for other_technique in list(best_techniques[partitioning_technique]):
                # Someone else is None when I am and never better than me when it is not None -> ditch the other guy!
                if all(edp[i] is None and energy_delay_product[other_technique][i] is None or (edp[i] is not None and (energy_delay_product[other_technique][i] is None or energy_delay_product[other_technique][i] > edp[i])) for i in range(len(file_data))):
                    best_techniques[partitioning_technique].remove(other_technique)
            if not best_techniques[partitioning_technique]:
                best_techniques[partitioning_technique].add(technique)
        pareto_techniques = reduce(lambda s1, s2 : s1 | s2, best_techniques.values()) | must_keep_techniques
        print("Dominated (excluded) techniques:", ', '.join(techniques - pareto_techniques))
        techniques = pareto_techniques
        synaptic_reuse_mean = {k : v for k, v in synaptic_reuse_mean.items() if k in techniques}
        synaptic_reuse_geomean = {k : v for k, v in synaptic_reuse_geomean.items() if k in techniques}
        connections_locality_mean = {k : v for k, v in connections_locality_mean.items() if k in techniques}
        connections_locality_geomean = {k : v for k, v in connections_locality_geomean.items() if k in techniques}
        init_connections_locality_mean = {k : v for k, v in init_connections_locality_mean.items() if k in techniques}
        init_connections_locality_geomean = {k : v for k, v in init_connections_locality_geomean.items() if k in techniques}
        energy_delay_product = {k : v for k, v in energy_delay_product.items() if k in techniques}

        # Replace 'None' with zero
        for technique in techniques:
            synaptic_reuse_mean[technique] = list(map(lambda x : x if x != None else math.nan, synaptic_reuse_mean[technique]))
            synaptic_reuse_geomean[technique] = list(map(lambda x : x if x != None else math.nan, synaptic_reuse_geomean[technique]))
            connections_locality_mean[technique] = list(map(lambda x : x if x != None else math.nan, connections_locality_mean[technique]))
            connections_locality_geomean[technique] = list(map(lambda x : x if x != None else math.nan, connections_locality_geomean[technique]))
            init_connections_locality_mean[technique] = list(map(lambda x : x if x != None else math.nan, init_connections_locality_mean[technique]))
            init_connections_locality_geomean[technique] = list(map(lambda x : x if x != None else math.nan, init_connections_locality_geomean[technique]))
            energy_delay_product[technique] = list(map(lambda x : x if x != None else math.nan, energy_delay_product[technique]))

        # Plotting (note: 25.6 = 2560 pixel)
        fig, (ghost_ax1, ax1, ax2, ghost_ax2) = plt.subplots(1, 4, figsize = (25.6, 7*(1 - 0.075)), sharex = True, tight_layout = True, width_ratios = [1/6, 1/3, 1/3, 1/6])
        ghost_ax1.remove()
        ghost_ax2.remove()

        # Decide the y-axis bounds by ignoring outliers (lower sigma is more brutal)
        def set_bounds(ax : matplotlib.axes.Axes, techniques : list[str], data : Optional[dict[str, float]], shapes : Optional[list[matplotlib.patches.Patch]], bar_width : float, sigma : float = 0.5, margin : float = 0.2):
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
                ax.set_xlim(min_x - bar_width, max_x + 2*bar_width)

        # Sets the y-scale for bar plots to be in percentage
        def format_y_bars(ax : matplotlib.axes.Axes, which : Union[str, list[float]] = "all", dec_digits : int = 1):
            ax.set_yscale("log", base = 10)
            ax.grid(axis = 'y', which = 'both')
            ax.yaxis.set_major_locator(LogLocator(base = 10.0, subs = which, numticks = 10))
            ax.yaxis.set_minor_locator(LogLocator(base = 10.0, subs = which, numticks = 10))
            #formatter = FuncFormatter(lambda v, _: f"{v*100:.0f}%" if v > 0 else "")
            def printer(v, _):
                d = dec_digits
                if v < 0:
                    return ""
                while v * 10**d < 1:
                    d += 1
                return f"{v:.{d}f}"
            formatter = FuncFormatter(printer)
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(formatter)

        # Synpatic reuse plot
        def synpatic_reuse_plot(ax : matplotlib.axes.Axes):
            # Prepare for bar-plot
            index = np.arange(len(x_labels))
            part_techniques = {}
            for technique in techniques:
                part_technique = technique.split('-', 1)[0]
                if part_technique not in part_techniques:
                    part_techniques[part_technique] = technique
            part_techniques_list = list(part_techniques.keys())
            part_techniques_list = sorted(part_techniques_list, reverse = True) # descending order on the word before the first '-', then ascending order as a tiebreak.
            for fo in forceful_order[::-1]:
                if fo in part_techniques_list:
                    part_techniques_list.remove(fo)
                    part_techniques_list.insert(0, fo)
            offset = (len(part_techniques_list) - 1)/2

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
            for technique in part_techniques_list:
                ongoing_linestyle = next(possible_linestyles)
                color = next(ongoing_color)
                style[technique] = {"color": color, "hatch": part_techniques_to_hatch[technique], "edgecolor": "white"}
                line_style[technique] = {"color": color, "marker" : next(possible_markers), "linestyle" : ongoing_linestyle, "markersize" : 8, "alpha" : 1.0}
            shadow_bars_alpha = 0.3
            
            rects = []
            for j, technique in enumerate(part_techniques_list):
                rects += ax.bar(index + (j - offset) * BAR_WIDTH_PART, synaptic_reuse_mean[part_techniques[technique]], BAR_WIDTH_PART, label = None, alpha = shadow_bars_alpha, **style[technique])
                rects += ax.bar(index + (j - offset) * BAR_WIDTH_PART, synaptic_reuse_geomean[part_techniques[technique]], BAR_WIDTH_PART, label = rename_label(technique), alpha = 1.0, **style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel(f"SNN (least → most {x_axis_order})")
            ax.set_ylabel("Partitioning Synaptic Reuse (higher is better)")
            #ax.set_ylabel("Partitioning Synaptic Reuse\n(normalized on highest geo. mean, higher is better)")
            ax.set_title("Reuse across SNNs")
            format_y_bars(ax, which = [1.0, 2.0, 4.0, 6.0, 8.0])
            #set_bounds(ax, list(part_techniques.values()), synaptic_reuse_mean, rects, BAR_WIDTH_PART)
            ax.set_ybound(0.8, 800)

        # Connections locality plot
        def connections_locality_plot(ax : matplotlib.axes.Axes):
            # Prepare for bar-plot
            index = np.arange(len(x_labels))
            plac_techniques = techniques
            for plac_technique in omit_plac_techniques:
                if plac_technique in plac_techniques:
                    plac_techniques.remove(plac_technique)
            #plac_techniques = sorted(techniques, key = lambda s : (''.join(chr(255 - ord(c)) for c in s.split('-', 1)[0]), s.split('-', 1)[1])) # descending order on the word before the first '-', then ascending order as a tiebreak.
            plac_techniques = sorted(techniques, key = lambda s : (''.join(chr(255 - ord(c)) for c in s.split('-', 1)[0]), chr(255 - ord(s.split('-', 1)[1][2])))) # descending order on the word before the first '-', then stupid hack to get truenorth to be first, then hilbert, then spectral.
            offset = (len(plac_techniques) - 1)/2

            # Assign style to partitioning techniques
            possible_colors = [
                    "#6C8EBF", # BLUE
                    "#336699", # DARKER BLUE
                    "#48617A", # DARK BLUE
                    #"#FFB700", # YELLOW # alts: D79B00
                    #"#B38000", # DARK YELLOW
                    #"#FF6978", # PINK
                    #"#A8516E", # DARK PINK
                    "#82B366", # GREEN
                    "#169E1B", # DARK GREEN
                    "#2F762F", # DARKER GREEN
                    "#EB6050", # RED # alts: cc3300, e63900, ec3c00, ff531a, ff3c2d, f03c2d, ea382a, ea3b2e, e7473a, e9493d, e94e3d, eb5847
                    "#CD0A00", # DARK RED
                    "#8E2B25", # DARKER RED
                    "#C2E812", # LIME
                    "#768E0B", # DARK LIME
                ]
            #possible_hatches = cycle(['', '/', '\\', 'x', '.']) #['', '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
            #possible_markers = cycle(['o', 'v', '^', 's', 'p', '*', 'p', 'X', 'D'])
            possible_markers = cycle(['o', 'v', '^', 's', 'p', 'H', 'D', 'P', 'X'])
            possible_linestyles = cycle(['-', ':', '--', '-.'])
            #part_techniques_to_hatch = defaultdict(lambda : '', hehiding = '/')
            part_techniques_to_hatch = defaultdict(lambda : '')#, spectral = '/')
            style = {}
            line_style = {}
            #prev_part_technique, ongoing_color, ongoing_hatch, ongoing_linestyle = None, cycle(possible_colors), next(possible_hatches), next(possible_linestyles)
            prev_part_technique, ongoing_color, ongoing_hatch, ongoing_linestyle = None, cycle(possible_colors), '', next(possible_linestyles)
            for technique in plac_techniques:
                part_technique, plac_technique = technique.split('-', 1)
                init_plac_technique, plac_ref_technique = plac_technique.split('-', 1) if '-' in plac_technique else ('', plac_technique)
                if prev_part_technique != part_technique:
                    prev_part_technique = part_technique
                    #ongoing_color = cycle(possible_colors)
                    #ongoing_hatch = next(possible_hatches)
                    ongoing_linestyle = next(possible_linestyles)
                color = next(ongoing_color)
                #style[technique] = {"color": color, "hatch": part_techniques_to_hatch[part_technique], "edgecolor": "white"}
                style[technique] = {"color": color, "hatch": part_techniques_to_hatch[init_plac_technique], "edgecolor": "white"}
                line_style[technique] = {"color": color, "marker" : next(possible_markers), "linestyle" : ongoing_linestyle, "markersize" : 8}
            shadow_bars_alpha = 0.3
            
            rects = []
            for j, technique in enumerate(plac_techniques):
                #rects += ax.bar(index + (j - offset) * BAR_WIDTH_PLAC, init_connections_locality[technique], BAR_WIDTH_PLAC, label = None, alpha = shadow_bars_alpha, **style[technique])
                rects += ax.bar(index + (j - offset) * BAR_WIDTH_PLAC, connections_locality_mean[technique], BAR_WIDTH_PLAC, label = None, alpha = shadow_bars_alpha, **style[technique])
                rects += ax.bar(index + (j - offset) * BAR_WIDTH_PLAC, connections_locality_geomean[technique], BAR_WIDTH_PLAC, label = rename_label(technique), alpha = 1.0, **style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_xlabel(f"SNN (least → most {x_axis_order})")
            ax.set_ylabel("Placement Connections Locality (lower is better)")
            #ax.set_ylabel("Placement Connections Locality\n(normalized on lowest geo. mean, lower is better)")
            ax.set_title("Locality across SNNs")
            #ax.set_ylim(0.8, 5)
            ax.set_ylim(1.0, 100.0)
            format_y_bars(ax, which = [1.0, 2.0, 4.0, 6.0, 8.0])
            #set_bounds(ax, plac_techniques, connections_locality_mean, rects, BAR_WIDTH_PLAC)
        
        synpatic_reuse_plot(ax1)
        connections_locality_plot(ax2)
        
        # Setup legends AFTER the tight layout
        plt.tight_layout(rect = [0, 0, 1, 1]) # TODO: comment me or use "gridspec" for a better scaling of plots!
        custom_handles = [
            #matplotlib.patches.Rectangle((0, 0,), 0, 0, color = "black", edgecolor = "white", label = "full: geometric mean"),
            #matplotlib.patches.Rectangle((0, 0,), 0, 0, color = "black", edgecolor = "white", label = "shade: average mean", alpha = 0.3)
            matplotlib.lines.Line2D([0], [0], marker = "s", color = "gray", linestyle = "", markersize = 10, label = "full: geometric mean"),
            matplotlib.lines.Line2D([0], [0], marker = "s", color = "gray", linestyle = "", markersize = 10, alpha = 0.3, label = "shade: arithmetic mean")
        ]
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles1 = custom_handles + handles1
        labels1 = [h.get_label() for h in handles1]
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles2 = custom_handles + handles2
        labels2 = [h.get_label() for h in handles2]
        ax1.legend(handles1, labels1, ncol = 1, loc = "center", bbox_to_anchor = (-0.33 - 0.1, 0.5))
        ax2.legend(handles2, labels2, ncol = 1, loc = "center", bbox_to_anchor = (1.33, 0.4)) # was (1.33, 0.5)
        
        # Show the plot
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