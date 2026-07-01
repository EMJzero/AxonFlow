from typing import TypeVar, Any, Optional
from types import FrameType

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import matplotlib.legend_handler
import matplotlib.patches
import matplotlib.axes
import matplotlib

from collections import defaultdict
from functools import reduce
from itertools import cycle
import numpy as np
import traceback
import random
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
        result += " + "
    return result[:-3]


# MATPLOTLIB SETTINGS:

SUPPORTED_EXTENSIONS = ['.pdf', '.eps', '.svg', '.png']
DPI = 300 #800
SAVE_NOT_SHOW = True

FONTSIZE = 17 #14 # was 13
BAR_WIDTH = 0.09
GROUP_SPACING = 1.35
GROUP_PADDING = 0.16
DOT_SIZE_MODEL = 70
DOT_SIZE_SIM = 45
DOT_LINE_WIDTH = 1.0
OVERFLOW_ARROW_FRACTION = 0.08
OVERFLOW_ARROW_SIZE = 8

random.seed(42)

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
        x_indices = list(range(len(files)))

        # Metrics to collect
        energy : dict[str, list[Optional[float]]] = defaultdict(list)
        latency : dict[str, list[Optional[float]]] = defaultdict(list)
        sim_energy : dict[str, list[Optional[float]]] = defaultdict(list)
        sim_latency : dict[str, list[Optional[float]]] = defaultdict(list)
        congestion : dict[str, list[Optional[float]]] = defaultdict(list)
        init_energy : dict[str, list[Optional[float]]] = defaultdict(list)
        init_latency : dict[str, list[Optional[float]]] = defaultdict(list)

        # Optional: specify techniques to omit
        #omit_techniques = {"sequential-hilbert-ps", "hmetis-truenorth", "hmetis-spectral-ps", "hehiding-spectral-ps", "hehiding-truenorth", "edgehiding", "unordered-sequential", "unordered"}
        omit_techniques = {
            "sequential-hilbert-ps",
            "sequential-spectral-ps",
            "sequential-spectral-fd",
            "sequential-truenorth",
            "hmetis-hilbert-ps",
            "hmetis-spectral-ps",
            "hmetis-spectral-fd",
            "hmetis-truenorth",
            "hehiding-hilbert-ps",
            "hehiding-spectral-ps",
            "hehiding-hilbert-fd",
            "hehiding-truenorth",
            "edgehiding",
            "unordered-sequential",
            "unordered"
        }
        # Optional: specify techniques that must be kept
        #must_keep_techniques = {"sequential-truenorth", "hmetis-hilbert-ps"}
        must_keep_techniques = set()
        # Optional: specify partitioning techniques to omit
        omit_part_techniques = {"setlist"}
        # Optional: disable "shades" for initial layout
        no_initial_layout = False
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
        for idx, (_, file, technique_entries) in enumerate(file_data):
            label = f"{os.path.splitext(file)[0]}"
            x_labels.append(label)

            for technique in techniques:
                entry = technique_entries.get(technique)
                if entry:
                    energy[technique].append(entry.get("plac_energy", None))
                    latency[technique].append(entry.get("plac_avg_lat", None))
                    sim_energy[technique].append(entry.get("sim_energy", None))
                    sim_latency[technique].append(entry.get("sim_latency", None))
                    congestion[technique].append(entry.get("plac_avg_cong", None))
                    init_energy[technique].append(entry.get("init_energy", None) if not no_initial_layout else None)
                    init_latency[technique].append(entry.get("init_avg_lat", None) if not no_initial_layout else None)
                else:
                    energy[technique].append(None)
                    latency[technique].append(None)
                    sim_energy[technique].append(None)
                    sim_latency[technique].append(None)
                    congestion[technique].append(None)
                    init_energy[technique].append(None)
                    init_latency[technique].append(None)

        # Optional: compute EDP w.r.t. the best partitioning for techniques selection
        energy_delay_product : dict[str, list[Optional[float]]] = {}
        best_energy_delay_product = [min([energy[technique][i]*latency[technique][i] for technique in techniques if energy[technique][i] != None and latency[technique][i] != None], default = 0) for i in range(len(x_indices))]
        init_energy_delay_product : dict[str, list[Optional[float]]] = {}
        for technique in techniques:
            energy_delay_product[technique] = list(map(lambda c : (c[0] * c[1]) / c[2] if c[0] != None and c[1] != None and c[2] != 0 else None, zip(energy[technique], latency[technique], best_energy_delay_product)))
            init_energy_delay_product[technique] = list(map(lambda c : (c[0] * c[1]) / c[2] if c[0] != None and c[1] != None and c[2] != 0 else None, zip(init_energy[technique], init_latency[technique], best_energy_delay_product)))

        # Optional: keep only the best placement by EDP for each partitioning technique
        #best_techniques = defaultdict(set) # best_technique[part_tech] -> set of techniques that are the best for at least one experiment size
        #for technique in sorted(energy_delay_product.keys()):
        #    edp = energy_delay_product[technique]
        #    partitioning_technique = technique.split('-', 1)[0]
        #    if partitioning_technique in omit_part_techniques or all(e is None for e in edp):
        #        continue
        #    # I am not None where someone else is or I am better than them at least once -> keep me!
        #    if all(any(edp[i] is not None and (energy_delay_product[other_technique][i] is None or energy_delay_product[other_technique][i] > edp[i]) for i in range(len(file_data))) for other_technique in best_techniques[partitioning_technique]):
        #        best_techniques[partitioning_technique].add(technique)
        #    for other_technique in list(best_techniques[partitioning_technique]):
        #        # Someone else is None when I am and never better than me when it is not None -> ditch the other guy!
        #        if all(edp[i] is None and energy_delay_product[other_technique][i] is None or (edp[i] is not None and (energy_delay_product[other_technique][i] is None or energy_delay_product[other_technique][i] > edp[i])) for i in range(len(file_data))):
        #            best_techniques[partitioning_technique].remove(other_technique)
        #    if not best_techniques[partitioning_technique]:
        #        best_techniques[partitioning_technique].add(technique)
        #pareto_techniques = reduce(lambda s1, s2 : s1 | s2, best_techniques.values()) | must_keep_techniques
        #print("Dominated (excluded) techniques:", ', '.join(techniques - pareto_techniques))
        #techniques = pareto_techniques
        #energy = {k : v for k, v in energy.items() if k in techniques}
        #latency = {k : v for k, v in latency.items() if k in techniques}
        #sim_energy = {k : v for k, v in sim_energy.items() if k in techniques}
        #sim_latency = {k : v for k, v in sim_latency.items() if k in techniques}
        #energy_delay_product = {k : v for k, v in energy_delay_product.items() if k in techniques}
        #init_energy = {k : v for k, v in init_energy.items() if k in techniques}
        #init_latency = {k : v for k, v in init_latency.items() if k in techniques}
        #init_energy_delay_product = {k : v for k, v in init_energy_delay_product.items() if k in techniques}

        # Keep raw data for validation statistics
        raw_energy = {k : list(v) for k, v in energy.items()}
        raw_latency = {k : list(v) for k, v in latency.items()}
        raw_sim_energy = {k : list(v) for k, v in sim_energy.items()}
        raw_sim_latency = {k : list(v) for k, v in sim_latency.items()}
        raw_congestion = {k : list(v) for k, v in congestion.items()}

        # Normalize w.r.t. the best simulated result for each SNN
        best_sim_energy = [min([sim_energy[technique][i] for technique in techniques if sim_energy[technique][i] != None], default = 0) for i in range(len(x_indices))]
        best_sim_latency = [min([sim_latency[technique][i] for technique in techniques if sim_latency[technique][i] != None], default = 0) for i in range(len(x_indices))]
        best_congestion = [min([congestion[technique][i] for technique in techniques if congestion[technique][i] != None], default = 0) for i in range(len(x_indices))]
        for technique in techniques:
            energy[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None and c[1] != 0 else None, zip(energy[technique], best_sim_energy)))
            latency[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None and c[1] != 0 else None, zip(latency[technique], best_sim_latency)))
            sim_energy[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None and c[1] != 0 else None, zip(sim_energy[technique], best_sim_energy)))
            sim_latency[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None and c[1] != 0 else None, zip(sim_latency[technique], best_sim_latency)))
            congestion[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None else None, zip(congestion[technique], best_congestion)))

        # Replace 'None' with zero
        for technique in techniques:
            energy[technique] = list(map(lambda x : x if x != None else math.nan, energy[technique]))
            latency[technique] = list(map(lambda x : x if x != None else math.nan, latency[technique]))
            #sim_energy[technique] = list(map(lambda x : x if x != None else math.nan, sim_energy[technique]))
            #sim_latency[technique] = list(map(lambda x : x if x != None else math.nan, sim_latency[technique]))
            congestion[technique] = list(map(lambda x : x if x != None else math.nan, congestion[technique]))
            energy_delay_product[technique] = list(map(lambda x : x if x != None else math.nan, energy_delay_product[technique]))
            init_energy[technique] = list(map(lambda x : x if x != None else math.nan, init_energy[technique]))
            init_latency[technique] = list(map(lambda x : x if x != None else math.nan, init_latency[technique]))
            init_energy_delay_product[technique] = list(map(lambda x : x if x != None else math.nan, init_energy_delay_product[technique]))

        # Prepare for dot-plot
        index = np.arange(len(x_labels)) * GROUP_SPACING
        offset = (len(techniques) - 1)/2
        #techniques = sorted(techniques, key = lambda s : (''.join(chr(255 - ord(c)) for c in s.split('-', 1)[0]), s.split('-', 1)[1])) # descending order on the word before the first '-', then ascending order as a tiebreak.
        techniques = sorted(techniques, key = lambda s : (''.join(chr(255 - ord(c)) for c in s.split('-', 1)[0]), chr(255 - ord(s.split('-', 1)[1][2])))) # descending order on the word before the first '-', then stupid hack to get truenorth to be first, then hilbert, then spectral.

        # Plotting (note: 25.6 = 2560 pixel)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2,
            figsize = (18, 9.1), #(18, 7) #(18, 6)
            gridspec_kw = {"height_ratios": [1.0, 0.65]},
            tight_layout = True
        )

        # Assign style to partitioning techniques
        possible_colors = [
                #"#6C8EBF", # BLUE
                "#336699", # DARKER BLUE
                #"#48617A", # DARK BLUE
                #"#FFB700", # YELLOW # alts: D79B00
                #"#B38000", # DARK YELLOW
                #"#FF6978", # PINK
                #"#A8516E", # DARK PINK
                #"#82B366", # GREEN
                "#169E1B", # DARK GREEN
                #"#2F762F", # DARKER GREEN
                #"#EB6050", # RED # alts: cc3300, e63900, ec3c00, ff531a, ff3c2d, f03c2d, ea382a, ea3b2e, e7473a, e9493d, e94e3d, eb5847
                #"#CD0A00", # DARKER RED
                "#8E2B25", # DARK RED
                "#C2E812", # LIME
                "#768E0B", # DARK LIME
            ]
        #part_techniques_to_hatch = defaultdict(lambda : '', hehiding = '/')
        part_techniques_to_hatch = defaultdict(lambda : '')#, spectral = '/')
        style = {}
        #prev_part_technique, ongoing_color, ongoing_hatch = None, cycle(possible_colors), next(possible_hatches)
        prev_part_technique, ongoing_color, ongoing_hatch = None, cycle(possible_colors), ''
        for technique in techniques:
            part_technique, plac_technique = technique.split('-', 1)
            init_plac_technique, plac_ref_technique = plac_technique.split('-', 1) if '-' in plac_technique else ('', plac_technique)
            if prev_part_technique != part_technique:
                prev_part_technique = part_technique
                #ongoing_color = cycle(possible_colors)
                #ongoing_hatch = next(possible_hatches)
            color = next(ongoing_color)
            #style[technique] = {"color": color, "hatch": part_techniques_to_hatch[part_technique], "edgecolor": "white"}
            style[technique] = {"color": color, "hatch": part_techniques_to_hatch[init_plac_technique], "edgecolor": "white"}
        shadow_bars_alpha = 0.3

        # Decide the y-axis bounds by ignoring outliers (lower sigma is more brutal)
        def set_bounds(ax : matplotlib.axes.Axes, data : Optional[dict[str, float] | list[dict[str, float]]], shapes : Optional[list[matplotlib.patches.Patch]], sigma : float = 0.5, margin : float = 0.2):
            if data:
                data = data if isinstance(data, list) else [data]
                data_by_x = defaultdict(list)
                for data_item in data:
                    for technique in techniques:
                        for xi, value in zip(x_labels, data_item[technique]):
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
                        valid_y = valid_y[valid_y > 0]
                        if len(valid_y) > 0:
                            ymin = max(0.001, min(ymin, valid_y.min() * (1.0 - margin / 2)))
                            ymax = max(ymax, 1.0 + margin / 2)
                    ax.set_ylim(ymin, ymax)
            
            if shapes:
                min_x, max_x = math.inf, 0
                for shape in shapes:
                    min_x = min(min_x, shape.get_x())
                    max_x = max(max_x, shape.get_x())
                ax.set_xlim(min_x - BAR_WIDTH, max_x + 2*BAR_WIDTH)
            else:
                half_group_width = offset * BAR_WIDTH + GROUP_PADDING
                ax.set_xlim(index[0] - half_group_width, index[-1] + half_group_width)

        # Sets the y-scale for dot plots to be in ratio to best simulated
        def format_y_bars(ax : matplotlib.axes.Axes, ymin : float, ymax : float):
            ax.set_yscale("log", base = 10)
            ax.set_ylim(ymin, ymax)
            ax.grid(axis = 'y', which = 'major')
            ax.yaxis.set_major_locator(FixedLocator(np.arange(math.ceil(ymin * 10), math.floor(ymax * 10) + 1) / 10))
            ax.yaxis.set_minor_locator(NullLocator())
            #formatter = FuncFormatter(lambda v, _: f"{v*100:.0f}%" if v > 0 else "")
            formatter = FuncFormatter(lambda v, _: f"{v:.1f}" if v > 0 else "")
            ax.yaxis.set_major_formatter(formatter)

        def add_snn_lanes(ax : matplotlib.axes.Axes):
            half_group_width = offset * BAR_WIDTH + GROUP_PADDING
            for i, xi in enumerate(index):
                #if i % 2 == 0:
                ax.axvspan(
                    xi - half_group_width,
                    xi + half_group_width,
                    color = "black",
                    alpha = 0.07, #0.035
                    linewidth = 0,
                    zorder = 0
                )
                ax.axvline(
                    xi,
                    color = "black",
                    alpha = 0.08,
                    linewidth = 0.8,
                    zorder = 0
                )
            for left, right in zip(index[:-1], index[1:]):
                ax.axvline(
                    (left + right) / 2,
                    color = "black",
                    alpha = 0.18,
                    linewidth = 0.9,
                    zorder = 0
                )

        def draw_overflow_arrows(ax : matplotlib.axes.Axes, model : dict[str, list[float]], sim : dict[str, list[float]]):
            ymin, ymax = ax.get_ylim()
            if ax.get_yscale() == "log":
                arrow_start = 10 ** (math.log10(ymax) - OVERFLOW_ARROW_FRACTION * (math.log10(ymax) - math.log10(ymin)))
            else:
                arrow_start = ymax - OVERFLOW_ARROW_FRACTION * (ymax - ymin)

            for j, technique in enumerate(techniques):
                color = style[technique]["color"]
                x = index + (j - offset) * BAR_WIDTH
                for xi, model_value, sim_value in zip(x, model[technique], sim[technique]):
                    if not np.isnan(model_value) and not np.isnan(sim_value) and model_value > ymax and sim_value > ymax:
                        ax.annotate(
                            "",
                            xy = (xi, ymax),
                            xytext = (xi, arrow_start),
                            arrowprops = {
                                "arrowstyle": "-|>",
                                "color": color,
                                "linewidth": 1.0,
                                "mutation_scale": OVERFLOW_ARROW_SIZE,
                                "shrinkA": 0,
                                "shrinkB": 0,
                            },
                            annotation_clip = False,
                            zorder = 5
                        )

        # Energy plot
        def energy_plot(ax : matplotlib.axes.Axes):
            y_lim = 2.5
            add_snn_lanes(ax)
            for j, technique in enumerate(techniques):
                color = style[technique]["color"]
                x = index + (j - offset) * BAR_WIDTH
                for xi, model_value, sim_value in zip(x, energy[technique], sim_energy[technique]):
                    if not np.isnan(model_value) and not np.isnan(sim_value):
                        ax.plot([xi, xi], [model_value, sim_value], color = color, alpha = 0.6, linewidth = DOT_LINE_WIDTH, zorder = 1)
                ax.scatter(x, energy[technique], s = DOT_SIZE_MODEL, marker = 'o', facecolors = 'white', edgecolors = color, linewidths = 1.4, label = technique, zorder = 3)
                ax.scatter(x, sim_energy[technique], s = DOT_SIZE_SIM, marker = 'o', facecolors = color, edgecolors = color, linewidths = 0.8, zorder = 4)
            ax.axhline(y = 1, color = 'black', alpha = 0.45, linestyle='--', linewidth = 1.0)
            ax.set_xticks(index)
            ax.set_xticklabels(x_labels, rotation = 25)
            ax.set_xlabel(f"SNN (least → most {x_axis_order})")
            ax.set_ylabel("Energy (ratio to best simulated)")
            ax.set_title("Simulated and Estimated Energy across SNNs")
            #ax.legend()
            format_y_bars(ax, 0.9, y_lim)
            #set_bounds(ax, [energy, sim_energy], None, sigma = 1.0)
            draw_overflow_arrows(ax, energy, sim_energy)

        # Latency plot
        def latency_plot(ax : matplotlib.axes.Axes):
            y_lim = 1.6
            add_snn_lanes(ax)
            for j, technique in enumerate(techniques):
                color = style[technique]["color"]
                x = index + (j - offset) * BAR_WIDTH
                for xi, model_value, sim_value in zip(x, latency[technique], sim_latency[technique]):
                    if not np.isnan(model_value) and not np.isnan(sim_value):
                        ax.plot([xi, xi], [model_value, sim_value], color = color, alpha = 0.6, linewidth = DOT_LINE_WIDTH, zorder = 1)
                ax.scatter(x, latency[technique], s = DOT_SIZE_MODEL, marker = 'o', facecolors = 'white', edgecolors = color, linewidths = 1.4, label = technique, zorder = 3)
                ax.scatter(x, sim_latency[technique], s = DOT_SIZE_SIM, marker = 'o', facecolors = color, edgecolors = color, linewidths = 0.8, zorder = 4)
            ax.axhline(y = 1, color = 'black', alpha = 0.45, linestyle='--', linewidth = 1.0)
            ax.set_xticks(index)
            ax.set_xticklabels(x_labels, rotation = 25)
            ax.set_xlabel(f"SNN (least → most {x_axis_order})")
            ax.set_ylabel("Avg. Latency (ratio to best simulated)")
            ax.set_title("Simulated and Estimated Latency across SNNs")
            #ax.legend()
            format_y_bars(ax, 0.7, y_lim)
            #set_bounds(ax, [latency, sim_latency], None, sigma = 1.0)
            draw_overflow_arrows(ax, latency, sim_latency)

        # Latency-model error versus congestion plot
        def latency_congestion_correlation_plot(ax : matplotlib.axes.Axes):
            all_congestion = []
            all_latency_ratio = []
            max_latency_ratio_point = None

            for technique in techniques:
                color = style[technique]["color"]
                technique_congestion = []
                technique_latency_ratio = []

                for congestion_value, model_value, sim_value in zip(raw_congestion[technique], raw_latency[technique], raw_sim_latency[technique]):
                    if congestion_value is None or model_value is None or sim_value is None or model_value == 0:
                        continue
                    if not np.isfinite(congestion_value) or not np.isfinite(model_value) or not np.isfinite(sim_value):
                        continue

                    latency_ratio = sim_value / model_value
                    if congestion_value <= 0 or latency_ratio <= 0:
                        continue
                    technique_congestion.append(congestion_value)
                    technique_latency_ratio.append(latency_ratio)
                    if max_latency_ratio_point is None or latency_ratio > max_latency_ratio_point[1]:
                        max_latency_ratio_point = (congestion_value, latency_ratio, color)

                if technique_congestion:
                    ax.scatter(
                        technique_congestion,
                        technique_latency_ratio,
                        s = DOT_SIZE_SIM,
                        marker = 'o',
                        facecolors = color,
                        edgecolors = color,
                        linewidths = 0.8,
                        alpha = 0.75,
                        zorder = 3
                    )
                    all_congestion.extend(technique_congestion)
                    all_latency_ratio.extend(technique_latency_ratio)

            if len(all_congestion) >= 2:
                x = np.array(all_congestion, dtype = float)
                y = np.array(all_latency_ratio, dtype = float)

                if np.std(x) > 0 and np.std(y) > 0:
                    log_x = np.log10(x)
                    log_y = np.log10(y)
                    pearson_r = np.corrcoef(log_x, log_y)[0, 1]
                    slope, intercept = np.polyfit(log_x, log_y, 1)
                    fit_x = np.logspace(np.min(log_x), np.max(log_x), 100)
                    fit_y = 10 ** (slope * np.log10(fit_x) + intercept)
                    ax.plot(fit_x, fit_y, color = 'white', linewidth = 4.0, alpha = 0.95, zorder = 4)
                    ax.plot(fit_x, fit_y, color = 'black', linewidth = 2.0, alpha = 0.95, zorder = 5)
                    ax.text(
                        0.05,
                        0.90,
                        f"Pearson $\\rho$ = {pearson_r:.2f}",
                        transform = ax.transAxes,
                        va = 'top',
                        bbox = dict(boxstyle = 'round,pad=0.25', facecolor = 'white', edgecolor = 'black', alpha = 0.75)
                    )
                    print(f"latency/congestion correlation: log-log Pearson $\\rho$ = {pearson_r:.3f}, slope = {slope:.6g}")

            #ax.axhline(y = 1, color = 'black', alpha = 0.45, linestyle = '--', linewidth = 1.0)
            ax.set_xscale("log", base = 10)
            ax.set_yscale("log", base = 10)
            ymin, ymax = ax.get_ylim()
            yticks = np.arange(max(1.0, math.ceil(ymin * 10) / 10), math.floor(ymax * 10) / 10 + 0.05, 0.1)
            ax.yaxis.set_major_locator(FixedLocator(yticks))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}" if v > 0 else ""))
            ax.yaxis.set_minor_locator(NullLocator())
            if max_latency_ratio_point is not None:
                x, y, color = max_latency_ratio_point
                ax.annotate(
                    "mobilenet",
                    xy = (x-0.03, y-0.008),
                    xytext = (-16, -18),
                    textcoords = 'offset points',
                    ha = 'right',
                    va = 'top',
                    color = color,
                    arrowprops = dict(arrowstyle = '->', color = color, linewidth = 1.2),
                    zorder = 6,
                    fontsize = FONTSIZE - 3
                )
            ax.grid(True, which = 'both', alpha = 0.25)
            ax.set_xlabel("Avg. Congestion")
            ax.set_ylabel("Simulated / Predicted\nAvg. Latency")
            ax.set_title("Latency Error vs. Congestion")

        energy_plot(ax1)
        latency_plot(ax2)
        latency_congestion_correlation_plot(ax3)
        ax4.axis('off')

        max_legend_rows = 8
        # HP: all axis have the same entries!
        handles, labels = ax1.get_legend_handles_labels()
        handles = [
            (
                ax1.scatter([], [], s = DOT_SIZE_MODEL, marker = 'o', facecolors = 'white', edgecolors = style[label]["color"], linewidths = 1.4),
                ax1.scatter([], [], s = DOT_SIZE_SIM, marker = 'o', facecolors = style[label]["color"], edgecolors = style[label]["color"], linewidths = 0.8)
            )
            for label in labels
        ]
        # Rename labels
        labels = list(map(rename_label, labels))
        handles += [
            ax1.scatter([], [], s = 0.0, marker = 'o', facecolors = 'white', edgecolors = 'white', linewidths = 0.0),
            ax1.scatter([], [], s = DOT_SIZE_MODEL, marker = 'o', facecolors = 'white', edgecolors = 'black', linewidths = 1.4),
            ax1.scatter([], [], s = DOT_SIZE_SIM, marker = 'o', facecolors = 'black', edgecolors = 'black', linewidths = 0.8)
        ]
        labels += ["", "empty: analytical model", "full: simulated"]
        ncols = math.ceil(len(labels) / max_legend_rows)
        ax4.legend(handles, labels, loc = 'center', ncol = ncols, handlelength = 3.0, handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(ndivide = None)}) # handlelength = 4.0

        for metric_name, model_raw, sim_raw in [("energy", raw_energy, raw_sim_energy), ("latency", raw_latency, raw_sim_latency)]:
            errors = []
            same_best = 0
            total_best = 0
            for i, label in enumerate(x_labels):
                available = [technique for technique in techniques if model_raw[technique][i] != None and sim_raw[technique][i] != None]
                if available:
                    total_best += 1
                    if min(available, key = lambda t : model_raw[t][i]) == min(available, key = lambda t : sim_raw[t][i]):
                        same_best += 1
                for technique in available:
                    if sim_raw[technique][i] != 0:
                        errors.append(abs(model_raw[technique][i] - sim_raw[technique][i]) / sim_raw[technique][i] * 100)
            print(f"{metric_name} validation: median error = {np.median(errors):.2f}%, mean error = {np.mean(errors):.2f}%, max error = {np.max(errors):.2f}%, min error = {np.min(errors):.2f}%, same best = {same_best}/{total_best}" if errors else f"{metric_name} validation: no valid simulated data found")
        
        # Show the plot
        plt.tight_layout() #[0, 0.125, 1, 1] #[0, 0.075, 1, 1]
        x0, y0, width, height = ax3.get_position().bounds
        bottom_plot_indent = 0.016
        ax3.set_position([x0 + bottom_plot_indent, y0, width - bottom_plot_indent, height])
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
