from typing import TypeVar, Any, Optional
from types import FrameType

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FixedLocator, FuncFormatter, NullFormatter
from matplotlib.lines import Line2D
import matplotlib.legend_handler
import matplotlib.patches
import matplotlib.axes
import matplotlib

from collections import defaultdict
from scipy.stats import zscore, spearmanr
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

FONTSIZE = 13
BAR_WIDTH_PART = 0.15
BAR_WIDTH_PLAC = 0.10

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
        file_data = []  # will hold tuples: (num_nodes, filename, technique_entries)
        x_labels = []

        # Metrics to collect
        synaptic_reuse_mean : dict[str, list[Optional[float]]] = defaultdict(list)
        synaptic_reuse_geomean : dict[str, list[Optional[float]]] = defaultdict(list)
        connections_locality_mean : dict[str, list[Optional[float]]] = defaultdict(list)
        connections_locality_geomean : dict[str, list[Optional[float]]] = defaultdict(list)
        init_connections_locality_mean : dict[str, list[Optional[float]]] = defaultdict(list)
        init_connections_locality_geomean : dict[str, list[Optional[float]]] = defaultdict(list)
        connectivity : dict[str, list[Optional[float]]] = defaultdict(list)
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
                    
                    connectivity[technique].append(entry.get("part_cost", None))
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
                    connectivity[technique].append(None)
        
        x_indices = list(range(len(x_labels)))
        
        # Optional: normalize w.r.t. the best partitioning
        #best_synaptic_reuse_geomean = [max([synaptic_reuse_geomean[technique][i] for technique in techniques if synaptic_reuse_geomean[technique][i] != None], default = 0) for i in range(len(x_indices))]
        #best_connections_locality_geomean = [min([connections_locality_geomean[technique][i] for technique in techniques if connections_locality_geomean[technique][i] != None], default = 0) for i in range(len(x_indices))]
        #best_connectivity = [min([connectivity[technique][i] for technique in techniques if connectivity[technique][i] != None], default = 0) for i in range(len(x_indices))]
        #best_edp = [min([energy_delay_product[technique][i] for technique in techniques if energy_delay_product[technique][i] != None], default = 0) for i in range(len(x_indices))]
        # ALT: always pick hMETIS to normalize against!
        best_connectivity = [connectivity["hmetis-hilbert-fd"][i] for i in range(len(x_indices))]
        best_edp = [energy_delay_product["hmetis-hilbert-fd"][i] for i in range(len(x_indices))]
        #best_synaptic_reuse_geomean = [synaptic_reuse_geomean["hmetis-hilbert-fd"][i] for i in range(len(x_indices))]
        #best_connections_locality_geomean = [connections_locality_geomean["hmetis-hilbert-fd"][i] for i in range(len(x_indices))]
        for technique in techniques:
            #synaptic_reuse_mean[technique] = list(map(lambda c : c[0] / c[1] if c[0] else None, zip(synaptic_reuse_mean[technique], best_synaptic_reuse_geomean)))
            #synaptic_reuse_geomean[technique] = list(map(lambda c : c[0] / c[1] if c[0] else None, zip(synaptic_reuse_geomean[technique], best_synaptic_reuse_geomean)))
            #connections_locality_mean[technique] = list(map(lambda c : c[0] / c[1] if c[0] else None, zip(connections_locality_mean[technique], best_connections_locality_geomean)))
            #connections_locality_geomean[technique] = list(map(lambda c : c[0] / c[1] if c[0] else None, zip(connections_locality_geomean[technique], best_connections_locality_geomean)))
            connectivity[technique] = list(map(lambda c : c[0] / c[1] if c[0] != None else None, zip(connectivity[technique], best_connectivity)))
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
        connectivity = {k : v for k, v in connectivity.items() if k in techniques}
        energy_delay_product = {k : v for k, v in energy_delay_product.items() if k in techniques}

        # Replace 'None' with zero
        for technique in techniques:
            synaptic_reuse_mean[technique] = list(map(lambda x : x if x != None else math.nan, synaptic_reuse_mean[technique]))
            synaptic_reuse_geomean[technique] = list(map(lambda x : x if x != None else math.nan, synaptic_reuse_geomean[technique]))
            connections_locality_mean[technique] = list(map(lambda x : x if x != None else math.nan, connections_locality_mean[technique]))
            connections_locality_geomean[technique] = list(map(lambda x : x if x != None else math.nan, connections_locality_geomean[technique]))
            init_connections_locality_mean[technique] = list(map(lambda x : x if x != None else math.nan, init_connections_locality_mean[technique]))
            init_connections_locality_geomean[technique] = list(map(lambda x : x if x != None else math.nan, init_connections_locality_geomean[technique]))
            connectivity[technique] = list(map(lambda x : x if x != None else math.nan, connectivity[technique]))
            energy_delay_product[technique] = list(map(lambda x : x if x != None else math.nan, energy_delay_product[technique]))

        # Plotting (note: 25.6 = 2560 pixel)
        fig, (ghost_ax1, ax1, ax2, ghost_ax2) = plt.subplots(1, 4, figsize = (25.6, 7*(1 - 0.075)), tight_layout = True, width_ratios = [1/6, 1/3, 1/3, 1/6])
        ghost_ax1.remove()
        ghost_ax2.remove()

        # Logarithmic tick labels on linear-scale axis
        """
        Set axis ticks to look and behave like a log10 axis
        when data are already transformed with np.log10.
        """
        def set_log10_ticks(ax : matplotlib.axes.Axes, axis : str = 'x', min_exp : float = None, max_exp : float = None):
            ax.grid(axis = axis, which = 'major')
            ax.grid(axis = axis, which = 'minor', alpha = 0.5)
            
            # Determine visible range
            if axis == 'x':
                lo, hi = ax.get_xlim()
            else:
                lo, hi = ax.get_ylim()

            # Limit range to integers around current view
            if min_exp is None:
                min_exp = int(np.floor(lo))
            if max_exp is None:
                max_exp = int(np.ceil(hi))

            # Major ticks = integer log10 values
            major_locs = np.arange(min_exp, max_exp + 1)
            # Minor ticks = log10 of 2..9 × each decade
            minor_locs = []
            for e in major_locs:
                minor_locs.extend(np.log10(np.arange(2, 10) * np.float_power(10, e)))
            minor_locs = [x for x in minor_locs if lo <= x <= hi]

            # Apply locators and formatters
            formatter = FuncFormatter(lambda val, _: f"$10^{{{int(val)}}}$")

            if axis == 'x':
                ax.xaxis.set_major_locator(FixedLocator(major_locs))
                ax.xaxis.set_minor_locator(FixedLocator(minor_locs))
                ax.xaxis.set_major_formatter(formatter)
            else:
                ax.yaxis.set_major_locator(FixedLocator(major_locs))
                ax.yaxis.set_minor_locator(FixedLocator(minor_locs))
                ax.yaxis.set_major_formatter(formatter)

        # Synpatic reuse plot
        def synpatic_reuse_plot(ax : matplotlib.axes.Axes):
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
            possible_markers = cycle(['o', 'v', '^', 's', 'p', '*', 'p', 'X', 'D'])
            possible_linestyles = cycle(['-', ':', '--', '-.', (5, (10, 3))])
            marker_style = {}
            line_style = {}
            prev_part_technique, ongoing_color, ongoing_hatch, ongoing_linestyle = None, cycle(possible_colors), '', next(possible_linestyles)
            for technique in part_techniques_list:
                ongoing_linestyle = next(possible_linestyles)
                color = next(ongoing_color)
                marker_style[technique] = {"color": color, "marker" : next(possible_markers)}
                line_style[technique] = {"color": color, "linestyle" : ongoing_linestyle}
            
            combined_handles = []
            text_idxs = defaultdict(int, {"hehiding" : 10, "sequential" : 8, "unordered" : 54, "hmetis" : 50, "edgehiding" : 48})
            for technique in part_techniques_list:
                #srg = synaptic_reuse_geomean[part_techniques[technique]]
                #con = connectivity[part_techniques[technique]]
                srg = np.log10(synaptic_reuse_geomean[part_techniques[technique]])
                con = np.log10(connectivity[part_techniques[technique]])
                #srg = zscore(srg)
                #con = zscore(con)
                ax.scatter(srg, con, label = rename_label(technique), alpha = 0.8, **marker_style[technique])
                z = np.polyfit(srg, con, 1)
                x = np.linspace(min(srg), max(srg), 100)
                y = np.poly1d(z)(x)
                ax.plot(x, y, **line_style[technique])
                combined_handles.append(Line2D([], [], **(marker_style[technique] | line_style[technique]), label = rename_label(technique)))
                rho, p = spearmanr(srg, con)
                print(f"[PART] {technique}: Spearman ρ = {rho:.3f}, p = {p:.3g}")
                if not math.isnan(rho) and not math.isnan(p):
                    text_idx = text_idxs[technique]
                    ax.text(x[text_idx], y[text_idx], f"ρ = {rho:.2f} p = {p:.2g}", color = line_style[technique]["color"], fontsize = 9, va = "bottom", ha = "left", alpha = 0.9, fontweight = "medium")
            ax.annotate("", xy = (2.62, 0.07), xytext = (2.5, 0.17), arrowprops = dict(arrowstyle = "->", color = "red", lw = 2.5, shrinkA = 0, shrinkB = 0, alpha = 0.9), zorder = 5)
            ax.text(2.5, 0.17, "goal", fontsize = 11, fontweight = "bold", color = "red", va = "bottom", ha = "center", zorder = 6)
            ax.set_xlabel(f"Synaptic Reuse Geometric Mean")
            ax.set_ylabel("Connectivity (normalized on hierarchical)")
            ax.set_title("Synaptic Reuse vs Connectivity")
            #ax.set_yscale("log", base = 10)
            #ax.set_xscale("log", base = 10)
            set_log10_ticks(ax, 'x')
            set_log10_ticks(ax, 'y')
            #format_y_bars(ax)
            #ax.set_ybound(0.8, 800)
            return combined_handles

        # Connections locality plot
        def connections_locality_plot(ax : matplotlib.axes.Axes):
            plac_techniques = techniques
            for plac_technique in omit_plac_techniques:
                if plac_technique in plac_techniques:
                    plac_techniques.remove(plac_technique)
            plac_techniques = sorted(techniques, key = lambda s : (''.join(chr(255 - ord(c)) for c in s.split('-', 1)[0]), chr(255 - ord(s.split('-', 1)[1][2])))) # descending order on the word before the first '-', then ascending order as a tiebreak.

            # Assign style to partitioning techniques
            possible_colors = [
                    "#6C8EBF", # BLUE
                    "#48617A", # DARK BLUE
                    "#336699", # DARKER BLUE
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
            possible_markers = cycle(['o', 'v', '^', 's', 'p', 'H', 'D', 'P', 'X'])
            possible_linestyles = cycle(['-', ':', '--', '-.'])
            marker_style = {} 
            line_style = {}
            prev_part_technique, ongoing_color, ongoing_hatch, ongoing_linestyle = None, cycle(possible_colors), '', next(possible_linestyles)
            for technique in plac_techniques:
                part_technique, plac_technique = technique.split('-', 1)
                init_plac_technique, plac_ref_technique = plac_technique.split('-', 1) if '-' in plac_technique else ('', plac_technique)
                if prev_part_technique != part_technique:
                    prev_part_technique = part_technique
                    ongoing_linestyle = next(possible_linestyles)
                color = next(ongoing_color)
                marker_style[technique] = {"color": color, "marker" : next(possible_markers)}
                line_style[technique] = {"color": color, "linestyle" : ongoing_linestyle}
            
            combined_handles = []
            text_idxs = defaultdict(int, {"hehiding-hilbert-fd" : -1, "hehiding-spectral-fd" : -1, "hehiding-truenorth" : -1,
                                          "hmetis-hilbert-fd" : -20, "hmetis-spectral-fd" : -5, "hmetis-truenorth" : -1,
                                          "sequential-hilbert-fd" : -35, "sequential-spectral-fd" : -15, "sequential-truenorth" : -14})
            text_posns = defaultdict(lambda : ("top", "left"), {"sequential-hilbert-fd" : ("bottom", "right"), "sequential-truenorth" : ("bottom", "right")})
            for technique in plac_techniques:
                #clg = connections_locality_geomean[technique]
                #edp = energy_delay_product[technique]
                clg = np.log10(connections_locality_geomean[technique])
                edp = np.log10(energy_delay_product[technique])
                #clg = zscore(clg)
                #edp = zscore(edp)
                ax.scatter(clg, edp, label = rename_label(technique), alpha = 0.8, **marker_style[technique])
                z = np.polyfit(clg, edp, 1)
                x = np.linspace(min(clg), max(clg), 100)
                y = np.poly1d(z)(x)
                ax.plot(x, y, **line_style[technique])
                combined_handles.append(Line2D([], [], **(marker_style[technique] | line_style[technique]), label = rename_label(technique)))
                rho, p = spearmanr(clg, edp)
                print(f"[PLAC] {technique}: Spearman ρ = {rho:.3f}, p = {p:.3g}")
                if not math.isnan(rho) and not math.isnan(p):
                    text_idx = text_idxs[technique]
                    va, ha = text_posns[technique]
                    ax.text(x[text_idx], y[text_idx], f"ρ = {rho:.2f} p = {p:.2g}", color = line_style[technique]["color"], fontsize = 9, va = va, ha = ha, alpha = 0.9, fontweight = "medium")
            ax.annotate("", xy = (0.4, -0.2), xytext = (0.3, -0.3), arrowprops = dict(arrowstyle = "<-", color = "red", lw = 2.5, shrinkA = 0, shrinkB = 0, alpha = 0.9), zorder = 5)
            ax.text(0.3, -0.3, "goal", fontsize = 11, fontweight = "bold", color = "red", va = "top", ha = "left", zorder = 6)
            ax.set_xlabel(f"Connections Locality Geometric Mean")
            ax.set_ylabel("Energy-Delay Product\n(normalized on hierarchical + hilbert + force-directed)")
            ax.set_title("Connections Locality vs Energy-Latency Product")
            #ax.set_yscale("log", base = 10)
            #ax.set_xscale("log", base = 10)
            set_log10_ticks(ax, 'x')
            set_log10_ticks(ax, 'y')
            #format_y_bars(ax)
            return combined_handles
        
        handles1 = synpatic_reuse_plot(ax1)
        handles2 = connections_locality_plot(ax2)
        
        # Setup legends AFTER the tight layout
        plt.tight_layout(rect = [0, 0, 1, 1]) # TODO: comment me or use "gridspec" for a better scaling of plots!
        labels1 = [h.get_label() for h in handles1]
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