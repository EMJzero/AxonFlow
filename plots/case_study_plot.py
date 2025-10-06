from typing import TypeVar, Any, Optional
from types import FrameType

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
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
        "file": args_match_and_remove(["-f", "--file"], with_value = True),
        "save": args_match_and_remove(["-s", "--save"], with_value = True),
        "quiet": args_match_and_remove(["-q", "--quiet"]),
    }
    return options

def help_options() -> None:
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-i --interactive\tOnce exploration has finished, instead of terminating the program, enter Python's interactive mode.")
    print(("-f, --file <path>\tPath to the JSON file containing the output of a run of 'targeted_main.py'."))
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

FONTSIZE = 13
BAR_WIDTH = 0.11

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
        file_path = options["file"]
        if not file_path:
            print(f"No file path provided, option '-f' or '--file' is mandatory.")
            sys.exit(0)
        if not os.path.exists(file_path):
            raise Exception(f"The provided path does not exist: {file_path}")
        elif os.path.isdir(file_path):
            raise Exception(f"The provided path is a directory, not a file: {file_path}")

        techniques = set()

        # Metrics to collect
        synaptic_reuse_mean : dict[str, Optional[float]] = defaultdict(list)
        synaptic_reuse_geomean : dict[str, Optional[float]] = defaultdict(list)
        connections_locality_mean : dict[str, Optional[float]] = defaultdict(list)
        connections_locality_geomean : dict[str, Optional[float]] = defaultdict(list)
        init_connections_locality_mean : dict[str, Optional[float]] = defaultdict(list)
        init_connections_locality_geomean : dict[str, Optional[float]] = defaultdict(list)
        
        energy : dict[str, Optional[float]] = defaultdict(list)
        energy_delay_product : dict[str, Optional[float]] = defaultdict(list)
        latency : dict[str, Optional[float]] = defaultdict(list)
        congestion : dict[str, Optional[float]] = defaultdict(list)
        times : dict[str, Optional[float]] = defaultdict(list)
        connectivity : dict[str, Optional[float]] = defaultdict(list)
        part_times : dict[str, Optional[float]] = defaultdict(list)
        init_energy : dict[str, Optional[float]] = defaultdict(list)
        init_latency : dict[str, Optional[float]] = defaultdict(list)
        init_energy_delay_product : dict[str, Optional[float]] = defaultdict(list)
        init_congestion : dict[str, Optional[float]] = defaultdict(list)

        # Optional: specify techniques to omit
        omit_techniques = {"unordered", "edgehiding", "hmetis-truenorth", "hmetis-hilbert-ps", "hmetis-spectral-ps", "hmetis-hilbert-fd", "hmetis-spectral-fd", "hehiding-hilbert-ps", "hehiding-spectral-ps", "sequential-hilbert-ps", "sequential-spectral-ps"}
        # Optional: disable "shades" for initial layout
        no_initial_layout = True
        # Optional: divide the connections locality by the partitions count
        norm_by_part_count = False

        # Read file
        file = os.path.basename(file_path)
        with open(file_path, "r") as f:
            print("Parsing:", file_path)
            data : list[dict[str, float]] = json.load(f)

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

            if not entry.get("part_valid", True):
                print(f"WARNING: Invalid partitioning for {file} -> {name}")
            if not entry.get("plac_valid", True):
                print(f"WARNING: Invalid placement for {file} -> {name}")

        # Extract data in increasing graph size order
        label = f"{os.path.splitext(file)[0]}"
        max_part_count = 1

        for technique in techniques:
            entry = technique_entries.get(technique)
            if entry:
                synaptic_reuse_mean[technique] = entry.get("part_synaptic_reuse", None)
                if isinstance(synaptic_reuse_mean[technique], dict):
                    synaptic_reuse_geomean[technique] = synaptic_reuse_mean[technique]["geo_mean"]
                    synaptic_reuse_mean[technique] = synaptic_reuse_mean[technique]["ar_mean"]
                else:
                    synaptic_reuse_geomean[technique] = synaptic_reuse_mean[technique]
                connections_locality_mean[technique] = entry.get("plac_connections_locality", None)
                if isinstance(connections_locality_mean[technique], dict):
                    connections_locality_geomean[technique] = connections_locality_mean[technique]["geo_mean_weighted"]/(entry["part_count"] if norm_by_part_count else 1)
                    connections_locality_mean[technique] = connections_locality_mean[technique]["ar_mean_weighted"]/(entry["part_count"] if norm_by_part_count else 1)
                else:
                    connections_locality_geomean[technique] = connections_locality_mean[technique]
                init_connections_locality_mean[technique] = entry.get("init_connections_locality", None) if not no_initial_layout else None
                if isinstance(init_connections_locality_mean[technique], dict):
                    init_connections_locality_geomean[technique] = init_connections_locality_mean[technique]["geo_mean"]/(entry["part_count"] if norm_by_part_count else 1)
                    init_connections_locality_mean[technique] = init_connections_locality_mean[technique]["ar_mean"]/(entry["part_count"] if norm_by_part_count else 1)
                else:
                    init_connections_locality_geomean[technique] = init_connections_locality_mean[technique]
                energy[technique] = entry.get("plac_energy", None)
                latency[technique] = entry.get("plac_avg_lat", None)
                energy_delay_product[technique] = energy[technique] * latency[technique] if energy[technique] != None and latency[technique] != None else None
                congestion[technique] = entry.get("plac_avg_cong", None)
                times[technique] = entry.get("time", None)
                connectivity[technique] = entry.get("part_cost", None)
                part_times[technique] = entry.get("part_time", None)
                init_energy[technique] = entry.get("init_energy", None) if not no_initial_layout else None
                init_latency[technique] = entry.get("init_avg_lat", None) if not no_initial_layout else None
                init_energy_delay_product[technique] = init_energy[technique] * init_latency[technique] if init_energy[technique] != None and init_latency[technique] != None else None
                init_congestion[technique] = entry.get("init_avg_cong", None) if not no_initial_layout else None
                if norm_by_part_count and (entry["part_count"] > max_part_count):
                    max_part_count = entry["part_count"]
            else:
                energy[technique] = None
                latency[technique] = None
                congestion[technique] = None
                times[technique] = None
                connectivity[technique] = None
                part_times[technique] = None
                init_energy[technique] = None
                init_latency[technique] = None
                init_congestion[technique] = None
                synaptic_reuse_mean[technique] = None
                synaptic_reuse_geomean[technique] = None
                connections_locality_mean[technique] = None
                connections_locality_geomean[technique] = None
                init_connections_locality_mean[technique] = None
                init_connections_locality_geomean[technique] = None
                energy_delay_product[technique] = None

        # Replace 'None' with zero
        for technique in techniques:
            energy[technique] = energy[technique] if energy[technique] != None else math.nan
            latency[technique] = latency[technique] if latency[technique] != None else math.nan
            congestion[technique] = congestion[technique] if congestion[technique] != None else math.nan
            times[technique] = times[technique] if times[technique] != None else math.nan
            connectivity[technique] = connectivity[technique] if connectivity[technique] != None else math.nan
            part_times[technique] = part_times[technique] if part_times[technique] != None else math.nan
            energy_delay_product[technique] = energy_delay_product[technique] if energy_delay_product[technique] != None else math.nan
            init_energy[technique] = init_energy[technique] if init_energy[technique] != None else math.nan
            init_latency[technique] = init_latency[technique] if init_latency[technique] != None else math.nan
            init_congestion[technique] = init_congestion[technique] if init_congestion[technique] != None else math.nan
            init_energy_delay_product[technique] = init_energy_delay_product[technique] if init_energy_delay_product[technique] != None else math.nan
            synaptic_reuse_mean[technique] = synaptic_reuse_mean[technique] if synaptic_reuse_mean[technique] != None else math.nan
            synaptic_reuse_geomean[technique] = synaptic_reuse_geomean[technique] if synaptic_reuse_geomean[technique] != None else math.nan
            connections_locality_mean[technique] = connections_locality_mean[technique] if connections_locality_mean[technique] != None else math.nan
            connections_locality_geomean[technique] = connections_locality_geomean[technique] if connections_locality_geomean[technique] != None else math.nan
            init_connections_locality_mean[technique] = init_connections_locality_mean[technique] if init_connections_locality_mean[technique] != None else math.nan
            init_connections_locality_geomean[technique] = init_connections_locality_geomean[technique] if init_connections_locality_geomean[technique] != None else math.nan

        # Optional: normalize w.r.t. the best partitioning
        best_synaptic_reuse_geomean = max([synaptic_reuse_geomean[technique] for technique in techniques], default = 0)
        best_connections_locality_geomean = min([connections_locality_geomean[technique] for technique in techniques], default = 0)
        best_edp = min([energy_delay_product[technique] for technique in techniques], default = 0)
        best_energy = min([energy[technique] for technique in techniques], default = 0)
        best_latency = min([latency[technique] for technique in techniques], default = 0)
        best_congestion = min([congestion[technique] for technique in techniques], default = 0)
        best_connectivity = min([connectivity[technique] for technique in techniques], default = 0)
        for technique in techniques:
            synaptic_reuse_mean[technique] = synaptic_reuse_mean[technique] / best_synaptic_reuse_geomean
            synaptic_reuse_geomean[technique] = synaptic_reuse_geomean[technique] / best_synaptic_reuse_geomean
            connections_locality_mean[technique] = connections_locality_mean[technique] / best_connections_locality_geomean
            connections_locality_geomean[technique] = connections_locality_geomean[technique] / best_connections_locality_geomean
            energy_delay_product[technique] = energy_delay_product[technique] / best_edp
            energy[technique] = energy[technique] / best_energy
            latency[technique] = latency[technique] / best_latency
            congestion[technique] = congestion[technique] / best_congestion
            connectivity[technique] = connectivity[technique] / best_connectivity
            # still normalize them w.r.t. the eventual best for their metric since they will be in the same plot
            init_connections_locality_mean[technique] = init_connections_locality_mean[technique] / best_connections_locality_geomean
            init_connections_locality_geomean[technique] = init_connections_locality_geomean[technique] / best_connections_locality_geomean
            init_energy_delay_product[technique] = init_energy_delay_product[technique] / best_edp
            init_energy[technique] = init_energy[technique] / best_energy
            init_latency[technique] = init_latency[technique] / best_latency
            init_congestion[technique] = init_congestion[technique] / best_congestion
            if norm_by_part_count:
                connections_locality_mean[technique] = connections_locality_mean[technique] / max_part_count
                connections_locality_geomean[technique] = connections_locality_geomean[technique] / max_part_count
                init_connections_locality_mean[technique] = init_connections_locality_mean[technique] / max_part_count
                init_connections_locality_geomean[technique] = init_connections_locality_geomean[technique] / max_part_count

        x_labels = ['']
        x_indices = [0]
        index = np.arange(len(x_labels))
        offset = (len(techniques) - 1)/2
        techniques = sorted(techniques, key = lambda s : (''.join(chr(255 - ord(c)) for c in s.split('-', 1)[0]), s.split('-', 1)[1])) # descending order on the word before the first '-', then ascending order as a tiebreak.

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize = (16, 10), sharex = True, tight_layout = True)

        # Assign style to partitioning techniques
        possible_colors = [
                "#6C8EBF", # BLUE
                "#48617A", # DARK BLUE
                "#336699", # DARKER BLUE
                #"#FFB700", # YELLOW # alts: D79B00
                #"#B38000", # DARK YELLOW
                #"#FF6978", # PINK
                #"#A8516E", # DARK PINK
                #"#82B366", # GREEN
                #"#169E1B", # DARK GREEN
                #"#2F762F", # DARKER GREEN
                "#EB6050", # RED # alts: cc3300, e63900, ec3c00, ff531a, ff3c2d, f03c2d, ea382a, ea3b2e, e7473a, e9493d, e94e3d, eb5847
                "#8E2B25", # DARK RED
                "#9F140D", # DARKER RED
                "#C2E812", # LIME
                "#768E0B", # DARK LIME
            ]
        possible_markers = cycle(['o', 'v', '^', 's', 'p', '*', 'D', 'X', 'p'])
        possible_linestyles = cycle(['-', ':', '--', '-.'])
        part_techniques_to_hatch = defaultdict(lambda : '', hehiding = '/')
        style = {}
        line_style = {}
        prev_part_technique, ongoing_color, ongoing_hatch, ongoing_linestyle = None, cycle(possible_colors), '', next(possible_linestyles)
        for technique in techniques:
            part_technique = technique.split('-', 1)[0]
            if prev_part_technique != part_technique:
                prev_part_technique = part_technique
                ongoing_linestyle = next(possible_linestyles)
            color = next(ongoing_color)
            style[technique] = {"color": color, "hatch": part_techniques_to_hatch[part_technique], "edgecolor": "white"}
            line_style[technique] = {"color": color, "marker" : next(possible_markers), "linestyle" : ongoing_linestyle, "markersize" : 8}
        shadow_bars_alpha = 0.3

        # Decide the y-axis bounds by ignoring outliers (lower sigma is more brutal)
        def set_bounds(ax : matplotlib.axes.Axes, data : Optional[dict[str, float]], shapes : Optional[list[matplotlib.patches.Patch]], sigma : float = 0.5, margin : float = 0.2):
            if data:
                data_by_x = defaultdict(list)
                for technique in techniques:
                    for xi, value in zip(x_labels, [data[technique]]):
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
            ax.grid(axis = 'y', which = 'major')
            ax.grid(axis = 'y', which = 'minor', alpha = 0.5)
            ax.yaxis.set_major_locator(LogLocator(base = 10.0, subs = np.arange(1.0, 10.0, 1.0), numticks = 10))
            ax.yaxis.set_minor_locator(LogLocator(base = 10.0, subs = np.arange(1.0, 9.0, 0.1), numticks = 10))
            #formatter = FuncFormatter(lambda v, _: f"{v*100:.0f}%" if v > 0 else "")
            formatter = FuncFormatter(lambda v, _: f"{v:.1f}" if v > 0 else "")
            ax.yaxis.set_major_formatter(formatter)
            #ax.yaxis.set_minor_formatter(formatter)
            ax.yaxis.set_minor_formatter(NullFormatter())

        # Energy plot
        def energy_plot(ax : matplotlib.axes.Axes):
            rects = []
            for j, technique in enumerate(techniques):
                rects += ax.bar(index + (j - offset) * BAR_WIDTH, init_energy[technique], BAR_WIDTH, label = None, alpha = shadow_bars_alpha, **style[technique])
                rects += ax.bar(index + (j - offset) * BAR_WIDTH, energy[technique], BAR_WIDTH, label = technique, **style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            #ax.set_xlabel("Problem Size (nodes)")
            ax.set_ylabel("Placement Energy (normalized w.r.t. lowest)")
            ax.set_title("Energy across SNNs")
            #ax.legend()
            format_y_bars(ax)
            set_bounds(ax, energy, rects, sigma = 0.4, margin = 0.1)

        # Latency plot
        def latency_plot(ax : matplotlib.axes.Axes):
            rects = []
            for j, technique in enumerate(techniques):
                rects += ax.bar(index + (j - offset) * BAR_WIDTH, init_latency[technique], BAR_WIDTH, label = None, alpha = shadow_bars_alpha, **style[technique])
                rects += ax.bar(index + (j - offset) * BAR_WIDTH, latency[technique], BAR_WIDTH, label = technique, **style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_ylabel("Avg Latency (normalized w.r.t. lowest)")
            ax.set_title("Latency")
            #ax.legend()
            format_y_bars(ax)
            set_bounds(ax, latency, rects, sigma = 2.0)

        # Congestion plot
        def congestion_plot(ax : matplotlib.axes.Axes):
            rects = []
            for j, technique in enumerate(techniques):
                rects += ax.bar(index + (j - offset) * BAR_WIDTH, init_congestion[technique], BAR_WIDTH, label = None, alpha = shadow_bars_alpha, **style[technique])
                rects += ax.bar(index + (j - offset) * BAR_WIDTH, congestion[technique], BAR_WIDTH, label = technique, **style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_ylabel("Avg. congestion (normalized w.r.t. lowest)")
            ax.set_title("Congestion")
            #ax.legend()
            format_y_bars(ax)
            set_bounds(ax, congestion, rects, sigma = 1.2)

        # TODO: does it even make sense to look at this? It is not like, the longer you run, the more you consume here...
        # Energy x Delay Product plot
        def edp_plot(ax : matplotlib.axes.Axes):
            rects = []
            for j, technique in enumerate(techniques):
                rects += ax.bar(index + (j - offset) * BAR_WIDTH, init_energy_delay_product[technique], BAR_WIDTH, label = None, alpha = shadow_bars_alpha, **style[technique])
                rects += ax.bar(index + (j - offset) * BAR_WIDTH, energy_delay_product[technique], BAR_WIDTH, label = technique, **style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_ylabel("Placement Energy x Latency (normalized w.r.t. lowest)")
            ax.set_title("Energy-Latency Product")
            #ax.legend()
            format_y_bars(ax)
            set_bounds(ax, energy_delay_product, rects, sigma = 0.0, margin = 0.1)

        # Partitioned Hypergraph Connectivity plot
        def conn_plot(ax : matplotlib.axes.Axes):
            rects = []
            for j, technique in enumerate(techniques):
                rects += ax.bar(index + (j - offset) * BAR_WIDTH, connectivity[technique], BAR_WIDTH, label = technique, **style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_ylabel("Partitioning Connectivity (normalized w.r.t. lowest)")
            ax.set_title("Connectivity")
            #ax.legend()
            format_y_bars(ax)
            set_bounds(ax, connectivity, rects)

        # Partitioning Time plot
        def part_time_plot(ax : matplotlib.axes.Axes):
            for technique in techniques:
                ax.plot(x_indices, part_times[technique], label = technique, **line_style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Time [s]")
            ax.set_title("Partitioning Time")
            #ax.legend()
            ax.grid(True)

        # Placement Time plot
        def plac_time_plot(ax : matplotlib.axes.Axes):
            for technique in techniques:
                ax.plot(x_indices, list(map(lambda t : t[0] - t[1], zip(times[technique], part_times[technique]))), label = technique, **line_style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Time [s]")
            ax.set_title("Placement Time")
            #ax.legend()
            ax.grid(True)

        # Total Time plot
        def tot_time_plot(ax : matplotlib.axes.Axes):
            for technique in techniques:
                ax.plot(x_indices, times[technique], label = technique, **line_style[technique])
                line_style[technique].pop("marker")
                ax.axhline(times[technique], label = None, **line_style[technique])
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation = 45)
            ax.set_yscale('log', base = 10)
            ax.set_ylabel("Time [s]")
            ax.set_title("Total Time")
            #ax.legend()
            ax.grid(True)
        
        energy_plot(ax1)
        latency_plot(ax2)
        congestion_plot(ax3)
        edp_plot(ax4)
        tot_time_plot(ax5)
        
        max_legend_rows = 2
        # HP: all axis have the same entries!
        handles, labels = ax1.get_legend_handles_labels()
        # Rename labels
        labels = list(map(rename_label, labels))
        # UNLESS: you use lines for time, instead of bars
        handles_lines, _ = ax5.get_legend_handles_labels()
        if not no_initial_layout:
            handles += [
                matplotlib.patches.Rectangle((0, 0,), 0, 0, facecolor = "gray", edgecolor = "white", alpha = 0.0),
                matplotlib.patches.Rectangle((0, 0,), 0, 0, facecolor = "gray", edgecolor = "white", alpha = 0.0)
            ]
            labels += ["full: refined placement", "shade: initial placement"]
            handles_lines += [
                matplotlib.lines.Line2D([0], [0], marker = "s", color = "gray", linestyle = "", markersize = 10),
                matplotlib.lines.Line2D([0], [0], marker = "s", color = "gray", linestyle = "", markersize = 10, alpha = 0.3)
            ]
        combined_handles = list(zip(handles, handles_lines))
        ncols = math.ceil(len(labels) / max_legend_rows)
        fig.legend(combined_handles, labels, loc = 'lower center', ncol = ncols, handler_map = {tuple: matplotlib.legend_handler.HandlerTuple(ndivide = None)}, handlelength = 5.0) # handlelength = 4.0
        
        plt.tight_layout(rect = [0, 0.065, 1, 1]) # TODO: comment me or use "gridspec" for a better scaling of plots!
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