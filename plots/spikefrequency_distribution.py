from typing import TypeVar, Any, Optional
from types import FrameType

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import traceback
import time
import code
import time
import sys
import os

# FIX IMPORTS: add to 'path' the absolute path to the parent directory of this script
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from graph_utils import *
from partitioner import *
from load_store import *
from settings import *
from placer import *
from prints import *
from utils import *
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
        "load": args_match_and_remove(["-l", "--load"], with_value = True),
        "reload": args_match_and_remove(["-r", "--reload"], with_value = True),
        "save": args_match_and_remove(["-s", "--save"], with_value = True),
        "fraction": args_match_and_remove(["-f", "--fraction"], with_value = True, value_type = float),
        "quiet": args_match_and_remove(["-q", "--quiet"]),
    }
    return options

def help_options() -> None:
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-i --interactive\tOnce exploration has finished, instead of terminating the program, enter Python's interactive mode.")
    print(("-l, --load <?path>\tLoads a true SNN graph instead of randomly generating one. If omitted, the default path is './snn_models/simple_cnn'.\n"
           "\t\t\tThe given path is concatenated with '_0.npz', '_input.npz', '.graphml', these are the three files expected to be found."))
    print("-r, --reload <path>\tReloads a previously saved SNN graph from 'path'. This takes priority on --load.")
    print(("-s, --save <name>\tSaves the produced plot with the given name, instead of showing it. There is automatic file overwrite prevention.\n"
           "The default extension is '.png', add an extension to <name> to override the file type, supported ones are '.pdf', '.eps', '.svg', '.png'."))
    print("-f, --fraction <num>\tFraction of the lowest-valued spike frequencies to mark as ignored, let it be a number in [0, 1].")
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
        seed = 192 #79
        if options["reload"]:
            print("\n------- reloading graph ------")
            path = options["reload"]
            if not os.path.exists(path):
                raise Exception(f"The provided path does not exist: {path}")
            print("Reloading model from:", path)
            snn = HyperGraph.load(path)
            print(f"Nodes count: {snn.nodes}\nEdges: {len(snn.hyperedges)}\nMean nodes per edge: {sum(he.connections() for he in snn)/len(snn.hyperedges)}\nSeed: {seed}")
        elif options["load"] or options["load"] is None:
            print("\n-------- loading graph -------")
            path = options["load"] if options["load"] else "./snn_models/simple_cnn"
            #snn = loadSNNGraphML(f"{path}.graphml")
            snn = loadSNNcomposite(f"{path}_0.npz", f"{path}_input.npz", f"{path}.graphml")
            print(f"Nodes count: {snn.nodes}\nEdges: {len(snn.hyperedges)}\nMean nodes per edge: {sum(he.connections() for he in snn)/len(snn.hyperedges)}\nSeed: {seed}")
        else:
            print("\n------ generating graph ------")
            nodes_count = 1024*32
            nodes_per_edge_mean, nodes_per_edge_variation = 48, 24
            print(f"Nodes count: {nodes_count}\nNodes per edge mean: {nodes_per_edge_mean}\nNodes per edge variation: {nodes_per_edge_variation}\nSeed: {seed}")
            snn = HyperGraph.generate_random(nodes_count, nodes_per_edge_mean, nodes_per_edge_variation, seed = seed)
        print()
        
        values_array = [
            # NOTE: ADD EXTRA DATA HERE FOR OVERLAPPED PLOTS!
            np.array([he.spike_frequency for he in snn for _ in range(he.connections())])
            ]
        names_array = [
            # NOTE: ADD EXTRA DATA HERE FOR OVERLAPPED PLOTS!
            "SNN"
            ]

        # LINE - AREA
        colors_array = [
            #('blue', 'skyblue'),
            ('#6C8EBF', '#DAE8FC'), #blue
            ('#D79B00', '#FFE6CC'), #orange
            ('#82B366', '#D5E8D4'), #green
            ('#B85450', '#F8CECC'), #red
            ('#9673A6', '#E1D5E7'), #purple
            ('#D6B656', '#FFF2CC'), #yellow
            ]

        # Set the number of intervals
        num_intervals = 200 #150

        # Create logarithmically spaced bins from the minimum to maximum value
        all_values = np.concatenate(values_array)
        # Filter out non-positive values to avoid log10(0) or log10 of negative
        num_zeros = np.sum(all_values <= 0)
        if num_zeros > 0:
            print(f"WARNING: {num_zeros} zero or negative values were ignored when computing logarithmic bins.")
        positive_values = all_values[all_values > 0]
        if len(positive_values) == 0:
            raise ValueError("All values are zero or negative; cannot compute log-scale histogram.")
        bins = np.logspace(
            np.log10(min(positive_values)),
            np.log10(max(positive_values)),
            num = num_intervals + 1
        )

        plt.figure(figsize=(16, 10)) #(13.8, 4)

        for values, colors, name in zip(values_array, colors_array, names_array):

            # Count the number of values in each bin
            counts, bin_edges = np.histogram(values, bins = bins)

            # Prepare data points for the "dash-like" plot
            # For each bin, we need two points: one for the start and one for the end of the bin
            x_points = np.empty(2 * num_intervals)
            y_points = np.empty(2 * num_intervals)

            for i in range(num_intervals):
                x_points[2 * i] = bin_edges[i]       # Start of the interval (left edge)
                x_points[2 * i + 1] = bin_edges[i + 1]  # End of the interval (right edge)
                y_points[2 * i] = counts[i]          # Count for the interval (both start and end points have the same Y)
                y_points[2 * i + 1] = counts[i]      # Same Y for the end of the interval

            # Plot the filled area under the curve
            plt.fill_between(x_points, y_points, color = colors[1], alpha = 0.3) # alpha was 0.5

            # Plot the line connecting the points (the long spline)
            plt.plot(x_points, y_points, color = colors[0], alpha = 0.8) # alpha was 1

            # Fraction-based cutoff line
            if options["fraction"] > 0:
                fraction = options["fraction"]
                sorted_values = np.sort(values)
                cutoff_index = int(len(sorted_values) * fraction)
                cutoff_value = sorted_values[cutoff_index]

                bin_index = np.searchsorted(bin_edges, cutoff_value, side='right') - 1
                if 0 <= bin_index < len(counts):
                    cutoff_y = counts[bin_index]
                else:
                    cutoff_y = 0

                total_cut_value = sum(sorted_values[:cutoff_index])
                total_value = sum(sorted_values)
                plt.vlines(cutoff_value, ymin=0, ymax=cutoff_y, color='red', linestyle='--', linewidth=2)
                print(f"Ignored lowest {100*fraction:.1f}% of values in {name}:")
                prettyPrintDict({"ignored count": f"{cutoff_index}/{len(sorted_values)}",
                                 "ignored below value": f"{cutoff_value:.3g}",
                                 "ignored total value": f"{total_cut_value:.3g}/{total_value:.3g} ({100*total_cut_value/total_value:.3g}%)"}, 1)

        patches = []
        for colors, name in zip(colors_array, names_array):
            patches.append(mpatches.Patch(facecolor = colors[1], edgecolor = colors[0], label = name))

        #plt.xlim(0.9, 100)

        # Add the custom legend
        plt.legend(handles  =patches)

        # Set X axis to be logarithmic
        plt.xscale('log')

        # Show grid only for the Y axis (with low alpha) and hide X axis grid
        plt.grid(axis = 'y', alpha = 0.3)
        plt.grid(axis = 'x', visible = False)

        # Labels and title
        plt.xlabel("Spike Frequency")
        plt.ylabel("Count")
        plt.title(f"Histogram of Spike Frequencies (Logarithmic Bins, {num_intervals} Intervals)")

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