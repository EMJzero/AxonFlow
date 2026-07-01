import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.ndimage import gaussian_filter1d
from functools import reduce
import matplotlib.axes
import numpy as np
import matplotlib
import math
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


# MATPLOTLIB SETTINGS:

SUPPORTED_EXTENSIONS = ['.pdf', '.eps', '.svg', '.png']
DPI = 300 #800

FONTSIZE = 17 # was 15
IN_PLOT_TEXT_FONTSIZE = 13

SAVE = False
SAVE_PATH = "spike_frequency_manual_plot.png"

LOG_SPACE_X = False
LOG_SPACE_Y = True
FIT_LOGNORM = True

# default font size
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}

matplotlib.rc('font', **font)

# specific element sizes
# autoscalable options: xx-small (0.58x), x-small (0.69x), small (0.83x), medium (1.0x), large (1.2x), x-large (1.44x), xx-large (1.73x)
plt.rcParams.update({
    #'axes.titlesize': 18, # default: FONTSIZE*1.2
    #'axes.labelsize': 15, # default: FONTSIZE*1
    'xtick.labelsize': 15, # default: FONTSIZE*1
    'ytick.labelsize': 15, # default: FONTSIZE*1
    #'legend.fontsize': 15, # default: FONTSIZE*1
    #'figure.titlesize': 18 # default: FONTSIZE*1.2
})


# MAIN:

if __name__ == "__main__":
    processed_snns_paths = {
        "16k model": "snn_models/8k_model_processed",
        "64k model": "snn_models/64k_model_ordered_processed",
        "lenet": "snn_models/lenet_cifar_ordered_processed",
        #"alexnet": "snn_models/alexnet_cifar_ordered_processed",
        "Allen V1": "snn_models/allen_v1_ordered_processed"
    }

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
    
    def rename_title(title : str) -> str:
        if title == "lenet": return "LeNet"
        else: return title
    
    # Set the number of intervals/bins
    num_intervals = 100 #200
    
    plots_cols = math.ceil(math.sqrt(len(processed_snns_paths)))
    plots_rows = math.ceil(len(processed_snns_paths)/plots_cols)
    #fig, axs = plt.subplots(plots_rows, plots_cols, figsize = (16, 10), tight_layout = True)
    fig, axs = plt.subplots(plots_rows, plots_cols, figsize = (16, 7), tight_layout = True)
    lin_axs : list[matplotlib.axes.Axes] = axs.flatten()[:len(processed_snns_paths)]

    #patches = []
    for (name, snn_path), ax, (line_color, area_color) in zip(processed_snns_paths.items(), lin_axs, colors_array):
        print("\n------- plotting graph -------")
        if not os.path.exists(snn_path):
            raise Exception(f"The provided path does not exist: {snn_path}")
        print("Reloading model from:", snn_path)
        snn = HyperGraph.load(snn_path)
        print(f"Nodes count: {snn.nodes}\nEdges: {len(snn.hyperedges)}\nMean nodes per edge: {sum(he.connections() for he in snn)/len(snn.hyperedges)}")

        values = np.array([he.spike_frequency for he in snn for _ in range(he.connections())])

        # Create logarithmically spaced bins from the minimum to maximum value
        # Filter out non-positive values to avoid log10(0) or log10 of negative
        num_zeros = np.sum(values <= 0)
        if num_zeros > 0:
            print(f"WARNING: {num_zeros} zero or negative values were ignored when computing logarithmic bins.")
        positive_values = values[values > 0]
        if len(positive_values) == 0:
            raise ValueError("All values are zero or negative; cannot compute log-scale histogram.")
        if LOG_SPACE_X:
            bins = np.logspace(np.log10(min(positive_values)), np.log10(max(positive_values)), num = num_intervals + 1)
        else:
            bins = np.linspace(min(positive_values), max(positive_values), num = num_intervals + 1)

        # Count the number of values in each bin
        counts, bin_edges = np.histogram(values, bins = bins)

        # Optional: Smooth the counts with a Gaussian kernel
        #raw_counts = counts
        #local = gaussian_filter1d(counts, sigma = 2)
        #wide = gaussian_filter1d(counts, sigma = 10)
        #counts = 0.5 * raw_counts + 0.3 * local + 0.2 * wide

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
        ax.fill_between(x_points, y_points, color = area_color, alpha = 0.3) # alpha was 0.5

        # Plot the line connecting the points (the long spline)
        ax.plot(x_points, y_points, color = line_color, alpha = 0.8) # alpha was 1

        #patches.append(mpatches.Patch(facecolor = area_color, edgecolor = line_color, label = name))

        # Fit lognormal distribution to positive values
        if FIT_LOGNORM:
            shape, loc, scale = lognorm.fit(positive_values, floc = 0)
            cv = np.sqrt(np.exp(shape**2) - 1)
            print(f"Fitted lognorm params: shape = {shape:.3f}, loc = {loc:.3g}, scale (median) = {scale:.3g}, cv = {cv:.3f}")
            
            # Evaluate the fitted PDF on bin centers
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            pdf_vals = lognorm.pdf(bin_centers, shape, loc = loc, scale = scale)

            # Scale PDF to histogram counts (so it overlays properly)
            pdf_scaled = pdf_vals * len(positive_values) * np.diff(bins)
            #pdf_scaled = np.clip(pdf_scaled, min(counts), max(counts))

            # Overlay the fitted curve
            lnorm, = ax.plot(bin_centers, pdf_scaled, 'k--', linewidth = 2, label = "Lognormal fit")
            lnorm.set_in_layout(False)
            
            # Add text inside the plot
            x_pos = bin_centers[3*len(pdf_scaled)//10]*1.05
            y_pos = pdf_scaled[3*len(pdf_scaled)//10]
            ax.text(
                x_pos, y_pos,
                f"Median = {scale:.2f}, Coeff. of Variation = {cv:.2f}",
                fontsize = IN_PLOT_TEXT_FONTSIZE, color = "black", # same color as curve
                ha = "left", va = "bottom"
            )

        # Set axis to be logarithmic
        ax.autoscale(enable = True, axis = 'y', tight = True)
        if LOG_SPACE_X:
            ax.set_xscale('log')
        if LOG_SPACE_Y:
            ax.set_yscale('log')
            #ax.set_ylim(bottom = y_points.min(where = np.greater(y_points, 0), initial = math.inf))
            low, high = np.percentile(y_points, [5, 95])
            ax.set_ylim(bottom = y_points.min(where = np.where((y_points > low) & (y_points <= high), True, False), initial = math.inf))

        # Show grid only for the Y axis (with low alpha) and hide X axis grid
        ax.grid(axis = 'y', alpha = 0.3)
        ax.grid(axis = 'x', visible = False)

        # Labels and title
        ax.set_xlabel("Spike Frequency") # [Hz]
        ax.set_ylabel("Neurons Count")
        ax.set_title(f"SNN: {rename_title(name)}")

    # Global title
    fig.suptitle(f"Histograms of Spike Frequencies ({num_intervals} bins)", fontsize = FONTSIZE + 2)

    # Add the custom legend
    #fig.legend(patches, loc = 'lower center')

    #plt.xlim(0.9, 100)

    # Show the plot
    #plt.tight_layout()
    if SAVE:
        filename = SAVE_PATH
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