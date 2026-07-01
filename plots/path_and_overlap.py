from matplotlib.ticker import LogLocator, FuncFormatter
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.ndimage import gaussian_filter1d
from functools import reduce
from typing import Union
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
BAR_WIDTH = 0.4

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
    data = {
        "lenet":      {"average_path_length": 3.191, "average_hedge_overlap": 0.649},
        #"16k_rand":   {"average_path_length": 3.091, "average_hedge_overlap": 0.068}, # pre-3D-generator
        "16k_rand":   {"average_path_length": 3.194, "average_hedge_overlap": 1.103},
        "16k_model":  {"average_path_length": 3.310, "average_hedge_overlap": 0.134},
        #"64k_rand":   {"average_path_length": 2.968, "average_hedge_overlap": 0.066},
        "64k_rand":   {"average_path_length": 3.243, "average_hedge_overlap": 0.603},
        "64k_model":  {"average_path_length": 2.956, "average_hedge_overlap": 0.704},
        "vgg11":      {"average_path_length": 6.933, "average_hedge_overlap": 7.722},
        "alexnet":    {"average_path_length": 4.232, "average_hedge_overlap": 6.593},
        "256k_model": {"average_path_length": 2.926, "average_hedge_overlap": 1.444},
        "allen_v1":   {"average_path_length": 4.436, "average_hedge_overlap": 0.489},
        #"256k_rand":  {"average_path_length": 2.941, "average_hedge_overlap": 0.065},
        "256k_rand":  {"average_path_length": 3.464, "average_hedge_overlap": 0.262},
        "1M_model":   {"average_path_length": 3.732, "average_hedge_overlap": 3.676},
        "mobilenet":  {"average_path_length": 18.549, "average_hedge_overlap": 0.005},
    }
    
    #order = ["lenet", "16k_rand", "16k_model", "64k_rand", "64k_model", "vgg11", "alexnet", "256k_model", "allen_v1", "256k_rand", "1M_model", "mobilenet"]
    order = ["16k_model", "lenet", "16k_rand", "64k_rand", "64k_model", "256k_rand", "allen_v1", "256k_model", "vgg11", "alexnet", "1M_model", "mobilenet"]
    for name in order:
        tmp = data.pop(name)
        data[name] = tmp

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
    
    ax1 : matplotlib.axes.Axes = None
    ax2 : matplotlib.axes.Axes = None
    #fig, ax1 = plt.subplots(1, 1, figsize = (16, 10), tight_layout = True)
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 10), tight_layout = True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 7), tight_layout = True)
    
    # Decide the y-axis bounds by ignoring outliers (lower sigma is more brutal)
    def set_bounds(ax : matplotlib.axes.Axes, metric : str, shapes : Optional[list[matplotlib.patches.Patch]], sigma : float = 0.5, margin : float = 0.2):
        data_by_x = defaultdict(list)
        for xi, value in data.items():
            value = value[metric]
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
    
    style_path = {"color": possible_colors[2], "edgecolor": "white"}
    style_overlap = {"color": possible_colors[3], "edgecolor": "white"}

    index = np.arange(len(data))
    offset = 0 #1/2 # (number of bars per group - 1)/2
    x_indices = list(range(len(data)))

    rects = []
    rects += ax1.bar(index + (0 - offset) * BAR_WIDTH, [d["average_path_length"] for d in data.values()], BAR_WIDTH, label = "expected path length", **style_path)
    #rects += ax1.bar(index + (1 - offset) * BAR_WIDTH, [d["average_hedge_overlap"] for d in data.values()], BAR_WIDTH, label = "average hyperedge overlap", **style_overlap)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(data.keys(), rotation = 45)
    ax1.set_xlabel("SNN (least → most nodes)")
    #ax1.set_ylabel("SNN Average Hyperedge Overlap and Path Length")
    ax1.set_ylabel("Average Path Length")
    #ax1.legend()
    format_y_bars(ax1)
    set_bounds(ax1, "average_path_length", rects)
    ax1.set_ylim(2.0, 20.0)
    
    rects += ax2.bar(index + (0 - offset) * BAR_WIDTH, [d["average_hedge_overlap"] for d in data.values()], BAR_WIDTH, label = "average hyperedge overlap", **style_overlap)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(data.keys(), rotation = 45)
    ax2.set_xlabel("SNN (least → most nodes)")
    ax2.set_ylabel("Average Hyperedge Overlap\n(# common destinations)")
    #ax2.legend()
    #format_y_bars(ax2, which = [1.0, 2.0, 4.0, 6.0, 8.0])
    format_y_bars(ax2, which = [2.0, 4.0, 6.0, 8.0])
    set_bounds(ax2, "average_path_length", rects)
    ax2.set_ylim(0.004, 8.0)

    # Show the plot
    fig.suptitle("Additional SNN Hypergraph Properties")
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