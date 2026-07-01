from typing import TypeVar, Any, Optional, Union
from types import FrameType

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib

from sklearn.linear_model import HuberRegressor

from collections import defaultdict
from itertools import cycle
import numpy as np
import traceback
import signal
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

# CLI MANAGEMENT

in_interactive_mode = False


def signal_handler(signal: int, frame: Optional[FrameType]) -> None:
    global in_interactive_mode
    if in_interactive_mode:
        print('EXITING...')
        sys.exit(0)
    else:
        print('\nHANDLING TERMINATION...\n')
        stack = traceback.format_stack(frame)
        print('------------ stack -----------')
        print(''.join(stack[:-1])[:-1])
        print('------------------------------')
        time.sleep(0.2)
        print('\nTERMINATION RECEIVED - SWITCHING TO INTERACTIVE MODE\n[type "exit()" or press "ctrl+c" again to terminate the program]\n')
        in_interactive_mode = True
        code.interact(local=globals())
        in_interactive_mode = False

class CutCornerBox:
    def __init__(self, pad=0.3, cut=3.0, angle=45, rounding=0.25):
        self.pad = pad
        self.cut = cut
        self.angle = angle
        self.rounding = rounding

    def __call__(self, x0, y0, width, height, mutation_size):
        if not 0 < self.angle < 90:
            raise ValueError("angle must be between 0 and 90 degrees")
        pad = mutation_size * self.pad
        r = mutation_size * self.rounding
        x0 -= pad
        y0 -= pad
        width += 2 * pad
        height += 2 * pad
        x1 = x0 + width
        y1 = y0 + height
        dx = mutation_size * self.cut
        dy = dx * math.tan(math.radians(self.angle))
        scale = min(1, width / dx, height / dy)
        dx *= scale
        dy *= scale
        r = min(r, width / 2, height / 2, x1 - dx - x0, y1 - (y0 + dy))
        verts = [
            # Start on bottom edge, after bottom-left rounding
            (x0 + r, y0),
            # Bottom edge to start of cut
            (x1 - dx, y0),
            # Diagonal cut
            (x1, y0 + dy),
            # Right edge, up to top-right rounding
            (x1, y1 - r),
            # Rounded top-right corner
            (x1, y1),
            (x1 - r, y1),
            # Top edge
            (x0 + r, y1),
            # Rounded top-left corner
            (x0, y1),
            (x0, y1 - r),
            # Left edge
            (x0, y0 + r),
            # Rounded bottom-left corner
            (x0, y0),
            (x0 + r, y0),
            # Close
            (x0 + r, y0),
        ]
        codes = [
            Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
            Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY,
        ]

        return Path(verts, codes)


T = TypeVar('T')


def args_match_and_remove(flags: Union[str, list[str]], with_value: bool = False, value_type: type[T] = str, flags_tag: str = '-') -> Union[bool, T, None]:
    if isinstance(flags, str):
        flags = [flags]
    for flag in flags:
        try:
            idx = sys.argv.index(flag)
            sys.argv.pop(idx)

            if with_value:
                if idx >= len(sys.argv) or sys.argv[idx].startswith(flags_tag):
                    return None
                try:
                    value = value_type(sys.argv[idx])
                    sys.argv.pop(idx)
                    return value
                except Exception:
                    return None
            else:
                return True
        except ValueError:
            continue
    return False


def parse_options() -> dict[str, Any]:
    options = {
        "help": args_match_and_remove(["-h", "--help"]),
        "interactive": args_match_and_remove(["-i", "--interactive"]),
        "dir": args_match_and_remove(["-d", "--dir"], with_value=True),
        "save": args_match_and_remove(["-s", "--save"], with_value=True),
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


#def rename_label(label : str) -> str:
#    result = ""
#    for piece in label.split("-"):
#        if piece == "sequential": result += "ordered sequential"
#        elif piece == "hmetis": result += "hierarchical"
#        elif piece == "hehiding": result += "h-edge overlap"
#        elif piece == "hilbert": result += "hilbert"
#        elif piece == "spectral": result += "spectral"
#        elif piece == "truenorth": result += "minimum distance"
#        elif piece == "fd": result += "force-directed"
#        elif piece == "ps": result += "particle swarm"
#        else:
#            result += piece
#            print("Could not fully rename label:", label, "-> technique note recognized:", piece)
#        result += " + "
#    return result[:-3]

def rename_label(label : str) -> str:
    if label == "sequential": return "ordered sequential"
    if label == "unordered": return "unordered sequential"
    if label == "edgehiding": return "edgemap"
    if label == "hmetis": return "hierarchical"
    if label == "hehiding": return "h-edge overlap"
    print("Could not rename label:", label)
    return label


# MATPLOTLIB SETTINGS

SUPPORTED_EXTENSIONS = ['.pdf', '.eps', '.svg', '.png']
DPI = 300
FONTSIZE = 16.3

# Regression toggle:
#   'linear'    -> old fit: log10(time) = m*log10(pins) + b
#   'quadratic' -> new fit: log10(time) = a*log10(pins)^2 + b*log10(pins) + c,
#                  with fallback to the old linear fit when curvature is tiny.
FIT_MODE = 'linear'

# Only used when FIT_MODE == 'quadratic'. Increase this to fall back more often.
# Decrease it to keep subtler curvature.
QUADRATIC_FLAT_EPS = 1e-3

font = {'family': 'sans-serif', 'weight': 'normal', 'size': FONTSIZE}
matplotlib.rc('font', **font)
plt.rcParams.update({
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
})


def clean_positive_xy(x, y):
    """Return x/y arrays filtered to positive finite pairs, required by log-log fitting."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def fit_power_law_loglog_huber(x, y, epsilon=1.35, return_info=False):
    """
    Robustly fit: y = C * x^m
    Equivalent log-space model: log10(y) = m * log10(x) + b
    Uses Huber regression, which behaves like least squares for normal points
    but downweights large residuals.
    Returns:
        m, b
    or, if return_info=True:
        m, b, info
    """
    x, y = clean_positive_xy(x, y)

    if len(x) < 2:
        return (None, None, None) if return_info else (None, None)

    lx = np.log10(x)
    ly = np.log10(y)

    # Need at least two distinct x-values to fit a slope.
    if len(np.unique(lx)) < 2:
        return (None, None, None) if return_info else (None, None)

    model = HuberRegressor(
        epsilon=epsilon,
        alpha=0.0,          # no regularization; closer to polyfit behavior
        fit_intercept=True,
        max_iter=1000
    )

    model.fit(lx.reshape(-1, 1), ly)

    m = float(model.coef_[0])
    b = float(model.intercept_)

    if return_info:
        info = {
            "outlier_mask": model.outliers_,
            "scale": model.scale_,
            "n_iter": model.n_iter_,
            "epsilon": epsilon,
        }
        return m, b, info

    return m, b

def fit_quadratic_loglog_with_linear_fallback(x, y, quadratic_flat_eps=QUADRATIC_FLAT_EPS):
    """
    Fit a quadratic curve in log-log space: log10(y) = a*log10(x)^2 + b*log10(x) + c
    If abs(a) is very small, the curve is effectively flat in curvature and this falls back to the old linear power-law model.
    Returns:
        ("quadratic", (a, b, c))
        ("linear", (m, b))
        (None, None)
    """
    x, y = clean_positive_xy(x, y)
    if len(x) < 2:
        return None, None

    lx = np.log10(x)
    ly = np.log10(y)

    if len(x) >= 3:
        a, b, c = np.polyfit(lx, ly, 2)
        if abs(a) >= quadratic_flat_eps:
            return "quadratic", (a, b, c)

    m, b = np.polyfit(lx, ly, 1)
    return "linear", (m, b)


def fit_selected_model_loglog(x, y):
    """Dispatch fit according to FIT_MODE."""
    if FIT_MODE == 'linear':
        m, b = fit_power_law_loglog_huber(x, y)
        if m is None:
            return None, None
        return 'linear', (m, b)

    if FIT_MODE == 'quadratic':
        return fit_quadratic_loglog_with_linear_fallback(x, y)

    raise ValueError(f"Unsupported FIT_MODE '{FIT_MODE}'. Use 'linear' or 'quadratic'.")


def valid_number(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and np.isfinite(float(value))


def positive_number(value) -> bool:
    return valid_number(value) and float(value) > 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    options = parse_options()

    if options["help"]:
        print("------------ HELP ------------")
        help_options()
        print("------------------------------")
        sys.exit(0)

    try:
        # Load JSON data
        path = options["dir"]
        if not path:
            print("No directory path provided, option '-d' or '--dir' is mandatory.")
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
        file_data = []  # tuples: (pins, filename, technique_entries)
        pins_by_file = []

        # Execution-time metrics kept from the mapping script.
        part_times: dict[str, list[Optional[float]]] = defaultdict(list)
        plac_times: dict[str, list[Optional[float]]] = defaultdict(list)
        total_times: dict[str, list[Optional[float]]] = defaultdict(list)

        # Optional: specify techniques to omit
        omit_techniques = {"sequential-truenorth"}
        # Optional: specify techniques to put first in the order, others will follow in descending alphabetical order
        forceful_order = ["sequential", "unordered", "edgehiding"]
        # Must: decide x-axis SNNs sort order, options are "nodes", "connections"/"edges"/"pins"
        x_axis_order = "pins"

        # Read files
        for file in files:
            #if "rand" in file or "allen" in file: continue
            file_path = os.path.join(path, file)

            with open(file_path, "r") as f:
                print("Parsing:", file_path)
                data: list[dict[str, float]] = json.load(f)

            pins = None
            technique_entries = {}

            for entry in data:
                name = entry["name"]
                technique = name.split("-", 1)[0]
                if technique in technique_entries:
                    print(f"Already seen technique '{technique}', skipping '{name}'...")
                    continue

                if name in omit_techniques:
                    print(f"Omitting technique '{name}'...")
                    continue

                if "note" in entry:
                    print(f"Failed entry '{file}' -> '{name}', note content:\n\t{entry['note']}")
                    continue

                techniques.add(technique)
                technique_entries[technique] = entry

                if pins is None:
                    if x_axis_order == "nodes" and "graph_nodes" in entry:
                        pins = entry["graph_nodes"]
                    elif "graph_pins" in entry:
                        pins = entry["graph_pins"]
                    elif "pins" in entry:
                        pins = entry["pins"]
                    elif "graph_edges" in entry:
                        # The original mapping script used graph_edges when x_axis_order was "pins".
                        pins = entry["graph_edges"]

                if not entry.get("part_valid", True):
                    print(f"WARNING: Invalid partitioning for {file} -> {name}")
                if not entry.get("plac_valid", True):
                    print(f"WARNING: Invalid placement for {file} -> {name}")

            if positive_number(pins):
                file_data.append((float(pins), file, technique_entries))
            else:
                print(f"WARNING: Could not determine positive pins count for {file}")

        file_data.sort(key=lambda x: x[0])

        # Extract execution-time data in increasing pins order.
        for pins, file, technique_entries in file_data:
            pins_by_file.append(pins)

            for technique in techniques:
                entry = technique_entries.get(technique)
                if not entry:
                    part_times[technique].append(None)
                    plac_times[technique].append(None)
                    total_times[technique].append(None)
                    continue

                part_time = entry.get("part_time", None)
                plac_time = 0 #entry.get("plac_time", None)
                recorded_total_time = entry.get("part_time", None) #entry.get("time", None)

                # Keep the original mapping behavior: if placement time is absent,
                # infer it from total time minus partitioning time when possible.
                if not positive_number(plac_time) and valid_number(recorded_total_time) and valid_number(part_time):
                    plac_time = float(recorded_total_time) - float(part_time)

                # Total mapping time is the sum of placement and partitioning when both are available.
                # Fall back to an explicit total-time field if the decomposed values are not available.
                if valid_number(part_time) and valid_number(plac_time):
                    total_time = float(part_time) + float(plac_time)
                elif valid_number(recorded_total_time):
                    total_time = float(recorded_total_time)
                else:
                    total_time = None

                part_times[technique].append(float(part_time) if valid_number(part_time) else None)
                plac_times[technique].append(float(plac_time) if valid_number(plac_time) else None)
                total_times[technique].append(float(total_time) if valid_number(total_time) else None)

        # Replace None / invalid / non-positive values with NaN for plotting and fitting.
        for technique in techniques:
            part_times[technique] = [float(x) if positive_number(x) else math.nan for x in part_times[technique]]
            plac_times[technique] = [float(x) if positive_number(x) else math.nan for x in plac_times[technique]]
            total_times[technique] = [float(x) if positive_number(x) else math.nan for x in total_times[technique]]

        techniques = sorted(techniques, reverse=True)
        for fo in forceful_order[::-1]:
            if fo in techniques:
                techniques.remove(fo)
                techniques.insert(0, fo)

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
            "#9BBD00", # LIME
            #"#768E0B", # DARK LIME
        ]
        possible_markers = cycle(['^', 'v', '*', 'o', 'p', 's', 'X', 'D'])
        colors = cycle(possible_colors)
        style = {
            technique: {
                "color": next(colors),
                "marker": next(possible_markers),
                "alpha": 0.85,
            }
            for technique in techniques
        }

        if len(pins_by_file) == 0:
            raise Exception("No positive pins counts found. Cannot create log-log scaling plot.")

        # FIGURE DEFINED HEREEEEEE!!!!!
        #fig, ax = plt.subplots(figsize=(5.5, 5.5), tight_layout = True)
        fig, ax = plt.subplots(figsize=(7, 6), tight_layout = True)

        pins_arr = np.asarray(pins_by_file, dtype=float)
        any_plotted = False

        for technique in techniques:
            times_arr = np.asarray(total_times[technique], dtype=float)
            mask = (pins_arr > 0) & (times_arr > 0) & np.isfinite(pins_arr) & np.isfinite(times_arr)
            x = pins_arr[mask]
            y = times_arr[mask]

            if len(x) == 0:
                print(f"Skipping {technique}: no valid positive total-time data.")
                continue

            any_plotted = True
            label = rename_label(technique)
            plt.scatter(
                x,
                y,
                label=label,
                marker=style[technique]["marker"],
                color=style[technique]["color"],
                alpha=style[technique]["alpha"],
            )

            fit_kind, coeffs = fit_selected_model_loglog(x, y)
            if fit_kind is not None:
                x_fit = np.logspace(np.log10(np.min(x)), np.log10(np.max(pins_arr)), 200)
                lx_fit = np.log10(x_fit)

                if fit_kind == "quadratic":
                    a, b, c = coeffs
                    y_fit = 10 ** (a * lx_fit**2 + b * lx_fit + c)
                    linestyle = "-."
                    print(
                        f"{technique} quadratic log-log scaling: "
                        f"log10(time) = {a:.4e}*log10(pins)^2 + {b:.4f}*log10(pins) + {c:.4f}"
                    )
                else:
                    m, b = coeffs
                    y_fit = 10 ** (m * lx_fit + b)
                    linestyle = "--"
                    print(f"{technique} linear log-log scaling: time = {10 ** b:.4e} * pins^{m:.4f}")

                plt.plot(
                    x_fit,
                    y_fit,
                    linestyle=linestyle,
                    color=style[technique]["color"],
                    alpha=0.9,
                )
            else:
                print(f"Skipping regression for {technique}: need at least two valid positive points.")

        if not any_plotted:
            raise Exception("No valid positive total-time data found. Cannot create log-log scaling plot.")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of Connections")
        plt.ylabel("Partitioning Time [s]")
        #plt.title(f"Mapping Time vs Pins (Log-Log Scaling)")
        #plt.title(f"Mapping Time vs Pins")
        plt.title(f"Partitioning Time vs Connections")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        
        handles, labels = ax.get_legend_handles_labels()
        ncols = 1
        leg = plt.legend( # was fig.legend
            handles,
            labels,
            loc= "upper left", #"lower center",
            ncol=ncols,
            handlelength=0.5, # was 4.3
            handletextpad=0.4,
            columnspacing=1.4,
            labelspacing=0.4,
            borderpad=0.3,
            frameon=True,
            fancybox=False,
        )
        #leg.get_frame().set_boxstyle(CutCornerBox(pad=0.03, cut=5.5, angle=27.5, rounding=0.25))
        #plt.tight_layout(rect=[0, 0.16, 1, 1])

        if options["save"]:
            filename = options["save"]
            if not any(filename.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                print("WARNING: the provided filename was missing a valid extension, adding '.png' by default.")
                filename += '.png'
            idx = 1
            while os.path.isfile(filename):
                fn1, fn2 = filename.rsplit('.', 1)
                filename = fn1 + f"_{idx}." + fn2
                idx += 1
            plt.savefig(filename, dpi=DPI)
        else:
            plt.show()
    except Exception:
        print(traceback.format_exc())

    if options["interactive"]:
        print("\n------ interactive mode ------")
        in_interactive_mode = True
        code.interact(local=globals())
        in_interactive_mode = False