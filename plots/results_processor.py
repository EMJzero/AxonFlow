from typing import TypeVar, Any, Optional
from types import FrameType

from prettytable import PrettyTable

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
        "quiet": args_match_and_remove(["-q", "--quiet"]),
    }
    return options

def help_options() -> None:
    print("Supported options:")
    print("-h, --help\t\tDisplay this help menu.")
    print("-i --interactive\tOnce exploration has finished, instead of terminating the program, enter Python's interactive mode.")
    print(("-d, --dir <path>\tPath to the directory (folder) containing one or more '.json' files, each being the output of a run of 'targeted_main.py'."
        "Only files immediately inside the directory (no nested directories) and with the '.json' extension will be considered."))
    print("-q, --quiet\t\tDisable verbose logging of optimization functions.")


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
        snns = []

        # custom SNNs lists
        layered = ["16k_model", "lenet", "64k_model", "256k_model", "vgg11", "alexnet", "1M_model", "mobilenet"]
        cyclic = ["16_rand", "64k_rand", "256k_rand", "allen_v1"]
        small = ["16k_model", "lenet", "16_rand", "64k_rand", "64k_model", "256k_rand"]
        large = ["allen_v1", "256k_model", "vgg11", "alexnet", "1M_model", "mobilenet"]

        # Metrics to collect
        energy : dict[str, list[Optional[float]]] = defaultdict(list)
        latency : dict[str, list[Optional[float]]] = defaultdict(list)
        congestion : dict[str, list[Optional[float]]] = defaultdict(list)
        times : dict[str, list[Optional[float]]] = defaultdict(list)
        connectivity : dict[str, list[Optional[float]]] = defaultdict(list)
        part_times : dict[str, list[Optional[float]]] = defaultdict(list)
        init_energy : dict[str, list[Optional[float]]] = defaultdict(list)
        init_latency : dict[str, list[Optional[float]]] = defaultdict(list)
        init_congestion : dict[str, list[Optional[float]]] = defaultdict(list)

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

                if "note" in entry:
                    print(f"Failed entry '{file}' -> '{name}', note content:\n\t{entry['note']}")
                    continue

                technique = name if '-' in name else name + "-NA"
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
            snns.append(label)

            for technique in techniques:
                entry = technique_entries.get(technique)
                if entry:
                    energy[technique].append(entry.get("plac_energy", None))
                    latency[technique].append(entry.get("plac_avg_lat", None))
                    congestion[technique].append(entry.get("plac_avg_cong", None))
                    times[technique].append(entry.get("time", None))
                    connectivity[technique].append(entry.get("part_cost", None))
                    part_times[technique].append(entry.get("part_time", None))
                    init_energy[technique].append(entry.get("init_energy", None))
                    init_latency[technique].append(entry.get("init_avg_lat", None))
                    init_congestion[technique].append(entry.get("init_avg_cong", None))
                else:
                    energy[technique].append(None)
                    latency[technique].append(None)
                    congestion[technique].append(None)
                    times[technique].append(None)
                    connectivity[technique].append(None)
                    part_times[technique].append(None)
                    init_energy[technique].append(None)
                    init_latency[technique].append(None)
                    init_congestion[technique].append(None)
        
        # Optional: use this to override the average mechanism to have only layered or cyclic SNNs
        #snns = layered
        
        # compute energy-delay product
        energy_delay_product : dict[str, list[Optional[float]]] = {}
        init_energy_delay_product : dict[str, list[Optional[float]]] = {}
        for technique in techniques:
            energy_delay_product[technique] = list(map(lambda c : c[0] * c[1] if c[0] != None and c[1] != None else None, zip(energy[technique], latency[technique])))
            init_energy_delay_product[technique] = list(map(lambda c : c[0] * c[1] if c[0] != None and c[1] != None else None, zip(init_energy[technique], init_latency[technique])))
        
        # Replace 'None' with zero
        for technique in techniques:
            energy[technique] = list(map(lambda x : x if x != None else math.nan, energy[technique]))
            latency[technique] = list(map(lambda x : x if x != None else math.nan, latency[technique]))
            congestion[technique] = list(map(lambda x : x if x != None else math.nan, congestion[technique]))
            times[technique] = list(map(lambda x : x if x != None else math.nan, times[technique]))
            connectivity[technique] = list(map(lambda x : x if x != None else math.nan, connectivity[technique]))
            part_times[technique] = list(map(lambda x : x if x != None else math.nan, part_times[technique]))
            energy_delay_product[technique] = list(map(lambda x : x if x != None else math.nan, energy_delay_product[technique]))
            init_energy[technique] = list(map(lambda x : x if x != None else math.nan, init_energy[technique]))
            init_latency[technique] = list(map(lambda x : x if x != None else math.nan, init_latency[technique]))
            init_congestion[technique] = list(map(lambda x : x if x != None else math.nan, init_congestion[technique]))
            init_energy_delay_product[technique] = list(map(lambda x : x if x != None else math.nan, init_energy_delay_product[technique]))
        
        # compute average metrics across SNNs
        energy_avg, latency_avg, congestion_avg, connectivity_avg, energy_delay_product_avg = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
        for i, snn in enumerate(snns):
            for tech in techniques:
                energy_avg[tech] += energy[tech][i]
                latency_avg[tech] += latency[tech][i]
                congestion_avg[tech] += congestion[tech][i]
                connectivity_avg[tech] += connectivity[tech][i]
                energy_delay_product_avg[tech] += energy_delay_product[tech][i]
        for tech in techniques:
            energy_avg[tech] /= len(snns)
            latency_avg[tech] /= len(snns)
            energy_delay_product_avg[tech] /= len(snns)
            congestion_avg[tech] /= len(snns)
            connectivity_avg[tech] /= len(snns)
        
        # normalize w.r.t. the best partitioning
        best_energy = [min([energy[technique][i] for technique in techniques if not math.isnan(energy[technique][i])], default = 0) for i in range(len(snns))]
        best_latency = [min([latency[technique][i] for technique in techniques if not math.isnan(latency[technique][i])], default = 0) for i in range(len(snns))]
        best_congestion = [min([congestion[technique][i] for technique in techniques if not math.isnan(congestion[technique][i])], default = 0) for i in range(len(snns))]
        best_connectivity = [min([connectivity[technique][i] for technique in techniques if not math.isnan(connectivity[technique][i])], default = 0) for i in range(len(snns))]
        best_energy_delay_product = [min([energy_delay_product[technique][i] for technique in techniques if not math.isnan(energy_delay_product[technique][i])], default = 0) for i in range(len(snns))]
        for technique in techniques:
            energy[technique] = list(map(lambda c : c[0] / c[1] if not math.isnan(c[0]) else math.nan, zip(energy[technique], best_energy)))
            latency[technique] = list(map(lambda c : c[0] / c[1] if not math.isnan(c[0]) else math.nan, zip(latency[technique], best_latency)))
            congestion[technique] = list(map(lambda c : c[0] / c[1] if not math.isnan(c[0]) else math.nan, zip(congestion[technique], best_congestion)))
            connectivity[technique] = list(map(lambda c : c[0] / c[1] if not math.isnan(c[0]) else math.nan, zip(connectivity[technique], best_connectivity)))
            energy_delay_product[technique] = list(map(lambda c : c[0] / c[1] if not math.isnan(c[0]) else math.nan, zip(energy_delay_product[technique], best_energy_delay_product)))
            # still normalize them w.r.t. the eventual best for their metric since they will be in the same plot
            init_energy[technique] = list(map(lambda c : c[0] / c[1] if not math.isnan(c[0]) else math.nan, zip(init_energy[technique], best_energy)))
            init_latency[technique] = list(map(lambda c : c[0] / c[1] if not math.isnan(c[0]) else math.nan, zip(init_latency[technique], best_latency)))
            init_congestion[technique] = list(map(lambda c : c[0] / c[1] if not math.isnan(c[0]) else math.nan, zip(init_congestion[technique], best_congestion)))
            init_energy_delay_product[technique] = list(map(lambda c : c[0] / c[1] if not math.isnan(c[0]) else math.nan, zip(init_energy_delay_product[technique], best_energy_delay_product)))
        # also normalize the averages
        best_avg_energy = min([energy_avg[technique] for technique in techniques if not math.isnan(energy_avg[technique])], default = 0)
        best_avg_latency = min([latency_avg[technique] for technique in techniques if not math.isnan(latency_avg[technique])], default = 0)
        best_avg_congestion = min([congestion_avg[technique] for technique in techniques if not math.isnan(congestion_avg[technique])], default = 0)
        best_avg_connectivity = min([connectivity_avg[technique] for technique in techniques if not math.isnan(connectivity_avg[technique])], default = 0)
        best_avg_energy_delay_product = min([energy_delay_product_avg[technique] for technique in techniques if not math.isnan(energy_delay_product_avg[technique])], default = 0)
        for tech in techniques:
            energy_avg[tech] /= best_avg_energy
            latency_avg[tech] /= best_avg_latency
            congestion_avg[tech] /= best_avg_congestion
            connectivity_avg[tech] /= best_avg_connectivity
            energy_delay_product_avg[tech] /= best_avg_energy_delay_product

        techniques = sorted(techniques, key = lambda s : (''.join(chr(255 - ord(c)) for c in s.split('-', 1)[0]), s.split('-', 1)[1])) # descending order on the word before the first '-', then ascending order as a tiebreak.

        def formatter(f : float):
            if f == 1.0:
                return "-> 1.0 <-"
            return f"{f:.3f}"

        # TODO:
        # - per-technique average across SNNs
        # - per-technique average across cyclic-only SNNs
        # - per-technique average across layered-only SNNs
        # - per-technique average across small-target SNNs
        # - per-technique average across large-target SNNs
        # - optional filter: keep best placement for each partitioning for each SNN

        if False:
            print("PER-SNN ALL RESULTS:")
            for i, snn in enumerate(snns):
                print("==>", snn)
                table = PrettyTable(["part tech", "plac tech", "energy", "latency", "ELP", "congestion", "tot-time", "connectivity", "part-time"])
                for tech in techniques:
                    part, plac = tech.split('-', 1)
                    table.add_row([part, plac] + list(map(formatter, [
                        energy[tech][i],
                        latency[tech][i],
                        energy_delay_product[tech][i],
                        congestion[tech][i],
                        times[tech][i],
                        connectivity[tech][i],
                        part_times[tech][i]
                    ])))
                print(table)
        
        if True:
            print("\nACROSS SNNs AVERAGE RESULTS:")
            table = PrettyTable(["part tech", "plac tech", "energy", "latency", "ELP", "congestion", "connectivity"])
            for tech in techniques:
                part, plac = tech.split('-', 1)
                table.add_row([part, plac] + list(map(formatter, [
                    energy_avg[tech],
                    latency_avg[tech],
                    energy_delay_product_avg[tech],
                    congestion_avg[tech],
                    connectivity_avg[tech]
                ])))
            print(table)
        
        if True:
            print("PER-SNN BEST (BY ELP) PLACEMENT FOR EACH PARTITIONING TECH:")
            for i, snn in enumerate(snns):
                print("==>", snn)
                best_per_part = {}
                table = PrettyTable(["part tech", "plac tech", "energy", "latency", "ELP", "congestion", "tot-time", "connectivity", "part-time"])
                for tech in techniques:
                    part, plac = tech.split('-', 1)
                    if part not in best_per_part or energy_delay_product[best_per_part[part]][i] > energy_delay_product[tech][i]:
                        best_per_part[part] = tech
                for tech in best_per_part.values():
                    part, plac = tech.split('-', 1)
                    table.add_row([part, plac] + list(map(formatter, [
                        energy[tech][i],
                        latency[tech][i],
                        energy_delay_product[tech][i],
                        congestion[tech][i],
                        times[tech][i],
                        connectivity[tech][i],
                        part_times[tech][i]
                    ])))
                print(table)

    except Exception:
        print(traceback.format_exc())

    if options["interactive"]:
        print("\n------ interactive mode ------")
        in_interactive_mode = True
        code.interact(local = globals())
        in_interactive_mode = False