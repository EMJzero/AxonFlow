from collections import defaultdict
from prettytable import PrettyTable
from itertools import product
import numpy as np
import time
import math
import re
import os

from settings import *
from prints import *
from utils import *
from snn import *

# HOW-TO IMPORT FROM SNN Toolbox:
# - GitHub: https://github.com/NeuromorphicProcessorProject/snn_toolbox
# - get SNN Toolbox up and running (good luck ;p), you can find the python3.8.20 requirements for it in 'snn_models/snn_toolbox_requirements.txt'
# - prepare a python script to run the tool, see 'snn_models/snn_toolbox_example.py' for an example
# - from the toolbox's output folder (usually './temp') recover, rename, and place all in the same folder the following files:
#   - x_norm.npz -> {modelname}_input.npz
#   - {modelname}.graphml
#   - /log/gui/test/log_vars/0.pkl -> {modelname}_0.npz
# - to speedup subsequent runs, run AxonFlow's main to import and save the preprocessed model:
#   - python3.13 main.py -d -l folder/{modelname} -s folder/{modelname}_processed
# - run experiments while re-importing the preprocessed model:
#   - python3.13 targeted_main.py -r folder/{modelname}_preprocessed

"""
Given a path relative to this script or absolute pointing to a GraphML graph,
loads it and returns it as an hypergraph.
The graph must have the attribute 'spike_frequency' on either nodes or edges.
"""
@core
def loadSNNGraphML(path : str) -> HyperGraph:
    import networkx

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise Exception(f"The provided path does not exist: {path}")
    elif not os.path.isfile(path):
        raise Exception(f"The provided path is not a file: {path}")
    elif path.split('.')[-1] != "graphml":
        print("WARNING: the provided file does not have the '.graphml' extension. Are you sure it stores a graph?")
    
    print("Loading SNN from:", path)
    g = networkx.read_graphml(path)
    if not isinstance(g, networkx.DiGraph) and not isinstance(g, networkx.MultiDiGraph):
        raise Exception(f"The provided graph does not get loaded as neither a DiGraph nor a MultiDiGraph instance by NetworkX. Its current class is {type(g)}.")
    
    print("Building HyperGraph:", g.name if g.name else f"NoName@{os.path.basename(path)}")
    nodes = {n : i for i, n in enumerate(g.nodes)} # node id -> node index
    hyperedges = defaultdict(list) # source node id -> list of destinations ids
    nodes_spike_frequencies = networkx.get_node_attributes(g, "spike_frequency") # node id -> spike frequency
    #edge_weights = networkx.get_edge_attributes(g, "weight")
    for edge in g.edges:
        src, dst = edge
        hyperedges[src].append(dst)
    
    return HyperGraph(len(nodes), [HyperEdge(nodes[k], tuple(map(lambda n : nodes[n], v)), nodes_spike_frequencies[k]) for k, v in hyperedges.items()])

"""
Given a path relative to this script or absolute pointing to a GraphML graph,
manually parses it and returns its set of node names and a list of edges,
each represented as a tuple '(src, dst)'.

NOTE: edges are stored as [src1, dst1, src2, dst2, ...] to minimize overhead!
NOTE: if the first line in the file says 'compact', then the graphml format is
      ignored and lines shall contain either one node or two nodes separated
      by ';' to represent edges.
"""
@core
def manualGraphMLparser(path : str) -> tuple[dict[str, int], list[str]]:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise Exception(f"The provided path does not exist: {path}")
    elif not os.path.isfile(path):
        raise Exception(f"The provided path is not a file: {path}")
    elif path.split('.')[-1] != "graphml":
        print("WARNING: the provided file does not have the '.graphml' extension. Are you sure it stores a graph?")
    
    total_size = os.path.getsize(path)
    next_node_idx = 0
    nodes = dict() # node id -> node index
    edges = []
    lines_count = 0
    total_bytes_read = 0
    print("Manually parsing SNN from:", path, f"(size: {fileSizeString(total_size)})")
    with open(path) as file_in:
        first_line = file_in.readline().strip()
        if first_line == "compact":
            nodeline_regex = re.compile(r'^(\w+)$')
            edgeline_regex = re.compile(r'^(\w+);(\w+)$')
        else:
            if first_line != "<?xml version='1.0' encoding='utf-8'?>":
                print(f"WARNING: the first line in the provided '.graphml' file reads \"{first_line}\" rather than the expected \"<?xml version='1.0' encoding='utf-8'?>\". Are you sure it stores a graph?")
            nodeline_regex = re.compile(r'<node id="(\w+)"\/>')
            edgeline_regex = re.compile(r'<edge source="(\w+)" target="(\w+)"\/>')
        weirdline_regex = re.compile(r'(?:<[\?\w\s\'\"\-\.\:\/\=]*>)*<node id="(\w+)"\/>')
        last_print_time = time.monotonic()
        for line in file_in:
            lines_count += 1
            total_bytes_read += len(line)
            if lines_count & 1023 == 0:
                now = time.monotonic()
                if now - last_print_time >= Settings.PRINT_INTERVAL and lines_count > 0:
                    last_print_time = now
                    avg_line_len = total_bytes_read / lines_count
                    estimated_total = int(total_size / avg_line_len)
                    print(f"Processed {lines_count} / ~{estimated_total} lines...")
            # assume line starts with the pattern
            match = nodeline_regex.match(line)
            if match:
                # MAYBE: internalize all node names to save memory
                #node = sys.intern(match.group(1))
                #nodes.add(node)
                nodes[match.group(1)] = next_node_idx
                next_node_idx += 1
                continue
            match = edgeline_regex.match(line)
            if match:
                edges.append(match.group(1))
                edges.append(match.group(2))
                continue
            # fall back on the weirdline regex
            match = weirdline_regex.search(line)
            if match:
                nodes[match.group(1)] = next_node_idx
                next_node_idx += 1
                continue
            print("Unmatched line:", line.strip())
    return nodes, edges

"""
Given a path relative to this script or absolute pointing to a NumPy array file
that has been generated as a "spiketrains var_log" from "SNN toolbox", a NumPy file
containing the matching input data, and a matching ".graphml" file with one node
per network neuron and connection matching the computation, loads them and returns
the resulting SNN with spike frequencies as an hypergraph.
"""
@core
def loadSNNcomposite(npz_log_path : str, npz_input_path : str, graphml_path : str) -> HyperGraph:
    npz_log_path = os.path.abspath(npz_log_path)
    if not os.path.exists(npz_log_path):
        raise Exception(f"The provided path does not exist: {npz_log_path}")
    elif not os.path.isfile(npz_log_path):
        raise Exception(f"The provided path is not a file: {npz_log_path}")
    elif npz_log_path.split('.')[-1] != "npz":
        print(f"WARNING: the '{npz_log_path}' file does not have the '.npz' extension. Are you sure it is a report from \"SNN toolbox\"?")
    
    print("Loading SNN log from:", npz_log_path)
    data = np.load(npz_log_path, allow_pickle = True)

    npz_input_path = os.path.abspath(npz_input_path)
    if not os.path.exists(npz_input_path):
        raise Exception(f"The provided path does not exist: {npz_input_path}")
    elif not os.path.isfile(npz_input_path):
        raise Exception(f"The provided path is not a file: {npz_input_path}")
    elif npz_input_path.split('.')[-1] != "npz":
        print(f"WARNING: the '{npz_input_path}' file does not have the '.npz' extension. Are you sure it is a report from \"SNN toolbox\"?")

    print("Loading SNN input from:", npz_input_path)
    input = np.load(npz_input_path, allow_pickle = True)

    table = PrettyTable(["Array Name", "Shape", "Data Type", "File"])
    table.border = False
    table.preserve_internal_border = True
    print(f"Arrays in files '{os.path.basename(npz_log_path)}', '{os.path.basename(npz_input_path)}':")
    for array_name in (data.files if not isinstance(data, dict) else data.keys()):
        array = np.array(data[array_name], dtype = object)
        table.add_row([array_name, array.shape, array.dtype, os.path.basename(npz_log_path)])
    for array_name in (input.files if not isinstance(input, dict) else input.keys()):
        array = np.array(input[array_name], dtype = object)
        table.add_row([array_name, array.shape, array.dtype, os.path.basename(npz_input_path)])
    print(table)
    
    if "spiketrains_n_b_l_t" not in (data.files if not isinstance(data, dict) else data.keys()):
        raise Exception(f"The '{npz_log_path}' file does not contain an 'spiketrains_n_b_l_t' array.")

    if "arr_0" not in (input.files if not isinstance(input, dict) else input.keys()):
        raise Exception(f"The '{npz_input_path}' file does not contain an 'arr_0' array.")

    graphml_path = os.path.abspath(graphml_path)
    if not os.path.exists(graphml_path):
        raise Exception(f"The provided path does not exist: {graphml_path}")
    elif not os.path.isfile(graphml_path):
        raise Exception(f"The provided path is not a file: {graphml_path}")
    elif graphml_path.split('.')[-1] != "graphml":
        print("WARNING: the provided file does not have the '.graphml' extension. Are you sure it stores a graph?")
    
    print("Loading SNN graph from:", graphml_path)
    #import networkx
    #g = networkx.read_graphml(graphml_path)
    #if not isinstance(g, networkx.DiGraph) and not isinstance(g, networkx.MultiDiGraph):
    #    raise Exception(f"The provided graph does not get loaded as neither a DiGraph nor a MultiDiGraph instance by NetworkX. Its current class is {type(g)}.")
    class GraphContainer:
        def __init__(self, nodes : dict[str, int], edges : list[str]):
            self.nodes = nodes
            self.edges = edges
    g = GraphContainer(*manualGraphMLparser(graphml_path))
    
    spike_frequencies = defaultdict(float) # node -> spike frequency
    
    table = PrettyTable(["Layer Name", "Shape", "Neurons", "Avg. spike freq."])
    table.border = False
    table.preserve_internal_border = True
    nodenames_regex = re.compile(r'^(\d+(?:_[a-zA-Z][a-zA-Z0-9]+)+)')
    layernames_regex = re.compile(r'(\d+)([\w\d]+)_[\dx]*')
    input_frequencies = np.array(input['arr_0'], dtype = object)
    del input # save memory wherever possible
    spiketrains = np.array(data["spiketrains_n_b_l_t"], dtype = object)
    del data # save memory wherever possible
    print("Recognized layers in SNN logs:")
    prettyPrintIterable(["input"] + list(spiketrains[:, 1]))
    print("Recognized layers in SNN graph:")
    #layer_names = set((s := n.split('_'))[0] + '_' + s[1] for n in g.nodes)
    graph_layer_names = set(nodenames_regex.match(n).group(1) for n in g.nodes)
    prettyPrintIterable(graph_layer_names)
    print("Parsing network layers and computing spike frequencies:")
    # guess the input layer's name
    input_layer_name_candidates = ["input", "inputlayer"]
    input_layer_name = next((iname for iname in input_layer_name_candidates if f"0_{iname}_0_0_0" in g.nodes), None)
    if not input_layer_name:
        raise Exception(f"No valid input layer name found for the SNN, tried:", ", ".join(input_layer_name_candidates))
    print(f"\t-> working on layer: 0_{input_layer_name}")
    input_shape = input_frequencies.shape
    if len(input_shape) != 4:
        raise Exception(f"Input shape of 'arr_0' in file '{npz_input_path}' does not contain 4 items.")
    # iterate over an arbitrary number of dimensions
    for index in product(*(range(dim) for dim in input_shape)):
        nodename = f"0_{input_layer_name}_" + '_'.join(map(str, index[1:]))
        if nodename not in g.nodes:
            raise Exception(f"Could not find node {nodename} in the graph.")
        # directly use the input's value as mean spike frequency
        spike_frequencies[nodename] += input_frequencies[index]
    del input_frequencies
    # average over the batch size
    for k in spike_frequencies.keys():
        spike_frequencies[k] /= input_shape[0]
    table.add_row(["input", input_shape, len(spike_frequencies), sum(sf for sf in spike_frequencies.values())/len(spike_frequencies)])
    for layer_spiketrains, layer_name in spiketrains:
        neurons_count = 0
        tot_spike_freq = 0
        match = layernames_regex.match(layer_name)
        layer_idx, layer_type = int(match.group(1)) + 1, camel_to_snake(match.group(2))
        compress_timesteps = layer_name.split("_")[1].count("x") + 1 + 1 == len(layer_spiketrains.shape) - 1
        graph_layer_name = f"{layer_idx}_{layer_type}"
        if graph_layer_name not in graph_layer_names:
            # if the log's layer name is not in the graph's layer names, pick the one with the same ids
            new_graph_layer_name = next((gln for gln in graph_layer_names if str(layer_idx) == gln.split('_', 1)[0]), None)
            if not new_graph_layer_name:
                raise Exception(f"Could not find layer {graph_layer_name} in the graph's nodes list.")
            print(f"WARNING: layer {graph_layer_name} not found in graph, taking layer {new_graph_layer_name} by index.")
            graph_layer_name = new_graph_layer_name
        print(f"\t-> working on layer: {graph_layer_name}")
        # isolate the first dimension (batch size) an the last (time steps) as we do the mean over them
        if compress_timesteps:
            batches, shape, time_steps = layer_spiketrains.shape[0], layer_spiketrains.shape[1:-1], layer_spiketrains.shape[-1]
        else:
            batches, shape = layer_spiketrains.shape[0], layer_spiketrains.shape[1:]
        # iterate over an arbitrary number of dimensions
        for index in product(*(range(dim) for dim in shape)):
            nodename = f"{graph_layer_name}_" + '_'.join(map(str, index))
            if nodename not in g.nodes:
                raise Exception(f"Could not find node {nodename} in the graph.")
            # compute the mean number of spikes across inputs in the batch and time steps
            if compress_timesteps:
                sf = sum(sum(layer_spiketrains[b][index][t] for t in range(time_steps)) for b in range(batches)) / (batches * time_steps)
            else:
                sf = sum(layer_spiketrains[b][index] for b in range(batches)) / batches
            spike_frequencies[nodename] += sf
            tot_spike_freq += sf
            neurons_count += 1
        table.add_row([layer_name, layer_spiketrains.shape, neurons_count, tot_spike_freq / neurons_count])
    print(table)
    del spiketrains

    print("Building HyperGraph...")
    nodes = g.nodes
    hyperedges = defaultdict(list) # source node id -> list of destinations ids
    for src, dst in zip(g.edges[::2], g.edges[1::2]):
        hyperedges[src].append(dst)
    del g
    
    # return data, g # test with 'd, g = loadSNNcomposite("./snn_models/mnist_cnn_0.npz", "./snn_models/mnist_cnn_input.npz", "./snn_models/mnist_cnn.graphml")'
    return HyperGraph(len(nodes), [HyperEdge(nodes[k], tuple(map(lambda n : nodes[n], v)), spike_frequencies[k]) for k, v in hyperedges.items()], no_checks = True)


# HOW-TO IMPORT V1 FROM Allen Brain Map:
# - download the network's files from here https://portal.brain-map.org/explore/models/mv1-all-layers (or directly here https://www.dropbox.com/sh/w5u31m3hq6u2x5m/AACpYpeWnm6s_qJDpmgrYgP7a?dl=0)
#   - v1_nodes.h5
#   - v1_v1_edges.h5
#   - spikes.h5
# - I am too lazy to make a CLI interface for this, so just use it manually:
#   - python3.13 main.py -d -i
#   - from the interactive console call 'hg = loadSNNh5(...)' with the paths to the downloaded files
#   - finally use 'hg.save("allen_v1_processed")' to save the processed hypergraph for later
#   - now you can dispose of the files and get back you ~4GB of disk space, yay!

"""
Given paths relative to this script or absolute pointing to three '.h5' files, one containing
the graph's nodes, one its edges, and one the timestamped spikes, loads them and returns
the resulting SNN with spike frequencies as an hypergraph.
These files are typical of the 'Sonata' format.
Nodes must be index from 0 onward if 'allow_nodes_renaming' is False, otherwise they can
arbitrary ids, but parsing time grows exponentially due to remaning.
"""
@core
def loadSNNh5(nodes_path : str, edges_path : str, spikes_path : str, allow_nodes_renaming : bool = False) -> HyperGraph:
    import h5py

    nodes_path = os.path.abspath(nodes_path)
    if not os.path.exists(nodes_path):
        raise Exception(f"The provided path does not exist: {nodes_path}")
    elif not os.path.isfile(nodes_path):
        raise Exception(f"The provided path is not a file: {nodes_path}")
    elif nodes_path.split('.')[-1] != "h5":
        print(f"WARNING: the '{nodes_path}' file does not have the '.h5' extension. Are you sure it describes a graph?")
    nodes_file = h5py.File(nodes_path, "r")
    if not "nodes" in nodes_file:
        raise Exception(f"The '{nodes_path}' file does not contain the 'nodes' key. Existing keys: {nodes_file.keys()}")
    if len(nodes_file["nodes"]) < 1:
        raise Exception(f"The '{nodes_path}' file does not contain anything under the 'nodes' key.")
    nodes_key = list(nodes_file["nodes"].keys())[0]
    if len(nodes_file["nodes"]) > 1:
        print(f"WARNING: the '{nodes_path}' file does contains multiple keys anything under the 'nodes' key. Opening only the first one: '{nodes_key}'.")
    if not "node_id" in nodes_file["nodes"][nodes_key]:
        raise Exception(f"The '{nodes_path}' file does not contain the 'node_id' key under the 'nodes/{nodes_key}' keys. Existing keys: {nodes_file['nodes'][nodes_key].keys()}")
    if all(i == n for i, n in enumerate(nodes_file["nodes"][nodes_key]["node_id"])): # if nodes are numbered progressively from 0, use a range for a compact representation
        nodes = range(len(nodes_file["nodes"][nodes_key]["node_id"]))
    elif not allow_nodes_renaming:
        raise Exception("Nodes are not numbered progressively from 0, this would be too slow to parse...\nIf you still want to parse this, set 'allow_nodes_renaming' to True...")
    else:
        nodes = {n : i for i, n in enumerate(nodes_file["nodes"][nodes_key]["node_id"])} # old node id -> new node id
    print(f"Found {len(nodes)} nodes...")
    
    edges_path = os.path.abspath(edges_path)
    if not os.path.exists(edges_path):
        raise Exception(f"The provided path does not exist: {edges_path}")
    elif not os.path.isfile(edges_path):
        raise Exception(f"The provided path is not a file: {edges_path}")
    elif edges_path.split('.')[-1] != "h5":
        print(f"WARNING: the '{edges_path}' file does not have the '.h5' extension. Are you sure it describes a graph?")
    edges_file = h5py.File(edges_path, "r")
    if not "edges" in edges_file:
        raise Exception(f"The '{edges_path}' file does not contain the 'edges' key. Existing keys: {edges_file.keys()}")
    if len(edges_file["edges"]) < 1:
        raise Exception(f"The '{edges_path}' file does not contain anything under the 'edges' key.")
    edges_key = list(edges_file["edges"].keys())[0]
    if len(edges_file["edges"]) > 1:
        print(f"WARNING: the '{edges_path}' file does contains multiple keys anything under the 'edges' key. Opening only the first one: '{edges_key}'.")
    if not "source_node_id" in edges_file["edges"][edges_key] or not "target_node_id" in edges_file["edges"][edges_key]:
        raise Exception(f"The '{edges_path}' file does not contain the 'source_node_id' and 'target_node_id' keys under the 'edges/{edges_key}' keys. Existing keys: {edges_file['edges'][edges_key].keys()}")
    hedges = defaultdict(list)
    edges_count = len(edges_file["edges"][edges_key]["source_node_id"])
    last_print_time = time.monotonic()
    srcs, dsts = np.array(edges_file["edges"][edges_key]["source_node_id"], dtype = np.int32), np.array(edges_file["edges"][edges_key]["target_node_id"], dtype = np.int32)
    for i, (src, dst) in enumerate(zip(srcs, dsts)):
        #if src not in nodes or dst not in nodes: # too slow...
        #    raise Exception(f"Edge {src}->{dst} uses invalid node ids.")
        hedges[src].append(dst)
        if i % 1024 == 0:
            now = time.monotonic()
            if now - last_print_time >= Settings.PRINT_INTERVAL:
                print(f"Working on edge {i}/{edges_count}...")
                last_print_time = now
    print(f"Found {len(hedges)} hedges...")
    
    if type(nodes) != range and allow_nodes_renaming:
        print("Renaming nodes in hedges, this is gonna take a while...")
        new_hedges = {nodes[k] : [nodes[d] for d in v] for k, v in hedges.items()}
        hedges = new_hedges
    
    spikes_path = os.path.abspath(spikes_path)
    if not os.path.exists(spikes_path):
        raise Exception(f"The provided path does not exist: {spikes_path}")
    elif not os.path.isfile(spikes_path):
        raise Exception(f"The provided path is not a file: {spikes_path}")
    elif spikes_path.split('.')[-1] != "h5":
        print(f"WARNING: the '{spikes_path}' file does not have the '.h5' extension. Are you sure it describes a graph?")
    spikes_file = h5py.File(spikes_path, "r")
    if not "spikes" in spikes_file:
        raise Exception(f"The '{spikes_path}' file does not contain the 'spikes' key. Existing keys: {spikes_file.keys()}")
    if len(spikes_file["spikes"]) < 1:
        raise Exception(f"The '{spikes_path}' file does not contain anything under the 'spikes' key.")
    spikes_key = list(spikes_file["spikes"].keys())[0]
    if len(spikes_file["spikes"]) > 1:
        print(f"WARNING: the '{spikes_path}' file does contains multiple keys anything under the 'spikes' key. Opening only the first one: '{spikes_key}'.")
    if not "node_ids" in spikes_file["spikes"][spikes_key] or not "timestamps" in spikes_file["spikes"][spikes_key]:
        raise Exception(f"The '{spikes_path}' file does not contain the 'node_ids' and 'timestamps' keys under the 'spikes/{spikes_key}' keys. Existing keys: {spikes_file['spikes'][spikes_key].keys()}")
    timestamps = np.array(spikes_file["spikes"][spikes_key]["timestamps"])
    low, high = np.percentile(timestamps, [5, 95]) # drop values below the 5th percentile and above the 95th percentile
    start, end = math.inf, 0
    for t in timestamps:
        if t < start and t >= low:
            start = t
        if t > end and t <= high:
            end = t
    duration = end - start
    spike_counts = defaultdict(int)
    for n in spikes_file["spikes"][spikes_key]["node_ids"]:
        spike_counts[n] += 1
    print(f"Found spikes for {len(spike_counts)} nodes over a measured duration of {duration}s...")

    if type(nodes) != range and allow_nodes_renaming:
        print("Renaming spike counts, this is gonna take a while...")
        new_spike_counts = {nodes[k] : v for k, v in spike_counts.items()}
        spike_counts = new_spike_counts
    
    print("Building HyperGraph...")
    return HyperGraph(len(nodes), [HyperEdge(src, tuple(dsts), spike_counts[src]/duration) for src, dsts in hedges.items()], no_checks = True)

"""
Export the given hypergraph in NetworkX's format.
Spike frequencies will be annotated on each node under the label 'spike_frequency'.

The provided 'path' will be automatically concatenated with the '.gml.gz' extension.
"""
def exportToNetworkX(hg : HyperGraph, path : str, silent_overwrite : bool = False) -> None:
    path = os.path.abspath(path + ".gml.gz")
    if os.path.exists(path) and not silent_overwrite:
        overwrite = input(f"Request to save the hypergraph to an existing path '{path}'.\nOverwrite it [y/n]? ")
        while overwrite not in ['y', 'n']:
            overwrite = input(f"Invalid answer. Choose [y/n]? ")
        if overwrite == 'n':
            print("WARNING: the hypergraph was NOT saved due to an overwrite conflict.")
    
    graph = hg.toGraph(collapse_overlapping_edges = True)
    nxgraph = graph.toNxGraph()
    nx.write_gml(nxgraph, path)