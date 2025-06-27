from collections import defaultdict
from prettytable import PrettyTable
from itertools import product
import numpy as np
import networkx
import re
import os

from prints import *
from utils import *
from snn import *

"""
Given a path relative to this script or absolute pointing to a GraphML graph,
loads it and returns it as an hypergraph.
The graph must have the attribute 'spike_frequency' on either nodes or edges.
"""
@core
def loadSNNGraphML(path : str) -> HyperGraph:
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
    for array_name in data.files:
        array = data[array_name]
        table.add_row([array_name, array.shape, array.dtype, os.path.basename(npz_log_path)])
    for array_name in input.files:
        array = input[array_name]
        table.add_row([array_name, array.shape, array.dtype, os.path.basename(npz_input_path)])
    print(table)
    
    if "spiketrains_n_b_l_t" not in data.files:
        raise Exception(f"The '{npz_log_path}' file does not contain an 'spiketrains_n_b_l_t' array.")

    if "arr_0" not in input.files:
        raise Exception(f"The '{npz_input_path}' file does not contain an 'arr_0' array.")

    graphml_path = os.path.abspath(graphml_path)
    if not os.path.exists(graphml_path):
        raise Exception(f"The provided path does not exist: {graphml_path}")
    elif not os.path.isfile(graphml_path):
        raise Exception(f"The provided path is not a file: {graphml_path}")
    elif graphml_path.split('.')[-1] != "graphml":
        print("WARNING: the provided file does not have the '.graphml' extension. Are you sure it stores a graph?")
    
    print("Loading SNN graph from:", graphml_path)
    g = networkx.read_graphml(graphml_path)
    if not isinstance(g, networkx.DiGraph) and not isinstance(g, networkx.MultiDiGraph):
        raise Exception(f"The provided graph does not get loaded as neither a DiGraph nor a MultiDiGraph instance by NetworkX. Its current class is {type(g)}.")
    
    spike_frequencies = defaultdict(lambda : 0) # node -> spike frequency
    
    table = PrettyTable(["Layer Name", "Shape", "Neurons", "Avg. spike freq."])
    table.border = False
    table.preserve_internal_border = True
    input_frequencies : np.ndarray = input['arr_0']
    spiketrains : np.ndarray = data["spiketrains_n_b_l_t"]
    print("Recognized layers in SNN logs:")
    prettyPrintIterable(["input"] + list(spiketrains[:, 1]))
    print("Recognized layers in SNN graph:")
    #prettyPrintIterable(set((s := n.split('_'))[0] + '_' + s[1] for n in g.nodes))
    prettyPrintIterable(set(re.match(r'^(\d+(?:_[a-zA-Z][a-zA-Z0-9]+)+)', n).group(1) for n in g.nodes))
    print("Parsing network layers and computing spike frequencies:")
    print("\t-> working on layer: 0_input")
    input_shape = input_frequencies.shape
    if len(input_shape) != 4:
        raise Exception(f"Input shape of 'arr_0' in file '{npz_input_path}' does not contain 4 items.")
    # iterate over an arbitrary number of dimensions
    for index in product(*(range(dim) for dim in input_shape)):
        nodename = f"0_input_" + '_'.join(map(str, index[1:]))
        if nodename not in g.nodes:
            raise Exception(f"Could not find node {nodename} in the graph.")
        # directly use the input's value as mean spike frequency
        spike_frequencies[nodename] += input_frequencies[index]
    # average over the batch size
    for k in spike_frequencies.keys():
        spike_frequencies[k] /= input_shape[0]
    table.add_row(["input", input_shape, len(spike_frequencies), sum(sf for sf in spike_frequencies.values())/len(spike_frequencies)])
    for layer_spiketrains, layer_name in spiketrains:
        neurons_count = 0
        tot_spike_freq = 0
        match = re.match(r'(\d\d)([\w\d]+)_[\dx]*', layer_name)
        layer_idx, layer_type = int(match.group(1)) + 1, camel_to_snake(match.group(2))
        print(f"\t-> working on layer: {layer_idx}_{layer_type}")
        # isolate the first dimension (batch size) an the last (time steps) as we do the mean over them
        batches, shape, time_steps = layer_spiketrains.shape[0], layer_spiketrains.shape[1:-1], layer_spiketrains.shape[-1]
        # iterate over an arbitrary number of dimensions
        for index in product(*(range(dim) for dim in shape)):
            nodename = f"{layer_idx}_{layer_type}_" + '_'.join(map(str, index))
            if nodename not in g.nodes:
                raise Exception(f"Could not find node {nodename} in the graph.")
            # compute the mean number of spikes across inputs in the batch and time steps
            sf = sum(sum(layer_spiketrains[b][index][t] for t in range(time_steps)) for b in range(batches)) / (batches * time_steps)
            spike_frequencies[nodename] += sf
            tot_spike_freq += sf
            neurons_count += 1
        table.add_row([layer_name, layer_spiketrains.shape, neurons_count, tot_spike_freq / neurons_count])
    print(table)

    print("Building HyperGraph:")
    nodes = {n : i for i, n in enumerate(g.nodes)} # node id -> node index
    hyperedges = defaultdict(list) # source node id -> list of destinations ids
    for edge in g.edges:
        src, dst = edge
        hyperedges[src].append(dst)
    
    # return data, g # test with 'd, g = loadSNNcomposite("./snn_models/mnist_cnn_0.npz", "./snn_models/mnist_cnn_input.npz", "./snn_models/mnist_cnn.graphml")'
    return HyperGraph(len(nodes), [HyperEdge(nodes[k], tuple(map(lambda n : nodes[n], v)), spike_frequencies[k]) for k, v in hyperedges.items()])

"""
FAILURE

Given a path relative to this script or absolute pointing to a NumPy array file
that has been generated as a "spiketrains var_log" from "SNN toolbox", loads it
and returns it as an hypergraph.

NOTE: if multiple paths are provided, they must refer to multiple batches ran on
      the same network.

@core
def loadSNNnpz(paths : list[str]) -> HyperGraph:
    paths = [os.path.abspath(path) for path in paths]
    data = {} # path -> data
    
    for path in paths:
        if not os.path.exists(path):
            raise Exception(f"The provided path does not exist: {path}")
        elif not os.path.isfile(path):
            raise Exception(f"The provided path is not a file: {path}")
        elif path.split('.')[-1] != "npz":
            print(f"WARNING: the '{path}' file does not have the '.npz' extension. Are you sure it is a report from \"SNN toolbox\"?")
        
        print("Loading SNN log from:", path)
    
        data[path] = np.load(path, allow_pickle = True)

        table = PrettyTable(["Array Name", "Shape", "Data Type"])
        table.border = False
        table.preserve_internal_border = True
        print(f"Arrays in file '{os.path.basename(path)}':")
        for array_name in data[path].files:
            array = data[path][array_name]
            table.add_row([array_name, array.shape, array.dtype])
        print(table)

        if "spiketrains_n_b_l_t" not in data[path].files:
            raise Exception(f"The '{path}' file does not contain an 'spiketrains_n_b_l_t' array.")

    # get the network data from the first file, then ensure it matches with the others
    spiketrains : np.ndarray = data[paths[0]]["spiketrains_n_b_l_t"]
    print(f"Network information:\n\tlayers count: {spiketrains.shape[0]}\n\tbatch size: {spiketrains[0][0].shape[0]}\n\tbatches: {len(paths)}\n\ttime steps: {spiketrains[0][0].shape[-1]}")
    if not all(v["spiketrains_n_b_l_t"].shape == spiketrains.shape for k, v in data.items() if k != paths[0]) \
        or not all(all(v["spiketrains_n_b_l_t"][l][1] == spiketrains[l][1] for l in range(spiketrains.shape[0])) for k, v in data.items() if k != paths[0]) \
        or not all(all(v["spiketrains_n_b_l_t"][l][0].shape[1:] == spiketrains[l][0].shape[1:] for l in range(spiketrains.shape[0])) for k, v in data.items() if k != paths[0]):
            raise Exception("Missmatch between network shapes and/or layer names across the provided files.")

    print(f"INFO: for now, we ignore the input layer (and thus how spikes are supplied to input neurons), as that only occurs on one tick out of 50 or more.\nFor reference, the input shape is: {data[paths[0]]['input_image_b_l'].shape}")

    print("Parsing network layers:")
    table = PrettyTable(["Layer Name", "Shape", "Neurons"])
    table.border = False
    table.preserve_internal_border = True
    for layer_spiketrains, layer_name in spiketrains:
        table.add_row([layer_name, layer_spiketrains.shape, "..."])

    return data
    #return HyperGraph(len(nodes), [HyperEdge(nodes[k], tuple(map(lambda n : nodes[n], v)), nodes_spike_frequencies[k]) for k, v in hyperedges.items()])
"""