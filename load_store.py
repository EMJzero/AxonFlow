from collections import defaultdict
import networkx
import os

from prints import *
from snn import *

"""
Given a path relative to this script or absolute pointing to a GraphML graph,
loads it and returns it as an hypergraph.
The graph must have the attribute 'spike_frequency' on either nodes or edges.
"""
@core
def loadSNN(path : str) -> HyperGraph:
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