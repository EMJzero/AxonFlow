import networkx as nx
import traceback
import argparse
import signal
import code
import time
import sys

in_interactive_mode = False

def signal_handler(signal: int, frame) -> None:
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
        #stop_threads()
        time.sleep(0.2)
        print('\nTERMINATION RECEIVED - SWITCHING TO INTERACTIVE MODE\n[type "exit()" or press "ctrl+c" again to terminate the program]\n')
        in_interactive_mode = True
        code.interact(local=globals())
        in_interactive_mode = False

def print_graph_statistics(graph):
    print('------ Graph Statistics ------')
    print(f"Directed: {graph.is_directed()}")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")

    degrees = dict(graph.degree())
    in_degrees = dict(graph.in_degree()) if graph.is_directed() else None
    out_degrees = dict(graph.out_degree()) if graph.is_directed() else None

    print(f"Average degree: {sum(degrees.values()) / len(degrees):.2f}")
    if in_degrees and out_degrees:
        print(f"Average in-degree: {sum(in_degrees.values()) / len(in_degrees):.2f}")
        print(f"Average out-degree: {sum(out_degrees.values()) / len(out_degrees):.2f}")

    if nx.is_directed_acyclic_graph(graph):
        print("Graph is a Directed Acyclic Graph (DAG).")
    else:
        print("Graph is not a DAG.")

    # Component info
    if graph.is_directed():
        components = list(nx.strongly_connected_components(graph))
        print(f"Number of strongly connected components: {len(components)}")
    else:
        components = list(nx.connected_components(graph))
        print(f"Number of connected components: {len(components)}")
    
    # Check for isolated nodes
    isolated = list(nx.isolates(graph))
    print(f"\nNumber of isolated nodes: {len(isolated)}")
    print("\nFirst 5 isolated nodes:")
    for node in isolated[:5]:
        print(node)
    print()

    # Show top 5 nodes by degree
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 nodes by degree:")
    for node, deg in top_nodes:
        print(f"{node}: {deg}")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Analyze a GraphML file.")
    parser.add_argument("file", help="Path to the GraphML file")
    args = parser.parse_args()

    print(f"Loading graph from '{args.file}' ...")
    G = nx.read_graphml(args.file)
    print_graph_statistics(G)

    print("Interactive, the graph is in the variable 'G'.")
    in_interactive_mode = True
    code.interact(local=globals())
    in_interactive_mode = False
