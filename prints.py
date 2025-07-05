from typing import Iterable, TypeVar, Generator, Any, Callable

from contextlib import contextmanager
from prettytable import PrettyTable
from termcolor import colored
from itertools import islice
import functools
import builtins
import time

from settings import *

T = TypeVar('T')

"""
Custom print to be used inside 'core' functions.
"""
@contextmanager
def hijack_print(prefix):
    original_print = builtins.print
    def custom_print(*args, **kwargs):
        if Settings.VERBOSE:
            sep = kwargs.get('sep', ' ')
            end = kwargs.get('end', '\n')
            lines = sep.join(str(arg) for arg in args).split('\n')
            first_prefix = f"[{prefix}]"
            other_prefix = ' '*len(prefix) + '└▶' #f"[{'-' * len(prefix)}]"
            for i, line in enumerate(lines):
                original_print(colored(first_prefix if i == 0 else other_prefix, Settings.VERBOSE_COLOR), line, end = end if i == len(lines) - 1 else '\n')
    builtins.print = custom_print
    try:
        yield
    finally:
        builtins.print = original_print

"""
Custom tag to be used like:
```
@core
def function():
    print("test")
```
This will result in the function's name being logged upon call and in every
print inside the function being preceded by the function's name.
"""
def core(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if Settings.VERBOSE:
            #print(f"Calling function: {func.__name__}")
            print(colored(f"[function call: {func.__name__}]", Settings.VERBOSE_COLOR))
            if Settings.TIMING:
                start = time.perf_counter()
        with hijack_print(func.__name__):
            result = func(*args, **kwargs)
        if Settings.VERBOSE and Settings.TIMING:
            end = time.perf_counter()
            elapsed = end - start
            print(colored(f"[{func.__name__}]", Settings.VERBOSE_COLOR), f"finished in {elapsed:.6f} seconds")
        return result
    return wrapper

"""
Returns a string with a pretty textual representation of the provided dictionary.
"""
def prettyFormatDict(dictionary : dict, indent_level : int = 0) -> str:
    string = ""
    for key, value in (dictionary.items() if isinstance(dictionary, dict) else zip(["" for i in dictionary], dictionary)):
        string += '    '*indent_level + (f"{key}: " if key != "" else "- ")
        if isinstance(value, dict):
            string += "\n" + prettyFormatDict(value, indent_level + 1)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            string += "\n" + prettyFormatDict(value, indent_level + 1)
        else:
            string += str(value)
        string += "\n"
    return string.rstrip()

"""
Prints a nicely formatted textual representation of the provided dictionary.
"""
def prettyPrintDict(dictionary : dict, indent_level : int = 0) -> None:
    print(prettyFormatDict(dictionary, indent_level))

"""
Given an iterable, prints its elements in an equi-spaced grid with each row having
'elem_per_row' (default is 4) elements.
"""
def prettyPrintIterable(iterable : Iterable, elem_per_row : int = 4) -> None:
    def iter_in_chunks(iterable : Iterable[T], chunk_size : int) -> Generator[None, list[T], None]:
        it = iter(iterable)
        while True:
            chunk = list(islice(it, chunk_size))
            if not chunk:
                break
            yield chunk

    table = PrettyTable([i for i in range(elem_per_row)])
    table.header = False
    table.border = False
    for chunk in iter_in_chunks(iterable, elem_per_row):
        table.add_row(chunk + ['' for _ in range(elem_per_row - len(chunk))])
    print(table)

"""
Given a file size in bytes, returns a string representing it in the closest unit
of measure between B, KB, MB, GB, and TB.
"""
def fileSizeString(filesize : int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(filesize)
    for unit in units:
        if size < 1024 or unit == 'TB':
            return f"{size:.1f} {unit}"
        size /= 1024

"""
Failure in drawing graphs with braill on the CLI.

import random
import shutil
import math
import drawille
import networkx as nx
from collections import defaultdict
# ANSI color codes
COLORS = [
    "\033[31m", "\033[32m", "\033[33m", "\033[34m",
    "\033[35m", "\033[36m", "\033[91m", "\033[92m", "\033[94m"
]
RESET = "\033[0m"
WHITE = "\033[97m"

def draw_graph_with_labels(num_nodes: int, edges: list[tuple[int, int]]) -> None:
    # Terminal width
    term_width = shutil.get_terminal_size().columns
    width_chars = max(40, min(150, term_width))
    width_px = width_chars * 2
    base_edge_length = width_px // 16
    height_px = width_px  # Use square canvas

    # Create graph
    G = nx.MultiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    # Compute layout
    pos = nx.spring_layout(G, seed=42, k=base_edge_length / 100.0)
    xs, ys = zip(*pos.values())
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def norm(val, vmin, vmax):
        return (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5

    node_positions = {}
    for node, (x, y) in pos.items():
        px = int(norm(x, min_x, max_x) * (width_px - 10)) + 5
        py = int(norm(y, min_y, max_y) * (height_px - 10)) + 5
        node_positions[node] = (px, py)

    canvas = drawille.Canvas()
    edge_colors = {}
    label_cells = set()

    # Handle multiple edges
    edge_groups = defaultdict(list)
    for u, v in G.edges():
        edge_groups[tuple(sorted((u, v)))].append((u, v))

    def draw_line_with_color(canvas, x1, y1, x2, y2, color, color_map):
        steps = max(abs(x2 - x1), abs(y2 - y1))
        for i in range(steps + 1):
            x = int(x1 + (x2 - x1) * i / steps)
            y = int(y1 + (y2 - y1) * i / steps)
            canvas.set(x, y)
            cell = (x // 2, y // 4)
            if cell not in color_map:
                color_map[cell] = color

    # Draw edges
    spacing = 4
    color_index = 0
    for (u, v), instances in edge_groups.items():
        x1, y1 = node_positions[u]
        x2, y2 = node_positions[v]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            continue

        ox, oy = -dy / length, dx / length
        for i in range(len(instances)):
            offset = (i - (len(instances) - 1) / 2) * spacing
            sx = int(x1 + ox * offset)
            sy = int(y1 + oy * offset)
            ex = int(x2 + ox * offset)
            ey = int(y2 + oy * offset)
            color = COLORS[color_index % len(COLORS)]
            draw_line_with_color(canvas, sx, sy, ex, ey, color, edge_colors)
            color_index += 1

    # Draw nodes
    node_labels = {}
    for node, (px, py) in node_positions.items():
        label = str(node)
        cx, cy = px // 2, py // 4
        for i, ch in enumerate(label):
            node_labels[(cx + i, cy)] = f"{WHITE}{ch}{RESET}"
            label_cells.add((cx + i, cy))

    # Render
    lines = canvas.frame().splitlines()
    for y, line in enumerate(lines):
        out = []
        for x, ch in enumerate(line):
            cell = (x, y)
            if cell in node_labels:
                out.append(node_labels[cell])
            elif cell in edge_colors and cell not in label_cells:
                out.append(f"{edge_colors[cell]}{ch}{RESET}")
            else:
                out.append(ch)
        print("".join(out))


# Test case
draw_graph_with_labels(15, [
    (0, 1), (0, 2), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 3), (7, 8), (8, 9), (10, 11), (12, 13), (13, 14)
])

if False and __name__ == "__main__":
    draw_graph_with_labels(15, [
        (0, 1), (0, 2), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 3), (7, 8), (8, 9), (10, 11), (12, 13), (13, 14),
        # node 10 is isolated
    ])

if False and __name__ == "__main__":
    num_nodes = 8
    edges = [
        (0, 1), (0, 1), (0, 1),  # multiple edges
        (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 6), (6, 3),
        # node 7 is isolated
    ]
    draw_graph_with_labels(num_nodes, edges)

if False and __name__ == "__main__":
    num_nodes = 12
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 0), (0, 6), (6, 7), (7, 8), (8, 9),
        (9, 10), (10, 11), (11, 6), (3, 9)
    ]
    draw_graph_with_labels(num_nodes, edges)

if False and __name__ == "__main__":
    num_nodes = 24
    edges = []
    for _ in range(32):
        edges.append(tuple(random.sample(range(24), 2)))
    draw_graph_with_labels(num_nodes, edges)
"""