import math
import numpy as np
import networkx as nx
from itertools import islice
from scipy.spatial import KDTree

# NOTE:
# This is a graph layout problem on a 2D lattice, aka a placement problem (for the VLSI guys).
# The goal is to minimize the total manhattan distance (and derived metrics, see "model.py")
# travelled by the graph's edges once their cores are placed on lattice points.
# Another way to put this, is that we want to maximize the locality of connections.
# 
# Techniques:
# - First get a topological ordering of the graph (making it acyclic if needed).
# - Hilbert Space Filling Curve as a starting point. Issue: works only on power-of-two lattice dimensions.
# - Spectal layout technique for a starting point.
# - Refine the placement with a Force-Directed algorithm (or Simulated Annealing, Particle Swarm, etc...).
# - Co-optimize with the partitining while refining the placement.

"""
Compute a compact, structure-aware layout of a graph onto a 2D integer lattice.
Embeds the nodes of an undirected, weighted graph into a fixed-size 2D lattice ('width' by 'height') such that:
- Nodes connected by high-weight edges are placed close together.
- The total (weighted) Manhattan distance of edges is minimized as a heuristic goal.
- Node placements are compact and avoid unnecessary dispersion across the grid.
- Each node is assigned a unique lattice coordinate (no collisions).

The algorithm uses a spectral embedding (via eigenvectors of the graph Laplacian) to project the graph structure
into 2D Euclidean space. The layout is then scaled to fit within a compact bounding box inside the given lattice,
and coordinates are discretized to the nearest available lattice points, resolving collisions.
"""
def spectralPlacement(graph: nx.Graph, width: int, height: int) -> list[tuple[int, int]]:
    nodes = graph.number_of_nodes()
    if width * height < nodes:
        raise Exception("Grid too small to hold all nodes.")

    # 1. Spectral layout using edge weights
    pos = nx.spectral_layout(graph, weight = 'spike_frequency', dim = 2)

    # 2. Layout coordinates -> [0, 1]^2 box
    coords = np.array([pos[i] for i in range(nodes)])
    coords -= coords.min(axis=0)
    coords /= coords.max(axis=0) + 1e-9
    # 3. Determine a tight box to pack nodes closely
    aspect_ratio = width / height
    box_w = min(width, math.ceil(math.sqrt(nodes * aspect_ratio)))
    box_h = min(height, math.ceil(nodes / box_w))
    # 4. Scale to the compact box
    coords[:, 0] *= box_w - 1
    coords[:, 1] *= box_h - 1
    # 5. Center the compact box in the full grid
    offset_x = (width - box_w) // 2
    offset_y = (height - box_h) // 2

    # 6. Create grid, KDTree, and resolve unique placement
    grid_points = [(x, y) for x in range(offset_x, offset_x + box_w) for y in range(offset_y, offset_y + box_h)]
    tree = KDTree(grid_points)
    used = set()
    embedding = [0 for _ in range(nodes)]
    for i, pt in enumerate(coords):
        _, idx = tree.query(pt + [offset_x, offset_y])
        while grid_points[idx] in used:
            grid_points.pop(idx)
            tree = KDTree(grid_points)
            _, idx = tree.query(pt + [offset_x, offset_y])
        embedding[i] = grid_points[idx]
        used.add(grid_points[idx])
    return embedding

"""
Compute a layout of a graph of 'nodes' nodes onto a 2D integer lattice via a 2D generalized Hilbert-like
space-filling curve that visits every point in a 'width' by 'height' rectangle exactly once.
It is highly recommended for 'width' and 'height' to be powers of two.
"""
def hilbertPlacement(nodes : int, width : int, height : int) -> list[tuple[int, int]]:
    #if width.bit_count() != 1 or height.bit_count() != 1:
    #    print("WARNING: building an HSC with 'width' and 'height' that are not powers of 2, results quality may vary.")
    if width % 2 != 0 or height % 2 != 0:
        print("WARNING: building an HSC with odd 'width' or 'height', results quality may vary.")

    def sgn(x):
        return (x > 0) - (x < 0)

    def generate(x, y, ax, ay, bx, by):
        # Credit: https://github.com/jakubcerveny/gilbert
        w = abs(ax + ay)
        h = abs(bx + by)
        dax, day = sgn(ax), sgn(ay)  # unit major direction
        dbx, dby = sgn(bx), sgn(by)  # unit orthogonal direction
        # trivial row fill
        if h == 1:
            for _ in range(w):
                yield (x, y)
                x += dax
                y += day
            return
        # trivial column fill
        if w == 1:
            for _ in range(h):
                yield (x, y)
                x += dbx
                y += dby
            return

        ax2, ay2 = ax // 2, ay // 2
        bx2, by2 = bx // 2, by // 2
        w2 = abs(ax2 + ay2)
        h2 = abs(bx2 + by2)
        if 2 * w > 3 * h:
            if w2 % 2 and w > 2:
                ax2 += dax
                ay2 += day
            # long case: split in two horizontal parts
            yield from generate(x, y, ax2, ay2, bx, by)
            yield from generate(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
        else:
            if h2 % 2 and h > 2:
                bx2 += dbx
                by2 += dby
            # standard case: three-way split
            yield from generate(x, y, bx2, by2, ax2, ay2)
            yield from generate(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
            yield from generate(
                x + (ax - dax) + (bx2 - dbx),
                y + (ay - day) + (by2 - dby),
                -bx2, -by2,
                -(ax - ax2), -(ay - ay2)
            )

    # reduce width and height to the smallest even values to fit "nodes"
    candidate_width = width - (1 if width % 2 != 0 else 2)
    candidate_height = height - (1 if height % 2 != 0 else 2)
    while candidate_width * candidate_height >= nodes:
        width = candidate_width
        height = candidate_height
        if width > height:
            candidate_width = width - 2
        else:
            candidate_height = height - 2
    # attempt that included odd numbers -> failed miserably
    #while (width - 1) * (height - 1) >= nodes:
    #    width, height = width - 1, height - 1

    if width >= height:
        return list(islice(generate(0, 0, width, 0, 0, height), nodes))
    else:
        return list(islice(generate(0, 0, 0, height, width, 0), nodes))


"""
Print method that shows a matrix of 0s and 1s, a digit per hardware core,
there the 1s mark active cores in the provided placement.
"""
def showActiveCores(layout, widht, height):
    m = [[0 for _ in range(widht)] for _ in range(height)]
    for x, y in layout:
        m[y][x] = 1
    for x in range(widht):
        for y in range(height):
            print(m[y][x], end = '')
        print('')