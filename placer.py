from collections import defaultdict
from functools import reduce
from scipy.spatial import KDTree
from itertools import islice
import networkx as nx
import numpy as np
import heapq
import math

from model import *
from utils import *

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
def spectralPlacement(graph: nx.Graph, width: int, height: int) -> list[Coord2D]:
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
    grid_points = [Coord2D(x, y) for x in range(offset_x, offset_x + box_w) for y in range(offset_y, offset_y + box_h)]
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
def hilbertPlacement(nodes : int, width : int, height : int) -> list[Coord2D]:
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
                yield Coord2D(x, y)
                x += dax
                y += day
            return
        # trivial column fill
        if w == 1:
            for _ in range(h):
                yield Coord2D(x, y)
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
Force-directed placement refinement method from Ouwen Jin's paper.
Using a potential function of the placement that considers a potential for each
connection among the partitioned SNN's nodes weighted by the distance between
the placements for the connected nodes. This is an heuristic to minimize such potential.
"""
def forceDirectedRefinement(hg : HyperGraph, placement : list[Coord2D], model : HardwareModel, batch : int = 16) -> list[Coord2D]:
    if hg.nodes != len(placement):
        raise Exception("The provided placement does not have an entry for each HyperGraph node.")
    min_x, max_x, min_y, max_y = reduce(lambda m, c : (c.x if c.x < m[0] else m[0], c.x if c.x > m[1] else m[1], c.y if c.y < m[2] else m[2], c.y if c.y > m[3] else m[3]), placement, (placement[0].x, placement[0].x, placement[0].y, placement[0].y))
    directions = (Coord2D(1, 0), Coord2D(0, 1), Coord2D(-1, 0), Coord2D(0, -1))
    forces = defaultdict(lambda : 0.0, {coords : model.getForces(hg, placement, node, directions) for node, coords in enumerate(placement)})
    #new_placement = defaultdict(lambda : -1, {coords : node for node, coords in enumerate(placement)})
    new_placement = BiMap({coords : node for node, coords in enumerate(placement)}, default_factory = lambda : -1)

    #candidates = [forces[coords][d_pos] + forces[coords + d_pos][d_neg] for coords in iter_major_diagonals(0, 0, width, height) for d_pos, d_neg in zip(directions[:2], directions[2:]) if coords in forces or coords + d_pos in forces]
    #heapq.heapify(candidates)
    #-
    candidates = []
    for coords in iter_major_diagonals(min_x, max_x, min_y, max_y):
        for d_pos, d_neg in zip(directions[:2], directions[2:]):
            if coords in forces or coords + d_pos in forces:
                other_coords = coords + d_pos
                tension = forces[coords][d_pos] + forces[other_coords][d_neg]
                if tension > 0:
                    heapq.heappush(candidates, (-tension, coords, other_coords)) # max-heap
    
    while len(candidates) > 0:
        moves = 0
        affected : set[Coord2D] = []
        while moves < batch and len(candidates) > 0:
            _, coords, other_coords = heapq.heappop(candidates)
            tension = forces[coords][other_coords - coords] + forces[other_coords][coords - other_coords]
            if tension > 0:
                moves += 1
                new_placement[coords], new_placement[other_coords] = new_placement[other_coords], new_placement[coords]
                forces[coords] = model.getForces(hg, new_placement, new_placement[coords], directions)
                forces[other_coords] = model.getForces(hg, new_placement, new_placement[other_coords], directions)
                for node in [new_placement[coords], new_placement[other_coords]]:
                    if node > 0:
                        for he in hg.getTouchingHyperedges(node):
                            for other_node in he.nodes:
                                if other_node != node:
                                    affected.add(new_placement.inv[other_node])
        # ISSUES: in the original version forces are never rebuilt for all nodes, only swapped ones (the following for was missing)
        for coords in affected:
            forces[coords] = model.getForces(hg, new_placement, new_placement[coords], directions)

        deduplicate = set()
        for i in range(0, len(candidates), -1):
            _, coords, other_coords = candidates[i]
            tension = forces[coords][other_coords - coords] + forces[other_coords][coords - other_coords]
            if tension <= 0:
                candidates.pop(i)
            else:
                candidates[i] = (-tension, coords, other_coords)
            deduplicate.add(coords + other_coords)
            deduplicate.add(other_coords + coords)
        for coords in affected:
            for d_pos in directions:
                d_neg = - d_pos
                other_coords = coords + d_pos
                if coords + other_coords not in deduplicate and other_coords + coords not in deduplicate:
                    tension = forces[coords][d_pos] + forces[other_coords][d_neg]
                    if tension > 0:
                        candidates.append((-tension, coords, other_coords))
                    deduplicate.add(coords + other_coords)
                    deduplicate.add(other_coords + coords)
        heapq.heapify(candidates)
    
    #final_placement = [None for _ in range(hg.nodes)]
    #for coords, node in new_placement.items():
    #    assert final_placement[node] == None, "Two placements produced for the same node, something broke."
    #    final_placement[node] = coords
    #return final_placement
    return [new_placement.inv[node] for node in range(hg.nodes)]

"""
Print method that shows a matrix of 0s and 1s, a digit per hardware core,
there the 1s mark active cores in the provided placement.
"""
def showActiveCores(layout : list[Coord2D], width : int, height : int) -> None:
    m = [[0 for _ in range(width)] for _ in range(height)]
    for x, y in layout:
        m[y][x] = 1
    for x in range(width):
        for y in range(height):
            print(m[y][x], end = '')
        print('')