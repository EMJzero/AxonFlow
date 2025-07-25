from typing import Optional

from itertools import islice, combinations
from collections import defaultdict
from scipy.spatial import KDTree
from functools import reduce
import networkx as nx
import numpy as np
import heapq
import math

from prints import *
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
@core
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

NOTE: requires nodes to be topologically ordered!
"""
@core
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
If 'fixes' is True, all forces are updated after every batch, preventing suboptimal moves (slow).
"""
@core
def forceDirectedRefinement(hg : HyperGraph, placement : list[Coord2D], model : HardwareModel, batch : int = 16, fixes : bool = True) -> list[Coord2D]:
    if hg.nodes != len(placement):
        raise Exception("The provided placement does not have an entry for each HyperGraph node.")
    min_x, max_x, min_y, max_y = reduce(lambda m, c : (c.x if c.x < m[0] else m[0], c.x if c.x > m[1] else m[1], c.y if c.y < m[2] else m[2], c.y if c.y > m[3] else m[3]), placement, (placement[0].x, placement[0].x, placement[0].y, placement[0].y))
    directions = (Coord2D(1, 0), Coord2D(0, 1), Coord2D(-1, 0), Coord2D(0, -1))
    forces : dict[Coord2D, dict[Coord2D, float]] = defaultdict(lambda : {d : 0.0 for d in directions}, {coords : model.getForces(hg, placement, node, directions) for node, coords in enumerate(placement)})
    new_placement = BiMap({coords : node for node, coords in enumerate(placement)}, default_factory = lambda : -1)

    timerPrint = getTimerPrinter(Settings.PRINT_INTERVAL)

    candidates = []
    for coords in iter_major_diagonals(min_x, max_x, min_y, max_y, end_included = True):
        for d_pos, d_neg in zip(directions[:2], directions[2:]):
            other_coords = coords + d_pos
            if coords in forces or other_coords in forces:
                tension = forces[coords][d_pos] + forces[other_coords][d_neg]
                if tension > 0:
                    heapq.heappush(candidates, (-tension, coords, other_coords)) # max-heap
    
    prev_moves_counts = [0, 0]
    while len(candidates) > 0:
        timerPrint(f"FD remaining candidates {len(candidates)}")
        moves = 0
        one_candidate = len(candidates) == 1
        affected : set[Coord2D] = set()
        while moves < batch and len(candidates) > 0:
            _, coords, other_coords = heapq.heappop(candidates)
            tension = forces[coords][other_coords - coords] + forces[other_coords][coords - other_coords]
            if tension > 0:
                moves += 1
                #print("FD moving:", new_placement[other_coords], "<->", new_placement[coords])
                new_placement[coords], new_placement[other_coords] = new_placement[other_coords], new_placement[coords]
                forces[coords] = model.getForces(hg, new_placement.inv, new_placement[coords], directions)
                forces[other_coords] = model.getForces(hg, new_placement.inv, new_placement[other_coords], directions)
                for node in [new_placement[coords], new_placement[other_coords]]:
                    if node >= 0:
                        for he in hg.getTouchingHyperedges(node):
                            for other_node in he.nodes:
                                if other_node != node:
                                    affected.add(new_placement.inv[other_node])
        # ISSUES: in the original version forces are never rebuilt for all nodes, only swapped ones (the following 'for' was missing)
        if fixes:
            for coords in affected:
                forces[coords] = model.getForces(hg, new_placement.inv, new_placement[coords], directions)

        deduplicate = set()
        for i in range(0, len(candidates), -1):
            _, coords, other_coords = candidates[i]
            tension = forces[coords][other_coords - coords] + forces[other_coords][coords - other_coords]
            if tension <= 0:
                candidates.pop(i)
            else:
                candidates[i] = (-tension, coords, other_coords)
            deduplicate.add((coords.x, coords.y, other_coords.x, other_coords.y))
            deduplicate.add((other_coords.x, other_coords.y, coords.x, coords.y))
        for coords in affected:
            for d_pos in directions:
                d_neg = - d_pos
                other_coords = coords + d_pos
                if (coords.x, coords.y, other_coords.x, other_coords.y) not in deduplicate and (other_coords.x, other_coords.y, coords.x, coords.y) not in deduplicate:
                    tension = forces[coords][d_pos] + forces[other_coords][d_neg]
                    if tension > 0:
                        candidates.append((-tension, coords, other_coords))
                    deduplicate.add((coords.x, coords.y, other_coords.x, other_coords.y))
                    deduplicate.add((other_coords.x, other_coords.y, coords.x, coords.y))
        heapq.heapify(candidates)
        # ISSUE: unless we stop using batches when candidates are few, we might have endless loops due to lazy updates
        # ALTERNATIVE FIX: do NOT rebuild forces for all nodes (see above "ISSUE")
        if fixes and all(m == moves for m in prev_moves_counts) and not one_candidate:
            batch = max(min(moves - 1, batch), 1)
        prev_moves_counts.pop(0)
        prev_moves_counts.append(moves)
    
    return [new_placement.inv[node] for node in range(hg.nodes)]

"""
Compute a layout of a graph of 'nodes' nodes onto a 2D integer lattice via particle swarm optimization.

Arguments:
- hg: hypergraph to layout.
- model: source of lattice constraints and layout cost estimator.
- num_particles: number of particles (candidate solutions) in the PSO swarm.
- num_iterations: number of optimization iterations to run.
- w: inertia weight, balances exploration vs exploitation.
- c1: cognitive coefficient, how much particles are influenced by their own best position.
- c2: social coefficient, how much particles are influenced by the global best.
- initial_layout): optional seed layout used to initialize the first particle.
"""
@core
def particleSwarmPlacement(hg : HyperGraph, model : HardwareModel, num_particles : int = 30, num_iterations : int = 200, w : float = 0.72, c1 : float = 1.49, c2 : float = 1.49, initial_layout : Optional[list[Coord2D]] = None) -> list[Coord2D]:
    n_nodes = hg.nodes
    # Initialize particle positions and velocities
    particles_pos = []  # List of numpy arrays shape (n_nodes,2)
    particles_vel = []  # Same shape
    pbest_pos = []
    pbest_cost = []

    lattice_width = model.coresAlongX()
    lattice_height = model.coresAlongY()

    gbest_pos = None
    gbest_cost = float('inf')

    for i in range(num_particles):
        if initial_layout is not None and i == 0:
            # Use provided layout for the first particle
            pos = np.array([(pt.x, pt.y) for pt in initial_layout], dtype=float)
        else:
            # Random integer positions in lattice
            xs = np.random.randint(0, lattice_width, size=n_nodes)
            ys = np.random.randint(0, lattice_height, size=n_nodes)
            pos = np.column_stack((xs, ys)).astype(float)
        vel = np.zeros((n_nodes, 2), dtype=float)

        particles_pos.append(pos)
        particles_vel.append(vel)

        # Evaluate initial cost
        rounded = [Coord2D(int(round(x)), int(round(y))) for x, y in pos]
        c = model.getCompoundMetric(hg, rounded)
        pbest_pos.append(pos.copy())
        pbest_cost.append(c)

        if c < gbest_cost:
            gbest_cost = c
            gbest_pos = pos.copy()

    timerPrint = getTimerPrinter(Settings.PRINT_INTERVAL)

    # Main PSO loop
    for it in range(num_iterations):
        timerPrint(f"PSO iteration: {it}, best cost: {gbest_cost}")
        for i in range(num_particles):
            # Random coefficients per node and dimension
            r1 = np.random.rand(n_nodes, 2)
            r2 = np.random.rand(n_nodes, 2)

            # Velocity update
            particles_vel[i] = (
                w * particles_vel[i]
                + c1 * r1 * (pbest_pos[i] - particles_pos[i])
                + c2 * r2 * (gbest_pos - particles_pos[i])
            )

            # Position update
            particles_pos[i] += particles_vel[i]
            # Enforce lattice bounds
            particles_pos[i][:, 0] = np.clip(particles_pos[i][:, 0], 0, lattice_width - 1)
            particles_pos[i][:, 1] = np.clip(particles_pos[i][:, 1], 0, lattice_height - 1)

            # Evaluate
            rounded = [Coord2D(int(round(x)), int(round(y))) for x, y in particles_pos[i]]
            c = model.getCompoundMetric(hg, rounded)

            # Update personal best
            if c < pbest_cost[i]:
                pbest_cost[i] = c
                pbest_pos[i] = particles_pos[i].copy()

                # Update global best
                if c < gbest_cost:
                    gbest_cost = c
                    gbest_pos = particles_pos[i].copy()

        # Optional: progress log
        # print(f"Iter {it+1}/{num_iterations} best cost={gbest_cost}")

    # Convert continuous global best to integer lattice coordinates
    best_placement = [Coord2D(int(round(x)), int(round(y))) for x, y in gbest_pos]
    return best_placement

"""
Compute a layout of a graph of 'nodes' nodes onto a 2D integer lattice via the algorithm proposed
in the "TrueNorth Ecosystem" paper.

NOTE: requires nodes to be topologically ordered!
Arguments:
- hg: hypergraph to layout, nodes must be in topological order.
- masked_edges: src->dst connections that were ignored to build the topological order.
- model: neuromorphic hardware model.
"""
@core
def trueNorthPlacement(hg : HyperGraph, masked_edges : list[tuple[int, int]], model : HardwareModel) -> list[Coord2D]:
    nodes_layers = [set()]
    for node in range(hg.nodes):
        for he in hg.getInboundHyperedges(node):
            if he.source() in nodes_layers[-1]:
                if (he.source(), node) in masked_edges:
                    continue
                nodes_layers.append(set())
                break
        nodes_layers[-1].add(node)
    
    inbound_sources = [{he.source() for he in hg.getInboundHyperedges(n)} for n in range(hg.nodes)]
    
    chips = defaultdict(set) # chip(x, y) -> set of nodes
    placement = BiMap() # node idx -> placement
    
    """
    Given a set of nodes, returns subset of N nodes with the highest
    number of shared sources of inbound hyperedges among them.
    """
    def closest_N_nodes(nodes : Iterable[int], N : int) -> Iterable[int]:
        best, best_iou = None, -1
        for subset in combinations(nodes, N):
            union = set()
            for s in subset:
                union.update(inbound_sources[s])
            union = len(union)
            intersection = sum(1 for n in inbound_sources[subset[0]] if all(n in inbound_sources[s] for s in subset[1:]))
            iou = intersection / union if union != 0 else 0
            if iou > best_iou:
                best, best_iou = subset, iou
        return best
    
    input = nodes_layers.pop(0)
    input_per_chip = len(input)//model.chipsCount()
    chips_with_extra_input = len(input) % model.chipsCount()
    for c_x in range(model.chips_per_system_x):
        for c_y in range(model.chips_per_system_y):
            chip_idx = c_x * model.chips_per_system_y + c_y
            input_count = input_per_chip + (1 if chip_idx < chips_with_extra_input else 0)
            coords = get_equispaced_lattice_points(model.cores_per_chip_x, model.cores_per_chip_y, input_count)
            #closest = [input.pop(0) for _ in input_count]
            closest = closest_N_nodes(input, input_count)
            for coord, n in zip(coords, closest):
                chip_coords = Coord2D(c_x * model.cores_per_chip_x, c_y * model.cores_per_chip_y)
                placement[n] = chip_coords + coord
                chips[chip_coords]
                input.remove(n)

    for layer in nodes_layers:
        for node in layer:
            inbound = inbound_sources[n]
            best_chip_coords, best_intersection = None, -1
            # select best chip based on intersection of stored nodes sets
            for chip_coords, nodes in chips.items():
                intersection = sum(1 for n in inbound if n in nodes)
                if intersection > best_intersection and len(nodes) < model.coresPerChipCount():
                    best_chip_coords, best_intersection = chip_coords, intersection
            # select best core in chip based on minimum total manhattan distance
            best_placement, best_distance = None, math.inf
            for x in range(model.cores_per_chip_x):
                for y in range(model.cores_per_chip_y):
                    coord = best_chip_coords + Coord2D(x, y)
                    if coord in placement.inv:
                        continue
                    distance = 0
                    for connected_node in inbound_sources[node]:
                        if connected_node in placement:
                            distance += manhattan(coord, placement[connected_node])
                    if distance < best_distance:
                        best_placement, best_distance = coord, distance
            placement[node] = best_placement
    
    return [placement[n] for n in range(hg.nodes)]

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