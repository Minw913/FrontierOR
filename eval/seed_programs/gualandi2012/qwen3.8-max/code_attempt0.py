import json
import argparse
import time
import heapq
import random
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Graph Coloring Optimization")
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to JSONL log file for intermediate solutions')
    return parser.parse_args()


def load_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['graph']


def build_adjacency(num_vertices, edges):
    adj = [set() for _ in range(num_vertices)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def dsatur_coloring(num_vertices, adj):
    """
    DSATUR (Degree of Saturation) greedy coloring algorithm.
    Selects the uncolored vertex with the highest saturation degree,
    breaking ties by the number of uncolored neighbors (then degree).
    """
    colors = [-1] * num_vertices
    saturation = [set() for _ in range(num_vertices)]
    degree = [len(adj[v]) for v in range(num_vertices)]

    # Max-heap: (-saturation, -degree, vertex)
    heap = [(0, -degree[v], v) for v in range(num_vertices)]
    heapq.heapify(heap)

    max_color = -1
    colored = [False] * num_vertices
    colored_count = 0

    while colored_count < num_vertices:
        best_v = -1
        while heap:
            neg_sat, neg_deg, v = heapq.heappop(heap)
            if not colored[v]:
                if -neg_sat == len(saturation[v]):
                    best_v = v
                    break
                else:
                    heapq.heappush(heap, (-len(saturation[v]), -degree[v], v))

        if best_v == -1:
            for v in range(num_vertices):
                if not colored[v]:
                    best_v = v
                    break
            if best_v == -1:
                break

        v = best_v
        used_colors = set()
        for u in adj[v]:
            if colored[u]:
                used_colors.add(colors[u])

        c = 0
        while c in used_colors:
            c += 1

        colors[v] = c
        colored[v] = True
        max_color = max(max_color, c)
        colored_count += 1

        for u in adj[v]:
            if not colored[u]:
                saturation[u].add(c)
                heapq.heappush(heap, (-len(saturation[u]), -degree[u], u))

    return colors, max_color + 1


def try_eliminate_color(num_vertices, adj, colors, target_color, rng):
    """
    Try to eliminate a specific color by recoloring all vertices that have it.
    Uses most-constrained-first strategy with random tie-breaking.
    Returns new coloring if successful, None otherwise.
    """
    target_vertices = [v for v in range(num_vertices) if colors[v] == target_color]
    if not target_vertices:
        return colors[:]

    available_colors = sorted(set(colors) - {target_color})
    if not available_colors:
        return None

    new_colors = colors[:]
    remaining = set(target_vertices)

    while remaining:
        best_v = None
        best_options = None
        best_count = float('inf')

        for v in remaining:
            neighbor_colors = set()
            for u in adj[v]:
                neighbor_colors.add(new_colors[u])

            options = [c for c in available_colors if c not in neighbor_colors]
            if len(options) < best_count:
                best_count = len(options)
                best_v = v
                best_options = options

        if best_count == 0:
            return None

        if len(best_options) > 1:
            new_colors[best_v] = rng.choice(best_options)
        else:
            new_colors[best_v] = best_options[0]

        remaining.remove(best_v)

    return new_colors


def local_search_reduce(num_vertices, adj, colors, time_limit, start_time, rng, time_fraction=0.5):
    """
    Iteratively try to reduce the number of colors by eliminating color classes.
    """
    best_colors = colors[:]

    while True:
        if time.time() - start_time > time_limit * time_fraction:
            break

        color_classes = defaultdict(list)
        for v in range(num_vertices):
            color_classes[best_colors[v]].append(v)

        if len(color_classes) <= 1:
            break

        sorted_colors = sorted(color_classes.keys(), key=lambda c: len(color_classes[c]))

        eliminated = False
        for target in sorted_colors:
            if time.time() - start_time > time_limit * time_fraction:
                break

            for attempt in range(25):
                result = try_eliminate_color(num_vertices, adj, best_colors, target, rng)
                if result is not None:
                    best_colors = result
                    eliminated = True
                    break

            if eliminated:
                break

        if not eliminated:
            break

    num_colors = len(set(best_colors))
    return best_colors, num_colors


def greedy_clique_lower_bound(num_vertices, adj):
    """Compute a lower bound on chromatic number via greedy clique finding."""
    vertices = sorted(range(num_vertices), key=lambda v: len(adj[v]), reverse=True)
    clique = []
    for v in vertices:
        if all(u in adj[v] for u in clique):
            clique.append(v)
    return len(clique)


def solve_with_gurobi(num_vertices, edges, adj, colors_hint, k, time_remaining):
    """
    Solve graph coloring using Gurobi ILP.
    k is the upper bound on the number of colors (from heuristic).
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, None

    if time_remaining <= 0.5:
        return None, None

    model = gp.Model("graph_coloring")
    model.setParam('OutputFlag', 0)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(time_remaining, 0.1))

    # Decision variables
    x = {}
    for v in range(num_vertices):
        for c in range(k):
            x[v, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")

    y = {}
    for c in range(k):
        y[c] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}")

    model.update()

    # Objective: minimize number of colors used
    model.setObjective(gp.quicksum(y[c] for c in range(k)), GRB.MINIMIZE)

    # Each vertex gets exactly one color
    for v in range(num_vertices):
        model.addConstr(gp.quicksum(x[v, c] for c in range(k)) == 1)

    # Adjacent vertices must have different colors
    for u, v in edges:
        for c in range(k):
            model.addConstr(x[u, c] + x[v, c] <= 1)

    # Link x and y
    for v in range(num_vertices):
        for c in range(k):
            model.addConstr(x[v, c] <= y[c])

    # Symmetry breaking: colors used in order
    for c in range(k - 1):
        model.addConstr(y[c] >= y[c + 1])

    # Lower bound from greedy clique
    clique_lb = greedy_clique_lower_bound(num_vertices, adj)
    if clique_lb > 1:
        model.addConstr(gp.quicksum(y[c] for c in range(k)) >= clique_lb)

    # Warm start with heuristic solution
    if colors_hint is not None:
        for v in range(num_vertices):
            for c in range(k):
                x[v, c].Start = 1.0 if colors_hint[v] == c else 0.0
        used_colors = set(colors_hint)
        for c in range(k):
            y[c].Start = 1.0 if c in used_colors else 0.0

    model.optimize()

    if model.SolCount > 0:
        result_colors = [0] * num_vertices
        for v in range(num_vertices):
            for c in range(k):
                if x[v, c].X > 0.5:
                    result_colors[v] = c
                    break

        result_num_colors = sum(1 for c in range(k) if y[c].X > 0.5)
        return result_colors, result_num_colors

    return None, None


def format_solution(colors, num_vertices):
    """Format the solution according to the output schema."""
    unique_colors = sorted(set(colors))
    color_map = {old: new for new, old in enumerate(unique_colors)}

    coloring = {}
    color_classes = defaultdict(list)

    for v in range(num_vertices):
        c = color_map[colors[v]]
        coloring[str(v)] = c
        color_classes[str(c)].append(v)

    return {
        "objective_value": len(unique_colors),
        "coloring": coloring,
        "color_classes": dict(color_classes)
    }


def main():
    args = parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    rng = random.Random(42)

    # Load instance
    graph = load_instance(args.instance_path)
    num_vertices = graph['num_vertices']
    num_edges = graph['num_edges']
    edges = [tuple(e) for e in graph['edges']]

    # Handle trivial case: no vertices
    if num_vertices == 0:
        solution = {"objective_value": 0, "coloring": {}, "color_classes": {}}
        if logger:
            logger.log_solution(0, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    # Build adjacency
    adj = build_adjacency(num_vertices, edges)

    # Handle trivial case: no edges
    if num_edges == 0:
        colors = [0] * num_vertices
        solution = format_solution(colors, num_vertices)
        if logger:
            logger.log_solution(solution["objective_value"], solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    # Step 1: DSATUR greedy coloring
    colors, num_colors = dsatur_coloring(num_vertices, adj)
    best_colors = colors[:]
    best_num_colors = num_colors

    solution = format_solution(best_colors, num_vertices)
    if logger:
        logger.log_solution(solution["objective_value"], solution)

    # Determine if Gurobi is feasible for this instance
    total_vars = num_vertices * best_num_colors
    total_edge_constraints = len(edges) * best_num_colors
    use_gurobi = (total_vars <= 50000 and total_edge_constraints <= 200000)

    # Step 2: Local search to reduce colors
    ls_fraction = 0.4 if use_gurobi else 0.9
    if time.time() - start_time < args.time_limit * ls_fraction:
        improved_colors, improved_num = local_search_reduce(
            num_vertices, adj, best_colors, args.time_limit, start_time, rng, ls_fraction
        )
        if improved_num < best_num_colors:
            best_colors = improved_colors
            best_num_colors = improved_num
            solution = format_solution(best_colors, num_vertices)
            if logger:
                logger.log_solution(solution["objective_value"], solution)

    # Step 3: Gurobi ILP for small/medium graphs
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed

    if use_gurobi and remaining > 2:
        # Renumber colors to be consecutive for Gurobi
        unique_colors = sorted(set(best_colors))
        color_map = {old: new for new, old in enumerate(unique_colors)}
        renumbered = [color_map[c] for c in best_colors]
        k = len(unique_colors)

        gurobi_result = solve_with_gurobi(
            num_vertices, edges, adj, renumbered, k, remaining * 0.9
        )

        if gurobi_result[0] is not None:
            g_colors, g_num = gurobi_result
            if g_num < best_num_colors:
                best_colors = g_colors
                best_num_colors = g_num
                solution = format_solution(best_colors, num_vertices)
                if logger:
                    logger.log_solution(solution["objective_value"], solution)

    # Write final solution
    solution = format_solution(best_colors, num_vertices)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()