import json
import argparse
import time
import random
import sys

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False


def read_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_adjacency(instance):
    n = instance['graph']['num_vertices']
    edges = instance['graph']['edges']
    adj = [set() for _ in range(n)]
    unique_edges = set()
    for e in edges:
        u, v = e[0], e[1]
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
            unique_edges.add((min(u, v), max(u, v)))
    return adj, n, list(unique_edges)


def dsatur_coloring(adj, n):
    colors = [-1] * n
    uncolored = set(range(n))
    nbr_colors = [set() for _ in range(n)]
    degrees = [len(adj[i]) for i in range(n)]
    while uncolored:
        v = max(uncolored, key=lambda x: (len(nbr_colors[x]), degrees[x]))
        used = nbr_colors[v]
        c = 0
        while c in used:
            c += 1
        colors[v] = c
        uncolored.remove(v)
        for u in adj[v]:
            if u in uncolored:
                nbr_colors[u].add(c)
    k = max(colors) + 1 if colors else 0
    return colors, k


def dsatur_coloring_rand(adj, n, seed=0):
    random.seed(seed)
    colors = [-1] * n
    uncolored = set(range(n))
    nbr_colors = [set() for _ in range(n)]
    degrees = [len(adj[i]) for i in range(n)]
    while uncolored:
        max_sat = max(len(nbr_colors[v]) for v in uncolored)
        candidates = [v for v in uncolored if len(nbr_colors[v]) == max_sat]
        if len(candidates) > 1:
            max_deg = max(degrees[v] for v in candidates)
            deg_candidates = [v for v in candidates if degrees[v] == max_deg]
            if len(deg_candidates) > 1:
                v = random.choice(deg_candidates)
            else:
                v = deg_candidates[0]
        else:
            v = candidates[0]
        used = nbr_colors[v]
        c = 0
        while c in used:
            c += 1
        colors[v] = c
        uncolored.remove(v)
        for u in adj[v]:
            if u in uncolored:
                nbr_colors[u].add(c)
    k = max(colors) + 1 if colors else 0
    return colors, k


def improve_coloring(adj, n, colors):
    colors = colors[:]
    while True:
        k = max(colors) + 1
        if k <= 1:
            break
        target = k - 1
        # direct moves
        moved = True
        while moved:
            moved = False
            for v in range(n):
                if colors[v] == target:
                    for c in range(target):
                        if all(colors[u] != c for u in adj[v]):
                            colors[v] = c
                            moved = True
                            break
        if target not in colors:
            unique = sorted(set(colors))
            mapping = {c: i for i, c in enumerate(unique)}
            colors = [mapping[c] for c in colors]
            continue
        # Kempe chains
        improved = False
        for v in [i for i in range(n) if colors[i] == target]:
            for c in range(target):
                if c == target:
                    continue
                if all(colors[u] != c for u in adj[v]):
                    colors[v] = c
                    improved = True
                    break
                visited = set()
                stack = [v]
                while stack:
                    u = stack.pop()
                    if u in visited:
                        continue
                    visited.add(u)
                    for w in adj[u]:
                        if colors[w] == target or colors[w] == c:
                            if w not in visited:
                                stack.append(w)
                for u in visited:
                    if colors[u] == target:
                        colors[u] = c
                    elif colors[u] == c:
                        colors[u] = target
                if target not in colors:
                    improved = True
                    break
            if improved:
                break
        if improved and target not in colors:
            unique = sorted(set(colors))
            mapping = {c: i for i, c in enumerate(unique)}
            colors = [mapping[c] for c in colors]
        else:
            break
    return colors


def format_solution(colors, n):
    unique = sorted(set(colors))
    mapping = {c: i for i, c in enumerate(unique)}
    new_colors = [mapping[c] for c in colors]
    k = len(unique)
    selected_sets = [[] for _ in range(k)]
    for v, c in enumerate(new_colors):
        selected_sets[c].append(v)
    for lst in selected_sets:
        lst.sort()
    coloring = {str(v): new_colors[v] for v in range(n)}
    return k, selected_sets, coloring


def find_max_clique_greedy(adj, n):
    if n == 0:
        return []
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    best = []
    for v in order[:20]:
        clique = [v]
        for u in order:
            if u == v:
                continue
            if all(u in adj[w] for w in clique):
                clique.append(u)
        if len(clique) > len(best):
            best = clique
    return best


def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    start_time = time.time()
    deadline = start_time + args.time_limit

    instance = read_instance(args.instance_path)
    adj, n, unique_edges = build_adjacency(instance)
    LB = instance['bounds']['clique_lower_bound']
    UB = instance['bounds']['greedy_upper_bound']

    if n == 0:
        solution = {"objective_value": 0, "selected_sets": [], "coloring": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # Initial coloring
    best_colors, best_k = dsatur_coloring(adj, n)
    best_colors = improve_coloring(adj, n, best_colors)
    best_k, best_sel_sets, best_coloring = format_solution(best_colors, n)
    if logger:
        logger.log_solution(best_k, {"objective_value": best_k, "selected_sets": best_sel_sets, "coloring": best_coloring})

    if best_k > LB:
        # Random restarts for initial improvement
        max_init_time = start_time + min(args.time_limit * 0.2, 5)
        for seed in range(1, 30):
            if time.time() > max_init_time:
                break
            colors, k = dsatur_coloring_rand(adj, n, seed)
            colors = improve_coloring(adj, n, colors)
            k = max(colors) + 1
            if k < best_k:
                best_colors = colors[:]
                best_k, best_sel_sets, best_coloring = format_solution(colors, n)
                if logger:
                    logger.log_solution(best_k, {"objective_value": best_k, "selected_sets": best_sel_sets, "coloring": best_coloring})
            if best_k == LB:
                break

        # Exact search with Gurobi
        if GUROBI_AVAILABLE and n <= 1000 and best_k > LB:
            clique = find_max_clique_greedy(adj, n)
            for k in range(best_k - 1, LB - 1, -1):
                if time.time() >= deadline:
                    break
                remaining = deadline - time.time()
                if remaining < 2:
                    break
                try:
                    model = gp.Model()
                    model.setParam('Seed', 0)
                    model.setParam('MIPGap', 1e-4)
                    model.setParam('NumericFocus', 0)
                    model.setParam('Threads', 1)
                    model.setParam('OutputFlag', 0)
                    model.setParam('TimeLimit', max(1, remaining - 1))

                    x = {}
                    for v in range(n):
                        for c in range(k):
                            x[v, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")

                    for v in range(n):
                        model.addConstr(gp.quicksum(x[v, c] for c in range(k)) == 1)

                    for u, v in unique_edges:
                        for c in range(k):
                            model.addConstr(x[u, c] + x[v, c] <= 1)

                    # Symmetry breaking
                    if clique and len(clique) > 1:
                        for idx, v in enumerate(clique):
                            if idx < k:
                                model.addConstr(x[v, idx] == 1)
                    else:
                        model.addConstr(x[0, 0] == 1)

                    model.optimize()

                    if model.SolCount > 0:
                        new_colors = [-1] * n
                        for v in range(n):
                            for c in range(k):
                                if x[v, c].X > 0.5:
                                    new_colors[v] = c
                                    break
                        best_colors = new_colors[:]
                        best_k, best_sel_sets, best_coloring = format_solution(new_colors, n)
                        if logger:
                            logger.log_solution(best_k, {"objective_value": best_k, "selected_sets": best_sel_sets, "coloring": best_coloring})
                        if model.status == GRB.OPTIMAL:
                            continue
                        else:
                            break
                    elif model.status == GRB.INFEASIBLE:
                        break
                    else:
                        break
                except Exception:
                    break

        # Fallback heuristic search for remaining time
        if (not GUROBI_AVAILABLE or n > 1000) and best_k > LB and time.time() < deadline:
            while time.time() < deadline - 2:
                seed = random.randint(0, 100000)
                colors, k = dsatur_coloring_rand(adj, n, seed)
                colors = improve_coloring(adj, n, colors)
                k = max(colors) + 1
                if k < best_k:
                    best_colors = colors[:]
                    best_k, best_sel_sets, best_coloring = format_solution(colors, n)
                    if logger:
                        logger.log_solution(best_k, {"objective_value": best_k, "selected_sets": best_sel_sets, "coloring": best_coloring})
                if best_k == LB:
                    break

    # Final output
    solution = {
        "objective_value": best_k,
        "selected_sets": best_sel_sets,
        "coloring": best_coloring
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    if logger:
        logger.log_solution(best_k, solution)


if __name__ == '__main__':
    solve()