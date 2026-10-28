import argparse
import json
import time
import random
from collections import defaultdict

from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)

    g = data["graph"]
    n = int(g["num_vertices"])
    edges_in = g["edges"]

    edge_set = set()
    for e in edges_in:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        if u > v:
            u, v = v, u
        edge_set.add((u, v))

    edges = sorted(edge_set)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    bounds = data.get("bounds", {})
    lb = int(bounds.get("clique_lower_bound", 1))
    ub_hint = bounds.get("greedy_upper_bound", None)
    ub_hint = int(ub_hint) if ub_hint is not None else None

    return n, edges, adj, lb, ub_hint


def dsatur_coloring(n, adj, randomize=False, rng=None):
    if rng is None:
        rng = random.Random(0)

    uncolored = set(range(n))
    colors = [-1] * n
    neighbor_colors = [set() for _ in range(n)]
    degrees = [len(adj[v]) for v in range(n)]

    while uncolored:
        # Select vertex with max saturation degree; tie-break by degree
        best_sat = -1
        best_deg = -1
        candidates = []
        for v in uncolored:
            sat = len(neighbor_colors[v])
            deg = degrees[v]
            if sat > best_sat or (sat == best_sat and deg > best_deg):
                best_sat = sat
                best_deg = deg
                candidates = [v]
            elif sat == best_sat and deg == best_deg:
                candidates.append(v)

        if randomize and len(candidates) > 1:
            v = rng.choice(candidates)
        else:
            v = min(candidates)

        forbidden = neighbor_colors[v]
        c = 0
        feasible = []
        while c <= n:
            if c not in forbidden:
                feasible.append(c)
                if not randomize:
                    break
            c += 1
        if randomize and feasible:
            # biased toward smaller colors but allows diversification
            pick_idx = 0 if rng.random() < 0.7 else rng.randrange(len(feasible))
            c = feasible[pick_idx]
        else:
            c = feasible[0]

        colors[v] = c
        uncolored.remove(v)

        for w in adj[v]:
            if w in uncolored:
                neighbor_colors[w].add(c)

    k = max(colors) + 1 if n > 0 else 0
    return colors, k


def compact_coloring(colors):
    used = sorted(set(colors))
    remap = {c: i for i, c in enumerate(used)}
    new_colors = [remap[c] for c in colors]
    k = len(used)
    classes = [[] for _ in range(k)]
    for v, c in enumerate(new_colors):
        classes[c].append(v)
    for cls in classes:
        cls.sort()
    return new_colors, classes, k


def build_solution_dict(colors):
    colors, classes, k = compact_coloring(colors)
    return {
        "objective_value": int(k),
        "selected_sets": classes,
        "coloring": {str(v): int(c) for v, c in enumerate(colors)},
    }


def heuristic_phase(n, adj, time_budget, lb, ub_hint, logger):
    t0 = time.time()
    rng = random.Random(42)

    best_colors, best_k = dsatur_coloring(n, adj, randomize=False, rng=rng)
    if ub_hint is not None:
        best_k = min(best_k, ub_hint) if ub_hint > 0 else best_k
    if logger:
        logger.log(best_k)

    # randomized multi-start improvements
    it = 0
    while time.time() - t0 < time_budget:
        it += 1
        cand_colors, cand_k = dsatur_coloring(n, adj, randomize=True, rng=rng)
        if cand_k < best_k:
            best_k = cand_k
            best_colors = cand_colors
            if logger:
                logger.log(best_k)
            if best_k <= lb:
                break
        # avoid too many iterations on large instances if no gain
        if it > 500:
            break

    return best_colors, best_k


def mip_improve(n, edges, adj, initial_colors, initial_k, lb, time_limit, logger):
    # If already at lower bound, no need for MIP.
    if initial_k <= lb or time_limit <= 0:
        return initial_colors, initial_k

    K = initial_k
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)

    model = Model("graph_coloring")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, time_limit)

    x = model.addVars(n, K, vtype=GRB.BINARY, name="x")
    y = model.addVars(K, vtype=GRB.BINARY, name="y")

    # Each vertex gets exactly one color
    model.addConstrs((quicksum(x[v, c] for c in range(K)) == 1 for v in range(n)))

    # Adjacent vertices cannot share a color
    for c in range(K):
        model.addConstrs((x[u, c] + x[v, c] <= 1 for (u, v) in edges))

    # Link color usage
    model.addConstrs((x[v, c] <= y[c] for v in range(n) for c in range(K)))

    # Symmetry breaking: contiguous use of colors
    model.addConstrs((y[c] >= y[c + 1] for c in range(K - 1)))

    # Objective lower bound
    model.addConstr(quicksum(y[c] for c in range(K)) >= lb)

    # Additional symmetry breaking
    if n > 0 and K > 0:
        model.addConstr(x[order[0], 0] == 1)
    cap = min(n, 30)
    for i in range(cap):
        v = order[i]
        for c in range(i + 1, K):
            model.addConstr(x[v, c] == 0)

    model.setObjective(quicksum(y[c] for c in range(K)), GRB.MINIMIZE)

    # Warm start from heuristic
    init_compact, _, init_k_compact = compact_coloring(initial_colors)
    for v in range(n):
        c0 = init_compact[v]
        for c in range(K):
            x[v, c].Start = 1.0 if c == c0 else 0.0
    for c in range(K):
        y[c].Start = 1.0 if c < init_k_compact else 0.0

    best_logged = [initial_k]

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            obj_int = int(round(obj))
            if obj_int < best_logged[0]:
                best_logged[0] = obj_int
                if logger:
                    logger.log(obj_int)

    model.optimize(callback)

    if model.SolCount == 0:
        return initial_colors, initial_k

    sol_colors = []
    for v in range(n):
        chosen = -1
        best_val = -1.0
        for c in range(K):
            val = x[v, c].X
            if val > best_val:
                best_val = val
                chosen = c
            if val > 0.5:
                chosen = c
                break
        sol_colors.append(chosen)

    sol_colors, _, sol_k = compact_coloring(sol_colors)
    if sol_k < initial_k:
        # in rare cases callback may not have logged (e.g. if disabled paths)
        if logger and sol_k < best_logged[0]:
            logger.log(sol_k)
        return sol_colors, sol_k
    return initial_colors, initial_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    global_start = time.time()
    n, edges, adj, lb, ub_hint = read_instance(args.instance_path)

    # Heuristic phase budget
    total_t = max(1, int(args.time_limit))
    heur_budget = min(10.0, max(0.5, 0.15 * total_t))

    best_colors, best_k = heuristic_phase(n, adj, heur_budget, lb, ub_hint, logger)

    elapsed = time.time() - global_start
    remain = max(0.0, total_t - elapsed)

    # MIP improvement/proof phase
    if remain > 0.2 and best_k > lb:
        best_colors, best_k = mip_improve(
            n=n,
            edges=edges,
            adj=adj,
            initial_colors=best_colors,
            initial_k=best_k,
            lb=lb,
            time_limit=remain,
            logger=logger,
        )

    solution = build_solution_dict(best_colors)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()