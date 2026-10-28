import argparse
import json
import time
import random
import sys

from solution_logger import SolutionLogger


def dsatur_color(selected, adj):
    """Color induced subgraph on 'selected' vertices with DSATUR. Returns dict v->color."""
    sel_set = set(selected)
    nbrs = {v: [u for u in adj[v] if u in sel_set] for v in selected}
    color = {}
    sat = {v: set() for v in selected}
    uncolored = set(selected)
    while uncolored:
        # pick vertex with max saturation, tie-break by degree
        v = max(uncolored, key=lambda u: (len(sat[u]), len(nbrs[u])))
        c = 0
        while c in sat[v]:
            c += 1
        color[v] = c
        uncolored.remove(v)
        for u in nbrs[v]:
            if u in uncolored:
                sat[u].add(c)
    return color


def greedy_select(clusters, adj, rng=None):
    """Pick one vertex per cluster greedily minimizing conflicts with already selected."""
    order = sorted(range(len(clusters)), key=lambda c: len(clusters[c]))
    sel_set = set()
    selected = {}
    for ci in order:
        cand = clusters[ci]
        best = []
        best_d = None
        for v in cand:
            d = 0
            for u in adj[v]:
                if u in sel_set:
                    d += 1
            if best_d is None or d < best_d:
                best_d = d
                best = [v]
            elif d == best_d:
                best.append(v)
        v = rng.choice(best) if rng else best[0]
        selected[ci] = v
        sel_set.add(v)
    return selected


def random_select(clusters, rng):
    return {ci: rng.choice(clusters[ci]) for ci in range(len(clusters))}


def make_solution(sel_by_cluster, coloring):
    used = sorted(set(coloring.values()))
    remap = {c: i for i, c in enumerate(used)}
    return {
        "objective_value": float(len(used)),
        "selected_vertices": {str(ci): int(v) for ci, v in sel_by_cluster.items()},
        "coloring": {str(v): int(remap[c]) for v, c in coloring.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = data["graph"]["num_vertices"]
    raw_edges = data["graph"]["edges"]
    clusters = data["partition"]["clusters"]
    num_clusters = len(clusters)

    adj = [[] for _ in range(n)]
    eset = set()
    for e in raw_edges:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in eset:
            continue
        eset.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
    edges = sorted(eset)

    vcluster = [-1] * n
    for ci, cl in enumerate(clusters):
        for v in cl:
            vcluster[v] = ci

    # ------------- Heuristic phase (multi-start) -------------
    rng = random.Random(0)
    best_sel = greedy_select(clusters, adj)
    best_col = dsatur_color(list(best_sel.values()), adj)
    best_k = len(set(best_col.values())) if best_col else 0
    if best_k == 0:
        best_k = 1  # degenerate; but clusters non-empty so shouldn't happen
    best_solution = make_solution(best_sel, best_col)
    if logger:
        logger.log_solution(best_solution["objective_value"], best_solution)

    heur_budget = min(max(2.0, args.time_limit * 0.10), 15.0)
    it = 0
    while time.time() - start_time < heur_budget and best_k > 1:
        it += 1
        if it % 3 == 0:
            sel = random_select(clusters, rng)
        else:
            sel = greedy_select(clusters, adj, rng)
        col = dsatur_color(list(sel.values()), adj)
        k = len(set(col.values()))
        if k < best_k:
            best_k = k
            best_sel = sel
            best_col = col
            best_solution = make_solution(best_sel, best_col)
            if logger:
                logger.log_solution(best_solution["objective_value"], best_solution)
        if it > 400:
            break

    # If trivially optimal (1 color or no clusters), finish now.
    remaining = args.time_limit - (time.time() - start_time) - 1.0
    if best_k <= 1 or num_clusters == 0 or remaining < 2.0:
        with open(args.solution_path, "w") as f:
            json.dump(best_solution, f, indent=2)
        return

    # ------------- Exact MIP phase (Gurobi) -------------
    try:
        import gurobipy as gp
        from gurobipy import GRB

        K = best_k  # upper bound on colors needed
        model = gp.Model("selective_coloring")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(1.0, remaining)

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        y = model.addVars(n, K, vtype=GRB.BINARY, name="y")
        z = model.addVars(K, vtype=GRB.BINARY, name="z")

        model.setObjective(gp.quicksum(z[c] for c in range(K)), GRB.MINIMIZE)

        for ci in range(num_clusters):
            model.addConstr(gp.quicksum(x[v] for v in clusters[ci]) == 1)

        for v in range(n):
            model.addConstr(gp.quicksum(y[v, c] for c in range(K)) == x[v])
            for c in range(K):
                model.addConstr(y[v, c] <= z[c])

        for (u, v) in edges:
            if vcluster[u] == vcluster[v]:
                continue  # both can never be selected simultaneously
            for c in range(K):
                model.addConstr(y[u, c] + y[v, c] <= z[c])

        # Symmetry breaking on color usage
        for c in range(K - 1):
            model.addConstr(z[c] >= z[c + 1])
        model.addConstr(z[0] == 1)

        # Warm start from heuristic
        for v in range(n):
            x[v].Start = 0
            for c in range(K):
                y[v, c].Start = 0
        used_cols = sorted(set(best_col.values()))
        cmap = {c: i for i, c in enumerate(used_cols)}
        for ci, v in best_sel.items():
            x[v].Start = 1
            y[v, cmap[best_col[v]]].Start = 1
        for c in range(K):
            z[c].Start = 1 if c < len(used_cols) else 0

        # Callback for incumbent logging
        xlist = [x[v] for v in range(n)]
        ylist = [y[v, c] for v in range(n) for c in range(K)]

        state = {"best_k": best_k, "best_sol": best_solution}

        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                obj = int(round(m.cbGet(GRB.Callback.MIPSOL_OBJ)))
                if obj < state["best_k"]:
                    xvals = m.cbGetSolution(xlist)
                    yvals = m.cbGetSolution(ylist)
                    sel = {}
                    col = {}
                    for v in range(n):
                        if xvals[v] > 0.5:
                            sel[vcluster[v]] = v
                            for c in range(K):
                                if yvals[v * K + c] > 0.5:
                                    col[v] = c
                                    break
                    if len(sel) == num_clusters and len(col) == num_clusters:
                        sol = make_solution(sel, col)
                        state["best_k"] = obj
                        state["best_sol"] = sol
                        if logger:
                            logger.log_solution(sol["objective_value"], sol)

        model.optimize(callback)

        # Extract final solution if better
        if model.SolCount > 0:
            obj = int(round(model.ObjVal))
            if obj <= state["best_k"]:
                sel = {}
                col = {}
                for v in range(n):
                    if x[v].X > 0.5:
                        sel[vcluster[v]] = v
                        for c in range(K):
                            if y[v, c].X > 0.5:
                                col[v] = c
                                break
                if len(sel) == num_clusters and len(col) == num_clusters:
                    sol = make_solution(sel, col)
                    if sol["objective_value"] < state["best_sol"]["objective_value"]:
                        state["best_sol"] = sol
                        if logger:
                            logger.log_solution(sol["objective_value"], sol)

        best_solution = state["best_sol"]

    except Exception:
        # Fall back to heuristic solution on any solver failure
        pass

    with open(args.solution_path, "w") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()