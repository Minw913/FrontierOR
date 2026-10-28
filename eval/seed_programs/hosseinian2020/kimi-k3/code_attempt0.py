import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Maximum edge-weight clique solver."
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the solution JSON file is written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional path to a JSONL convergence log file.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = (SolutionLogger(args.log_path, sense="maximize")
              if args.log_path else None)

    start_time = time.time()

    # ------------------------- read instance -------------------------
    with open(args.instance_path) as fh:
        inst = json.load(fh)

    vertices = [int(v) for v in inst["vertices"]]
    n = len(vertices)
    index = {v: i for i, v in enumerate(vertices)}

    adj = [set() for _ in range(n)]
    edges = []          # list of (i, j, weight) with i < j
    wdeg = [0.0] * n    # weighted degree
    seen = set()
    for e in inst["edges"]:
        i = index[e["i"]]
        j = index[e["j"]]
        w = float(e["weight"])
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        adj[i].add(j)
        adj[j].add(i)
        edges.append((key[0], key[1], w))
        wdeg[i] += w
        wdeg[j] += w
    m = len(edges)

    # ------------------------- helpers -------------------------
    def weight_of(clique):
        s = set(clique)
        return sum(w for (i, j, w) in edges if i in s and j in s)

    def solution_dict(obj, clique_idx):
        return {
            "objective_value": float(obj),
            "clique": sorted(int(vertices[i]) for i in clique_idx),
        }

    # ------------------- greedy initial incumbent -------------------
    def greedy(order):
        clique = []
        for v in order:
            for u in clique:
                if u not in adj[v]:
                    break
            else:
                clique.append(v)
        return clique

    orders = [
        sorted(range(n), key=lambda v: -wdeg[v]),
        sorted(range(n), key=lambda v: (-len(adj[v]), -wdeg[v])),
    ]
    best_obj = -1.0
    best_clique = []
    for order in orders:
        c = greedy(order)
        w = weight_of(c)
        if w > best_obj:
            best_obj, best_clique = w, c
    if best_obj < 0.0:
        best_obj = 0.0
        best_clique = []

    final_obj = float(best_obj)
    final_clique = list(best_clique)
    last_logged = [float(best_obj)]

    if logger:
        logger.log_solution(float(best_obj),
                            solution_dict(best_obj, best_clique))

    # ------------------------- MIP refinement -------------------------
    remaining = args.time_limit - (time.time() - start_time)
    if n > 0 and remaining > 0.05:
        model = gp.Model("max_edge_weight_clique")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        y = model.addVars(m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")

        # An edge contributes only if both endpoints are selected.
        for k, (i, j, w) in enumerate(edges):
            model.addConstr(y[k] <= x[i])
            model.addConstr(y[k] <= x[j])

        # Conflict constraints: two non-adjacent vertices cannot both be
        # selected.  If there are too many, add them lazily instead.
        nonedge_count = n * (n - 1) // 2 - m
        use_lazy = nonedge_count > 2_000_000
        if use_lazy:
            model.Params.LazyConstraints = 1
        else:
            for i in range(n):
                ai = adj[i]
                for j in range(i + 1, n):
                    if j not in ai:
                        model.addConstr(x[i] + x[j] <= 1)

        model.setObjective(
            gp.quicksum(w * y[k] for k, (i, j, w) in enumerate(edges)),
            GRB.MAXIMIZE,
        )

        # MIP start from the greedy solution.
        in_best = set(best_clique)
        for v in range(n):
            x[v].Start = 1.0 if v in in_best else 0.0
        for k, (i, j, w) in enumerate(edges):
            y[k].Start = 1.0 if (i in in_best and j in in_best) else 0.0

        xvars = [x[i] for i in range(n)]

        def callback(mdl, where):
            if where != GRB.Callback.MIPSOL:
                return
            vals = mdl.cbGetSolution(xvars)
            sel = [i for i in range(n) if vals[i] > 0.5]
            if use_lazy:
                feasible = True
                for a in range(len(sel)):
                    i = sel[a]
                    ai = adj[i]
                    for b in range(a + 1, len(sel)):
                        j = sel[b]
                        if j not in ai:
                            mdl.cbLazy(xvars[i] + xvars[j] <= 1)
                            feasible = False
                if not feasible:
                    return
            w = weight_of(sel)
            if w > last_logged[0] + 1e-7:
                last_logged[0] = w
                if logger:
                    logger.log_solution(float(w), solution_dict(w, sel))

        remaining = args.time_limit - (time.time() - start_time)
        model.Params.TimeLimit = max(remaining, 0.05)
        model.optimize(callback)

        if model.SolCount > 0:
            vals = [x[i].X for i in range(n)]
            cl = [i for i in range(n) if vals[i] > 0.5]
            w = weight_of(cl)
            if w >= final_obj:
                final_obj = w
                final_clique = cl

    # ------------------------- final output -------------------------
    sol = solution_dict(final_obj, final_clique)

    if logger and final_obj > last_logged[0] + 1e-9:
        logger.log_solution(float(final_obj), sol)

    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh)


if __name__ == "__main__":
    main()