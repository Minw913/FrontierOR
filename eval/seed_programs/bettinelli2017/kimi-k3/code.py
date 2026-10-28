import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_solution_dict(selected, profits):
    selected = sorted(int(i) for i in selected)
    return {
        "objective_value": float(sum(profits[i] for i in selected)),
        "selected_items": selected,
    }


def greedy_solution(n, capacity, profits, weights, adj):
    """Simple feasible fallback: density-ordered greedy respecting conflicts."""
    order = sorted(range(n), key=lambda i: (profits[i] / max(weights[i], 1), profits[i]), reverse=True)
    chosen = set()
    total_w = 0
    total_p = 0
    for i in order:
        if total_w + weights[i] > capacity:
            continue
        if any(j in chosen for j in adj[i]):
            continue
        chosen.add(i)
        total_w += weights[i]
        total_p += profits[i]
    return total_p, chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    start = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["num_items"])
    capacity = int(inst["capacity"])
    profits = [int(p) for p in inst["items"]["profits"]]
    weights = [int(w) for w in inst["items"]["weights"]]

    adj = [set() for _ in range(n)]
    edge_list = []
    seen = set()
    for e in inst["conflict_graph"]["edges"]:
        i, j = int(e[0]), int(e[1])
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        edge_list.append(key)
        adj[key[0]].add(key[1])
        adj[key[1]].add(key[0])

    # Greedy feasible solution as an initial incumbent / fallback.
    g_val, g_set = greedy_solution(n, capacity, profits, weights, adj)
    best_val = float(g_val)
    best_sel = set(g_set)
    if logger:
        logger.log_solution(best_val, build_solution_dict(best_sel, profits))

    remaining = args.time_limit - (time.time() - start)
    if remaining > 0.05 and n > 0:
        try:
            model = gp.Model("knapsack_conflicts")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(remaining, 0.05)

            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            model.setObjective(gp.quicksum(profits[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
            model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "cap")
            for (i, j) in edge_list:
                model.addConstr(x[i] + x[j] <= 1, name=f"conf_{i}_{j}")

            # Provide greedy start.
            for i in range(n):
                x[i].Start = 1.0 if i in g_set else 0.0

            def callback(m, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        obj = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
                        sol = m.cbGetSolution([x[i] for i in range(n)])
                        sel = [i for i in range(n) if sol[i] > 0.5]
                        if logger:
                            logger.log_solution(float(obj), build_solution_dict(sel, profits))
                    except Exception:
                        pass

            model.optimize(callback)

            if model.SolCount > 0:
                vals = model.getAttr("X", [x[i] for i in range(n)])
                sel = {i for i in range(n) if vals[i] > 0.5}
                val = float(sum(profits[i] for i in sel))
                if val > best_val + 1e-9:
                    best_val = val
                    best_sel = sel
        except Exception:
            pass

    solution = build_solution_dict(best_sel, profits)
    solution["objective_value"] = float(best_val)

    if logger:
        logger.log_solution(best_val, solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()