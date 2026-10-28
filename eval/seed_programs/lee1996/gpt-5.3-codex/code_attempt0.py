import argparse
import json
import math
from typing import Dict, List, Tuple, Any

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def read_instance(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)

    m = data.get("parameters", {}).get("m", data["target_nodes"]["count"])
    n = data.get("parameters", {}).get("n", data["steiner_nodes"]["count"])

    assignment_costs = data["cost_matrices"]["assignment_costs"]["matrix"]
    inter_costs = data["cost_matrices"]["interconnection_costs"]["matrix"]
    weights = data["steiner_nodes"]["node_weights"]

    return {
        "m": m,
        "n": n,
        "assignment_costs": assignment_costs,
        "inter_costs": inter_costs,
        "weights": weights,
    }


def single_hub_heuristic(inst: Dict[str, Any]) -> Dict[str, Any]:
    m, n = inst["m"], inst["n"]
    c = inst["assignment_costs"]
    w = inst["weights"]

    best_j = 0
    best_obj = float("inf")
    for j in range(n):
        obj = w[j] + sum(c[i][j] for i in range(m))
        if obj < best_obj:
            best_obj = obj
            best_j = j

    assignments = {str(i): best_j for i in range(m)}
    return {
        "objective_value": float(best_obj),
        "active_steiner_nodes": [best_j],
        "assignments": assignments,
        "steiner_tree_edges": [],
    }


def compute_objective(sol: Dict[str, Any], inst: Dict[str, Any]) -> float:
    c = inst["assignment_costs"]
    d = inst["inter_costs"]
    w = inst["weights"]

    obj = 0.0
    for ti, sj in sol["assignments"].items():
        i = int(ti)
        obj += c[i][sj]
    for j in sol["active_steiner_nodes"]:
        obj += w[j]
    for i, j in sol["steiner_tree_edges"]:
        obj += d[i][j]
    return float(obj)


class IncumbentCallback:
    def __init__(self, logger: SolutionLogger):
        self.logger = logger
        self.best = float("inf")

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < self.best:
                self.best = obj
                self.logger.log(float(obj))


def solve_mip(inst: Dict[str, Any], time_limit: int, logger: SolutionLogger = None) -> Dict[str, Any]:
    m, n = inst["m"], inst["n"]
    c = inst["assignment_costs"]
    d = inst["inter_costs"]
    w = inst["weights"]

    # Heuristic baseline (always feasible if n >= 1)
    best_sol = single_hub_heuristic(inst)
    if logger:
        logger.log(best_sol["objective_value"])

    model = gp.Model("weighted_euclidean_steiner_tree_assignment")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0, int(time_limit))

    # Variables
    x = model.addVars(m, n, vtype=GRB.BINARY, name="x")  # assignment target->steiner
    y = model.addVars(n, vtype=GRB.BINARY, name="y")     # steiner active
    r = model.addVars(n, vtype=GRB.BINARY, name="r")     # chosen root steiner
    a = model.addVars(((i, j) for i in range(n) for j in range(n) if i != j),
                      vtype=GRB.BINARY, name="a")        # directed tree arcs
    u = model.addVars(n, lb=0.0, ub=float(n), vtype=GRB.CONTINUOUS, name="u")  # MTZ order

    # Objective
    model.setObjective(
        gp.quicksum(c[i][j] * x[i, j] for i in range(m) for j in range(n)) +
        gp.quicksum(d[i][j] * a[i, j] for i in range(n) for j in range(n) if i != j) +
        gp.quicksum(w[j] * y[j] for j in range(n)),
        GRB.MINIMIZE
    )

    # Assignment constraints
    for i in range(m):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f"assign_{i}")
        for j in range(n):
            model.addConstr(x[i, j] <= y[j], name=f"assign_active_{i}_{j}")

    # Root selection
    model.addConstr(gp.quicksum(r[j] for j in range(n)) == 1, name="one_root")
    for j in range(n):
        model.addConstr(r[j] <= y[j], name=f"root_active_{j}")

    # In-degree structure: each active node has exactly one parent (root comes from super-root via r)
    for j in range(n):
        model.addConstr(
            gp.quicksum(a[i, j] for i in range(n) if i != j) + r[j] == y[j],
            name=f"indeg_{j}"
        )

    # Arc activation linking
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            model.addConstr(a[i, j] <= y[i], name=f"arc_from_active_{i}_{j}")
            model.addConstr(a[i, j] <= y[j], name=f"arc_to_active_{i}_{j}")

    # MTZ acyclicity constraints
    # u_j >= u_i + 1 - n*(1-a_ij) - n*(1-y_j)
    # plus y_j <= u_j <= n*y_j
    for j in range(n):
        model.addConstr(u[j] >= y[j], name=f"u_lb_{j}")
        model.addConstr(u[j] <= n * y[j], name=f"u_ub_{j}")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            model.addConstr(
                u[j] >= u[i] + 1 - n * (1 - a[i, j]) - n * (1 - y[j]),
                name=f"mtz_{i}_{j}"
            )

    # MIP start from heuristic (single hub)
    hub = best_sol["active_steiner_nodes"][0]
    for j in range(n):
        y[j].Start = 1.0 if j == hub else 0.0
        r[j].Start = 1.0 if j == hub else 0.0
        u[j].Start = 1.0 if j == hub else 0.0
    for i in range(m):
        for j in range(n):
            x[i, j].Start = 1.0 if j == hub else 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                a[i, j].Start = 0.0

    cb = IncumbentCallback(logger) if logger else None
    model.optimize(cb if cb else None)

    # If MIP found feasible, extract
    if model.SolCount > 0:
        active = [j for j in range(n) if y[j].X > 0.5]
        if not active:
            return best_sol

        assignments = {}
        for i in range(m):
            # pick assigned steiner by max value (robust to tolerances)
            j_best = max(range(n), key=lambda j: x[i, j].X)
            if y[j_best].X <= 0.5:
                # fallback among active steiners by min assignment cost
                j_best = min(active, key=lambda j: c[i][j])
            assignments[str(i)] = int(j_best)

        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and a[i, j].X > 0.5:
                    edges.append([int(i), int(j)])

        mip_sol = {
            "objective_value": float(model.ObjVal),
            "active_steiner_nodes": [int(j) for j in sorted(active)],
            "assignments": assignments,
            "steiner_tree_edges": edges,
        }

        # Recompute objective for consistency
        mip_sol["objective_value"] = compute_objective(mip_sol, inst)

        if mip_sol["objective_value"] + 1e-7 < best_sol["objective_value"]:
            best_sol = mip_sol

    return best_sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    inst = read_instance(args.instance_path)
    sol = solve_mip(inst, args.time_limit, logger)

    # Ensure exact output schema fields
    out = {
        "objective_value": float(sol["objective_value"]),
        "active_steiner_nodes": [int(v) for v in sol["active_steiner_nodes"]],
        "assignments": {str(k): int(v) for k, v in sol["assignments"].items()},
        "steiner_tree_edges": [[int(i), int(j)] for i, j in sol["steiner_tree_edges"]],
    }

    with open(args.solution_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()