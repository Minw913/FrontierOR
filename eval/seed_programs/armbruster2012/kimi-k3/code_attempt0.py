import argparse
import json
import sys
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Balanced graph bisection solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    n = int(data["num_nodes"])
    m = int(data["num_edges"])
    F = int(data["bisection_capacity_F"])
    W = int(data.get("total_node_weight", sum(data["node_weights"])))
    node_weights = [int(w) for w in data["node_weights"]]
    edges = [[int(u), int(v)] for u, v in data["edges"]]
    edge_weights = [float(c) for c in data["edge_weights"]]
    assert len(node_weights) == n
    assert len(edges) == m
    assert len(edge_weights) == m
    return n, m, F, W, node_weights, edges, edge_weights


def build_solution_dict(bits, edges, edge_weights):
    """bits: list of 0/1 group assignments. Returns schema-compliant dict
    with node 0 in partition_S."""
    n = len(bits)
    S = [i for i in range(n) if bits[i] > 0.5]
    C = [i for i in range(n) if bits[i] <= 0.5]
    if 0 not in S:
        S, C = C, S
    obj = 0.0
    for (u, v), c in zip(edges, edge_weights):
        if (bits[u] > 0.5) != (bits[v] > 0.5):
            obj += c
    return {
        "objective_value": float(obj),
        "partition_S": S,
        "partition_complement": C,
    }


def greedy_fallback(n, W, F, node_weights, edges, edge_weights):
    """Simple feasible construction: assign nodes to S until it cannot grow,
    then swap if needed to make both groups satisfy the capacity."""
    order = sorted(range(n), key=lambda i: -node_weights[i])
    bits = [0] * n
    sumS = 0
    target_min = W - F
    for i in order:
        if sumS + node_weights[i] <= F and sumS < target_min:
            bits[i] = 1
            sumS += node_weights[i]
    # Ensure minimum size of S
    if sumS < target_min:
        for i in sorted(range(n), key=lambda i: node_weights[i]):
            if bits[i] == 0 and sumS + node_weights[i] <= F:
                bits[i] = 1
                sumS += node_weights[i]
            if sumS >= target_min:
                break
    return bits


def main():
    args = parse_args()
    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n, m, F, W, node_weights, edges, edge_weights = load_instance(args.instance_path)

    if n == 0:
        sol = {"objective_value": 0.0, "partition_S": [], "partition_complement": []}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    remaining = max(1, args.time_limit - 1)

    model = gp.Model("balanced_bisection")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = remaining

    # x[i] = 1 iff node i is in group A (the group containing node 0).
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    # Symmetry breaking: node 0 fixed in group A.
    x[0].lb = 1
    x[0].ub = 1

    # y[e] = 1 iff edge e is cut.
    y = model.addVars(m, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="y")

    # Capacity constraints: both groups must respect the bisection capacity.
    wsum_A = gp.quicksum(node_weights[i] * x[i] for i in range(n))
    model.addConstr(wsum_A <= F, name="cap_A")
    model.addConstr(wsum_A >= W - F, name="cap_B")

    # Cut linearization.
    for e, (u, v) in enumerate(edges):
        model.addConstr(y[e] >= x[u] - x[v], name=f"cut1_{e}")
        model.addConstr(y[e] >= x[v] - x[u], name=f"cut2_{e}")

    model.setObjective(
        gp.quicksum(edge_weights[e] * y[e] for e in range(m)), GRB.MINIMIZE
    )

    best = {"sol": None, "obj": float("inf")}

    # Closure variables for callback
    xvars = [x[i] for i in range(n)]

    def incumbent_callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xv = mdl.cbGetSolution(xvars)
                bits = [1 if xv[i] > 0.5 else 0 for i in range(n)]
                sol = build_solution_dict(bits, edges, edge_weights)
                if sol["objective_value"] < best["obj"] - 1e-9:
                    best["obj"] = sol["objective_value"]
                    best["sol"] = sol
                    if logger:
                        logger.log_solution(sol["objective_value"], sol)
            except Exception:
                pass

    model.optimize(incumbent_callback)

    # Retrieve final best solution
    final_sol = None
    if model.SolCount > 0:
        xv = model.getAttr("X", xvars)
        bits = [1 if xv[i] > 0.5 else 0 for i in range(n)]
        final_sol = build_solution_dict(bits, edges, edge_weights)
    elif best["sol"] is not None:
        final_sol = best["sol"]
    else:
        # Fallback: greedy feasible construction
        bits = greedy_fallback(n, W, F, node_weights, edges, edge_weights)
        final_sol = build_solution_dict(bits, edges, edge_weights)

    if logger and (best["sol"] is None or final_sol["objective_value"] < best["obj"] - 1e-9):
        logger.log_solution(final_sol["objective_value"], final_sol)

    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f)


if __name__ == "__main__":
    main()