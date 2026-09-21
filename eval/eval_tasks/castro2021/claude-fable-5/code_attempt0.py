import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=600)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = inst["n"]
    m = inst["m"]
    supplies = np.array(inst["supplies"], dtype=float)
    demands = np.array(inst["demands"], dtype=float)
    arc_capacity = float(inst["arc_capacity"])
    linear_costs = np.array(inst["linear_costs"], dtype=float)
    cost_type = inst.get("cost_type", "linear_fractional")
    quadratic_costs = None
    if cost_type == "quadratic":
        quadratic_costs = np.array(inst["quadratic_costs"], dtype=float)

    remaining = max(1.0, args.time_limit - (time.time() - start_time) - 1.0)

    model = gp.Model("transport")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = remaining

    # Flow variables x[i, j], 0 <= x <= arc_capacity
    x = model.addMVar((n, m), lb=0.0, ub=arc_capacity, name="x")

    # Demand constraints: sum over suppliers == demand[j]
    model.addConstr(x.sum(axis=0) == demands, name="demand")

    # Supply constraints: sum over customers <= supply[i]
    model.addConstr(x.sum(axis=1) <= supplies, name="supply")

    # Objective
    if quadratic_costs is not None:
        obj = (linear_costs * x).sum() + (quadratic_costs * x * x).sum()
    else:
        obj = (linear_costs * x).sum()
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    flows_out = {}
    objective_value = None

    if model.SolCount > 0:
        xv = x.X
        # clip tiny negatives
        xv = np.clip(xv, 0.0, arc_capacity)
        for i in range(n):
            for j in range(m):
                flows_out[f"x_{i}_{j}"] = float(xv[i, j])
        if quadratic_costs is not None:
            objective_value = float(
                (linear_costs * xv).sum() + (quadratic_costs * xv * xv).sum()
            )
        else:
            objective_value = float((linear_costs * xv).sum())

        if logger:
            logger.log_solution(objective_value, {
                "objective_value": objective_value,
                "flows": flows_out,
            })
    else:
        # Fallback: greedy feasible construction (should rarely be needed)
        xv = np.zeros((n, m))
        rem_supply = supplies.copy()
        order = np.argsort(linear_costs, axis=0)
        for j in range(m):
            need = demands[j]
            for i in order[:, j]:
                if need <= 1e-12:
                    break
                amt = min(need, rem_supply[i], arc_capacity)
                xv[i, j] += amt
                rem_supply[i] -= amt
                need -= amt
        for i in range(n):
            for j in range(m):
                flows_out[f"x_{i}_{j}"] = float(xv[i, j])
        if quadratic_costs is not None:
            objective_value = float(
                (linear_costs * xv).sum() + (quadratic_costs * xv * xv).sum()
            )
        else:
            objective_value = float((linear_costs * xv).sum())
        if logger:
            logger.log_solution(objective_value, {
                "objective_value": objective_value,
                "flows": flows_out,
            })

    solution = {
        "objective_value": objective_value,
        "flows": flows_out,
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()