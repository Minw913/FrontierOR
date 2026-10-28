import argparse
import json
import math
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def greedy_feasible_solution(
    demands: List[float],
    capacities: List[float],
    fixed_costs: List[float],
    shipping_costs: List[List[float]],
) -> Tuple[float, List[int], Dict[Tuple[int, int], float], Dict[str, Dict[str, float]]]:
    """
    Build a feasible solution by assigning each customer demand greedily
    to cheapest plants with remaining capacity. Demand can be split.
    Returns:
      objective, open_plants_list, x_frac_dict[(i,j)], assignments_json_dict
    """
    n = len(demands)
    m = len(capacities)

    rem_cap = capacities[:]
    x = {}  # (i,j) -> fraction in [0,1]

    # Precompute plant order by shipping cost for each customer
    sorted_plants = [sorted(range(m), key=lambda j: shipping_costs[i][j]) for i in range(n)]

    for i in range(n):
        d = demands[i]
        if d <= 0:
            # Zero-demand customer: assign fully to cheapest plant (no capacity impact)
            j0 = sorted_plants[i][0]
            x[(i, j0)] = 1.0
            continue

        remaining = d
        for j in sorted_plants[i]:
            if remaining <= 1e-12:
                break
            if rem_cap[j] <= 1e-12:
                continue
            alloc = min(remaining, rem_cap[j])  # in demand units
            frac = alloc / d
            if frac > 0:
                x[(i, j)] = x.get((i, j), 0.0) + frac
                rem_cap[j] -= alloc
                remaining -= alloc

        # Safety fallback (should not happen when total capacity > total demand)
        if remaining > 1e-8:
            # assign remaining fraction to cheapest plant regardless of residual
            # (keeps a complete assignment; capacity may be violated only in pathological case)
            j0 = sorted_plants[i][0]
            frac = remaining / d
            x[(i, j0)] = x.get((i, j0), 0.0) + frac
            remaining = 0.0

    # Determine open plants from usage
    used = set(j for (_, j), v in x.items() if v > 1e-12)
    open_plants = sorted(list(used))

    shipping = 0.0
    for (i, j), frac in x.items():
        shipping += shipping_costs[i][j] * frac
    fixed = sum(fixed_costs[j] for j in open_plants)
    obj = shipping + fixed

    assignments = {}
    for i in range(n):
        inner = {}
        s = 0.0
        for j in range(m):
            v = x.get((i, j), 0.0)
            if v > 1e-12:
                inner[str(j)] = float(v)
                s += v
        if not inner:
            # defensive fill
            j0 = min(range(m), key=lambda j: shipping_costs[i][j])
            inner[str(j0)] = 1.0
        else:
            # normalize to sum exactly 1 for output robustness
            if abs(s - 1.0) > 1e-9 and s > 0:
                for k in list(inner.keys()):
                    inner[k] = float(inner[k] / s)
        assignments[str(i)] = inner

    return obj, open_plants, x, assignments


def build_and_solve(instance: dict, time_limit: int, logger: Optional[SolutionLogger]):
    n = instance["num_customers"]
    m = instance["num_plants"]

    demands = [float(v) for v in instance["customers"]["demands"]]
    capacities = [float(v) for v in instance["plants"]["capacities"]]
    fixed_costs = [float(v) for v in instance["plants"]["fixed_costs"]]
    shipping_costs = [[float(c) for c in row] for row in instance["shipping_costs"]]

    # Heuristic incumbent
    heur_obj, heur_open, heur_x, heur_assignments = greedy_feasible_solution(
        demands, capacities, fixed_costs, shipping_costs
    )
    best_logged = math.inf
    if logger:
        logger.log(float(heur_obj))
        best_logged = heur_obj

    model = gp.Model("splittable_cflp")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = int(time_limit)
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    # Variables
    y = model.addVars(m, vtype=GRB.BINARY, name="open")
    x = model.addVars(n, m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="frac")

    # Demand fulfillment: each customer fully served
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(m)) == 1.0, name=f"demand_{i}")

    # Capacity and activation
    for j in range(m):
        model.addConstr(
            gp.quicksum(demands[i] * x[i, j] for i in range(n)) <= capacities[j] * y[j],
            name=f"cap_{j}",
        )

    # Objective
    ship_term = gp.quicksum(shipping_costs[i][j] * x[i, j] for i in range(n) for j in range(m))
    fixed_term = gp.quicksum(fixed_costs[j] * y[j] for j in range(m))
    model.setObjective(ship_term + fixed_term, GRB.MINIMIZE)

    # Warm start from heuristic
    heur_open_set = set(heur_open)
    for j in range(m):
        y[j].Start = 1.0 if j in heur_open_set else 0.0
    for (i, j), val in heur_x.items():
        if val > 1e-12:
            x[i, j].Start = val

    model._logger = logger
    model._best_logged = best_logged

    def cb(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < mdl._best_logged:
                mdl._best_logged = obj
                if mdl._logger:
                    mdl._logger.log(float(obj))

    model.optimize(cb)

    # If no MIP solution found in time, return heuristic fallback
    if model.SolCount == 0:
        return {
            "objective_value": float(heur_obj),
            "open_plants": [int(j) for j in heur_open],
            "assignments": heur_assignments,
        }

    # Extract best found solution
    open_plants = [j for j in range(m) if y[j].X > 0.5]
    open_set = set(open_plants)

    assignments: Dict[str, Dict[str, float]] = {}
    for i in range(n):
        inner = {}
        total = 0.0
        for j in range(m):
            v = x[i, j].X
            if v > 1e-9:
                inner[str(j)] = float(v)
                total += v

        if not inner:
            # defensive (should not occur)
            if open_plants:
                j0 = min(open_plants, key=lambda jj: shipping_costs[i][jj])
            else:
                j0 = min(range(m), key=lambda jj: shipping_costs[i][jj])
            inner[str(j0)] = 1.0
            total = 1.0

        if abs(total - 1.0) > 1e-8 and total > 0:
            for k in list(inner.keys()):
                inner[k] = float(inner[k] / total)

        assignments[str(i)] = inner

    result = {
        "objective_value": float(model.ObjVal),
        "open_plants": [int(j) for j in open_plants],
        "assignments": assignments,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    instance = load_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = build_and_solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()