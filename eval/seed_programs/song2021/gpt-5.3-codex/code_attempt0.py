import argparse
import json
import math
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_feasible_intervals(order: dict, H: int) -> List[Tuple[int, int, int]]:
    """
    Returns list of feasible intervals (s, f, tardiness) for one order.
    Periods are 1-based and inclusive.
    """
    r = int(order["release_date_r"])
    d = int(order["due_date_d"])
    LB = float(order["intensity_lower_bound_LB"])
    UB = float(order["intensity_upper_bound_UB"])

    l_given = int(order["lower_exec_interval_length_l"])
    lbar_given = int(order["upper_exec_interval_length_l_bar"])

    # Length bounds implied by intensity bounds
    l_from_ub = math.ceil(1.0 / UB)
    lbar_from_lb = math.floor(1.0 / LB)

    l_min = max(1, l_given, l_from_ub)
    l_max = min(lbar_given, lbar_from_lb, H - r + 1)

    intervals = []
    if l_min > l_max:
        return intervals

    for s in range(r, H + 1):
        max_len_here = min(l_max, H - s + 1)
        if l_min > max_len_here:
            continue
        for L in range(l_min, max_len_here + 1):
            f = s + L - 1
            tard = max(0, f - d)
            intervals.append((s, f, tard))
    return intervals


def solve_instance(instance: dict, time_limit: int, logger: SolutionLogger = None) -> dict:
    H = int(instance["planning_horizon_H"])
    n = int(instance["num_orders_n"])
    sigma = float(instance["unit_cost_nonregular_capacity_sigma"])
    capacities = instance["capacities"]
    orders = instance["orders"]

    # Basic validation
    if len(capacities) != H:
        raise ValueError("capacities length must equal planning_horizon_H")
    if len(orders) != n:
        raise ValueError("orders length must equal num_orders_n")

    # Precompute feasible intervals for each order
    intervals_by_i: Dict[int, List[Tuple[int, int, int]]] = {}
    for i, od in enumerate(orders):
        ivals = build_feasible_intervals(od, H)
        intervals_by_i[i] = ivals

    # Early infeasibility check
    for i in range(n):
        if not intervals_by_i[i]:
            # No feasible execution interval for this order
            return {
                "objective_value": 1e30,
                "orders_solution": [],
                "nonregular_capacity_by_period": {},
            }

    model = gp.Model("order_scheduling")
    model.Params.TimeLimit = int(time_limit)
    model.Params.Threads = 1

    # Decision variables
    y = {}  # interval selection
    x = {}  # intensity per period
    o = {}  # overtime per period

    # y vars
    for i in range(n):
        for (s, f, tard) in intervals_by_i[i]:
            y[(i, s, f)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{s}_{f}")

    # x vars
    for i in range(n):
        for t in range(1, H + 1):
            x[(i, t)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"x_{i}_{t}")

    # overtime vars
    for t in range(1, H + 1):
        o[t] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"ot_{t}")

    model.update()

    # One interval per order
    for i in range(n):
        model.addConstr(
            gp.quicksum(y[(i, s, f)] for (s, f, tard) in intervals_by_i[i]) == 1,
            name=f"one_interval_{i}",
        )

    # Intensity-window coupling + completion
    for i, od in enumerate(orders):
        LB = float(od["intensity_lower_bound_LB"])
        UB = float(od["intensity_upper_bound_UB"])

        # Precompute coverage lists by period for efficiency
        covers_t = {t: [] for t in range(1, H + 1)}
        for (s, f, tard) in intervals_by_i[i]:
            for t in range(s, f + 1):
                covers_t[t].append((s, f))

        for t in range(1, H + 1):
            if covers_t[t]:
                cover_expr = gp.quicksum(y[(i, s, f)] for (s, f) in covers_t[t])
            else:
                cover_expr = 0.0
            model.addConstr(x[(i, t)] <= UB * cover_expr, name=f"xub_{i}_{t}")
            model.addConstr(x[(i, t)] >= LB * cover_expr, name=f"xlb_{i}_{t}")

        model.addConstr(
            gp.quicksum(x[(i, t)] for t in range(1, H + 1)) == 1.0,
            name=f"complete_{i}",
        )

    # Capacity with overtime
    for t in range(1, H + 1):
        model.addConstr(
            gp.quicksum(float(orders[i]["work_content_p"]) * x[(i, t)] for i in range(n))
            <= float(capacities[t - 1]) + o[t],
            name=f"cap_{t}",
        )

    # Objective: weighted tardiness + overtime cost
    tardiness_cost_expr = gp.quicksum(
        float(orders[i]["tardiness_cost_w"]) * tard * y[(i, s, f)]
        for i in range(n)
        for (s, f, tard) in intervals_by_i[i]
    )
    overtime_cost_expr = sigma * gp.quicksum(o[t] for t in range(1, H + 1))
    model.setObjective(tardiness_cost_expr + overtime_cost_expr, GRB.MINIMIZE)

    # Incumbent logging callback
    best_logged = {"obj": float("inf")}

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged["obj"]:
                best_logged["obj"] = obj
                logger.log(float(obj))

    model.optimize(cb)

    if model.SolCount == 0:
        return {
            "objective_value": 1e30,
            "orders_solution": [],
            "nonregular_capacity_by_period": {},
        }

    # Extract solution
    orders_solution = []
    for i, od in enumerate(orders):
        chosen = None
        best_val = -1.0
        for (s, f, tard) in intervals_by_i[i]:
            val = y[(i, s, f)].X
            if val > best_val:
                best_val = val
                chosen = (s, f)
        if chosen is None:
            # fallback (should not happen if feasible)
            chosen = (int(od["release_date_r"]), int(od["release_date_r"]))

        s, f = chosen
        intensities = {}
        for t in range(s, f + 1):
            intensities[str(t)] = float(x[(i, t)].X)

        orders_solution.append(
            {
                "order_id": int(od["order_id"]),
                "chosen_interval": [int(s), int(f)],
                "intensities": intensities,
            }
        )

    nonregular = {}
    for t in range(1, H + 1):
        val = float(o[t].X)
        if val > 1e-6:
            nonregular[str(t)] = val

    solution = {
        "objective_value": float(model.ObjVal),
        "orders_solution": orders_solution,
        "nonregular_capacity_by_period": nonregular,
    }
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve_instance(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()