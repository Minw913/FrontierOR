import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-7


def period_costs(value, num_periods):
    """Accept the documented scalar holding costs and tolerate period arrays."""
    if isinstance(value, (list, tuple)):
        if len(value) != num_periods:
            raise ValueError("Holding-cost array has incorrect length")
        return [float(v) for v in value]
    return [float(value)] * num_periods


def clean_number(value):
    value = float(value)
    if abs(value) < EPS:
        return 0.0
    return max(0.0, value)


def lot_sizing_plan(demand, setup_cost, holding_cost):
    """
    Solve a single-facility uncapacitated dynamic lot-sizing problem by
    shortest-path dynamic programming.
    """
    T = len(demand)
    demand = [float(v) for v in demand]
    setup_cost = [float(v) for v in setup_cost]
    holding_cost = period_costs(holding_cost, T)

    if T == 0:
        return [], [], []

    if sum(demand) <= EPS:
        return [0.0] * T, [0.0] * T, [0] * T

    # arc_cost[t][q] is the cost of ordering at t and covering demand
    # in periods t,...,q-1. q may equal T.
    arc_cost = [[math.inf] * (T + 1) for _ in range(T)]
    for t in range(T):
        cost = setup_cost[t]
        accumulated_holding = 0.0
        arc_cost[t][t] = setup_cost[t]
        for q in range(t + 1, T + 1):
            k = q - 1
            if k > t:
                accumulated_holding += holding_cost[k - 1]
            cost += demand[k] * accumulated_holding
            arc_cost[t][q] = cost

    prefix = [0.0]
    for value in demand:
        prefix.append(prefix[-1] + value)

    # dp[t] is the cost of all completed replenishment cycles before an
    # order placed in period t.
    dp = [math.inf] * T
    predecessor = [-1] * T

    for t in range(T):
        if prefix[t] <= EPS:
            dp[t] = 0.0

    for t in range(T):
        if not math.isfinite(dp[t]):
            continue
        for q in range(t + 1, T):
            candidate = dp[t] + arc_cost[t][q]
            if candidate + 1e-12 < dp[q]:
                dp[q] = candidate
                predecessor[q] = t

    last = min(
        (t for t in range(T) if math.isfinite(dp[t])),
        key=lambda t: dp[t] + arc_cost[t][T],
    )

    order_periods = []
    current = last
    while current >= 0:
        order_periods.append(current)
        current = predecessor[current]
    order_periods.reverse()

    x = [0.0] * T
    for idx, t in enumerate(order_periods):
        q = order_periods[idx + 1] if idx + 1 < len(order_periods) else T
        quantity = sum(demand[t:q])
        if quantity > EPS:
            x[t] = quantity

    inventory = [0.0] * T
    stock = 0.0
    for t in range(T):
        stock += x[t] - demand[t]
        if abs(stock) < EPS:
            stock = 0.0
        inventory[t] = max(0.0, stock)

    y = [1 if x[t] > EPS else 0 for t in range(T)]
    return x, inventory, y


def compute_objective(instance, solution):
    W = instance["num_warehouses"]
    R = instance["num_retailers"]
    T = instance["num_periods"]

    fixed = instance["fixed_costs"]
    holding = instance["holding_costs"]

    hp = period_costs(holding["plant"], T)
    hw = period_costs(holding["warehouses"], T)
    hr = [period_costs(holding["retailers"][r], T) for r in range(R)]

    objective = 0.0

    for t in range(T):
        objective += float(fixed["plant"][t]) * solution["y_plant"][t]
        objective += hp[t] * solution["s_plant"][t]

    for w in range(W):
        for t in range(T):
            objective += (
                float(fixed["warehouses"][w][t])
                * solution["y_warehouses"][w][t]
            )
            objective += hw[t] * solution["s_warehouses"][w][t]

    for r in range(R):
        for t in range(T):
            objective += (
                float(fixed["retailers"][r][t])
                * solution["y_retailers"][r][t]
            )
            objective += hr[r][t] * solution["s_retailers"][r][t]

    return float(objective)


def finalize_solution(instance, solution):
    """Clean numerical noise and make setup indicators match positive inflows."""
    T = instance["num_periods"]
    W = instance["num_warehouses"]
    R = instance["num_retailers"]

    for t in range(T):
        solution["x_plant"][t] = clean_number(solution["x_plant"][t])
        solution["s_plant"][t] = clean_number(solution["s_plant"][t])
        solution["y_plant"][t] = 1 if solution["x_plant"][t] > EPS else 0

    for w in range(W):
        for t in range(T):
            solution["x_warehouses"][w][t] = clean_number(
                solution["x_warehouses"][w][t]
            )
            solution["s_warehouses"][w][t] = clean_number(
                solution["s_warehouses"][w][t]
            )
            solution["y_warehouses"][w][t] = (
                1 if solution["x_warehouses"][w][t] > EPS else 0
            )

    for r in range(R):
        for t in range(T):
            solution["x_retailers"][r][t] = clean_number(
                solution["x_retailers"][r][t]
            )
            solution["s_retailers"][r][t] = clean_number(
                solution["s_retailers"][r][t]
            )
            solution["y_retailers"][r][t] = (
                1 if solution["x_retailers"][r][t] > EPS else 0
            )

    solution["objective_value"] = compute_objective(instance, solution)
    return solution


def make_jit_solution(instance):
    """Feasible just-in-time policy used as an immediate fallback."""
    W = instance["num_warehouses"]
    R = instance["num_retailers"]
    T = instance["num_periods"]
    assignment = instance["retailer_warehouse_assignment"]
    demands = instance["demands"]

    xr = [[float(demands[r][t]) for t in range(T)] for r in range(R)]
    xw = [[0.0] * T for _ in range(W)]
    for r in range(R):
        w = assignment[r]
        for t in range(T):
            xw[w][t] += xr[r][t]

    xp = [sum(xw[w][t] for w in range(W)) for t in range(T)]

    solution = {
        "objective_value": 0.0,
        "x_plant": xp,
        "s_plant": [0.0] * T,
        "y_plant": [1 if v > EPS else 0 for v in xp],
        "x_warehouses": xw,
        "s_warehouses": [[0.0] * T for _ in range(W)],
        "y_warehouses": [
            [1 if xw[w][t] > EPS else 0 for t in range(T)]
            for w in range(W)
        ],
        "x_retailers": xr,
        "s_retailers": [[0.0] * T for _ in range(R)],
        "y_retailers": [
            [1 if xr[r][t] > EPS else 0 for t in range(T)]
            for r in range(R)
        ],
    }
    return finalize_solution(instance, solution)


def make_bottom_up_dp_solution(instance):
    """
    Optimize retailer replenishments independently, then use those shipments
    as warehouse demand, and finally warehouse receipts as plant demand.
    """
    W = instance["num_warehouses"]
    R = instance["num_retailers"]
    T = instance["num_periods"]
    assignment = instance["retailer_warehouse_assignment"]
    demands = instance["demands"]
    fixed = instance["fixed_costs"]
    holding = instance["holding_costs"]

    xr = [[0.0] * T for _ in range(R)]
    sr = [[0.0] * T for _ in range(R)]
    yr = [[0] * T for _ in range(R)]

    for r in range(R):
        xr[r], sr[r], yr[r] = lot_sizing_plan(
            demands[r],
            fixed["retailers"][r],
            holding["retailers"][r],
        )

    warehouse_outflow = [[0.0] * T for _ in range(W)]
    for r in range(R):
        w = assignment[r]
        for t in range(T):
            warehouse_outflow[w][t] += xr[r][t]

    xw = [[0.0] * T for _ in range(W)]
    sw = [[0.0] * T for _ in range(W)]
    yw = [[0] * T for _ in range(W)]

    for w in range(W):
        xw[w], sw[w], yw[w] = lot_sizing_plan(
            warehouse_outflow[w],
            fixed["warehouses"][w],
            holding["warehouses"],
        )

    plant_outflow = [sum(xw[w][t] for w in range(W)) for t in range(T)]
    xp, sp, yp = lot_sizing_plan(
        plant_outflow,
        fixed["plant"],
        holding["plant"],
    )

    solution = {
        "objective_value": 0.0,
        "x_plant": xp,
        "s_plant": sp,
        "y_plant": yp,
        "x_warehouses": xw,
        "s_warehouses": sw,
        "y_warehouses": yw,
        "x_retailers": xr,
        "s_retailers": sr,
        "y_retailers": yr,
    }
    return finalize_solution(instance, solution)


def solution_to_flat_values(solution, W, R, T):
    values = []
    values.extend(solution["x_plant"])
    values.extend(solution["s_plant"])
    values.extend(solution["y_plant"])

    for w in range(W):
        values.extend(solution["x_warehouses"][w])
    for w in range(W):
        values.extend(solution["s_warehouses"][w])
    for w in range(W):
        values.extend(solution["y_warehouses"][w])

    for r in range(R):
        values.extend(solution["x_retailers"][r])
    for r in range(R):
        values.extend(solution["s_retailers"][r])
    for r in range(R):
        values.extend(solution["y_retailers"][r])

    return values


def flat_values_to_solution(instance, values):
    W = instance["num_warehouses"]
    R = instance["num_retailers"]
    T = instance["num_periods"]
    position = 0

    def take_vector():
        nonlocal position
        result = [float(v) for v in values[position:position + T]]
        position += T
        return result

    xp = take_vector()
    sp = take_vector()
    _yp = take_vector()

    xw = [take_vector() for _ in range(W)]
    sw = [take_vector() for _ in range(W)]
    _yw = [take_vector() for _ in range(W)]

    xr = [take_vector() for _ in range(R)]
    sr = [take_vector() for _ in range(R)]
    _yr = [take_vector() for _ in range(R)]

    solution = {
        "objective_value": 0.0,
        "x_plant": xp,
        "s_plant": sp,
        "y_plant": [0] * T,
        "x_warehouses": xw,
        "s_warehouses": sw,
        "y_warehouses": [[0] * T for _ in range(W)],
        "x_retailers": xr,
        "s_retailers": sr,
        "y_retailers": [[0] * T for _ in range(R)],
    }
    return finalize_solution(instance, solution)


def build_model(instance):
    W = instance["num_warehouses"]
    R = instance["num_retailers"]
    T = instance["num_periods"]
    assignment = instance["retailer_warehouse_assignment"]
    demands = instance["demands"]
    fixed = instance["fixed_costs"]
    holding = instance["holding_costs"]

    hp = period_costs(holding["plant"], T)
    hw = period_costs(holding["warehouses"], T)
    hr = [period_costs(holding["retailers"][r], T) for r in range(R)]

    warehouse_retailers = [[] for _ in range(W)]
    for r, w in enumerate(assignment):
        warehouse_retailers[w].append(r)

    plant_demand = [
        sum(float(demands[r][t]) for r in range(R))
        for t in range(T)
    ]
    warehouse_demand = [
        [
            sum(float(demands[r][t]) for r in warehouse_retailers[w])
            for t in range(T)
        ]
        for w in range(W)
    ]

    def remaining(values):
        result = [0.0] * T
        total = 0.0
        for t in range(T - 1, -1, -1):
            total += values[t]
            result[t] = total
        return result

    plant_remaining = remaining(plant_demand)
    warehouse_remaining = [remaining(warehouse_demand[w]) for w in range(W)]
    retailer_remaining = [
        remaining([float(v) for v in demands[r]]) for r in range(R)
    ]

    model = gp.Model("three_echelon_lot_sizing")

    xp = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="x_plant")
    sp = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="s_plant")
    yp = model.addVars(T, vtype=GRB.BINARY, name="y_plant")

    xw = model.addVars(W, T, lb=0.0, vtype=GRB.CONTINUOUS, name="x_warehouse")
    sw = model.addVars(W, T, lb=0.0, vtype=GRB.CONTINUOUS, name="s_warehouse")
    yw = model.addVars(W, T, vtype=GRB.BINARY, name="y_warehouse")

    xr = model.addVars(R, T, lb=0.0, vtype=GRB.CONTINUOUS, name="x_retailer")
    sr = model.addVars(R, T, lb=0.0, vtype=GRB.CONTINUOUS, name="s_retailer")
    yr = model.addVars(R, T, vtype=GRB.BINARY, name="y_retailer")

    for t in range(T):
        previous = 0.0 if t == 0 else sp[t - 1]
        model.addConstr(
            previous + xp[t]
            == gp.quicksum(xw[w, t] for w in range(W)) + sp[t],
            name=f"plant_balance_{t}",
        )
        model.addConstr(
            xp[t] <= plant_remaining[t] * yp[t],
            name=f"plant_setup_{t}",
        )

    for w in range(W):
        served = warehouse_retailers[w]
        for t in range(T):
            previous = 0.0 if t == 0 else sw[w, t - 1]
            model.addConstr(
                previous + xw[w, t]
                == gp.quicksum(xr[r, t] for r in served) + sw[w, t],
                name=f"warehouse_balance_{w}_{t}",
            )
            model.addConstr(
                xw[w, t] <= warehouse_remaining[w][t] * yw[w, t],
                name=f"warehouse_setup_{w}_{t}",
            )

    for r in range(R):
        for t in range(T):
            previous = 0.0 if t == 0 else sr[r, t - 1]
            model.addConstr(
                previous + xr[r, t]
                == float(demands[r][t]) + sr[r, t],
                name=f"retailer_balance_{r}_{t}",
            )
            model.addConstr(
                xr[r, t] <= retailer_remaining[r][t] * yr[r, t],
                name=f"retailer_setup_{r}_{t}",
            )

    # An optimal solution never needs terminal surplus under nonnegative costs.
    if T > 0:
        model.addConstr(sp[T - 1] == 0.0, name="plant_terminal_inventory")
        for w in range(W):
            model.addConstr(
                sw[w, T - 1] == 0.0,
                name=f"warehouse_terminal_inventory_{w}",
            )
        for r in range(R):
            model.addConstr(
                sr[r, T - 1] == 0.0,
                name=f"retailer_terminal_inventory_{r}",
            )

    objective = gp.LinExpr()
    for t in range(T):
        objective += float(fixed["plant"][t]) * yp[t]
        objective += hp[t] * sp[t]

    for w in range(W):
        for t in range(T):
            objective += float(fixed["warehouses"][w][t]) * yw[w, t]
            objective += hw[t] * sw[w, t]

    for r in range(R):
        for t in range(T):
            objective += float(fixed["retailers"][r][t]) * yr[r, t]
            objective += hr[r][t] * sr[r, t]

    model.setObjective(objective, GRB.MINIMIZE)

    all_vars = []
    all_vars.extend(xp[t] for t in range(T))
    all_vars.extend(sp[t] for t in range(T))
    all_vars.extend(yp[t] for t in range(T))

    all_vars.extend(xw[w, t] for w in range(W) for t in range(T))
    all_vars.extend(sw[w, t] for w in range(W) for t in range(T))
    all_vars.extend(yw[w, t] for w in range(W) for t in range(T))

    all_vars.extend(xr[r, t] for r in range(R) for t in range(T))
    all_vars.extend(sr[r, t] for r in range(R) for t in range(T))
    all_vars.extend(yr[r, t] for r in range(R) for t in range(T))

    return model, all_vars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    W = int(instance["num_warehouses"])
    R = int(instance["num_retailers"])
    T = int(instance["num_periods"])

    best_solution = make_jit_solution(instance)
    best_objective = best_solution["objective_value"]
    if logger:
        logger.log_solution(best_objective, best_solution)

    dp_solution = make_bottom_up_dp_solution(instance)
    if dp_solution["objective_value"] < best_objective - 1e-9:
        best_solution = dp_solution
        best_objective = dp_solution["objective_value"]
        if logger:
            logger.log_solution(best_objective, best_solution)

    try:
        elapsed = time.monotonic() - start_time
        remaining_time = float(args.time_limit) - elapsed

        if remaining_time > 0.01 and T > 0:
            model, all_vars = build_model(instance)

            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = max(0.01, remaining_time)

            start_values = solution_to_flat_values(best_solution, W, R, T)
            for variable, value in zip(all_vars, start_values):
                variable.Start = value

            callback_state = {"best": best_objective}

            def incumbent_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL or logger is None:
                    return
                try:
                    raw_values = cb_model.cbGetSolution(all_vars)
                    candidate = flat_values_to_solution(instance, raw_values)
                    candidate_objective = candidate["objective_value"]
                    if candidate_objective < callback_state["best"] - 1e-9:
                        callback_state["best"] = candidate_objective
                        logger.log_solution(candidate_objective, candidate)
                except Exception:
                    # Logging must not interrupt optimization.
                    pass

            if logger:
                model.optimize(incumbent_callback)
            else:
                model.optimize()

            if model.SolCount > 0:
                raw_values = [variable.X for variable in all_vars]
                candidate = flat_values_to_solution(instance, raw_values)
                candidate_objective = candidate["objective_value"]

                if candidate_objective < best_objective - 1e-7:
                    best_solution = candidate
                    best_objective = candidate_objective
                    if logger and candidate_objective < callback_state["best"] - 1e-9:
                        logger.log_solution(candidate_objective, candidate)

    except gp.GurobiError:
        # The precomputed heuristic remains a valid fallback.
        pass

    best_solution = finalize_solution(instance, best_solution)

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(best_solution, file, indent=2)


if __name__ == "__main__":
    main()