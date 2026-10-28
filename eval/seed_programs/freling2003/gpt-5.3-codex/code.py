import argparse
import json
import time
from typing import List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    p = data["parameters"]
    F = int(p["num_facilities"])
    C = int(p["num_customers"])
    T = int(p["num_periods"])

    seasonal = data["seasonal_factors"]
    demands = data["demands"]
    holding = data["holding_costs"]
    capacities = data["capacities"]
    transportation = data["transportation_costs"]

    return F, C, T, seasonal, demands, holding, capacities, transportation


def aggregate_transport_costs(F, C, T, transportation):
    # Per problem statement objective: sum over periods cost * assignment
    cost_fc = [[0.0] * C for _ in range(F)]
    for f in range(F):
        for c in range(C):
            s = 0.0
            for t in range(T):
                s += float(transportation[f][c][t])
            cost_fc[f][c] = s
    return cost_fc


def compute_capacity_upper_bounds(F, T, capacities, seasonal):
    # Sufficient/necessary upper bound on total assigned demand q_f for feasibility:
    # cumulative_demand(t) <= cumulative_capacity(t) for all t.
    S = 0.0
    upper = [float("inf")] * F
    for t in range(T):
        S += seasonal[t]
        if S <= 1e-12:
            continue
        for f in range(F):
            cum_cap = sum(capacities[f][k] for k in range(t + 1))
            upper[f] = min(upper[f], cum_cap / S)
    return upper


def greedy_assignment(F, C, demands, cost_fc, upper_bounds):
    customers = list(range(C))
    customers.sort(key=lambda c: demands[c], reverse=True)

    rem = upper_bounds[:]
    assign = [-1] * C

    for c in customers:
        d = demands[c]
        feasible_fs = [f for f in range(F) if rem[f] + 1e-9 >= d]
        if feasible_fs:
            best_f = min(feasible_fs, key=lambda f: cost_fc[f][c])
        else:
            # fallback: least transport cost (may violate bound)
            best_f = min(range(F), key=lambda f: cost_fc[f][c])
        assign[c] = best_f
        rem[best_f] -= d
    return assign


def build_and_solve_main_mip(
    F, C, T, seasonal, demands, holding, capacities, cost_fc, time_limit, logger, start_assign=None
):
    model = gp.Model("facility_assignment_inventory")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.0, float(time_limit))
    model.Params.MIPFocus = 1

    x = model.addVars(F, C, vtype=GRB.BINARY, name="x")
    q = model.addVars(F, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    inv = model.addVars(F, T, lb=0.0, vtype=GRB.CONTINUOUS, name="inv")

    # Each customer assigned to exactly one facility
    for c in range(C):
        model.addConstr(gp.quicksum(x[f, c] for f in range(F)) == 1, name=f"assign_{c}")

    # Total assigned demand per facility
    for f in range(F):
        model.addConstr(q[f] == gp.quicksum(float(demands[c]) * x[f, c] for c in range(C)), name=f"qdef_{f}")

    # Capacity + inventory flow constraints
    for f in range(F):
        for t in range(T):
            if t == 0:
                model.addConstr(
                    float(seasonal[t]) * q[f] + inv[f, t] <= float(capacities[f][t]),
                    name=f"cap_{f}_{t}",
                )
            else:
                model.addConstr(
                    float(seasonal[t]) * q[f] + inv[f, t] - inv[f, t - 1] <= float(capacities[f][t]),
                    name=f"cap_{f}_{t}",
                )

    transport_obj = gp.quicksum(float(cost_fc[f][c]) * x[f, c] for f in range(F) for c in range(C))
    hold_obj = gp.quicksum(float(holding[f][t]) * inv[f, t] for f in range(F) for t in range(T))
    model.setObjective(transport_obj + hold_obj, GRB.MINIMIZE)

    # Warm start
    if start_assign is not None and len(start_assign) == C:
        for c in range(C):
            fs = start_assign[c]
            if 0 <= fs < F:
                x[fs, c].Start = 1.0
                for f in range(F):
                    if f != fs:
                        x[f, c].Start = 0.0

    best_logged = {"val": float("inf")}

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged["val"]:
                best_logged["val"] = obj
                logger.log(float(obj))

    model.optimize(cb if logger is not None else None)
    return model, x, inv


def solve_inventory_lp_fixed_assignment(
    F, C, T, seasonal, demands, holding, capacities, cost_fc, assignment, time_limit
):
    # Continuous LP to get best inventory for fixed assignment
    q_val = [0.0] * F
    for c in range(C):
        q_val[assignment[c]] += demands[c]

    m = gp.Model("fixed_assignment_inventory")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(0.0, float(time_limit))

    inv = m.addVars(F, T, lb=0.0, vtype=GRB.CONTINUOUS, name="inv")
    for f in range(F):
        for t in range(T):
            rhs = float(capacities[f][t]) - float(seasonal[t]) * q_val[f]
            if t == 0:
                m.addConstr(inv[f, t] <= rhs)
            else:
                m.addConstr(inv[f, t] - inv[f, t - 1] <= rhs)

    m.setObjective(gp.quicksum(float(holding[f][t]) * inv[f, t] for f in range(F) for t in range(T)), GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        return None, None

    inv_sol = [[0.0] * T for _ in range(F)]
    for f in range(F):
        for t in range(T):
            inv_sol[f][t] = float(inv[f, t].X)

    transport = 0.0
    for c in range(C):
        transport += float(cost_fc[assignment[c]][c])
    hold_cost = sum(float(holding[f][t]) * inv_sol[f][t] for f in range(F) for t in range(T))
    return transport + hold_cost, inv_sol


def fallback_assignment_mip(F, C, demands, cost_fc, upper_bounds, time_limit):
    m = gp.Model("fallback_assign")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(0.0, float(time_limit))

    y = m.addVars(F, C, vtype=GRB.BINARY, name="y")
    for c in range(C):
        m.addConstr(gp.quicksum(y[f, c] for f in range(F)) == 1)
    for f in range(F):
        m.addConstr(gp.quicksum(float(demands[c]) * y[f, c] for c in range(C)) <= float(upper_bounds[f]) + 1e-9)

    m.setObjective(gp.quicksum(float(cost_fc[f][c]) * y[f, c] for f in range(F) for c in range(C)), GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        return None
    assign = [-1] * C
    for c in range(C):
        for f in range(F):
            if y[f, c].X > 0.5:
                assign[c] = f
                break
        if assign[c] < 0:
            assign[c] = 0
    return assign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    F, C, T, seasonal, demands, holding, capacities, transportation = load_instance(args.instance_path)
    cost_fc = aggregate_transport_costs(F, C, T, transportation)

    upper_bounds = compute_capacity_upper_bounds(F, T, capacities, seasonal)
    greedy_start = greedy_assignment(F, C, demands, cost_fc, upper_bounds)

    # Main MIP
    elapsed = time.time() - t0
    remaining = max(0.0, args.time_limit - elapsed)
    model, x, inv = build_and_solve_main_mip(
        F, C, T, seasonal, demands, holding, capacities, cost_fc, remaining, logger, start_assign=greedy_start
    )

    final_assignment = None
    final_inventory = None
    final_obj = None

    if model.SolCount > 0:
        final_obj = float(model.ObjVal)
        final_assignment = {}
        for c in range(C):
            chosen = 0
            bestv = -1.0
            for f in range(F):
                v = x[f, c].X
                if v > bestv:
                    bestv = v
                    chosen = f
            final_assignment[str(c)] = int(chosen)

        final_inventory = {}
        for f in range(F):
            final_inventory[str(f)] = [float(inv[f, t].X) for t in range(T)]
    else:
        # Fallback within remaining time budget
        elapsed = time.time() - t0
        remaining = max(0.0, args.time_limit - elapsed)

        assign = None
        if remaining > 0.1:
            assign = fallback_assignment_mip(F, C, demands, cost_fc, upper_bounds, remaining * 0.6)
        if assign is None:
            assign = greedy_start

        elapsed = time.time() - t0
        remaining = max(0.0, args.time_limit - elapsed)

        obj_lp, inv_lp = solve_inventory_lp_fixed_assignment(
            F, C, T, seasonal, demands, holding, capacities, cost_fc, assign, max(0.05, remaining)
        )

        if obj_lp is None:
            # Last resort: zero inventory projection
            inv_lp = [[0.0] * T for _ in range(F)]
            obj_lp = 0.0
            for c in range(C):
                obj_lp += float(cost_fc[assign[c]][c])

        final_obj = float(obj_lp)
        final_assignment = {str(c): int(assign[c]) for c in range(C)}
        final_inventory = {str(f): [float(inv_lp[f][t]) for t in range(T)] for f in range(F)}

        if logger:
            logger.log(final_obj)

    solution = {
        "objective_value": float(final_obj),
        "assignment": final_assignment,
        "inventory": final_inventory,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()