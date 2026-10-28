import argparse
import json
import math
import os
from typing import List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_greedy_open_plan(
    opening_costs: List[float],
    capacities: List[float],
    transport_costs: List[List[float]],
    demand_mean: List[float],
    max_total_demand: float,
) -> List[int]:
    F = len(opening_costs)
    C = len(demand_mean)
    total_mean = sum(demand_mean) if sum(demand_mean) > 0 else 1.0

    scores = []
    for f in range(F):
        avg_tc = sum(transport_costs[f][c] * demand_mean[c] for c in range(C)) / total_mean
        score = opening_costs[f] / max(capacities[f], 1e-9) + 0.01 * avg_tc
        scores.append((score, f))

    scores.sort()
    x = [0] * F
    cap_sum = 0.0
    for _, f in scores:
        if cap_sum + 1e-9 >= max_total_demand:
            break
        x[f] = 1
        cap_sum += capacities[f]

    # If still short, open everything needed
    if cap_sum + 1e-9 < max_total_demand:
        for f in range(F):
            if x[f] == 0:
                x[f] = 1
                cap_sum += capacities[f]
                if cap_sum + 1e-9 >= max_total_demand:
                    break
    return x


def greedy_shipping(
    x_open: List[int],
    capacities: List[float],
    demands: List[float],
    sorted_fac_by_customer: List[List[int]],
) -> Optional[List[List[float]]]:
    F = len(x_open)
    C = len(demands)

    remaining = [capacities[f] * x_open[f] for f in range(F)]
    alloc = [[0.0 for _ in range(C)] for _ in range(F)]

    for c in range(C):
        need = demands[c]
        for f in sorted_fac_by_customer[c]:
            if need <= 1e-9:
                break
            avail = remaining[f]
            if avail <= 1e-12:
                continue
            q = min(avail, need)
            alloc[f][c] += q
            remaining[f] -= q
            need -= q

        if need > 1e-6:
            return None  # infeasible for this open set
    return alloc


def heuristic_solution(
    opening_costs: List[float],
    capacities: List[float],
    transport_costs: List[List[float]],
    scenarios_demands: List[List[float]],
    probabilities: List[float],
    max_total_demand: float,
    demand_mean: List[float],
) -> Tuple[Optional[float], Optional[List[int]], Optional[List[List[List[float]]]]]:
    F = len(opening_costs)
    C = len(demand_mean)
    S = len(scenarios_demands)

    sorted_fac_by_customer = []
    for c in range(C):
        order = list(range(F))
        order.sort(key=lambda f: transport_costs[f][c])
        sorted_fac_by_customer.append(order)

    x = build_greedy_open_plan(opening_costs, capacities, transport_costs, demand_mean, max_total_demand)

    y_all = []
    for s in range(S):
        alloc = greedy_shipping(x, capacities, scenarios_demands[s], sorted_fac_by_customer)
        if alloc is None:
            return None, None, None
        y_all.append(alloc)

    fixed = sum(opening_costs[f] * x[f] for f in range(F))
    expected_transport = 0.0
    for s in range(S):
        scen_cost = 0.0
        for f in range(F):
            row = y_all[s][f]
            tc_row = transport_costs[f]
            scen_cost += sum(tc_row[c] * row[c] for c in range(C))
        expected_transport += probabilities[s] * scen_cost

    obj = fixed + expected_transport
    return obj, x, y_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    F = inst["num_facilities"]
    C = inst["num_customers"]
    S = inst["num_scenarios"]

    facilities = inst["facilities"]
    scenarios = inst["scenarios"]
    transport_costs = inst["transportation_costs"]

    opening_costs = [facilities[f]["opening_cost"] for f in range(F)]
    capacities = [facilities[f]["capacity"] for f in range(F)]
    probabilities = [scenarios[s]["probability"] for s in range(S)]
    scenarios_demands = [scenarios[s]["demands"] for s in range(S)]

    # demand mean for heuristic scoring
    demand_mean = [0.0] * C
    prob_sum = sum(probabilities)
    if prob_sum <= 0:
        prob_sum = 1.0
    for s in range(S):
        p = probabilities[s] / prob_sum
        for c in range(C):
            demand_mean[c] += p * scenarios_demands[s][c]

    max_total_demand = max(sum(scenarios_demands[s][c] for c in range(C)) for s in range(S))

    # Heuristic warm start
    heur_obj, heur_x, heur_y = heuristic_solution(
        opening_costs,
        capacities,
        transport_costs,
        scenarios_demands,
        probabilities,
        max_total_demand,
        demand_mean,
    )

    if logger and heur_obj is not None:
        logger.log(float(heur_obj))

    # Build MIP
    model = gp.Model("stochastic_facility_location")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, int(args.time_limit))

    x = model.addVars(F, vtype=GRB.BINARY, name="x")
    y = model.addVars(S, F, C, lb=0.0, vtype=GRB.CONTINUOUS, name="y")

    # Demand satisfaction for each scenario/customer
    for s in range(S):
        for c in range(C):
            model.addConstr(gp.quicksum(y[s, f, c] for f in range(F)) >= scenarios_demands[s][c],
                            name=f"demand_s{s}_c{c}")

    # Facility capacity in each scenario
    for s in range(S):
        for f in range(F):
            model.addConstr(gp.quicksum(y[s, f, c] for c in range(C)) <= capacities[f] * x[f],
                            name=f"cap_s{s}_f{f}")

    # Strengthening feasibility cut
    model.addConstr(gp.quicksum(capacities[f] * x[f] for f in range(F)) >= max_total_demand,
                    name="total_capacity_cover")

    obj = gp.quicksum(opening_costs[f] * x[f] for f in range(F))
    obj += gp.quicksum(
        probabilities[s] * transport_costs[f][c] * y[s, f, c]
        for s in range(S) for f in range(F) for c in range(C)
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Warm start
    if heur_x is not None and heur_y is not None:
        for f in range(F):
            x[f].Start = heur_x[f]
        for s in range(S):
            for f in range(F):
                for c in range(C):
                    y[s, f, c].Start = heur_y[s][f][c]

    # Incumbent logging callback
    model._logger = logger
    model._best_logged = float(heur_obj) if heur_obj is not None else float("inf")

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val + 1e-9 < m._best_logged:
                if m._logger:
                    m._logger.log(float(obj_val))
                m._best_logged = float(obj_val)

    model.optimize(cb)

    use_model_sol = model.SolCount > 0

    if use_model_sol:
        objective_value = float(model.ObjVal)
        x_sol = [int(round(x[f].X)) for f in range(F)]
        y_sol = [[[0.0 for _ in range(C)] for _ in range(F)] for _ in range(S)]
        for s in range(S):
            for f in range(F):
                for c in range(C):
                    val = y[s, f, c].X
                    if abs(val) < 1e-10:
                        val = 0.0
                    y_sol[s][f][c] = float(max(0.0, val))
    elif heur_obj is not None:
        objective_value = float(heur_obj)
        x_sol = [int(v) for v in heur_x]
        y_sol = heur_y
    else:
        # Last-resort fallback (should rarely happen)
        objective_value = 1e100
        x_sol = [0 for _ in range(F)]
        y_sol = [[[0.0 for _ in range(C)] for _ in range(F)] for _ in range(S)]

    open_facilities = [f for f in range(F) if x_sol[f] == 1]

    x_out = {str(f): int(x_sol[f]) for f in range(F)}
    y_out = {}
    for s in range(S):
        fs = {}
        for f in range(F):
            cs = {}
            for c in range(C):
                cs[str(c)] = float(y_sol[s][f][c])
            fs[str(f)] = cs
        y_out[str(s)] = fs

    solution = {
        "objective_value": float(objective_value),
        "open_facilities": open_facilities,
        "x": x_out,
        "y": y_out,
    }

    os.makedirs(os.path.dirname(args.solution_path), exist_ok=True) if os.path.dirname(args.solution_path) else None
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()