import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None, type=str)
    return parser.parse_args()


def greedy_subset(order, capacities, required_capacity):
    selected = []
    total = 0.0
    for i in order:
        if capacities[i] <= 0:
            continue
        selected.append(i)
        total += capacities[i]
        if total + 1e-9 >= required_capacity:
            break
    return tuple(sorted(selected)) if total + 1e-9 >= required_capacity else None


def greedy_routing(selected, demands, capacities, facility_orders, costs,
                   return_flows=False):
    """
    Produces a feasible routing for a complete facility-customer network.
    Customers with larger demands are handled first, and each customer uses
    its cheapest selected facilities with remaining capacity.
    """
    remaining_capacity = [float(c) if i in selected else 0.0
                          for i, c in enumerate(capacities)]
    customer_order = sorted(
        range(len(demands)),
        key=lambda j: (-demands[j], j)
    )

    routing_cost = 0.0
    flows = {} if return_flows else None
    selected_set = set(selected)

    for j in customer_order:
        remaining_demand = float(demands[j])
        if remaining_demand <= 1e-12:
            continue

        for i in facility_orders[j]:
            if i not in selected_set or remaining_capacity[i] <= 1e-12:
                continue

            amount = min(remaining_demand, remaining_capacity[i])
            if amount > 0:
                routing_cost += costs[i][j] * amount
                if return_flows:
                    flows[(i, j)] = amount
                remaining_capacity[i] -= amount
                remaining_demand -= amount

            if remaining_demand <= 1e-9:
                break

        if remaining_demand > 1e-7:
            return math.inf, None

    return routing_cost, flows


def build_initial_solution(fixed_costs, capacities, routing_costs,
                           probabilities, scenario_demands):
    num_facilities = len(fixed_costs)
    num_customers = len(routing_costs[0]) if num_facilities else 0
    required_capacity = max(
        (sum(demand) for demand in scenario_demands),
        default=0
    )

    if sum(max(0, c) for c in capacities) + 1e-9 < required_capacity:
        raise ValueError("Instance is infeasible: insufficient total capacity.")

    facility_orders = [
        sorted(range(num_facilities),
               key=lambda i, j=j: (routing_costs[i][j], i))
        for j in range(num_customers)
    ]

    expected_demands = [
        sum(probabilities[s] * scenario_demands[s][j]
            for s in range(len(scenario_demands)))
        for j in range(num_customers)
    ]
    expected_total = sum(expected_demands)

    average_routing = []
    for i in range(num_facilities):
        if expected_total > 0:
            value = sum(expected_demands[j] * routing_costs[i][j]
                        for j in range(num_customers)) / expected_total
        else:
            value = 0.0
        average_routing.append(value)

    positive_capacity = lambda i: max(float(capacities[i]), 1e-9)

    candidate_orders = [
        sorted(range(num_facilities),
               key=lambda i: (fixed_costs[i] / positive_capacity(i),
                              fixed_costs[i], -capacities[i], i)),
        sorted(range(num_facilities),
               key=lambda i: (fixed_costs[i], -capacities[i], i)),
        sorted(range(num_facilities),
               key=lambda i: (fixed_costs[i] / positive_capacity(i)
                              + average_routing[i],
                              fixed_costs[i], i)),
        sorted(range(num_facilities),
               key=lambda i: (-capacities[i], fixed_costs[i], i)),
    ]

    candidates = {tuple(range(num_facilities))}
    if required_capacity <= 1e-9:
        candidates.add(tuple())

    for order in candidate_orders:
        candidate = greedy_subset(order, capacities, required_capacity)
        if candidate is not None:
            candidates.add(candidate)

    best_objective = math.inf
    best_selected = None

    for selected_tuple in candidates:
        selected = set(selected_tuple)
        fixed_part = sum(fixed_costs[i] for i in selected)
        expected_routing = 0.0
        feasible = True

        for s, demands in enumerate(scenario_demands):
            route_cost, _ = greedy_routing(
                selected, demands, capacities, facility_orders,
                routing_costs, return_flows=False
            )
            if not math.isfinite(route_cost):
                feasible = False
                break
            expected_routing += probabilities[s] * route_cost

        if feasible:
            objective = fixed_part + expected_routing
            if objective < best_objective:
                best_objective = objective
                best_selected = selected

    if best_selected is None:
        raise ValueError("Unable to construct a feasible initial solution.")

    scenario_flows = []
    expected_routing = 0.0
    for s, demands in enumerate(scenario_demands):
        route_cost, flows = greedy_routing(
            best_selected, demands, capacities, facility_orders,
            routing_costs, return_flows=True
        )
        scenario_flows.append(flows)
        expected_routing += probabilities[s] * route_cost

    y_values = [1 if i in best_selected else 0
                for i in range(num_facilities)]
    objective = (
        sum(fixed_costs[i] * y_values[i] for i in range(num_facilities))
        + expected_routing
    )

    return objective, y_values, scenario_flows


def incumbent_callback(model, where):
    if where != GRB.Callback.MIPSOL:
        return

    try:
        objective = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        tolerance = 1e-7 * max(1.0, abs(objective))
        if objective >= model._logged_best - tolerance:
            return

        values = model.cbGetSolution(model._y_list)
        y_values = [1 if value >= 0.5 else 0 for value in values]
        solution = {
            "objective_value": objective,
            "y": y_values
        }

        if model._solution_logger:
            model._solution_logger.log_solution(objective, solution)
        model._logged_best = objective
    except Exception:
        # Logging must not interrupt the optimizer.
        pass


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_facilities = int(data["num_facilities"])
    num_customers = int(data["num_customers"])
    num_scenarios = int(data["num_scenarios"])

    fixed_costs = data["facilities"]["fixed_costs"]
    capacities = data["facilities"]["capacities"]
    routing_costs = data["routing_costs"]
    probabilities = data["scenarios"]["probabilities"]
    scenario_demands = data["scenarios"]["demands"]

    if len(probabilities) != num_scenarios:
        raise ValueError("Scenario probability count does not match num_scenarios.")
    if len(scenario_demands) != num_scenarios:
        raise ValueError("Scenario demand count does not match num_scenarios.")

    max_total_demand = max(
        (sum(scenario_demands[s]) for s in range(num_scenarios)),
        default=0
    )

    warm_objective, warm_y, warm_flows = build_initial_solution(
        fixed_costs, capacities, routing_costs,
        probabilities, scenario_demands
    )

    warm_solution = {
        "objective_value": float(warm_objective),
        "y": warm_y
    }
    if logger:
        logger.log_solution(float(warm_objective), warm_solution)

    model = gp.Model("stochastic_capacitated_facility_location")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    y = model.addVars(
        num_facilities,
        vtype=GRB.BINARY,
        name="y"
    )
    x = model.addVars(
        num_scenarios,
        num_facilities,
        num_customers,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="x"
    )

    model.ModelSense = GRB.MINIMIZE
    for i in range(num_facilities):
        y[i].Obj = float(fixed_costs[i])

    for s in range(num_scenarios):
        probability = float(probabilities[s])
        for i in range(num_facilities):
            for j in range(num_customers):
                x[s, i, j].Obj = probability * float(routing_costs[i][j])

    all_nonnegative_costs = all(
        routing_costs[i][j] >= 0
        for i in range(num_facilities)
        for j in range(num_customers)
    )

    for s in range(num_scenarios):
        for j in range(num_customers):
            inbound = gp.quicksum(x[s, i, j] for i in range(num_facilities))
            demand = float(scenario_demands[s][j])
            if all_nonnegative_costs:
                model.addConstr(inbound == demand, name=f"demand_{s}_{j}")
            else:
                model.addConstr(inbound >= demand, name=f"demand_{s}_{j}")

        for i in range(num_facilities):
            model.addConstr(
                gp.quicksum(x[s, i, j] for j in range(num_customers))
                <= float(capacities[i]) * y[i],
                name=f"capacity_{s}_{i}"
            )

    model.addConstr(
        gp.quicksum(float(capacities[i]) * y[i]
                    for i in range(num_facilities))
        >= float(max_total_demand),
        name="total_open_capacity"
    )

    # Provide a complete feasible MIP start.
    for i in range(num_facilities):
        y[i].Start = warm_y[i]

    for s in range(num_scenarios):
        flows = warm_flows[s]
        for i in range(num_facilities):
            for j in range(num_customers):
                x[s, i, j].Start = flows.get((i, j), 0.0)

    elapsed = time.monotonic() - start_time
    remaining_time = max(0.0, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining_time

    model._y_list = [y[i] for i in range(num_facilities)]
    model._solution_logger = logger
    model._logged_best = float(warm_objective)

    if remaining_time > 0:
        model.optimize(incumbent_callback)

    if model.SolCount > 0:
        objective_value = float(model.ObjVal)
        y_values = [
            1 if y[i].X >= 0.5 else 0
            for i in range(num_facilities)
        ]
        solution = {
            "objective_value": objective_value,
            "y": y_values
        }

        tolerance = 1e-7 * max(1.0, abs(objective_value))
        if logger and objective_value < model._logged_best - tolerance:
            logger.log_solution(objective_value, solution)
    else:
        solution = warm_solution

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(solution_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()