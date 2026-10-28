import json
import argparse
import time
from gurobipy import *


def compute_min_inventory(demand_per_period, capacity_per_period, num_periods):
    """Compute minimum inventory levels using backward pass."""
    if num_periods == 0:
        return []
    min_inv = [0.0] * num_periods
    for t in range(num_periods - 2, -1, -1):
        min_inv[t] = max(0.0, demand_per_period[t + 1] + min_inv[t + 1] - capacity_per_period[t + 1])
    # Check period 0 feasibility
    if demand_per_period[0] + min_inv[0] > capacity_per_period[0] + 1e-9:
        return None
    return min_inv


def greedy_solution(instance):
    """Compute a greedy feasible solution."""
    num_facilities = instance['parameters']['num_facilities']
    num_customers = instance['parameters']['num_customers']
    num_periods = instance['parameters']['num_periods']
    seasonal = instance['seasonal_factors']
    demands = instance['demands']
    holding_costs = instance['holding_costs']
    capacities = instance['capacities']
    transport_costs = instance['transportation_costs']

    # Precompute cumulative seasonal factors
    cum_seasonal = [0.0] * num_periods
    cum_seasonal[0] = seasonal[0]
    for t in range(1, num_periods):
        cum_seasonal[t] = cum_seasonal[t - 1] + seasonal[t]

    # Precompute cumulative capacities
    cum_capacities = [[0.0] * num_periods for _ in range(num_facilities)]
    for f in range(num_facilities):
        cum_capacities[f][0] = capacities[f][0]
        for t in range(1, num_periods):
            cum_capacities[f][t] = cum_capacities[f][t - 1] + capacities[f][t]

    # Sort customers by demand (largest first for better packing)
    customer_order = sorted(range(num_customers), key=lambda c: -demands[c])

    assignment = [-1] * num_customers
    facility_cum_demand = [[0.0] * num_periods for _ in range(num_facilities)]

    for c in customer_order:
        # Compute cost for each facility
        facility_costs = []
        for f in range(num_facilities):
            cost = sum(transport_costs[f][c][t] for t in range(num_periods))
            facility_costs.append((cost, f))
        facility_costs.sort()

        assigned = False
        for cost, f in facility_costs:
            # Check feasibility: cumulative demand must not exceed cumulative capacity
            feasible = True
            for t in range(num_periods):
                new_cum = facility_cum_demand[f][t] + demands[c] * cum_seasonal[t]
                if new_cum > cum_capacities[f][t] + 1e-9:
                    feasible = False
                    break
            if feasible:
                assignment[c] = f
                for t in range(num_periods):
                    facility_cum_demand[f][t] += demands[c] * cum_seasonal[t]
                assigned = True
                break

        if not assigned:
            # Assign to facility with most total remaining capacity
            best_f = 0
            best_slack = -float('inf')
            for f in range(num_facilities):
                slack = cum_capacities[f][num_periods - 1] - facility_cum_demand[f][num_periods - 1]
                if slack > best_slack:
                    best_slack = slack
                    best_f = f
            assignment[c] = best_f
            for t in range(num_periods):
                facility_cum_demand[best_f][t] += demands[c] * cum_seasonal[t]

    # Compute minimum inventory for each facility
    inventory = {}
    for f in range(num_facilities):
        demand_per_period = [0.0] * num_periods
        for t in range(num_periods):
            if t == 0:
                demand_per_period[t] = facility_cum_demand[f][0]
            else:
                demand_per_period[t] = facility_cum_demand[f][t] - facility_cum_demand[f][t - 1]

        min_inv = compute_min_inventory(demand_per_period, capacities[f], num_periods)
        if min_inv is None:
            min_inv = [0.0] * num_periods
        inventory[str(f)] = min_inv

    # Compute objective
    obj_val = 0.0
    for c in range(num_customers):
        f = assignment[c]
        for t in range(num_periods):
            obj_val += transport_costs[f][c][t]
    for f in range(num_facilities):
        for t in range(num_periods):
            obj_val += holding_costs[f][t] * inventory[str(f)][t]

    return obj_val, {str(c): assignment[c] for c in range(num_customers)}, inventory


def extract_solution_from_model(model, x, I, num_customers, num_facilities, num_periods):
    """Extract solution from solved Gurobi model."""
    assignment = {}
    for c in range(num_customers):
        best_f = 0
        best_val = -1.0
        for f in range(num_facilities):
            val = x[c, f].X
            if val > best_val:
                best_val = val
                best_f = f
        assignment[str(c)] = best_f

    inventory = {}
    for f in range(num_facilities):
        inventory[str(f)] = [max(0.0, I[f, t].X) for t in range(num_periods)]

    return assignment, inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_facilities = instance['parameters']['num_facilities']
    num_customers = instance['parameters']['num_customers']
    num_periods = instance['parameters']['num_periods']
    seasonal = instance['seasonal_factors']
    demands = instance['demands']
    holding_costs = instance['holding_costs']
    capacities = instance['capacities']
    transport_costs = instance['transportation_costs']

    # Get greedy solution for warm start and fallback
    greedy_obj, greedy_assignment, greedy_inventory = greedy_solution(instance)

    best_solution = {
        "objective_value": greedy_obj,
        "assignment": greedy_assignment,
        "inventory": greedy_inventory
    }

    if logger:
        logger.log_solution(greedy_obj, best_solution)

    # Check remaining time
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed

    if remaining_time <= 2:
        with open(args.solution_path, 'w') as f:
            json.dump(best_solution, f)
        return

    # Build Gurobi MIP model
    model = Model("facility_assignment")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', remaining_time)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = {}
    for c in range(num_customers):
        for f in range(num_facilities):
            x[c, f] = model.addVar(vtype=GRB.BINARY, name=f"x_{c}_{f}")

    I = {}
    for f in range(num_facilities):
        for t in range(num_periods):
            I[f, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"I_{f}_{t}")

    model.update()

    # Set warm start from greedy solution
    for c in range(num_customers):
        for f in range(num_facilities):
            x[c, f].Start = 1.0 if greedy_assignment[str(c)] == f else 0.0
    for f in range(num_facilities):
        for t in range(num_periods):
            I[f, t].Start = greedy_inventory[str(f)][t]

    # Objective: minimize total supply cost + total inventory holding cost
    obj = LinExpr()
    for c in range(num_customers):
        for f in range(num_facilities):
            cost_cf = sum(transport_costs[f][c][t] for t in range(num_periods))
            obj += x[c, f] * cost_cf

    for f in range(num_facilities):
        for t in range(num_periods):
            obj += holding_costs[f][t] * I[f, t]

    model.setObjective(obj, GRB.MINIMIZE)

    # Assignment constraints: each customer assigned to exactly one facility
    for c in range(num_customers):
        model.addConstr(quicksum(x[c, f] for f in range(num_facilities)) == 1, name=f"assign_{c}")

    # Capacity/inventory balance constraints
    for f in range(num_facilities):
        for t in range(num_periods):
            demand_expr = quicksum(x[c, f] * (demands[c] * seasonal[t]) for c in range(num_customers))
            if t == 0:
                model.addConstr(demand_expr + I[f, t] <= capacities[f][t], name=f"cap_{f}_{t}")
            else:
                model.addConstr(demand_expr + I[f, t] <= capacities[f][t] + I[f, t - 1], name=f"cap_{f}_{t}")

    # Callback to log incumbent solutions
    best_obj = [greedy_obj]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    assignment = {}
                    for c in range(num_customers):
                        best_f = 0
                        best_val = -1.0
                        for f in range(num_facilities):
                            val = model.cbGetSolution(x[c, f])
                            if val > best_val:
                                best_val = val
                                best_f = f
                        assignment[str(c)] = best_f
                    inventory = {}
                    for f in range(num_facilities):
                        inventory[str(f)] = [max(0.0, model.cbGetSolution(I[f, t])) for t in range(num_periods)]
                    solution = {
                        "objective_value": obj_val,
                        "assignment": assignment,
                        "inventory": inventory
                    }
                    logger.log_solution(obj_val, solution)

    try:
        model.optimize(callback)
    except GurobiError:
        pass

    # Extract final solution from Gurobi
    if model.SolCount > 0:
        assignment, inventory = extract_solution_from_model(model, x, I, num_customers, num_facilities, num_periods)
        objective_value = model.ObjVal

        solution = {
            "objective_value": objective_value,
            "assignment": assignment,
            "inventory": inventory
        }

        if objective_value < best_solution["objective_value"]:
            best_solution = solution

        if logger and objective_value < best_obj[0] - 1e-6:
            logger.log_solution(objective_value, solution)

    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f)


if __name__ == '__main__':
    main()