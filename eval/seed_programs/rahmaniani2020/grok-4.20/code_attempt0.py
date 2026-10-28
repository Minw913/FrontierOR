import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_facilities = data['num_facilities']
    num_customers = data['num_customers']
    num_scenarios = data['num_scenarios']
    fixed_costs = data['facilities']['fixed_costs']
    capacities = data['facilities']['capacities']
    routing_costs = data['routing_costs']
    probabilities = data['scenarios']['probabilities']
    demands = data['scenarios']['demands']
    max_total_demand = data['max_total_demand']

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model("StochasticFacilityLocation")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 0.01
    model.Params.LogToConsole = 0

    y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")
    x = model.addVars(num_facilities, num_scenarios, num_customers, lb=0, name="x")

    # Objective: fixed costs + expected routing cost
    fixed_cost_expr = gp.quicksum(fixed_costs[i] * y[i] for i in range(num_facilities))
    routing_expr = 0.0
    for s in range(num_scenarios):
        for i in range(num_facilities):
            for j in range(num_customers):
                routing_expr += probabilities[s] * routing_costs[i][j] * x[i, s, j]
    model.setObjective(fixed_cost_expr + routing_expr, GRB.MINIMIZE)

    # Demand satisfaction
    for s in range(num_scenarios):
        for j in range(num_customers):
            model.addConstr(
                gp.quicksum(x[i, s, j] for i in range(num_facilities)) >= demands[s][j],
                name=f"demand_s{s}_c{j}"
            )

    # Capacity constraints
    for s in range(num_scenarios):
        for i in range(num_facilities):
            model.addConstr(
                gp.quicksum(x[i, s, j] for j in range(num_customers)) <= capacities[i] * y[i],
                name=f"cap_s{s}_f{i}"
            )

    # Total capacity covering max demand
    model.addConstr(
        gp.quicksum(capacities[i] * y[i] for i in range(num_facilities)) >= max_total_demand,
        name="total_capacity"
    )

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        objective_value = model.objVal
        y_sol = [int(y[i].X + 0.5) for i in range(num_facilities)]
        
        # If y_sol does not satisfy total capacity, force feasibility by opening cheapest sufficient set
        total_cap_open = sum(capacities[i] * y_sol[i] for i in range(num_facilities))
        if total_cap_open < max_total_demand:
            # Simple greedy: open facilities in order of increasing fixed cost until capacity is met
            fac_list = list(range(num_facilities))
            fac_list.sort(key=lambda i: fixed_costs[i])
            total_cap = 0
            to_open = set()
            for i in fac_list:
                to_open.add(i)
                total_cap += capacities[i]
                if total_cap >= max_total_demand:
                    break
            y_sol = [1 if i in to_open else 0 for i in range(num_facilities)]
            # Recompute a feasible routing cost (upper bound)
            # Use simple assignment to lowest cost open facility per customer per scenario
            routing_cost = 0.0
            open_set = [i for i in range(num_facilities) if y_sol[i] == 1]
            for s in range(num_scenarios):
                for j in range(num_customers):
                    min_cost = min(routing_costs[i][j] for i in open_set)
                    routing_cost += probabilities[s] * min_cost * demands[s][j]
            objective_value = sum(fixed_costs[i] for i in range(num_facilities) if y_sol[i] == 1) + routing_cost

        solution = {
            "objective_value": float(objective_value),
            "y": y_sol
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Fallback: open all facilities
        y_sol = [1] * num_facilities
        objective_value = sum(fixed_costs) + sum(
            probabilities[s] * sum(routing_costs[i][j] * demands[s][j] for i in range(num_facilities) for j in range(num_customers))
            for s in range(num_scenarios)
        )
        solution = {
            "objective_value": float(objective_value),
            "y": y_sol
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()