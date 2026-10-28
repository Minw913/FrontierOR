import argparse
import json
from gurobipy import Model, GRB, quicksum
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, log_path=None):
    num_facilities = instance['parameters']['num_facilities']
    num_customers = instance['parameters']['num_customers']
    num_periods = instance['parameters']['num_periods']
    seasonal_factors = instance['seasonal_factors']
    demands = instance['demands']
    holding_costs = instance['holding_costs']
    capacities = instance['capacities']
    transportation_costs = instance['transportation_costs']

    customer_demands = [[demands[c] * seasonal_factors[t] for t in range(num_periods)] for c in range(num_customers)]

    model = Model("LogisticsDistribution")
    model.setParam("Threads", 1)
    model.modelSense = GRB.MINIMIZE

    assign_vars = model.addVars(num_customers, num_facilities, vtype=GRB.BINARY, name='assign')
    for c in range(num_customers):
        model.addConstr(quicksum(assign_vars[c, f] for f in range(num_facilities)) == 1, name=f'assign_customer_{c}')

    inventory_vars = model.addVars(num_facilities, num_periods, vtype=GRB.CONTINUOUS, name='inventory', lb=0)
    demand_vars = model.addVars(num_facilities, num_periods, vtype=GRB.CONTINUOUS, name='demand')
    for f in range(num_facilities):
        for t in range(num_periods):
            model.addConstr(demand_vars[f, t] == quicksum(customer_demands[c][t] * assign_vars[c, f] for c in range(num_customers)), name=f'demand_facility_{f}_period_{t}')

    for f in range(num_facilities):
        for t in range(num_periods):
            if t == 0:
                model.addConstr(inventory_vars[f, t] == max(0, demand_vars[f, t] - capacities[f][t]), name=f'inventory_balance_{f}_{t}')
            else:
                model.addConstr(inventory_vars[f, t] == inventory_vars[f, t-1] + demand_vars[f, t] - capacities[f][t], name=f'inventory_balance_{f}_{t}')
            model.addConstr(demand_vars[f, t] <= capacities[f][t] + (inventory_vars[f, t-1] if t > 0 else 0), name=f'capacity_constraint_{f}_{t}')

    total_supply_cost = quicksum(transportation_costs[f][c][t] * customer_demands[c][t] * assign_vars[c, f] for f in range(num_facilities) for c in range(num_customers) for t in range(num_periods))
    total_inventory_holding_cost = quicksum(holding_costs[f][t] * inventory_vars[f, t] for f in range(num_facilities) for t in range(num_periods))
    model.setObjective(total_supply_cost + total_inventory_holding_cost)

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                logger.log(model.getObjective().getValue())

    model.setParam('TimeLimit', time_limit)
    model.optimize(callback)
    
    if model.status == GRB.INFEASIBLE:
        return {
            'objective_value': float('inf'),
            'assignment': {str(c): None for c in range(num_customers)},
            'inventory': {str(f): [0.0 for _ in range(num_periods)] for f in range(num_facilities)}
        }

    if model.SolCount > 0:
        assignment = {str(c): np.argmax([assign_vars[c, f].X for f in range(num_facilities)]) for c in range(num_customers)}
        inventory = {str(f): [max(inventory_vars[f, t].X, 0) for t in range(num_periods)] for f in range(num_facilities)}
        return {
            'objective_value': model.getObjective().getValue(),
            'assignment': assignment,
            'inventory': inventory
        }
    else:
        return {
            'objective_value': float('inf'),
            'assignment': {str(c): None for c in range(num_customers)},
            'inventory': {str(f): [0.0 for _ in range(num_periods)] for f in range(num_facilities)}
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, args.log_path)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()