import json
import argparse
import time
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description="Stochastic Facility Location Problem")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Extract data
    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    num_scenarios = instance['scenarios']['num_scenarios']

    fixed_costs = instance['facilities']['fixed_costs']
    capacities = instance['facilities']['capacities']
    routing_costs = instance['routing_costs']
    probabilities = instance['scenarios']['probabilities']
    demands = instance['scenarios']['demands']

    # Compute max total demand across scenarios
    max_total_demand = max(sum(demands[s]) for s in range(num_scenarios))

    import gurobipy as gp
    from gurobipy import GRB

    # Build MIP model
    model = gp.Model("stochastic_facility_location")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")

    # Flow variables: x[i,j,s] = flow from facility i to customer j in scenario s
    x = model.addVars(num_facilities, num_customers, num_scenarios,
                      vtype=GRB.CONTINUOUS, lb=0.0, name="x")

    # Objective: fixed costs + expected routing costs
    obj = gp.LinExpr()
    for i in range(num_facilities):
        obj += fixed_costs[i] * y[i]
    for s in range(num_scenarios):
        for i in range(num_facilities):
            for j in range(num_customers):
                obj += probabilities[s] * routing_costs[i][j] * x[i, j, s]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint 1: Demand satisfaction for each customer in each scenario
    for s in range(num_scenarios):
        for j in range(num_customers):
            model.addConstr(
                gp.quicksum(x[i, j, s] for i in range(num_facilities)) >= demands[s][j],
                name=f"demand_s{s}_c{j}"
            )

    # Constraint 2: Capacity constraints for each facility in each scenario
    for s in range(num_scenarios):
        for i in range(num_facilities):
            model.addConstr(
                gp.quicksum(x[i, j, s] for j in range(num_customers)) <= capacities[i] * y[i],
                name=f"cap_s{s}_f{i}"
            )

    # Constraint 3: Total capacity of opened facilities >= max total demand
    model.addConstr(
        gp.quicksum(capacities[i] * y[i] for i in range(num_facilities)) >= max_total_demand,
        name="total_capacity"
    )

    # Callback to log incumbent solutions
    best_obj = [float('inf')]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0]:
                best_obj[0] = obj_val
                y_vals = [int(round(model.cbGetSolution(y[i]))) for i in range(num_facilities)]
                solution = {
                    "objective_value": obj_val,
                    "y": y_vals
                }
                if logger:
                    logger.log_solution(obj_val, solution)

    model.setCallback(callback)

    # Solve
    model.optimize()

    # Extract final solution
    if model.SolCount > 0:
        y_vals = [int(round(y[i].X)) for i in range(num_facilities)]
        obj_val = model.ObjVal

        solution = {
            "objective_value": obj_val,
            "y": y_vals
        }

        # Log if not already logged via callback
        if obj_val < best_obj[0] or best_obj[0] == float('inf'):
            if logger:
                logger.log_solution(obj_val, solution)
    else:
        # Fallback: open all facilities as a feasible solution
        y_vals = [1] * num_facilities
        # Estimate objective: fixed costs + greedy routing cost
        total_fixed = sum(fixed_costs)
        # Compute a rough routing cost estimate
        routing_cost_est = 0.0
        for s in range(num_scenarios):
            for j in range(num_customers):
                # Assign to cheapest open facility
                min_cost = min(routing_costs[i][j] for i in range(num_facilities))
                routing_cost_est += probabilities[s] * demands[s][j] * min_cost
        obj_val = total_fixed + routing_cost_est

        solution = {
            "objective_value": obj_val,
            "y": y_vals
        }
        if logger:
            logger.log_solution(obj_val, solution)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()