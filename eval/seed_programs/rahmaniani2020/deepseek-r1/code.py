import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Initialize solution logger if log_path is provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    num_scenarios = instance['num_scenarios']
    max_total_demand = instance['max_total_demand']
    facilities = instance['facilities']
    customers = instance['customers']
    routing_costs = instance['routing_costs']
    scenarios = instance['scenarios']
    
    fixed_costs = facilities['fixed_costs']
    capacities = facilities['capacities']
    probabilities = scenarios['probabilities']
    demands = scenarios['demands']

    # Create Gurobi model
    model = gp.Model('FacilityLocation')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    # Add binary variables for facility openings
    y = model.addVars(num_facilities, vtype=GRB.BINARY, name='y')
    
    # Add continuous flow variables for each facility-customer-scenario combination
    x = model.addVars(
        num_facilities, 
        num_customers, 
        num_scenarios, 
        vtype=GRB.CONTINUOUS, 
        name='x'
    )

    # Total capacity constraint: opened facilities must cover max total demand
    model.addConstr(
        gp.quicksum(capacities[i] * y[i] for i in range(num_facilities)) >= max_total_demand,
        name='total_capacity'
    )

    # Customer demand constraints (per scenario)
    for s in range(num_scenarios):
        for j in range(num_customers):
            model.addConstr(
                gp.quicksum(x[i, j, s] for i in range(num_facilities)) >= demands[s][j],
                name=f'demand_s{s}_c{j}'
            )

    # Facility capacity constraints (per scenario)
    for i in range(num_facilities):
        for s in range(num_scenarios):
            model.addConstr(
                gp.quicksum(x[i, j, s] for j in range(num_customers)) <= capacities[i] * y[i],
                name=f'capacity_f{i}_s{s}'
            )

    # Objective function components
    fixed_cost_component = gp.quicksum(fixed_costs[i] * y[i] for i in range(num_facilities))
    routing_cost_component = gp.quicksum(
        probabilities[s] * routing_costs[i][j] * x[i, j, s]
        for i in range(num_facilities)
        for j in range(num_customers)
        for s in range(num_scenarios)
    )
    model.setObjective(fixed_cost_component + routing_cost_component, GRB.MINIMIZE)

    # Callback function for logging intermediate solutions
    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            # Get current objective value and facility status
            obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            y_vals = [round(m.cbGetSolution(y[i])) for i in range(num_facilities)]
            
            # Log solution if logger is available
            if logger:
                solution_dict = {
                    "objective_value": obj_val,
                    "y": y_vals
                }
                logger.log_solution(obj_val, solution_dict)

    # Optimize model with callback
    model._y = y  # Store variable reference for callback
    model.optimize(callback)

    # Prepare final solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            obj_val = model.objVal
            y_sol = [round(y[i].X) for i in range(num_facilities)]
        else:
            # Fallback: open all facilities if no solution found
            obj_val = sum(fixed_costs)
            y_sol = [1] * num_facilities
    else:
        # Default solution if optimization fails
        obj_val = sum(fixed_costs)
        y_sol = [1] * num_facilities

    # Write final solution
    solution = {
        "objective_value": float(obj_val),
        "y": y_sol
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()