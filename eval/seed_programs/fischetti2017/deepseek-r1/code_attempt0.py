import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Solve quadratic Uncapacitated Facility Location (qUFL) problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the JSON file containing the problem instance.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path where the solution JSON file will be written.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions (optional).')
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    num_facilities = data['num_facilities']
    num_customers = data['num_customers']
    opening_costs = data['opening_costs']
    allocation_costs = data['allocation_costs']  # Shape: [num_facilities][num_customers]

    # Initialize solution logger if log_path is provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Build a trivial feasible solution: assign each customer to the facility with the smallest allocation cost
    y_trivial = [0] * num_facilities
    x_trivial = [[0.0] * num_customers for _ in range(num_facilities)]
    
    for j in range(num_customers):
        min_cost = float('inf')
        best_i = -1
        for i in range(num_facilities):
            if allocation_costs[i][j] < min_cost:
                min_cost = allocation_costs[i][j]
                best_i = i
        x_trivial[best_i][j] = 1.0
        y_trivial[best_i] = 1

    # Create Gurobi model
    model = gp.Model("qUFL")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    # Add variables
    y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")
    x = model.addVars(num_facilities, num_customers, vtype=GRB.CONTINUOUS, name="x", lb=0.0)

    # Set initial solution (MIP start)
    for i in range(num_facilities):
        y[i].Start = y_trivial[i]
    for i in range(num_facilities):
        for j in range(num_customers):
            x[i, j].Start = x_trivial[i][j]

    # Set objective: minimize total cost = sum(opening_costs[i] * y_i) + sum(allocation_costs[i][j] * x_ij^2)
    obj = gp.quicksum(opening_costs[i] * y[i] for i in range(num_facilities))
    for i in range(num_facilities):
        for j in range(num_customers):
            obj += allocation_costs[i][j] * x[i, j] * x[i, j]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints: each customer must be fully served
    for j in range(num_customers):
        model.addConstr(gp.quicksum(x[i, j] for i in range(num_facilities)) == 1, name=f"demand_{j}")

    # Constraints: a customer can only be served by an open facility
    for i in range(num_facilities):
        for j in range(num_customers):
            model.addConstr(x[i, j] <= y[i], name=f"link_{i}_{j}")

    # Store variables and logger in model for callback access
    model._y = y
    model._x = x
    model._logger = logger

    # Callback function to log intermediate solutions
    def gurobi_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Get current solution values
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            y_vals = [model.cbGetSolution(model._y[i]) for i in range(num_facilities)]
            x_vals = [[model.cbGetSolution(model._x[i, j]) for j in range(num_customers)] for i in range(num_facilities)]
            
            # Prepare solution dictionary
            sol_dict = {
                "objective_value": obj_val,
                "variant": "quadratic",
                "y": [int(round(y_val)) for y_val in y_vals],
                "x": x_vals
            }
            
            # Log the solution
            if model._logger:
                model._logger.log_solution(obj_val, sol_dict)

    # Optimize the model
    model.optimize(gurobi_callback)

    # Extract the best solution found
    if model.SolCount > 0:
        obj_val = model.ObjVal
        y_sol = [int(round(y[i].X)) for i in range(num_facilities)]
        x_sol = [[x[i, j].X for j in range(num_customers)] for i in range(num_facilities)]
    else:
        # Fallback to the initial solution if no solution found
        obj_val = 0.0
        for i in range(num_facilities):
            if y_trivial[i]:
                obj_val += opening_costs[i]
        for i in range(num_facilities):
            for j in range(num_customers):
                if x_trivial[i][j] > 0:
                    obj_val += allocation_costs[i][j] * (x_trivial[i][j] ** 2)
        y_sol = y_trivial
        x_sol = x_trivial

    # Prepare final solution dictionary
    solution_dict = {
        "objective_value": obj_val,
        "variant": "quadratic",
        "y": y_sol,
        "x": x_sol
    }

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == '__main__':
    main()