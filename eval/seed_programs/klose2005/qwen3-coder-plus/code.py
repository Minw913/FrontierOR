import argparse
import json
import math
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def solve_facility_location(instance_data, time_limit, logger=None):
    num_customers = instance_data["num_customers"]
    num_plants = instance_data["num_plants"]
    
    demands = instance_data["customers"]["demands"]
    capacities = instance_data["plants"]["capacities"]
    fixed_costs = instance_data["plants"]["fixed_costs"]
    shipping_costs = instance_data["shipping_costs"]
    
    # Create a model
    model = gp.Model("facility_location")
    model.setParam("Threads", 1)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.OutputFlag, 0)  # Turn off verbose output
    
    # Variables: y[j] - binary, whether plant j is open
    y = model.addVars(num_plants, vtype=GRB.BINARY, name="y")
    
    # Variables: x[i, j] - continuous [0,1], fraction of customer i's demand served by plant j
    x = model.addVars(num_customers, num_plants, lb=0, ub=1, name="x")
    
    # Objective function: minimize total cost (shipping + fixed opening costs)
    obj_expr = gp.quicksum(fixed_costs[j] * y[j] for j in range(num_plants))
    obj_expr += gp.quicksum(demands[i] * shipping_costs[i][j] * x[i, j] 
                            for i in range(num_customers) for j in range(num_plants))
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Constraints: Each customer's demand is fully satisfied
    for i in range(num_customers):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_plants)) == 1, f"demand_satisfaction_{i}")
    
    # Constraints: Capacity constraint for each plant
    for j in range(num_plants):
        model.addConstr(
            gp.quicksum(demands[i] * x[i, j] for i in range(num_customers)) <= capacities[j] * y[j],
            f"capacity_{j}"
        )
    
    # Solve the model
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objval)
    
    model.optimize(callback)
    
    # Extract the solution
    if model.status == GRB.INFEASIBLE:
        raise Exception("Model is infeasible.")
    
    open_plants = []
    assignments = {}
    
    for j in range(num_plants):
        if y[j].X > 0.5:  # Plant is open
            open_plants.append(j)
    
    for i in range(num_customers):
        cust_assignments = {}
        for j in range(num_plants):
            if x[i, j].X > 1e-6:  # Only include non-zero assignments
                cust_assignments[str(j)] = x[i, j].X
        assignments[str(i)] = cust_assignments
    
    return {
        "objective_value": model.objVal,
        "open_plants": open_plants,
        "assignments": assignments
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path where the solution should be written")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", help="Path to the log file (optional)")
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    try:
        solution = solve_facility_location(instance_data, args.time_limit, logger)
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)

    except Exception as e:
        print(f"Error solving instance: {e}")
        raise


if __name__ == "__main__":
    main()