import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, log_path, sense="minimize"):
            pass
        def log(self, objective_value):
            pass

def main():
    parser = argparse.ArgumentParser(description="Solve Generalized Assignment Problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to the output solution JSON file.")
    parser.add_argument("--time_limit", type=int, default=None, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path for logging intermediate solutions.")
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_agents = instance['num_agents']
    num_jobs = instance['num_jobs']
    profits = instance['profits']
    weights = instance['weights']
    capacities = instance['capacities']

    # Initialize objective logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Initialize Gurobi model
    model = gp.Model("GeneralizedAssignment")
    model.setParam("Threads", 1)  # Execute in a single CPU core
    if args.time_limit is not None:
        model.setParam("TimeLimit", args.time_limit)

    # Flatten profits to serve as objective coefficients
    profits_dict = {(i, j): profits[i][j] for i in range(num_agents) for j in range(num_jobs)}

    # Add binary decision variables x_ij: 1 if job j is assigned to agent i
    x = model.addVars(num_agents, num_jobs, vtype=GRB.BINARY, obj=profits_dict, name="x")
    
    # We want to maximize total profit
    model.ModelSense = GRB.MAXIMIZE

    # Constraint 1: Each job must be assigned to exactly one single agent
    model.addConstrs((x.sum('*', j) == 1 for j in range(num_jobs)), name="assign_job")

    # Constraint 2: Total weight of jobs assigned to an agent must not exceed its capacity
    for i in range(num_agents):
        model.addConstr(
            gp.quicksum(weights[i][j] * x[i, j] for j in range(num_jobs)) <= capacities[i],
            name=f"capacity_agent_{i}"
        )

    # Define the callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Solve the problem
    model.optimize(callback)

    # Extract best found solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        assignment = {}
        for j in range(num_jobs):
            for i in range(num_agents):
                # Checking > 0.5 because values could be subject to floating point slight inaccuracies 
                if x[i, j].X > 0.5:
                    assignment[str(j)] = i
                    break
        
        solution = {
            "objective_value": objective_value,
            "assignment": assignment
        }
    else:
        # If no feasible solution was found during the given time limit
        solution = {
            "objective_value": 0.0,
            "assignment": {}
        }

    # Write output matching the solution schema
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()