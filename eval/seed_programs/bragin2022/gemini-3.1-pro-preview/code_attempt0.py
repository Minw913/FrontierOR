import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Solve Generalized Assignment Problem using Gurobi.")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the JSON file containing the problem instance.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path where the final solution JSON file must be written.")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument('--log_path', type=str, default=None, help="Path to a JSONL file for logging intermediate solutions (optional).")
    
    args = parser.parse_args()
    
    # Initialize the solution logger if a log path is provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    num_machines = data['num_machines']
    num_jobs = data['num_jobs']
    cost_matrix = data['cost_matrix']
    resource_matrix = data['resource_matrix']
    capacities = data['capacities']
    
    # Configure Gurobi Environment
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    
    # Build Gurobi Model
    model = gp.Model("GeneralizedAssignment", env=env)
    
    # Constrain to single CPU core
    model.setParam('Threads', 1)
    
    # Set time limit
    if args.time_limit > 0:
        model.setParam('TimeLimit', args.time_limit)
        
    # Create binary decision variables
    x = model.addVars(num_machines, num_jobs, vtype=GRB.BINARY, name="x")
    
    # Set Objective
    cost_dict = {(i, j): cost_matrix[i][j] for i in range(num_machines) for j in range(num_jobs)}
    model.setObjective(x.prod(cost_dict), GRB.MINIMIZE)
    
    # Constraint 1: Extactly one machine per job
    model.addConstrs((x.sum('*', j) == 1 for j in range(num_jobs)), name="job_assignment")
    
    # Constraint 2: Resource capacity on each machine
    for i in range(num_machines):
        model.addConstr(
            gp.quicksum(resource_matrix[i][j] * x[i, j] for j in range(num_jobs)) <= capacities[i],
            name=f"capacity_{i}"
        )
        
    # Callback to log intermediate solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
                
    # Optimize
    model.optimize(callback)
    
    # Extract Solution
    if model.SolCount > 0:
        assignments = {}
        for j in range(num_jobs):
            for i in range(num_machines):
                # Checking > 0.5 properly accounts for floating point imprecision
                if x[i, j].X > 0.5:
                    assignments[str(j)] = i
                    break
                    
        solution = {
            "objective_value": float(model.ObjVal),
            "assignments": assignments
        }
    else:
        # Give fallback empty JSON block if no feasible solution was found
        solution = {}
        
    # Write output to solution file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()