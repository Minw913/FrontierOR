import json
import argparse
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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_agents = instance['num_agents']
    num_jobs = instance['num_jobs']
    profits = instance['profits']
    weights = instance['weights']
    capacities = instance['capacities']

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    model = gp.Model("AssignmentWithCapacity")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('LogToConsole', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('Threads', 1)

    # Binary variables: x[a][j] = 1 if agent a does job j
    x = []
    for a in range(num_agents):
        x.append([model.addVar(vtype=GRB.BINARY, name=f"x_{a}_{j}") for j in range(num_jobs)])

    # Objective: maximize total profit
    obj = gp.quicksum(profits[a][j] * x[a][j] for a in range(num_agents) for j in range(num_jobs))
    model.setObjective(obj, GRB.MAXIMIZE)

    # Each job assigned to exactly one agent
    for j in range(num_jobs):
        model.addConstr(gp.quicksum(x[a][j] for a in range(num_agents)) == 1, f"assign_{j}")

    # Capacity constraints
    for a in range(num_agents):
        model.addConstr(gp.quicksum(weights[a][j] * x[a][j] for j in range(num_jobs)) <= capacities[a], f"cap_{a}")

    # Callback to log improved solutions
    best_obj = float('-inf')
    start_time = time.time()

    def callback(model_local, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model_local.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val > best_obj + 1e-6:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        assignment = {}
        for j in range(num_jobs):
            for a in range(num_agents):
                if x[a][j].X > 0.5:
                    assignment[str(j)] = a
                    break

        solution = {
            "objective_value": float(objective_value),
            "assignment": assignment
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No feasible solution found
        solution = {
            "objective_value": 0.0,
            "assignment": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()