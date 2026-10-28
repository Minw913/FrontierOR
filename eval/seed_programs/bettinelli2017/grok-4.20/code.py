import argparse
import json
import time
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Read instance
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = instance["num_items"]
    capacity = instance["capacity"]
    profits = instance["items"]["profits"]
    weights = instance["items"]["weights"]
    edges = instance["conflict_graph"]["edges"]

    # Build adjacency list for conflicts
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Create Gurobi model
    model = gp.Model("KnapsackWithConflicts")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-4
    model.Params.LogToConsole = 0

    # Variables: x[i] = 1 if item i is selected
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Objective: maximize profit
    model.setObjective(gp.quicksum(profits[i] * x[i] for i in range(n)), GRB.MAXIMIZE)

    # Weight constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "capacity")

    # Conflict constraints
    for u, v in edges:
        if u < v:  # avoid duplicate
            model.addConstr(x[u] + x[v] <= 1, f"conflict_{u}_{v}")

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    selected = []
    if model.SolCount > 0:
        for i in range(n):
            if x[i].X > 0.5:
                selected.append(i)
        objective = float(model.ObjVal)
    else:
        # No solution found, return empty
        objective = 0.0

    # Write solution
    solution = {
        "objective_value": objective,
        "selected_items": selected
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()