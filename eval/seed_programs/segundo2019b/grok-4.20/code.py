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

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    graph = instance['graph']
    vertex_weights = instance['vertex_weights']
    n = graph['num_vertices']
    edges = graph['edges']
    vertices = graph['vertices']

    # Build adjacency matrix (symmetric)
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    for i in range(n):
        adj[i][i] = True  # self is always adjacent

    # Create Gurobi model
    model = gp.Model("Max_Weighted_Clique")
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-4
    model.Params.LogToConsole = 0

    # Binary variables
    x = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(n)]

    # Objective
    obj = gp.quicksum(vertex_weights[i] * x[i] for i in range(n))
    model.setObjective(obj, GRB.MAXIMIZE)

    # Clique constraints: for every pair of non-adjacent vertices, at most one can be selected
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                model.addConstr(x[i] + x[j] <= 1, name=f"no_edge_{i}_{j}")

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(float(obj_val))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        clique_vertices = [i for i in range(n) if x[i].X > 0.5]
    else:
        # No solution found, return empty
        objective_value = 0.0
        clique_vertices = []

    # Write solution
    solution = {
        "objective_value": float(objective_value),
        "clique_vertices": clique_vertices
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()