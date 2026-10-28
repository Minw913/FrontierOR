import json
import argparse
import time
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_vertices']
    edges = instance['edges']
    adj = instance['adjacency_list']
    greedy_size = instance['greedy_clique_size']
    greedy_clique = instance['greedy_clique']

    # Build set of non-edges for constraints (u < v)
    non_edges = set()
    for i in range(n):
        neighbors = set(adj[str(i)])
        for j in range(i+1, n):
            if j not in neighbors:
                non_edges.add((i, j))

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Record greedy solution immediately
    best_obj = greedy_size
    best_clique = greedy_clique[:]
    if logger:
        logger.log(best_obj)

    model = gp.Model("MaxClique")
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.MIPGap = 0.0
    model.Params.LogToConsole = 0

    x = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in range(n)]

    # Objective
    model.setObjective(gp.quicksum(x), GRB.MAXIMIZE)

    # Constraints: for every non-edge, at most one vertex selected
    for u, v in non_edges:
        model.addConstr(x[u] + x[v] <= 1)

    # Warm start with greedy clique
    for v in range(n):
        x[v].setAttr("Start", 1 if v in greedy_clique else 0)
    model.setAttr("StartNumber", 1)

    # Callback to log improved solutions
    def callback(model, where):
        nonlocal best_obj, best_clique
        if where == GRB.Callback.MIPSOL:
            obj_val = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
            if obj_val > best_obj:
                best_obj = obj_val
                sol = model.cbGetSolution(x)
                clique = [i for i in range(n) if sol[i] > 0.5]
                best_clique = clique
                if logger:
                    logger.log(best_obj)

    model.optimize(callback)

    # If optimization finished or time expired, get best solution
    if model.SolCount > 0:
        obj_val = int(round(model.ObjVal))
        if obj_val > best_obj:
            best_obj = obj_val
            best_clique = [i for i in range(n) if x[i].X > 0.5]
            if logger:
                logger.log(best_obj)

    # Final output
    solution = {
        "objective_value": float(best_obj),
        "clique_vertices": best_clique
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()