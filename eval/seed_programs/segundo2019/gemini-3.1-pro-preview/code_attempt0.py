import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB

# Attempt to import SolutionLogger; fallback if not available in testing environment manually.
try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, *args, **kwargs):
            pass
        def log(self, *args, **kwargs):
            pass

def main():
    parser = argparse.ArgumentParser(description="MEWCP solver using Gurobi MILP formulation.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where the output solution JSON must be written.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to a JSONL file for logging intermediate solutions.")
    args = parser.parse_args()

    start_time = time.time()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    graph = instance['graph']
    vertices = graph['vertices']
    edges = graph['edges']

    # Standardize edge representations and accumulate weights to handle any duplicate edges simply by summing weights.
    edge_weights = {}
    for u, v, w in edges:
        curr_u, curr_v = (u, v) if u < v else (v, u)
        if (curr_u, curr_v) not in edge_weights:
            edge_weights[(curr_u, curr_v)] = 0.0
        edge_weights[(curr_u, curr_v)] += w

    # Identify non-edges to formulate the independent set constraints over complement graph
    non_edges = []
    n = len(vertices)
    for i in range(n):
        u_i = vertices[i]
        for j in range(i + 1, n):
            v_j = vertices[j]
            curr_u, curr_v = (u_i, v_j) if u_i < v_j else (v_j, u_i)
            if (curr_u, curr_v) not in edge_weights:
                non_edges.append((curr_u, curr_v))

    # Initialize Gurobi API footprint
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()

    model = gp.Model("MEWCP", env=env)
    
    # Restrict to a single CPU core
    model.setParam('Threads', 1)

    # Automatically adapt TimeLimit deducting parsing/formulation elapsed time
    elapsed = time.time() - start_time
    rem_time = args.time_limit - elapsed
    model.setParam('TimeLimit', max(1, rem_time))

    # Construct the MILP variables
    x = {}
    for v in vertices:
        x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")

    y = {}
    for (u, v), w in edge_weights.items():
        y[(u, v)] = model.addVar(vtype=GRB.BINARY, obj=w, name=f"y_{u}_{v}")
        
    # Construct model constraints
    # y <= x_u and y <= x_v strictly force the inclusion of an edge to conditionally require both endpoints.
    if y:
        model.addConstrs((y[k] <= x[k[0]] for k in edge_weights.keys()), name="bound_u")
        model.addConstrs((y[k] <= x[k[1]] for k in edge_weights.keys()), name="bound_v")

    # Clique enforcement -> complement graph pairs sum to at most 1, avoiding selecting disconnected vertices
    if non_edges:
        model.addConstrs((x[u] + x[v] <= 1 for u, v in non_edges), name="clique")

    # Objective is to maximize sum of weights in clique edges
    model.ModelSense = GRB.MAXIMIZE

    # Register intermediate solution logger instance
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    class CallbackData:
        def __init__(self):
            # Best valid starting objective for logging
            self.best_obj = -1e15

    cb_data = CallbackData()

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj > cb_data.best_obj + 1e-5:
                cb_data.best_obj = obj
                if logger:
                    logger.log(obj)

    # Perform solving operation
    model.optimize(callback)

    # Extract final best solutions found within constraints/time limit
    if model.SolCount > 0:
        clique_vertices = [v for v in vertices if x[v].X > 0.5]
        clique_vertices.sort()

        clique_edges = []
        for (u, v), var in y.items():
            if var.X > 0.5:
                clique_edges.append([u, v])
        clique_edges.sort(key=lambda e: (e[0], e[1]))

        obj_val = float(model.ObjVal)
    else:
        # Fallback to essentially an empty setup if no nodes are possible
        clique_vertices = []
        clique_edges = []
        obj_val = 0.0

    # Package output according to solution schema specification
    solution_json = {
        "objective_value": obj_val,
        "clique_vertices": clique_vertices,
        "clique_edges": clique_edges
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution_json, f, indent=2)

if __name__ == "__main__":
    main()