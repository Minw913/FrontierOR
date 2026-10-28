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

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_vertices = instance['num_vertices']
    vertices = instance['vertices']
    edges_data = instance['edges']

    # Map vertex labels to 0-based indices if needed, but assume labels == 0..n-1 or use dict
    vertex_set = set(vertices)
    edge_dict = {}
    for e in edges_data:
        i, j, w = e['i'], e['j'], e['weight']
        # Ensure i < j
        if i > j:
            i, j = j, i
        edge_dict[(i, j)] = w

    # Create model
    model = gp.Model("MaxCliqueWeight")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)

    # Binary variables for vertices
    x = {}
    for v in vertices:
        x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")

    # Objective: sum of weights of edges in the clique
    obj_expr = gp.LinExpr()
    edge_vars = {}
    for (i, j), w in edge_dict.items():
        if i not in vertex_set or j not in vertex_set:
            continue
        # y_ij <= x_i, y_ij <= x_j  ->  y_ij = 1 only if both are selected
        y = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
        edge_vars[(i, j)] = y
        obj_expr.add(y, w)
        model.addConstr(y <= x[i])
        model.addConstr(y <= x[j])

    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Clique constraints: for every pair of vertices with NO edge, cannot both be selected
    # To enforce that selected vertices form a clique
    for i_idx in range(len(vertices)):
        for j_idx in range(i_idx + 1, len(vertices)):
            u = vertices[i_idx]
            v = vertices[j_idx]
            if (u, v) not in edge_dict and (v, u) not in edge_dict:
                model.addConstr(x[u] + x[v] <= 1)

    # Callback to log improved solutions
    best_obj = -float('inf')
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val > best_obj + 1e-6:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        clique = []
        for v in vertices:
            if x[v].X > 0.5:
                clique.append(v)
        objective_value = model.objVal if model.objVal > 0 else 0.0
    else:
        # No feasible solution
        clique = []
        objective_value = 0.0

    # If no edges and isolated vertices, objective is 0
    if len(edge_dict) == 0:
        objective_value = 0.0
        clique = [vertices[0]] if vertices else []

    # Write solution
    solution = {
        "objective_value": float(objective_value),
        "clique": [int(v) for v in sorted(clique)]
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # Final log of best
    if logger and objective_value > 0:
        logger.log(float(objective_value))

if __name__ == "__main__":
    main()