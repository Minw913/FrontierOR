import json
import argparse
import time
from typing import List, Dict, Any
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path: str) -> Dict[str, Any]:
    with open(instance_path, 'r') as f:
        return json.load(f)

def write_solution(solution_path: str, objective_value: float, visited_vertices: List[int], tour_edges: List[List[int]]):
    solution = {
        "objective_value": float(objective_value),
        "visited_vertices": visited_vertices,
        "tour_edges": tour_edges
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

def solve_op(instance: Dict[str, Any], time_limit: int, logger: SolutionLogger = None) -> Dict[str, Any]:
    n = instance["num_nodes"]
    D = instance["distance_limitation_d0"]
    depot = instance["depot_vertex"]
    scores = instance["scores"]
    dist = instance["distance_matrix"]
    
    # Filter nodes with positive score (depot has 0)
    positive_nodes = [i for i in range(n) if scores[i] > 0]
    N = len(positive_nodes)
    node_map = {positive_nodes[i]: i for i in range(N)}
    if depot not in positive_nodes:
        positive_nodes.append(depot)
        node_map[depot] = N
        N += 1
    
    # Precompute distances and scores for selected nodes
    node_list = positive_nodes
    c = [[dist[i][j] for j in node_list] for i in node_list]
    p = [scores[i] for i in node_list]
    depot_idx = node_map[depot]
    
    # Model
    m = gp.Model("PrizeCollectingTSP")
    m.Params.TimeLimit = time_limit
    m.Params.Threads = 1
    m.Params.MIPGap = 0.01  # Allow some gap for speed
    
    # Variables
    x = m.addVars(N, N, vtype=GRB.BINARY, name="x")
    y = m.addVars(N, vtype=GRB.BINARY, name="y")
    
    # Objective: maximize total prize (excluding depot)
    obj = gp.quicksum(p[i] * y[i] for i in range(N) if i != depot_idx)
    m.setObjective(obj, GRB.MAXIMIZE)
    
    # Degree constraints
    for i in range(N):
        if i == depot_idx:
            m.addConstr(gp.quicksum(x[i, j] for j in range(N) if j != i) == 2 * y[i])
            m.addConstr(gp.quicksum(x[j, i] for j in range(N) if j != i) == 2 * y[i])
        else:
            m.addConstr(gp.quicksum(x[i, j] for j in range(N) if j != i) == y[i])
            m.addConstr(gp.quicksum(x[j, i] for j in range(N) if j != i) == y[i])
    
    # Distance budget
    m.addConstr(gp.quicksum(c[i][j] * x[i, j] for i in range(N) for j in range(N) if i != j) <= D)
    
    # No self-loops
    for i in range(N):
        m.addConstr(x[i, i] == 0)
    
    # Subtour elimination (MTZ formulation)
    u = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, ub=N, name="u")
    for i in range(N):
        if i != depot_idx:
            m.addConstr(u[i] >= 1)
            m.addConstr(u[i] <= N - 1)
    
    for i in range(N):
        for j in range(N):
            if i != depot_idx and j != depot_idx and i != j:
                m.addConstr(u[i] - u[j] + N * x[i, j] <= N - 1)
    
    # Symmetry breaking: force first node in some order if used
    if N > 2:
        m.addConstr(y[0] >= y[1])  # arbitrary
    
    # Callback for logging improved solutions
    best_obj = 0
    solution_found = False
    best_visited = []
    best_edges = []
    
    def callback(model, where):
        nonlocal best_obj, solution_found, best_visited, best_edges
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val > best_obj + 1e-6:
                best_obj = obj_val
                # Extract solution
                y_val = model.cbGetSolution(y)
                x_val = model.cbGetSolution(x)
                visited = [node_list[i] for i in range(N) if y_val[i] > 0.5]
                edges = []
                for i in range(N):
                    for j in range(N):
                        if i < j and x_val[i, j] > 0.5:
                            edges.append([node_list[i], node_list[j]])
                        elif j < i and x_val[j, i] > 0.5:
                            edges.append([node_list[j], node_list[i]])
                if logger:
                    logger.log(float(obj_val))
                solution_found = True
                best_visited = sorted(visited)
                best_edges = sorted(edges, key=lambda e: (e[0], e[1]))
    
    m.optimize(callback)
    
    # If no solution found via callback, extract from final model
    if not solution_found and m.SolCount > 0:
        obj_val = m.objVal
        y_val = m.getAttr('X', y)
        x_val = m.getAttr('X', x)
        visited = [node_list[i] for i in range(N) if y_val[i] > 0.5]
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if x_val[i, j] > 0.5 or x_val[j, i] > 0.5:
                    edges.append([node_list[i], node_list[j]])
        best_obj = obj_val
        best_visited = sorted(visited)
        best_edges = sorted(edges, key=lambda e: (e[0], e[1]))
    
    # If still no solution, return trivial solution: just the depot
    if not solution_found and m.SolCount == 0:
        best_obj = 0.0
        best_visited = [depot]
        best_edges = []
    
    return {
        "objective_value": best_obj,
        "visited_vertices": best_visited,
        "tour_edges": best_edges
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    start_time = time.time()
    instance = read_instance(args.instance_path)
    result = solve_op(instance, args.time_limit, logger)
    
    # Ensure we don't exceed time
    elapsed = time.time() - start_time
    if elapsed > args.time_limit:
        print("Warning: Solution took longer than time limit")
    
    write_solution(
        args.solution_path,
        result["objective_value"],
        result["visited_vertices"],
        result["tour_edges"]
    )

if __name__ == "__main__":
    main()