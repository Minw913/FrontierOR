import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve_maxcut(instance, time_limit, log_path):
    # Parse parameters and data
    params = instance.get("parameters", {})
    data = instance.get("data", {})
    
    # Determine number of nodes
    if "n_nodes" in params:
        n = params["n_nodes"]
    elif "num_nodes" in params:
        n = params["num_nodes"]
    else:
        edges = data.get("edges", [])
        if edges:
            n = max(max(u, v) for u, v in edges) + 1
        else:
            n = 0
            
    # Determine edges
    edges = data.get("edges", [])
    if not edges and "adjacency_matrix" in data:
        adj = data["adjacency_matrix"]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] > 0:
                    edges.append([i, j])
                    
    logger = SolutionLogger(log_path, sense="maximize") if (SolutionLogger and log_path) else None
    
    m = gp.Model()
    m.setParam("Seed", 0)
    m.setParam("MIPGap", 1e-4)
    m.setParam("NumericFocus", 0)
    m.setParam("Threads", 1)
    if time_limit:
        m.setParam("TimeLimit", time_limit)
        
    # Decision variables for MaxCut formulation
    x = m.addVars(n, vtype=GRB.BINARY, name="x")
    z = m.addVars(len(edges), vtype=GRB.BINARY, name="z")
    
    # MaxCut constraints (Linearization of XOR for each edge)
    for idx, edge in enumerate(edges):
        u, v = edge[0], edge[1]
        m.addConstr(z[idx] <= x[u] + x[v])
        m.addConstr(z[idx] <= 2 - x[u] - x[v])
        
    m.setObjective(gp.quicksum(z[idx] for idx in range(len(edges))), GRB.MAXIMIZE)
    
    def maxcut_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                vals = model.cbGetSolution(x)
                sol_dict = {f"x_{i}": int(round(vals[i])) for i in range(n)}
                logger.log_solution(obj, sol_dict)
                
    m.optimize(maxcut_callback)
    
    # Extract final solution based on Schema
    if m.SolCount > 0:
        sol_dict = {f"x_{i}": int(round(x[i].X)) for i in range(n)}
        obj = m.ObjVal
        return {"objective_value": obj, "solution": sol_dict}
    else:
        return {"objective_value": 0.0, "solution": {f"x_{i}": 0 for i in range(n)}}

def solve_tsp(instance, time_limit, log_path):
    params = instance.get("parameters", {})
    data = instance.get("data", {})
    
    n = params.get("n_cities", 0)
    if n == 0:
        n = len(data.get("coordinates", []))
    dist = data.get("distance_matrix", [])
    
    if n == 1:
        return {"objective_value": 0.0, "solution": {"x_0_0": 1}, "tour": [0]}
        
    logger = SolutionLogger(log_path, sense="minimize") if (SolutionLogger and log_path) else None
    
    m = gp.Model()
    m.setParam("Seed", 0)
    m.setParam("MIPGap", 1e-4)
    m.setParam("NumericFocus", 0)
    m.setParam("Threads", 1)
    if time_limit:
        m.setParam("TimeLimit", time_limit)
        
    # Decision variables for directed edges in tour
    vars = m.addVars(n, n, vtype=GRB.BINARY, name='e')
    for i in range(n):
        m.addConstr(vars.sum(i, '*') == 1)  # out-degree = 1
        m.addConstr(vars.sum('*', i) == 1)  # in-degree = 1
        vars[i, i].ub = 0                   # Disallow self-loops
        
    m.setObjective(gp.quicksum(dist[i][j] * vars[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
    
    # Enable lazy constraints for Subtour Elimination
    m.Params.LazyConstraints = 1
    
    def tsp_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                vals = model.cbGetSolution(vars)
                edges_dict = {}
                for i in range(n):
                    for j in range(n):
                        if vals[i, j] > 0.5:
                            edges_dict[i] = j
                
                # Find all disjoint subtours
                tours = []
                unvisited = set(range(n))
                while unvisited:
                    start = unvisited.pop()
                    tour = [start]
                    curr = edges_dict.get(start, start)
                    while curr != start and curr in unvisited:
                        unvisited.remove(curr)
                        tour.append(curr)
                        curr = edges_dict.get(curr, start)
                    tours.append(tour)
                    
                if not tours:
                    return
                    
                tours.sort(key=len)
                shortest = tours[0]
                
                # If a true subtour exists, eliminate it lazily
                if len(shortest) < n:
                    model.cbLazy(gp.quicksum(vars[i, j] for i in shortest for j in shortest if i != j) <= len(shortest) - 1)
                else:
                    # Valid continuous Hamiltonian cycle found
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if logger:
                        sol_dict = {}
                        for t in range(n):
                            city = shortest[t]
                            sol_dict[f"x_{city}_{t}"] = 1
                        for i in range(n):
                            for t in range(n):
                                if f"x_{i}_{t}" not in sol_dict:
                                    sol_dict[f"x_{i}_{t}"] = 0
                        logger.log_solution(obj, sol_dict)
            except Exception:
                pass
                
    m.optimize(tsp_callback)
    
    # Extract final optimal permutation mapping based on Schema
    if m.SolCount > 0:
        edges = [(i, j) for i in range(n) for j in range(n) if vars[i, j].X > 0.5]
        edges_dict = {i: j for i, j in edges}
        
        tour = [0]
        curr = edges_dict.get(0, 0)
        while curr != 0 and len(tour) < n:
            tour.append(curr)
            curr = edges_dict.get(curr, 0)
            
        sol_dict = {}
        for t in range(n):
            if t < len(tour):
                city = tour[t]
                sol_dict[f"x_{city}_{t}"] = 1
                
        # Fill non-selected steps with 0 to make it dense
        for i in range(n):
            for t in range(n):
                if f"x_{i}_{t}" not in sol_dict:
                    sol_dict[f"x_{i}_{t}"] = 0
                    
        obj = m.ObjVal
        return {"objective_value": obj, "solution": sol_dict, "tour": tour}
    else:
        return {"objective_value": float('inf'), "solution": {}, "tour": []}

def main():
    parser = argparse.ArgumentParser(description="Solve Optimization Problem (MaxCut or TSP)")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default="", help="Path to log intermediate solutions")
    
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
        
    params = instance.get("parameters", {})
    # Determine if the problem is TSP logically based on parameters dict
    if "n_cities" in params or "coordinate_range" in params:
        result = solve_tsp(instance, args.time_limit, args.log_path)
    else:
        result = solve_maxcut(instance, args.time_limit, args.log_path)
        
    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()