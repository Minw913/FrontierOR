import json
import argparse
import time
import sys
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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance['graph']
    n = graph['num_vertices']
    edges = graph['edges']
    
    if n == 0:
        result = {
            "objective_value": 0,
            "coloring": {},
            "color_classes": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(result, f, indent=2)
        return

    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Greedy coloring for upper bound and initial solution
    def greedy_color(order):
        color = [-1] * n
        max_c = 0
        for v in order:
            used = set()
            for nei in adj[v]:
                if color[nei] != -1:
                    used.add(color[nei])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            max_c = max(max_c, c)
        return color, max_c + 1
    
    # Multiple greedy heuristics
    orders = [
        list(range(n)),  # natural
        sorted(range(n), key=lambda v: -len(adj[v])),  # largest degree first
        sorted(range(n), key=lambda v: len(adj[v])),   # smallest degree first
    ]
    
    best_coloring = None
    best_k = n + 1
    for order in orders:
        coloring, k = greedy_color(order)
        if k < best_k:
            best_k = k
            best_coloring = coloring
            if logger:
                logger.log(best_k)
    
    # Try to improve with simple local search (DSATUR-like)
    def dsatur():
        color = [-1] * n
        sat = [0] * n
        deg = [len(adj[i]) for i in range(n)]
        uncolored = set(range(n))
        max_c = 0
        
        # Color highest degree first
        v = max(range(n), key=lambda x: deg[x])
        color[v] = 0
        uncolored.remove(v)
        
        for nei in adj[v]:
            sat[nei] = 1
        
        while uncolored:
            # Select vertex with highest saturation, break ties by degree
            best_v = None
            best_sat = -1
            best_deg = -1
            for u in uncolored:
                s = sat[u]
                d = deg[u]
                if (s > best_sat) or (s == best_sat and d > best_deg):
                    best_sat = s
                    best_deg = d
                    best_v = u
            v = best_v
            
            used = set()
            for nei in adj[v]:
                if color[nei] != -1:
                    used.add(color[nei])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            max_c = max(max_c, c)
            uncolored.remove(v)
            
            for nei in adj[v]:
                if color[nei] == -1:
                    # Update saturation if this is a new color
                    neighbor_used = {color[x] for x in adj[nei] if color[x] != -1}
                    sat[nei] = len(neighbor_used)
        
        return color, max_c + 1
    
    dsat_color, dsat_k = dsatur()
    if dsat_k < best_k:
        best_k = dsat_k
        best_coloring = dsat_color
        if logger:
            logger.log(best_k)
    
    incumbent_k = best_k
    incumbent_coloring = best_coloring[:]
    
    # Now use Gurobi with formulation that tries to improve chromatic number
    # We will run MIP with decreasing k until time runs out or optimality proven
    start_time = time.time()
    time_limit = args.time_limit
    remaining_time = max(1, time_limit - (time.time() - start_time))
    
    # Upper bound on chromatic number from greedy
    current_best = incumbent_k
    
    # Try to find better colorings with MIP using k = current_best-1 down to 1
    for target_k in range(current_best - 1, 0, -1):
        elapsed = time.time() - start_time
        if elapsed >= time_limit - 2:
            break
        mip_time = max(5, int(time_limit - elapsed - 2))
        
        try:
            m = gp.Model("GraphColoring")
            m.setParam(GRB.Param.TimeLimit, mip_time)
            m.setParam(GRB.Param.Threads, 1)
            m.setParam(GRB.Param.LogToConsole, 0)
            m.setParam(GRB.Param.MIPFocus, 1)  # focus on feasible solutions
            m.setParam(GRB.Param.Heuristics, 0.8)
            
            # x[v,c] = 1 if vertex v gets color c
            x = m.addVars(n, target_k, vtype=GRB.BINARY, name="x")
            
            # Each vertex gets exactly one color
            for v in range(n):
                m.addConstr(gp.quicksum(x[v, c] for c in range(target_k)) == 1)
            
            # Adjacent vertices different colors
            for u, v in edges:
                if u < v:  # avoid duplicate
                    for c in range(target_k):
                        m.addConstr(x[u, c] + x[v, c] <= 1)
            
            # Objective: minimize max color used (but since we fix k, we just feasibility)
            # Use a dummy objective
            m.setObjective(0, GRB.MINIMIZE)
            
            # Start with greedy solution if possible (warm start)
            if incumbent_coloring is not None and max(incumbent_coloring) < target_k:
                for v in range(n):
                    c = incumbent_coloring[v]
                    if c < target_k:
                        x[v, c].start = 1.0
                    else:
                        x[v, 0].start = 1.0
            
            m.optimize()
            
            if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT and m.SolCount > 0:
                if m.objVal is not None or (m.status == GRB.OPTIMAL):
                    # Extract solution
                    new_coloring = [-1] * n
                    for v in range(n):
                        for c in range(target_k):
                            if x[v, c].X > 0.5:
                                new_coloring[v] = c
                                break
                    new_k = max(new_coloring) + 1
                    
                    # Verify feasibility
                    feasible = True
                    for u, v in edges:
                        if new_coloring[u] == new_coloring[v]:
                            feasible = False
                            break
                    if feasible and new_k < incumbent_k:
                        incumbent_k = new_k
                        incumbent_coloring = new_coloring
                        if logger:
                            logger.log(incumbent_k)
                        # We can break early since we go from high k to low
                        break
                        
        except Exception:
            # If Gurobi fails (license, memory, etc.), just continue with heuristic solution
            pass
        
        # Check remaining time
        if time.time() - start_time > time_limit - 3:
            break
    
    # Final best solution
    final_coloring = incumbent_coloring
    final_k = max(final_coloring) + 1 if final_coloring else 0
    
    # Build output structures
    coloring_dict = {}
    color_classes = {}
    for v, c in enumerate(final_coloring):
        coloring_dict[str(v)] = int(c)
        if str(c) not in color_classes:
            color_classes[str(c)] = []
        color_classes[str(c)].append(v)
    
    result = {
        "objective_value": int(final_k),
        "coloring": coloring_dict,
        "color_classes": color_classes
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()