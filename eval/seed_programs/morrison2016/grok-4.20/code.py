import argparse
import json
import time
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    graph = instance['graph']
    bounds = instance.get('bounds', {})
    n = graph['num_vertices']
    edges = graph['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    best_k = bounds.get('greedy_upper_bound', n)
    best_coloring = None
    best_sets = None
    
    # Try to find good colorings with increasing k
    # Start from lower bound if available
    lower = bounds.get('clique_lower_bound', 1)
    k = lower
    
    # Use a simple but effective greedy coloring first to get a good upper bound
    def greedy_coloring():
        color = [-1] * n
        for v in range(n):
            used = set()
            for nei in adj[v]:
                if color[nei] != -1:
                    used.add(color[nei])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        num_colors = max(color) + 1
        sets = [[] for _ in range(num_colors)]
        for v in range(n):
            sets[color[v]].append(v)
        for s in sets:
            s.sort()
        return num_colors, color, sets
    
    # Run greedy to get initial upper bound
    init_k, init_color, init_sets = greedy_coloring()
    if init_k < best_k:
        best_k = init_k
        best_coloring = init_color
        best_sets = init_sets
        if logger:
            logger.log(best_k)
    
    # Also try ordering by degree for better greedy
    degrees = [(len(adj[i]), i) for i in range(n)]
    degrees.sort(reverse=True)
    order = [v for _, v in degrees]
    color = [-1] * n
    for v in order:
        used = set()
        for nei in adj[v]:
            if color[nei] != -1:
                used.add(color[nei])
        c = 0
        while c in used:
            c += 1
        color[v] = c
    num_c = max(color) + 1
    if num_c < best_k:
        best_k = num_c
        sets = [[] for _ in range(num_c)]
        for v in range(n):
            sets[color[v]].append(v)
        for s in sets:
            s.sort()
        best_coloring = color[:]
        best_sets = sets
        if logger:
            logger.log(best_k)
    
    # If we have a very small graph or tight bounds, try exact MIP
    time_left = args.time_limit - (time.time() - start_time)
    if time_left < 5:
        # Just return best found so far
        pass
    elif n <= 50 or (best_k - lower <= 3 and n <= 80) or (best_k <= 5 and n <= 120):
        # Try MIP formulation with k = best_k - 1 down to lower
        for target_k in range(best_k - 1, lower - 1, -1):
            time_left = args.time_limit - (time.time() - start_time)
            if time_left < 3:
                break
            
            try:
                model = gp.Model("GraphColoring")
                model.setParam(GRB.Param.TimeLimit, time_left - 1)
                model.setParam(GRB.Param.Threads, 1)
                model.setParam(GRB.Param.LogToConsole, 0)
                
                # x[v,c] = 1 if vertex v gets color c
                x = model.addVars(n, target_k, vtype=GRB.BINARY, name="x")
                
                # Each vertex gets exactly one color
                for v in range(n):
                    model.addConstr(gp.quicksum(x[v, c] for c in range(target_k)) == 1)
                
                # Adjacent vertices different colors
                for u, v in edges:
                    if u < v:  # avoid duplicates
                        for c in range(target_k):
                            model.addConstr(x[u, c] + x[v, c] <= 1)
                
                # Objective: minimize k is fixed, but we can use dummy objective
                model.setObjective(0, GRB.MINIMIZE)
                
                # Symmetry breaking: color 0 used by lowest index vertex
                model.addConstr(x[0, 0] == 1)
                
                # Try to solve
                model.optimize()
                
                if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT and model.SolCount > 0:
                    if model.status == GRB.OPTIMAL:
                        new_k = target_k
                    else:
                        # Check if feasible solution found
                        if hasattr(model, 'objVal') and model.SolCount > 0:
                            sol = model.getVars()
                            feasible = True
                            color = [-1] * n
                            for v in range(n):
                                assigned = False
                                for c in range(target_k):
                                    if x[v, c].X > 0.5:
                                        color[v] = c
                                        assigned = True
                                        break
                                if not assigned:
                                    feasible = False
                            if feasible:
                                new_k = target_k
                                sets = [[] for _ in range(new_k)]
                                for v in range(n):
                                    sets[color[v]].append(v)
                                for s in sets:
                                    s.sort()
                                best_k = new_k
                                best_coloring = color
                                best_sets = sets
                                if logger:
                                    logger.log(best_k)
                            continue
                    
                    # Extract solution
                    color = [-1] * n
                    for v in range(n):
                        for c in range(target_k):
                            if x[v, c].X > 0.5:
                                color[v] = c
                                break
                    
                    sets = [[] for _ in range(target_k)]
                    for v in range(n):
                        sets[color[v]].append(v)
                    for s in sets:
                        s.sort()
                    
                    best_k = target_k
                    best_coloring = color
                    best_sets = sets
                    if logger:
                        logger.log(best_k)
                    
                    # Since we found a better one, we can try even smaller
                    continue
                else:
                    # Infeasible, increase lower bound
                    lower = max(lower, target_k + 1)
                    break
            except:
                break
    else:
        # For larger graphs, run multiple greedy heuristics with random ordering
        import random
        best_color_list = best_coloring
        for _ in range(30):
            time_left = args.time_limit - (time.time() - start_time)
            if time_left < 0.5:
                break
            ordering = list(range(n))
            random.shuffle(ordering)
            color = [-1] * n
            for v_idx in ordering:
                used = set()
                for nei in adj[v_idx]:
                    if color[nei] >= 0:
                        used.add(color[nei])
                c = 0
                while c in used:
                    c += 1
                color[v_idx] = c
            num_colors = max(color) + 1
            if num_colors < best_k:
                best_k = num_colors
                sets = [[] for _ in range(num_colors)]
                for v in range(n):
                    sets[color[v]].append(v)
                for s in sets:
                    s.sort()
                best_coloring = color[:]
                best_sets = sets
                if logger:
                    logger.log(best_k)
    
    # Final output
    if best_coloring is None:
        # Fallback: assign each vertex its own color
        best_k = n
        best_coloring = list(range(n))
        best_sets = [[i] for i in range(n)]
    
    # Build coloring dict
    coloring_dict = {}
    for v in range(n):
        coloring_dict[str(v)] = int(best_coloring[v])
    
    solution = {
        "objective_value": int(best_k),
        "selected_sets": best_sets,
        "coloring": coloring_dict
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()