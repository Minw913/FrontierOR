import sys
import json
import random
import time
import argparse
import gurobipy as gp
from gurobipy import GRB

# Attempt to import SolutionLogger as requested in the requirements.
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve_max_clique(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
        
    graph_data = data["graph"]
    num_vertices = graph_data["num_vertices"]
    num_edges = graph_data["num_edges"]
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path and SolutionLogger else None
    
    if num_vertices == 0:
        if logger:
            logger.log(0)
        sol = {
            "objective_value": 0,
            "clique_vertices": []
        }
        with open(solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return

    # Parse adjacency list and ensure integer keys/values
    adj_set = {}
    for k, v in graph_data["adjacency_list"].items():
        adj_set[int(k)] = set(v)
        
    vertices = list(adj_set.keys())
    
    # ----------------------------------------------------
    # Heuristic phase: Construct strong initial solutions
    # ----------------------------------------------------
    best_clique = []
    
    def try_update(clique):
        nonlocal best_clique
        if len(clique) > len(best_clique):
            best_clique = clique.copy()
            if logger:
                logger.log(len(best_clique))

    # 1. Standard greedy (highest degree first)
    sorted_v_deg = sorted(vertices, key=lambda v: len(adj_set[v]), reverse=True)
    c1 = []
    for v in sorted_v_deg:
        if all(u in adj_set[v] for u in c1):
            c1.append(v)
    try_update(c1)
    
    random.seed(42)
    heuristic_time_start = time.time()
    
    # 2. Targeted seed heuristics: start from top 200 highest-degree vertices
    for start_v in sorted_v_deg[:min(num_vertices, 200)]:
        if time.time() - heuristic_time_start > min(1.0, time_limit * 0.03):
            break
        c_v = [start_v]
        # Evaluate neighbors by picking ones with generally higher degrees
        neighbors = sorted(list(adj_set[start_v]), key=lambda u: len(adj_set[u]), reverse=True)
        for neighbor in neighbors:
            if all(u in adj_set[neighbor] for u in c_v):
                c_v.append(neighbor)
        try_update(c_v)

    # 3. Randomized greedy exploration
    while time.time() - heuristic_time_start < min(2.5, time_limit * 0.05):
        c_rand = []
        rand_v = vertices[:]
        random.shuffle(rand_v)
        # Occasionally bias with degree + slight random noise
        if random.random() < 0.5:
            rand_v.sort(key=lambda v: len(adj_set[v]) + random.randint(0, 5), reverse=True)
            
        for v in rand_v:
            if all(u in adj_set[v] for u in c_rand):
                c_rand.append(v)
        try_update(c_rand)

    # ----------------------------------------------------
    # Exact phase: Gurobi Model creation & solving
    # ----------------------------------------------------
    time_left = time_limit - (time.time() - start_time)
    if time_left <= 0:
        sol = {
            "objective_value": len(best_clique),
            "clique_vertices": sorted(best_clique)
        }
        with open(solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return

    # Decide explicit vs lazy setup based on non-edge graph density. 
    # If explicit constraints overwhelm realistic Python memory, we drop to pure lazy.
    num_non_edges = num_vertices * (num_vertices - 1) // 2 - num_edges
    use_lazy = (num_non_edges > 5_000_000)

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    
    model = gp.Model("max_clique", env=env)
    model.setParam("TimeLimit", max(1.0, time_left))
    model.setParam("Threads", 1)  # Strict container single core limitation requirement
    
    if use_lazy:
        model.setParam("LazyConstraints", 1)
        
    # Variables representing whether vertex v is in the clique
    x = model.addVars(vertices, vtype=GRB.BINARY, name="x")
    model.setObjective(x.sum(), GRB.MAXIMIZE)
    
    # Guide Gurobi with our best heuristic sequence
    for v in best_clique:
        x[v].Start = 1.0
    for v in vertices:
        if v not in best_clique:
            x[v].Start = 0.0

    # Add Clique Cover / Graph Coloring inequalities on the complement graph.
    # This acts as an extraordinarily tight bounding constraint for Max Clique LP.
    coloring_time_start = time.time()
    num_colorings = 10 if num_vertices < 5000 else 3
    added_ind_sets = set()
    for i in range(num_colorings):
        if time.time() - coloring_time_start > min(5.0, time_limit * 0.1):
            break
        colors = {}
        sorted_v = vertices[:]
        random.shuffle(sorted_v)
        if i == 0:
            sorted_v = sorted_v_deg  # Fallback to degree distribution for at least one robust coloring
        
        for v in sorted_v:
            used = {colors[u] for u in adj_set[v] if u in colors}
            c = 0
            while c in used:
                c += 1
            colors[v] = c
        
        classes = {}
        for v, c in colors.items():
            classes.setdefault(c, []).append(v)
            
        for c_class in classes.values():
            if len(c_class) > 1:
                t_class = tuple(sorted(c_class))
                if t_class not in added_ind_sets:
                    added_ind_sets.add(t_class)
                    model.addConstr(gp.quicksum(x[v] for v in c_class) <= 1)

    # Supply exact pairwise limits if memory scaling permits it
    if not use_lazy:
        non_edges = []
        for i in range(len(vertices)):
            vi = vertices[i]
            for j in range(i+1, len(vertices)):
                vj = vertices[j]
                if vj not in adj_set[vi]:
                    non_edges.append((vi, vj))
        
        if non_edges:
            model.addConstrs((x[u] + x[v] <= 1 for (u, v) in non_edges), name="ne")
            
    # Setup state for standard MIPSOL callback tracking
    model._best_obj = len(best_clique)
    model._logger = logger
    model._adj_set = adj_set
    model._vertices = vertices
    model._x = x
    model._use_lazy = use_lazy

    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            # Reconstruct the true clique state from intermediate relaxation branch jumps
            sol = cb_model.cbGetSolution(cb_model._x)
            chosen = [v for v in cb_model._vertices if sol[v] > 0.5]
            
            violated = False
            if cb_model._use_lazy:
                # Dynamically locate structural conflict loops inside Gurobi node
                for i in range(len(chosen)):
                    vi = chosen[i]
                    for j in range(i+1, len(chosen)):
                        vj = chosen[j]
                        if vj not in cb_model._adj_set[vi]:
                            cb_model.cbLazy(cb_model._x[vi] + cb_model._x[vj] <= 1)
                            violated = True
            
            if not violated:
                # Properly validate an internal discovery
                obj = len(chosen)
                if obj > cb_model._best_obj:
                    cb_model._best_obj = obj
                    if cb_model._logger:
                        cb_model._logger.log(obj)

    # Execute
    model.optimize(callback)
    
    # ----------------------------------------------------
    # Solution evaluation & saving routines
    # ----------------------------------------------------
    final_clique = best_clique
    if model.SolCount > 0:
        try:
            chosen = [v for v in vertices if x[v].X > 0.5]
            
            valid = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    if chosen[j] not in adj_set[chosen[i]]:
                        valid = False
                        break
                if not valid:
                    break
                    
            if valid and len(chosen) > len(best_clique):
                final_clique = chosen
        except gp.GurobiError:
            pass

    sol = {
        "objective_value": len(final_clique),
        "clique_vertices": sorted(final_clique)
    }
    
    with open(solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Solve Maximum Clique Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to parsing instance schema file JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where model stores finalized solution")
    parser.add_argument("--time_limit", type=int, required=True, help="Available processor time in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Utility module intermediate execution progress logger stream")
    args = parser.parse_args()
    
    solve_max_clique(args.instance_path, args.solution_path, args.time_limit, args.log_path)

if __name__ == "__main__":
    main()