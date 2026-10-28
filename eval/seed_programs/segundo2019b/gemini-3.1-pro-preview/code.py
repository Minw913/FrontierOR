import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB

def solve(args):
    start_time = time.time()
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    graph = data['graph']
    num_vertices = graph['num_vertices']
    original_vertices = graph['vertices']
    edges = graph['edges']
    weights = data['vertex_weights']
    
    # Map each vertex ID to its corresponding weight based on the input arrays
    weight_map = {original_vertices[i]: weights[i] for i in range(num_vertices)}
    
    # Maintain sorted mappings for deterministic processing and variable indexing
    vertices = sorted(original_vertices)
    idx_to_v = {i: vertices[i] for i in range(num_vertices)}
    v_to_idx = {vertices[i]: i for i in range(num_vertices)}
    
    # Build complete adjacency list internally mapped to index space 0..num_vertices-1
    full_adj = {i: set() for i in range(num_vertices)}
    for u, v in edges:
        i, j = v_to_idx[u], v_to_idx[v]
        full_adj[i].add(j)
        full_adj[j].add(i)
        
    # Discover all non-edge pairs (complement graph edges) efficiently
    non_edges = []
    for i in range(num_vertices - 1):
        missing = set(range(i + 1, num_vertices)) - full_adj[i]
        if missing:
            non_edges.extend([(i, j) for j in missing])
            
    # Compute a high-quality initial feasible solution using a simple greedy heuristic
    sorted_indices = sorted(range(num_vertices), key=lambda i: weight_map[idx_to_v[i]], reverse=True)
    initial_clique = []
    for i in sorted_indices:
        is_connected = True
        for c in initial_clique:
            if c not in full_adj[i]:
                is_connected = False
                break
        if is_connected:
            initial_clique.append(i)
            
    # Set up Gurobi environment and core setup constraints
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()
    
    m = gp.Model("max_weight_clique", env=env)
    
    # Restrict environment to single CPU core as per requirements
    m.setParam("Threads", 1)
    
    x = m.addVars(num_vertices, vtype=GRB.BINARY, name="x")
    
    # Implement the greedy initial solution as a valid start point
    initial_clique_set = set(initial_clique)
    for i in range(num_vertices):
        x[i].Start = 1.0 if i in initial_clique_set else 0.0
        
    # Objective Formulation: Maximize total weights
    obj_expr = gp.quicksum(weight_map[idx_to_v[i]] * x[i] for i in range(num_vertices))
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # For every pair of non-connected vertices, they cannot naturally both occur in a valid clique
    if non_edges:
        m.addConstrs((x[i] + x[j] <= 1 for i, j in non_edges))
        
    # Strictly respect max timeout constraint including model construction overhead
    time_spent = time.time() - start_time
    time_limit = max(1, args.time_limit - time_spent - 1)
    m.setParam("TimeLimit", time_limit)
    
    # Try importing SolutionLogger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    except ImportError:
        logger = None
        
    # Process intermediate solutions optimally across callback stages
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj)
            except Exception:
                pass

    m.optimize(cb)
    
    # Determine the absolute best parameters regardless of the run's final execution endpoint
    if m.Status == GRB.OPTIMAL or m.SolCount > 0:
        obj_val = m.ObjVal
        clique_vertices = [idx_to_v[i] for i in range(num_vertices) if x[i].X > 0.5]
    else:
        # Fallback to initial heuristic if solver somehow found nothing useful within deep constraint time limits
        obj_val = sum(weight_map[idx_to_v[i]] for i in initial_clique)
        clique_vertices = [idx_to_v[i] for i in initial_clique]
        
    out = {
        "objective_value": float(obj_val),
        "clique_vertices": clique_vertices
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    
    args = parser.parse_args()
    solve(args)