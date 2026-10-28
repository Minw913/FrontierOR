import json
import argparse
import time
import random

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def compress_coloring(coloring):
    """
    Compress the colors of a partial or feasible coloring to contiguous integers 0..k-1
    """
    unique_colors = sorted(list(set(coloring.values())))
    mapping = {old_c: new_c for new_c, old_c in enumerate(unique_colors)}
    new_coloring = {n: mapping[c] for n, c in coloring.items()}
    return new_coloring, len(unique_colors)

def find_large_clique(nodes, adj):
    """Greedy procedure to heuristically find a large clique in the input graph."""
    best_clique = []
    # Try starting cliques from nodes with largest degrees
    nodes_sorted = sorted(list(nodes), key=lambda x: len(adj[x]), reverse=True)
    for start_node in nodes_sorted[:50]:
        clique = [start_node]
        for n in nodes_sorted:
            if n == start_node: 
                continue
            # A node can be added to the clique if connected to all current elements
            is_connected = True
            for c in clique:
                if n not in adj[c]:
                    is_connected = False
                    break
            if is_connected:
                clique.append(n)
        if len(clique) > len(best_clique):
            best_clique = clique
    return best_clique

def dsatur_with_clique(nodes, adj, clique, randomize=False):
    """
    DSATUR algorithm implemented with a specified initial clique assigned distinctly.
    Returns: dictionary {node: assigned_color}
    """
    color_assigned = {}
    sat_degrees = {n: 0 for n in nodes}
    neighbor_colors = {n: set() for n in nodes}
    uncolored = set(nodes)
    degrees = {n: len(adj[n]) for n in nodes}
    
    # Pre-assign clique elements to distinct colors to fix permutation symmetry
    for i, u in enumerate(clique):
        c = i
        color_assigned[u] = c
        uncolored.remove(u)
        for v in adj[u]:
            degrees[v] -= 1
            if c not in neighbor_colors[v]:
                neighbor_colors[v].add(c)
                sat_degrees[v] += 1
                
    while uncolored:
        max_sat = -1
        max_deg = -1
        candidates = []
        for n in uncolored:
            if sat_degrees[n] > max_sat:
                max_sat = sat_degrees[n]
                max_deg = degrees[n]
                candidates = [n]
            elif sat_degrees[n] == max_sat:
                if degrees[n] > max_deg:
                    max_deg = degrees[n]
                    candidates = [n]
                elif degrees[n] == max_deg:
                    candidates.append(n)
        
        # Determine the next node to color
        if randomize and candidates:
            u = random.choice(candidates)
        else:
            u = candidates[0]
            
        uncolored.remove(u)
        
        # Find the smallest available color
        used_colors = neighbor_colors[u]
        c = 0
        while c in used_colors:
            c += 1
            
        color_assigned[u] = c
        
        # Update neighbors with new configuration state
        for v in adj[u]:
            if v in uncolored:
                degrees[v] -= 1
            if c not in neighbor_colors[v]:
                neighbor_colors[v].add(c)
                sat_degrees[v] += 1
                
    return color_assigned

def solve_feasibility(nodes, edges, C, clique, time_limit, adj):
    """
    Formulates Graph Coloring as a sequence of purely feasible Gurobi Binary Programs.
    Uses SOS1-style logical constraints to break remaining permutation symmetry. 
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    model = gp.Model("GCP_feas", env=env)
    model.Params.TimeLimit = max(1.0, time_limit)
    model.Params.Threads = 1  # Limited to single core as described
    model.Params.MIPFocus = 1 # Focused internally on feasibility over bound-tightening
    
    valid_colors = {n: set() for n in nodes}
    clique_set = set(clique)
    remaining_nodes = [n for n in nodes if n not in clique_set]
    # Sorting ensures nodes with higher degrees face smaller color domains sequentially, thus strongly breaking symmetry
    remaining_nodes.sort(key=lambda x: len(adj[x]), reverse=True)
    
    for i, v in enumerate(clique):
        valid_colors[v] = {i}
        
    for k, n in enumerate(remaining_nodes):
        max_col = min(C - 1, len(clique) + k)
        valid_colors[n] = set(range(max_col + 1))
        
    # Variables: only allow theoretically sound colourings to dramatically narrow hypothesis space mapping
    x = {}
    for n in nodes:
        for c in valid_colors[n]:
            x[n, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{n}_{c}")
            
    # Each node uniquely falls into exactly 1 color index mapping
    for n in nodes:
        model.addConstr(gp.quicksum(x[n, c] for c in valid_colors[n]) == 1)
        
    # No two closely connected adjacent nodes fall on equivalent color paths
    for u, v in edges:
        common_colors = valid_colors[u].intersection(valid_colors[v])
        for c in common_colors:
            model.addConstr(x[u, c] + x[v, c] <= 1)
            
    model.setObjective(0, GRB.MINIMIZE)
    model.optimize()
    
    if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
        # Extract the node mappings upon hitting feasible configuration nodes
        coloring = {}
        for n in nodes:
            for c in valid_colors[n]:
                if x[n, c].X > 0.5:
                    coloring[n] = int(c)
                    break
        return True, coloring, model.Status
    elif model.Status == GRB.INFEASIBLE:
        return False, None, model.Status
    else:
        return None, None, model.Status

def main():
    start_time = time.time()
    args = parse_arguments()
    
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None
        
    # 1. Loading Instance Data Pipeline Configured Schema Structure
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    graph_data = data['graph']
    num_edges = graph_data['num_edges']
    nodes_list = graph_data['nodes']
    edges_list = graph_data['edges']
    
    nodes = set(nodes_list)
    adj = {n: [] for n in nodes}
    edges = []
    
    for u, v in edges_list:
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        
    # Exceptional mapping path required for complete structural disconnections 
    if num_edges == 0:
        best_c = 1
        best_coloring = {n: 0 for n in nodes}
        color_classes = [[n for n in nodes]]
        if logger:
            logger.log(best_c)
        solution = {
            "objective_value": best_c,
            "vertex_colors": {str(k): v for k, v in best_coloring.items()},
            "color_classes": color_classes
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # 2. Mathematical Lower Bound / Symmetry Extraction
    clique = find_large_clique(nodes, adj)
    if not clique:
        clique = [nodes_list[0]]
        
    # 3. Base DSATUR Resolution Path
    best_coloring = dsatur_with_clique(nodes, adj, clique, randomize=False)
    best_coloring, best_c = compress_coloring(best_coloring)
    if logger:
        logger.log(best_c)
        
    # 4. Randomized Phase Evaluation - Escaping Local Minimas (Max 2 seconds bound)
    dsatur_end = time.time() + min(2.0, args.time_limit * 0.1)
    while time.time() < dsatur_end:
        c_assign = dsatur_with_clique(nodes, adj, clique, randomize=True)
        c_assign, c_count = compress_coloring(c_assign)
        if c_count < best_c:
            best_c = c_count
            best_coloring = c_assign
            if logger:
                logger.log(best_c)
                
    end_time = start_time + args.time_limit
    
    # 5. Iterative MILP Feasibility Exploration Loop bounded efficiently within timeframe allowances
    while best_c > len(clique):
        C = best_c - 1
        rem_time = end_time - time.time() - 0.5
        if rem_time <= 1.0:
            break
            
        feasible, new_coloring, status = solve_feasibility(nodes, edges, C, clique, rem_time, adj)
        if feasible:
            best_coloring, actual_c = compress_coloring(new_coloring)
            best_c = actual_c
            if logger:
                logger.log(best_c)
        elif status == GRB.INFEASIBLE:
            # Structurally impossible to colour effectively lower than this threshold bounded bound - Optimal Solution Identified
            break
        else:
            # Time limit breach, exiting phase
            break
            
    # 6. Structured Schema Validation and Return Dump
    color_classes = [[] for _ in range(best_c)]
    for n, c in best_coloring.items():
        color_classes[c].append(n)
        
    solution = {
        "objective_value": best_c,
        "vertex_colors": {str(k): v for k, v in best_coloring.items()},
        "color_classes": color_classes
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()