import json
import argparse
import time
import sys
import random
from solution_logger import SolutionLogger


def load_instance(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    graph = data['graph']
    num_vertices = graph['num_vertices']
    num_edges = graph['num_edges']
    vertices = graph['vertices']
    edges = graph['edges']
    return num_vertices, num_edges, vertices, edges


def build_graph_structures(vertices, edges):
    adj = {v: set() for v in vertices}
    weight = {}
    for u, v, w in edges:
        adj[u].add(v)
        adj[v].add(u)
        key = (min(u, v), max(u, v))
        weight[key] = w
    return adj, weight


def compute_clique_weight(clique_set, weight):
    total = 0
    clique_list = sorted(clique_set)
    for i in range(len(clique_list)):
        for j in range(i + 1, len(clique_list)):
            key = (clique_list[i], clique_list[j])
            if key in weight:
                total += weight[key]
    return total


def greedy_construct(vertices, adj, weight):
    """Greedy construction: add vertices one by one maximizing marginal gain."""
    # Score each vertex by sum of incident edge weights
    vertex_score = {}
    for v in vertices:
        score = sum(weight.get((min(v, u), max(v, u)), 0) for u in adj[v])
        vertex_score[v] = score
    
    # Sort by score descending
    sorted_vertices = sorted(vertices, key=lambda v: -vertex_score[v])
    
    clique = set()
    for v in sorted_vertices:
        # Check if v is connected to all vertices in current clique
        if all(v in adj[u] for u in clique):
            clique.add(v)
    
    return clique


def greedy_construct_weighted(vertices, adj, weight, current_clique=None):
    """Greedy construction considering marginal weight gain."""
    if current_clique is None:
        clique = set()
    else:
        clique = set(current_clique)
    
    # Get candidates that can be added
    changed = True
    while changed:
        changed = False
        best_gain = 0
        best_v = None
        for v in vertices:
            if v in clique:
                continue
            if all(v in adj[u] for u in clique):
                # Compute marginal gain
                gain = sum(weight.get((min(v, u), max(v, u)), 0) for u in clique)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
        if best_v is not None and best_gain > 0:
            clique.add(best_v)
            changed = True
    
    return clique


def local_search(vertices, adj, weight, clique, time_limit, start_time):
    """Local search: try adding, removing, and swapping vertices."""
    best_clique = set(clique)
    best_weight = compute_clique_weight(best_clique, weight)
    
    improved = True
    iteration = 0
    while improved and time.time() - start_time < time_limit * 0.3:
        improved = False
        iteration += 1
        
        # Try adding vertices
        for v in vertices:
            if v in best_clique:
                continue
            if all(v in adj[u] for u in best_clique):
                gain = sum(weight.get((min(v, u), max(v, u)), 0) for u in best_clique)
                if gain > 0:
                    best_clique.add(v)
                    best_weight += gain
                    improved = True
        
        # Try swapping: remove one vertex, try to add others
        if not improved and len(best_clique) > 1:
            clique_list = list(best_clique)
            for remove_v in clique_list:
                if time.time() - start_time > time_limit * 0.3:
                    break
                # Weight lost by removing remove_v
                lost = sum(weight.get((min(remove_v, u), max(remove_v, u)), 0) 
                          for u in best_clique if u != remove_v)
                
                # Try adding vertices that become available
                candidate_clique = best_clique - {remove_v}
                gained = 0
                for v in vertices:
                    if v in candidate_clique or v == remove_v:
                        continue
                    if all(v in adj[u] for u in candidate_clique):
                        g = sum(weight.get((min(v, u), max(v, u)), 0) for u in candidate_clique)
                        if g > 0:
                            candidate_clique.add(v)
                            gained += g
                
                # Check if we can add more after initial additions
                more_added = True
                while more_added:
                    more_added = False
                    for v in vertices:
                        if v in candidate_clique:
                            continue
                        if all(v in adj[u] for u in candidate_clique):
                            g = sum(weight.get((min(v, u), max(v, u)), 0) for u in candidate_clique)
                            if g > 0:
                                candidate_clique.add(v)
                                gained += g
                                more_added = True
                
                if gained > lost:
                    best_clique = candidate_clique
                    best_weight = best_weight - lost + gained
                    improved = True
                    break
    
    return best_clique, best_weight


def multi_greedy(vertices, adj, weight, time_limit, start_time, logger):
    """Run multiple greedy constructions with different orderings."""
    best_clique = set()
    best_weight = 0
    
    # Strategy 1: Greedy by total incident weight
    clique = greedy_construct(vertices, adj, weight)
    w = compute_clique_weight(clique, weight)
    if w > best_weight:
        best_weight = w
        best_clique = set(clique)
        if logger:
            sol_dict = make_solution_dict(best_clique, best_weight, adj, weight)
            logger.log_solution(best_weight, sol_dict)
    
    # Strategy 2: Weighted greedy
    clique = greedy_construct_weighted(vertices, adj, weight)
    w = compute_clique_weight(clique, weight)
    if w > best_weight:
        best_weight = w
        best_clique = set(clique)
        if logger:
            sol_dict = make_solution_dict(best_clique, best_weight, adj, weight)
            logger.log_solution(best_weight, sol_dict)
    
    # Strategy 3: Random restarts
    random.seed(0)
    max_restarts = 50
    for _ in range(max_restarts):
        if time.time() - start_time > time_limit * 0.2:
            break
        shuffled = list(vertices)
        random.shuffle(shuffled)
        clique = set()
        for v in shuffled:
            if all(v in adj[u] for u in clique):
                clique.add(v)
        # Improve with weighted greedy
        clique = greedy_construct_weighted(vertices, adj, weight, clique)
        w = compute_clique_weight(clique, weight)
        if w > best_weight:
            best_weight = w
            best_clique = set(clique)
            if logger:
                sol_dict = make_solution_dict(best_clique, best_weight, adj, weight)
                logger.log_solution(best_weight, sol_dict)
    
    # Local search improvement
    if time.time() - start_time < time_limit * 0.5:
        clique, w = local_search(vertices, adj, weight, best_clique, time_limit, start_time)
        if w > best_weight:
            best_weight = w
            best_clique = set(clique)
            if logger:
                sol_dict = make_solution_dict(best_clique, best_weight, adj, weight)
                logger.log_solution(best_weight, sol_dict)
    
    return best_clique, best_weight


def make_solution_dict(clique_set, obj_value, adj, weight):
    clique_list = sorted(clique_set)
    clique_edges = []
    for i in range(len(clique_list)):
        for j in range(i + 1, len(clique_list)):
            u, v = clique_list[i], clique_list[j]
            if v in adj[u]:
                clique_edges.append([u, v])
    
    return {
        "objective_value": float(obj_value),
        "clique_vertices": clique_list,
        "clique_edges": clique_edges
    }


def solve_with_gurobi(vertices, edges, adj, weight, time_limit, start_time, logger):
    """Solve using Gurobi MIP."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, 0
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        return None, 0
    
    try:
        model = gp.Model("MEWCP")
        model.setParam('OutputFlag', 0)
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', remaining_time)
        
        # Binary variables for vertices
        x = {}
        for v in vertices:
            x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")
        
        # For each edge, create y variable = x_u * x_v
        edge_vars = {}
        for u, v, w in edges:
            key = (min(u, v), max(u, v))
            y = model.addVar(vtype=GRB.BINARY, name=f"y_{key[0]}_{key[1]}")
            edge_vars[key] = y
            # Linearization constraints
            model.addConstr(y <= x[u], name=f"y_le_xu_{key[0]}_{key[1]}")
            model.addConstr(y <= x[v], name=f"y_le_xv_{key[0]}_{key[1]}")
            model.addConstr(y >= x[u] + x[v] - 1, name=f"y_ge_sum_{key[0]}_{key[1]}")
        
        # Non-edge constraints: for pairs not connected, at most one can be selected
        vertex_set = set(vertices)
        edge_set = set()
        for u, v, w in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = vertices[i], vertices[j]
                key = (min(u, v), max(u, v))
                if key not in edge_set:
                    model.addConstr(x[u] + x[v] <= 1, name=f"nonedge_{key[0]}_{key[1]}")
        
        # Objective: maximize sum of weights * y_e
        obj = gp.LinExpr()
        for u, v, w in edges:
            key = (min(u, v), max(u, v))
            obj += w * edge_vars[key]
        
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # Set initial solution if we have one from heuristic
        model.update()
        
        model.optimize()
        
        if model.SolCount > 0:
            clique = set()
            for v in vertices:
                if x[v].X > 0.5:
                    clique.add(v)
            
            obj_val = compute_clique_weight(clique, weight)
            return clique, obj_val
        
    except Exception as e:
        print(f"Gurobi error: {e}", file=sys.stderr)
    
    return None, 0


def main():
    parser = argparse.ArgumentParser(description="Maximum Edge Weight Clique Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # Load instance
    num_vertices, num_edges, vertices, edges = load_instance(args.instance_path)
    adj, weight = build_graph_structures(vertices, edges)
    
    # Phase 1: Heuristic solution
    best_clique, best_weight = multi_greedy(vertices, adj, weight, args.time_limit, start_time, logger)
    
    # Phase 2: Try Gurobi MIP if time permits and instance is not too large
    remaining = args.time_limit - (time.time() - start_time)
    # Use Gurobi for instances where the formulation is manageable
    if remaining > 5 and num_vertices <= 2000:
        gurobi_clique, gurobi_weight = solve_with_gurobi(
            vertices, edges, adj, weight, args.time_limit, start_time, logger
        )
        if gurobi_clique is not None and gurobi_weight > best_weight:
            best_clique = gurobi_clique
            best_weight = gurobi_weight
            if logger:
                sol_dict = make_solution_dict(best_clique, best_weight, adj, weight)
                logger.log_solution(best_weight, sol_dict)
    
    # Final local search if time remains
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 1:
        clique, w = local_search(vertices, adj, weight, best_clique, args.time_limit, start_time)
        if w > best_weight:
            best_weight = w
            best_clique = set(clique)
            if logger:
                sol_dict = make_solution_dict(best_clique, best_weight, adj, weight)
                logger.log_solution(best_weight, sol_dict)
    
    # Write solution
    solution = make_solution_dict(best_clique, best_weight, adj, weight)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    print(f"Objective value: {best_weight}")
    print(f"Clique size: {len(best_clique)}")


if __name__ == "__main__":
    main()