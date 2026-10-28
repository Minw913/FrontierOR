import argparse
import json
import itertools
from collections import defaultdict, deque
import gurobipy as gp
from gurobipy import GRB
import time
from solution_logger import SolutionLogger


def is_connected(vertices, edges):
    """Check if subgraph induced by vertices is connected."""
    if len(vertices) <= 1:
        return True

    vertex_set = set(vertices)
    edges_subgraph = [(u, v) for u, v in edges if u in vertex_set and v in vertex_set]
    
    if len(edges_subgraph) == 0:
        return False  # More than 1 isolated vertex without connecting edges
    
    adj = defaultdict(list)
    for u, v in edges_subgraph:
        adj[u].append(v)
        adj[v].append(u)
        
    visited = set()
    start = vertices[0]
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
    return len(visited) == len(vertices)


def is_s_plex(vertices, edges_dict, s):
    """
    Check if the subgraph induced by vertices is an s-plex.
    Each vertex in the subgraph has degree >= |vertices| - s
    """
    n = len(vertices)
    vertices_set = set(vertices)
    
    for v in vertices:
        degree = sum(1 for u in vertices_set if u != v and edges_dict[v].get(u))
        if degree < n - s:
            return False
    return True


def find_all_s_plexes(vertices, edges_list, s, connectivity_required):
    """Find all s-plexes in the graph."""
    edges_dict = defaultdict(lambda: {})
    for u, v in edges_list:
        edges_dict[u][v] = 1
        edges_dict[v][u] = 1
        
    all_s_plexes = []
    n = len(vertices)
    
    # Generate all possible subsets of vertices
    for r in range(1, n + 1):  # Size of s-plex
        for subset in itertools.combinations(vertices, r):
            subset = list(subset)
            if is_s_plex(subset, edges_dict, s):
                if connectivity_required:
                    if is_connected(subset, edges_list):
                        all_s_plexes.append(subset)
                else:
                    all_s_plexes.append(subset)
                    
    return all_s_plexes  # Return all s-plexes


def solve_set_partition_cover_model(
    vertices,
    all_relaxed_cliques,  # These are precomputed subsets meeting relaxed conditions
    decomposition_type,
    initial_objective=None,
    time_limit=3600
):
    """
    Formulates and solves the set covering/partitioning problem with Gurobi.
    """
    n = len(vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Create model
    model = gp.Model("vertex_decomposition")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)  # Silent mode
    
    if initial_objective:
        # Set a warm start objective hint
        model.setParam(GRB.Param.Cutoff, initial_objective) 

    # Variables: binary variables for each possible relaxed clique
    x = []
    for i, clique in enumerate(all_relaxed_cliques):
        var_name = f"splex_{i}"
        x.append(model.addVar(vtype=GRB.BINARY, name=var_name))
    
    # Constraints: Each vertex must be covered based on decomposition_type
    for v in vertices:
        expr = gp.LinExpr()
        for i, clique in enumerate(all_relaxed_cliques):
            if v in clique:
                expr += x[i]
        
        if decomposition_type == "partitioning":
            model.addConstr(expr == 1, name=f"cover_vertex_{v}")
        elif decomposition_type == "covering":
            model.addConstr(expr >= 1, name=f"cover_vertex_{v}")
    
    # Objective: Minimize the total number of selected groups
    model.setObjective(gp.quicksum(x), GRB.MINIMIZE)
    
    model.optimize()

    if model.SolCount > 0:
        selected_groups = []
        for i in range(len(x)):
            if x[i].X > 0.5:
                selected_groups.append(all_relaxed_cliques[i])
                
        return selected_groups, model.ObjVal
    else:
        return [], float('inf')  # No solution found


def main():
    parser = argparse.ArgumentParser(description='Solve vertex decomposition into relaxed cliques')
    parser.add_argument('--instance_path', required=True, help='Path to input JSON instance')
    parser.add_argument('--solution_path', required=True, help='Path to output JSON solution')
    parser.add_argument('--time_limit', required=True, type=int, help='Time limit in seconds')
    parser.add_argument('--log_path', help='Path to log file (optional)')
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Extract instance details
    graph = instance['graph']
    problem_settings = instance['problem_settings']

    vertices = graph['vertices']
    edges = graph['edges']
    s = problem_settings.get('parameter_s')
    decomposition_type = problem_settings['decomposition_type']
    connectivity_required = problem_settings['connectivity_required']

    if problem_settings['relaxed_clique_type'] != 's-plex':
        raise ValueError(f"Only s-plex type is currently supported, got: {problem_settings['relaxed_clique_type']}")

    # Initialize logger if requested
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    # Find all valid s-plexes that satisfy additional constraints
    all_s_plexes = find_all_s_plexes(vertices, edges, s, connectivity_required)

    # For very large problem instances, we might want to limit the enumeration
    # But for simplicity here, we'll work with all found s-plexes
    remaining_time = args.time_limit - (time.time() - start_time)
    
    if remaining_time <= 8:  # Reserve some time for solving
        with open(args.solution_path, 'w') as f:
            json.dump({'objective_value': float('inf'), 'solution': []}, f)
        return

    # Solve the set partition/cover problem
    try:
        solution_groups, obj_val = solve_set_partition_cover_model(
            vertices, 
            all_s_plexes, 
            decomposition_type,
            initial_objective=len(vertices),  # At worst, use singletons
            time_limit=remaining_time
        )
    except Exception as e:
        print(f"Error in optimization: {e}")
        # As fallback, use each singleton component
        solution_groups = [[v] for v in vertices]
        obj_val = len(vertices)

    # Update the logger if a solution was improved
    if logger and obj_val != float('inf'):
        logger.log(obj_val)

    # Prepare output
    output = {
        'objective_value': obj_val,
        'solution': solution_groups
    }

    # Write solution to output file
    with open(args.solution_path, 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    main()