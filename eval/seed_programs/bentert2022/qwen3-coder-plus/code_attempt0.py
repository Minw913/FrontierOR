import argparse
import json
import math
import time
import itertools
from solution_logger import SolutionLogger


def calculate_distance_squared(p1, p2):
    """Calculate squared Euclidean distance between two points"""
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def compute_all_weights(vertices):
    """Compute weights for all possible edges"""
    n = len(vertices)
    vertex_positions = {v['id']: (v['x'], v['y']) for v in vertices}
    weights = {}
    
    for i in range(n):
        for j in range(i+1, n):
            id1, id2 = vertices[i]['id'], vertices[j]['id']
            pos1, pos2 = vertex_positions[id1], vertex_positions[id2]
            dist_sq = calculate_distance_squared(pos1, pos2)
            weights[(id1, id2)] = dist_sq
            weights[(id2, id1)] = dist_sq
    
    return weights


def greedy_solution(vertices, weights):
    """Greedy algorithm for finding a good spanning tree solution"""
    n = len(vertices)
    vertex_ids = [v['id'] for v in vertices]
    
    # Start with the minimum spanning tree using Prim's algorithm
    mst_edges = []
    visited = set([vertex_ids[0]])
    unvisited = set(vertex_ids[1:])
    
    while unvisited:
        min_edge = None
        min_weight = float('inf')
        
        for u in visited:
            for v in unvisited:
                edge_key = (min(u,v), max(u,v))
                w = weights[edge_key]
                if w < min_weight:
                    min_weight = w
                    min_edge = (u, v)
        
        mst_edges.append((min_edge[0], min_edge[1]))
        visited.add(min_edge[1])
        unvisited.remove(min_edge[1])
    
    # Compute costs for MST solution
    selected_edges = mst_edges
    degree = {v_id: 0 for v_id in vertex_ids}
    edge_max_weight = {}
    
    for u, v in selected_edges:
        degree[u] += 1
        degree[v] += 1
        edge_key = (min(u,v), max(u,v))
        w = weights[edge_key]
        edge_max_weight[(u,v)] = w
        edge_max_weight[(v,u)] = w
    
    vertex_costs = {}
    for v_id in vertex_ids:
        incident_weights = []
        for u, v in selected_edges:
            if u == v_id or v == v_id:
                edge_key = (min(u,v), max(u,v))
                incident_weights.append(weights[edge_key])
        
        if incident_weights:
            vertex_costs[v_id] = max(incident_weights)
        else:
            vertex_costs[v_id] = 0
    
    objective_value = sum(vertex_costs.values())
    
    return objective_value, selected_edges, vertex_costs


def local_search_improvement(vertices, initial_objective, initial_edges, initial_costs, weights, time_limit, start_time, logger):
    """Local search improvement of initial solution with time budget"""
    n = len(vertices)
    vertex_ids = [v['id'] for v in vertices]
    current_obj = initial_objective
    current_edges = [(u, v) for u, v in initial_edges]
    current_costs = {k: v for k, v in initial_costs.items()}

    iteration = 0
    improvement_found = True

    while improvement_found:
        if time.time() - start_time >= time_limit * 0.9:
            break  # Reserve time for final solution processing

        improvement_found = False
        iteration += 1

        # Try to improve by swapping one edge
        edges_set = set(current_edges)
        remaining_vertices = set(vertex_ids)
        
        for u_target, v_target in list(edges_set):
            neighbors_dict = build_neighbors_dict(current_edges)
            
            # Remove the edge temporarily
            temp_edges = [e for e in current_edges if e != (u_target, v_target) and e != (v_target, u_target)]
            
            # Check if removing this edge breaks connectivity
            if is_connected(temp_edges, vertex_ids):
                remaining_vertices = set(vertex_ids)
                
                # Try adding alternative edges one by one to maintain connectivity
                for x in vertex_ids:
                    for y in vertex_ids:
                        if x < y and (x,y) not in edges_set:
                            candidate_edges = temp_edges + [(x,y)]
                            
                            # Check if this maintains connectivity and is a spanning tree (with correct number of edges)
                            if is_connected(candidate_edges, vertex_ids) and len(candidate_edges) == n - 1:
                                new_cost = compute_objective_with_cost_details(candidate_edges, weights, vertex_ids)
                                if new_cost[0] < current_obj:
                                    current_obj = new_cost[0]
                                    current_edges = candidate_edges
                                    current_costs = new_cost[1]
                                    if logger:
                                        logger.log(current_obj)
                                    improvement_found = True
                                    
                                    # Break early if timeout is near
                                    if time.time() - start_time >= time_limit * 0.85:
                                        return current_obj, current_edges, current_costs
        
        # If we're taking too much time, bail out of main loop
        if time.time() - start_time >= time_limit * 0.85:
            break

    return current_obj, current_edges, current_costs


def build_neighbors_dict(edges):
    """Build adjacency dictionary from edge list"""
    neighbors = {}
    for u, v in edges:
        if u not in neighbors:
            neighbors[u] = []
        if v not in neighbors:
            neighbors[v] = []
        neighbors[u].append(v)
        neighbors[v].append(u)
    return neighbors


def dfs_component_check(start_node, neighbors_dict, visited):
    """DFS to check component connectivity"""
    stack = [start_node]
    component_size = 0
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        component_size += 1
        for neighbor in neighbors_dict.get(node, []):
            if neighbor not in visited:
                stack.append(neighbor)
    return component_size


def is_connected(edges, vertex_ids):
    """Check if the given edges form a connected graph with all vertices"""
    if not edges:
        return len(vertex_ids) <= 1
        
    neighbors_dict = build_neighbors_dict(edges)
    visited = set()
    start_vertex = vertex_ids[0]
    
    comp_size = dfs_component_check(start_vertex, neighbors_dict, visited)
    
    # Check that all vertices are visited and that components size matches
    return comp_size == len(vertex_ids)


def compute_objective_with_cost_details(edges, weights, vertex_ids):
    """Calculate objective and detailed costs"""
    degree = {v_id: 0 for v_id in vertex_ids}
    vertex_costs = {}
    
    # Initialize costs to 0
    for v_id in vertex_ids:
        vertex_costs[v_id] = 0

    # Update for actual edges
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
        edge_key = (min(u,v), max(u,v))
        w = weights[edge_key]
        
        # Update max for each vertex
        if w > vertex_costs[u]:
            vertex_costs[u] = w
        if w > vertex_costs[v]:
            vertex_costs[v] = w
    
    obj_value = sum(vertex_costs.values())
    
    # For vertices with no edges in the solution, cost remains 0
    return obj_value, vertex_costs


def spanning_tree_with_pruning(vertices, weights):
    """Alternative method focusing on minimizing degree-cost interactions"""
    n = len(vertices)
    vertex_ids = [v['id'] for v in vertices]
    
    # Sort edges by weight ascending for Prim-like expansion
    weight_list = []
    for u, v in itertools.combinations(vertex_ids, 2):
        edge_key = (min(u, v), max(u, v))
        weight_list.append((weights[edge_key], u, v))
    weight_list.sort()
    
    parent = {v_id: v_id for v_id in vertex_ids}
    
    def find_root(v):
        if parent[v] != v:
            parent[v] = find_root(parent[v])
        return parent[v]
    
    def union(u, v):
        root_u, root_v = find_root(u), find_root(v)
        if root_u != root_v:
            parent[root_u] = root_v
            return True
        return False
    
    selected_edges = []
    for w, u, v in weight_list:
        if union(u, v):
            selected_edges.append((u, v))
            if len(selected_edges) == n - 1:
                break
    
    # Now compute vertex costs based on the MST
    vertex_costs = {}
    for v_id in vertex_ids:
        incident_weights = []
        for u, v in selected_edges:
            if u == v_id or v == v_id:
                edge_key = (min(u,v), max(u,v))
                incident_weights.append(weights[edge_key])
        
        vertex_costs[v_id] = max(incident_weights) if incident_weights else 0
    
    objective_value = sum(vertex_costs.values())
    
    return objective_value, selected_edges, vertex_costs


def solve(args):
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    start_time = time.time()
    time_limit = args.time_limit
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    vertices = instance_data['vertices']
    weights = compute_all_weights(vertices)
    
    # Compute an initial solution using a greedy approach (MST-based)
    obj_val, edges, vertex_costs = greedy_solution(vertices, weights)

    if logger:
        logger.log(obj_val)

    # Attempt additional improvements using local search within time limits
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed
    if remaining_time > 0:
        try:
            obj_val, edges, vertex_costs = local_search_improvement(
                vertices, obj_val, edges, vertex_costs, 
                weights, remaining_time, start_time, logger
            )
        except KeyboardInterrupt:
            pass  # Gracefully exit if interrupted

    # Ensure solution validity and structure
    solution = {
        "objective_value": obj_val,
        "selected_edges": [[min(u,v), max(u,v)] for u, v in edges],
        "vertex_costs": {str(k): v for k, v in vertex_costs.items()}
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    
    args = parser.parse_args()
    solve(args)