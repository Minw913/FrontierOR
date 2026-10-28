import argparse
import json
import math
from collections import defaultdict, deque
import gurobipy as gp
from gurobipy import GRB
import time
import itertools


def solve_arc_routing(instance_data, time_limit):
    """
    Solves the Capacitated Arc Routing Problem with multiple identical vehicles.
    
    Args:
        instance_data: Dictionary containing the problem instance
        time_limit: Maximum allowed computation time in seconds
    
    Returns:
        Solution dictionary with the specified output schema
    """
    print("Starting to build the model...")
    required_edges = [e for e in instance_data['edges'] if e['is_required']]
    all_edges = instance_data['edges']
    num_nodes = instance_data['graph']['num_nodes']
    depot = instance_data['depot']
    num_vehicles = instance_data['fleet']['num_vehicles']
    vehicle_capacity = instance_data['fleet']['vehicle_capacity']
    
    # Create node -> edge mapping to identify edges connected to nodes
    node_to_edge_map = defaultdict(list)
    for e in all_edges:
        edge_id = e['edge_id']
        u, v = e['endpoints']
        node_to_edge_map[u].append(edge_id)
        node_to_edge_map[v].append(edge_id)
    
    # Maps for easy retrieval of edge properties
    edge_demand_map = {e['edge_id']: e['demand'] for e in all_edges}
    service_cost_map = {e['edge_id']: e['service_cost'] for e in all_edges}
    deadhead_cost_map = {e['edge_id']: e['cost'] for e in all_edges}
    endpoint_map = {e['edge_id']: tuple(e['endpoints']) for e in all_edges}
    endpoints_list = [(e['endpoints'][0], e['endpoints'][1]) for e in all_edges]
    endpoint_to_idx = {tuple(sorted(e['endpoints'])): i for i, e in enumerate(all_edges)}
    
    # Calculate total service cost (this part is fixed)
    total_service_cost = sum(service_cost_map[edge['edge_id']] for edge in required_edges if edge['is_required'])

    # Build a graph for computing deadhead distances between important locations
    def compute_shortest_paths():
        # Use Floyd-Warshall to compute shortest paths between all pairs of vertices
        dist = [[float('inf')] * num_nodes for _ in range(num_nodes)]
        
        # Initialize direct connections
        for e in all_edges:
            u, v = e['endpoints']
            cost = e['cost']  # deadheading cost
            dist[u][v] = min(dist[u][v], cost)
            dist[v][u] = min(dist[v][u], cost)
        
        # Distance to node itself is 0
        for i in range(num_nodes):
            dist[i][i] = 0
        
        # Floyd-Warshall algorithm
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        return dist

    print("Computing shortest paths...")
    distance_matrix = compute_shortest_paths()

    # Formulation of the problem (heuristic approach due to complexity)
    start_time = time.time()
    
    # Phase 1: Assign required edges to vehicles while respecting capacity
    print("Assigning required edges to vehicles...")
    # Greedy assignment based on capacity constraint
    routes = []
    remaining_required_edges = list(required_edges)
    
    for v in range(num_vehicles):
        vehicle_route = {
            'vehicle': v,
            'serviced_edges': [],
            'deadheaded_edges': []
        }
        current_load = 0
        
        # Assign as many as possible to the current vehicle without exceeding capacity
        remaining_items = []
        for req_edge in remaining_required_edges:
            if current_load + req_edge['demand'] <= vehicle_capacity:
                vehicle_route['serviced_edges'].append(req_edge['edge_id'])
                current_load += req_edge['demand']
            else:
                remaining_items.append(req_edge)
        
        remaining_required_edges = remaining_items
        routes.append(vehicle_route)
        
        if not remaining_required_edges:
            # If all required edges are allocated, pad with empty vehicles
            for v_rest in range(v + 1, num_vehicles):
                routes.append({
                    'vehicle': v_rest,
                    'serviced_edges': [],
                    'deadheaded_edges': []
                })
            break
            
    # If we still have unassigned required edges after using all vehicles
    if remaining_required_edges:
        print(f"WARNING: Capacity constraints make solution infeasible. Still need to assign {len(remaining_required_edges)} required edges.")
        # This indicates infeasibility, but let's try to continue anyway
    
    print(f"Finished assignment. Processing {len(routes)} vehicle routes...")

    # Phase 2: For each vehicle route, find a good sequence starting and ending at depot
    # Simple approach: Find a tour for each vehicle to visit all its required edges
    total_deadhead_cost = 0

    for route_info in routes:
        route_serviced_edges = route_info['serviced_edges']
        if not route_serviced_edges:
            continue  # Skip if no work needed

        # Build a complete graph of locations of interest for this specific vehicle's route
        # Locations: depot + nodes incident to required edges in the route
        interesting_nodes = set([depot])
        for edge_id in route_serviced_edges:
            u, v = endpoint_map[edge_id]
            interesting_nodes.add(u)
            interesting_nodes.add(v)

        location_list = sorted(list(interesting_nodes))

        # For each pair of these locations, we have the shortest path distance
        sub_dist = {
            n1: {n2: distance_matrix[n1][n2] for n2 in location_list} 
            for n1 in location_list
        }

        # Now find a path that visits all required edges, starting and ending at depot
        # Using a heuristic: MST + DFS to get Eulerian multigraph approximation
        # Or simpler: nearest neighbor heuristic with respect to the required edges and depot
        
        # Create a simplified tour visiting all endpoints of the required edges for this vehicle
        visited_nodes = set([depot])
        current_location = depot
        deadhead_edges_used = defaultdict(int)
        
        # Visit each required edge exactly once, in a greedy nearest neighbor fashion
        unserviced = list(route_serviced_edges)
        
        while unserviced:
            next_edge = unserviced[0]
            u, v = endpoint_map[next_edge]
            
            # Choose the closest end point from the current location
            cost_u = sub_dist[current_location][u]
            cost_v = sub_dist[current_location][v]
            
            if cost_u <= cost_v:
                # Go to u then v
                if current_location != u:
                    # Add deadhead path from current_location to u
                    path_u = reconstruct_path(distance_matrix, current_location, u)
                    for i in range(len(path_u) - 1):
                        uv_edge_ids = find_edge_ids_between_nodes(node_to_edge_map, path_u[i], path_u[i+1])
                        for eid in uv_edge_ids:
                            if (path_u[i+1], path_u[i]) == tuple(sorted(endpoint_map[eid])):
                                deadhead_edges_used[eid] += 1
                                break
                        
                # Service edge going from u to v
                current_location = v
                
                # Go to node v (or stay at v since we'll use this edge later)
            else:
                # Go to v then u
                if current_location != v:
                    # Add deadhead path from current_location to v
                    path_v = reconstruct_path(distance_matrix, current_location, v)
                    for i in range(len(path_v) - 1):
                        uv_edge_ids = find_edge_ids_between_nodes(node_to_edge_map, path_v[i], path_v[i+1])
                        for eid in uv_edge_ids:
                            if (path_v[i+1], path_v[i]) == tuple(sorted(endpoint_map[eid])):
                                deadhead_edges_used[eid] += 1
                                break
                            
                # Service edge but go from v to u
                current_location = u
            
            # In either case, we've handled the required traversal of the edge
            # Mark as serviced (add to deadhead if we need to come back)
            unserviced.pop(0)
            
            # In practice, we may need both directions depending on overall tour construction
            # For simplicity, just mark that we used the deadhead for reaching
        
        # Return to depot after servicing all edges for this vehicle
        if current_location != depot:
            path_back = reconstruct_path(distance_matrix, current_location, depot)
            for i in range(len(path_back) - 1):
                uv_edge_ids = find_edge_ids_between_nodes(node_to_edge_map, path_back[i], path_back[i+1])
                for eid in uv_edge_ids:
                    if (path_back[i+1], path_back[i]) == tuple(sorted(endpoint_map[eid])):
                        deadhead_edges_used[eid] += 1
                        break
        
        # Update route info
        route_info['deadheaded_edges'] = [{'edge_id': edge_id, 'times': count} 
                                          for edge_id, count in deadhead_edges_used.items()]
        
        # Calculate the cost of these deadhead traversals for updating the total
        total_deadhead_cost += sum(
            deadhead_edges_used[edge_id] * deadhead_cost_map[edge_id] 
            for edge_id in deadhead_edges_used.keys()
        )
    
    print(f"Finished route construction phase.")
    
    # Final cost calculation
    total_cost = total_service_cost + total_deadhead_cost
            
    print(f"Total cost calculated: {total_cost}")

    solution = {
        'objective_value': total_cost,
        'routes': routes
    }

    return solution


def find_edge_ids_between_nodes(node_to_edge_map, u, v):
    """Finds edge IDs connecting nodes u and v"""
    u_edges = set(node_to_edge_map.get(u, []))
    v_edges = set(node_to_edge_map.get(v, []))
    common_edges = u_edges.intersection(v_edges)
    return list(common_edges)


def reconstruct_path(distance_matrix, start, end):
    """
    Reconstruct actual path nodes from distance matrix computed by Floyd-Warshall
    Note: This simplified version will not return actual path nodes, so we need a different approach
    """
    # Instead, we'll compute the path again using a separate shortest path algorithm that keeps parent pointers
    num_nodes = len(distance_matrix)
    visited = [False] * num_nodes
    dist = [float('inf')] * num_nodes
    parent = [-1] * num_nodes
    
    dist[start] = 0
    
    for _ in range(num_nodes):
        # Find min distance vertex
        u = -1
        for v in range(num_nodes):
            if not visited[v] and (u == -1 or dist[v] < dist[u]):
                u = v
        
        if u == -1 or u == end:
            break
            
        visited[u] = True
        
        # Update dist values of adjacent vertices
        for v in range(num_nodes):
            if (not visited[v] and
                distance_matrix[u][v] != float('inf') and
                dist[u] + distance_matrix[u][v] < dist[v]):
                
                dist[v] = dist[u] + distance_matrix[u][v]
                parent[v] = u
    
    # Backtrack to find path
    path = []
    curr = end
    while curr != -1:
        path.append(curr)
        curr = parent[curr]
    
    return path[::-1]


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Solve the arc routing problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Log path for solution logger')
    args = parser.parse_args()

    # Initialize logger
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_arc_routing(instance_data, args.time_limit)
    
    # If we found a better solution during processing, log it
    if logger:
        logger.log(solution['objective_value'])
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
    

if __name__ == '__main__':
    main()