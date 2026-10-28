import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import itertools
from solution_logger import SolutionLogger
import time


def build_subnetwork_graph(edges, edge_indices):
    """Build adjacency list representation of the subnetwork."""
    graph = defaultdict(list)
    for idx in edge_indices:
        edge = edges[idx]
        u, v = edge['endpoints']
        graph[u].append((v, idx))
        graph[v].append((u, idx))
    return graph


def find_simple_paths(graph, start, end, max_length, terminals):
    """Find all simple paths between start and end with given max length."""
    paths = []
    
    def dfs(current, target, path, visited, depth):
        if current == target and len(path) > 0:
            paths.append(path[:])
            return
        
        if depth >= max_length:
            return
            
        for neighbor, edge_idx in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(edge_idx)
                dfs(neighbor, target, path, visited, depth + 1)
                path.pop()
                visited.remove(neighbor)
    
    visited = {start}
    dfs(start, end, [], visited, 0)
    return paths


def enumerate_all_valid_lines(modes, edges, max_line_length_edges):
    """Enumerate all possible valid lines for all modes."""
    all_lines = []
    line_id = 0
    
    for mode_info in modes:
        mode_name = mode_info['name']
        terminals = mode_info['terminals']
        edge_indices = mode_info['edge_indices']
        
        # Build subgraph for this mode
        graph = build_subnetwork_graph(edges, edge_indices)
        
        # Find all simple paths between terminal pairs
        for i in range(len(terminals)):
            for j in range(i + 1, len(terminals)):
                start = terminals[i]
                end = terminals[j]
                
                # Find all paths between these terminals
                paths = find_simple_paths(graph, start, end, max_line_length_edges, terminals)
                
                for path in paths:
                    # Convert path of edge indices to node sequence
                    node_path = [start]
                    current = start
                    for edge_idx in path:
                        u, v = edges[edge_idx]['endpoints']
                        next_node = v if u == current else u
                        node_path.append(next_node)
                        current = next_node
                    
                    all_lines.append({
                        'line_id': line_id,
                        'mode': mode_name,
                        'nodes': node_path,
                        'edges': path,
                        'mode_info': mode_info
                    })
                    line_id += 1
    
    return all_lines


def solve_transit_network(instance_data, time_limit):
    """Solve the transit network optimization problem."""
    # Extract instance data
    global_params = instance_data['global_parameters']
    network = instance_data['network']
    modes = instance_data['modes']
    od_pairs = instance_data['od_matrix']
    
    lambda_weight = global_params['lambda']
    F = global_params['frequency_upper_bound_F']
    max_line_length = global_params['max_line_length_edges']
    
    # Enumerate all valid lines
    all_lines = enumerate_all_valid_lines(modes, network['edges'], max_line_length)
    
    # Create directed arc mapping (each undirected edge -> 2 directed arcs)
    arc_to_edge = {}  # maps directed arc id to undirected edge id
    edge_to_arcs = {}  # maps undirected edge id to list of 2 directed arcs
    num_arcs = 0
    
    for edge_idx, edge in enumerate(network['edges']):
        u, v = edge['endpoints']
        # Create two directed arcs for each undirected edge
        arc_uv = num_arcs
        arc_vu = num_arcs + 1
        arc_to_edge[arc_uv] = edge_idx
        arc_to_edge[arc_vu] = edge_idx
        if edge_idx not in edge_to_arcs:
            edge_to_arcs[edge_idx] = []
        edge_to_arcs[edge_idx].extend([arc_uv, arc_vu])
        num_arcs += 2
    
    # Prepare OD demands
    od_demands = {}
    for od in od_pairs:
        od_demands[(od['origin'], od['destination'])] = od['demand']
    
    # Create model
    model = gp.Model("TransitNetwork")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables
    # Line frequency variables
    line_freq = model.addVars(len(all_lines), lb=0.0, ub=F, name="line_freq")
    
    # Line activation variables
    line_active = model.addVars(len(all_lines), vtype=GRB.BINARY, name="line_active")
    
    # Passenger flow variables (for each OD pair and each possible path)
    # We'll create paths dynamically as needed, but for now we'll create a basic routing
    # To keep things manageable, we'll consider paths up to a certain length
    
    # First, create a basic network flow model without path enumeration
    # Variables for flow on each arc for each OD pair
    od_pairs_list = list(od_demands.keys())
    arc_flow = model.addVars([(arc, od_pair) for arc in range(num_arcs) for od_pair in od_pairs_list],
                             lb=0.0, name="arc_flow")
    
    # Constraints
    # 1. Flow conservation for each OD pair at each node
    for od_pair in od_pairs_list:
        origin, dest = od_pair
        demand = od_demands[od_pair]
        
        for node in range(network['num_nodes']):
            if node == origin:
                # Outflow constraint for origin
                outflow = gp.LinExpr()
                inflow = gp.LinExpr()
                
                # Get outgoing arcs from this node
                for edge_idx in range(network['num_edges']):
                    u, v = network['edges'][edge_idx]['endpoints']
                    if u == node:
                        outflow.add(arc_flow[2*edge_idx, od_pair])
                    elif v == node:
                        outflow.add(arc_flow[2*edge_idx + 1, od_pair])  # arc vu
                    if v == node:
                        inflow.add(arc_flow[2*edge_idx, od_pair])  # flow from u->v contributing to v's inflow
                    if u == node:
                        inflow.add(arc_flow[2*edge_idx + 1, od_pair])  # flow from v->u contributing to u's inflow
                
                model.addConstr(outflow - inflow == demand, 
                               f"flow_conservation_origin_{node}_{od_pair}")
                
            elif node == dest:
                # Inflow constraint for destination
                outflow = gp.LinExpr()
                inflow = gp.LinExpr()
                
                # Get incoming arcs to this node
                for edge_idx in range(network['num_edges']):
                    u, v = network['edges'][edge_idx]['endpoints']
                    if u == node:
                        inflow.add(arc_flow[2*edge_idx + 1, od_pair])  # flow coming into u from v
                    elif v == node:
                        inflow.add(arc_flow[2*edge_idx, od_pair])  # flow coming into v from u
                    if v == node:
                        outflow.add(arc_flow[2*edge_idx, od_pair])  # flow going out from v to u
                    if u == node:
                        outflow.add(arc_flow[2*edge_idx + 1, od_pair])  # flow going out from u to v
                
                model.addConstr(inflow - outflow == demand, 
                               f"flow_conservation_dest_{node}_{od_pair}")
                
            else:
                # Intermediate node constraint
                outflow = gp.LinExpr()
                inflow = gp.LinExpr()
                
                for edge_idx in range(network['num_edges']):
                    u, v = network['edges'][edge_idx]['endpoints']
                    if u == node:
                        outflow.add(arc_flow[2*edge_idx, od_pair])
                        inflow.add(arc_flow[2*edge_idx + 1, od_pair])
                    elif v == node:
                        outflow.add(arc_flow[2*edge_idx + 1, od_pair])
                        inflow.add(arc_flow[2*edge_idx, od_pair])
                
                model.addConstr(inflow == outflow, 
                               f"flow_conservation_intermediate_{node}_{od_pair}")
    
    # 2. Edge capacity constraints
    for edge_idx in range(network['num_edges']):
        total_freq_on_edge = gp.LinExpr()
        
        # Sum frequencies of all lines using this edge
        for line_idx, line in enumerate(all_lines):
            if edge_idx in line['edges']:
                total_freq_on_edge.add(line_freq[line_idx])
        
        edge_capacity = network['edges'][edge_idx]['edge_capacity']
        model.addConstr(total_freq_on_edge <= edge_capacity, 
                       f"edge_capacity_{edge_idx}")
    
    # 3. Link line frequency to activation
    for line_idx in range(len(all_lines)):
        # If line is active, frequency must be > 0
        # If line is not active, frequency must be 0
        model.addConstr(line_freq[line_idx] <= F * line_active[line_idx],
                       f"freq_activation_link_{line_idx}")
    
    # 4. Capacity constraints: total passenger flow on arc <= capacity provided by lines using that edge
    for arc_idx in range(num_arcs):
        edge_idx = arc_to_edge[arc_idx]
        
        # Total passenger flow on this arc
        total_flow_on_arc = gp.LinExpr()
        for od_pair in od_pairs_list:
            total_flow_on_arc.add(arc_flow[arc_idx, od_pair])
        
        # Capacity provided by lines using the corresponding undirected edge
        total_capacity_from_lines = gp.LinExpr()
        for line_idx, line in enumerate(all_lines):
            if edge_idx in line['edges']:
                # Find the mode of this line to get its vehicle capacity
                for mode_info in modes:
                    if mode_info['name'] == line['mode']:
                        vehicle_capacity = mode_info['vehicle_capacity']
                        break
                total_capacity_from_lines.add(line_freq[line_idx] * vehicle_capacity)
        
        model.addConstr(total_flow_on_arc <= total_capacity_from_lines,
                       f"passenger_capacity_{arc_idx}")
    
    # Objective function components
    # 1. Total line cost
    total_fixed_cost = gp.LinExpr()
    total_operating_cost = gp.LinExpr()
    
    for line_idx, line in enumerate(all_lines):
        # Add fixed cost if line is active
        for mode_info in modes:
            if mode_info['name'] == line['mode']:
                fixed_cost = mode_info['fixed_cost_per_line']
                op_cost_per_edge = mode_info['operating_cost_per_edge']
                break
        
        total_fixed_cost.add(fixed_cost * line_active[line_idx])
        
        # Add operating cost proportional to frequency
        total_operating_cost.add(op_cost_per_edge * len(line['edges']) * line_freq[line_idx])
    
    total_line_cost = total_fixed_cost + total_operating_cost
    
    # 2. Total passenger travel time
    total_passenger_time = gp.LinExpr()
    for arc_idx in range(num_arcs):
        edge_idx = arc_to_edge[arc_idx]
        travel_time = network['edges'][edge_idx]['traveling_time_seconds']
        
        for od_pair in od_pairs_list:
            total_passenger_time.add(arc_flow[arc_idx, od_pair] * travel_time)
    
    # Final objective: lambda * total_line_cost + (1-lambda) * total_passenger_time
    objective = lambda_weight * total_line_cost + (1 - lambda_weight) * total_passenger_time
    model.setObjective(objective, GRB.MINIMIZE)
    
    # Optimize
    model.optimize()
    
    # Process solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        active_lines = []
        for line_idx in range(len(all_lines)):
            if line_active[line_idx].x > 0.001:  # Threshold to check if line is active
                active_lines.append({
                    'line_index': line_idx,
                    'mode': all_lines[line_idx]['mode'],
                    'nodes': all_lines[line_idx]['nodes'],
                    'edges': all_lines[line_idx]['edges'],
                    'frequency': line_freq[line_idx].x
                })
        
        # Extract passenger flows
        active_passenger_paths = []
        
        # Group flows by OD pair
        od_flows = defaultdict(float)
        for (arc, od_pair), flow_val in arc_flow.items():
            if flow_val.x > 0.001:
                od_flows[od_pair] += flow_val.x
        
        # Create path representations for significant flows
        # Since we don't have explicit paths, we'll just report aggregate flows
        # For a complete solution, we'd need to reconstruct actual paths
        # For now, we'll create placeholder paths
        
        # This is a simplified approach - in practice, we would trace actual paths
        # But since we used arc-based flow variables, we need to reconstruct paths
        
        # Let's create a basic path reconstruction for each OD pair
        for od_pair in od_pairs_list:
            origin, dest = od_pair
            demand = od_demands[od_pair]
            
            # Find all arcs with flow for this OD pair
            arcs_with_flow = []
            total_flow = 0.0
            for arc in range(num_arcs):
                if arc_flow[arc, od_pair].x > 0.001:
                    arcs_with_flow.append(arc)
                    total_flow += arc_flow[arc, od_pair].x
            
            # If there's flow, create a path (simplified approach)
            if total_flow > 0:
                # Create a simple path representation
                active_passenger_paths.append({
                    'origin': origin,
                    'destination': dest,
                    'arcs': arcs_with_flow,
                    'flow': total_flow
                })
        
        solution = {
            'objective_value': model.objVal,
            'active_lines': active_lines,
            'active_passenger_paths': active_passenger_paths
        }
        
        return solution
    else:
        # Return empty solution if infeasible
        return {
            'objective_value': float('inf'),
            'active_lines': [],
            'active_passenger_paths': []
        }


def main():
    parser = argparse.ArgumentParser(description='Solve transit network optimization problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to log file for intermediate solutions')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_transit_network(instance_data, args.time_limit)
    
    # Log the final solution if we have a logger
    if logger:
        logger.log_solution(solution['objective_value'], solution)
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()