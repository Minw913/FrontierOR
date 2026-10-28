import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger


def solve_vrptw_with_edge_sets(args):
    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_customers = instance["num_customers"]
    num_vehicles = instance["num_vehicles"]
    vehicle_capacity = instance["vehicle_capacity"]
    num_edge_sets = instance["num_edge_sets"]
    schedule_horizon = instance["schedule_horizon"]
    nodes = instance["nodes"]
    distance_matrix = instance["distance_matrix"]

    # Create a map from nodeId to index in nodes array
    node_id_to_idx = {node['id']: i for i, node in enumerate(nodes)}
    idx_to_node_id = {i: node['id'] for i, node in enumerate(nodes)}

    # Edge sets data
    edge_sets = []
    set_idx_to_original_id = {}  # mapping from our 0-based index to original id
    for i, es in enumerate(instance["edge_sets"]):
        # Convert edges from node ids to indices
        edges_as_indices = []
        for edge in es['edges']:
            u_id, v_id = edge[0], edge[1]
            u_idx, v_idx = node_id_to_idx[u_id], node_id_to_idx[v_id]
            edges_as_indices.append((u_idx, v_idx))
        edge_sets.append({
            'id': i,  # Our internal index
            'original_id': es['id'],  # Keep track of original id
            'access_cost': es['access_cost'],
            'edges': edges_as_indices,
        })
        set_idx_to_original_id[i] = es['id']

    # Free edges data (not used explicitly but part of data)
    free_edges_indices = []
    for edge in instance["free_edges"]:
        u_id, v_id = edge[0], edge[1]
        u_idx, v_idx = node_id_to_idx[u_id], node_id_to_idx[v_id]
        free_edges_indices.append((u_idx, v_idx))

    # Build edge set membership map for directed constraints
    # edge_dir_to_sets[(i, j)] -> set of edge set IDs containing edge (i,j)
    edge_dir_to_sets = {}
    for set_idx, edge_set_data in enumerate(edge_sets):
        for u_idx, v_idx in edge_set_data['edges']:
            # This is an undirected edge
            if (u_idx, v_idx) not in edge_dir_to_sets:
                edge_dir_to_sets[(u_idx, v_idx)] = set()
            edge_dir_to_sets[(u_idx, v_idx)].add(set_idx)
            
            if (v_idx, u_idx) not in edge_dir_to_sets:
                edge_dir_to_sets[(v_idx, u_idx)] = set()
            edge_dir_to_sets[(v_idx, u_idx)].add(set_idx)

    # Depot index is always 0 based on problem specification (nodes[0] is depot)
    depot_idx = 0

    # Create model
    model = gp.Model("VRPTW_with_edge_sets")
    
    # Set model parameters
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)  # Enforce single thread usage
    model.Params.MIPGap = 1e-4    # Allow suboptimal solution for faster time

    # Add variables
    # x[k,i,j]: 1 if vehicle k uses edge (i,j), directed
    x = {}
    for k in range(num_vehicles):
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j:  # No self-loops
                    x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")

    # y[s]: 1 if edge set s is used
    y = {}
    for s_idx in range(len(edge_sets)):
        y[s_idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{s_idx}")

    # t[i,k]: arrival time at node i for vehicle k
    t = {}
    for k in range(num_vehicles):
        for i in range(num_customers + 1):
            t[k, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=nodes[i]["time_window_start"], 
                                   ub=min(nodes[i]["time_window_end"], schedule_horizon), name=f"t_{k}_{i}")

    # u[k,i]: load on vehicle k when leaving node i
    u = {}
    for k in range(num_vehicles):
        for i in range(num_customers + 1):
            u[k, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=vehicle_capacity, name=f"u_{k}_{i}")

    model.update()

    # Constraints
    # 1. Each customer is visited exactly once
    for i in range(1, num_customers + 1):  # Skip depot
        model.addConstr(gp.quicksum(x[k, j, i] for k in range(num_vehicles) 
                   for j in range(num_customers + 1) if i != j) == 1, 
                       name=f"visit_customer_{i}_once")

    # 2. Flow conservation: # incoming = # outgoing for all nodes and vehicles
    for k in range(num_vehicles):
        for i in range(num_customers + 1):
            in_flow = gp.quicksum(x[k, j, i] for j in range(num_customers + 1) if i != j)
            out_flow = gp.quicksum(x[k, i, j] for j in range(num_customers + 1) if i != j)
            model.addConstr(in_flow == out_flow, name=f"flow_cons_{k}_{i}")
        
        # If a vehicle leaves depot, it must eventually come back (flow balance at depot)
        out_flow_from_depot = gp.quicksum(x[k, depot_idx, j] for j in range(1, num_customers + 1))
        in_flow_to_depot = gp.quicksum(x[k, j, depot_idx] for j in range(1, num_customers + 1))
        model.addConstr(out_flow_from_depot == in_flow_to_depot, name=f"out_eq_in_vehicle_{k}")

    # 3. Vehicle k must start at the depot if used
    for k in range(num_vehicles):
        # A vehicle is "used" if it leaves the depot
        model.addConstr(gp.quicksum(x[k, depot_idx, j] for j in range(1, num_customers + 1)) >= 0, name=f"start_{k}")
        
        # If not used, then no flow throughout the route
        # Use the fact that it leaves depot at most once
        model.addConstr(gp.quicksum(x[k, depot_idx, j] for j in range(1, num_customers + 1)) <= 1, name=f"leave_dep_at_most_once_{k}")

    # 4. Capacity constraints: total demand on vehicle k <= vehicle_capacity
    for k in range(num_vehicles):
        model.addConstr(gp.quicksum(x[k, i, j] * nodes[j]['demand'] 
                              for i in range(num_customers + 1) 
                              for j in range(1, num_customers + 1) 
                              if i != j) <= vehicle_capacity, name=f"cap_k_{k}")

    # 5. Time constraints: t[k,j] >= t[k,i] + dist[i][j] + service_time[i] - M*(1-x[k,i,j])
    big_M_time = schedule_horizon  # Safe upper bound for time window
    for k in range(num_vehicles):
        for i in range(num_customers + 1):
            for j in range(1, num_customers + 1):  # Skip depot as starting point of time propagation
                if i != j:
                    model.addConstr(
                        t[k, j] >= t[k, i] + distance_matrix[i][j] + nodes[i]["service_time"] - 
                        big_M_time * (1 - x[k, i, j]),
                        name=f"time_prop_{k}_{i}_{j}"
                    )
    
    # Additional constraint for time propagation between depot and first customer
    for k in range(num_vehicles):
        for j in range(1, num_customers + 1):
            model.addConstr(
                t[k, j] >= t[k, depot_idx] + distance_matrix[depot_idx][j] + nodes[depot_idx]["service_time"] - 
                big_M_time * (1 - x[k, depot_idx, j]),
                name=f"depot_to_cust_time_{k}_{j}"
            )
    
    # 6. Link edge selection to edge set usage
    # Each arc can only be used if its corresponding edge set (if any) is selected
    for k in range(num_vehicles):
        for (i, j) in edge_dir_to_sets:
            if i != j:
                set_ids = edge_dir_to_sets[(i, j)]
                for s in set_ids:
                    model.addConstr(x[k, i, j] <= y[s], name=f"arc_uses_set_{k}_{i}_{j}_set_{s}")

    # 7. Update vehicle load constraints
    big_M_load = vehicle_capacity  # When we don't take the arc, no load constraints apply
    for k in range(num_vehicles):
        for i in range(num_customers + 1):
            for j in range(1, num_customers + 1):
                if i != j:
                    # u[k,j] >= u[k,i] + demand[j] - big_M_load*(1-x[k,i,j])
                    model.addConstr(
                        u[k, j] >= u[k, i] + nodes[j]['demand'] - big_M_load * (1 - x[k, i, j]),
                        name=f"load_upd_{k}_{i}_{j}_1"
                    )
                    
                    # u[k,j] >= 0 (ensure positivity)
                    # u[k,j] <= vehicle_capacity (already bounded by variable definition)
                
                # Load after serving depot is unchanged except for what was loaded there
                if i == depot_idx and j != depot_idx:
                    model.addConstr(
                        u[k, j] >= nodes[j]['demand'] * x[k, depot_idx, j],
                        name=f"load_from_depot_{k}_{j}_via_{depot_idx}"
                    )

        # Initial load after departure from depot should be sum of all demands served from depot
        total_initial_demand = gp.quicksum(nodes[j]['demand'] * x[k, depot_idx, j]
                                          for j in range(1, num_customers + 1))
        model.addConstr(u[k, depot_idx] >= total_initial_demand, name=f"depot_load_{k}")
        
        # Load at departure from depot should not exceed capacity
        model.addConstr(u[k, depot_idx] <= vehicle_capacity, name=f"depot_cap_{k}")

    # Additional load and flow constraints
    for k in range(num_vehicles):
        # Load at depot (after initial loading) must respect capacity
        model.addConstr(u[k, depot_idx] <= vehicle_capacity, name=f"cap_dep_{k}_dep")
    
    # Objective function
    # Minimize sum of traversal costs + access costs
    total_traversal_cost = gp.LinExpr()
    for k in range(num_vehicles):
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j:
                    total_traversal_cost.addTerms(distance_matrix[i][j], x[k, i, j],
                                                  name=f"traversal_{k}_{i}_{j}")
    
    total_access_cost = gp.quicksum(edge_sets[s_idx]['access_cost'] * y[s_idx] 
                                         for s_idx in range(len(edge_sets)))
    
    model.setObjective(total_traversal_cost + total_access_cost, GRB.MINIMIZE)

    # Solve the model
    model.optimize()
    
    # Process solution
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        if model.SolCount > 0:
            objective_value = model.ObjVal

            # Collect edge sets used
            edge_sets_used = []
            for s_idx in range(len(edge_sets)):
                if y[s_idx].X > 0.5:
                    edge_sets_used.append(set_idx_to_original_id[s_idx])  # use original id
            
            # Extract routes
            routes = []
            for k in range(num_vehicles):
                # Check if vehicle k is active - check if it leaves depot
                if gp.quicksum(x[k, depot_idx, j].X for j in range(1, num_customers + 1)).getValue() < 0.9:
                    continue  # Vehicle k is not used
                    
                # Start building route from depot
                route = [idx_to_node_id[depot_idx]]
                visit_times = {str(idx_to_node_id[depot_idx]): t[k, depot_idx].X}
                
                current = depot_idx
                visited_customers = {depot_idx}
                
                # Loop to build the path
                while True:
                    next_node = None
                    for j in range(1, num_customers + 1):  # Only visit customers (excluding depot)
                        if x[k, current, j].X > 0.5 and j not in visited_customers:
                            next_node = j
                            break
                    
                    if next_node is None:
                        # If none found, try to return to depot
                        if x[k, current, depot_idx].X > 0.5:
                            route.append(idx_to_node_id[depot_idx])
                            visit_times[str(idx_to_node_id[depot_idx])] = t[k, depot_idx].X
                        break  # end of route
                    
                    route.append(idx_to_node_id[next_node])
                    visit_times[str(idx_to_node_id[next_node])] = t[k, next_node].X
                    visited_customers.add(next_node)
                    current = next_node
                    
                    # Safety to prevent infinite loops
                    if len(route) > 2 + num_customers:  # Depot + customers + return to depot
                        break
                
                # Construct arcs based on route
                arcs = []
                for i in range(len(route) - 1):
                    arcs.append([route[i], route[i+1]])
                        
                if len(route) > 2:  # Meaning it went out and back
                    routes.append({
                        "vehicle": k,
                        "sequence": route,
                        "arcs": arcs,
                        "visit_times": visit_times
                    })

            solution = {
                "objective_value": objective_value,
                "routes": routes,
                "edge_sets_used": sorted(edge_sets_used)  # Sort for consistent output
            }
            
            if solution:
                return solution
        else:
            # No solution found
            return {"objective_value": float('inf'), "routes": [], "edge_sets_used": []}
    else:
        return {"objective_value": float('inf'), "routes": [], "edge_sets_used": []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    # Initialize logger if needed
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Solve
    solution = solve_vrptw_with_edge_sets(args)

    # Log objective value if found and logger is available
    if logger and 'objective_value' in solution and isinstance(solution['objective_value'], (int, float)) and solution['objective_value'] < float('inf'):
        logger.log(solution['objective_value'])

    # Write results
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()