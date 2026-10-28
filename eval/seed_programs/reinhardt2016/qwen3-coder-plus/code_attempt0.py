import argparse
import json
import math
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
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

    # Edge sets data
    edge_sets = []
    for es in instance["edge_sets"]:
        # Convert edges from node ids to indices
        edges_as_indices = []
        for edge in es['edges']:
            u_id, v_id = edge[0], edge[1]
            u_idx, v_idx = node_id_to_idx[u_id], node_id_to_idx[v_id]
            edges_as_indices.append((u_idx, v_idx))
        edge_sets.append({
            'id': es['id'],
            'access_cost': es['access_cost'],
            'edges': edges_as_indices,
            'edges_set': set(edges_as_indices + [(v, u) for (u, v) in edges_as_indices])  # Undirected edges
        })

    # Free edges data
    free_edges_indices = []
    for edge in instance["free_edges"]:
        u_id, v_id = edge[0], edge[1]
        u_idx, v_idx = node_id_to_idx[u_id], node_id_to_idx[v_id]
        free_edges_indices.append((u_idx, v_idx))

    # Build edge set membership map
    # edge_in_set[(i, j)] -> set of edge set IDs containing edge (i,j)
    edge_in_set = {}
    for set_id, edge_set_data in enumerate(edge_sets):
        for u_idx, v_idx in edge_set_data['edges']:
            # Add to both directions since edge is undirected
            if (u_idx, v_idx) not in edge_in_set:
                edge_in_set[(u_idx, v_idx)] = set()
            edge_in_set[(u_idx, v_idx)].add(set_id)
            
            if (v_idx, u_idx) not in edge_in_set:
                edge_in_set[(v_idx, u_idx)] = set()
            edge_in_set[(v_idx, u_idx)].add(set_id)

    # Depot index is always 0 based on problem specification (id=0 is depot)
    depot_idx = 0

    print(f"Building model...")

    # Create model
    model = gp.Model("VRPTW_with_edge_sets")
    
    # Set model parameters
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions

    # Add variables
    # x[k,i,j]: 1 if vehicle k uses edge (i,j)
    x = {}
    for k in range(num_vehicles):
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j:
                    x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")

    # y[s]: 1 if edge set s is used
    y = {}
    for s in range(len(edge_sets)):
        y[s] = model.addVar(vtype=GRB.BINARY, name=f"y_{s}")

    # t[i,k]: arrival time at node i for vehicle k
    t = {}
    for k in range(num_vehicles):
        for i in range(num_customers + 1):
            t[k, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=nodes[i]["time_window_start"], 
                                   ub=min(nodes[i]["time_window_end"], schedule_horizon), name=f"t_{k}_{i}")

    # Flow and capacity constraints might require u[k,i] variables for load
    # u[k,i]: load after visiting node i with vehicle k
    u = {}
    for k in range(num_vehicles):
        for i in range(num_customers + 1):
            u[k, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=vehicle_capacity, name=f"u_{k}_{i}")

    model.update()

    print(f"Adding constraints...")

    # Constraints
    # 1. Each customer is visited exactly once
    for i in range(1, num_customers + 1):  # Customer nodes only (skip depot)
        model.addConstr(
            gp.quicksum(x[k, i, j] for k in range(num_vehicles) 
                       for j in range(num_customers + 1) if i != j) == 1, 
            name=f"visit_customer_{i}"
        )

    # 2. Flow conservation for all nodes and all vehicles
    for k in range(num_vehicles):
        for i in range(num_customers + 1):
            model.addConstr(
                gp.quicksum(x[k, i, j] for j in range(num_customers + 1) if i != j) ==
                gp.quicksum(x[k, j, i] for j in range(num_customers + 1) if i != j),
                name=f"flow_conservation_k_{k}_i_{i}"
            )
        
        # Each vehicle leaves depot once if it is used
        model.addConstr(
            gp.quicksum(x[k, depot_idx, j] for j in range(1, num_customers + 1)) <= 1,
            name=f"start_once_k_{k}"
        )

    # 3. Vehicle capacity constraint
    for k in range(num_vehicles):
        model.addConstr(
            gp.quicksum(nodes[i]['demand'] * gp.quicksum(x[k, j, i] for j in range(num_customers + 1) if j != i) 
                       for i in range(1, num_customers + 1)) <= vehicle_capacity,
            name=f"capacity_k_{k}"
        )

    # 4. Time window constraints and time consistency
    for k in range(num_vehicles):
        for i in range(num_customers + 1):
            # Time window bounds already handled by variable definition
            pass
        
        # Time consistency along arcs: t[j] >= t[i] + dist[i][j] + service time at i  
        # Only add this constraint when arc is used
        # Linearization needed
        big_m = schedule_horizon  # Use a reasonable upper bound
        
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j:
                    # If vehicle k uses arc (i,j), then t[k,j] >= t[k,i] + dist[i][j] + service_time[i]
                    # t[k,j] >= t[k,i] + dist[i][j] + service_time[i] - M*(1-x[k,i,j])
                    # equivalently: t[k,j] >= t[k,i] + dist[i][j] + service_time[i] - M + M*x[k,i,j]
                    # or: t[k,j] >= t[k,i] + dist[i][j] + service_time[i] - M*(1-x[k,i,j])
                    model.addConstr(
                        t[k, j] >= t[k, i] + distance_matrix[i][j] + nodes[i]["service_time"] - 
                        big_m * (1 - x[k, i, j]),
                        name=f"time_link_k_{k}_i_{i}_j_{j}"
                    )

    # 5. Link edge set use to edge use
    for s_idx, edge_set_info in enumerate(edge_sets):
        for (u, v) in edge_set_info['edges']:
            # For each edge in set s, ensure we have access to the set to use it
            model.addConstr(
                x[k, u, v] <= y[s_idx] for k in range(num_vehicles)
            )
            model.addConstr(
                x[k, v, u] <= y[s_idx] for k in range(num_vehicles)  # since edge is undirected, both directions
            )
    
    # 6. Subtour elimination would be too numerous for full MTZ, so relax for now
    # We may need to add cutting planes dynamically, but for this implementation,
    # we'll keep the model simple and handle potential subtours with lazy constraints later if needed.

    # 7. Initial load at depot is 0 for each vehicle
    # This is automatically satisfied due to binary nature and load computation
    
    # 8. Update load variables
    for k in range(num_vehicles):
        # Vehicle starts with 0 load at depot
        model.addConstr(u[k, depot_idx] == 0, name=f"init_load_{k}")
        
        # Conservation of loads: u[k,j] >= u[k,i] + demand[j] when arc (i,j) is used
        # Formulation: u[k,j] >= u[k,i] + demand[j] - bigM*(1 - x[k,i,j])
        for i in range(num_customers + 1):
            for j in range(1, num_customers + 1):  # j starts from 1 (not depot)
                if i != j:
                    model.addConstr(
                        u[k, j] >= u[k, i] + nodes[j]['demand'] - 
                            vehicle_capacity * (1 - x[k, i, j]),
                        name=f"load_flow_{k}_{i}_{j}"
                    )
                    
    # Objective function
    # Minimize sum of traversal costs + edge set access costs
    total_traversal_cost = gp.quicksum(
        x[k, i, j] * distance_matrix[i][j]
        for k in range(num_vehicles)
        for i in range(num_customers + 1)
        for j in range(num_customers + 1)
        if i != j
    )
    
    total_access_cost = gp.quicksum(
        y[s] * edge_sets[s]['access_cost']
        for s in range(len(edge_sets))
    )
    
    model.setObjective(total_traversal_cost + total_access_cost, GRB.MINIMIZE)

    print(f"Optimization starting...")
    start_time = time.time()
    
    # Optimize
    model.optimize()
    
    print(f"Solving completed. Status: {model.Status}")
    
    # Process solution
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        objective_value = model.ObjVal if model.SolCount > 0 else float('inf')
        
        if model.SolCount == 0:
            print("No solution found!")
            return {"objective_value": float('inf'), "routes": [], "edge_sets_used": []}
            
        routes = []
        edge_sets_used = []
        
        # Determine edge sets used
        for s_idx in range(len(edge_sets)):
            if y[s_idx].X > 0.5:
                edge_sets_used.append(edge_sets[s_idx]['id'])
        
        # Extract routes
        for k in range(num_vehicles):
            vehicle_route = [0]  # Start at depot (idx 0)
            current_node = 0
            
            # Find next node in the route
            while True:
                next_node_found = False
                for j in range(num_customers + 1):
                    if j != current_node and x[k, current_node, j].X > 0.5:
                        vehicle_route.append(j)
                        current_node = j
                        next_node_found = True
                        break
                        
                if current_node == 0 and len(vehicle_route) > 1:  # Back to depot
                    break
                
                if not next_node_found:
                    # If not found a next node and didn't reach depot, this vehicle may not be used
                    break
            
            if len(vehicle_route) > 1:  # Only add non-empty routes
                # Create arcs for this route
                arcs = [[vehicle_route[i], vehicle_route[i+1]] for i in range(len(vehicle_route)-1)]
                
                # Get visit times for this vehicle
                visit_times = {f"{nodes[i]['id']}": round(t[k, i].X) for i in vehicle_route}
                
                routes.append({
                    "vehicle": k,
                    "sequence": [nodes[i]['id'] for i in vehicle_route],
                    "arcs": arcs,
                    "visit_times": visit_times
                })
        
        solution = {
            "objective_value": objective_value,
            "routes": routes,
            "edge_sets_used": edge_sets_used[:]
        }
        
        print(f"Found solution with value: {objective_value}")
        return solution
    else:
        return {"objective_value": float('inf'), "routes": [], "edge_sets_used": []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    # Initialize logger if a log path is provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Solve the instance
    solution = solve_vrptw_with_edge_sets(args)

    # Log the solution objective if logger exists and solution exists
    if logger and solution and 'objective_value' in solution and solution['objective_value'] < float('inf'):
        logger.log(solution['objective_value'])

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()