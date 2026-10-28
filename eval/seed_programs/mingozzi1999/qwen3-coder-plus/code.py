import argparse
import json
import math
import random
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def solve_vrpb(instance):
    """
    Solve the Vehicle Routing Problem with Backhauls (VRPB) using Gurobi.
    """
    # Extract instance parameters
    n_linehaul = instance["num_linehaul_customers"]
    n_backhaul = instance["num_backhaul_customers"]
    n_tot_cus = instance["num_total_customers"]
    n_veh = instance["num_vehicles"]
    v_cap = instance["vehicle_capacity"]
    dist_m = instance["distance_matrix"]
    
    depot_idx = instance["depot"]["index"]
    
    # Customer indexing
    L = list(range(0, n_linehaul))  # Linehaul customers (0, ..., n_linehaul-1)
    B = list(range(n_linehaul, n_linehaul + n_backhaul))  # Backhaul customers
    V = list(range(n_tot_cus))  # All customers
    
    # Demands
    l_demands = [c['demand'] for c in instance["linehaul_customers"]]
    b_demands = [c['demand'] for c in instance["backhaul_customers"]]
    
    # Create Gurobi model
    model = gp.Model('VRPB')
    model.Params.LogToConsole = 0  # Turn off console logging
    model.setParam(GRB.Param.Threads, 1)
    
    # Decision Variables - for a multi-commodity flow-like model
    # x[i,j,k]: binary if vehicle k travels from i to j
    x = {}
    for i in range(len(dist_m)):
        for j in range(len(dist_m[0])):
            if i != j:  # No self loops except depot ones
                for k in range(n_veh):
                    x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    # Flow variables for load
    # f_l[i,j,k]: flow of linehaul goods from i to j for vehicle k
    # f_b[i,j,k]: flow of backhaul goods from i to j for vehicle k
    f_l = {}
    f_b = {}
    for i in range(len(dist_m)):
        for j in range(len(dist_m[0])):
            if i != j:
                for k in range(n_veh):
                    f_l[i, j, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f"f_l_{i}_{j}_{k}", lb=0)
                    f_b[i, j, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f"f_b_{i}_{j}_{k}", lb=0)

    # Objective function
    obj = 0
    for k in range(n_veh):
        for i in range(len(dist_m)):
            for j in range(len(dist_m[0])):
                if i != j:
                    obj += dist_m[i][j] * x[i, j, k]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints:

    # 1. Each linehaul customer is visited exactly once
    for i in L:
        constraint_expr = 0
        for k in range(n_veh):
            for j in range(len(dist_m[0])):
                if i != j:
                    constraint_expr += x[i, j, k]
        model.addConstr(constraint_expr == 1, name=f"visit_linehaul_{i}")

    # 2. Each backhaul customer is visited exactly once
    for j in B:
        constraint_expr = 0
        for k in range(n_veh):
            for i in range(len(dist_m)):
                if i != j:
                    constraint_expr += x[i, j, k]
        model.addConstr(constraint_expr == 1, name=f"visit_backhaul_{j}")
        
    # 3. Flow conservation at depot for all vehicles
    for k in range(n_veh):
        inflow_depot = gp.LinExpr()
        outflow_depot = gp.LinExpr()
        for i in range(len(dist_m)):
            if i != depot_idx:
                inflow_depot += x[i, depot_idx, k]
                outflow_depot += x[depot_idx, i, k]
        model.addConstr(inflow_depot == x[depot_idx, depot_idx, k], name=f"net_flow_depot_in_{k}")
        model.addConstr(outflow_depot == x[depot_idx, depot_idx, k], name=f"net_flow_depot_out_{k}")
        
        # Ensure a vehicle enters if it leaves and vice versa
        model.addConstr(x[depot_idx, depot_idx, k] <= 500)  # Upper limit if used
        # Ensure consistent usage of the vehicle
        model.addConstr(gp.quicksum(x[i, j, k] for i in range(len(dist_m)) for j in range(len(dist_m[0])) if i != j and i != depot_idx) >= gp.quicksum(x[depot_idx, i, k] for i in range(len(dist_m[0])) if i != depot_idx))

    # We adjust flow cons for non-depot nodes differently:
    # For linehaul - only outflow if not depot and is linehaul
    for i in L:
        for k in range(n_veh):
            inflow = gp.quicksum(x[j, i, k] for j in range(len(dist_m)) if j != i)
            outflow = gp.quicksum(x[i, j, k] for j in range(len(dist_m[0])) if j != i)
            model.addConstr(inflow == outflow, name=f"flow_conservation_L_{i}_{k}")
            
    for j in B:
        for k in range(n_veh):
            inflow = gp.quicksum(x[i, j, k] for i in range(len(dist_m)) if i != j)
            outflow = gp.quicksum(x[j, i, k] for i in range(len(dist_m[0])) if i != j)
            model.addConstr(inflow == outflow, name=f"flow_conservation_B_{j}_{k}")

    # 4. Flow conservation for linehaul flows (at depot, non-zero if vehicle leaves with linehaul load)
    for k in range(n_veh):
        total_l_loads = sum(l_demands[i] for i in L if next((True for c in instance["linehaul_customers"] if c['index'] == i), False))
        l_load_flows = gp.quicksum(f_l[i, j, k] for i in range(len(dist_m)) for j in range(len(dist_m[0])) if i != j and i==depot_idx)
        actual_load = gp.quicksum(f_l[i, j, k] for i in range(len(dist_m)) for j in range(len(dist_m[0])) if i != j and i==depot_idx)
        model.addConstr(actual_load == gp.quicksum(x[depot_idx, i, k] * l_demands[L.index(i)] for i in L if i < len(l_demands)), name=f"l_flow_origin_{k}")

    # Simplified flow for depot for each vehicle k
    for k in range(n_veh):
        # Linehaul flow at depot (what flows out when leaving depot)
        out_l_flow = gp.quicksum(f_l[depot_idx, i, k] for i in L)
        outflow = gp.quicksum(x[depot_idx, i, k] for i in len(dist_m[0]) if i < len(dist_m[0]))
        # The flow should match the total demand on the route
        model.addConstr(out_l_flow <= v_cap, name=f"depot_l_flow_balance_{k}")
        
    # Now explicitly defining the more sophisticated flow balance constraints:
    for k in range(n_veh):
        for i in L:
            # Inflow and outflow of L for each customer i on a route
            netoutflow = gp.quicksum(f_l[i, j, k] for j in range(len(dist_m[0])) if j != i) - \
                         gp.quicksum(f_l[j, i, k] for j in range(len(dist_m)) if j != i)
            model.addConstr(netoutflow == l_demands[L.index(i)] * gp.quicksum(x[i, j, k] for j in range(len(dist_m[0])) if j != i),
                            name=f"net_flow_balance_L_{i}_{k}")
        
        for j in B:  
            # For backhauls, net inflow should match demand
            netinflow = gp.quicksum(f_b[i, j, k] for i in range(len(dist_m)) if i != j) - \
                        gp.quicksum(f_b[j, i, k] for i in range(len(dist_m[0])) if i != j)
            model.addConstr(netinflow == b_demands[B.index(j)] * gp.quicksum(x[i, j, k] for i in range(len(dist_m)) if i != j),
                            name=f"net_flow_balance_B_{j}_{k}")
    
    # Capacity constraints on edges for flows
    bigM = v_cap  # Maximum possible load
    for k in range(n_veh):
        for i in range(len(dist_m)):
            for j in range(len(dist_m[0])):
                if i != j:
                    model.addConstr(f_l[i, j, k] <= bigM * x[i, j, k], name=f"capacity_link_f_l_{i}_{j}_{k}")
                    model.addConstr(f_b[i, j, k] <= bigM * x[i, j, k], name=f"capacity_link_f_b_{i}_{j}_{k}")

    # Flow for linehaul does not exceed vehicle capacity at origin
    for k in range(n_veh):
        total_l_flow_leaving_from_depot = gp.quicksum(f_l[depot_idx, i, k] for i in range(len(dist_m[0])) if i != depot_idx)
        model.addConstr(total_l_flow_leaving_from_depot <= v_cap, name=f"capacity_vehicle_l_{k}")
        
        total_b_flow_entering_to_depot = gp.quicksum(f_b[i, depot_idx, k] for i in range(len(dist_m)) if i != depot_idx)
        model.addConstr(total_b_flow_entering_to_depot <= v_cap, name=f"capacity_vehicle_b_{k}")
    
    # Subtour elimination - we use a simple constraint ensuring that for each non-empty subset S,
    # the sum of edges from S to V \ S is at least the required min vehicles.
    # This is a complex combinatorial constraint, often we rely on cuts or use formulations 
    # that implicitly handle subtours via flow variables.
    # Our flow formulation already takes care of connectedness within routes.
    
    # Ensure that a route visiting backhaul customers must visit linehaul customers too
    for k in range(n_veh):
        # If there are transitions to backhaul customers, check if linehaul path exists
        # For simplicity in flow formulation: ensure no direct path from depot solely to backhaul without first visiting a linehaul node
        # Sum of outflow from depot of backhaul items is zero if no linehaul visited by the vehicle
        # A more precise implementation of the rule "no backhaul-only routes":
        for j in B:
            y_l_visited_by_k = gp.min_(gp.quicksum(x[i, j_temp, k] for i in range(len(dist_m)) 
                                                  for j_temp in L if i != j_temp ), 1)
            outflow_to_j_directly = gp.quicksum(x[i, j, k] for i in range(len(dist_m)) if i != j)
            model.addConstr(outflow_to_j_directly <= y_l_visited_by_k * n_tot_cus, name=f"no_back_only_{k}_{j}")
    

    # Set time limit
    model.Params.TimeLimit = instance["time_limit"]

    # Optimize
    model.optimize()

    # Extract solution
    solution = {
        "objective_value": model.objVal if model.solCount > 0 else float('inf'),
        "routes": []
    }

    if model.solCount > 0:
        routes = []
        for k in range(n_veh):
            route = [depot_idx]
            current_node = depot_idx
            
            while True:
                next_node = None
                for j in range(len(dist_m[0])):
                    if j != current_node and x[current_node, j, k].x > 0.9:
                        next_node = j
                        break
                
                if next_node is None or next_node == depot_idx:
                    break
                    
                route.append(next_node)
                current_node = next_node
            
            if len(route) > 1:  # Non-trivial route
                route.append(depot_idx)
                routes.append(route)
        
        solution["routes"] = routes

    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to output the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", required=False, help="Path to log intermediate solutions")
    args = parser.parse_args()

    # Load the instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    # Add time limit to instance for consistency with solver needs
    instance["time_limit"] = args.time_limit

    # Initialize logger if provided
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    try:
        # Solve the VRPB instance
        solution = solve_vrpb(instance)

        # Update logger with final objective value if found
        if logger and solution["objective_value"] != float('inf'):
            logger.log(solution["objective_value"])

    except Exception as e:
        print(f"Error solving instance: {e}")
        solution = {
            "objective_value": float('inf'),
            "routes": []
        }

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()