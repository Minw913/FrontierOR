import argparse
import json
import math
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import itertools


def solve_inventory_routing(instance_path, solution_path, time_limit, logger=None):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    # Extract data
    num_customers = instance["num_customers"]
    num_periods = instance["num_periods"]
    num_vehicles = instance["num_vehicles"]
    vehicle_capacity = instance["vehicle_capacity"]
    num_nodes = instance["num_nodes"]
    coordinates = instance["coordinates"]
    distance_matrix = instance["distance_matrix"]
    holding_costs = instance["holding_costs"]
    customer_demands = instance["customer_demands"]  # [customer][period]
    supplier_production = instance["supplier_production"]
    inventory_capacities = instance["inventory_capacities"]
    initial_inventories = instance["initial_inventories"]
    
    depot_idx = 0
    customer_indices = list(range(1, num_customers + 1))
    all_nodes = list(range(num_nodes))
    
    # Create model
    model = gp.Model("inventory_routing")
    model.setParam("Threads", 1)
    model.params.TimeLimit = time_limit
    model.params.MIPGap = 1e-4  # Stop when gap reaches 5%
    
    # Decision Variables
    # x[i,j,k,t]: 1 if vehicle k travels arc (i,j) in period t
    x = {}
    for t in range(num_periods):
        for k in range(num_vehicles):
            for i in all_nodes:
                for j in all_nodes:
                    if i != j:
                        x[i, j, k, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}_{t}")
    
    # d[c,k,t]: quantity delivered to customer c by vehicle k in period t
    d = {}
    for t in range(num_periods):
        for k in range(num_vehicles):
            for c in customer_indices:
                d[c, k, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"d_{c}_{k}_{t}")
    
    # I[loc, t]: inventory at location loc at end of period t
    I = {}
    for t in range(num_periods):
        for loc in all_nodes:
            I[loc, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"I_{loc}_{t}")

    # Binary variable: y[c, k, t] = 1 if customer c is visited by vehicle k in period t
    y = {}
    for t in range(num_periods):
        for k in range(num_vehicles):
            for c in customer_indices:
                y[c, k, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}_{k}_{t}")
    
    model.update()
    
    # Objective function: minimize total travel cost + holding cost
    obj_expr = gp.quicksum(
        distance_matrix[i][j] * x[i, j, k, t]
        for t in range(num_periods)
        for k in range(num_vehicles)
        for i in all_nodes
        for j in all_nodes
        if i != j
    )
    
    obj_expr += gp.quicksum(
        holding_costs[loc] * I[loc, t]
        for t in range(num_periods)
        for loc in all_nodes
    )
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Constraints
    
    # Each customer is visited at most once per period
    for t in range(num_periods):
        for c in customer_indices:
            model.addConstr(gp.quicksum(y[c, k, t] for k in range(num_vehicles)) <= 1)
    
    # For each vehicle k in each period t, form consistent flows
    for t in range(num_periods):
        for k in range(num_vehicles):
            # Flow conservation at depot
            model.addConstr(
                gp.quicksum(x[depot_idx, j, k, t] for j in all_nodes if j != depot_idx) == 
                gp.quicksum(x[i, depot_idx, k, t] for i in all_nodes if i != depot_idx)
            )
            
            # Each customer can have at most 1 visit by this vehicle
            for c in customer_indices:
                model.addConstr(
                    gp.quicksum(x[i, c, k, t] for i in all_nodes if i != c) == y[c, k, t]
                )  # Enter to customer c by vehicle k
                
                model.addConstr(
                    gp.quicksum(x[c, j, k, t] for j in all_nodes if j != c) == y[c, k, t]
                )  # Leave from customer c by vehicle k
            
            # Subtour elimination using degree constraints and ensuring paths flow properly
            # If visited, must arrive and depart through the same vehicle
            for c in customer_indices:
                model.addConstr(
                    gp.quicksum(x[c, j, k, t] for j in all_nodes if j != c) == 
                    gp.quicksum(x[j, c, k, t] for j in all_nodes if j != c)
                )
    
    # Vehicle capacity constraint
    for t in range(num_periods):
        for k in range(num_vehicles):
            model.addConstr(gp.quicksum(d[c, k, t] for c in customer_indices) <= vehicle_capacity)

    # Ensure delivery only to visited customers and respect storage capacity
    for t in range(num_periods):
        for k in range(num_vehicles):
            for c in customer_indices:
                # d[c,k,t] > 0 only if y[c,k,t] = 1
                max_delivery = min(inventory_capacities[c], vehicle_capacity)
                model.addConstr(d[c, k, t] <= max_delivery * y[c, k, t])

    # Inventory balance at depot
    for t in range(num_periods):
        delivery_sum = gp.quicksum(
            d[c, k, t] for c in customer_indices for k in range(num_vehicles)
        )
        
        if t == 0:
            model.addConstr(
                initial_inventories[depot_idx] + supplier_production[t] - delivery_sum == I[depot_idx, t]
            )
        else:
            model.addConstr(
                I[depot_idx, t-1] + supplier_production[t] - delivery_sum == I[depot_idx, t]
            )
    
    # Inventory upper bound for depot
    for t in range(num_periods):
        model.addConstr(I[depot_idx, t] <= inventory_capacities[depot_idx])
    
    # Inventory balance at customers
    for t in range(num_periods):
        for c in customer_indices:
            delivery_sum = gp.quicksum(d[c, k, t] for k in range(num_vehicles))
            
            if t == 0:
                model.addConstr(
                    initial_inventories[c] + delivery_sum - customer_demands[c-1][t] == I[c, t]
                )
            else:
                model.addConstr(
                    I[c, t-1] + delivery_sum - customer_demands[c-1][t] == I[c, t]
                )
    
    # Inventory bounds for customers
    for t in range(num_periods):
        for c in customer_indices:
            model.addConstr(I[c, t] <= inventory_capacities[c])
    
    # Solve model
    model.optimize()

    def get_routes_from_solution():
        routes = defaultdict(dict)
        
        for t in range(num_periods):
            for k in range(num_vehicles):
                # Extract route for vehicle k in period t
                visited_customers = []
                for c in customer_indices:
                    if y[c, k, t].x > 0.5:  # Customer c is visited by vehicle k in period t
                        visited_customers.append(c)
                
                if len(visited_customers) == 0:
                    continue  # No customers visited by this vehicle
                
                # Build route from arcs in x variables
                current_route = [depot_idx]  # Start and end at depot
                
                # Find first customer from depot
                next_node = depot_idx
                visited_arc_set = set()
                
                # Build full route by following arcs
                for step in range(len(visited_customers) + 1):
                    found = False
                    for node in all_nodes:
                        if node != next_node and (next_node, node, k, t) in x and x[next_node, node, k, t].x > 0.5:
                            visited_arc_set.add((next_node, node))
                            if node != depot_idx:
                                current_route.append(node)
                            else:
                                current_route.append(node)  # End back at depot
                                found = True
                                break
                            next_node = node
                            found = True
                            break
                    if not found:
                        break  # Stop if no forward arc found
                
                # Also get deliveries for this vehicle
                deliveries = {}
                for c in visited_customers:
                    deliveries[str(c)] = round(d[c, k, t].x)
                    
                routes[str(t)][str(k)] = {
                    "route": current_route,
                    "deliveries": deliveries
                }
        
        return routes

    # Check model status and output solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        solution = {}

        # Get routes
        routes = get_routes_from_solution()
        solution["routes"] = routes
        
        # Get inventories
        inventories = {}
        for loc in all_nodes:
            inv_dict = {}
            for t in range(num_periods):
                inv_dict[str(t)] = round(I[loc, t].x)
            inventories[str(loc)] = inv_dict
        solution["inventories"] = inventories

        # Calculate objective value
        travel_cost = sum(
            distance_matrix[i][j] * x[i, j, k, t].x
            for t in range(num_periods)
            for k in range(num_vehicles)
            for i in all_nodes
            for j in all_nodes
            if i != j and (i, j, k, t) in x
        )

        holding_cost = sum(
            holding_costs[loc] * I[loc, t].x
            for t in range(num_periods)
            for loc in all_nodes
            if (loc, t) in I
        )

        solution["objective_value"] = travel_cost + holding_cost
        
        # If logging enabled, log the final value when improving
        if logger is not None and solution["objective_value"] < float('inf'):
            logger.log(solution["objective_value"])
        
        # Write solution to file
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Even if infeasible, create a blank solution to avoid crash
        solution = {
            "objective_value": float("inf"),
            "routes": {},
            "inventories": {}
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Solve Inventory Routing Problem')
    parser.add_argument('--instance_path', required=True, type=str, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', required=True, type=str, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', required=True, type=int, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log file (JSONL format)')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    solve_inventory_routing(args.instance_path, args.solution_path, args.time_limit, logger)


if __name__ == '__main__':
    main()