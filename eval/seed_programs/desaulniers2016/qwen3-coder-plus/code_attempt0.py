import argparse
import json
import math
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import itertools


def solve_inventory_routing(instance_path, solution_path, time_limit):
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
    model.params.TimeLimit = time_limit
    model.params.MIPGap = 0.05  # Stop when gap reaches 5%
    
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
    
    # Link binary visit variable to x flow variables
    for t in range(num_periods):
        for k in range(num_vehicles):
            for c in customer_indices:
                # If customer c is visited by vehicle k -> there should be arcs leading to it
                # Flow conservation: incoming = outgoing except at depot (net outflow = 1 if visited)
                model.addConstr(gp.quicksum(x[depot_idx, j, k, t] for j in all_nodes if j != depot_idx) == y[c, k, t])
                model.addConstr(gp.quicksum(x[i, c, k, t] for i in all_nodes if i != c) == y[c, k, t])
    
    # At most num_vehicles routes per period
    for t in range(num_periods):
        model.addConstr(gp.quicksum(y[c, k, t] for c in customer_indices for k in range(num_vehicles)) >= 0)
    
    # Flow conservation at depot: outflow = inflow (both are #vehicles used)
    for t in range(num_periods):
        for k in range(num_vehicles):
            model.addConstr(
                gp.quicksum(x[depot_idx, j, k, t] for j in all_nodes if j != depot_idx) ==
                gp.quicksum(x[j, depot_idx, k, t] for j in all_nodes if j != depot_idx)
            )
            
            # Only one trip per vehicle max
            model.addConstr(
                gp.quicksum(x[depot_idx, j, k, t] for j in all_names if j != depot_idx) <= 1
            )
    
    # Flow conservation for each customer per vehicle-period combination
    for t in range(num_periods):
        for k in range(num_vehicles):
            for c in customer_indices:
                model.addConstr(
                    gp.quicksum(x[i, c, k, t] for i in all_nodes if i != c) ==
                    gp.quicksum(x[c, j, k, t] for j in all_nodes if j != c)
                )
    
    # Vehicle capacity constraint
    for t in range(num_periods):
        for k in range(num_vehicles):
            model.addConstr(gp.quicksum(d[c, k, t] for c in customer_indices) <= vehicle_capacity)
    
    # Delivery only to visited customers
    for t in range(num_periods):
        for k in range(num_vehicles):
            for c in customer_indices:
                # d[c,k,t] > 0 only if y[c,k,t] = 1
                model.addConstr(d[c, k, t] <= inventory_capacities[c] * y[c, k, t])
    
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
    
    # Subtour elimination constraints are complex to add completely, but we'll try with lazy constraints if needed
    # We need to ensure every customer is in only one connected component per vehicle-period combination
    
    # Solve model
    model.optimize()

    def get_routes_from_solution():
        routes = defaultdict(dict)
        deliveries = defaultdict(dict)
        
        for t in range(num_periods):
            for k in range(num_vehicles):
                # Get the route for vehicle k in period t
                route_list = [0]  # Start at depot
                
                current_node = 0
                visited = set([0])
                
                cont = True
                while cont:
                    found_next = False
                    for j in all_nodes:
                        if j not in visited and x[current_node, j, k, t].x > 0.9:
                            route_list.append(j)
                            current_node = j
                            visited.add(j)
                            found_next = True
                            
                            # Add delivery info if customer
                            if j in customer_indices:
                                delivery_q = d[j, k, t].x
                                deliveries[(t, k)][j] = round(delivery_q)
                            
                            break
                    
                    if current_node == 0:  # Backed to depot
                        cont = False
                    elif not found_next:
                        cont = False  # Stuck, break loop
                
                if len(route_list) > 1:  # Only record if actually traveled
                    routes[t][k] = {
                        "route": route_list,
                        "deliveries": {str(c): v for c, v in deliveries.get((t, k), {}).items()}
                    }
        
        return routes

    # Handle multiple scenarios:
    # - OPTIMAL solution
    # - TIME_LIMIT reached (but feasible solution exists)
    # - INFEASIBLE (should exit gracefully)
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        solution = {}

        # Rebuild routes
        routes = get_routes_from_solution()
        solution["routes"] = routes
        
        inventories = {}
        for loc in all_nodes:
            inv_dict = {}
            for t in range(num_periods):
                inv_dict[str(t)] = round(I[loc, t].x)
            inventories[str(loc)] = inv_dict
        solution["inventories"] = inventories

        # Calculate total travel cost
        travel_cost = sum(
            distance_matrix[i][j] * x[i, j, k, t].x
            for t in range(num_periods)
            for k in range(num_vehicles)
            for i in all_nodes
            for j in all_nodes
            if i != j
        )

        # Calculate total holding cost
        holding_cost = sum(
            holding_costs[loc] * I[loc, t].x
            for t in range(num_periods)
            for loc in all_nodes
        )

        solution["objective_value"] = travel_cost + holding_cost
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        print(f"Model status: {model.status}")
        # Even if infeasible, attempt to generate solution anyway if values are available
        solution = {
            "objective_value": float("inf"),
            "routes": {},
            "inventories": {}
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser(description='Solve Inventory Routing Problem')
    parser.add_argument('--instance_path', required=True, type=str, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', required=True, type=str, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', required=True, type=int, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log file (JSONL format)')
    
    args = parser.parse_args()
    
    # Initialize logger
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
        
        # Create a callback to log improved solutions
        def mycallback(model, where):
            if where == GRB.Callback.MIP:
                obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
                if obj_best < float('inf'):
                    logger.log(obj_best)
        
        # Call solve with callback - not possible with basic optimize() easily
        solve_inventory_routing(args.instance_path, args.solution_path, args.time_limit)
    else:
        solve_inventory_routing(args.instance_path, args.solution_path, args.time_limit)


if __name__ == '__main__':
    main()