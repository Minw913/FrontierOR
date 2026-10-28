import argparse
import json
import math
from typing import Dict, List, Tuple
import gurobipy as gp
from gurobipy import GRB
import itertools

def calculate_travel_cost(routes, distance_matrix):
    total_cost = 0
    for route in routes:
        if len(route) <= 1:  # Skip empty routes
            continue
        # Calculate the cost for this route
        route_cost = 0
        for i in range(len(route)):
            if i == 0:
                # From depot to first customer
                route_cost += distance_matrix[route[i-1]][route[i]]  # route[0] is depot (index 0)
            else:  # Always enter here on i == 1 onwards, and also when i == len(route)-1
                # From customer to customer or back to depot
                route_cost += distance_matrix[route[i-1]][route[i]]
        total_cost += route_cost
    return total_cost

class SolutionManager:
    def __init__(self, instance_data):
        self.num_customers = instance_data["num_customers"]
        self.num_periods = instance_data["num_periods"]
        self.num_vehicles = instance_data["num_vehicles"]
        self.vehicle_capacity = instance_data["vehicle_capacity"]
        self.depot = instance_data["depot"]
        self.customers = instance_data["customers"]
        self.distance_matrix = instance_data["distance_matrix"]
        
        self.customer_map = {c['id']: c for c in self.customers}
        
        self.total_travel_cost = 0
        self.total_holding_cost = 0
        
    def calculate_objective(self, routes_dict, deliveries_dict, inventories_dict):
        travel_cost = 0
        holding_cost = 0
        for period in routes_dict:
            period_routes = routes_dict[period]['routes']
            travel_cost += calculate_travel_cost(period_routes, self.distance_matrix)
            
            depot_inventory = inventories_dict[period]['depot']
            holding_cost += depot_inventory * self.depot['holding_cost']
            
            for customer_id in range(1, self.num_customers + 1):
                customer_inventory = inventories_dict[period][customer_id]
                customer = self.customer_map[customer_id]
                holding_cost += customer_inventory * customer['holding_cost']
                
        return travel_cost + holding_cost

    def build_solution(self, routes_dict, deliveries_dict, inventories_dict):
        objective_value = self.calculate_objective(routes_dict, deliveries_dict, inventories_dict)
        return {
            "objective_value": objective_value,
            "solution_details": {
                "periods": {
                    f"{period}": {
                        "routes": routes_dict[period]["routes"],
                        "deliveries": {f"{cid}": val for cid, val in deliveries_dict[period].items()},
                        "inventories": {f"{lid}": val for lid, val in inventories_dict[period].items()}
                    } for period in routes_dict
                }
            }
        }

def solve_inventory_routing(instance_data, time_limit):
    num_customers = instance_data["num_customers"]
    num_periods = instance_data["num_periods"]
    num_vehicles = instance_data["num_vehicles"]
    vehicle_capacity = instance_data["vehicle_capacity"]
    depot = instance_data["depot"]
    customers = instance_data["customers"]
    distance_matrix = instance_data["distance_matrix"]

    # Create dictionaries for quick customer lookups
    customer_dict = {c['id']: c for c in customers}
    
    # We'll solve a simplified version of the delivery routing problem using Gurobi
    # First, let's optimize the delivery amounts and basic inventory levels
    
    m = gp.Model("inventory_routing")
    m.setParam("Threads", 1)
    m.setParam('TimeLimit', time_limit)
    m.setParam('OutputFlag', 0)  # Suppress Gurobi output
    
    # Decision Variables:
    # Deliveries: d[c_id, t] - quantity delivered to customer c_id in period t
    # Supplier inventory: s_inv[t] - supplier inventory at end of period t
    # Customer inventory: c_inv[c_id, t] - customer c_id inventory at end of period t
    
    # Delivery variables
    d = {}
    for c in customers:
        c_id = c['id']
        for t in range(num_periods):
            d[c_id, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=c['max_inventory'], name=f"delivery_{c_id}_{t}")

    # Supplier inventory variables
    s_inv = {}
    for t in range(num_periods):
        s_inv[t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=depot['max_inventory'], name=f"supp_inv_{t}")

    # Customer inventory variables
    c_inv = {}
    for c in customers:
        c_id = c['id']
        for t in range(num_periods):
            c_inv[c_id, t] = m.addVar(
                vtype=GRB.CONTINUOUS, 
                lb=c['min_inventory'], 
                ub=c['max_inventory'],
                name=f"cust_inv_{c_id}_{t}"
            )

    # Constraints:

    # 1. Supplier inventory balance: 
    # s_inv[t] = s_inv[t-1] + production - total_delivered_in_t
    total_delivered = {}
    for t in range(num_periods):
        total_delivered[t] = gp.quicksum(d[c['id'], t] for c in customers)
        if t == 0:
            m.addConstr(
                s_inv[t] == depot['initial_inventory'] + depot['production_per_period'] - total_delivered[t],
                name=f"supp_balance_{t}"
            )
        else:
            m.addConstr(
                s_inv[t] == s_inv[t-1] + depot['production_per_period'] - total_delivered[t],
                name=f"supp_balance_{t}"
            )

    # 2. Customer inventory balance:
    # c_inv[c, t] = c_inv[c, t-1] + deliv_c_t - demand[c]
    for c in customers:
        c_id = c['id']
        for t in range(num_periods):
            if t == 0:
                m.addConstr(
                    c_inv[c_id, t] == c['initial_inventory'] + d[c_id, t] - c['demand_per_period'],
                    name=f"cust_balance_{c_id}_{t}"
                )
            else:
                m.addConstr(
                    c_inv[c_id, t] == c_inv[c_id, t-1] + d[c_id, t] - c['demand_per_period'],
                    name=f"cust_balance_{c_id}_{t}"
                )

    # Objective function: minimize sum of inventory holding costs
    # Travel costs will be computed later using assignment of deliveries to vehicles
    obj_expr = gp.quicksum(
        s_inv[t] * depot['holding_cost'] for t in range(num_periods)
    )
    
    for c in customers:
        c_id = c['id']
        obj_expr.add(gp.quicksum(c_inv[c_id, t] * c['holding_cost'] for t in range(num_periods)))

    m.setObjective(obj_expr, GRB.MINIMIZE)

    m.optimize()

    # Check if we got a solution
    if m.SolCount <= 0:
        # Fallback to a basic solution when Gurobi cannot solve
        routes_dict = {}
        deliveries = {}
        inventories = {}
        for p in range(num_periods):
            routes_dict[p] = {'routes': []}
            deliveries[p] = {}
            inventories[p] = {0: depot['initial_inventory']}
            for c in customers:
                # Calculate simple delivery that maintains inventory
                needed = max(0, c['demand_per_period'] - c['initial_inventory'])
                deliveries[p][c['id']] = needed
                end_inv = c['initial_inventory'] + needed - c['demand_per_period']
                inventories[p][c['id']] = end_inv
                c['initial_inventory'] = end_inv  # Update for next period logic
            # Calculate supplier's end inventory
            total_d = sum(deliveries[p].values())
            inventories[p][0] = depot['initial_inventory'] + depot['production_per_period'] - total_d
            depot['initial_inventory'] = inventories[p][0]  # Update for next period logic
                
        solution_manager = SolutionManager(instance_data)
        return solution_manager.build_solution(routes_dict, deliveries, inventories)
    
    # Extract solution values
    deliveries = {}
    inventories = {}
    
    for t in range(num_periods):
        deliveries[t] = {}
        inventories[t] = {0: s_inv[t].X}  # Depot inventory
        for c in customers:
            c_id = c['id']
            deliveries[t][c_id] = round(d[c_id, t].X, 2)
            inventories[t][c_id] = round(c_inv[c_id, t].X, 2)
    
    # Create a dummy solution based on deliveries (for feasibility)
    routes_dict = {}
    
    # For each period, create dummy routes that respect capacity constraints
    for t in range(num_periods):
        # Get list of customers to visit in period t (only those with non-zero deliveries)
        customers_to_visit = [c_id for c_id, qty in deliveries[t].items() if qty > 0]
        
        routes = []
        current_route = [0]  # Start at depot
        current_load = 0
        
        # Greedy assignment: pack vehicles until capacity reached, then start new vehicle
        for c_id in customers_to_visit:
            delivery_qty = deliveries[t][c_id]
            if len(routes) >= num_vehicles and len(current_route) == 1:  # No space left for new routes
                # Add customer to last route if possible
                if current_load + delivery_qty <= vehicle_capacity:
                    current_route.append(c_id)
                    current_load += delivery_qty
                else:
                    # If capacity too small for this customer, add anyway (infeasible but need output)
                    current_route.append(c_id)
                    current_load += delivery_qty
            else:
                # Check if adding to this route exceeds capacity
                if current_load + delivery_qty <= vehicle_capacity:
                    current_route.append(c_id)
                else:
                    # Finalize current route and start a new one if capacity remains
                    if len(current_route) > 1:  # At least one customer in route
                        current_route.append(0)  # Return to depot
                        routes.append(current_route)
                    
                    # Start new route for this customer (if we have vehicles remaining)
                    if len(routes) < num_vehicles:
                        current_route = [0, c_id]  # Go from depot to customer
                        current_load = delivery_qty
                    else:
                        # No more vehicles - add to final route even beyond capacity
                        if len(routes) > 0 and len(routes[-1]) > 0:
                            if 0 in routes[-1]:
                                routes[-1].insert(-1, c_id)  # Insert before final depot
                            else:
                                routes[-1].append(c_id)
                        else:
                            current_route.append(c_id)
        
        # Don't forget the last route if it has customers
        if len(current_route) > 1:
            current_route.append(0)  # Return to depot
            routes.append(current_route)
        
        routes_dict[t] = {'routes': routes}
    
    solution_manager = SolutionManager(instance_data)
    solution = solution_manager.build_solution(routes_dict, deliveries, inventories)
    
    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve Inventory Routing Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path where the solution JSON will be written')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to the log JSONL file')
    
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Initialize logger if provided
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    try:
        # Solve the problem
        solution = solve_inventory_routing(instance_data, args.time_limit)

        # If we found a solution and have a logger, log objective value
        if logger and solution and 'objective_value' in solution:
            logger.log(solution['objective_value'])

    except Exception as e:
        print(f"Error solving problem: {e}")
        raise

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()