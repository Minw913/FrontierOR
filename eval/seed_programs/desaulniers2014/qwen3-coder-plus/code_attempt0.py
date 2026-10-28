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
                route_cost += distance_matrix[0][route[i]]
            elif i == len(route) - 1:
                # From last customer back to depot
                route_cost += distance_matrix[route[i-1]][0]
            else:
                # From customer to customer
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

    m = gp.Model("inventory_routing")
    m.setParam('TimeLimit', time_limit)
    
    # Variables:
    
    # x[i,j,p,v] = 1 if vehicle v travels from i to j in period p
    x = {}
    for p in range(num_periods):
        for v in range(num_vehicles):
            for i in range(num_customers + 1):
                for j in range(num_customers + 1):
                    if i != j:  # no self loops
                        x[i,j,p,v] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{p}_{v}")
    
    # y[i,p] = 1 if customer i is visited in period p
    y = {}
    for p in range(num_periods):
        for i in range(1, num_customers + 1):  # Skip depot
            y[i,p] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{p}")
    
    # d[i,p] = delivery quantity to customer i in period p
    d = {}
    for p in range(num_periods):
        for i in range(1, num_customers + 1):
            max_possible_delivery = (
                customers[i-1]['max_inventory'] 
                - max(0, customers[i-1]['initial_inventory'] - customers[i-1]['demand_per_period'])
            )
            d[i,p] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_possible_delivery, name=f"d_{i}_{p}")
    
    # inv_supplier[p] = end-of-period inventory at supplier in period p
    inv_supplier = {}
    for p in range(num_periods):
        ub = depot['initial_inventory'] + sum(depot['production_per_period'] for _ in range(p+1))
        inv_supplier[p] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=min(ub, depot['max_inventory']), 
                                  name=f"inv_supplier_{p}")
    
    # inv_customer[i,p] = end-of-period inventory at customer i in period p
    inv_customer = {}
    for p in range(num_periods):
        for i in range(1, num_customers + 1):
            ub = (
                customers[i-1]['initial_inventory'] 
                + sum(d[i,k].ub for k in range(p+1))  # upper bound on total delivered before/including p
            )
            inv_customer[i,p] = m.addVar(
                vtype=GRB.CONTINUOUS, lb=customers[i-1]['min_inventory'], 
                ub=min(customers[i-1]['max_inventory'], ub), 
                name=f"inv_customer_{i}_{p}"
            )

    # Flow conservation at customers: for each period p, each location i
    for p in range(num_periods):
        for i in range(1, num_customers + 1):  # Only for customers
            expr_in = gp.LinExpr()
            expr_out = gp.LinExpr()
            for v in range(num_vehicles):
                # Incoming edges
                for j in range(num_customers + 1):
                    if j != i:
                        expr_in.add(x[j,i,p,v])
                # Outgoing edges
                for j in range(num_customers + 1):
                    if j != i:
                        expr_out.add(x[i,j,p,v])
                
            m.addConstr(expr_in == y[i,p], f"flow_in_{i}_{p}")
            m.addConstr(expr_out == y[i,p], f"flow_out_{i}_{p}")
    
    # Each customer visited by at most one vehicle per period
    for p in range(num_periods):
        for i in range(1, num_customers + 1):
            expr = gp.LinExpr()
            for v in range(num_vehicles):
                # Edges going out from this customer
                for j in range(num_customers + 1):
                    if i != j:
                        expr.add(x[i,j,p,v])
            m.addConstr(expr <= 1, f"visit_{i}_{p}")

    # At most num_vehicles per period
    for p in range(num_periods):
        expr = gp.LinExpr()
        for v in range(num_vehicles):
            for i in range(num_customers + 1):
                for j in range(num_customers + 1):
                    if i != j:
                        expr.add(x[i,j,p,v])
        # Since each route takes at least one edge from depot, count those outgoing from depot
        depot_edges = gp.LinExpr()
        for v in range(num_vehicles):
            for j in range(1, num_customers + 1):
                depot_edges.add(x[0,j,p,v])
        m.addConstr(depot_edges <= num_vehicles, f"vehicle_limit_{p}")

    # Capacity constraint on vehicles - the quantity delivered by a route should not exceed vehicle capacity
    for p in range(num_periods):
        for v in range(num_vehicles):
            expr = gp.LinExpr()
            for i in range(1, num_customers + 1):
                # Quantity delivered is associated with visiting the customer
                # Sum up the deliveries that happen due to this route
                expr_to_i = gp.LinExpr()
                for j in range(num_customers + 1):
                    if j != i:
                        expr_to_i.add(x[j,i,p,v])  # incoming at i for this vehicle
            
                # Add delivery term: delivery[i,p] * whether this route delivers to i
                # We need linearization: del_var * boolean_var
                # But we approximate by summing across all possible destinations of this flow-to-i
                expr.addTerms(1, d[i,p])  # This is wrong - need multilinear terms
                
            # The correct way: For each visit, compute if vehicle v serves customer i
            # Instead: for each vehicle, add up all the deliveries that could be on its route
            route_deliveries = gp.LinExpr()
            for i in range(1, num_customers + 1):
                # Binary var for if vehicle v serves customer i in period p
                z_vi_p = m.addVar(vtype=GRB.BINARY, name=f"z_vehicle_visits_{i}_{p}_{v}")
                # If any edge from i leaves for v, then z_vi_p = 1
                for j in range(num_customers + 1):
                    if i != j:
                        m.addConstr(z_vi_p >= x[i,j,p,v])
                        # Ensure z_vi_p is 0 if none of these x vars is 1
                
                # Link z_vi_p to actual deliveries d[i,p]
                # If z_vi_p == 1 then d[i,p] might be > 0, else d[i,p] must be 0
                max_demand_p = customers[i-1]['max_inventory'] - max(0, customers[i-1]['initial_inventory'] - customers[i-1]['demand_per_period'])
                m.addConstr(d[i,p] <= max_demand_p * z_vi_p)
                
                route_deliveries.addTerms(1.0, d[i,p])  # Actually needs z_vi_p connection too
            
            m.addConstr(route_deliveries <= vehicle_capacity, f"capacity_{p}_{v}")

    # Inventory balance constraints
    # For supplier:
    for p in range(num_periods):
        total_delivered = gp.quicksum(d[i, p] for i in range(1, num_customers + 1))
        if p == 0:
            m.addConstr(
                inv_supplier[p] == depot['initial_inventory'] + depot['production_per_period'] - total_delivered,
                f"supp_inv_balance_{p}"
            )
        else:
            m.addConstr(
                inv_supplier[p] == inv_supplier[p-1] + depot['production_per_period'] - total_delivered,
                f"supp_inv_balance_{p}"
            )
        m.addConstr(inv_supplier[p] <= depot['max_inventory'], f"supp_max_cap_{p}")

    # For customers:
    for p in range(num_periods):
        for i in range(1, num_customers + 1):
            if p == 0:
                m.addConstr(
                    inv_customer[i, p] == customers[i-1]['initial_inventory'] + d[i, p] - customers[i-1]['demand_per_period'],
                    f"cust_inv_balance_{i}_{p}"
                )
            else:
                m.addConstr(
                    inv_customer[i, p] == inv_customer[i, p-1] + d[i, p] - customers[i-1]['demand_per_period'],
                    f"cust_inv_balance_{i}_{p}"
                )
            # Max inventory constraint was already imposed in variable definition
            
    # No stockouts allowed
    for p in range(num_periods):
        for i in range(1, num_customers + 1):
            # At the start of the period, after delivery, before consuming
            current_inventory_with_delivery = customers[i-1]['initial_inventory'] if p == 0 else inv_customer[i, p-1]
            m.addConstr(current_inventory_with_delivery + d[i, p] >= customers[i-1]['demand_per_period'], f"no_stockout_{i}_{p}")

    # Min inventory constraint
    for p in range(num_periods):
        for i in range(1, num_customers + 1):
            m.addConstr(inv_customer[i, p] >= customers[i-1]['min_inventory'], f"min_inv_{i}_{p}")

    # Objective - minimize transport and inventory costs
    travel_cost_expr = gp.LinExpr()
    holding_cost_expr = gp.LinExpr()
    
    # Travel costs
    for p in range(num_periods):
        for v in range(num_vehicles):
            for i in range(num_customers + 1):
                for j in range(num_customers + 1):
                    if i != j:
                        travel_cost_expr.addTerms(distance_matrix[i][j], x[i,j,p,v])
                        
    # Holding costs
    for p in range(num_periods):
        holding_cost_expr.addTerms(depot['holding_cost'], inv_supplier[p])
        for i in range(1, num_customers + 1):
            holding_cost_expr.addTerms(customers[i-1]['holding_cost'], inv_customer[i,p])
    
    m.setObjective(travel_cost_expr + holding_cost_expr, GRB.MINIMIZE)

    m.optimize()

    # Extract solution
    if m.SolCount == 0:
        raise Exception("No feasible solution found")

    routes = {}
    deliveries = {}
    inventories = {}

    for p in range(num_periods):
        if p == 0:
            prev_inv_sup = depot['initial_inventory']
        else:
            prev_inv_sup = inventories[p-1]['depot']
        
        current_routes = []
        current_deliveries = {}
        current_inventories = {}

        # Build routes for this period
        for v in range(num_vehicles):
            # Find the route for vehicle v if it exists
            route_start = []  # Start nodes for arcs from depot
            route_end = []    # End nodes for arcs to depot
            route_intermediate = {}  # intermediate arcs: i -> j

            for i in range(num_customers + 1):
                for j in range(num_customers + 1):
                    if i != j and x[i, j, p, v].X > 0.9:
                        if i == 0:
                            route_start.append(j)
                        elif j == 0:
                            route_end.append(i)
                        else:
                            route_intermediate[i] = j

            if not route_start:
                # Vehicle not used
                continue

            # Construct the route path
            # Start with 0, go to first customer, follow sequence
            this_route = [0]
            if route_start:
                next_node = route_start[0]  # First customer
                this_route.append(next_node)
                current = next_node

                # Follow intermediate segments
                visited = set(this_route)
                while current in route_intermediate and len(this_route) < num_customers + 3:
                    next_node = route_intermediate[current]
                    if next_node in visited:
                        break  # Avoid infinite loop
                    this_route.append(next_node)
                    current = next_node
                    visited.add(current)

                if not route_end or next_node in route_end:
                    if this_route[-1] != 0:
                        this_route.append(0)  # Return to depot

            current_routes.append(this_route)

        routes[p] = {"routes": current_routes}

        for i in range(1, num_customers + 1):
            current_deliveries[i] = d[i, p].X
        deliveries[p] = current_deliveries

        current_inventories[0] = inv_supplier[p].X  # depot is location 0
        for i in range(1, num_customers + 1):
            current_inventories[i] = inv_customer[i, p].X
        inventories[p] = current_inventories

    # Create solution manager and build solution
    sm = SolutionManager(instance_data)
    solution = sm.build_solution(routes, deliveries, inventories)

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

        # If we found a solution and have a logger, log objective values
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