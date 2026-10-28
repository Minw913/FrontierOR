import argparse
import json
import math
from itertools import permutations
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import itertools
import random
import threading
import time
import copy
from solution_logger import SolutionLogger

def calculate_euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class RobustVRPSolver:
    def __init__(self, instance_data, time_limit=60, log_path=None):
        self.instance = instance_data
        self.time_limit = time_limit
        self.num_customers = instance_data["num_customers"]
        self.num_vehicles = instance_data["num_vehicles"]
        self.vehicle_capacity = instance_data["vehicle_capacity"]
        self.Gamma = instance_data["uncertainty_set"]["Gamma"]
        
        # Extract customer information
        self.customers = {c["id"]: c for c in instance_data["customers"]}
        self.customer_ids = sorted([c["id"] for c in instance_data["customers"]])
        # Map customer IDs to 0-based indices for internal use
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.customer_ids)}
        self.idx_to_id = {idx: cid for idx, cid in enumerate(self.customer_ids)}
        
        # Depot info - assuming always uses index 0 in the distance matrix
        self.depot_id = 0  # Per schema, depot is always index 0
        self.distance_matrix = instance_data["distance_matrix"]
        
        self.nominal_demands = [self.customers[self.idx_to_id[i]]['nominal_demand'] for i in range(self.num_customers)]
        self.demand_deviations = [self.customers[self.idx_to_id[i]]['demand_deviation'] for i in range(self.num_customers)]
        
        # Initialize logger
        self.logger = SolutionLogger(log_path, sense="minimize") if log_path else None
        self.best_solution = None
        self.best_objective = float('inf')
        self.x = None  # Store reference to variables for extraction
        self.y = None

    def solve(self):
        # Create a model
        m = gp.Model("RobustVRP")
        m.setParam("Threads", 1)
        
        # Set time limit
        m.Params.TimeLimit = self.time_limit
        
        # Suppress Gurobi output to avoid cluttering our console
        m.Params.OutputFlag = 0
        
        # Variables: x[i,j,k] = 1 if vehicle k travels from i to j
        # 0 to n (depot to last customer) 
        x = {}
        for k in range(self.num_vehicles):
            for i in range(self.num_customers + 1):  # depot included as index 0
                for j in range(self.num_customers + 1):
                    if i != j:
                        x[i, j, k] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
        
        # y[i, k] = 1 if customer i (internal index 1 to n) is assigned to vehicle k
        y = {}
        for i in range(1, self.num_customers + 1):  # Customers' internal indices are 1 to n
            for k in range(self.num_vehicles):
                y[i, k] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")
        
        # z[i, k] indicates if customer i contributes to worst-case deviation on vehicle k
        z = {}
        for i in range(1, self.num_customers + 1):
            for k in range(self.num_vehicles):
                z[i, k] = m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}")
        
        # Objective: minimize total travel cost
        m.setObjective(
            gp.quicksum(
                self.distance_matrix[i][j] * x[i, j, k]
                for k in range(self.num_vehicles)
                for i in range(self.num_customers + 1)
                for j in range(self.num_customers + 1)
                if i != j
            ),
            GRB.MINIMIZE
        )
        
        # Each customer appears exactly once on exactly one route
        for i in range(1, self.num_customers + 1):  # Customer indices from 1 to n
            m.addConstr(gp.quicksum(y[i, k] for k in range(self.num_vehicles)) == 1)
        
        # If customer i is assigned to vehicle k, it must have incoming/outgoing arcs on that vehicle
        for k in range(self.num_vehicles):
            for i in range(1, self.num_customers + 1):
                # Customer i is assigned to k -> vehicle k serves i
                m.addConstr(gp.quicksum(x[i, j, k] for j in range(self.num_customers + 1) if j != i) == y[i, k])
                m.addConstr(gp.quicksum(x[j, i, k] for j in range(self.num_customers + 1) if j != i) == y[i, k])
                
        # Each vehicle starts and ends at depot (index 0)
        for k in range(self.num_vehicles):
            m.addConstr(gp.quicksum(x[0, i, k] for i in range(1, self.num_customers + 1)) == 1)  # Departure
            m.addConstr(gp.quicksum(x[i, 0, k] for i in range(1, self.num_customers + 1)) == 1)  # Return
            
        # Flow conservation at customers for each vehicle
        for k in range(self.num_vehicles):
            for i in range(1, self.num_customers + 1):
                in_flow = gp.quicksum(x[j, i, k] for j in range(self.num_customers + 1) if j != i)
                out_flow = gp.quicksum(x[i, j, k] for j in range(self.num_customers + 1) if j != i)
                m.addConstr(in_flow == out_flow)
        
        # Robust capacity constraints: worst-case demand <= capacity for each vehicle
        for k in range(self.num_vehicles):
            worst_case_demand_k = gp.quicksum(self.nominal_demands[i-1] * y[i, k] for i in range(1, self.num_customers + 1)) \
                                 + gp.quicksum(self.demand_deviations[i-1] * z[i, k] for i in range(1, self.num_customers + 1))
            m.addConstr(worst_case_demand_k <= self.vehicle_capacity)
        
        # z[i,k] constraints: link z to y and limit deviations by Gamma per route
        for k in range(self.num_vehicles):
            for i in range(1, self.num_customers + 1):
                # z can only be 1 if y is also 1
                m.addConstr(z[i, k] <= y[i, k])
                
            # Total number of deviations per route bounded by Gamma
            # Sum of z[i, k] <= Gamma for integer Gamma, otherwise adjust appropriately for continuous Gamma
            max_devs = min(int(self.Gamma) if self.Gamma == int(self.Gamma) else int(self.Gamma) + 1, self.num_customers)
            m.addConstr(gp.quicksum(z[i, k] for i in range(1, self.num_customers + 1)) <= max_devs)

        # Save references to variables for solution extraction
        self.x = x
        self.y = y

        # Optimize with callback
        if self.logger:
            def mycallback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_val < self.best_objective:
                        self.best_objective = obj_val
                        
                        # Get current variable values directly in callback
                        current_solution = self.get_routes_for_callback(model)
                        if current_solution:
                            self.best_solution = current_solution
                            self.logger.log(obj_val)

            m.optimize(mycallback)
        else:
            m.optimize()

        if m.SolCount > 0:
            # Extract final solution even if not improved
            x_vals = m.getAttr('X', x)
            final_solution = self.get_routes_from_solution_direct(x_vals, {key: var.X for key, var in y.items()})
            if final_solution:
                if not self.best_solution or m.ObjVal < self.best_objective:
                    self.best_objective = m.ObjVal
                    self.best_solution = final_solution
        elif self.best_solution is None:
            # Handle case where no solution found
            self.best_solution = None
            self.best_objective = float('inf')
                
        return self.best_solution
    
    def get_routes_for_callback(self, model):
        """Extract routes using model.cbGetSolution."""
        try:
            # Get variables directly
            result_routes = []
            
            # Determine which customers are assigned to each vehicle
            assignment = {}
            for i in range(1, self.num_customers + 1):
                for k in range(self.num_vehicles):
                    var_val = model.cbGetSolution(self.y[i, k])
                    if abs(var_val - 1.0) < 1e-6:  # If customer i is assigned to vehicle k
                        assignment[i] = k
                        break
            
            # Group customers per vehicle
            veh_to_customers = defaultdict(list)
            for i in range(1, self.num_customers + 1):
                if i in assignment:
                    veh_to_customers[assignment[i]].append(i)
            
            # Build routes for each vehicle
            for k in range(self.num_vehicles):
                if k not in veh_to_customers or not veh_to_customers[k]:
                    continue  # Skip empty routes

                current_route = []
                    
                # Start from depot (node 0)
                current_node = 0
                remaining_customers = set(veh_to_customers[k])
                
                # Build sequence using model.cbGetSolution
                while remaining_customers:
                    next_node = None
                    for j in remaining_customers:
                        if j != current_node:
                            var_val = model.cbGetSolution(self.x[current_node, j, k])
                            if abs(var_val - 1.0) < 1e-5:
                                next_node = j
                                break
    
                    if next_node is None:
                        # If couldn't find a next node from remaining customers, might be going back to depot
                        if current_node in veh_to_customers[k]:  # We are leaving a customer node
                            if abs(model.cbGetSolution(self.x[current_node, 0, k]) - 1.0) < 1e-5:
                                break
                            else:
                                # Something is wrong - break
                                break
                        else:
                            break
                    
                    current_route.append(next_node)
                    remaining_customers.remove(next_node)
                    current_node = next_node
                        
                # Calculate route metrics
                if current_route:  # Only process if there are customers in the route
                    route_ids = [self.idx_to_id[cust_idx - 1] for cust_idx in current_route]  # Convert back to original IDs
                    
                    # Cost calculation
                    cost = 0
                    if current_route:
                        # Depot to first
                        cost += self.distance_matrix[0][current_route[0]]
                        # Between customers
                        for idx in range(len(current_route) - 1):
                            cost += self.distance_matrix[current_route[idx]][current_route[idx + 1]]
                        # Last to depot
                        cost += self.distance_matrix[current_route[-1]][0]

                    # Worst case demand calculation (accounting for Gamma uncertainty)
                    nominal_sum = sum(self.nominal_demands[c_idx - 1] for c_idx in current_route)
                    deviations_list = [self.demand_deviations[c_idx - 1] for c_idx in current_route]
                    ordered_deviations_desc = sorted(deviations_list, reverse=True)
                    gamma_cap = min(int(self.Gamma) if self.Gamma == int(self.Gamma) else int(self.Gamma) + 1, len(ordered_deviations_desc))
                    gamma_deviation = sum(ordered_deviations_desc[:gamma_cap])
                    
                    worst_case_demand = nominal_sum + gamma_deviation

                    result_routes.append({
                        "customers": route_ids,
                        "cost": int(round(cost)),
                        "worst_case_demand": int(round(worst_case_demand))
                    })

            # Calculate total cost
            total_cost = sum(route['cost'] for route in result_routes)
            
            return {
                "objective_value": float(total_cost),
                "routes": result_routes
            }
        except Exception as e:
            # If callback extraction fails, return None to skip logging that solution
            return None

    def get_routes_from_solution_direct(self, x, y):
        """Extract routes from the solution variables."""
        result_routes = []
        
        # Determine which customers are assigned to each vehicle
        assignment = {}
        for i in range(1, self.num_customers + 1):
            for k in range(self.num_vehicles):
                if abs(y[i, k] - 1.0) < 1e-6:  # If customer i is assigned to vehicle k
                    assignment[i] = k
                    break
        
        # Group customers per vehicle
        veh_to_customers = defaultdict(list)
        for i in range(1, self.num_customers + 1):
            if i in assignment:
                veh_to_customers[assignment[i]].append(i)
        
        # Build routes for each vehicle by traversing from the depot
        for k in range(self.num_vehicles):
            if k not in veh_to_customers or not veh_to_customers[k]:
                continue  # Skip empty routes

            current_route = []
            
            # Start from depot (node 0)
            current_node = 0
            remaining_customers = set(veh_to_customers[k])
            
            # Build the sequence using x variables
            while remaining_customers:
                next_node = None
                # Look for next customer in the route for vehicle k
                for j in remaining_customers:
                    if j != current_node and abs(x[current_node, j, k] - 1.0) < 1e-5:
                        next_node = j
                        break

                if next_node is None:
                    # If couldn't find a next node from remaining customers, might be going back to depot
                    if abs(x[current_node, 0, k] - 1.0) < 1e-5:
                        # End of route without more customer stops
                        break
                    else:
                        # Something is wrong
                        break
                
                current_route.append(next_node)
                remaining_customers.remove(next_node)
                current_node = next_node

            # Calculate route metrics
            if current_route:  # Only process if there are customers in the route
                route_ids = [self.idx_to_id[cust_idx - 1] for cust_idx in current_route]  # Convert back to original IDs
                
                # Cost calculation
                cost = 0
                if current_route:
                    # Depot to first
                    cost += self.distance_matrix[0][current_route[0]]
                    # Between customers
                    for idx in range(len(current_route) - 1):
                        cost += self.distance_matrix[current_route[idx]][current_route[idx + 1]]
                    # Last to depot
                    cost += self.distance_matrix[current_route[-1]][0]

                # Worst case demand calculation (accounting for Gamma uncertainty)
                nominal_sum = sum(self.nominal_demands[c_idx - 1] for c_idx in current_route)
                deviations_list = [self.demand_deviations[c_idx - 1] for c_idx in current_route]
                ordered_deviations_desc = sorted(deviations_list, reverse=True)
                gamma_cap = min(int(self.Gamma) if self.Gamma == int(self.Gamma) else int(self.Gamma) + 1, len(ordered_deviations_desc))
                gamma_deviation = sum(ordered_deviations_desc[:gamma_cap])
                
                worst_case_demand = nominal_sum + gamma_deviation

                result_routes.append({
                    "customers": route_ids,
                    "cost": int(round(cost)),
                    "worst_case_demand": int(round(worst_case_demand))
                })

        # Calculate total cost
        total_cost = sum(route['cost'] for route in result_routes)
        
        return {
            "objective_value": float(total_cost),
            "routes": result_routes
        }


def main():
    parser = argparse.ArgumentParser(description='Solve the Robust VRP problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the input JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the output JSON solution file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, required=False, help='Path for logging intermediate solutions')
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Create solver
    solver = RobustVRPSolver(instance_data, time_limit=args.time_limit, log_path=args.log_path)

    # Solve
    solution = solver.solve()

    # Write solution
    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("No solution found.")


if __name__ == '__main__':
    main()