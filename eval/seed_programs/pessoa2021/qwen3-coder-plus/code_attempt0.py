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
        
        # Depot info
        self.depot_id = instance_data["depot"]["id"]
        self.distance_matrix = instance_data["distance_matrix"]
        
        self.nominal_demands = [self.customers[self.idx_to_id[i]]['nominal_demand'] for i in range(self.num_customers)]
        self.demand_deviations = [self.customers[self.idx_to_id[i]]['demand_deviation'] for i in range(self.num_customers)]
        
        # Initialize logger
        self.logger = SolutionLogger(log_path, sense="minimize") if log_path else None
        self.best_solution = None
        self.best_objective = float('inf')

    def solve(self):
        # Create a model
        m = gp.Model("RobustVRP")
        
        # Set time limit
        m.Params.TimeLimit = self.time_limit
        
        # Suppress Gurobi output to avoid cluttering our console
        m.Params.OutputFlag = 0
        
        # Variable: x[i,j,k] = 1 if vehicle k travels from i to j
        # i,j from 0 (depot) to n (after mapping - assume depot is mapped to index 0)
        # To handle general ID mapping properly, we'll map everything to 0-based indices internally
        # For ease of implementation, we map customer idx to 0-based and depot to the last index
        depot_idx = self.num_customers  # Map depot to n-th index
        loc_idx_map = {}  # Maps original IDs to internal indexing
        loc_idx_map[self.depot_id] = depot_idx
        for i in range(self.num_customers):
            loc_idx_map[self.customer_ids[i]] = i
            
        x = {}
        for k in range(self.num_vehicles):
            for i in range(self.num_customers + 1):  # Include depot in index set
                for j in range(self.num_customers + 1):
                    if i != j:
                        x[i, j, k] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
        
        # Additional variables needed
        # y[i, k] = 1 if customer i is assigned to vehicle k
        y = {}
        for i in range(self.num_customers):  # Only for customer nodes
            for k in range(self.num_vehicles):
                y[i, k] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")
        
        # Worst case demand per route constraint
        z = {}  # z[i,k] indicates if customer i contributes to worst-case on vehicle k
        for i in range(self.num_customers):
            for k in range(self.num_vehicles):
                z[i, k] = m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}")
        
        # Objective function
        m.setObjective(
            gp.quicksum(
                self.distance_matrix[loc_idx_map[self.idx_to_id[i]]][loc_idx_map[self.idx_to_id[j]]] * x[i, j, k]
                for k in range(self.num_vehicles)
                for i in range(self.num_customers + 1)
                for j in range(self.num_customers + 1)
                if i != j
            ),
            GRB.MINIMIZE
        )
        
        # Each customer appears exactly once on exactly one route
        for i in range(self.num_customers):
            m.addConstr(gp.quicksum(y[i, k] for k in range(self.num_vehicles)) == 1)
        
        # Connection constraints
        for k in range(self.num_vehicles):
            for i in range(self.num_customers):  # customer i
                m.addConstr(gp.quicksum(x[depot_idx, i, k] for k_iter in [k]) == y[i, k])  # Leaving depot
                m.addConstr(gp.quicksum(x[i, depot_idx, k] for k_iter in [k]) == y[i, k])  # Going to depot
        
        # Flow conservation - each customer entered exactly once
        for k in range(self.num_vehicles):
            for i in range(self.num_customers):
                if i != depot_idx:
                    m.addConstr(
                        gp.quicksum(x[j, i, k] for j in range(self.num_customers + 1) if j != i) ==
                        gp.quicksum(x[i, j, k] for j in range(self.num_customers + 1) if j != i)
                    )
                    
                    # Ensure the flow equals binary assignment
                    m.addConstr(gp.quicksum(x[j, i, k] for j in range(self.num_customers + 1) if j != i) == y[i, k])
        
        # Vehicle start and end at depot
        for k in range(self.num_vehicles):
            m.addConstr(gp.quicksum(x[depot_idx, i, k] for i in range(self.num_customers)) == 1)
            m.addConstr(gp.quicksum(x[i, depot_idx, k] for i in range(self.num_customers)) == 1)
        
        # Capacity constraint with robust version
        bigM = sum(self.nominal_demands) + sum(self.demand_deviations)  # Upper bound of possible demand
        for k in range(self.num_vehicles):
            # Nominal and deviation parts of worst-case demand calculation
            nominal_part = gp.quicksum(self.nominal_demands[i] * y[i, k] for i in range(self.num_customers))
            
            # For robustness: compute worst-case demand considering Gamma deviation budget
            deviation_part = gp.quicksum(self.demand_deviations[i] * z[i, k] for i in range(self.num_customers))
            
            total_worst_case_demand = nominal_part + deviation_part
            m.addConstr(total_worst_case_demand <= self.vehicle_capacity)
            
            # Link z and y appropriately
            for i in range(self.num_customers):
                m.addConstr(z[i, k] <= y[i, k])
                
            # At most Gamma customers in route k can have full deviation
            # We model this using a special knapsack-like constraint:
            m.addConstr(gp.quicksum(z[i, k] for i in range(self.num_customers)) <= self.Gamma)
        
        # Solve model
        def solution_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < self.best_objective:
                    self.best_objective = obj_val
                    # Get current solution and check feasibility
                    sol_x = model.cbGetSolution(x)
                    current_sol = self.extract_solution(sol_x, y, depot_idx, loc_idx_map, model)
                    if current_sol:
                        self.best_solution = current_sol
                        if self.logger:
                            self.logger.log(obj_val)

        try:
            m.optimize(solution_callback)
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")

        if m.SolCount > 0:
            # Extract final solution
            x_vals = m.getAttr('X', x)
            final_solution = self.extract_solution(x_vals, y, depot_idx, loc_idx_map, m)
            if not self.best_solution or m.ObjVal < self.best_objective:
                self.best_objective = m.ObjVal
                self.best_solution = final_solution
                
        return self.best_solution
    
    def get_cost_from_route(self, route):
        """Calculates the distance for a given route."""
        cost = 0
        # From depot to first customer
        cost += self.distance_matrix[0][self.id_to_idx[route[0]]]
        
        # Between customers
        for i in range(len(route) - 1):
            cust1 = self.id_to_idx[route[i]]
            cust2 = self.id_to_idx[route[i + 1]]
            cost += self.distance_matrix[cust1][cust2]
        
        # To depot from last customer
        if route:
            cost += self.distance_matrix[self.id_to_idx[route[-1]]][0]
        
        return cost
    
    def extract_solution(self, x, y, depot_idx, loc_idx_map, model):
        """Extract solution from Gurobi variables."""
        # Map back to original customer IDs and construct routes
        solution_routes = []
    
        # Convert Gurobi variable results to assignments: route[k] contains customers of k-th vehicle
        y_vals = {}
        for k in range(self.num_vehicles):
            y_vals[k] = []
            for i in range(self.num_customers):
                if round(y[i, k].X) == 1:
                    y_vals[k].append(i)  # This is index id in range(num_customers), convert to real id later

        # For each assigned customer list, derive a valid sequence using the x variables
        for k in range(self.num_vehicles):
            assigned_customer_indices = y_vals[k]
            assigned_customers = [self.idx_to_id[i] for i in assigned_customer_indices]
            
            if not assigned_customers:
                continue  # No customers assigned to this vehicle

            # Find the order of these customers in route using edges
            route_order = []
            current_pos = depot_idx  # Start from depot
            
            # Build the route step-by-step following the path
            customers_left = set(assigned_customer_indices)
            
            while customers_left:
                next_pos = None
                for j in assigned_customer_indices:
                    if round(x[current_pos, j, k].X) == 1:
                        next_pos = j
                        break
                if next_pos is None:
                    # Check if returning to depot
                    if round(x[current_pos, depot_idx, k].X) == 1:
                        # Done with this route
                        break
                    else:
                        # Error in routing - no valid edge to continue
                        raise Exception("Invalid route configuration detected in Gurobi solution.")
                
                route_order.append(next_pos)
                customers_left.remove(next_pos)
                current_pos = next_pos

            # Convert indices back to original IDs
            route_customer_ids = [self.idx_to_id[i] for i in route_order]

            # Calculate worst-case demand on route: nominal + top Gamma deviations if possible
            relevant_nominals = [self.customers[cid]['nominal_demand'] for cid in route_customer_ids]
            relevant_deviations = [self.customers[cid]['demand_deviation'] for cid in route_customer_ids]

            nominal_total = sum(relevant_nominals)
            # Get up to min(Gamma, len(deviations)) largest possible deviations
            max_devs_for_this_route = sorted(relevant_deviations, reverse=True)[:int(self.Gamma)]
            worst_case_deviation = sum(max_devs_for_this_route)

            worst_case_demand = nominal_total + worst_case_deviation

            cost = 0
            # Path from depot -> first customer
            depot_idx_internal = self.id_to_idx[self.depot_id]  # which is probably 0 based on distance_matrix structure
            first_customer_idx = self.id_to_idx[route_customer_ids[0]] if route_customer_ids else depot_idx_internal
            cost += self.distance_matrix[depot_idx_internal][first_customer_idx]
            
            # Between customers
            for i in range(len(route_customer_ids) - 1):
                cust1_idx = self.id_to_idx[route_customer_ids[i]]
                cust2_idx = self.id_to_idx[route_customer_ids[i + 1]]
                cost += self.distance_matrix[cust1_idx][cust2_idx]
            
            # Last customer -> depot
            if route_customer_ids:
                last_customer_idx = self.id_to_idx[route_customer_ids[-1]]
                cost += self.distance_matrix[last_customer_idx][depot_idx_internal]

            solution_routes.append({
                "customers": route_customer_ids,
                "cost": int(cost),
                "worst_case_demand": int(worst_case_demand)
            })
        
        # Verify all constraints are satisfied
        all_cust = set(self.customer_ids)
        routed_cust = set()
        for route in solution_routes:
            routed_cust.update(route['customers'])
            
        assert len(all_cust.difference(routed_cust)) == 0, "Not all customers were covered in the route"
        assert len(routed_cust.difference(all_cust)) == 0, "More customers than expected covered"
        
        total_cost = sum(route['cost'] for route in solution_routes)
        
        solution_output = {
            "objective_value": float(total_cost),
            "routes": solution_routes
        }
        
        return solution_output


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