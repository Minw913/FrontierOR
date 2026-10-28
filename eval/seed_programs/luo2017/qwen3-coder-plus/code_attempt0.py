import argparse
import json
import math
from typing import Dict, List, Tuple
import gurobipy as gp
from gurobipy import GRB
import time
from solution_logger import SolutionLogger


def calculate_euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


class VRPSolver:
    def __init__(self, instance_data, time_limit, log_path=None):
        self.instance_data = instance_data
        self.time_limit = time_limit
        self.log_path = log_path
        
        # Extract parameters
        self.num_customers = instance_data["num_customers"]
        self.vehicle_capacity = instance_data["vehicle_capacity"]
        self.scheduling_horizon = instance_data["scheduling_horizon"]
        
        self.cost_a = instance_data["cost_function"]["a"]
        self.cost_b = instance_data["cost_function"]["b"]
        
        self.depot = instance_data["depot"]
        self.customers = {c["id"]: c for c in instance_data["customers"]}
        
        self.distance_matrix = instance_data["distance_matrix"]
        
        # Virtual depot for return (node n+1 in the solution schema)
        self.virtual_depot_id = self.num_customers + 1
        
        # Initialize logger
        self.logger = SolutionLogger(log_path, sense="minimize") if log_path else None
        
        # Create model
        self.model = gp.Model("VRP")
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('Seed', 0)
        self.model.setParam('MIPGap', 1e-4)
        self.model.setParam('NumericFocus', 0)
        self.model.setParam('Threads', 1)
        
        # Variables and sets
        self.x = {}  # x[i,j,k] = 1 if vehicle k travels from i to j
        self.f = {}  # f[i,j,k] = flow (weight) on edge (i,j) for vehicle k
        self.t = {}  # t[i,k] = arrival time at node i for vehicle k
        self.d = {}  # d[i,k] = delivery amount at node i by vehicle k
        self.y = {}  # y[i,k] = 1 if vehicle k visits node i
        
        # Node indices: 0 is depot, 1 to n are customers, n+1 is virtual depot
        self.nodes = list(range(self.num_customers + 2))
        self.customer_nodes = list(range(1, self.num_customers + 1))
        self.physical_nodes = list(range(self.num_customers + 1))  # without virtual depot
        self.all_vehicles = list(range(self.num_customers))  # upper bound on needed vehicles
        
    def build_model(self):
        # Add variables
        for k in self.all_vehicles:
            for i in self.physical_nodes:
                for j in self.physical_nodes:
                    if i != j:
                        self.x[i, j, k] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
                        
            # Flow variables for each edge (including virtual depot return)
            for i in self.nodes:
                for j in self.nodes:
                    if i != j and ((i < len(self.physical_nodes) and j < len(self.physical_nodes)) or 
                                   (i < len(self.physical_nodes) and j == self.virtual_depot_id) or
                                   (i == 0 and j == self.virtual_depot_id)):
                        self.f[i, j, k] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, 
                                                            ub=self.vehicle_capacity, name=f"f_{i}_{j}_{k}")
            
            # Arrival time variables
            for i in self.nodes:
                if i == self.virtual_depot_id:
                    max_time = self.scheduling_horizon
                else:
                    max_time = self.scheduling_horizon
                self.t[i, k] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=max_time, name=f"t_{i}_{k}")
                
            # Delivery variables
            for i in self.customer_nodes:
                self.d[i, k] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, 
                                                 ub=min(self.customers[i]["demand"], self.vehicle_capacity),
                                                 name=f"d_{i}_{k}")
                
            # Visit variables
            for i in self.physical_nodes:
                self.y[i, k] = self.model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")
                
        # Objective function: minimize total cost
        obj_expr = gp.LinExpr()
        for k in self.all_vehicles:
            for i in self.nodes:
                for j in self.nodes:
                    if i != j and ((i < len(self.physical_nodes) and j < len(self.physical_nodes)) or 
                                   (i < len(self.physical_nodes) and j == self.virtual_depot_id)):
                        # Calculate distance
                        if j == self.virtual_depot_id:
                            dist = self.distance_matrix[i][0]
                        else:
                            dist = self.distance_matrix[i][j]
                        
                        # Cost = distance * (a * load + b)
                        # We'll linearize this later using the flow variables
                        obj_expr.addTerms(dist * self.cost_a, self.f[i, j, k])
                        obj_expr.addConstant(dist * self.cost_b)
        
        self.model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Constraints
        # 1. Each customer's demand must be satisfied
        for i in self.customer_nodes:
            demand_expr = gp.LinExpr()
            for k in self.all_vehicles:
                demand_expr.addTerms(1.0, self.d[i, k])
            self.model.addConstr(demand_expr == self.customers[i]["demand"], 
                                name=f"demand_satisfaction_{i}")
        
        # 2. Flow conservation at depot
        for k in self.all_vehicles:
            outflow = gp.LinExpr()
            inflow = gp.LinExpr()
            for j in self.customer_nodes:
                outflow.addTerms(1.0, self.f[0, j, k])
            for j in self.customer_nodes:
                inflow.addTerms(1.0, self.f[j, 0, k])
            # Outflow equals inflow minus deliveries
            delivery_sum = gp.LinExpr()
            for i in self.customer_nodes:
                delivery_sum.addTerms(1.0, self.d[i, k])
            self.model.addConstr(outflow - inflow == delivery_sum, name=f"flow_conservation_depot_{k}")
        
        # 3. Flow conservation at customers
        for k in self.all_vehicles:
            for i in self.customer_nodes:
                in_flow = gp.LinExpr()
                out_flow = gp.LinExpr()
                
                # Incoming flows to customer i
                for j in self.physical_nodes:
                    if j != i:
                        in_flow.addTerms(1.0, self.f[j, i, k])
                
                # Outgoing flows from customer i
                for j in self.physical_nodes:
                    if j != i:
                        out_flow.addTerms(1.0, self.f[i, j, k])
                
                # Flow conservation: in - out = delivery
                self.model.addConstr(in_flow - out_flow == self.d[i, k], 
                                    name=f"flow_conservation_customer_{i}_vehicle_{k}")
        
        # 4. Link x and f: if x[i,j,k]=1 then f[i,j,k] > 0
        big_m = self.vehicle_capacity
        for k in self.all_vehicles:
            for i in self.physical_nodes:
                for j in self.physical_nodes:
                    if i != j:
                        self.model.addConstr(self.f[i, j, k] <= big_m * self.x[i, j, k], 
                                            name=f"flow_x_link_{i}_{j}_{k}")
        
        # 5. Vehicle visit constraints
        for k in self.all_vehicles:
            # At most one outgoing edge from depot
            out_from_depot = gp.LinExpr()
            for j in self.customer_nodes:
                out_from_depot.addTerms(1.0, self.x[0, j, k])
            self.model.addConstr(out_from_depot <= 1, name=f"out_from_depot_{k}")
        
        # 6. Flow out from depot equals sum of deliveries
        for k in self.all_vehicles:
            out_flow = gp.LinExpr()
            for j in self.customer_nodes:
                out_flow.addTerms(1.0, self.f[0, j, k])
            
            delivery_sum = gp.LinExpr()
            for i in self.customer_nodes:
                delivery_sum.addTerms(1.0, self.d[i, k])
            
            self.model.addConstr(out_flow == delivery_sum, name=f"depot_out_flow_{k}")
        
        # 7. Time constraints
        for k in self.all_vehicles:
            # Departure from depot respects time window
            self.model.addConstr(self.t[0, k] >= self.depot["time_window"][0], 
                                name=f"depot_departure_early_{k}")
            self.model.addConstr(self.t[0, k] + self.depot["service_time"] <= self.depot["time_window"][1],
                                name=f"depot_departure_late_{k}")
        
        # 8. Time consistency for edges
        for k in self.all_vehicles:
            for i in self.physical_nodes:
                for j in self.physical_nodes:
                    if i != j:
                        # If vehicle travels from i to j, time at j must be after arrival from i
                        service_time_i = self.depot["service_time"] if i == 0 else self.customers[i]["service_time"]
                        travel_time = self.distance_matrix[i][j]
                        
                        # Big-M constraint: t[j,k] >= t[i,k] + service_time[i] + travel_time - M*(1-x[i,j,k])
                        big_m = self.scheduling_horizon
                        self.model.addConstr(
                            self.t[j, k] >= self.t[i, k] + service_time_i + travel_time - 
                            big_m * (1 - self.x[i, j, k]),
                            name=f"time_consistency_{i}_{j}_{k}"
                        )
        
        # 9. Return to virtual depot time window
        for k in self.all_vehicles:
            # If vehicle returns to virtual depot
            self.model.addConstr(self.t[self.virtual_depot_id, k] >= self.depot["time_window"][0],
                                name=f"return_early_{k}")
            self.model.addConstr(self.t[self.virtual_depot_id, k] <= self.depot["time_window"][1],
                                name=f"return_late_{k}")
        
        # 10. Link time variables to actual returns
        for k in self.all_vehicles:
            for i in self.customer_nodes:
                # If vehicle goes from i to virtual depot
                service_time_i = self.customers[i]["service_time"]
                travel_time = self.distance_matrix[i][0]  # back to depot
                
                # Big-M constraint for virtual depot arrival time
                big_m = self.scheduling_horizon
                self.model.addConstr(
                    self.t[self.virtual_depot_id, k] >= self.t[i, k] + service_time_i + travel_time - 
                    big_m * (1 - self.x[i, 0, k]),
                    name=f"virtual_return_time_{i}_{k}"
                )
        
        # 11. Capacity constraints
        for k in self.all_vehicles:
            total_delivery = gp.LinExpr()
            for i in self.customer_nodes:
                total_delivery.addTerms(1.0, self.d[i, k])
            self.model.addConstr(total_delivery <= self.vehicle_capacity,
                                name=f"capacity_vehicle_{k}")
        
        # 12. Time window constraints at customers
        for i in self.customer_nodes:
            for k in self.all_vehicles:
                # If customer i is visited by vehicle k
                self.model.addConstr(self.t[i, k] >= self.customers[i]["time_window"][0],
                                    name=f"customer_early_{i}_vehicle_{k}")
                self.model.addConstr(self.t[i, k] <= self.customers[i]["time_window"][1],
                                    name=f"customer_late_{i}_vehicle_{k}")
    
    def solve(self):
        # Solve the model
        self.model.optimize()
        
        # Extract solution
        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            solution = self.extract_solution()
            return solution
        else:
            print(f"Model status: {self.model.status}")
            return None
    
    def extract_solution(self):
        routes = []
        
        for k in self.all_vehicles:
            # Check if vehicle k is used
            vehicle_used = False
            for i in self.physical_nodes:
                for j in self.physical_nodes:
                    if i != j and self.x[i, j, k].X > 0.5:
                        vehicle_used = True
                        break
                if vehicle_used:
                    break
            
            if not vehicle_used:
                continue
            
            # Build route for vehicle k
            route = [0]  # Start at depot
            current_node = 0
            visited = {0}
            
            # Find next nodes in route
            while True:
                next_node_found = False
                for j in self.physical_nodes:
                    if j not in visited and self.x[current_node, j, k].X > 0.5:
                        route.append(j)
                        current_node = j
                        visited.add(j)
                        next_node_found = True
                        break
                
                if not next_node_found:
                    break
            
            route.append(self.virtual_depot_id)  # End at virtual depot
            
            # Get deliveries
            deliveries = {}
            for i in self.customer_nodes:
                delivery_val = self.d[i, k].X
                if delivery_val > 1e-5:  # Only include significant deliveries
                    deliveries[str(i)] = delivery_val
            
            # Get edges with load weights
            edges = []
            for idx in range(len(route) - 1):
                from_node = route[idx]
                to_node = route[idx + 1]
                
                # Find the corresponding flow variable
                if to_node == self.virtual_depot_id:
                    # This is a return to virtual depot from physical depot
                    if from_node == 0:
                        load_weight = self.f[0, 0, k].X if (0, 0, k) in self.f else 0
                    else:
                        # Edge from customer to depot
                        load_weight = self.f[from_node, 0, k].X
                else:
                    load_weight = self.f[from_node, to_node, k].X
                
                edges.append({
                    "from": from_node,
                    "to": to_node,
                    "load_weight": load_weight
                })
            
            routes.append({
                "vehicle": k,
                "route": route,
                "deliveries": deliveries,
                "edges": edges
            })
        
        # Calculate objective value
        total_cost = self.model.ObjVal if self.model.status == GRB.OPTIMAL else float('inf')
        
        solution = {
            "objective_value": total_cost,
            "routes": routes
        }
        
        return solution


def main():
    parser = argparse.ArgumentParser(description='Solve VRP with Split Deliveries')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file')
    
    args = parser.parse_args()
    
    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Create solver
    solver = VRPSolver(instance_data, args.time_limit, args.log_path)
    
    # Build and solve the model
    solver.build_model()
    
    # Solve and get solution
    solution = solver.solve()
    
    if solution is not None:
        # Write solution to file
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("No solution found")


if __name__ == "__main__":
    main()