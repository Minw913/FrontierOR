import argparse
import json
import math
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def solve_vrptw_sop(instance_path, solution_path, time_limit, log_path=None):
    with open(instance_path, 'r') as f:
        inst = json.load(f)
    
    # Initialize logger if path provided
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_customers = inst['num_customers'] 
    num_days = inst['num_days']
    num_vehicles = inst['num_vehicles']
    vehicle_capacity = inst['vehicle_capacity']
    max_route_duration = inst['max_route_duration']
    
    depot = inst['depot']
    customers = inst['customers']
    
    # Create ID to index mapping
    customer_id_to_idx = {c['id']: i for i, c in enumerate(customers)}
    
    # Calculate distances
    locations = [depot] + customers
    n = len(locations)
    distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            xi, yi = locations[i]['x'], locations[i]['y']
            xj, yj = locations[j]['x'], locations[j]['y']
            dist = calculate_distance((xi, yi), (xj, yj))
            distance_matrix[i][j] = dist
    
    # Gurobi model
    model = gp.Model('VRPTW-SOP')
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', False) 
    model.setParam('TimeLimit', time_limit)
    
    # Decision vars
    # Schedule selection: which schedule to choose for each customer
    y = {}  # y[customer_idx, schedule_idx]
    for cust_idx in range(num_customers):
        for sched_idx in range(len(customers[cust_idx]['schedules'])):
            y[cust_idx, sched_idx] = model.addVar(vtype=GRB.BINARY, 
                                                  name=f'y_c{cust_idx}_s{sched_idx}')
    
    # Routes - define a set of potential routes per day
    # We'll use the idea of arcs: x[day][k][i][j] = whether vehicle k travels from i to j on day d
    # We'll create a simplified model by defining routes based on visit sequences
    # Let R be a set of all possible (feasible) routes per day
    # However, explicitly enumerating routes is prohibitive, so we go for flows
    
    # Arc flow variables
    V_0 = [0]  # Depot (indexed as 0)
    V_plus = [i + 1 for i in range(len(customers))]  # Customers (indexed as 1..n)
    V = V_0 + V_plus  # All locations: depot (index 0) + customers (1..n)
    
    # Add flow variables based on arcs
    # x[d][k][i][j] for each day d, vehicle k, arc i->j
    x = {}
    for d in range(num_days):
        for k in range(num_vehicles):
            for i in V:
                for j in V:
                    if i != j:
                        x[i, j, k, d] = model.addVar(vtype=GRB.BINARY, 
                                                     name=f'x_d{d}_k{k}_i{i}_j{j}')
    
    # Load variables to enforce capacity
    load = {}
    for d in range(num_days):
        for k in range(num_vehicles):
            for i in V:
                load[i, k, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, 
                                           name=f'load_i{i}_k{k}_d{d}')
    
    # Arrival time variables
    T = {}
    big_M_time = 10000  # Large number for time constraints
    for d in range(num_days):
        for k in range(num_vehicles):
            for i in V:
                T[i, k, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, 
                                       ub=big_M_time, name=f'T_i{i}_k{k}_d{d}')
    
    # Add constraints
    # Each customer selects exactly one schedule
    for cust_idx in range(num_customers):
        model.addConstr(sum(y[cust_idx, sched_idx] 
                           for sched_idx in range(len(customers[cust_idx]['schedules']))) == 1,
                       name=f'single_schedule_c{cust_idx}')
    
    # Flow conservation at each node and day
    for d in range(num_days):
        for k in range(num_vehicles):
            # At depot: total outflow = inflow = number of vehicles used? Not necessarily.
            # Better: let vehicle k start at depot and return to depot.
            # So: out-deg(depot)=1 if vehicle k is used, =0 otherwise
            #     in-deg(depot)=1 if vehicle k is used, =0 otherwise
            
            # We approximate by constraining that a vehicle can only leave depot once and return once
            model.addConstr(
                sum(x[0, j, k, d] for j in V_plus if (0, j, k, d) in x) <= 1,
                name=f'depot_out_bound_k{k}_d{d}'
            ) 
            
            model.addConstr(
                sum(x[i, 0, k, d] for i in V_plus if (i, 0, k, d) in x) <= 1,
                name=f'depart_in_bound_k{k}_d{d}'
            )
    
    # Link customer visit to selected schedule
    # For each customer on each day, the number of times they are served must match their schedule  
    for cust_idx in range(num_customers):
        for d in range(num_days):
            # How many times this customer should be visited on this day according to current schedule?
            visits_on_day_d = gp.LinExpr()
            
            # Add up all schedules for customer that include a visit on day d
            for s_idx, sched in enumerate(customers[cust_idx]['schedules']):
                visit_count_in_sched = sum(1 for part in sched['parts'] if part['visit_day'] == d)
                visits_on_day_d.addTerms(visit_count_in_sched, y[cust_idx, s_idx])
                
            # Now count how often we visit this node
            customer_node_id = cust_idx + 1
            visit_count_actual = \
                sum(x[0, customer_node_id, k, d] for k in range(num_vehicles) if (0, customer_node_id, k, d) in x) + \
                sum(x[i, customer_node_id, k, d] for k in range(num_vehicles) 
                    for i in V_plus if i != customer_node_id and (i, customer_node_id, k, d) in x)
            
            model.addConstr(visits_on_day_d == visit_count_actual,
                           name=f'customer_service_match_c{cust_idx}_d{d}')
    
    # Capacity constraint
    for d in range(num_days):
        for k in range(num_vehicles):
            total_load = gp.quicksum(
                sum(p['demand_per_visit'] for si in range(len(c['schedules'])) 
                    for p in c['schedules'][si]['parts'] if p['visit_day'] == d) *
                sum(x[i, cj_idx+1, k, d] for i in V if (i, cj_idx+1, k, d) in x)
                for cj_idx, c in enumerate(customers)
            )
            model.addConstr(total_load <= vehicle_capacity,
                           name=f'vehicle_cap_k{k}_d{d}')
    
    # Set up callbacks if logger exists
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objval)
    
    # Set objective: minimize total travel cost
    total_cost = gp.quicksum(
        distance_matrix[i][j] * x[i, j, k, d]
        for d in range(num_days)
        for k in range(num_vehicles)
        for i in V
        for j in V
        if i != j and (i, j, k, d) in x
    )
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Try solving - add basic checks to avoid crash
    try:
        model.optimize(callback=mycallback if logger else None)
    except Exception as e:
        print("Gurobi error:", e)
        # Return empty solution in case of failure
        result = {
            "objective_value": float('inf'),
            "routes": {str(day): [] for day in range(num_days)},
            "schedules": {}
        }
        with open(solution_path, 'w') as f:
            json.dump(result, f, indent=2)
        return

    # Formulate output from model solution
    if model.SolCount == 0:
        result = {
            "objective_value": float('inf'),
            "routes": {str(day): [] for day in range(num_days)},
            "schedules": {}
        }
        with open(solution_path, 'w') as f:
            json.dump(result, f, indent=2)
        return

    objective_value = model.ObjVal if model.Status != GRB.INFEASIBLE else float('inf')

    # Extract schedule choices
    schedules = {}
    for cust_idx in range(num_customers):
        selected_sched = -1
        for sched_idx in range(len(customers[cust_idx]['schedules'])):
            if y[cust_idx, sched_idx].X > 0.5:
                selected_sched = sched_idx
                break
        if selected_sched != -1:
            schedules[customers[cust_idx]['id']] = {
                "schedule_index": selected_sched,
                "days": customers[cust_idx]["schedules"][selected_sched]["days"],
                "visit_frequency": customers[cust_idx]["schedules"][selected_sched]["visit_frequency"]
            }

    # Extract routes
    routes = {}
    for d in range(num_days):
        day_routes = []
        for k in range(num_vehicles):
            # Build the path taken by vehicle k on day d
            path = []
            for i in V_plus:  # For each customer
                # Check if this customer was visited by this vehicle
                visited_by_k = False
                
                # Check edges from depot to customer
                if (0, i, k, d) in x and x[0, i, k, d].X > 0.5:
                    visited_by_k = True
                    path.append(("start", i))  # depot to i
                # Check edges from another customer to i 
                for j in V_plus:
                    if i != j and (j, i, k, d) in x and x[j, i, k, d].X > 0.5:
                        visited_by_k = True
                        path.append((j, i))  # j to i

            if visited_by_k:  # Simplified logic: check if any edge is used
                # Actually reconstruct the tour
                vehicle_edges = []
                for i in V:
                    for j in V:
                        if i != j and (i, j, k, d) in x and x[i, j, k, d].X > 0.5:
                            vehicle_edges.append((i, j))
                
                if len(vehicle_edges) > 0:
                    # Reconstruct path
                    start_points = {e[0] for e in vehicle_edges}
                    end_points = {e[1] for e in vehicle_edges}
                    
                    # Find source nodes (out>in) - depot is usually source if not sink
                    path = extract_path(vehicle_edges, V_0, V_plus)
                    
                    customers_served = []
                    # Map index back to customer id
                    for node in path:
                        if node != 0:  # not depot
                            customers_served.append(customers[node-1]['id'])
                    
                    # Calculate cost of this route
                    total_distance = 0
                    prev = 0   # Start at depot
                    for node in path:
                        total_distance += distance_matrix[prev][node]
                        prev = node 
                    total_distance += distance_matrix[prev][0]  # Return to depot
                    
                    # Calculate route demand
                    total_demand = 0
                    route_parts = {}
                    for c_id in customers_served:
                        c_idx = customer_id_to_idx[c_id]
                        
                        # Find selected schedule for c_id
                        selected_si = -1
                        for si in range(len(customers[c_idx]['schedules'])):
                            if y[c_idx, si].X > 0.5:
                                selected_si = si
                                break
                        
                        # Look for visits for that customer on day d
                        if selected_si != -1:
                            for part in customers[c_idx]['schedules'][selected_si]['parts']:
                                if part['visit_day'] == d and customers[c_idx]['id'] == c_id:
                                    total_demand += part['demand_per_visit']
                                    route_parts[c_id] = {
                                        "visit_day": part["visit_day"],
                                        "days_covered": part["days_covered"], 
                                        "demand_per_visit": part["demand_per_visit"]
                                    }
                                    break

                    route_entry = {
                        "customers": customers_served,
                        "cost": total_distance,
                        "demand": total_demand,
                        "parts": route_parts
                    }
                    day_routes.append(route_entry)

        routes[str(d)] = day_routes

    result = {
        "objective_value": objective_value,
        "routes": routes,
        "schedules": schedules
    }

    # Write to output file
    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)

def extract_path(edges, depot_nodes, customer_nodes):
    # Create a map from each start point to its ends
    succ = {}
    for (i, j) in edges:
        succ[i] = j

    # Start pathfinding from the depot
    path = []
    current = 0  # Assuming 0 is always depot id
    visited = set()
    
    while current in succ and succ[current] not in visited:
        next_node = succ[current]
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    
    return path


def main():
    parser = argparse.ArgumentParser(description='Solve VRP SOP')
    parser.add_argument('--instance_path', required=True, help='Path to instance JSON')
    parser.add_argument('--solution_path', required=True, help='Path to output solution JSON')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', required=False, help='Path to log file (optional)')
    
    args = parser.parse_args()
    
    solve_vrptw_sop(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()