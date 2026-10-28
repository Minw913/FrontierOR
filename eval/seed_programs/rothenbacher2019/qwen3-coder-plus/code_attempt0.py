import argparse
import json
import math
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def distance_and_cost_matrices(depot_data, customer_data):
    all_locations = [depot_data] + customer_data
    n = len(all_locations)
    dist_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                pos_i = (all_locations[i]['x'], all_locations[i]['y'])
                pos_j = (all_locations[j]['x'], all_locations[j]['y'])
                dist_matrix[i][j] = euclidean_distance(pos_i, pos_j)
    
    return dist_matrix

def get_distance(location_a, location_b, dist_matrix, depot_data, customer_data):
    # Map depot and customers to indices in the distance matrix
    all_locs = [depot_data] + customer_data
    
    idx_a = 0 if 'time_window' in location_a else customer_data.index(location_a) + 1
    idx_b = 0 if 'time_window' in location_b else customer_data.index(location_b) + 1
    
    return dist_matrix[idx_a][idx_b]

def solve_vrptw_sop(inst_path, sol_path, time_limit, log_path=None):
    # Load problem instance
    with open(inst_path, 'r') as f:
        inst = json.load(f)

    num_customers = inst["num_customers"]
    num_days = inst["num_days"]
    num_vehicles = inst["num_vehicles"]
    vehicle_capacity = inst["vehicle_capacity"]
    max_route_duration = inst["max_route_duration"]
    depot = inst["depot"]
    customers = inst["customers"]

    # Prepare distance matrix
    dist_matrix = distance_and_cost_matrices(depot, customers)

    # Create mapping to quickly refer to customers and their IDs
    id_to_customer_idx = {c['id']: i for i, c in enumerate(customers)}

    # Model setup with Gurobi
    model = gp.Model("VRPTW_SOP")
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", 0.001)
    if log_path:
        model.setParam("LogFile", log_path.replace(".jsonl", "_model.log"))

    # Decision variables:
    # 1. Schedule selection variables
    sch_vars = {}
    for cust_idx, customer in enumerate(customers):
        sch_vars[cust_idx] = []
        for s_idx, sched in enumerate(customer['schedules']):
            var = model.addVar(vtype=GRB.BINARY, name=f"sched_{cust_idx}_{s_idx}")
            sch_vars[cust_idx].append(var)

    # 2. Route-related variables
    # x[i,j,k,d] = 1 if vehicle k on day d goes from i to j
    x = {}
    # All locations (depots and customers): we'll use 0 for depot and 1..n for customers
    locs = range(0, num_customers + 1)  # 0=depot, 1..num_customers=customers

    for k in range(num_vehicles):
        for d in range(num_days):
            for i in locs:
                for j in locs:
                    if i != j:
                        x[i, j, k, d] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}_{d}")

    # u[i,k,d] = load before visiting customer i with vehicle k on day d
    # y[i,k,d] = start time of service at customer i with vehicle k on day d
    u = {}
    y = {}
    for k in range(num_vehicles):
        for d in range(num_days):
            for i in range(1, num_customers + 1):  # Only for customers (not depot)
                u[i, k, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                      ub=float(vehicle_capacity), name=f"u_{i}_{k}_{d}")
                y[i, k, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=depot['time_window'][0],
                                          ub=depot['time_window'][1], name=f"y_{i}_{k}_{d}")

    # Now handle visit assignments:
    # visit_assign_vars[cust_idx][part][k][d] = 1 if part-th visit of customer's selected schedule is assigned to vehicle k on day d
    visit_assign_vars = {}
    visit_map = {}  # Map to keep track of visit part IDs for constraints
    
    all_visits = []  # List of tuples: (cust_idx, part_idx, visit_day, days_covered, demand_per_visit)
    visit_id_counter = 0
    
    for cust_idx, customer in enumerate(customers):
        visit_assign_vars[cust_idx] = []
        
        for s_idx, sched in enumerate(customer['schedules']):
            visit_assign_vars[cust_idx].append([])
            
            for p_idx, part in enumerate(sched['parts']):
                visit_assign_vars[cust_idx][s_idx].append({})
                
                v_key = (cust_idx, p_idx, s_idx)
                visit_map[v_key] = visit_id_counter
                visit_id_counter += 1
                
                for k in range(num_vehicles):
                    for d in range(num_days):
                        if d == part['visit_day']:
                            var = model.addVar(vtype=GRB.BINARY, 
                                              name=f"visit_assign_c{cust_idx}_s{s_idx}_p{p_idx}_v{k}_d{d}")
                            visit_assign_vars[cust_idx][s_idx][p_idx][(k, d)] = var
                            all_visits.append((cust_idx, p_idx, visit_id_counter - 1))
    
    model.update()

    # Constraints
    # 1. Each customer selects exactly one schedule
    for cust_idx in range(len(customers)):
        model.addConstr(gp.quicksum(sch_vars[cust_idx][s_idx] 
                                   for s_idx in range(len(customers[cust_idx]['schedules']))) == 1)
    
    # 2. Flow conservation: each customer is left and entered by same vehicle
    for k in range(num_vehicles):
        for d in range(num_days):
            for i in range(1, num_customers + 1):  
                # Leaving customer i
                leaving = gp.quicksum(x[i, j, k, d] for j in locs if i != j)
                entering = gp.quicksum(x[j, i, k, d] for j in locs if i != j)
                model.addConstr(leaving == entering)
    
    # 3. Each vehicle starts and ends at depot
    for k in range(num_vehicles):
        for d in range(num_days):
            # Departure from depot equals return to depot
            out_from_depot = gp.quicksum(x[0, j, k, d] for j in range(1, num_customers + 1))
            into_depot = gp.quicksum(x[i, 0, k, d] for i in range(1, num_customers + 1))
            model.addConstr(out_from_depot == into_depot)
    
    # 4. Each vehicle must start at depot (but might not serve anyone)
    for k in range(num_vehicles):
        for d in range(num_days):
            model.addConstr(gp.quicksum(x[0, j, k, d] for j in range(1, num_customers + 1)) <= 1)

    # 5. Visit assignments according to selected schedule
    for cust_idx, customer in enumerate(customers):
        for s_idx, sched in enumerate(customer['schedules']):
            # Link schedule selection to visit assignments
            for p_idx, part in enumerate(sched['parts']):
                day = part['visit_day']
                total_visits_for_part = gp.quicksum(
                    visit_assign_vars[cust_idx][s_idx][p_idx].get((k, day), 0)
                    for k in range(num_vehicles)
                )
                model.addConstr(total_visits_for_part == sch_vars[cust_idx][s_idx])

    # 6. Each visit assignment must correspond to a route
    all_cust_parts_with_day = []
    for cust_idx, customer in enumerate(customers):
        for s_idx, sched in enumerate(customer['schedules']):
            for p_idx, part in enumerate(sched['parts']):
                visit_day = part['visit_day']
                
                for k in range(num_vehicles):
                    # If this is a valid visit assignment
                    visit_var = visit_assign_vars[cust_idx][s_idx][p_idx].get((k, visit_day), None)
                    if visit_var is not None:
                        # Then there must be an incoming arc to that customer on that day by that vehicle
                        # and outgoing arc too
                        incoming_sum = gp.quicksum(x[j, cust_idx+1, k, visit_day] for j in range(num_customers + 1) if j != cust_idx+1)
                        outgoing_sum = gp.quicksum(x[cust_idx+1, j, k, visit_day] for j in range(num_customers + 1) if j != cust_idx+1)
                        
                        model.addConstr(incoming_sum >= visit_var)
                        model.addConstr(outgoing_sum >= visit_var)

    # 7. Capacity constraints
    for k in range(num_vehicles):
        for d in range(num_days):
            total_load = gp.quicksum(
                x[i, j, k, d] * customers[i-1]['demand_per_day'] * 1  # placeholder
                for i in range(1, num_customers + 1)
                for j in locs if i != j
            )
            model.addConstr(total_load <= vehicle_capacity)

    # For proper capacity handling with actual visit demands:
    for k in range(num_vehicles):
        for d in range(num_days):
            route_demand = gp.quicksum(
                visit_assign_vars[cust_idx][s_idx][p_idx].get((k, d), 0) *
                cust_sched_p['demand_per_visit']
                for cust_idx, customer in enumerate(customers)
                for s_idx, sched in enumerate(customer['schedules'])
                for p_idx, cust_sched_p in enumerate(sched['parts'])
                if d == cust_sched_p['visit_day']
            )
            model.addConstr(route_demand == gp.quicksum(
                x[cust_idx+1, j, k, d] * customers[cust_idx]['demand_per_day']
                for cust_idx in range(num_customers)
                for j in locs if cust_idx+1 != j))

    # Add cumulative load constraints
    for k in range(num_vehicles):
        for d in range(num_days):
            for i in range(1, num_customers + 1):
                for j in locs:
                    if i != j:
                        # If edge (i->j) is used, update load at j
                        demand_i = customers[i-1]['demand_per_day']
                        model.addConstr(u[i, k, d] + demand_i * x[i, j, k, d] <= u[j, k, d] + vehicle_capacity*(1 - x[i, j, k, d]),
                                       f"load_flow_{i}_{j}_{k}_{d}")

    # Time-based constraints
    depot_tw_start = depot['time_window'][0]
    depot_tw_end = depot['time_window'][1]
    depot_service_time = depot['service_time']

    for k in range(num_vehicles):
        for d in range(num_days):
            # Departure time - must start within depot time window
            departure_times = []
            for j in range(1, num_customers + 1):
                travel_time = dist_matrix[0][j]  # Depot to customer j
                # Start time at j must allow arrival after travel from depot if using this arc
                model.addConstr(y[j, k, d] >= (depot_tw_start + dist_matrix[0][j]) * x[0, j, k, d])
                departure_times.append(y[j, k, d])
            
            # Arrival back at depot
            for i in range(1, num_customers + 1):
                # Ensure return to depot happens on time
                service_time_at_i = customers[i-1]['service_time']
                travel_time_back = dist_matrix[i][0]  # Customer i back to depot
                model.addConstr(
                    y[i, k, d] + service_time_at_i + dist_matrix[i][0]
                    <= depot_tw_end + (1- x[i, 0, k, d]) * 1e6
                )

                # Link travel time between consecutive customers
                for j in range(1, num_customers + 1):
                    if i != j and i > 0:  # Avoid depot
                        service_time_at_i = customers[i-1]['service_time']
                        travel_time = dist_matrix[i][j]
                        model.addConstr(
                            y[j, k, d] >= y[i, k, d] + service_time_at_i + dist_matrix[i][j]
                            - 1e6 * (1 - x[i, j, k, d]),
                            f"travel_{i}_{j}_{k}_{d}"
                        )

    # Customer time windows
    for ci, customer in enumerate(customers):
        for k in range(num_vehicles):
            for d in range(num_days):
                tw_start, tw_end = customer['time_window']
                model.addConstr(y[ci + 1, k, d] >= tw_start)
                model.addConstr(y[ci + 1, k, d] <= tw_end)

    # Route limit constraint: on each day, each vehicle used at most once
    for d in range(num_days):
        total_used_vehicles = gp.quicksum(
            x[0, j, k, d] for k in range(num_vehicles) for j in range(1, num_customers + 1)
        )
        model.addConstr(total_used_vehicles <= num_vehicles)

    # Objective function: minimize total travel cost
    obj_expr = gp.LinExpr()
    for k in range(num_vehicles):
        for d in range(num_days):
            for i in locs:
                for j in locs:
                    if i != j:
                        obj_expr += x[i, j, k, d] * dist_matrix[i][j]
    
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Solver callback for logging
    def logger_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if log_path:
                sl.callback_log(obj_val)

    # Setup logging object
    sl = SolutionLogger(log_path, "minimize") if log_path else None
    
    # Solve
    if sl:
        model.optimize(logger_callback)
    else:
        model.optimize()
    
    if model.SolCount == 0:
        raise Exception("No solution found")

    # Extract solution
    objective_value = model.objVal
    
    # Determine selected schedules
    final_schedules = {}
    for cust_idx, customer in enumerate(customers):
        for s_idx, sched in enumerate(customer['schedules']):
            if sch_vars[cust_idx][s_idx].X > 0.9:
                final_schedules[customer['id']] = {
                    'schedule_index': s_idx,
                    'days': sched['days'],
                    'visit_frequency': sched['visit_frequency']
                }
    
    # Determine routes
    final_routes = {}
    for d in range(num_days):
        day_routes = []
        for k in range(num_vehicles):
            route_customers = []
            route_parts = {}
            total_cost = 0
            total_demand = 0
            
            current_pos = 0  # Start at depot
            while True:
                next_visited = False
                for j in range(0, num_customers + 1):
                    if current_pos != j and x[current_pos, j, k, d].X > 0.9:
                        if j == 0:  # Back to depot
                            break
                        else:  # Visit to customer j
                            cust_id = customers[j-1]['id']
                            route_customers.append(cust_id)
                            
                            # Find corresponding visit details
                            for cust_idx, customer in enumerate(customers):
                                if customer['id'] == cust_id:
                                    for s_idx, sched in enumerate(customer['schedules']):
                                        for p_idx, part in enumerate(sched['parts']):
                                            visit_day = part['visit_day']
                                            if visit_day == d:
                                                visit_var = visit_assign_vars[cust_idx][s_idx][p_idx].get((k, d), None)
                                                if visit_var and visit_var.X > 0.9:
                                                    route_parts[cust_id] = {
                                                        'visit_day': part['visit_day'],
                                                        'days_covered': part['days_covered'],
                                                        'demand_per_visit': part['demand_per_visit']
                                                    }
                                                    total_demand += part['demand_per_visit']
                            
                            total_cost += dist_matrix[current_pos][j]
                            current_pos = j
                            next_visited = True
                            break
                if not next_visited or current_pos == 0:  # No next customer or returned to depot
                    if len(route_customers) > 0:  # Only record non-trivial routes
                        total_cost += dist_matrix[current_pos][0]  # Add cost to return to depot
                    break
        
        if len(route_customers) > 0:
            day_routes.append({
                'customers': route_customers.copy(),
                'cost': total_cost,
                'demand': total_demand,
                'parts': route_parts.copy()
            })
        
        final_routes[d] = day_routes.copy()

    solution = {
        "objective_value": objective_value,
        "routes": final_routes,
        "schedules": final_schedules
    }

    # Write solution
    with open(sol_path, 'w') as f:
        json.dump(solution, f, indent=2)

    return


def main():
    parser = argparse.ArgumentParser(description='Solve VRPTW SOP')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    solve_vrptw_sop(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()