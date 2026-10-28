import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
import math
from solution_logger import SolutionLogger

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    time_limit = args.time_limit

    # Extract data
    num_customers = instance['num_customers']
    num_days = instance['num_days']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    max_route_duration = instance['max_route_duration']
    depot = instance['depot']
    customers = instance['customers']
    planning_horizon_cyclic = instance.get('planning_horizon_cyclic', True)
    
    # Locations: 0 = depot, 1..N = customers
    N = num_customers
    locations = [depot] + customers
    x_coords = [loc['x'] for loc in locations]
    y_coords = [loc['y'] for loc in locations]
    service_times = [loc.get('service_time', 0) for loc in locations]
    time_windows = [loc.get('time_window', [0, 1440]) for loc in locations]

    # Precompute distances and travel times (distance + service at origin)
    dist = [[0.0] * (N+1) for _ in range(N+1)]
    travel_time = [[0.0] * (N+1) for _ in range(N+1)]
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                d = euclidean_distance(x_coords[i], y_coords[i], x_coords[j], y_coords[j])
                dist[i][j] = d
                travel_time[i][j] = d + service_times[i]

    depot_tw = time_windows[0]
    depot_earliest = float(depot_tw[0])
    depot_latest = float(depot_tw[1])

    # Customer data
    demand_per_day_list = [c['demand_per_day'] for c in customers]
    cust_time_windows = [c['time_window'] for c in customers]
    cust_service = [c['service_time'] for c in customers]
    cust_ids = [c['id'] for c in customers]

    # Build visit_options: for each customer, list of possible (day, days_covered, demand, schedule_idx, part_idx)
    visit_options = []
    for c_idx, cust in enumerate(customers):
        cust_visits = []
        for s_idx, sched in enumerate(cust['schedules']):
            for p_idx, part in enumerate(sched.get('parts', [])):
                vd = part['visit_day']
                dc = part['days_covered']
                dem = part['demand_per_visit']
                cust_visits.append((vd, dc, dem, s_idx, p_idx))
        visit_options.append(cust_visits)

    # Create MIP
    model = gp.Model("PeriodicVRPTW")
    model.setParam('TimeLimit', max(1, time_limit - 10))
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('LogToConsole', 0)
    model.setParam('Presolve', 2)
    model.setParam('Heuristics', 0.8)

    # Select one schedule per customer
    schedule_vars = {}
    for c in range(N):
        num_sched = len(customers[c]['schedules'])
        for s in range(num_sched):
            schedule_vars[c, s] = model.addVar(vtype=GRB.BINARY, name=f"sel_c{c}_s{s}")
        model.addConstr(gp.quicksum(schedule_vars[c, s] for s in range(num_sched)) == 1, name=f"one_sched_c{c}")

    # Visit use variables
    visit_use = {}
    for c in range(N):
        for o_idx, (vd, dc, dem, s_idx, _) in enumerate(visit_options[c]):
            visit_use[c, vd, o_idx] = model.addVar(vtype=GRB.BINARY, name=f"vu_c{c}_d{vd}_o{o_idx}")
            model.addConstr(visit_use[c, vd, o_idx] <= schedule_vars[c, s_idx])

    # Force visits according to selected schedule
    for c in range(N):
        cust = customers[c]
        for s_idx, sched in enumerate(cust['schedules']):
            for v_idx, part in enumerate(sched.get('parts', [])):
                vd = part['visit_day']
                dc = part['days_covered']
                # find matching o_idx
                for o_idx, opt in enumerate(visit_options[c]):
                    if opt[0] == vd and opt[1] == dc and opt[3] == s_idx:
                        model.addConstr(visit_use[c, vd, o_idx] == schedule_vars[c, s_idx])
                        break

    # y[d, c] = 1 if customer c visited on day d
    y = {}
    for d in range(num_days):
        for c in range(N):
            y[d, c] = model.addVar(vtype=GRB.BINARY, name=f"y_d{d}_c{c}")
            possible_o = [o_idx for o_idx, (vd, _, _, _, _) in enumerate(visit_options[c]) if vd == d]
            if possible_o:
                model.addConstr(y[d, c] == gp.quicksum(visit_use[c, d, o_idx] for o_idx in possible_o))
            else:
                model.addConstr(y[d, c] == 0)

    # Routing variables per day
    x = {}
    for d in range(num_days):
        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    x[d, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x{d}_{i}_{j}")

    # Flow conservation
    for d in range(num_days):
        for c in range(N):
            # inflow = outflow = y
            model.addConstr(gp.quicksum(x[d, i, c+1] for i in range(N+1) if i != c+1) == y[d, c])
            model.addConstr(gp.quicksum(x[d, c+1, j] for j in range(N+1) if j != c+1) == y[d, c])

    # Vehicle limit
    for d in range(num_days):
        vehicles = gp.quicksum(x[d, 0, j] for j in range(1, N+1))
        model.addConstr(vehicles <= num_vehicles)
        model.addConstr(gp.quicksum(x[d, i, 0] for i in range(1, N+1)) == vehicles)

    # Capacity
    load = {}
    M_cap = vehicle_capacity * 2
    for d in range(num_days):
        for i in range(N + 1):
            load[d, i] = model.addVar(0, vehicle_capacity, vtype=GRB.CONTINUOUS, name=f"load{d}_{i}")
        
        model.addConstr(load[d, 0] == 0)
        
        for c in range(N):
            i = c + 1
            dem_expr = gp.LinExpr()
            for o_idx, (vd, dc, dem, _, _) in enumerate(visit_options[c]):
                if vd == d:
                    dem_expr += dem * visit_use[c, d, o_idx]
            model.addConstr(load[d, i] >= dem_expr - M_cap * (1 - y[d, c]))
            model.addConstr(load[d, i] <= dem_expr + M_cap * (1 - y[d, c]))

        for i in range(N + 1):
            for j in range(1, N + 1):
                if i == j: continue
                i_c = i - 1 if i > 0 else -1
                dem_j = gp.LinExpr()
                if j >= 1:
                    for o_idx, (vd, dc, dem, _, _) in enumerate(visit_options[j-1]):
                        if vd == d:
                            dem_j += dem * visit_use[j-1, d, o_idx]
                model.addConstr(load[d, j] >= load[d, i] + dem_j - M_cap * (1 - x[d, i, j]))
                model.addConstr(load[d, j] <= load[d, i] + dem_j + M_cap * (1 - x[d, i, j]))

    # Time and TW
    t = {}
    bigM_time = 3000.0
    for d in range(num_days):
        for i in range(N + 1):
            t[d, i] = model.addVar(0, 3000, vtype=GRB.CONTINUOUS, name=f"t{d}_{i}")

    return_time = {}
    for d in range(num_days):
        return_time[d] = model.addVar(0, 3000, vtype=GRB.CONTINUOUS, name=f"rt{d}")

    for d in range(num_days):
        model.addConstr(t[d, 0] >= depot_earliest)
        model.addConstr(return_time[d] <= depot_latest)
        model.addConstr(return_time[d] >= t[d, 0])
        model.addConstr(return_time[d] - t[d, 0] <= max_route_duration)

        for j in range(1, N+1):
            model.addConstr(return_time[d] >= t[d, j] + travel_time[j][0] - bigM_time * (1 - x[d, j, 0]))

        # Time windows on customers
        for c in range(N):
            i = c + 1
            e, l = cust_time_windows[c]
            model.addConstr(t[d, i] >= e - bigM_time * (1 - y[d, c]))
            model.addConstr(t[d, i] <= l + bigM_time * (1 - y[d, c]))

        # Time propagation on arcs
        for i in range(N + 1):
            for j in range(N + 1):
                if i == j: continue
                if i == 0:
                    model.addConstr(t[d, j] >= t[d, 0] + travel_time[0][j] - bigM_time * (1 - x[d, i, j]))
                else:
                    model.addConstr(t[d, j] >= t[d, i] + travel_time[i][j] - bigM_time * (1 - x[d, i, j]))

    # Objective
    total_cost = gp.quicksum(dist[i][j] * x[d, i, j] for d in range(num_days) 
                            for i in range(N+1) for j in range(N+1) if i != j)
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Callback for incumbent logging
    best_obj = float('inf')
    incumbent_count = [0]
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-4:
                best_obj = obj
                incumbent_count[0] += 1
                if logger and incumbent_count[0] % 3 == 0:  # limit logging frequency
                    logger.log(float(obj))

    # Solve with timeout protection
    try:
        model.optimize(callback)
    except Exception:
        pass

    # Get best objective safely
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.SOLUTION_LIMIT:
        final_obj = model.objVal
    else:
        final_obj = 0.0

    # Extract solution ONLY from variables with solution (use .X with care)
    schedules_out = {}
    solution_found = (model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT) and 
                     hasattr(model, 'objVal') and model.objVal < GRB.INFINITY)

    if solution_found:
        for c in range(N):
            cust_id = str(customers[c]['id'])
            chosen_s = None
            for s_idx in range(len(customers[c]['schedules'])):
                if schedule_vars[c, s_idx].X > 0.5:
                    chosen_s = s_idx
                    break
            if chosen_s is not None:
                sched = customers[c]['schedules'][chosen_s]
                schedules_out[cust_id] = {
                    "schedule_index": int(chosen_s),
                    "days": [int(day) for day in sched.get("days", [])],
                    "visit_frequency": int(sched.get("visit_frequency", 0))
                }
            else:
                # fallback: pick first schedule
                schedules_out[cust_id] = {
                    "schedule_index": 0,
                    "days": [int(day) for day in customers[c]['schedules'][0].get("days", [])],
                    "visit_frequency": int(customers[c]['schedules'][0].get("visit_frequency", 0))
                }
    else:
        # No solution: return empty but valid structure
        for c in range(N):
            cust_id = str(customers[c]['id'])
            schedules_out[cust_id] = {
                "schedule_index": 0,
                "days": [int(day) for day in customers[c]['schedules'][0].get("days", [])],
                "visit_frequency": int(customers[c]['schedules'][0].get("visit_frequency", 0))
            }

    routes_out = {}
    for d in range(num_days):
        day_routes = []
        if solution_found:
            visited = [False] * (N + 1)
            for start in range(1, N + 1):
                if x.get((d, 0, start)) and x[d, 0, start].X > 0.5 and not visited[start]:
                    route_model_idx = []
                    current = start
                    visited[current] = True
                    route_model_idx.append(current)
                    
                    while True:
                        next_c = None
                        for nxt in range(1, N + 1):
                            if x.get((d, current, nxt)) and x[d, current, nxt].X > 0.5 and not visited[nxt]:
                                next_c = nxt
                                break
                        if next_c is None:
                            break
                        current = next_c
                        visited[current] = True
                        route_model_idx.append(current)

                    # Build output route
                    customers_list = []
                    parts = {}
                    route_cost = dist[0][route_model_idx[0]]
                    route_demand = 0.0
                    prev = 0

                    for k, model_idx in enumerate(route_model_idx):
                        c_idx = model_idx - 1
                        c_id = customers[c_idx]['id']
                        customers_list.append(int(c_id))

                        # Get visit info
                        assigned = False
                        for o_idx, (vd, dc, dem, _, _) in enumerate(visit_options[c_idx]):
                            key = (c_idx, d, o_idx)
                            if vd == d and key in visit_use and visit_use[key].X > 0.5:
                                parts[str(c_id)] = {
                                    "visit_day": int(vd),
                                    "days_covered": int(dc),
                                    "demand_per_visit": int(dem)
                                }
                                route_demand += dem
                                assigned = True
                                break
                        if not assigned and visit_options[c_idx]:
                            # fallback
                            vd, dc, dem, _, _ = visit_options[c_idx][0]
                            parts[str(c_id)] = {
                                "visit_day": int(vd),
                                "days_covered": int(dc),
                                "demand_per_visit": int(dem)
                            }
                            route_demand += dem

                        if k < len(route_model_idx) - 1:
                            nxt_idx = route_model_idx[k + 1]
                            route_cost += dist[model_idx][nxt_idx]
                        else:
                            route_cost += dist[model_idx][0]

                    route_entry = {
                        "customers": customers_list,
                        "cost": float(route_cost),
                        "demand": float(route_demand),
                        "parts": parts
                    }
                    day_routes.append(route_entry)
        
        routes_out[str(d)] = day_routes

    solution = {
        "objective_value": float(final_obj),
        "routes": routes_out,
        "schedules": schedules_out
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger and final_obj < float('inf'):
        logger.log(float(final_obj))

if __name__ == "__main__":
    main()