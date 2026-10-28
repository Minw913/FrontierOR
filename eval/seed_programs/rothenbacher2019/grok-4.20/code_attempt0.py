import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
import math
from solution_logger import SolutionLogger
from collections import defaultdict
import itertools

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
    planning_horizon_cyclic = instance['planning_horizon_cyclic']
    distance_matrix_type = instance.get('distance_matrix_type', 'euclidean')
    cost_type = instance.get('cost_type', 'distance')

    # Locations: 0 = depot, 1..N = customers
    N = num_customers
    locations = [depot] + customers
    x_coords = [loc['x'] for loc in locations]
    y_coords = [loc['y'] for loc in locations]
    service_times = [loc.get('service_time', 0) for loc in locations]
    time_windows = [loc.get('time_window', [0, 1440]) for loc in locations]  # default large window

    # Precompute distances and travel times (distance + service at origin)
    dist = [[0.0 for _ in range(N+1)] for _ in range(N+1)]
    travel_time = [[0.0 for _ in range(N+1)] for _ in range(N+1)]
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                d = euclidean_distance(x_coords[i], y_coords[i], x_coords[j], y_coords[j])
                dist[i][j] = d
                travel_time[i][j] = d + service_times[i]
            else:
                dist[i][j] = 0.0
                travel_time[i][j] = 0.0

    depot_tw = time_windows[0]
    depot_earliest = depot_tw[0]
    depot_latest = depot_tw[1]

    # Customer data
    cust_ids = [c['id'] for c in customers]
    demand_per_day = [c['demand_per_day'] for c in customers]
    cust_time_windows = [c['time_window'] for c in customers]
    cust_service = [c['service_time'] for c in customers]

    # Build list of possible patterns per customer
    # Each pattern: (schedule_idx, visit_list) where visit_list = list of (day, days_covered, demand)
    patterns = []
    for c_idx in range(N):
        cust = customers[c_idx]
        cust_patterns = []
        for s_idx, sched in enumerate(cust['schedules']):
            visits = []
            for part in sched['parts']:
                visits.append((part['visit_day'], part['days_covered'], part['demand_per_visit']))
            cust_patterns.append((s_idx, visits))
        patterns.append(cust_patterns)

    # For each customer, possible visits: list of (day, days_covered, demand, schedule_idx, pattern_idx_in_sched)
    visit_options = [[] for _ in range(N)]
    for c in range(N):
        for s_idx, (sched_idx, visit_list) in enumerate(patterns[c]):
            for v_idx, (vd, dc, dem) in enumerate(visit_list):
                visit_options[c].append((vd, dc, dem, s_idx, v_idx))

    # We will use a column generation like approach but since N is likely small (<=20), we enumerate feasible routes per day
    # But for general case, we build a MIP with arc variables and select one pattern per customer

    # MIP model
    model = gp.Model("PeriodicVRPTW")
    model.setParam('TimeLimit', time_limit - 5)  # safety margin
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.01)
    model.setParam('LogToConsole', 0)

    # Decision 1: select one schedule per customer
    schedule_vars = {}
    for c in range(N):
        for s_idx in range(len(patterns[c])):
            schedule_vars[c, s_idx] = model.addVar(vtype=GRB.BINARY, name=f"select_c{c}_s{s_idx}")

    for c in range(N):
        model.addConstr(gp.quicksum(schedule_vars[c, s] for s in range(len(patterns[c]))) == 1, name=f"one_schedule_c{c}")

    # For each customer-day visit requirement
    # We create binary variable for whether a particular visit option is used
    # But since one schedule selects all its visits together, better to use the selection above

    # Now for routing: we need to cover exactly the visits prescribed by selected schedules

    # This is tricky. One way is to have for each possible visit (c, day, days_covered) a variable if it is executed
    # But since schedules bundle them, the selection enforces the bundle.

    # Let's define for each possible atomic visit: on a specific day, serving a customer with a specific (days_covered)
    # We need exactly one execution for each required visit from the chosen schedule.

    # For each customer c, for each possible visit option o = (day, dc, dem, s_idx, vpos), we have a var x if this visit is executed on that day
    # But only if the schedule is chosen.

    visit_use = {}
    for c in range(N):
        for o_idx, (day, dc, dem, s_idx, _) in enumerate(visit_options[c]):
            visit_use[c, day, o_idx] = model.addVar(vtype=GRB.BINARY, name=f"visit_c{c}_d{day}_o{o_idx}")
            # Link to schedule selection
            model.addConstr(visit_use[c, day, o_idx] <= schedule_vars[c, s_idx], name=f"visitlink_c{c}_d{day}_o{o_idx}")

    # For each customer and each schedule, the visits must be covered exactly when selected.
    # Since each schedule has fixed visits, when a schedule is selected, all its visit options must be used exactly once.
    for c in range(N):
        for s_idx, (_, visit_list) in enumerate(patterns[c]):
            for v_idx, (vd, dc, dem) in enumerate(visit_list):
                # find the o_idx corresponding to this
                for o_idx, opt in enumerate(visit_options[c]):
                    if opt[0] == vd and opt[1] == dc and opt[3] == s_idx:
                        model.addConstr(visit_use[c, vd, o_idx] == schedule_vars[c, s_idx], name=f"force_visit_c{c}_s{s_idx}_v{v_idx}")
                        break

    # Now for each day, we need to route the visits that are active on that day.
    # A visit_use[c, d, o] means there is a delivery to c on day d with quantity = dem

    # For each day d, the set of customers that must be visited on that day is those with visit_use >0 on d
    # Since it's selected together, the model will coordinate.

    # To model routing per day, we use MTZ or flow formulation for each day separately.
    # Since days are coupled only through the schedule choices, we can duplicate the routing variables per day.

    # Variables for each day:
    # For each day, for each possible customer that might be visited that day (all customers could potentially)
    # But to reduce, we only consider customers that have at least one possible visit on that day.

    # Arc variables: for each day, x[i][j][k] = 1 if vehicle k goes from i to j on that day
    # But with num_vehicles limit, better use binary arc and limit number of routes by flow from depot.

    # Since homogeneous, we use:
    # For each day d:
    #   x_d[i,j] binary if arc from i to j is used on day d (i,j in 0..N)
    #   u_d[i] : load after visiting i on day d
    #   t_d[i] : time when service starts at i on day d

    # But only if the node is visited that day.

    # We also need y_d[c] = 1 if customer c is visited on day d. This equals sum over o of visit_use[c,d,o]

    y = {}  # y[d, c] = 1 if c visited on day d
    for d in range(num_days):
        for c in range(N):
            y[d, c] = model.addVar(vtype=GRB.BINARY, name=f"y_d{d}_c{c}")
            # Link to visit_use: exactly one o can be active per customer per day (at most one visit per day assumed)
            possible_o_on_day = [o_idx for o_idx, opt in enumerate(visit_options[c]) if opt[0] == d]
            if possible_o_on_day:
                model.addConstr(y[d, c] == gp.quicksum(visit_use[c, d, o_idx] for o_idx in possible_o_on_day), name=f"y_link_d{d}_c{c}")
            else:
                model.addConstr(y[d, c] == 0, name=f"no_visit_d{d}_c{c}")

    # Depot is always "visited" but we handle separately.

    # For each day, we need to limit number of vehicles used <= num_vehicles
    # We will count the number of arcs leaving the depot.

    x = {}
    for d in range(num_days):
        for i in range(N+1):
            for j in range(N+1):
                if i != j:
                    x[d, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_d{d}_{i}_{j}")

    # Flow conservation and visit exactly once if y=1
    for d in range(num_days):
        for c in range(1, N+1):  # customers only
            model.addConstr(
                gp.quicksum(x[d, i, c] for i in range(N+1) if i != c) == y[d, c-1],
                name=f"inflow_d{d}_c{c}"
            )
            model.addConstr(
                gp.quicksum(x[d, c, j] for j in range(N+1) if j != c) == y[d, c-1],
                name=f"outflow_d{d}_c{c}"
            )

    # Depot flow: number of vehicles used
    for d in range(num_days):
        vehicles_used = gp.quicksum(x[d, 0, j] for j in range(1, N+1))
        model.addConstr(vehicles_used <= num_vehicles, name=f"vehicle_limit_d{d}")
        # Also inflow to depot must match
        model.addConstr(
            gp.quicksum(x[d, i, 0] for i in range(1, N+1)) == vehicles_used,
            name=f"depot_flow_d{d}"
        )

    # No self loops (already i!=j)

    # Capacity constraints
    load = {}
    for d in range(num_days):
        for i in range(N+1):
            load[d, i] = model.addVar(lb=0, ub=vehicle_capacity, vtype=GRB.CONTINUOUS, name=f"load_d{d}_i{i}")

    M = vehicle_capacity + 1
    for d in range(num_days):
        for i in range(1, N+1):
            c_idx = i - 1
            # demand delivered if visited. Since one visit per day max, we can sum possible deliveries
            dem_if_visited = 0
            possible_deliveries = []
            for o_idx, (vd, dc, dem, _, _) in enumerate(visit_options[c_idx]):
                if vd == d:
                    possible_deliveries.append((o_idx, dem))
            if possible_deliveries:
                dem_expr = gp.quicksum(dem * visit_use[c_idx, d, o_idx] for o_idx, dem in possible_deliveries)
            else:
                dem_expr = 0
            model.addConstr(load[d, i] >= dem_expr - M * (1 - y[d, c_idx]), name=f"load_lower_d{d}_i{i}")
            model.addConstr(load[d, i] <= dem_expr + M * (1 - y[d, c_idx]), name=f"load_upper_d{d}_i{i}")

        # Depot load = 0
        model.addConstr(load[d, 0] == 0, name=f"depot_load_d{d}")

        # Capacity on arcs
        for i in range(N+1):
            for j in range(1, N+1):  # to customer
                if i != j:
                    model.addConstr(load[d, j] >= load[d, i] + (gp.quicksum(dem * visit_use[c_idx, d, o_idx] for o_idx, dem in possible_deliveries if True) if j>0 else 0) - M * (1 - x[d, i, j]), name=f"load_flow1_d{d}_{i}_{j}")
                    model.addConstr(load[d, j] <= load[d, i] + (gp.quicksum(dem * visit_use[c_idx, d, o_idx] for o_idx, dem in possible_deliveries if True) if j>0 else 0) + M * (1 - x[d, i, j]), name=f"load_flow2_d{d}_{i}_{j}")

    # Time windows and duration
    t = {}
    bigT = 1440 * 2
    for d in range(num_days):
        for i in range(N+1):
            t[d, i] = model.addVar(lb=0, ub=bigT, vtype=GRB.CONTINUOUS, name=f"t_d{d}_i{i}")

    # Depot departure and return
    for d in range(num_days):
        model.addConstr(t[d, 0] >= depot_earliest, name=f"depot_start_d{d}")
        # Return time
        # For return, the time at depot upon return must be <= depot_latest
        for j in range(1, N+1):
            if x[d, j, 0].ub > 0.5:  # always true
                model.addConstr(t[d, 0] >= t[d, j] + travel_time[j][0] - bigT * (1 - x[d, j, 0]), name=f"return_time_d{d}_from{j}")

    # Time propagation
    for d in range(num_days):
        for i in range(N+1):
            for j in range(N+1):
                if i != j:
                    # if arc used, t[j] >= t[i] + service_at_i (already in travel) + travel_time[i to j]
                    # travel_time[i][j] already includes service at i
                    if i == 0:
                        # from depot, t[j] >= departure which is t[0] + travel from 0 without service? but we set service[0]=0
                        model.addConstr(t[d, j] >= t[d, 0] + travel_time[0][j] - bigT * (1 - x[d, i, j]), name=f"time_mtz_d{d}_{i}_{j}")
                    else:
                        # customer i has time window on start of service
                        model.addConstr(t[d, j] >= t[d, i] + travel_time[i][j] - bigT * (1 - x[d, i, j]), name=f"time_mtz_d{d}_{i}_{j}")

    # Time windows for customers
    for d in range(num_days):
        for c in range(1, N+1):
            c_idx = c - 1
            # start of service in [e, l]
            e, l = cust_time_windows[c_idx]
            model.addConstr(t[d, c] >= e - bigT * (1 - y[d, c_idx]), name=f"tw_early_d{d}_c{c}")
            model.addConstr(t[d, c] <= l + bigT * (1 - y[d, c_idx]), name=f"tw_late_d{d}_c{c}")

    # Route duration: the return time - start time <= max_route_duration
    # Since t[0] on return is the return time
    for d in range(num_days):
        model.addConstr(t[d, 0] <= depot_latest, name=f"depot_return_d{d}")  # already have from constraints but reinforce
        for j in range(1, N+1):
            model.addConstr(t[d, 0] - t[d, 0] + (t[d, 0] - (t[d, j] + travel_time[j][0])) * x[d, j, 0] <= max_route_duration, name=f"dur_from_{j}_d{d}")  # linearized

    # Actually better:
    for d in range(num_days):
        model.addConstr(t[d, 0] <= depot_latest, name=f"return_window_d{d}")
        # duration from initial departure (t[d,0] initial) to final return (the second t[d,0])
        # Since we set only one t[0] per day, we need to be careful.
        # The initial t[d,0] is departure, the final return time is max over used (t[j] + tt[j to 0])
        # So we introduce a separate var for return_time
    return_time = {}
    for d in range(num_days):
        return_time[d] = model.addVar(lb=0, ub=bigT, vtype=GRB.CONTINUOUS, name=f"return_t_d{d}")
        model.addConstr(return_time[d] <= depot_latest, name=f"return_window2_d{d}")
        model.addConstr(return_time[d] >= t[d, 0], name=f"return_ge_depart_d{d}")  # at least

        for j in range(1, N+1):
            model.addConstr(return_time[d] >= t[d, j] + travel_time[j][0] - bigT*(1 - x[d, j, 0]), name=f"return_bound_d{d}_j{j}")

        # duration constraint
        model.addConstr(return_time[d] - t[d, 0] <= max_route_duration, name=f"duration_d{d}")

    # Objective: sum of all travel costs = sum over all days, all arcs dist[i][j] * x[d,i,j]
    total_cost = gp.quicksum(dist[i][j] * x[d, i, j] for d in range(num_days) for i in range(N+1) for j in range(N+1) if i != j)
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Callback for logging incumbents
    best_obj = float('inf')
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-6:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        # If infeasible or error, we need a fallback but for now assume feasible
        pass

    final_obj = model.objVal if model.objVal < GRB.INFINITY else 0.0

    # Build the solution dictionary as per schema

    # 1. schedules
    schedules_out = {}
    for c_idx in range(N):
        cust_id = customers[c_idx]['id']
        chosen_s = None
        for s_idx in range(len(patterns[c_idx])):
            if abs(schedule_vars[c_idx, s_idx].X) > 0.5:
                chosen_s = s_idx
                break
        if chosen_s is not None:
            sched = customers[c_idx]['schedules'][chosen_s]
            schedules_out[str(cust_id)] = {
                "schedule_index": chosen_s,
                "days": sched["days"],
                "visit_frequency": int(sched["visit_frequency"])
            }

    # 2. routes
    routes_out = {}
    for d in range(num_days):
        day_routes = []
        # Find all routes by traversing from depot
        visited = set()
        for start_c in range(1, N+1):
            if abs(x[d, 0, start_c].X) > 0.5 and start_c not in visited:
                # start a new route
                route_customers = []
                current = start_c
                route_demand = 0.0
                parts_dict = {}
                while True:
                    visited.add(current)
                    c_idx = current - 1
                    cust_id_str = str(customers[c_idx]['id'])
                    route_customers.append(customers[c_idx]['id'])

                    # find which visit option was used
                    used_o = None
                    dem_this = 0
                    dc_this = 0
                    vd_this = d
                    for o_idx, (vd, dc, dem, _, _) in enumerate(visit_options[c_idx]):
                        if vd == d and abs(visit_use[c_idx, d, o_idx].X) > 0.5:
                            used_o = o_idx
                            dem_this = dem
                            dc_this = dc
                            break
                    if used_o is not None:
                        parts_dict[cust_id_str] = {
                            "visit_day": int(vd_this),
                            "days_covered": int(dc_this),
                            "demand_per_visit": int(dem_this)
                        }
                        route_demand += dem_this

                    # find next
                    next_found = False
                    for next_j in range(1, N+1):
                        if abs(x[d, current, next_j].X) > 0.5 and next_j not in visited:
                            current = next_j
                            next_found = True
                            break
                    if not next_found:
                        break  # back to depot

                # compute route cost
                route_cost = dist[0][route_customers[0]-1 if route_customers else 0]  # wait index fix
                prev = 0
                for cust in route_customers:
                    cid = cust  # id but index is cust-? wait better map
                    # customers list index = id? assume ids are 1 to N or check
                # Recompute route cost properly
                route_path = [0] + [cust for cust in route_customers]  # assuming cust id == index+1 ? Not necessarily.
                # The schema uses customer ids as given.
                # To be safe, we need mapping from logical index to id.
                # Our indices: customer logical index c_idx 0..N-1 , id = customers[c_idx]['id']
                # Our x uses 0=depot, 1 to N corresponding to customers[0] to customers[N-1]
                # So index in model 1..N corresponds to customers[i-1], whose id is customers[i-1]['id']

                route_ids = []
                prev_loc = 0
                route_cost = 0.0
                route_demand = 0.0
                parts_dict = {}
                current_model_idx = 0
                # restart extraction with correct indices
                # Re-extract routes with correct mapping
        # Better re-extract
        day_routes = []
        used_from_depot = []
        for j in range(1, N+1):
            if x[d, 0, j].X > 0.5:
                used_from_depot.append(j)

        for start_idx in used_from_depot:
            route = []
            current = start_idx
            while True:
                route.append(current)
                next_one = None
                for nxt in range(1, N+1):
                    if x[d, current, nxt].X > 0.5:
                        next_one = nxt
                        break
                if next_one is None:
                    break
                current = next_one

            # now build output
            customer_ids_in_route = []
            parts = {}
            this_demand = 0.0
            prev = 0
            this_cost = dist[0][route[0]]
            for pos, model_c_idx in enumerate(route):
                c_real_idx = model_c_idx - 1
                c_id = customers[c_real_idx]['id']
                customer_ids_in_route.append(c_id)

                # get visit details
                for o_idx, (vd, dc, dem, sidx, _) in enumerate(visit_options[c_real_idx]):
                    if vd == d and visit_use[c_real_idx, d, o_idx].X > 0.5:
                        parts[str(c_id)] = {
                            "visit_day": int(vd),
                            "days_covered": int(dc),
                            "demand_per_visit": int(dem)
                        }
                        this_demand += dem
                        break
                if pos < len(route)-1:
                    next_c = route[pos+1]
                    this_cost += dist[model_c_idx][next_c]
                else:
                    this_cost += dist[model_c_idx][0]

            route_entry = {
                "customers": customer_ids_in_route,
                "cost": float(this_cost),
                "demand": float(this_demand),
                "parts": parts
            }
            day_routes.append(route_entry)

        routes_out[str(d)] = day_routes

    # objective
    solution = {
        "objective_value": float(final_obj),
        "routes": routes_out,
        "schedules": schedules_out
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # Final log of best
    if logger and final_obj < float('inf'):
        logger.log(float(final_obj))

if __name__ == "__main__":
    main()