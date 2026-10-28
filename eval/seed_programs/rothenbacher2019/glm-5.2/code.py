#!/usr/bin/env python3
import argparse
import json
import math
import time
import random

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

class Instance:
    def __init__(self, data):
        self.num_customers = data['num_customers']
        self.num_days = data['num_days']
        self.num_vehicles = data['num_vehicles']
        self.vehicle_capacity = data['vehicle_capacity']
        self.max_route_duration = data['max_route_duration']
        
        self.depot = data['depot']
        self.depot_loc = (self.depot['x'], self.depot['y'])
        self.depot_tw = list(self.depot['time_window'])
        self.depot_st = self.depot['service_time']
        
        self.customers = data['customers']
        
        self.locs = [self.depot_loc] + [(c['x'], c['y']) for c in self.customers]
        self.n = len(self.locs)
        
        self.dist = [[0.0]*self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                self.dist[i][j] = euclidean(self.locs[i], self.locs[j])
        
        self.service_times = [self.depot_st] + [c['service_time'] for c in self.customers]
        self.time_windows = [list(self.depot_tw)] + [list(c['time_window']) for c in self.customers]
        
        self.cust_id_to_idx = {}
        for i, c in enumerate(self.customers):
            self.cust_id_to_idx[c['id']] = i + 1
        self.cust_idx_to_id = {v: k for k, v in self.cust_id_to_idx.items()}


def check_route(route, inst, demands):
    """Check if route is feasible. route is list of 1-indexed customer indices."""
    if not route:
        return True, 0.0, 0
    total_demand = sum(demands.get(idx, 0) for idx in route)
    if total_demand > inst.vehicle_capacity:
        return False, 0.0, 0
    cost = 0.0
    current_time = float(inst.depot_tw[0])
    prev = 0
    for idx in route:
        tt = inst.dist[prev][idx] + inst.service_times[prev]
        arrival = current_time + tt
        tw = inst.time_windows[idx]
        service_start = max(arrival, tw[0])
        if service_start > tw[1]:
            return False, 0.0, 0
        current_time = service_start + inst.service_times[idx]
        cost += inst.dist[prev][idx]
        prev = idx
    tt = inst.dist[prev][0] + inst.service_times[prev]
    arrival = current_time + tt
    cost += inst.dist[prev][0]
    if arrival > inst.depot_tw[1]:
        return False, 0.0, 0
    route_duration = arrival - inst.depot_tw[0]
    if route_duration > inst.max_route_duration:
        return False, 0.0, 0
    return True, cost, total_demand


def compute_route_cost(route, inst):
    if not route:
        return 0.0
    cost = inst.dist[0][route[0]]
    for i in range(len(route)-1):
        cost += inst.dist[route[i]][route[i+1]]
    cost += inst.dist[route[-1]][0]
    return cost


def insertion_heuristic(day_tasks, inst, deadline, order_key='deadline', seed=0):
    """Greedy insertion heuristic for VRPTW."""
    if not day_tasks:
        return []
    
    n = len(day_tasks)
    demands = {t['customer_idx']: t['demand'] for t in day_tasks}
    
    if order_key == 'deadline':
        order = sorted(range(n), key=lambda i: inst.time_windows[day_tasks[i]['customer_idx']][1])
    elif order_key == 'distance':
        order = sorted(range(n), key=lambda i: inst.dist[0][day_tasks[i]['customer_idx']])
    elif order_key == 'angle':
        def angle(i):
            idx = day_tasks[i]['customer_idx']
            dx = inst.locs[idx][0] - inst.depot_loc[0]
            dy = inst.locs[idx][1] - inst.depot_loc[1]
            return math.atan2(dy, dx)
        order = sorted(range(n), key=angle)
    elif order_key == 'random':
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
    else:
        order = list(range(n))
    
    routes = []
    unassigned = set(order)
    
    while unassigned:
        if time.time() > deadline:
            return None
        route = []
        changed = True
        while changed and unassigned:
            changed = False
            best_task = None
            best_pos = -1
            best_ins_cost = float('inf')
            
            for ti in list(unassigned):
                if time.time() > deadline:
                    return None
                cust_idx = day_tasks[ti]['customer_idx']
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [cust_idx] + route[pos:]
                    feas, cost, dem = check_route(new_route, inst, demands)
                    if feas:
                        if route:
                            old_cost = compute_route_cost(route, inst)
                            ins_cost = cost - old_cost
                        else:
                            ins_cost = cost
                        if ins_cost < best_ins_cost:
                            best_ins_cost = ins_cost
                            best_task = ti
                            best_pos = pos
            
            if best_task is not None:
                cust_idx = day_tasks[best_task]['customer_idx']
                route = route[:best_pos] + [cust_idx] + route[best_pos:]
                unassigned.remove(best_task)
                changed = True
        
        if route:
            routes.append(route)
        else:
            return None
        if len(routes) > inst.num_vehicles:
            return None
    
    return routes


def local_search(routes, inst, demands, deadline):
    """Improve routes using 2-opt, 2-opt*, or-opt, and inter-route swap."""
    if not routes:
        return routes
    
    improved = True
    max_iter = 100
    it = 0
    
    while improved and it < max_iter:
        it += 1
        if time.time() > deadline:
            break
        improved = False
        
        # 2-opt within each route
        for ri in range(len(routes)):
            if time.time() > deadline:
                break
            route = routes[ri]
            if len(route) < 3:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    if time.time() > deadline:
                        break
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    old_cost = compute_route_cost(route, inst)
                    new_cost = compute_route_cost(new_route, inst)
                    if new_cost < old_cost - 1e-6:
                        feas, _, _ = check_route(new_route, inst, demands)
                        if feas:
                            routes[ri] = new_route
                            route = new_route
                            improved = True
        
        if improved:
            continue
        
        # Or-opt: move customer from one route to another
        for ri in range(len(routes)):
            if improved or time.time() > deadline:
                break
            for pos in range(len(routes[ri])):
                if time.time() > deadline:
                    break
                cust = routes[ri][pos]
                new_ri = routes[ri][:pos] + routes[ri][pos+1:]
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    if time.time() > deadline:
                        break
                    for new_pos in range(len(routes[rj]) + 1):
                        new_rj = routes[rj][:new_pos] + [cust] + routes[rj][new_pos:]
                        feas_i = True
                        feas_j, _, _ = check_route(new_rj, inst, demands)
                        if new_ri:
                            feas_i, _, _ = check_route(new_ri, inst, demands)
                        if feas_i and feas_j:
                            old_cost = compute_route_cost(routes[ri], inst) + compute_route_cost(routes[rj], inst)
                            new_cost = (compute_route_cost(new_ri, inst) if new_ri else 0) + compute_route_cost(new_rj, inst)
                            if new_cost < old_cost - 1e-6:
                                routes[ri] = new_ri
                                routes[rj] = new_rj
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Inter-route swap
        for ri in range(len(routes)):
            if improved or time.time() > deadline:
                break
            for pi in range(len(routes[ri])):
                if time.time() > deadline:
                    break
                for rj in range(ri+1, len(routes)):
                    if time.time() > deadline:
                        break
                    for pj in range(len(routes[rj])):
                        if time.time() > deadline:
                            break
                        new_ri = list(routes[ri])
                        new_rj = list(routes[rj])
                        new_ri[pi], new_rj[pj] = new_rj[pj], new_ri[pi]
                        feas_i, _, _ = check_route(new_ri, inst, demands)
                        feas_j, _, _ = check_route(new_rj, inst, demands)
                        if feas_i and feas_j:
                            old_cost = compute_route_cost(routes[ri], inst) + compute_route_cost(routes[rj], inst)
                            new_cost = compute_route_cost(new_ri, inst) + compute_route_cost(new_rj, inst)
                            if new_cost < old_cost - 1e-6:
                                routes[ri] = new_ri
                                routes[rj] = new_rj
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        if improved:
            continue
        
        # 2-opt*: cross-route 2-opt
        for ri in range(len(routes)):
            if improved or time.time() > deadline:
                break
            for rj in range(ri+1, len(routes)):
                if time.time() > deadline:
                    break
                for i in range(-1, len(routes[ri])):
                    if time.time() > deadline:
                        break
                    for j in range(-1, len(routes[rj])):
                        if time.time() > deadline:
                            break
                        new_ri = routes[ri][:i+1] + routes[rj][j+1:]
                        new_rj = routes[rj][:j+1] + routes[ri][i+1:]
                        if new_ri == routes[ri] and new_rj == routes[rj]:
                            continue
                        if not new_ri and not new_rj:
                            continue
                        feas_i = True
                        feas_j = True
                        if new_ri:
                            feas_i, _, _ = check_route(new_ri, inst, demands)
                        if new_rj:
                            feas_j, _, _ = check_route(new_rj, inst, demands)
                        if feas_i and feas_j:
                            old_cost = compute_route_cost(routes[ri], inst) + compute_route_cost(routes[rj], inst)
                            new_cost = 0.0
                            if new_ri:
                                new_cost += compute_route_cost(new_ri, inst)
                            if new_rj:
                                new_cost += compute_route_cost(new_rj, inst)
                            if new_cost < old_cost - 1e-6:
                                routes[ri] = new_ri
                                routes[rj] = new_rj
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    
    routes = [r for r in routes if r]
    return routes


def solve_vrp_mip(day_tasks, inst, deadline):
    """Solve VRPTW for a single day using Gurobi MIP."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    
    n = len(day_tasks)
    if n == 0:
        return []
    if n > 12:
        return None
    
    remaining = max(1.0, deadline - time.time())
    
    demands = {t['customer_idx']: t['demand'] for t in day_tasks}
    
    node_to_cust = {}
    for i, t in enumerate(day_tasks):
        node_to_cust[i+1] = t['customer_idx']
    
    N = n + 2  # 0=depot_start, 1..n=customers, n+1=depot_end
    local_dist = [[0.0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            ci = 0 if (i == 0 or i == n+1) else node_to_cust[i]
            cj = 0 if (j == 0 or j == n+1) else node_to_cust[j]
            local_dist[i][j] = inst.dist[ci][cj]
    
    local_st = [0] * N
    local_st[0] = inst.depot_st
    for i in range(1, n+1):
        local_st[i] = inst.service_times[node_to_cust[i]]
    
    local_tw = [list(inst.depot_tw)] + [list(inst.time_windows[node_to_cust[i]]) for i in range(1, n+1)] + [list(inst.depot_tw)]
    local_dem = [0] + [demands[node_to_cust[i]] for i in range(1, n+1)] + [0]
    
    try:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        
        model = gp.Model(env=env)
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = remaining
        
        x = {}
        for i in range(N):
            for j in range(N):
                if i != j and i != n+1 and j != 0:
                    if i == 0 and j == n+1:
                        continue
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
        
        t = {}
        for i in range(N):
            lb = local_tw[i][0]
            ub = local_tw[i][1]
            t[i] = model.addVar(lb=lb, ub=ub, name=f't_{i}')
        
        load = {}
        for i in range(N):
            load[i] = model.addVar(lb=0, ub=inst.vehicle_capacity, name=f'load_{i}')
        
        model.update()
        
        model.setObjective(gp.quicksum(local_dist[i][j] * x[i, j] for (i, j) in x), GRB.MINIMIZE)
        
        for j in range(1, n+1):
            model.addConstr(gp.quicksum(x[i, j] for i in range(N) if (i, j) in x) == 1)
            model.addConstr(gp.quicksum(x[j, k] for k in range(N) if (j, k) in x) == 1)
        
        model.addConstr(gp.quicksum(x[0, j] for j in range(1, n+1) if (0, j) in x) <= inst.num_vehicles)
        model.addConstr(
            gp.quicksum(x[0, j] for j in range(1, n+1) if (0, j) in x) ==
            gp.quicksum(x[i, n+1] for i in range(1, n+1) if (i, n+1) in x)
        )
        
        M_time = 100000.0
        for (i, j) in x:
            travel = local_dist[i][j] + local_st[i]
            model.addConstr(t[j] >= t[i] + travel - M_time * (1 - x[i, j]))
            model.addConstr(load[j] >= load[i] + local_dem[j] - inst.vehicle_capacity * (1 - x[i, j]))
        
        model.addConstr(load[0] == 0)
        model.addConstr(t[0] >= inst.depot_tw[0])
        model.addConstr(t[n+1] <= inst.depot_tw[1])
        model.addConstr(t[n+1] - t[0] <= inst.max_route_duration)
        
        model.optimize()
        
        if model.SolCount == 0:
            return None
        
        routes = []
        visited = set()
        
        for start_j in range(1, n+1):
            if start_j in visited:
                continue
            route = []
            # Find the start of this route (go backward to depot)
            current = start_j
            while True:
                pred = None
                for i in range(N):
                    if (i, current) in x and x[i, current].X > 0.5:
                        pred = i
                        break
                if pred is None or pred == 0:
                    break
                current = pred
            
            # Now traverse forward from current
            if current == 0:
                # current is depot_start, find first customer
                for j in range(1, n+1):
                    if (0, j) in x and x[0, j].X > 0.5 and j not in visited:
                        current = j
                        break
                else:
                    continue
            
            while current != n+1 and current != 0:
                route.append(node_to_cust[current])
                visited.add(current)
                next_node = None
                for k in range(N):
                    if (current, k) in x and x[current, k].X > 0.5:
                        next_node = k
                        break
                if next_node is None:
                    break
                current = next_node
            
            if route:
                routes.append(route)
        
        return routes if routes else None
        
    except Exception:
        return None


def select_schedules(inst, strategy, seed=0):
    selected = {}
    for c in inst.customers:
        cid = c['id']
        schedules = c['schedules']
        if strategy == 'min_visits':
            best_idx = 0
            min_visits = float('inf')
            for si, sch in enumerate(schedules):
                if sch['visit_frequency'] < min_visits:
                    min_visits = sch['visit_frequency']
                    best_idx = si
            selected[cid] = best_idx
        elif strategy == 'first':
            selected[cid] = 0
        elif strategy == 'last':
            selected[cid] = len(schedules) - 1
        elif strategy == 'random':
            rng = random.Random(seed)
            selected[cid] = rng.randint(0, len(schedules)-1)
        elif strategy == 'min_demand_per_visit':
            best_idx = 0
            min_max_demand = float('inf')
            for si, sch in enumerate(schedules):
                max_demand = max(p['demand_per_visit'] for p in sch['parts'])
                if max_demand < min_max_demand:
                    min_max_demand = max_demand
                    best_idx = si
            selected[cid] = best_idx
        else:
            selected[cid] = 0
    return selected


def select_schedules_balanced(inst, seed=0):
    """Select schedules to balance visits per day."""
    rng = random.Random(seed)
    day_loads = [0] * inst.num_days
    selected = {}
    
    customer_list = list(inst.customers)
    rng.shuffle(customer_list)
    
    for c in customer_list:
        cid = c['id']
        schedules = c['schedules']
        
        best_idx = 0
        best_score = float('inf')
        
        for si, sch in enumerate(schedules):
            new_loads = list(day_loads)
            for part in sch['parts']:
                new_loads[part['visit_day']] += 1
            score = sum(l * l for l in new_loads)
            if score < best_score:
                best_score = score
                best_idx = si
        
        selected[cid] = best_idx
        for part in schedules[best_idx]['parts']:
            day_loads[part['visit_day']] += 1
    
    return selected


def build_daily_tasks(schedule_selection, inst):
    daily_tasks = {d: [] for d in range(inst.num_days)}
    for c in inst.customers:
        cid = c['id']
        si = schedule_selection[cid]
        sch = c['schedules'][si]
        for part in sch['parts']:
            day = part['visit_day']
            daily_tasks[day].append({
                'customer_idx': inst.cust_id_to_idx[cid],
                'customer_id': cid,
                'demand': part['demand_per_visit'],
                'days_covered': part['days_covered'],
                'visit_day': day
            })
    return daily_tasks


def solve_vrp_for_day(day_tasks, inst, deadline):
    if not day_tasks:
        return []
    
    n = len(day_tasks)
    demands = {t['customer_idx']: t['demand'] for t in day_tasks}
    
    # Try MIP for small instances
    if n <= 10:
        mip_routes = solve_vrp_mip(day_tasks, inst, deadline)
        if mip_routes is not None:
            return mip_routes
    
    # Heuristic approach with multiple orderings
    best_routes = None
    best_cost = float('inf')
    
    orderings = [
        ('deadline', 0),
        ('distance', 0),
        ('angle', 0),
        ('random', 1),
        ('random', 2),
        ('random', 3),
    ]
    
    for order_key, seed in orderings:
        if time.time() > deadline:
            break
        
        routes = insertion_heuristic(day_tasks, inst, deadline, order_key, seed)
        if routes is None:
            continue
        
        routes = local_search(routes, inst, demands, deadline)
        
        total_cost = sum(compute_route_cost(r, inst) for r in routes)
        if total_cost < best_cost:
            best_cost = total_cost
            best_routes = [list(r) for r in routes]
    
    return best_routes


def build_solution_dict(routes_by_day, schedule_selection, inst):
    routes_output = {}
    
    daily_tasks = build_daily_tasks(schedule_selection, inst)
    
    for day in range(inst.num_days):
        day_key = str(day)
        routes_output[day_key] = []
        
        if day not in routes_by_day or not routes_by_day[day]:
            continue
        
        day_tasks = daily_tasks[day]
        task_map = {}
        for t in day_tasks:
            task_map[t['customer_idx']] = t
        
        for route in routes_by_day[day]:
            route_customers = [inst.cust_idx_to_id[idx] for idx in route]
            cost = compute_route_cost(route, inst)
            demand = sum(task_map[idx]['demand'] for idx in route)
            
            parts = {}
            for idx in route:
                cid = inst.cust_idx_to_id[idx]
                t = task_map[idx]
                parts[str(cid)] = {
                    'visit_day': t['visit_day'],
                    'days_covered': t['days_covered'],
                    'demand_per_visit': t['demand']
                }
            
            routes_output[day_key].append({
                'customers': route_customers,
                'cost': cost,
                'demand': demand,
                'parts': parts
            })
    
    schedules_output = {}
    for c in inst.customers:
        cid = c['id']
        si = schedule_selection[cid]
        sch = c['schedules'][si]
        schedules_output[str(cid)] = {
            'schedule_index': si,
            'days': sch['days'],
            'visit_frequency': sch['visit_frequency']
        }
    
    total_cost = 0.0
    for day_key in routes_output:
        for r in routes_output[day_key]:
            total_cost += r['cost']
    
    return {
        'objective_value': total_cost,
        'routes': routes_output,
        'schedules': schedules_output
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    deadline = start_time + args.time_limit
    
    with open(args.instance_path) as f:
        data = json.load(f)
    
    inst = Instance(data)
    
    best_solution = None
    best_obj = float('inf')
    
    # Build list of strategies to try
    strategies = [
        ('min_visits', 0),
        ('first', 0),
        ('min_demand_per_visit', 0),
        ('balanced', 0),
        ('balanced', 1),
        ('balanced', 2),
        ('random', 1),
        ('random', 2),
        ('random', 3),
        ('random', 4),
        ('random', 5),
        ('random', 6),
        ('random', 7),
        ('random', 8),
        ('random', 9),
        ('random', 10),
    ]
    
    for strategy, seed in strategies:
        if time.time() > deadline:
            break
        
        if strategy == 'balanced':
            schedule_selection = select_schedules_balanced(inst, seed)
        else:
            schedule_selection = select_schedules(inst, strategy, seed)
        
        daily_tasks = build_daily_tasks(schedule_selection, inst)
        
        routes_by_day = {}
        total_cost = 0.0
        feasible = True
        
        for day in range(inst.num_days):
            if time.time() > deadline:
                feasible = False
                break
            
            day_tasks = daily_tasks[day]
            # Give each day a sub-deadline
            remaining = deadline - time.time()
            num_remaining_days = inst.num_days - day
            day_deadline = time.time() + min(remaining, remaining / max(num_remaining_days, 1) * 1.5)
            
            routes = solve_vrp_for_day(day_tasks, inst, day_deadline)
            
            if routes is None:
                feasible = False
                break
            
            routes_by_day[day] = routes
            for r in routes:
                total_cost += compute_route_cost(r, inst)
        
        if feasible and total_cost < best_obj:
            best_obj = total_cost
            sol_dict = build_solution_dict(routes_by_day, schedule_selection, inst)
            best_solution = sol_dict
            if logger:
                logger.log_solution(best_obj, sol_dict)
    
    if best_solution is None:
        # Fallback: try harder with first schedule
        for strategy, seed in [('first', 0), ('min_visits', 0), ('balanced', 0)]:
            if time.time() > deadline:
                break
            if strategy == 'balanced':
                schedule_selection = select_schedules_balanced(inst, seed)
            else:
                schedule_selection = select_schedules(inst, strategy, seed)
            daily_tasks = build_daily_tasks(schedule_selection, inst)
            routes_by_day = {}
            feasible = True
            for day in range(inst.num_days):
                if time.time() > deadline:
                    feasible = False
                    break
                day_tasks = daily_tasks[day]
                routes = solve_vrp_for_day(day_tasks, inst, deadline)
                if routes is None:
                    feasible = False
                    break
                routes_by_day[day] = routes
            if feasible:
                best_solution = build_solution_dict(routes_by_day, schedule_selection, inst)
                if logger:
                    logger.log_solution(best_solution['objective_value'], best_solution)
                break
    
    if best_solution is None:
        # Last resort: empty solution
        schedule_selection = select_schedules(inst, 'first', 0)
        best_solution = build_solution_dict({}, schedule_selection, inst)
    
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


if __name__ == '__main__':
    main()