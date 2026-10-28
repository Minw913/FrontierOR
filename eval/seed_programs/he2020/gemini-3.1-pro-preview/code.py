import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import random
import time

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_customers = instance['num_customers']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    distance_matrix = instance['distance_matrix']
    depot = instance['depot']
    depot_end = instance['depot_end']
    inconvenience_cost_function = instance['inconvenience_cost_function']

    customers = {}
    for i, c in enumerate(instance['customers']):
        customers[i + 1] = c

    def compute_route_dist(route):
        if not route: return 0
        dist = distance_matrix[0][route[0]]
        for i in range(len(route) - 1):
            dist += distance_matrix[route[i]][route[i+1]]
        dist += distance_matrix[route[-1]][num_customers + 1]
        return dist

    def optimize_route_times(r):
        if not r: return []
        times = []
        curr = depot['time_window'][0]
        prev = 0
        for c in r:
            arr = curr + (0 if prev == 0 else customers[prev]['service_time']) + distance_matrix[prev][c]
            curr = max(customers[c]['time_window'][0], arr)
            times.append(curr)
            prev = c
            
        for i in range(len(r) - 1, -1, -1):
            c = r[i]
            pref = customers[c]['preferred_time']
            if times[i] < pref:
                if i == len(r) - 1:
                    latest = depot_end['time_window'][1] - distance_matrix[c][num_customers+1] - customers[c]['service_time']
                else:
                    next_c = r[i+1]
                    latest = times[i+1] - distance_matrix[c][next_c] - customers[c]['service_time']
                    
                latest = min(latest, customers[c]['time_window'][1])
                if latest > times[i]:
                    times[i] = min(pref, latest)
                    
        return times

    def evaluate_routes(routes):
        obj = 0.0
        rho = inconvenience_cost_function['rho']
        for r in routes:
            obj += compute_route_dist(r)
            opt_times = optimize_route_times(r)
            for idx, c in enumerate(r):
                pref = customers[c]['preferred_time']
                obj += rho * (opt_times[idx] - pref) ** 2
        return obj

    def grasp_heuristic(seed, deterministic=False):
        rng = random.Random(seed)
        unassigned = set(range(1, num_customers + 1))
        routes = []
        
        while unassigned:
            if len(routes) == num_vehicles:
                return None
            
            curr_route = []
            curr_time = depot['time_window'][0]
            curr_node = 0
            curr_demand = 0
            
            while True:
                candidates = []
                
                for c in unassigned:
                    if curr_demand + customers[c]['demand'] > vehicle_capacity:
                        continue
                    
                    arrival = curr_time + (0 if curr_node == 0 else customers[curr_node]['service_time']) + distance_matrix[curr_node][c]
                    if arrival > customers[c]['time_window'][1]:
                        continue
                        
                    new_time = max(customers[c]['time_window'][0], arrival)
                    arrival_end = new_time + customers[c]['service_time'] + distance_matrix[c][num_customers+1]
                    if arrival_end > depot_end['time_window'][1]:
                        continue
                        
                    wait = max(0, customers[c]['time_window'][0] - arrival)
                    
                    if deterministic:
                        w1, w2, w3 = 1.0, 0.2, 0.1
                    else:
                        w1 = rng.uniform(0.5, 1.5)
                        w2 = rng.uniform(0.1, 0.5)
                        w3 = rng.uniform(0.05, 0.2)
                    
                    score = w1 * distance_matrix[curr_node][c] + w2 * wait + w3 * customers[c]['time_window'][1]
                    candidates.append((score, c, new_time))
                    
                if not candidates:
                    break
                    
                candidates.sort(key=lambda x: x[0])
                if deterministic:
                    chosen = candidates[0]
                else:
                    top_k = min(3, len(candidates))
                    chosen = candidates[rng.randint(0, top_k - 1)]
                    
                best_c = chosen[1]
                best_arrival = chosen[2]
                
                curr_route.append(best_c)
                unassigned.remove(best_c)
                curr_time = best_arrival
                curr_node = best_c
                curr_demand += customers[best_c]['demand']
                
            if not curr_route:
                return None
                
            routes.append(curr_route)
            
        return routes

    def get_heuristic_solution():
        best_sol = None
        best_obj = float('inf')
        
        sol = grasp_heuristic(0, deterministic=True)
        if sol is not None:
            obj = evaluate_routes(sol)
            if obj < best_obj:
                best_obj = obj
                best_sol = sol
                
        seed = 1
        while time.time() - start_time < args.time_limit - 2:
            if best_sol is not None and num_customers <= 120 and time.time() - start_time > min(5, args.time_limit * 0.2):
                break
                
            sol = grasp_heuristic(seed, deterministic=False)
            if sol is not None:
                obj = evaluate_routes(sol)
                if obj < best_obj:
                    best_obj = obj
                    best_sol = sol
                    
            seed += 1
            
        return best_sol

    heuristic_routes = get_heuristic_solution()
    
    C = list(range(1, num_customers + 1))
    V = [0] + C + [num_customers + 1]
    
    fallback_solution = None
    if heuristic_routes:
        final_routes = []
        final_schedule = {}
        final_obj = 0.0
        for r in heuristic_routes:
            route = [0] + r + [num_customers + 1]
            final_routes.append(route)
            
            opt_times = optimize_route_times(r)
            for idx, c in enumerate(r):
                final_schedule[str(customers[c]['id'])] = round(opt_times[idx], 2)
                
            dist = compute_route_dist(r)
            final_obj += dist
        
        for c in C:
            t_val = final_schedule[str(customers[c]['id'])]
            pref = customers[c]['preferred_time']
            final_obj += inconvenience_cost_function['rho'] * (t_val - pref) ** 2
            
        fallback_solution = {
            "objective_value": round(final_obj, 2),
            "routes": final_routes,
            "schedule": final_schedule
        }
        if logger:
            logger.log_solution(fallback_solution["objective_value"], fallback_solution)
    else:
        fallback_solution = {
            "objective_value": 1e9,
            "routes": [],
            "schedule": {}
        }

    final_solution = fallback_solution

    def extract_routes(x_vals, n_cust):
        routes = []
        for j in range(1, n_cust + 1):
            if x_vals.get((0, j), 0) > 0.5:
                route = [0]
                curr = j
                visited = set()
                while curr != n_cust + 1:
                    route.append(curr)
                    visited.add(curr)
                    next_node = None
                    for k in range(1, n_cust + 2):
                        if curr != k and x_vals.get((curr, k), 0) > 0.5:
                            next_node = k
                            break
                    if next_node is None or next_node in visited:
                        break
                    curr = next_node
                route.append(n_cust + 1)
                routes.append(route)
        return routes

    if num_customers <= 120:
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0) 
            env.setParam("LogToConsole", 0)
            env.setParam("Seed", 0)
            env.setParam("MIPGap", 1e-4)
            env.setParam("NumericFocus", 0)
            env.setParam("Threads", 1)
            env.start()
            
            model = gp.Model("VRPTW_Inconvenience", env=env)
            remaining_time = args.time_limit - (time.time() - start_time) - 2
            if remaining_time < 1:
                remaining_time = 1
            model.Params.TimeLimit = remaining_time

            x = {}
            valid_edges = set()
            
            for i in V:
                for j in V:
                    if i != j and i != num_customers + 1 and j != 0:
                        possible = True
                        if i in C and j in C:
                            ea = customers[i]['time_window'][0] + customers[i]['service_time'] + distance_matrix[i][j]
                            if ea > customers[j]['time_window'][1]:
                                possible = False
                            if customers[i]['demand'] + customers[j]['demand'] > vehicle_capacity:
                                possible = False
                        elif i == 0 and j in C:
                            ea = depot['time_window'][0] + distance_matrix[0][j]
                            if ea > customers[j]['time_window'][1]:
                                possible = False
                        elif i in C and j == num_customers + 1:
                            ea = customers[i]['time_window'][0] + customers[i]['service_time'] + distance_matrix[i][num_customers + 1]
                            if ea > depot_end['time_window'][1]:
                                possible = False
                                
                        if possible:
                            valid_edges.add((i, j))

            if heuristic_routes:
                for r in heuristic_routes:
                    if not r: continue
                    valid_edges.add((0, r[0]))
                    for idx in range(len(r) - 1):
                        valid_edges.add((r[idx], r[idx+1]))
                    valid_edges.add((r[-1], num_customers + 1))

            for (i, j) in valid_edges:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

            t = {}
            for i in C:
                t[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=customers[i]['time_window'][0], ub=customers[i]['time_window'][1], name=f"t_{i}")

            u = {}
            for i in C:
                u[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=customers[i]['demand'], ub=vehicle_capacity, name=f"u_{i}")

            v = {}
            for i in C:
                v[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=num_customers, name=f"v_{i}")

            rho = inconvenience_cost_function['rho']
            obj_routing = gp.quicksum(distance_matrix[i][j] * x[i, j] for i, j in x)
            obj_inconv = gp.quicksum(rho * (t[i] - customers[i]['preferred_time']) * (t[i] - customers[i]['preferred_time']) for i in C)
            model.setObjective(obj_routing + obj_inconv, GRB.MINIMIZE)

            for i in C:
                model.addConstr(gp.quicksum(x[i, j] for j in V if (i, j) in x) == 1)
                model.addConstr(gp.quicksum(x[j, i] for j in V if (j, i) in x) == 1)

            model.addConstr(gp.quicksum(x[0, j] for j in C if (0, j) in x) <= num_vehicles)
            model.addConstr(gp.quicksum(x[i, num_customers + 1] for i in C if (i, num_customers + 1) in x) == gp.quicksum(x[0, j] for j in C if (0, j) in x))

            for (i, j) in valid_edges:
                if i in C and j in C:
                    M_ij = customers[i]['time_window'][1] + customers[i]['service_time'] + distance_matrix[i][j] - customers[j]['time_window'][0]
                    if M_ij > 0:
                        model.addConstr(t[j] >= t[i] + customers[i]['service_time'] + distance_matrix[i][j] - M_ij * (1 - x[i, j]))
                elif i == 0 and j in C:
                    earliest_arrival = depot['time_window'][0] + distance_matrix[0][j]
                    M0 = earliest_arrival - customers[j]['time_window'][0]
                    if M0 > 0:
                        model.addConstr(t[j] >= customers[j]['time_window'][0] + M0 * x[0, j])

            for i in C:
                if (i, num_customers + 1) in x:
                    max_t_i = depot_end['time_window'][1] - customers[i]['service_time'] - distance_matrix[i][num_customers+1]
                    M_in = customers[i]['time_window'][1] - max_t_i
                    if M_in > 0:
                        model.addConstr(t[i] <= max_t_i + M_in * (1 - x[i, num_customers + 1]))

            for (i, j) in valid_edges:
                if i in C and j in C:
                    model.addConstr(u[j] >= u[i] + customers[j]['demand'] - vehicle_capacity * (1 - x[i, j]))
                    model.addConstr(v[j] >= v[i] + 1 - num_customers * (1 - x[i, j]))

            if heuristic_routes:
                for (i, j) in x:
                    x[i, j].Start = 0
                    
                for r in heuristic_routes:
                    if not r: continue
                    if (0, r[0]) in x:
                        x[0, r[0]].Start = 1
                    for i in range(len(r) - 1):
                        if (r[i], r[i+1]) in x:
                            x[r[i], r[i+1]].Start = 1
                    if (r[-1], num_customers + 1) in x:
                        x[r[-1], num_customers + 1].Start = 1
                    
                    opt_times = optimize_route_times(r)
                    curr_u = 0
                    curr_v = 0
                    for idx, c in enumerate(r):
                        t[c].Start = opt_times[idx]
                        curr_u += customers[c]['demand']
                        u[c].Start = curr_u
                        curr_v += 1
                        v[c].Start = curr_v

            model._x_keys = list(x.keys())
            model._x_vars = list(x.values())
            model._t_keys = list(t.keys())
            model._t_vars = list(t.values())
            model._num_customers = num_customers
            model._customers = customers
            model._logger = logger
            model._distance_matrix = distance_matrix
            model._rho = rho

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    try:
                        x_vals_list = model.cbGetSolution(model._x_vars)
                        x_vals = dict(zip(model._x_keys, x_vals_list))
                        
                        t_vals_list = model.cbGetSolution(model._t_vars)
                        t_vals = dict(zip(model._t_keys, t_vals_list))
                        
                        routes = extract_routes(x_vals, model._num_customers)
                        visited_customers = sum(len(r) - 2 for r in routes)
                        
                        if visited_customers == model._num_customers:
                            schedule = {}
                            for i in range(1, model._num_customers + 1):
                                schedule[str(model._customers[i]['id'])] = round(t_vals[i], 2)
                                
                            final_obj_cb = 0.0
                            for r in routes:
                                dist = 0
                                for i in range(len(r) - 1):
                                    dist += model._distance_matrix[r[i]][r[i+1]]
                                final_obj_cb += dist
                                
                            for i in range(1, model._num_customers + 1):
                                t_val = schedule[str(model._customers[i]['id'])]
                                pref = model._customers[i]['preferred_time']
                                final_obj_cb += model._rho * (t_val - pref) ** 2
                                
                            solution_dict = {
                                "objective_value": round(final_obj_cb, 2),
                                "routes": routes,
                                "schedule": schedule
                            }
                            if model._logger:
                                model._logger.log_solution(final_obj_cb, solution_dict)
                        else:
                            if model._logger:
                                model._logger.log(obj)
                    except Exception:
                        if model._logger:
                            model._logger.log(obj)

            model.optimize(cb)

            if model.SolCount > 0:
                x_vals = {k: v.X for k, v in x.items()}
                t_vals = {k: v.X for k, v in t.items()}
                
                extracted_routes = extract_routes(x_vals, num_customers)
                visited_customers = sum(len(r) - 2 for r in extracted_routes)
                
                if visited_customers == num_customers:
                    extracted_schedule = {}
                    for i in range(1, num_customers + 1):
                        extracted_schedule[str(customers[i]['id'])] = round(t_vals[i], 2)
                        
                    obj_val = 0.0
                    for r in extracted_routes:
                        dist = 0
                        for i in range(len(r) - 1):
                            dist += distance_matrix[r[i]][r[i+1]]
                        obj_val += dist
                        
                    for i in range(1, num_customers + 1):
                        t_val = extracted_schedule[str(customers[i]['id'])]
                        pref = customers[i]['preferred_time']
                        obj_val += rho * (t_val - pref) ** 2
                        
                    final_solution = {
                        "objective_value": round(obj_val, 2),
                        "routes": extracted_routes,
                        "schedule": extracted_schedule
                    }

        except Exception:
            pass

    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)

if __name__ == "__main__":
    main()