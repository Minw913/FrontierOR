import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def main():
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

    def is_feasible(seq):
        demand = sum(customers[i]['demand'] for i in seq)
        if demand > vehicle_capacity:
            return False
        current_time = depot['time_window'][0]
        prev = 0
        for c in seq:
            arrival = current_time + (0 if prev == 0 else customers[prev]['service_time']) + distance_matrix[prev][c]
            if arrival > customers[c]['time_window'][1]:
                return False
            current_time = max(customers[c]['time_window'][0], arrival)
            prev = c
        
        arrival_end = current_time + customers[prev]['service_time'] + distance_matrix[prev][num_customers+1]
        if arrival_end > depot_end['time_window'][1]:
            return False
        return True

    def get_heuristic_solution():
        def try_solve(order):
            routes = []
            for c in order:
                best_insertion = None
                best_cost = float('inf')
                for r_idx, route in enumerate(routes):
                    for i in range(len(route) + 1):
                        new_route = route[:i] + [c] + route[i:]
                        if is_feasible(new_route):
                            cost = compute_route_dist(new_route) - compute_route_dist(route)
                            if cost < best_cost:
                                best_cost = cost
                                best_insertion = (r_idx, i)
                if best_insertion is not None:
                    r_idx, i = best_insertion
                    routes[r_idx].insert(i, c)
                else:
                    if len(routes) < num_vehicles:
                        if is_feasible([c]):
                            routes.append([c])
                        else:
                            return None
                    else:
                        return None
            return routes

        orders_to_try = [
            sorted(range(1, num_customers + 1), key=lambda x: customers[x]['time_window'][0]),
            sorted(range(1, num_customers + 1), key=lambda x: customers[x]['time_window'][1]),
            sorted(range(1, num_customers + 1), key=lambda x: (customers[x]['time_window'][0], distance_matrix[0][x])),
            list(range(1, num_customers + 1))
        ]
        
        import random
        rng = random.Random(0)
        for _ in range(15):
            base = sorted(range(1, num_customers + 1), key=lambda x: customers[x]['time_window'][0])
            for i in range(len(base)-1):
                if rng.random() < 0.3:
                    base[i], base[i+1] = base[i+1], base[i]
            orders_to_try.append(base)
            
        for order in orders_to_try:
            sol = try_solve(order)
            if sol is not None:
                return sol
        return None

    heuristic_routes = get_heuristic_solution()

    model = gp.Model("VRPTW_Inconvenience")
    model.Params.TimeLimit = args.time_limit
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    C = list(range(1, num_customers + 1))
    V = [0] + C + [num_customers + 1]

    x = {}
    for i in V:
        for j in V:
            if i != j and i != num_customers + 1 and j != 0:
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

    for j in C:
        earliest_arrival = depot['time_window'][0] + distance_matrix[0][j]
        M0 = earliest_arrival - customers[j]['time_window'][0]
        if M0 > 0:
            model.addConstr(t[j] >= customers[j]['time_window'][0] + M0 * x[0, j])

    for i in C:
        for j in C:
            if i != j:
                M_ij = customers[i]['time_window'][1] + customers[i]['service_time'] + distance_matrix[i][j] - customers[j]['time_window'][0]
                if M_ij > 0:
                    model.addConstr(t[j] >= t[i] + customers[i]['service_time'] + distance_matrix[i][j] - M_ij * (1 - x[i, j]))

    for i in C:
        max_t_i = depot_end['time_window'][1] - customers[i]['service_time'] - distance_matrix[i][num_customers+1]
        M_in = customers[i]['time_window'][1] - max_t_i
        if M_in > 0:
            model.addConstr(t[i] <= max_t_i + M_in * (1 - x[i, num_customers + 1]))

    for i in C:
        for j in C:
            if i != j:
                model.addConstr(u[j] >= u[i] + customers[j]['demand'] - vehicle_capacity * (1 - x[i, j]))
                model.addConstr(v[j] >= v[i] + 1 - num_customers * (1 - x[i, j]))

    if heuristic_routes:
        for i, j in x:
            x[i, j].Start = 0
            
        for r in heuristic_routes:
            if not r: continue
            x[0, r[0]].Start = 1
            for i in range(len(r) - 1):
                x[r[i], r[i+1]].Start = 1
            x[r[-1], num_customers + 1].Start = 1
            
            curr_time = depot['time_window'][0]
            curr_u = 0
            curr_v = 0
            prev = 0
            for c in r:
                arrival = curr_time + (0 if prev == 0 else customers[prev]['service_time']) + distance_matrix[prev][c]
                curr_time = max(customers[c]['time_window'][0], arrival)
                t[c].Start = curr_time
                
                curr_u += customers[c]['demand']
                u[c].Start = curr_u
                
                curr_v += 1
                v[c].Start = curr_v
                
                prev = c

    model._x = x
    model._t = t
    model._num_customers = num_customers
    model._customers = customers
    model._logger = logger

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

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            try:
                x_vals = model.cbGetSolution(model._x)
                t_vals = model.cbGetSolution(model._t)
                
                routes = extract_routes(x_vals, model._num_customers)
                schedule = {}
                for i in range(1, model._num_customers + 1):
                    schedule[str(model._customers[i]['id'])] = round(t_vals[i], 2)
                    
                solution_dict = {
                    "objective_value": round(obj, 2),
                    "routes": routes,
                    "schedule": schedule
                }
                if model._logger:
                    model._logger.log_solution(obj, solution_dict)
            except Exception as e:
                if model._logger:
                    model._logger.log(obj)

    model.optimize(cb)

    if model.SolCount > 0:
        x_vals = {k: v.X for k, v in x.items()}
        t_vals = {k: v.X for k, v in t.items()}
        
        final_routes = extract_routes(x_vals, num_customers)
        final_schedule = {}
        for i in range(1, num_customers + 1):
            final_schedule[str(customers[i]['id'])] = round(t_vals[i], 2)
            
        final_solution = {
            "objective_value": model.ObjVal,
            "routes": final_routes,
            "schedule": final_schedule
        }
    else:
        final_routes = []
        final_schedule = {}
        final_obj = 0.0
        if heuristic_routes:
            for r in heuristic_routes:
                route = [0] + r + [num_customers + 1]
                final_routes.append(route)
                
                curr_time = depot['time_window'][0]
                prev = 0
                for c in r:
                    arrival = curr_time + (0 if prev == 0 else customers[prev]['service_time']) + distance_matrix[prev][c]
                    curr_time = max(customers[c]['time_window'][0], arrival)
                    final_schedule[str(customers[c]['id'])] = round(curr_time, 2)
                    prev = c
                    
                dist = compute_route_dist(r)
                final_obj += dist
            
            for c in C:
                t_val = final_schedule[str(customers[c]['id'])]
                pref = customers[c]['preferred_time']
                final_obj += rho * (t_val - pref) ** 2
                
            final_solution = {
                "objective_value": final_obj,
                "routes": final_routes,
                "schedule": final_schedule
            }
        else:
            final_solution = {
                "objective_value": float('inf'),
                "routes": [],
                "schedule": {}
            }

    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)

if __name__ == "__main__":
    main()