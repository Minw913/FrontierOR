import json
import argparse
import sys
import time
import random
import gurobipy as gp
from gurobipy import GRB

def check_route(route, capacity, demand, ready_time, due_date, eff_time):
    t = float(ready_time[0])
    load = 0
    for k in range(len(route) - 1):
        u = route[k]
        v = route[k + 1]
        arr = t + eff_time[u][v]
        if arr > due_date[v]:
            return False
        t = max(arr, float(ready_time[v]))
        if v != 0:
            load += demand[v]
            if load > capacity:
                return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", required=False)
    args = parser.parse_args()

    start_time = time.time()

    # Load instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data['num_customers']
    capacity = data['vehicle_capacity']
    depot_data = data['depot']
    customers_data = data['customers']
    dist_matrix = data['distance_matrix']
    travel_time_includes_service_time = data.get('travel_time_includes_service_time', False)

    ready_time = [0] * (N + 1)
    due_date = [0] * (N + 1)
    demand = [0] * (N + 1)
    service_time = [0] * (N + 1)

    ready_time[0] = depot_data['ready_time']
    due_date[0] = depot_data['due_date']
    demand[0] = depot_data['demand']
    service_time[0] = depot_data['service_time']

    for c in customers_data:
        idx = c['id']
        ready_time[idx] = c['ready_time']
        due_date[idx] = c['due_date']
        demand[idx] = c['demand']
        service_time[idx] = c['service_time']

    # Effective travel time calculation
    eff_time = [[0.0] * (N + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        for j in range(N + 1):
            if i == j:
                continue
            base_t = float(dist_matrix[i][j])
            if not travel_time_includes_service_time:
                base_t += float(service_time[i])
            diff = float(ready_time[j] - due_date[i])
            eff_time[i][j] = max(base_t, diff)

    # Validate and filter possible arcs based on capacity and time rules
    valid_arcs = []
    for i in range(N + 1):
        for j in range(N + 1):
            if i == j:
                continue
            if i != 0 and j != 0 and demand[i] + demand[j] > capacity:
                continue
            if ready_time[i] + eff_time[i][j] > due_date[j]:
                continue
            valid_arcs.append((i, j))

    # Initialize Logger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Fast insertion heuristic to establish a strong initial upper-bound solution
    best_routes = []
    best_vehicles = float('inf')
    best_dist = float('inf')

    orderings = []
    orderings.append(sorted(range(1, N + 1), key=lambda x: due_date[x]))
    orderings.append(sorted(range(1, N + 1), key=lambda x: dist_matrix[0][x], reverse=True))
    orderings.append(sorted(range(1, N + 1), key=lambda x: demand[x], reverse=True))
    rng = random.Random(42)
    for _ in range(8):
        l = list(range(1, N + 1))
        rng.shuffle(l)
        orderings.append(l)

    for order in orderings:
        unrouted = set(range(1, N + 1))
        routes = []
        possible = True

        while unrouted:
            route = [0, 0]
            while True:
                best_cost = float('inf')
                best_c = None
                best_ins = None

                for c in order:
                    if c not in unrouted:
                        continue
                    for pos in range(1, len(route)):
                        new_r = route[:pos] + [c] + route[pos:]
                        if check_route(new_r, capacity, demand, ready_time, due_date, eff_time):
                            u = route[pos - 1]
                            v = route[pos]
                            dist_inc = dist_matrix[u][c] + dist_matrix[c][v] - dist_matrix[u][v]
                            cost = dist_inc + 0.05 * due_date[c]
                            if cost < best_cost:
                                best_cost = cost
                                best_c = c
                                best_ins = pos

                if best_c is None:
                    break
                route.insert(best_ins, best_c)
                unrouted.remove(best_c)

            if len(route) == 2:
                possible = False
                break
            routes.append(route)

        if not possible:
            continue

        vehicles = len(routes)
        dist = sum(dist_matrix[routes[i][k]][routes[i][k + 1]] for i in range(len(routes)) for k in range(len(routes[i]) - 1))

        if vehicles < best_vehicles or (vehicles == best_vehicles and dist < best_dist):
            best_routes = routes
            best_vehicles = vehicles
            best_dist = dist
            if logger:
                logger.log(float(best_vehicles))

    # Gurobi optimization setup
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model("VRPTW", env=env)

    rem_time = max(1.0, args.time_limit - (time.time() - start_time) - 1.0)
    model.Params.TimeLimit = rem_time
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-4
    model.Params.MIPFocus = 1

    x = model.addVars(valid_arcs, vtype=GRB.BINARY, name="x")
    t = model.addVars(N + 1, vtype=GRB.CONTINUOUS, name="t")
    l = model.addVars(N + 1, vtype=GRB.CONTINUOUS, name="l")

    for i in range(N + 1):
        t[i].lb = float(ready_time[i])
        t[i].ub = float(due_date[i])

    l[0].lb = 0.0
    l[0].ub = float(capacity)
    for i in range(1, N + 1):
        l[i].lb = float(demand[i])
        l[i].ub = float(capacity)

    # Degree constraints for exact node visitation
    for j in range(1, N + 1):
        model.addConstr(gp.quicksum(x[i, j] for i, _j in valid_arcs if _j == j) == 1)
        model.addConstr(gp.quicksum(x[j, k] for _j, k in valid_arcs if _j == j) == 1)

    # Sub-tour elimination via MTZ tracking for time and accumulated capacity 
    for i, j in valid_arcs:
        if j != 0:
            M_ij_T = due_date[i] + eff_time[i][j] - ready_time[j]
            if M_ij_T > 0:
                model.addConstr(t[j] >= t[i] + eff_time[i][j] - M_ij_T * (1 - x[i, j]))
            else:
                model.addConstr(t[j] >= t[i] + eff_time[i][j])

            if i != 0:
                model.addConstr(l[j] >= l[i] + demand[j] - capacity * (1 - x[i, j]))

        if j == 0:
            M_i0 = due_date[i] + eff_time[i][0] - due_date[0]
            if M_i0 > 0:
                model.addConstr(t[i] + eff_time[i][0] <= due_date[0] + M_i0 * (1 - x[i, 0]))
            else:
                model.addConstr(t[i] + eff_time[i][0] <= due_date[0])

    # Construct primary & secondary scalarized objective
    num_vehicles_expr = gp.quicksum(x[0, j] for _i, j in valid_arcs if _i == 0)
    total_distance_expr = gp.quicksum(dist_matrix[i][j] * x[i, j] for i, j in valid_arcs)
    max_d = max(dist_matrix[i][j] for i in range(N + 1) for j in range(N + 1))
    veh_weight = max(1000000.0, float(N * max_d + 1000.0))
    model.setObjective(num_vehicles_expr * veh_weight + total_distance_expr, GRB.MINIMIZE)

    # Inject evaluated heuristic as MIP Start upper-bound
    if best_routes:
        used_arcs = set()
        for r in best_routes:
            for k in range(len(r) - 1):
                used_arcs.add((r[k], r[k + 1]))

        for (i, j), var in x.items():
            if (i, j) in used_arcs:
                var.Start = 1
            else:
                var.Start = 0

    # Callback routine matching solver improvements with Solution Logging rules 
    def cb(mdl, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                v_count = int(obj / veh_weight)
                if logger:
                    logger.log(float(v_count))
            except Exception:
                pass

    model.optimize(cb)

    # Route extraction from MIP solver final state
    final_routes = []
    if model.SolCount > 0:
        x_val = {k: v.X for k, v in x.items()}
        for j in range(1, N + 1):
            if (0, j) in valid_arcs and x_val[0, j] > 0.5:
                r = [0]
                curr = j
                while curr != 0:
                    r.append(curr)
                    next_node = 0
                    for nxt in range(N + 1):
                        if (curr, nxt) in valid_arcs and x_val[curr, nxt] > 0.5:
                            next_node = nxt
                            break
                    if next_node == curr or next_node is None:
                        break
                    curr = next_node
                r.append(0)
                final_routes.append(r)
    else:
        final_routes = best_routes

    # Recalculate deterministic and minimal bounds per discovered schedule block
    departure_times = {}
    loads = {}
    total_dist = 0.0

    for route in final_routes:
        t = float(ready_time[0])
        load = 0
        for k in range(len(route) - 1):
            u = route[k]
            v = route[k + 1]
            total_dist += dist_matrix[u][v]
            
            arr = t + eff_time[u][v]
            t = max(arr, float(ready_time[v]))
            
            if v != 0:
                load += demand[v]
                departure_times[str(v)] = round(t, 2)
                loads[str(v)] = int(load)

    # Materialize output payload fulfilling solution formatting schema constraints
    out_dict = {
        "objective_value": float(len(final_routes)),
        "num_vehicles": len(final_routes),
        "routes": final_routes,
        "total_distance": round(total_dist, 2),
        "departure_times": departure_times,
        "loads": loads
    }

    with open(args.solution_path, 'w') as f:
        json.dump(out_dict, f, indent=2)

if __name__ == "__main__":
    main()