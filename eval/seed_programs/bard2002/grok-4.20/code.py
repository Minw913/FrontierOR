import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_customers = instance['num_customers']
    capacity = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    distance_matrix = instance['distance_matrix']
    horizon = instance['scheduling_horizon']
    travel_includes_service = instance.get('travel_time_includes_service_time', False)

    N = num_customers
    nodes = [0] + [c['id'] for c in customers]
    customer_list = list(range(1, N+1))
    all_nodes = [0] + customer_list

    # Precompute effective travel times
    tt = [[0.0 for _ in all_nodes] for _ in all_nodes]
    ready = {0: depot['ready_time']}
    due = {0: depot['due_date']}
    demand = {0: 0}
    service = {0: 0}

    for c in customers:
        idx = c['id']
        ready[idx] = c['ready_time']
        due[idx] = c['due_date']
        demand[idx] = c['demand']
        service[idx] = c['service_time']

    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                tt[i][j] = 0.0
                continue
            base = distance_matrix[i][j]
            if i == 0:
                t = base
            else:
                if travel_includes_service:
                    t = base
                else:
                    t = base + service[i]
            diff = ready[j] - due[i]
            if diff > t:
                t = float(diff)
            tt[i][j] = t

    # Feasible arcs
    arcs = []
    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                continue
            if i != 0 and j != 0:
                if due[i] + tt[i][j] > ready[j] + 1e-5:
                    continue
                if demand[i] + demand[j] > capacity:
                    continue
            arcs.append((i, j))

    model = gp.Model("VRPTW")
    model.Params.TimeLimit = float(args.time_limit)
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.LogToConsole = 0
    model.Params.Heuristics = 0.8
    model.Params.MIPGap = 1e-4

    # Variables
    x = {}
    max_vehicles = min(N, 5) if N <= 15 else min(N, 8)
    vehicle_range = list(range(1, max_vehicles + 1))
    for i_idx, j_idx in arcs:
        for k in vehicle_range:
            x[i_idx, j_idx, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i_idx}_{j_idx}_{k}")

    y = {}
    for k in vehicle_range:
        y[k] = model.addVar(vtype=GRB.BINARY, name=f"use_{k}")

    tvar = {}
    for i in all_nodes:
        for k in vehicle_range:
            tvar[i, k] = model.addVar(lb=0, ub=horizon, vtype=GRB.CONTINUOUS, name=f"t_{i}_{k}")

    loadvar = {}
    for i in all_nodes:
        for k in vehicle_range:
            loadvar[i, k] = model.addVar(lb=0, ub=capacity, vtype=GRB.CONTINUOUS, name=f"load_{i}_{k}")

    M = float(horizon + 10.0)
    M_load = float(capacity + 1.0)

    model.setObjective(gp.quicksum(y[k] for k in vehicle_range), GRB.MINIMIZE)

    # Each customer visited exactly once
    for i in customer_list:
        expr = gp.LinExpr()
        for j in all_nodes:
            for k in vehicle_range:
                if (i, j, k) in x:
                    expr.add(x[i, j, k])
        model.addConstr(expr == 1, f"visit_{i}")

    # Flow conservation
    for i in customer_list:
        for k in vehicle_range:
            incoming = gp.LinExpr()
            outgoing = gp.LinExpr()
            for j in all_nodes:
                if (j, i, k) in x:
                    incoming.add(x[j, i, k])
                if (i, j, k) in x:
                    outgoing.add(x[i, j, k])
            model.addConstr(incoming == outgoing, f"flow_{i}_{k}")

    # Link to vehicle use
    for k in vehicle_range:
        expr = gp.LinExpr()
        for j in customer_list:
            if (0, j, k) in x:
                expr.add(x[0, j, k])
        model.addConstr(expr <= y[k], f"use_link_{k}")

    # Depot constraints
    for k in vehicle_range:
        model.addConstr(tvar[0, k] >= ready[0] * y[k], f"depot_ready_{k}")
        model.addConstr(tvar[0, k] <= due[0] * y[k] + M * (1 - y[k]), f"depot_due_{k}")
        model.addConstr(loadvar[0, k] == 0, f"depot_load_{k}")

    # Time windows
    for i in customer_list:
        for k in vehicle_range:
            vis = gp.LinExpr()
            for j_idx in all_nodes:
                if (j_idx, i, k) in x:
                    vis.add(x[j_idx, i, k])
            model.addConstr(tvar[i, k] >= ready[i] - M * (1 - vis), f"tw_ready_{i}_{k}")
            model.addConstr(tvar[i, k] <= due[i] + M * (1 - vis), f"tw_due_{i}_{k}")

    # Travel time constraints
    for i in all_nodes:
        for j in customer_list:
            for k in vehicle_range:
                if (i, j, k) in x:
                    model.addConstr(tvar[j, k] >= tvar[i, k] + tt[i][j] - M * (1 - x[i, j, k]), 
                                  f"time_{i}_{j}_{k}")

    # Return to depot
    for k in vehicle_range:
        for i in customer_list:
            if (i, 0, k) in x:
                model.addConstr(tvar[0, k] >= tvar[i, k] + tt[i][0] - M * (1 - x[i, 0, k]), 
                              f"return_time_{i}_{k}")

    # Load constraints
    for i in customer_list:
        for k in vehicle_range:
            vis = gp.LinExpr()
            for j_idx in all_nodes:
                if (j_idx, i, k) in x:
                    vis.add(x[j_idx, i, k])
            model.addConstr(loadvar[i, k] >= demand[i] - M_load * (1 - vis), f"minload_{i}_{k}")
            model.addConstr(loadvar[i, k] <= capacity + M_load * (1 - vis), f"cap_{i}_{k}")

    for i in all_nodes:
        for j in customer_list:
            for k in vehicle_range:
                if (i, j, k) in x:
                    model.addConstr(loadvar[j, k] >= loadvar[i, k] + demand[j] - M_load * (1 - x[i, j, k]), 
                                  f"load_flow_{i}_{j}_{k}")

    # Symmetry breaking
    for k_idx in range(len(vehicle_range)-1):
        model.addConstr(y[vehicle_range[k_idx]] >= y[vehicle_range[k_idx+1]], f"sym_{k_idx}")

    # Callback for logging
    best_obj = None
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if best_obj is None or obj + 1e-6 < best_obj:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    model.optimize(callback)

    status = model.Status
    if status == GRB.INFEASIBLE:
        solution = {
            "objective_value": float(N),
            "num_vehicles": int(N),
            "routes": [[0, i, 0] for i in customer_list],
            "total_distance": 0.0,
            "departure_times": {},
            "loads": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # Extract solution
    vehicle_routes = []
    departure_dict = {}
    load_dict = {}
    total_dist = 0.0

    used_vehicles = []
    for k in vehicle_range:
        if y[k].X > 0.5:
            used_vehicles.append(k)

    for k in used_vehicles:
        route = [0]
        current = 0
        visited = set()
        while True:
            found = False
            for j in all_nodes:
                if j in visited:
                    continue
                if (current, j, k) in x and x[current, j, k].X > 0.5:
                    if j == 0 and len(route) > 1:
                        route.append(0)
                        found = True
                        break
                    elif j != 0:
                        route.append(j)
                        visited.add(j)
                        current = j
                        found = True
                        break
            if not found:
                if route[-1] != 0:
                    route.append(0)
                break
        if len(route) > 2:
            vehicle_routes.append(route)

    # Fill departure and load from solution
    for i in customer_list:
        assigned = False
        for k in used_vehicles:
            if any((i == r) for r in route for route in vehicle_routes if i in route):
                if (i, k) in tvar:
                    val = tvar[i, k].X
                    departure_dict[str(i)] = round(float(val), 6)
                    assigned = True
                if (i, k) in loadvar:
                    lval = loadvar[i, k].X
                    load_dict[str(i)] = round(float(lval), 6)
                break

    for route in vehicle_routes:
        for r_idx in range(len(route) - 1):
            a = route[r_idx]
            b = route[r_idx + 1]
            total_dist += distance_matrix[a][b]

    num_veh = len(vehicle_routes)
    objective = float(num_veh)

    solution = {
        "objective_value": objective,
        "num_vehicles": int(num_veh),
        "routes": vehicle_routes,
        "total_distance": round(total_dist, 6),
        "departure_times": departure_dict,
        "loads": load_dict
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger and best_obj is not None:
        logger.log(best_obj)

if __name__ == "__main__":
    main()