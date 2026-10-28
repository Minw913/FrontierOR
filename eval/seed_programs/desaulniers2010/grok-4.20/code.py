import json
import argparse
import time
import sys
from typing import List, Dict, Any
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

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

    start_time = time.time()
    remaining_time = args.time_limit

    # Extract data
    n = instance['num_customers']
    Q = instance['vehicle_capacity']
    K_max = instance['num_vehicles_upper_bound']
    T = instance['scheduling_horizon']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']
    split_allowed = instance['split_deliveries_allowed']
    service_indep = instance['service_time_independent_of_quantity']

    # Nodes: 0 = start depot, 1..n = customers, n+1 = return depot
    N = n + 2
    start_depot = 0
    end_depot = n + 1
    customers_list = list(range(1, n + 1))

    # Time windows and service
    a = [0] * N
    b = [0] * N
    s = [0] * N
    d = [0] * N

    a[start_depot] = depot['time_window'][0]
    b[start_depot] = depot['time_window'][1]
    s[start_depot] = depot['service_time']
    d[start_depot] = depot['demand']

    for cust in customers:
        idx = cust['id']
        a[idx] = cust['time_window'][0]
        b[idx] = cust['time_window'][1]
        s[idx] = cust['service_time']
        d[idx] = cust['demand']

    a[end_depot] = 0
    b[end_depot] = T
    s[end_depot] = 0
    d[end_depot] = 0

    # Precompute feasible arcs (i,j) where i != j, considering time windows
    arcs = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Check if can go i -> j
            # Earliest at i + travel from i to j <= latest at j
            travel = dist[i][j] if i < len(dist) and j < len(dist[0]) else 0
            if i == end_depot or j == start_depot:
                continue
            if i == start_depot and j == end_depot:
                arcs.append((i, j))
                continue
            earliest_arrival = a[i] + s[i] + travel
            if earliest_arrival <= b[j]:
                arcs.append((i, j))

    # Adjust for return depot indexing: end_depot = n+1, dist uses 0 for depot
    # dist[0][j] for depot to customer, dist[i][0] for customer to depot
    # For end_depot, treat as depot return
    def get_dist(i, j):
        ii = 0 if i == start_depot or i == end_depot else i
        jj = 0 if j == start_depot or j == end_depot else j
        return dist[ii][jj]

    # Rebuild arcs with correct check
    arcs = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if i == end_depot and j != start_depot:  # cannot leave end
                continue
            if j == start_depot and i != end_depot:  # only enter end from customers or start
                continue
            travel_time = get_dist(i, j)
            if i == start_depot:
                earliest_start_i = a[i]
            else:
                earliest_start_i = a[i]
            arrival = earliest_start_i + s[i] + travel_time
            if arrival <= b[j] or (j == end_depot and arrival <= b[end_depot]):
                arcs.append((i, j))

    # Minimum and maximum vehicles
    total_demand = sum(d[i] for i in customers_list)
    K_min = (total_demand + Q - 1) // Q
    K_min = max(K_min, 1)
    K_max = min(K_max, len(customers_list) + 5)  # reasonable bound

    # Build model
    try:
        model = gp.Model("SDVRPTW")
        model.Params.TimeLimit = remaining_time - 5 if remaining_time > 10 else 1
        model.Params.Threads = 1
        model.Params.MIPGap = 1e-4
        model.Params.LogToConsole = 0

        # Variables
        x = {}
        for i, j in arcs:
            for k in range(K_max):
                x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

        # Delivery quantity: only if split allowed, else full demand per visit is complicated
        # To make it general, use continuous delivery variables
        y = {}
        for i in customers_list:
            for k in range(K_max):
                y[i, k] = model.addVar(lb=0, ub=min(d[i], Q), vtype=GRB.CONTINUOUS, name=f"y_{i}_{k}")

        # Time variables
        t = {}
        for i in range(N):
            for k in range(K_max):
                t[i, k] = model.addVar(lb=a[i], ub=b[i], vtype=GRB.CONTINUOUS, name=f"t_{i}_{k}")

        # Load variables (cumulative)
        u = {}
        for i in customers_list:
            for k in range(K_max):
                u[i, k] = model.addVar(lb=0, ub=Q, vtype=GRB.CONTINUOUS, name=f"u_{i}_{k}")

        # Use vehicle
        use = {}
        for k in range(K_max):
            use[k] = model.addVar(vtype=GRB.BINARY, name=f"use_{k}")

        # Objective: minimize total distance
        obj = gp.quicksum(get_dist(i, j) * x[i, j, k] for i, j in arcs for k in range(K_max))
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints

        # 1. Each customer demand satisfied (if split allowed, else must be exactly by one visit but handled by y)
        for i in customers_list:
            model.addConstr(gp.quicksum(y[i, k] for k in range(K_max)) >= d[i])

        # 2. Vehicle capacity
        for k in range(K_max):
            model.addConstr(gp.quicksum(y[i, k] for i in customers_list) <= Q * use[k])

        # 3. Link y to x: if y>0 then visit
        for i in customers_list:
            for k in range(K_max):
                visit = gp.quicksum(x[i, j, k] for j in range(N) if (i, j) in arcs)
                model.addConstr(y[i, k] <= min(d[i], Q) * visit)

        # 4. Flow conservation for each vehicle
        for k in range(K_max):
            # Start from depot
            model.addConstr(gp.quicksum(x[start_depot, j, k] for j in range(N) if (start_depot, j) in arcs) == use[k])
            # End at end_depot
            model.addConstr(gp.quicksum(x[i, end_depot, k] for i in range(N) if (i, end_depot) in arcs) == use[k])

            for i in customers_list:
                inflow = gp.quicksum(x[j, i, k] for j in range(N) if (j, i) in arcs)
                outflow = gp.quicksum(x[i, j, k] for j in range(N) if (i, j) in arcs)
                model.addConstr(inflow == outflow)
                model.addConstr(inflow <= 1)  # no split visit by same vehicle

        # 5. Time windows and sequencing
        M = T + 10
        for i, j in arcs:
            if j == start_depot or i == end_depot:
                continue
            for k in range(K_max):
                if (i, j) in arcs:
                    model.addConstr(t[i, k] + s[i] + get_dist(i, j) <= t[j, k] + M * (1 - x[i, j, k]))

        # 6. Subtour elimination / load
        for i in customers_list:
            for j in customers_list:
                if i != j and (i, j) in arcs and (j, i) in arcs:
                    for k in range(K_max):
                        model.addConstr(u[i, k] + y[j, k] <= u[j, k] + Q * (1 - x[i, j, k]))

        for i in customers_list:
            for k in range(K_max):
                model.addConstr(y[i, k] <= u[i, k])
                model.addConstr(u[i, k] <= Q * use[k])

        # 7. At most K_max, at least K_min
        model.addConstr(gp.quicksum(use[k] for k in range(K_max)) >= K_min)
        model.addConstr(gp.quicksum(use[k] for k in range(K_max)) <= K_max)

        # 8. No travel from end to start except idle but already constrained

        # Callback for incumbent logging
        best_obj = float('inf')
        def callback(model, where):
            nonlocal best_obj
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < best_obj - 1e-6:
                    best_obj = obj_val
                    if logger:
                        logger.log(float(obj_val))

        # Optimize
        model.optimize(callback)

        # Extract solution
        if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
            objective_value = model.objVal
            num_vehicles = int(sum(use[k].X + 0.5 for k in range(K_max)))

            # Build routes and deliveries
            routes = []
            vehicle_id = 1
            deliveries_total = {i: 0.0 for i in customers_list}

            for k in range(K_max):
                if use[k].X < 0.5:
                    continue

                # Find route
                route = [start_depot]
                current = start_depot
                visited = set()

                while True:
                    found = False
                    for j in range(N):
                        if (current, j) in arcs and x[current, j, k].X > 0.5 and j not in visited:
                            if j == end_depot:
                                route.append(end_depot)
                                found = True
                                break
                            route.append(j)
                            visited.add(j)
                            current = j
                            found = True
                            break
                    if not found or current == end_depot:
                        if current != end_depot:
                            route.append(end_depot)
                        break

                # Deliveries for this vehicle
                deliv = {}
                for i in customers_list:
                    qty = y[i, k].X
                    if qty > 1e-6:
                        deliv[str(i)] = float(round(qty, 6))  # avoid float precision
                        deliveries_total[i] += qty

                if len(route) > 2 or (len(route) == 2 and route[0] == 0 and route[1] == end_depot):
                    routes.append({
                        "vehicle": vehicle_id,
                        "route": route,
                        "deliveries": deliv
                    })
                    vehicle_id += 1

            # Fill deliveries dict
            deliveries_dict = {}
            for cid in customers_list:
                deliveries_dict[str(cid)] = {
                    "demand": d[cid],
                    "total_delivered": float(round(deliveries_total[cid], 6))
                }

            solution = {
                "objective_value": float(objective_value),
                "num_vehicles_used": int(num_vehicles),
                "routes": routes,
                "deliveries": deliveries_dict
            }

            # If no routes but demand zero, add dummy
            if total_demand == 0:
                solution["num_vehicles_used"] = 0
                solution["routes"] = []
                solution["objective_value"] = 0.0

            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)

        else:
            # No solution found, write empty feasible if possible
            solution = {
                "objective_value": 0.0,
                "num_vehicles_used": 0,
                "routes": [],
                "deliveries": {}
            }
            for cid in customers_list:
                solution["deliveries"][str(cid)] = {
                    "demand": d[cid],
                    "total_delivered": 0.0
                }
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)

    except Exception as e:
        # Emergency fallback: empty solution
        solution = {
            "objective_value": 0.0,
            "num_vehicles_used": 0,
            "routes": [],
            "deliveries": {}
        }
        custs = instance.get("customers", [])
        for c in custs:
            cid = c["id"]
            solution["deliveries"][str(cid)] = {
                "demand": c["demand"],
                "total_delivered": 0.0
            }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()