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

    n_cust = instance['num_customers']
    n_nodes = instance['num_nodes']
    alpha = instance['alpha']
    nodes = instance['nodes']
    truck_tt = instance['truck_travel_times']
    drone_tt = instance['drone_travel_times']

    # Node indices: 0 = depot origin, 1..n_cust = customers, n_cust+1 = depot destination
    depot_o = 0
    depot_d = n_cust + 1
    customers = list(range(1, n_cust + 1))
    all_nodes = list(range(n_nodes))  # 0 to n_cust+1

    # Precompute possible combined nodes (all customers can be)
    # We will model with binary variables for types

    model = gp.Model("TruckDrone")
    model.setParam(GRB.Param.TimeLimit, args.time_limit - 5)  # leave margin
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPFocus, 1)
    model.setParam(GRB.Param.Heuristics, 0.2)
    model.setParam(GRB.Param.Cuts, 2)

    # Decision variables
    # x[i][j] = 1 if truck travels i -> j
    x = {}
    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                continue
            if i == depot_d or j == depot_o:
                continue
            if i == depot_o and j == depot_d:
                continue
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # y[k][l] = 1 if drone travels k -> l
    y = {}
    for k in all_nodes:
        for l in all_nodes:
            if k == l:
                continue
            if k == depot_d or l == depot_o:
                continue
            if k == depot_o and l == depot_d and n_cust >= 1:  # forbid direct, but will add constraint
                pass
            y[k, l] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}_{l}")

    # Customer type
    # t_i = 1 if truck serves i (truck-only or combined)
    # d_i = 1 if drone serves i (drone-only or combined)
    # c_i = 1 if combined
    t = {}
    d = {}
    c = {}
    for i in customers:
        t[i] = model.addVar(vtype=GRB.BINARY, name=f"t_{i}")
        d[i] = model.addVar(vtype=GRB.BINARY, name=f"d_{i}")
        c[i] = model.addVar(vtype=GRB.BINARY, name=f"c_{i}")

    # Arrival times
    a = {}
    for i in all_nodes:
        if i == depot_d:
            a[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"a_{i}")
        else:
            a[i] = model.addVar(lb=0, ub=1000000, vtype=GRB.CONTINUOUS, name=f"a_{i}")

    # Objective
    model.setObjective(a[depot_d], GRB.MINIMIZE)

    # Constraints

    # 1. Exactly one departure from depot_o for truck and drone
    model.addConstr(gp.quicksum(x[depot_o, j] for j in customers) == 1, "truck_depart")
    model.addConstr(gp.quicksum(y[depot_o, k] for k in customers) == 1, "drone_depart")

    # 2. Exactly one arrival to depot_d
    model.addConstr(gp.quicksum(x[i, depot_d] for i in customers) == 1, "truck_return")
    model.addConstr(gp.quicksum(y[k, depot_d] for k in customers) == 1, "drone_return")

    # 3. Flow conservation for truck at customers
    for i in customers:
        model.addConstr(
            gp.quicksum(x[j, i] for j in all_nodes if (j, i) in x) ==
            gp.quicksum(x[i, j] for j in all_nodes if (i, j) in x),
            f"truck_flow_{i}"
        )
        # Number of truck arcs leaving = 1 if truck serves (t_i), 0 otherwise
        model.addConstr(
            gp.quicksum(x[i, j] for j in all_nodes if (i, j) in x) == t[i],
            f"truck_leave_{i}"
        )

    # 4. Flow conservation for drone at customers
    for i in customers:
        model.addConstr(
            gp.quicksum(y[k, i] for k in all_nodes if (k, i) in y) ==
            gp.quicksum(y[i, l] for l in all_nodes if (i, l) in y),
            f"drone_flow_{i}"
        )
        model.addConstr(
            gp.quicksum(y[i, l] for l in all_nodes if (i, l) in y) == d[i],
            f"drone_leave_{i}"
        )

    # 5. Type linking: c_i <= t_i, c_i <= d_i, t_i + d_i <= 1 + c_i
    for i in customers:
        model.addConstr(c[i] <= t[i], f"c_le_t_{i}")
        model.addConstr(c[i] <= d[i], f"c_le_d_{i}")
        model.addConstr(t[i] + d[i] <= 1 + c[i], f"td_le_1pc_{i}")

    # 6. Timing constraints
    M = 1000000
    for i, j in x.keys():
        if i == depot_d or j == depot_o:
            continue
        model.addConstr(
            a[j] >= a[i] + truck_tt[i][j] - M * (1 - x[i, j]),
            f"truck_time_{i}_{j}"
        )

    for k, l in y.keys():
        if k == depot_d or l == depot_o:
            continue
        model.addConstr(
            a[l] >= a[k] + drone_tt[k][l] - M * (1 - y[k, l]),
            f"drone_time_{k}_{l}"
        )

    # 7. Drone arc restrictions based on combined
    # For every pair of distinct customers i,j, y[i,j] + y[j,i] <= c[i] + c[j]
    for i_idx in range(len(customers)):
        for j_idx in range(i_idx + 1, len(customers)):
            i = customers[i_idx]
            j = customers[j_idx]
            if (i, j) in y and (j, i) in y:
                model.addConstr(
                    y[i, j] + y[j, i] <= c[i] + c[j],
                    f"drone_pair_{i}_{j}"
                )

    # 8. Forbid drone serving single customer directly out and back
    for i in customers:
        if (depot_o, i) in y and (i, depot_d) in y:
            model.addConstr(
                y[depot_o, i] + y[i, depot_d] <= 1,
                f"no_single_drone_{i}"
            )

    # 9. At least 2 customers served (but since >=2 given)

    # 10. Depot destination has no outgoing, origin no incoming already by var definition

    # Initial solution heuristic: simple truck-only tour
    try:
        # Simple nearest neighbor for truck only
        visited = set()
        current = depot_o
        truck_path = [depot_o]
        current_time = 0
        for _ in range(n_cust):
            best_next = None
            best_time = float('inf')
            for nxt in customers:
                if nxt not in visited:
                    if truck_tt[current][nxt] < best_time:
                        best_time = truck_tt[current][nxt]
                        best_next = nxt
            if best_next is None:
                break
            truck_path.append(best_next)
            current_time += truck_tt[current][best_next]
            visited.add(best_next)
            current = best_next
        truck_path.append(depot_d)
        current_time += truck_tt[current][depot_d]

        # Set initial solution
        model.setParam(GRB.Param.SolutionLimit, 5)
        for i in range(len(truck_path) - 1):
            u = truck_path[i]
            v = truck_path[i + 1]
            if (u, v) in x:
                x[u, v].start = 1.0

        for i in customers:
            t[i].start = 1.0
            d[i].start = 0.0
            c[i].start = 0.0

        for i in customers:
            a[i].start = 0.0  # will be updated by model

        a[depot_o].start = 0.0
        a[depot_d].start = current_time * 1.2  # loose

        # Also add a simple drone tour: go to farthest then back
        if n_cust >= 1:
            drone_start = max(customers, key=lambda k: drone_tt[depot_o][k])
            y[depot_o, drone_start].start = 1.0
            y[drone_start, depot_d].start = 1.0
            d[drone_start].start = 1.0
            c[drone_start].start = 0.0
            t[drone_start].start = 0.0 if n_cust > 1 else 1.0

    except:
        pass

    # Callback for logging incumbents
    best_obj = float('inf')
    start_time = time.time()

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-4:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
        # Get values
        try:
            obj_val = model.objVal
        except:
            obj_val = best_obj if best_obj < float('inf') else 999999.0

        # Reconstruct truck route
        truck_route = [depot_o]
        current = depot_o
        visited_truck = set()
        while True:
            found = False
            for j in all_nodes:
                if (current, j) in x and abs(x[current, j].X) > 0.5:
                    if j == depot_d:
                        truck_route.append(depot_d)
                        found = True
                        break
                    if j not in visited_truck:
                        truck_route.append(j)
                        visited_truck.add(j)
                        current = j
                        found = True
                        break
            if not found or j == depot_d:
                break
        if truck_route[-1] != depot_d:
            truck_route.append(depot_d)

        # Reconstruct drone route
        drone_route = [depot_o]
        current = depot_o
        visited_drone = set()
        while True:
            found = False
            for j in all_nodes:
                if (current, j) in y and abs(y[current, j].X) > 0.5:
                    if j == depot_d:
                        drone_route.append(depot_d)
                        found = True
                        break
                    if j not in visited_drone:
                        drone_route.append(j)
                        visited_drone.add(j)
                        current = j
                        found = True
                        break
            if not found or j == depot_d:
                break
        if drone_route[-1] != depot_d:
            drone_route.append(depot_d)

        # Customer types
        customer_types = {}
        for i in customers:
            if abs(c[i].X) > 0.5:
                customer_types[str(i)] = "combined"
            elif abs(t[i].X) > 0.5:
                customer_types[str(i)] = "truck"
            else:
                customer_types[str(i)] = "drone"

        # Arrival times
        arrival_times = {}
        for i in all_nodes:
            try:
                arrival_times[str(i)] = float(a[i].X)
            except:
                arrival_times[str(i)] = 0.0
        arrival_times[str(depot_d)] = float(obj_val)

        # If routes are invalid or empty, fallback to truck-only
        if len(truck_route) < 3 or len(drone_route) < 3:
            # Build truck only solution
            truck_route = [0] + customers + [depot_d]
            drone_route = [0, customers[0], depot_d]
            customer_types = {}
            for cid in customers:
                customer_types[str(cid)] = "truck"
            customer_types[str(customers[0])] = "combined"  # at least one
            arrival_times = {str(i): 0.0 for i in all_nodes}
            cum_time = 0.0
            for k in range(1, len(truck_route)):
                prev = truck_route[k-1]
                curr = truck_route[k]
                cum_time += truck_tt[prev][curr]
                arrival_times[str(curr)] = cum_time
            obj_val = cum_time
            arrival_times[str(depot_d)] = cum_time

        solution = {
            "objective_value": float(obj_val),
            "truck_route": [int(x) for x in truck_route],
            "drone_route": [int(x) for x in drone_route],
            "customer_types": customer_types,
            "arrival_times": arrival_times
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

    else:
        # Emergency fallback solution
        cust_list = list(range(1, n_cust+1))
        truck_route = [0] + cust_list + [n_cust + 1]
        drone_route = [0, cust_list[0], n_cust + 1]
        customer_types = {}
        for cid in cust_list:
            customer_types[str(cid)] = "truck"
        if n_cust > 0:
            customer_types[str(cust_list[0])] = "combined"
        arrival_times = {"0": 0.0}
        cum = 0.0
        prev = 0
        for nxt in cust_list + [n_cust+1]:
            cum += truck_tt[prev][nxt]
            arrival_times[str(nxt)] = float(cum)
            prev = nxt
        solution = {
            "objective_value": float(cum),
            "truck_route": truck_route,
            "drone_route": drone_route,
            "customer_types": customer_types,
            "arrival_times": arrival_times
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()