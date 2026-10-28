import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

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
    n_nodes_input = instance['num_nodes']
    alpha = instance['alpha']
    nodes = instance['nodes']
    truck_tt = instance['truck_travel_times']
    drone_tt = instance['drone_travel_times']

    # Correct node definitions based on schema and common convention
    # 0 = origin depot, 1..n_cust = customers, n_cust+1 = destination depot
    depot_o = 0
    depot_d = n_cust + 1
    customers = list(range(1, n_cust + 1))
    all_nodes = list(range(depot_d + 1))  # 0 ... n_cust+1

    model = gp.Model("TruckDrone")
    model.setParam(GRB.Param.TimeLimit, max(1, args.time_limit - 5))
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPFocus, 1)
    model.setParam(GRB.Param.Heuristics, 0.2)
    model.setParam(GRB.Param.Cuts, 2)
    model.setParam(GRB.Param.LogToConsole, 0)

    # Decision variables
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

    y = {}
    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                continue
            if i == depot_d or j == depot_o:
                continue
            if i == depot_o and j == depot_d:
                continue
            y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    t = {}
    d_var = {}
    c = {}
    for i in customers:
        t[i] = model.addVar(vtype=GRB.BINARY, name=f"t_{i}")
        d_var[i] = model.addVar(vtype=GRB.BINARY, name=f"d_{i}")
        c[i] = model.addVar(vtype=GRB.BINARY, name=f"c_{i}")

    a = {}
    for i in all_nodes:
        a[i] = model.addVar(lb=0.0, ub=1e6, vtype=GRB.CONTINUOUS, name=f"a_{i}")

    model.setObjective(a[depot_d], GRB.MINIMIZE)

    # Constraints
    model.addConstr(gp.quicksum(x[depot_o, j] for j in customers) == 1, "truck_depart")
    model.addConstr(gp.quicksum(y[depot_o, k] for k in customers) == 1, "drone_depart")
    model.addConstr(gp.quicksum(x[i, depot_d] for i in customers) == 1, "truck_return")
    model.addConstr(gp.quicksum(y[k, depot_d] for k in customers) == 1, "drone_return")

    for i in customers:
        in_truck = gp.quicksum(x[j, i] for j in all_nodes if (j, i) in x)
        out_truck = gp.quicksum(x[i, j] for j in all_nodes if (i, j) in x)
        model.addConstr(in_truck == out_truck, f"truck_flow_{i}")
        model.addConstr(out_truck == t[i], f"truck_leave_{i}")

        in_drone = gp.quicksum(y[j, i] for j in all_nodes if (j, i) in y)
        out_drone = gp.quicksum(y[i, j] for j in all_nodes if (i, j) in y)
        model.addConstr(in_drone == out_drone, f"drone_flow_{i}")
        model.addConstr(out_drone == d_var[i], f"drone_leave_{i}")

    for i in customers:
        model.addConstr(c[i] <= t[i])
        model.addConstr(c[i] <= d_var[i])
        model.addConstr(t[i] + d_var[i] <= 1 + c[i])

    M = 1e6
    for i, j in x.keys():
        model.addConstr(a[j] >= a[i] + truck_tt[i][j] - M * (1 - x[i, j]), f"truck_time_{i}_{j}")
    for i, j in y.keys():
        model.addConstr(a[j] >= a[i] + drone_tt[i][j] - M * (1 - y[i, j]), f"drone_time_{i}_{j}")

    for ii in range(len(customers)):
        for jj in range(ii + 1, len(customers)):
            i = customers[ii]
            j = customers[jj]
            if (i, j) in y and (j, i) in y:
                model.addConstr(y[i, j] + y[j, i] <= c[i] + c[j], f"drone_pair_{i}_{j}")

    for i in customers:
        if (depot_o, i) in y and (i, depot_d) in y:
            model.addConstr(y[depot_o, i] + y[i, depot_d] <= 1, f"no_single_{i}")

    # Warm-start with a pure truck tour (and one drone customer if possible)
    try:
        cust_list = list(range(1, n_cust + 1))
        truck_route = [depot_o] + cust_list + [depot_d]
        for k in range(len(truck_route) - 1):
            u = truck_route[k]
            v = truck_route[k + 1]
            if (u, v) in x:
                x[u, v].start = 1.0

        for i in customers:
            t[i].start = 1.0
            d_var[i].start = 0.0
            c[i].start = 0.0

        a[depot_o].start = 0.0
        cum_time = 0.0
        prev = depot_o
        for nxt in cust_list + [depot_d]:
            cum_time += truck_tt[prev][nxt]
            if nxt in a:
                a[nxt].start = cum_time
            prev = nxt

        if n_cust >= 1:
            # Let drone serve the last customer (often farthest)
            last_c = cust_list[-1]
            if (depot_o, last_c) in y:
                y[depot_o, last_c].start = 1.0
            if (last_c, depot_d) in y:
                y[last_c, depot_d].start = 1.0
            d_var[last_c].start = 1.0
            t[last_c].start = 0.0
            c[last_c].start = 0.0
    except:
        pass

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

    model.optimize(callback)

    # Extract solution or fallback
    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT) or model.SolCount == 0:
        # Pure truck fallback
        cust_list = list(range(1, n_cust + 1))
        truck_route = [0] + cust_list + [depot_d]
        drone_route = [0, cust_list[0] if cust_list else 0, depot_d]
        customer_types = {}
        for cid in cust_list:
            customer_types[str(cid)] = "truck"
        if cust_list:
            customer_types[str(cust_list[0])] = "combined"
        arrival_times = {"0": 0.0}
        cum = 0.0
        prev = 0
        for nxt_node in cust_list + [depot_d]:
            cum += float(truck_tt[prev][nxt_node])
            arrival_times[str(nxt_node)] = float(cum)
            prev = nxt_node
        obj_val = cum
    else:
        obj_val = model.objVal

        # Reconstruct truck route
        truck_route = [depot_o]
        current = depot_o
        visited = set()
        for _ in range(n_cust + 5):
            found = False
            for nxt in all_nodes:
                if (current, nxt) in x and x[current, nxt].X > 0.5:
                    if nxt != depot_d and nxt in visited:
                        continue
                    truck_route.append(nxt)
                    visited.add(nxt)
                    current = nxt
                    found = True
                    break
            if not found or current == depot_d:
                break
        if truck_route[-1] != depot_d:
            truck_route.append(depot_d)

        # Reconstruct drone route
        drone_route = [depot_o]
        current = depot_o
        visited = set()
        for _ in range(n_cust + 5):
            found = False
            for nxt in all_nodes:
                if (current, nxt) in y and y[current, nxt].X > 0.5:
                    if nxt != depot_d and nxt in visited:
                        continue
                    drone_route.append(nxt)
                    visited.add(nxt)
                    current = nxt
                    found = True
                    break
            if not found or current == depot_d:
                break
        if drone_route[-1] != depot_d:
            drone_route.append(depot_d)

        customer_types = {}
        for i in customers:
            if c[i].X > 0.5:
                customer_types[str(i)] = "combined"
            elif t[i].X > 0.5:
                customer_types[str(i)] = "truck"
            else:
                customer_types[str(i)] = "drone"

        arrival_times = {}
        for i in all_nodes:
            arrival_times[str(i)] = float(a[i].X)
        arrival_times[str(depot_d)] = float(obj_val)

    # Final validation and robust fallback
    if len(truck_route) < 2 or truck_route[-1] != depot_d:
        cust_list = list(range(1, n_cust + 1))
        truck_route = [0] + cust_list + [depot_d]
        drone_route = [0] + ([cust_list[0]] if cust_list else []) + [depot_d]
        customer_types = {str(cid): "truck" for cid in cust_list}
        if cust_list:
            customer_types[str(cust_list[0])] = "combined"
        arrival_times = {"0": 0.0}
        cum = 0.0
        prev = 0
        for nxt_node in cust_list + [depot_d]:
            cum += float(truck_tt[prev][nxt_node])
            arrival_times[str(nxt_node)] = float(cum)
            prev = nxt_node
        obj_val = cum

    solution = {
        "objective_value": float(obj_val),
        "truck_route": [int(x) for x in truck_route],
        "drone_route": [int(x) for x in drone_route],
        "customer_types": customer_types,
        "arrival_times": arrival_times
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()