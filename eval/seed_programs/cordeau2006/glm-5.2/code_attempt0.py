import json
import math
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_vehicles = data["num_vehicles"]
    num_users = data["num_users"]
    Q = data["vehicle_capacity"]
    R_max = data["maximum_ride_time"]
    T_max = data["maximum_route_duration"]
    nodes = data["nodes"]
    num_nodes = len(nodes)

    # node properties
    x_coord = [0.0] * num_nodes
    y_coord = [0.0] * num_nodes
    e = [0.0] * num_nodes
    l = [0.0] * num_nodes
    s = [0.0] * num_nodes
    q = [0.0] * num_nodes
    node_type = [""] * num_nodes
    user_id = [-1] * num_nodes
    paired_node = [-1] * num_nodes

    for node in nodes:
        nid = node["node_id"]
        x_coord[nid] = node["x"]
        y_coord[nid] = node["y"]
        e[nid] = node["earliest_time"]
        l[nid] = node["latest_time"]
        s[nid] = node["service_duration"]
        q[nid] = node["load"]
        node_type[nid] = node["node_type"]
        if "user_id" in node:
            user_id[nid] = node["user_id"]
        if "paired_node" in node:
            paired_node[nid] = node["paired_node"]

    dist = [[0.0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist[i][j] = math.sqrt((x_coord[i] - x_coord[j])**2 + (y_coord[i] - y_coord[j])**2)

    depot_start = 0
    depot_end = num_nodes - 1
    P = [i for i in range(num_nodes) if node_type[i] == "pickup"]
    D = [i for i in range(num_nodes) if node_type[i] == "dropoff"]
    PD = P + D

    # Precompute tight Big-M for time constraints
    M_time = [[0.0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                M_time[i][j] = max(0.0, l[i] + s[i] + dist[i][j] - e[j])

    # M for route duration
    M_duration = max(0.0, l[depot_end] - e[depot_start] - T_max)

    model = gp.Model("DARP")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)

    # Variables
    x = {}
    for v in range(num_vehicles):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    if i == depot_end or j == depot_start:
                        continue
                    x[i, j, v] = model.addVar(vtype=GRB.BINARY, obj=dist[i][j], name=f"x_{i}_{j}_{v}")

    T = {}
    for i in range(num_nodes):
        if i == depot_start or i == depot_end:
            for v in range(num_vehicles):
                T[i, v] = model.addVar(lb=e[i], ub=l[i], vtype=GRB.CONTINUOUS, name=f"T_{i}_{v}")
        else:
            T[i] = model.addVar(lb=e[i], ub=l[i], vtype=GRB.CONTINUOUS, name=f"T_{i}")

    L = {}
    for i in range(num_nodes):
        if i == depot_start or i == depot_end:
            for v in range(num_vehicles):
                L[i, v] = model.addVar(lb=0, ub=Q, vtype=GRB.CONTINUOUS, name=f"L_{i}_{v}")
        else:
            lb_l = max(0, q[i])
            ub_l = min(Q, Q + q[i])
            L[i] = model.addVar(lb=lb_l, ub=ub_l, vtype=GRB.CONTINUOUS, name=f"L_{i}")

    model.update()

    # 1. Degree constraints
    for v in range(num_vehicles):
        model.addConstr(gp.quicksum(x[depot_start, j, v] for j in range(num_nodes) if (depot_start, j, v) in x) <= 1)
        model.addConstr(gp.quicksum(x[i, depot_end, v] for i in range(num_nodes) if (i, depot_end, v) in x) <= 1)

    for k in PD:
        model.addConstr(gp.quicksum(x[i, k, v] for v in range(num_vehicles) for i in range(num_nodes) if (i, k, v) in x) == 1)
        model.addConstr(gp.quicksum(x[k, j, v] for v in range(num_vehicles) for j in range(num_nodes) if (k, j, v) in x) == 1)

    for v in range(num_vehicles):
        for k in PD:
            model.addConstr(
                gp.quicksum(x[i, k, v] for i in range(num_nodes) if (i, k, v) in x) == 
                gp.quicksum(x[k, j, v] for j in range(num_nodes) if (k, j, v) in x)
            )

    # Coupling pickup and dropoff
    for u in range(num_users):
        p_node = -1
        d_node = -1
        for i in PD:
            if user_id[i] == u and node_type[i] == "pickup":
                p_node = i
            if user_id[i] == u and node_type[i] == "dropoff":
                d_node = i
        for v in range(num_vehicles):
            model.addConstr(
                gp.quicksum(x[i, p_node, v] for i in range(num_nodes) if (i, p_node, v) in x) == 
                gp.quicksum(x[i, d_node, v] for i in range(num_nodes) if (i, d_node, v) in x)
            )

    # 2. Time constraints
    for v in range(num_vehicles):
        for j in PD:
            if (depot_start, j, v) in x:
                model.addConstr(
                    T[j] >= T[depot_start, v] + s[depot_start] + dist[depot_start][j] - M_time[depot_start][j] * (1 - x[depot_start, j, v])
                )
        for i in PD:
            if (i, depot_end, v) in x:
                model.addConstr(
                    T[depot_end, v] >= T[i] + s[i] + dist[i][depot_end] - M_time[i][depot_end] * (1 - x[i, depot_end, v])
                )

    for i in PD:
        for j in PD:
            if i != j:
                sum_x = gp.quicksum(x[i, j, v] for v in range(num_vehicles) if (i, j, v) in x)
                if sum_x.size() > 0:
                    model.addConstr(
                        T[j] >= T[i] + s[i] + dist[i][j] - M_time[i][j] * (1 - sum_x)
                    )

    for v in range(num_vehicles):
        sum_x_out = gp.quicksum(x[depot_start, j, v] for j in range(num_nodes) if (depot_start, j, v) in x)
        model.addConstr(
            T[depot_end, v] - T[depot_start, v] <= T_max + M_duration * (1 - sum_x_out)
        )

    # 3. Load constraints
    for v in range(num_vehicles):
        for j in PD:
            if (depot_start, j, v) in x:
                model.addConstr(
                    L[j] >= L[depot_start, v] + q[j] - Q * (1 - x[depot_start, j, v])
                )
        for i in PD:
            if (i, depot_end, v) in x:
                model.addConstr(
                    L[depot_end, v] >= L[i] + q[depot_end] - Q * (1 - x[i, depot_end, v])
                )
    for i in PD:
        for j in PD:
            if i != j:
                sum_x = gp.quicksum(x[i, j, v] for v in range(num_vehicles) if (i, j, v) in x)
                if sum_x.size() > 0:
                    model.addConstr(
                        L[j] >= L[i] + q[j] - Q * (1 - sum_x)
                    )

    # 4. Ride time constraints
    for u in range(num_users):
        p_node = -1
        d_node = -1
        for i in PD:
            if user_id[i] == u and node_type[i] == "pickup":
                p_node = i
            if user_id[i] == u and node_type[i] == "dropoff":
                d_node = i
        model.addConstr(T[d_node] - T[p_node] - s[p_node] <= R_max)
        model.addConstr(T[d_node] - T[p_node] - s[p_node] >= dist[p_node][d_node])

    model.setObjective(gp.quicksum(x[i, j, v] * dist[i][j] for (i, j, v) in x), GRB.MINIMIZE)

    model.optimize()

    if model.SolCount > 0:
        obj_val = model.ObjVal
        routes = {}
        service_times = {}
        ride_times = {}

        for v in range(num_vehicles):
            route = [depot_start]
            curr = depot_start
            while curr != depot_end:
                next_node = -1
                for j in range(num_nodes):
                    if (curr, j, v) in x and x[curr, j, v].X > 0.5:
                        next_node = j
                        break
                if next_node == -1:
                    if curr == depot_start:
                        next_node = depot_end
                    else:
                        break
                route.append(next_node)
                curr = next_node
            routes[str(v)] = route

        for v in range(num_vehicles):
            if any((depot_start, j, v) in x and x[depot_start, j, v].X > 0.5 for j in range(num_nodes)):
                service_times[f"depot_start_{v}"] = T[depot_start, v].X
            if any((i, depot_end, v) in x and x[i, depot_end, v].X > 0.5 for i in range(num_nodes)):
                service_times[f"depot_end_{v}"] = T[depot_end, v].X

        for i in PD:
            if any((j, i, v) in x and x[j, i, v].X > 0.5 for j in range(num_nodes) for v in range(num_vehicles)):
                service_times[str(i)] = T[i].X

        for u in range(num_users):
            p_node = -1
            d_node = -1
            for i in PD:
                if user_id[i] == u and node_type[i] == "pickup":
                    p_node = i
                if user_id[i] == u and node_type[i] == "dropoff":
                    d_node = i
            if str(p_node) in service_times and str(d_node) in service_times:
                ride_times[str(u)] = service_times[str(d_node)] - service_times[str(p_node)] - s[p_node]

        sol = {
            "objective_value": obj_val,
            "routes": routes,
            "service_times": service_times,
            "ride_times": ride_times
        }

        if logger:
            logger.log_solution(obj_val, sol)

        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
    else:
        sol = {"objective_value": 0, "routes": {}, "service_times": {}, "ride_times": {}}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)

if __name__ == "__main__":
    solve()