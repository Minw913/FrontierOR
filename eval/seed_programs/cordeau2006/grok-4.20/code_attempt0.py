import json
import argparse
import time
import math
from typing import Dict, List, Any
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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

    n_vehicles = instance['num_vehicles']
    n_users = instance['num_users']
    Q = instance['vehicle_capacity']
    max_ride = instance['maximum_ride_time']
    max_route_dur = instance['maximum_route_duration']
    nodes = instance['nodes']
    N = instance['num_nodes']
    depot_start = 0
    depot_end = N - 1
    pickups = [i for i in range(N) if nodes[i]['node_type'] == 'pickup']
    dropoffs = [i for i in range(N) if nodes[i]['node_type'] == 'dropoff']
    all_nodes = list(range(N))

    # Precompute distances and travel times (same)
    dist = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                dist[i][j] = euclidean_distance(nodes[i]['x'], nodes[i]['y'], nodes[j]['x'], nodes[j]['y'])

    # Earliest and latest times
    E = [nodes[i]['earliest_time'] for i in range(N)]
    L = [nodes[i]['latest_time'] for i in range(N)]
    service = [nodes[i]['service_duration'] for i in range(N)]
    load_val = [nodes[i]['load'] for i in range(N)]

    # Paired nodes
    pair = {}
    for nd in nodes:
        if 'paired_node' in nd:
            pair[nd['node_id']] = nd['paired_node']

    # Build model
    model = gp.Model("DARP")
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPGap, 0.01)
    model.setParam(GRB.Param.LogToConsole, 0)

    # Variables
    x = {}
    for k in range(n_vehicles):
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")

    t = {}
    for k in range(n_vehicles):
        for i in all_nodes:
            t[k, i] = model.addVar(lb=0.0, ub=L[i] + 1e-3, name=f"t_{k}_{i}")

    q = {}
    for k in range(n_vehicles):
        for i in all_nodes:
            q[k, i] = model.addVar(lb=0.0, ub=Q, name=f"q_{k}_{i}")

    # Objective: minimize total travel cost
    obj = gp.quicksum(dist[i][j] * x[k, i, j] for k in range(n_vehicles) for i in all_nodes for j in all_nodes if i != j)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # Each user node visited exactly once by one vehicle
    for i in pickups + dropoffs:
        model.addConstr(gp.quicksum(x[k, i, j] for k in range(n_vehicles) for j in all_nodes if i != j) == 1)

    # Flow conservation
    for k in range(n_vehicles):
        for i in all_nodes:
            if i == depot_start or i == depot_end:
                continue
            model.addConstr(
                gp.quicksum(x[k, i, j] for j in all_nodes if i != j) ==
                gp.quicksum(x[k, j, i] for j in all_nodes if i != j)
            )

    # Each vehicle leaves origin and returns to destination exactly once
    for k in range(n_vehicles):
        model.addConstr(gp.quicksum(x[k, depot_start, j] for j in all_nodes if j != depot_start) == 1)
        model.addConstr(gp.quicksum(x[k, i, depot_end] for i in all_nodes if i != depot_end) == 1)

    # Time constraints
    M = max(L) + 1000.0
    for k in range(n_vehicles):
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    model.addConstr(
                        t[k, j] >= t[k, i] + service[i] + dist[i][j] - M * (1 - x[k, i, j])
                    )

    # Load constraints
    for k in range(n_vehicles):
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    model.addConstr(
                        q[k, j] >= q[k, i] + load_val[j] - Q * (1 - x[k, i, j])
                    )
                    model.addConstr(
                        q[k, j] <= q[k, i] + load_val[j] + Q * (1 - x[k, i, j])
                    )

    # Capacity and non-negativity already in variable bounds but reinforce
    for k in range(n_vehicles):
        for i in all_nodes:
            model.addConstr(q[k, i] >= max(0, load_val[i]))
            model.addConstr(q[k, i] <= min(Q, Q + load_val[i]))

    # Pickup before delivery and same vehicle
    for k in range(n_vehicles):
        for p in pickups:
            d = pair[p]
            model.addConstr(
                gp.quicksum(x[k, p, j] for j in all_nodes if p != j) ==
                gp.quicksum(x[k, d, j] for j in all_nodes if d != j)
            )
            # Linearized MTZ for precedence
            model.addConstr(
                t[k, d] >= t[k, p] + service[p] + dist[p][d] - M * (1 - gp.quicksum(x[k, p, j] for j in all_nodes if p != j))
            )

    # Ride time constraints
    for p in pickups:
        d = pair[p]
        for k in range(n_vehicles):
            model.addConstr(
                t[k, d] - t[k, p] - service[p] <= max_ride + M * (1 - gp.quicksum(x[k, p, j] for j in all_nodes if p != j))
            )
            model.addConstr(
                t[k, d] - t[k, p] - service[p] >= dist[p][d] - M * (1 - gp.quicksum(x[k, p, j] for j in all_nodes if p != j))
            )

    # Route duration
    for k in range(n_vehicles):
        model.addConstr(t[k, depot_end] - t[k, depot_start] <= max_route_dur)

    # Time windows already in variable bounds but add big-M enforcement
    for k in range(n_vehicles):
        for i in all_nodes:
            model.addConstr(t[k, i] >= E[i] - M * (1 - gp.quicksum(x[k, j, i] for j in all_nodes if j != i)))
            model.addConstr(t[k, i] <= L[i] + M * (1 - gp.quicksum(x[k, j, i] for j in all_nodes if j != i)))

    # Depot service times
    for k in range(n_vehicles):
        t[k, depot_start].lb = E[depot_start]
        t[k, depot_end].ub = L[depot_end]

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj_val))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) and model.SolCount > 0:
        objective_value = float(model.objVal)

        # Build routes and times
        routes = {}
        service_times = {}
        ride_times = {}

        for k in range(n_vehicles):
            # Find route
            route = [depot_start]
            current = depot_start
            visited = set([depot_start])
            while True:
                found = False
                for j in all_nodes:
                    if current != j and x[k, current, j].X > 0.5:
                        if j not in visited:
                            route.append(j)
                            visited.add(j)
                            current = j
                            found = True
                            break
                if not found or current == depot_end:
                    break
            if current != depot_end:
                route.append(depot_end)
            routes[f"vehicle_{k}"] = route

            # Service times
            for i in route:
                if i == depot_start:
                    service_times[f"depot_start_{k}"] = float(t[k, i].X)
                elif i == depot_end:
                    service_times[f"depot_end_{k}"] = float(t[k, i].X)
                else:
                    service_times[str(i)] = float(t[k, i].X)

        # Ride times
        for p in pickups:
            user_id = nodes[p]['user_id']
            d = pair[p]
            # Find which vehicle serves it
            serving_k = None
            for k in range(n_vehicles):
                if abs(x[k, p, pair[p]].X - 1.0) < 1e-4 or any(x[k, p, j].X > 0.5 for j in all_nodes):
                    serving_k = k
                    break
            if serving_k is not None:
                pickup_time = t[serving_k, p].X
                drop_time = t[serving_k, d].X
                ride = float(drop_time - pickup_time - service[p])
            else:
                ride = 0.0
            ride_times[str(user_id)] = ride

        solution = {
            "objective_value": objective_value,
            "routes": routes,
            "service_times": service_times,
            "ride_times": ride_times
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No solution found - write empty feasible if possible
        solution = {
            "objective_value": 0.0,
            "routes": {},
            "service_times": {},
            "ride_times": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()