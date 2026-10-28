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
    user_from_pickup = {}
    for nd in nodes:
        if nd['node_type'] in ('pickup', 'dropoff'):
            pair[nd['node_id']] = nd['paired_node']
            if nd['node_type'] == 'pickup':
                user_from_pickup[nd['node_id']] = nd['user_id']

    # Build model
    model = gp.Model("DARP")
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPGap, 1e-4)
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
        model.addConstr(gp.quicksum(x[k, i, j] for k in range(n_vehicles) for j in all_nodes if i != j) == 1, name=f"visit_{i}")

    # Flow conservation
    for k in range(n_vehicles):
        for i in all_nodes:
            model.addConstr(
                gp.quicksum(x[k, i, j] for j in all_nodes if i != j) ==
                gp.quicksum(x[k, j, i] for j in all_nodes if i != j),
                name=f"flow_{k}_{i}"
            )

    # Each vehicle leaves origin and returns to destination exactly once
    for k in range(n_vehicles):
        model.addConstr(gp.quicksum(x[k, depot_start, j] for j in all_nodes if j != depot_start) == 1, name=f"leave_{k}")
        model.addConstr(gp.quicksum(x[k, i, depot_end] for i in all_nodes if i != depot_end) == 1, name=f"return_{k}")

    # Time constraints
    M = max(L) + max_route_dur + 1000.0
    for k in range(n_vehicles):
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    model.addConstr(
                        t[k, j] >= t[k, i] + service[i] + dist[i][j] - M * (1 - x[k, i, j]),
                        name=f"time_{k}_{i}_{j}"
                    )

    # Load constraints
    for k in range(n_vehicles):
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    model.addConstr(
                        q[k, j] >= q[k, i] + load_val[j] - Q * (1 - x[k, i, j]),
                        name=f"load_lower_{k}_{i}_{j}"
                    )
                    model.addConstr(
                        q[k, j] <= q[k, i] + load_val[j] + Q * (1 - x[k, i, j]),
                        name=f"load_upper_{k}_{i}_{j}"
                    )

    # Capacity and non-negativity
    for k in range(n_vehicles):
        for i in all_nodes:
            model.addConstr(q[k, i] >= max(0.0, float(load_val[i])), name=f"qmin_{k}_{i}")
            model.addConstr(q[k, i] <= min(float(Q), float(Q + load_val[i])), name=f"qmax_{k}_{i}")

    # Pickup before delivery on same vehicle + subtour elimination
    for k in range(n_vehicles):
        for p_idx, p in enumerate(pickups):
            d = pair[p]
            model.addConstr(
                gp.quicksum(x[k, p, j] for j in all_nodes if p != j) ==
                gp.quicksum(x[k, d, j] for j in all_nodes if d != j),
                name=f"sameveh_{k}_{p}"
            )
            # MTZ-like for precedence
            model.addConstr(
                t[k, d] >= t[k, p] + service[p] + dist[p][d] - M * (1 - gp.quicksum(x[k, p, j] for j in all_nodes if p != j)),
                name=f"precedence_{k}_{p}"
            )

    # Ride time constraints
    for p in pickups:
        d = pair[p]
        for k in range(n_vehicles):
            model.addConstr(
                t[k, d] - t[k, p] - service[p] <= max_ride + M * (1 - gp.quicksum(x[k, p, j] for j in all_nodes if p != j)),
                name=f"ride_upper_{k}_{p}"
            )
            model.addConstr(
                t[k, d] - t[k, p] - service[p] >= dist[p][d] - M * (1 - gp.quicksum(x[k, p, j] for j in all_nodes if p != j)),
                name=f"ride_lower_{k}_{p}"
            )

    # Route duration
    for k in range(n_vehicles):
        model.addConstr(t[k, depot_end] - t[k, depot_start] <= max_route_dur, name=f"duration_{k}")

    # Time windows
    for k in range(n_vehicles):
        for i in all_nodes:
            model.addConstr(t[k, i] >= E[i] - M * (1 - gp.quicksum(x[k, j, i] for j in all_nodes if j != i)),
                           name=f"tw_early_{k}_{i}")
            model.addConstr(t[k, i] <= L[i] + M * (1 - gp.quicksum(x[k, j, i] for j in all_nodes if j != i)),
                           name=f"tw_late_{k}_{i}")

    # Depot bounds
    for k in range(n_vehicles):
        t[k, depot_start].lb = E[depot_start]
        t[k, depot_end].ub = L[depot_end]

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger and obj_val < GRB.INFINITY:
                logger.log(float(obj_val))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        objective_value = float(model.objVal)

        # Build routes and times
        routes = {}
        service_times: Dict[str, float] = {}
        ride_times: Dict[str, float] = {}

        for k in range(n_vehicles):
            # Reconstruct route using x variables
            route = [depot_start]
            current = depot_start
            visited = set([depot_start, depot_end])
            
            while current != depot_end:
                found = False
                for j in all_nodes:
                    if current != j and abs(x[k, current, j].X - 1.0) < 0.5:
                        continue
                    if j not in visited:
                        route.append(j)
                        visited.add(j)
                        current = j
                        found = True
                        break
                if not found:
                    # Fallback: add depot_end if stuck
                    route.append(depot_end)
                    break
            routes[f"vehicle_{k}"] = route

            # Service times
            for i in route:
                val = float(t[k, i].X)
                if i == depot_start:
                    service_times[f"depot_start_{k}"] = val
                elif i == depot_end:
                    service_times[f"depot_end_{k}"] = val
                else:
                    service_times[str(i)] = val

        # Ride times - more robust lookup
        for p in pickups:
            user_id = user_from_pickup[p]
            d = pair[p]
            # Find vehicle that serves this pickup
            serving_k = None
            for k_cand in range(n_vehicles):
                if any(abs(x[k_cand, p, j].X) > 0.5 for j in all_nodes if p != j):
                    serving_k = k_cand
                    break
            if serving_k is not None:
                pickup_time = t[serving_k, p].X
                drop_time = t[serving_k, d].X
                ride = float(drop_time - pickup_time - service[p])
                ride = max(ride, dist[p][d])  # numerical safety
            else:
                ride = float(dist[p][d])
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
        # No feasible solution found
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