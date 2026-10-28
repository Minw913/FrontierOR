import argparse
import json
import math
import time
from typing import Dict, List, Tuple, Set

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_extended_travel_times(
    n_customers: int,
    truck_mat: List[List[float]],
    drone_mat: List[List[float]],
):
    """
    Extended node set:
      0               : depot origin
      1..n_customers  : customers
      n_customers + 1 : depot destination (duplicate of depot)
    """
    sink = n_customers + 1

    def t_truck(i: int, j: int) -> float:
        bi = 0 if i == 0 else i
        bj = 0 if j == sink else j
        return truck_mat[bi][bj]

    def t_drone(i: int, j: int) -> float:
        bi = 0 if i == 0 else i
        bj = 0 if j == sink else j
        return drone_mat[bi][bj]

    return sink, t_truck, t_drone


def nearest_neighbor_truck_route(n: int, truck_mat: List[List[float]]) -> List[int]:
    unvisited = set(range(1, n + 1))
    route = [0]
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: truck_mat[cur][j] if cur != 0 else truck_mat[0][j])
        route.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    route.append(0)
    return route


def compute_arrival_times_from_routes(
    n: int,
    truck_route: List[int],
    drone_route: List[int],
    truck_mat: List[List[float]],
    drone_mat: List[List[float]],
) -> Dict[int, float]:
    """
    Computes minimal feasible shared arrival times t over extended nodes
    for fixed truck/drone routes by longest-path relaxation on precedence edges.
    """
    sink = n + 1
    edges = []

    def truck_time(i, j):
        bi = 0 if i == 0 else i
        bj = 0 if j == sink else j
        return truck_mat[bi][bj]

    def drone_time(i, j):
        bi = 0 if i == 0 else i
        bj = 0 if j == sink else j
        return drone_mat[bi][bj]

    for k in range(len(truck_route) - 1):
        i = truck_route[k]
        j = truck_route[k + 1]
        jj = sink if (j == 0 and k + 1 == len(truck_route) - 1) else j
        edges.append((i, jj, truck_time(i, jj)))

    for k in range(len(drone_route) - 1):
        i = drone_route[k]
        j = drone_route[k + 1]
        jj = sink if (j == 0 and k + 1 == len(drone_route) - 1) else j
        edges.append((i, jj, drone_time(i, jj)))

    t = {i: 0.0 for i in range(0, sink + 1)}
    # Longest-path relaxation (works if no positive cycle; should hold for consistent routes)
    for _ in range(sink + 2):
        updated = False
        for i, j, w in edges:
            nv = t[i] + w
            if nv > t[j] + 1e-12:
                t[j] = nv
                updated = True
        if not updated:
            break
    return t


def heuristic_solution(instance: dict):
    n = instance["num_customers"]
    truck_mat = instance["truck_travel_times"]
    drone_mat = instance["drone_travel_times"]

    truck_route = nearest_neighbor_truck_route(n, truck_mat)

    # Ensure at least two combined customers for drone route feasibility
    c1, c2 = truck_route[1], truck_route[2]
    drone_route = [0, c1, c2, 0]

    customer_types = {}
    for i in range(1, n + 1):
        if i == c1 or i == c2:
            customer_types[i] = "combined"
        else:
            customer_types[i] = "truck"

    arr = compute_arrival_times_from_routes(n, truck_route, drone_route, truck_mat, drone_mat)
    obj = arr[n + 1]

    return {
        "objective_value": float(obj),
        "truck_route": truck_route,
        "drone_route": drone_route,
        "customer_types": customer_types,
        "arrival_times": arr,
    }


def route_from_arcs(arcs_used: Set[Tuple[int, int]], sink: int) -> List[int]:
    out = {}
    for i, j in arcs_used:
        out[i] = j

    route = [0]
    cur = 0
    seen = set([0])
    max_steps = len(arcs_used) + 5
    steps = 0
    while cur != sink and steps <= max_steps:
        if cur not in out:
            break
        nxt = out[cur]
        route.append(nxt)
        cur = nxt
        steps += 1
        if cur in seen and cur != sink:
            break
        seen.add(cur)

    # Convert sink back to depot id 0 for output
    route = [0 if x == sink else x for x in route]
    if route[-1] != 0:
        route.append(0)
    return route


def solve(instance: dict, time_limit: int, logger: SolutionLogger = None):
    n = instance["num_customers"]
    truck_mat = instance["truck_travel_times"]
    drone_mat = instance["drone_travel_times"]

    customers = list(range(1, n + 1))
    sink, t_truck, t_drone = build_extended_travel_times(n, truck_mat, drone_mat)

    truck_arcs = []
    drone_arcs = []

    for j in customers:
        truck_arcs.append((0, j))
        drone_arcs.append((0, j))
    for i in customers:
        for j in customers:
            if i != j:
                truck_arcs.append((i, j))
                drone_arcs.append((i, j))
    for i in customers:
        truck_arcs.append((i, sink))
        drone_arcs.append((i, sink))

    max_arc = 0.0
    for row in truck_mat:
        max_arc = max(max_arc, max(row))
    for row in drone_mat:
        max_arc = max(max_arc, max(row))
    M = (n + 2) * max_arc + 1.0
    if M < 1.0:
        M = 1.0

    # Heuristic fallback + MIP start
    h = heuristic_solution(instance)
    best_logged = [float("inf")]
    if logger is not None:
        logger.log(h["objective_value"])
        best_logged[0] = h["objective_value"]

    model = gp.Model("truck_drone_synchronized")
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, int(time_limit))
    model.Params.OutputFlag = 0

    x = model.addVars(truck_arcs, vtype=GRB.BINARY, name="x")  # truck arcs
    y = model.addVars(drone_arcs, vtype=GRB.BINARY, name="y")  # drone arcs

    u = model.addVars(customers, vtype=GRB.BINARY, name="u_truck_only")
    v = model.addVars(customers, vtype=GRB.BINARY, name="v_drone_only")
    w = model.addVars(customers, vtype=GRB.BINARY, name="w_combined")

    t = model.addVars(range(0, sink + 1), lb=0.0, ub=M, vtype=GRB.CONTINUOUS, name="t")
    model.addConstr(t[0] == 0.0, name="depot_origin_time")

    # Customer type partition
    for i in customers:
        model.addConstr(u[i] + v[i] + w[i] == 1, name=f"type_partition_{i}")

    # Depart/return exactly once for truck and drone
    model.addConstr(gp.quicksum(x[0, j] for j in customers) == 1, name="truck_depart_once")
    model.addConstr(gp.quicksum(x[i, sink] for i in customers) == 1, name="truck_return_once")
    model.addConstr(gp.quicksum(y[0, j] for j in customers) == 1, name="drone_depart_once")
    model.addConstr(gp.quicksum(y[i, sink] for i in customers) == 1, name="drone_return_once")

    for i in customers:
        truck_out = gp.quicksum(x[i, j] for j in customers if j != i) + x[i, sink]
        truck_in = x[0, i] + gp.quicksum(x[j, i] for j in customers if j != i)

        drone_out = gp.quicksum(y[i, j] for j in customers if j != i) + y[i, sink]
        drone_in = y[0, i] + gp.quicksum(y[j, i] for j in customers if j != i)

        model.addConstr(truck_out == u[i] + w[i], name=f"truck_out_{i}")
        model.addConstr(truck_in == u[i] + w[i], name=f"truck_in_{i}")

        model.addConstr(drone_out == v[i] + w[i], name=f"drone_out_{i}")
        model.addConstr(drone_in == v[i] + w[i], name=f"drone_in_{i}")

    # Drone pairwise customer arc restriction
    for idx_i in range(len(customers)):
        i = customers[idx_i]
        for idx_j in range(idx_i + 1, len(customers)):
            j = customers[idx_j]
            model.addConstr(y[i, j] + y[j, i] <= w[i] + w[j], name=f"drone_pair_{i}_{j}")

    # Drone cannot do single-customer out-and-back
    for i in customers:
        model.addConstr(y[0, i] + y[i, sink] <= 1, name=f"drone_no_single_outback_{i}")

    # Time precedence constraints on used arcs
    for i, j in truck_arcs:
        model.addConstr(
            t[j] >= t[i] + t_truck(i, j) - M * (1 - x[i, j]),
            name=f"time_truck_{i}_{j}",
        )
    for i, j in drone_arcs:
        model.addConstr(
            t[j] >= t[i] + t_drone(i, j) - M * (1 - y[i, j]),
            name=f"time_drone_{i}_{j}",
        )

    model.setObjective(t[sink], GRB.MINIMIZE)

    # MIP start from heuristic
    h_types = h["customer_types"]
    h_truck_route = h["truck_route"]
    h_drone_route = h["drone_route"]
    h_arr = h["arrival_times"]

    truck_start_arcs = set()
    for k in range(len(h_truck_route) - 1):
        i = h_truck_route[k]
        j = h_truck_route[k + 1]
        jj = sink if (j == 0 and k + 1 == len(h_truck_route) - 1) else j
        truck_start_arcs.add((i, jj))
    drone_start_arcs = set()
    for k in range(len(h_drone_route) - 1):
        i = h_drone_route[k]
        j = h_drone_route[k + 1]
        jj = sink if (j == 0 and k + 1 == len(h_drone_route) - 1) else j
        drone_start_arcs.add((i, jj))

    for a in truck_arcs:
        x[a].Start = 1.0 if a in truck_start_arcs else 0.0
    for a in drone_arcs:
        y[a].Start = 1.0 if a in drone_start_arcs else 0.0

    for i in customers:
        typ = h_types[i]
        u[i].Start = 1.0 if typ == "truck" else 0.0
        v[i].Start = 1.0 if typ == "drone" else 0.0
        w[i].Start = 1.0 if typ == "combined" else 0.0

    for node in range(0, sink + 1):
        t[node].Start = float(h_arr.get(node, 0.0))

    def callback(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged[0]:
                best_logged[0] = obj
                logger.log(float(obj))

    model.optimize(callback)

    # Extract best solution if available; else fallback heuristic
    if model.SolCount > 0:
        x_used = {(i, j) for (i, j) in truck_arcs if x[i, j].X > 0.5}
        y_used = {(i, j) for (i, j) in drone_arcs if y[i, j].X > 0.5}

        truck_route = route_from_arcs(x_used, sink)
        drone_route = route_from_arcs(y_used, sink)

        customer_types = {}
        for i in customers:
            ui = u[i].X
            vi = v[i].X
            wi = w[i].X
            if wi >= ui and wi >= vi:
                customer_types[str(i)] = "combined"
            elif ui >= vi:
                customer_types[str(i)] = "truck"
            else:
                customer_types[str(i)] = "drone"

        arrival_times = {str(node): float(t[node].X) for node in range(0, sink + 1)}
        obj = float(model.ObjVal)

        sol = {
            "objective_value": obj,
            "truck_route": [int(vv) for vv in truck_route],
            "drone_route": [int(vv) for vv in drone_route],
            "customer_types": customer_types,
            "arrival_times": arrival_times,
        }
    else:
        # Fallback to heuristic
        customer_types = {str(k): v for k, v in h["customer_types"].items()}
        arrival_times = {str(k): float(v) for k, v in h["arrival_times"].items()}
        sol = {
            "objective_value": float(h["objective_value"]),
            "truck_route": [int(vv) for vv in h["truck_route"]],
            "drone_route": [int(vv) for vv in h["drone_route"]],
            "customer_types": customer_types,
            "arrival_times": arrival_times,
        }

    return sol


def main():
    parser = argparse.ArgumentParser(description="Truck-Drone synchronized routing solver")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to JSONL log file (optional)")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start = time.time()
    remaining = max(1, int(args.time_limit - (time.time() - start)))
    solution = solve(instance, remaining, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()