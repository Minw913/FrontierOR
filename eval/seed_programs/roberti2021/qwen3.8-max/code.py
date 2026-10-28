import json
import argparse
import sys
import time as time_module
from gurobipy import *


def main():
    parser = argparse.ArgumentParser(description='Truck-Drone Delivery Optimization')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = inst['num_customers']
    num_nodes = inst['num_nodes']
    truck_times = inst['truck_travel_times']
    drone_times = inst['drone_travel_times']

    depot_o = 0
    depot_d = n + 1
    customers = list(range(1, n + 1))

    def phys(i):
        return i if i <= n else 0

    def tt(i, j):
        return truck_times[phys(i)][phys(j)]

    def dt(i, j):
        return drone_times[phys(i)][phys(j)]

    start_time = time_module.time()

    # Try MIP
    solution = solve_mip(n, num_nodes, customers, depot_o, depot_d,
                         truck_times, drone_times, tt, dt,
                         args.time_limit, logger, start_time)

    if solution is None:
        solution = solve_heuristic(n, customers, depot_o, depot_d,
                                   truck_times, drone_times, tt, dt)

    if solution is not None:
        if logger:
            logger.log_solution(solution['objective_value'], solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)


def solve_mip(n, num_nodes, customers, depot_o, depot_d,
              truck_times, drone_times, tt, dt, time_limit, logger, start_time):
    # Build arcs
    truck_arcs = []
    drone_arcs = []
    for j in customers:
        truck_arcs.append((depot_o, j))
        drone_arcs.append((depot_o, j))
    for i in customers:
        for j in customers:
            if i != j:
                truck_arcs.append((i, j))
                drone_arcs.append((i, j))
    for i in customers:
        truck_arcs.append((i, depot_d))
        drone_arcs.append((i, depot_d))

    # Big-M computation
    max_time = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if truck_times[i][j] > max_time:
                max_time = truck_times[i][j]
            if drone_times[i][j] > max_time:
                max_time = drone_times[i][j]

    UB = (n + 1) * max_time
    M = UB + max_time + 1

    model = Model("truck_drone")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0

    # Decision variables
    xt = {}
    for (i, j) in truck_arcs:
        xt[i, j] = model.addVar(vtype=GRB.BINARY, name=f"xt_{i}_{j}")

    xd = {}
    for (i, j) in drone_arcs:
        xd[i, j] = model.addVar(vtype=GRB.BINARY, name=f"xd_{i}_{j}")

    u = {}
    v = {}
    w = {}
    for i in customers:
        u[i] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}")
        v[i] = model.addVar(vtype=GRB.BINARY, name=f"v_{i}")
        w[i] = model.addVar(vtype=GRB.BINARY, name=f"w_{i}")

    t = {}
    for i in range(n + 2):
        t[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"t_{i}")

    model.update()

    # Objective: minimize arrival time at depot destination
    model.setObjective(t[depot_d], GRB.MINIMIZE)

    # Each customer must be served
    for i in customers:
        model.addConstr(u[i] + v[i] >= 1, name=f"serve_{i}")

    # Combined indicator linearization
    for i in customers:
        model.addConstr(w[i] <= u[i], name=f"wlu_{i}")
        model.addConstr(w[i] <= v[i], name=f"wlv_{i}")
        model.addConstr(w[i] >= u[i] + v[i] - 1, name=f"wge_{i}")

    # Truck flow conservation
    model.addConstr(quicksum(xt[depot_o, j] for j in customers) == 1, name="truck_dep")
    model.addConstr(quicksum(xt[i, depot_d] for i in customers) == 1, name="truck_arr")

    for i in customers:
        preds = [depot_o] + [k for k in customers if k != i]
        succs = [depot_d] + [k for k in customers if k != i]
        model.addConstr(quicksum(xt[j, i] for j in preds) == u[i], name=f"tin_{i}")
        model.addConstr(quicksum(xt[i, j] for j in succs) == u[i], name=f"tout_{i}")

    # Drone flow conservation
    model.addConstr(quicksum(xd[depot_o, j] for j in customers) == 1, name="drone_dep")
    model.addConstr(quicksum(xd[i, depot_d] for i in customers) == 1, name="drone_arr")

    for i in customers:
        preds = [depot_o] + [k for k in customers if k != i]
        succs = [depot_d] + [k for k in customers if k != i]
        model.addConstr(quicksum(xd[j, i] for j in preds) == v[i], name=f"din_{i}")
        model.addConstr(quicksum(xd[i, j] for j in succs) == v[i], name=f"dout_{i}")

    # Arrival time constraints (MTZ-style subtour elimination)
    for (i, j) in truck_arcs:
        c = tt(i, j)
        model.addConstr(t[j] >= t[i] + c - M * (1 - xt[i, j]), name=f"tt_{i}_{j}")

    for (i, j) in drone_arcs:
        c = dt(i, j)
        model.addConstr(t[j] >= t[i] + c - M * (1 - xd[i, j]), name=f"dtc_{i}_{j}")

    # Drone pair constraint: drone arcs between customers limited by combined status
    for i in customers:
        for j in customers:
            if i < j:
                model.addConstr(xd[i, j] + xd[j, i] <= w[i] + w[j], name=f"dp_{i}_{j}")

    # Drone cannot do depot -> single customer -> depot
    for i in customers:
        model.addConstr(xd[depot_o, i] + xd[i, depot_d] <= 1, name=f"dnd_{i}")

    # Depot origin time is 0
    model.addConstr(t[depot_o] == 0, name="t0")

    # Valid inequalities
    model.addConstr(quicksum(v[i] for i in customers) >= 2, name="drone_min")
    model.addConstr(quicksum(u[i] for i in customers) >= 1, name="truck_min")

    model.update()

    # Callback for logging incumbent solutions
    logged_objectives = set()

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            obj_rounded = round(obj, 6)
            if obj_rounded not in logged_objectives:
                logged_objectives.add(obj_rounded)
                try:
                    xt_v = {k: mdl.cbGetSolution(var) for k, var in xt.items()}
                    xd_v = {k: mdl.cbGetSolution(var) for k, var in xd.items()}
                    u_v = {k: mdl.cbGetSolution(var) for k, var in u.items()}
                    v_v = {k: mdl.cbGetSolution(var) for k, var in v.items()}
                    t_v = {k: mdl.cbGetSolution(var) for k, var in t.items()}
                    sol = build_solution(n, customers, depot_o, depot_d,
                                         xt_v, xd_v, u_v, v_v, t_v)
                    if sol is not None and logger:
                        logger.log_solution(obj, sol)
                except Exception:
                    if logger:
                        logger.log(obj)

    model.optimize(callback)

    if model.SolCount > 0:
        xt_v = {k: var.X for k, var in xt.items()}
        xd_v = {k: var.X for k, var in xd.items()}
        u_v = {k: var.X for k, var in u.items()}
        v_v = {k: var.X for k, var in v.items()}
        t_v = {k: var.X for k, var in t.items()}
        return build_solution(n, customers, depot_o, depot_d, xt_v, xd_v, u_v, v_v, t_v)

    return None


def build_solution(n, customers, depot_o, depot_d, xt_v, xd_v, u_v, v_v, t_v):
    """Build solution dictionary from variable values."""
    obj_val = t_v.get(depot_d, 0)

    # Extract truck route by following arcs
    truck_route = [depot_o]
    current = depot_o
    visited = {depot_o}
    max_iter = n + 2
    for _ in range(max_iter):
        if current == depot_d:
            break
        next_node = None
        for j in customers + [depot_d]:
            if (current, j) in xt_v and xt_v[current, j] > 0.5:
                next_node = j
                break
        if next_node is None:
            break
        truck_route.append(next_node)
        visited.add(next_node)
        current = next_node

    # Extract drone route by following arcs
    drone_route = [depot_o]
    current = depot_o
    visited = {depot_o}
    for _ in range(max_iter):
        if current == depot_d:
            break
        next_node = None
        for j in customers + [depot_d]:
            if (current, j) in xd_v and xd_v[current, j] > 0.5:
                next_node = j
                break
        if next_node is None:
            break
        drone_route.append(next_node)
        visited.add(next_node)
        current = next_node

    # Customer types
    customer_types = {}
    for i in customers:
        ui = round(u_v.get(i, 0))
        vi = round(v_v.get(i, 0))
        if ui == 1 and vi == 1:
            customer_types[str(i)] = "combined"
        elif ui == 1:
            customer_types[str(i)] = "truck"
        else:
            customer_types[str(i)] = "drone"

    # Arrival times
    arrival_times = {}
    for i in range(n + 2):
        val = t_v.get(i, 0)
        arrival_times[str(i)] = round(float(val), 6) if val is not None else 0.0

    return {
        "objective_value": float(obj_val),
        "truck_route": truck_route,
        "drone_route": drone_route,
        "customer_types": customer_types,
        "arrival_times": arrival_times
    }


def solve_heuristic(n, customers, depot_o, depot_d, truck_times, drone_times, tt, dt):
    """Fallback heuristic: try different drone customer pairs and truck routes."""
    best_solution = None
    best_obj = float('inf')

    # Try all pairs of drone customers
    for i in customers:
        for j in customers:
            if i >= j:
                continue

            # Try both drone orderings
            for drone_cust_order in [(i, j), (j, i)]:
                dc1, dc2 = drone_cust_order

                # Drone route: depot -> dc1 -> dc2 -> depot
                drone_route = [depot_o, dc1, dc2, depot_d]

                # Truck customers: all customers
                # Use nearest neighbor for truck route
                truck_custs = list(customers)
                truck_route = nearest_neighbor_route(depot_o, depot_d, truck_custs, tt)

                # Customer types
                customer_types = {}
                for c in customers:
                    if c == dc1 or c == dc2:
                        customer_types[str(c)] = "combined"
                    else:
                        customer_types[str(c)] = "truck"

                # Compute arrival times
                arrival = compute_arrival_times(truck_route, drone_route, tt, dt, n, depot_d)
                obj = arrival[depot_d]

                if obj < best_obj:
                    best_obj = obj
                    arrival_times = {str(k): round(float(v), 6) for k, v in arrival.items()}
                    best_solution = {
                        "objective_value": float(obj),
                        "truck_route": truck_route,
                        "drone_route": drone_route,
                        "customer_types": customer_types,
                        "arrival_times": arrival_times
                    }

    # Also try with more drone customers if n >= 3
    if n >= 3:
        for i in customers:
            for j in customers:
                if i >= j:
                    continue
                for k in customers:
                    if k == i or k == j:
                        continue
                    # Drone visits i, j, k (all combined)
                    # Try order i -> j -> k
                    for perm in [(i, j, k), (i, k, j), (j, i, k)]:
                        drone_route = [depot_o] + list(perm) + [depot_d]
                        truck_custs = list(customers)
                        truck_route = nearest_neighbor_route(depot_o, depot_d, truck_custs, tt)

                        customer_types = {}
                        drone_set = set(perm)
                        for c in customers:
                            if c in drone_set:
                                customer_types[str(c)] = "combined"
                            else:
                                customer_types[str(c)] = "truck"

                        arrival = compute_arrival_times(truck_route, drone_route, tt, dt, n, depot_d)
                        obj = arrival[depot_d]

                        if obj < best_obj:
                            best_obj = obj
                            arrival_times = {str(k2): round(float(v), 6) for k2, v in arrival.items()}
                            best_solution = {
                                "objective_value": float(obj),
                                "truck_route": truck_route,
                                "drone_route": drone_route,
                                "customer_types": customer_types,
                                "arrival_times": arrival_times
                            }

    return best_solution


def nearest_neighbor_route(depot_o, depot_d, custs, tt):
    """Build a nearest-neighbor route for the truck."""
    if not custs:
        return [depot_o, depot_d]

    route = [depot_o]
    unvisited = list(custs)
    current = depot_o

    while unvisited:
        nearest = min(unvisited, key=lambda j: tt(current, j))
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    route.append(depot_d)
    return route


def compute_arrival_times(truck_route, drone_route, tt, dt, n, depot_d):
    """Compute arrival times given truck and drone routes."""
    arrival = [0.0] * (n + 2)

    # Truck arrival times
    for idx in range(len(truck_route) - 1):
        i = truck_route[idx]
        j = truck_route[idx + 1]
        new_time = arrival[i] + tt(i, j)
        if new_time > arrival[j]:
            arrival[j] = new_time

    # Drone arrival times
    for idx in range(len(drone_route) - 1):
        i = drone_route[idx]
        j = drone_route[idx + 1]
        new_time = arrival[i] + dt(i, j)
        if new_time > arrival[j]:
            arrival[j] = new_time

    return arrival


if __name__ == '__main__':
    main()