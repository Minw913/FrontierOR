import json
import argparse
import time
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Multi-Depot Vehicle Scheduling Problem")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n_trips = instance['parameters']['n_trips']
    m_depots = instance['parameters']['m_depots']
    vehicle_capacities = instance['vehicle_capacities']

    # Build lookup structures
    depot_to_trip_cost = {}
    for entry in instance['depot_to_trip_costs']:
        depot_to_trip_cost[(entry['depot'], entry['trip'])] = entry['cost']

    trip_to_depot_cost = {}
    for entry in instance['trip_to_depot_costs']:
        trip_to_depot_cost[(entry['trip'], entry['depot'])] = entry['cost']

    compat_arc_cost = {}
    compat_arcs_list = []
    for entry in instance['compatible_arcs']:
        compat_arc_cost[(entry['from_trip'], entry['to_trip'])] = entry['cost']
        compat_arcs_list.append((entry['from_trip'], entry['to_trip']))

    # Adjacency lists
    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    for (i, j) in compat_arcs_list:
        out_arcs[i].append(j)
        in_arcs[j].append(i)

    # Get trip start times for sorting (for heuristic)
    trip_start_time = {}
    trip_end_time = {}
    for t in instance['trips']:
        trip_start_time[t['trip_id']] = t['start_time']
        trip_end_time[t['trip_id']] = t['end_time']

    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("MDVSP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.OutputFlag = 0

    # Decision variables
    x = {}

    for k in range(1, m_depots + 1):
        # Depot to trip arcs
        for i in range(1, n_trips + 1):
            if (k, i) in depot_to_trip_cost:
                x[k, 0, i] = model.addVar(
                    vtype=GRB.BINARY,
                    obj=depot_to_trip_cost[(k, i)],
                    name=f"x_{k}_s_{i}"
                )

        # Trip to trip arcs
        for (i, j) in compat_arcs_list:
            x[k, i, j] = model.addVar(
                vtype=GRB.BINARY,
                obj=compat_arc_cost[(i, j)],
                name=f"x_{k}_{i}_{j}"
            )

        # Trip to depot arcs
        for i in range(1, n_trips + 1):
            if (i, k) in trip_to_depot_cost:
                x[k, i, n_trips + 1] = model.addVar(
                    vtype=GRB.BINARY,
                    obj=trip_to_depot_cost[(i, k)],
                    name=f"x_{k}_{i}_t"
                )

    model.update()

    # Constraint 1: Each trip covered exactly once
    for i in range(1, n_trips + 1):
        expr = gp.LinExpr()
        for k in range(1, m_depots + 1):
            if (k, 0, i) in x:
                expr += x[k, 0, i]
            for j in in_arcs[i]:
                if (k, j, i) in x:
                    expr += x[k, j, i]
        model.addConstr(expr == 1, name=f"cover_{i}")

    # Constraint 2: Flow conservation at each trip node for each depot
    for k in range(1, m_depots + 1):
        for i in range(1, n_trips + 1):
            flow_in = gp.LinExpr()
            flow_out = gp.LinExpr()

            if (k, 0, i) in x:
                flow_in += x[k, 0, i]
            for j in in_arcs[i]:
                if (k, j, i) in x:
                    flow_in += x[k, j, i]

            if (k, i, n_trips + 1) in x:
                flow_out += x[k, i, n_trips + 1]
            for j in out_arcs[i]:
                if (k, i, j) in x:
                    flow_out += x[k, i, j]

            model.addConstr(flow_in == flow_out, name=f"flow_{k}_{i}")

    # Constraint 3: Depot capacity
    for k in range(1, m_depots + 1):
        expr = gp.LinExpr()
        for i in range(1, n_trips + 1):
            if (k, 0, i) in x:
                expr += x[k, 0, i]
        model.addConstr(expr <= vehicle_capacities[k - 1], name=f"cap_{k}")

    model.update()

    # Callback for logging incumbent solutions
    logged_solutions = []

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            solution = extract_solution_cb(model, x, n_trips, m_depots, out_arcs)
            if solution is not None:
                solution["objective_value"] = obj_val
                logged_solutions.append(solution)
                if logger:
                    logger.log_solution(obj_val, solution)

    def extract_solution_cb(model, x, n_trips, m_depots, out_arcs):
        routes = []
        for k in range(1, m_depots + 1):
            starts = []
            for i in range(1, n_trips + 1):
                if (k, 0, i) in x:
                    try:
                        val = model.cbGetSolution(x[k, 0, i])
                        if val is not None and val > 0.5:
                            starts.append(i)
                    except Exception:
                        pass

            for start_trip in starts:
                route_trips = [start_trip]
                current = start_trip
                visited = {start_trip}
                while True:
                    found_next = False
                    for j in out_arcs[current]:
                        if (k, current, j) in x and j not in visited:
                            try:
                                val = model.cbGetSolution(x[k, current, j])
                                if val is not None and val > 0.5:
                                    route_trips.append(j)
                                    visited.add(j)
                                    current = j
                                    found_next = True
                                    break
                            except Exception:
                                pass
                    if not found_next:
                        break
                routes.append({"depot": k, "trips": route_trips})

        return {
            "objective_value": 0.0,
            "routes": routes,
            "num_vehicles": len(routes)
        }

    def extract_final_solution(model, x, n_trips, m_depots, out_arcs):
        routes = []
        for k in range(1, m_depots + 1):
            starts = []
            for i in range(1, n_trips + 1):
                if (k, 0, i) in x and x[k, 0, i].X > 0.5:
                    starts.append(i)

            for start_trip in starts:
                route_trips = [start_trip]
                current = start_trip
                visited = {start_trip}
                while True:
                    found_next = False
                    for j in out_arcs[current]:
                        if (k, current, j) in x and j not in visited:
                            if x[k, current, j].X > 0.5:
                                route_trips.append(j)
                                visited.add(j)
                                current = j
                                found_next = True
                                break
                    if not found_next:
                        break
                routes.append({"depot": k, "trips": route_trips})

        return {
            "objective_value": model.ObjVal,
            "routes": routes,
            "num_vehicles": len(routes)
        }

    # Solve
    model.optimize(callback)

    # Extract solution
    solution = None
    if model.SolCount > 0:
        try:
            solution = extract_final_solution(model, x, n_trips, m_depots, out_arcs)
        except Exception:
            if logged_solutions:
                solution = logged_solutions[-1]
    elif logged_solutions:
        solution = logged_solutions[-1]

    # If MIP failed, try greedy heuristic
    if solution is None:
        solution = greedy_heuristic(
            n_trips, m_depots, vehicle_capacities,
            depot_to_trip_cost, trip_to_depot_cost,
            compat_arc_cost, compat_arcs_list,
            out_arcs, in_arcs, trip_start_time
        )
        if solution and logger:
            logger.log_solution(solution["objective_value"], solution)

    # Write output
    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(solution["objective_value"], solution)


def greedy_heuristic(n_trips, m_depots, vehicle_capacities,
                     depot_to_trip_cost, trip_to_depot_cost,
                     compat_arc_cost, compat_arcs_list,
                     out_arcs, in_arcs, trip_start_time):
    """Greedy heuristic to construct a feasible solution."""
    # Sort trips by start time
    sorted_trips = sorted(range(1, n_trips + 1), key=lambda t: trip_start_time.get(t, 0))

    # Build routes greedily
    routes = []  # Each route: (depot, [list of trips])
    assigned = set()
    remaining_cap = list(vehicle_capacities)

    # Try to chain trips together
    # For each trip in order, try to append to an existing route
    route_last_trip = {}  # route_index -> last trip
    route_depot = {}  # route_index -> depot

    for trip in sorted_trips:
        if trip in assigned:
            continue

        # Try to append to an existing route
        best_route_idx = None
        best_append_cost = float('inf')

        for r_idx in range(len(routes)):
            last = route_last_trip[r_idx]
            if (last, trip) in compat_arc_cost:
                cost = compat_arc_cost[(last, trip)]
                if cost < best_append_cost:
                    best_append_cost = cost
                    best_route_idx = r_idx

        if best_route_idx is not None:
            routes[best_route_idx].append(trip)
            route_last_trip[best_route_idx] = trip
            assigned.add(trip)
        else:
            # Start a new route from the best depot
            best_depot = None
            best_cost = float('inf')
            for k in range(1, m_depots + 1):
                if remaining_cap[k - 1] > 0 and (k, trip) in depot_to_trip_cost:
                    cost = depot_to_trip_cost[(k, trip)]
                    if cost < best_cost:
                        best_cost = cost
                        best_depot = k

            if best_depot is not None:
                r_idx = len(routes)
                routes.append([trip])
                route_depot[r_idx] = best_depot
                route_last_trip[r_idx] = trip
                remaining_cap[best_depot - 1] -= 1
                assigned.add(trip)

    # Build solution
    result_routes = []
    obj = 0.0
    for r_idx, trips in enumerate(routes):
        depot = route_depot[r_idx]
        obj += depot_to_trip_cost.get((depot, trips[0]), 0)
        for idx in range(len(trips) - 1):
            obj += compat_arc_cost.get((trips[idx], trips[idx + 1]), 0)
        obj += trip_to_depot_cost.get((trips[-1], depot), 0)
        result_routes.append({"depot": depot, "trips": trips})

    return {
        "objective_value": obj,
        "routes": result_routes,
        "num_vehicles": len(result_routes)
    }


if __name__ == "__main__":
    main()