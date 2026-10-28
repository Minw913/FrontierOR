import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def route_cost(route, depot, truck_times):
    if not route:
        return float(truck_times[depot][depot])

    total = float(truck_times[depot][route[0]])
    for a, b in zip(route, route[1:]):
        total += float(truck_times[a][b])
    total += float(truck_times[route[-1]][depot])
    return total


def nearest_neighbor_route(customers, depot, truck_times):
    remaining = set(customers)
    route = []
    current = depot

    while remaining:
        nxt = min(
            remaining,
            key=lambda j: (truck_times[current][j], j),
        )
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt

    return route


def improve_route_2opt(route, depot, truck_times, deadline):
    route = list(route)
    current_cost = route_cost(route, depot, truck_times)

    # Exact directed-cost evaluation, limited to a few passes.
    for _ in range(3):
        if time.monotonic() >= deadline or len(route) < 3:
            break

        best_cost = current_cost
        best_route = None
        n = len(route)

        for left in range(n - 1):
            if time.monotonic() >= deadline:
                break
            for right in range(left + 1, n):
                candidate = (
                    route[:left]
                    + list(reversed(route[left:right + 1]))
                    + route[right + 1:]
                )
                candidate_cost = route_cost(candidate, depot, truck_times)
                if candidate_cost < best_cost - 1e-9:
                    best_cost = candidate_cost
                    best_route = candidate

        if best_route is None:
            break

        route = best_route
        current_cost = best_cost

    return route


def make_solution_from_arcs(
    arcs,
    drone_assignments,
    depot,
    sink,
    truck_times,
    drone_times,
):
    normalized_arcs = [[int(i), int(j)] for i, j in arcs]
    normalized_assignments = [
        {
            "drone": int(d),
            "dispatch_node": int(i),
            "delivery_node": int(j),
        }
        for d, i, j in drone_assignments
    ]

    visited = {depot}
    for i, j in arcs:
        if i != sink:
            visited.add(i)
        if j != sink:
            visited.add(j)

    drone_loads = {}
    for d, i, j in drone_assignments:
        key = (int(i), int(d))
        drone_loads[key] = drone_loads.get(key, 0.0) + float(
            drone_times[d][i][j]
        )

    waiting_times = {}
    for i in sorted(visited):
        longest = 0.0
        for d, dispatch in ((d, node) for node, d in drone_loads if node == i):
            longest = max(longest, drone_loads[(dispatch, d)])
        waiting_times[str(i)] = float(longest)

    truck_total = 0.0
    for i, j in arcs:
        destination = depot if j == sink else j
        truck_total += float(truck_times[i][destination])

    objective = truck_total + sum(waiting_times.values())

    normalized_arcs.sort(key=lambda arc: (arc[0], arc[1]))
    normalized_assignments.sort(
        key=lambda item: (
            item["dispatch_node"],
            item["drone"],
            item["delivery_node"],
        )
    )

    return {
        "objective_value": float(objective),
        "truck_route": normalized_arcs,
        "drone_assignments": normalized_assignments,
        "waiting_times": waiting_times,
    }


def make_solution_from_order(
    route,
    assignments,
    depot,
    sink,
    truck_times,
    drone_times,
):
    arcs = []
    if route:
        arcs.append((depot, route[0]))
        arcs.extend(zip(route, route[1:]))
        arcs.append((route[-1], sink))
    else:
        arcs.append((depot, sink))

    return make_solution_from_arcs(
        arcs,
        assignments,
        depot,
        sink,
        truck_times,
        drone_times,
    )


def greedy_drone_improvement(
    route,
    assignments,
    depot,
    sink,
    truck_times,
    battery_consumption,
    drone_times,
    battery_capacities,
    deadline,
    incumbent_callback,
):
    route = list(route)
    assignments = list(assignments)
    num_drones = len(battery_capacities)

    battery_used = [0.0] * num_drones
    loads = {depot: [0.0] * num_drones}
    protected_launch_nodes = set()

    for d, i, j in assignments:
        battery_used[d] += float(battery_consumption[d][i][j])
        loads.setdefault(i, [0.0] * num_drones)
        loads[i][d] += float(drone_times[d][i][j])
        if i != depot:
            protected_launch_nodes.add(i)

    while route and time.monotonic() < deadline:
        best = None
        best_improvement = 1e-9

        route_nodes = [depot] + route
        route_index = {node: idx for idx, node in enumerate(route)}

        for j in list(route):
            if time.monotonic() >= deadline:
                break
            if j in protected_launch_nodes:
                continue

            pos = route_index[j]
            predecessor = depot if pos == 0 else route[pos - 1]
            successor = depot if pos == len(route) - 1 else route[pos + 1]

            saving = (
                float(truck_times[predecessor][j])
                + float(truck_times[j][successor])
                - float(truck_times[predecessor][successor])
            )

            for dispatch in route_nodes:
                if dispatch == j:
                    continue

                dispatch_loads = loads.get(dispatch)
                if dispatch_loads is None:
                    dispatch_loads = [0.0] * num_drones

                old_wait = max(dispatch_loads) if dispatch_loads else 0.0

                for d in range(num_drones):
                    energy = float(battery_consumption[d][dispatch][j])
                    if battery_used[d] + energy > float(battery_capacities[d]) + 1e-9:
                        continue

                    duration = float(drone_times[d][dispatch][j])
                    new_drone_load = dispatch_loads[d] + duration
                    new_wait = max(old_wait, new_drone_load)
                    added_wait = new_wait - old_wait
                    improvement = saving - added_wait

                    if improvement > best_improvement + 1e-9:
                        best_improvement = improvement
                        best = (j, dispatch, d, energy, duration)

        if best is None:
            break

        j, dispatch, d, energy, duration = best
        route.remove(j)
        assignments.append((d, dispatch, j))
        battery_used[d] += energy
        loads.setdefault(dispatch, [0.0] * num_drones)
        loads[dispatch][d] += duration

        if dispatch != depot:
            protected_launch_nodes.add(dispatch)

        incumbent_callback(route, assignments)

    return route, assignments


def main():
    args = parse_args()
    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    num_nodes = int(instance["num_nodes"])
    customers = [int(j) for j in instance["customer_indices"]]
    depot = int(instance["depot_index"])
    sink = num_nodes

    params = instance["parameters"]
    num_drones = int(params["num_drones"])
    battery_capacities = [
        float(v) for v in params["drone_battery_capacities"]
    ]

    truck_times = instance["truck_travel_times"]
    battery_consumption = instance["drone_battery_consumption"]
    drone_times = instance["drone_delivery_time"]

    best_solution = None
    best_value = math.inf

    def record_solution(solution):
        nonlocal best_solution, best_value
        value = float(solution["objective_value"])
        if value < best_value - 1e-7:
            best_value = value
            best_solution = solution
            if logger:
                logger.log_solution(value, solution)

    # Always establish a robust truck-only feasible incumbent.
    route = nearest_neighbor_route(customers, depot, truck_times)
    assignments = []
    record_solution(
        make_solution_from_order(
            route, assignments, depot, sink, truck_times, drone_times
        )
    )

    heuristic_deadline = min(
        start_time + max(0.0, 0.15 * max(0, args.time_limit)),
        time.monotonic() + 2.0,
    )

    if len(route) <= 180 and time.monotonic() < heuristic_deadline:
        improved_route = improve_route_2opt(
            route, depot, truck_times, heuristic_deadline
        )
        improved_solution = make_solution_from_order(
            improved_route,
            assignments,
            depot,
            sink,
            truck_times,
            drone_times,
        )
        record_solution(improved_solution)
        route = improved_route

    if num_drones > 0 and time.monotonic() < heuristic_deadline:
        route, assignments = greedy_drone_improvement(
            route=route,
            assignments=assignments,
            depot=depot,
            sink=sink,
            truck_times=truck_times,
            battery_consumption=battery_consumption,
            drone_times=drone_times,
            battery_capacities=battery_capacities,
            deadline=heuristic_deadline,
            incumbent_callback=lambda r, a: record_solution(
                make_solution_from_order(
                    r, a, depot, sink, truck_times, drone_times
                )
            ),
        )

    remaining_time = args.time_limit - (time.monotonic() - start_time)

    if remaining_time > 0.05:
        try:
            model = gp.Model()
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, remaining_time)

            physical_route_nodes = [depot] + customers

            arc_keys = []
            for i in physical_route_nodes:
                for j in customers:
                    if i != j:
                        arc_keys.append((i, j))
                arc_keys.append((i, sink))

            x = model.addVars(arc_keys, vtype=GRB.BINARY)
            z = model.addVars(customers, vtype=GRB.BINARY)
            u = model.addVars(
                customers,
                lb=0.0,
                ub=max(1, len(customers)),
                vtype=GRB.CONTINUOUS,
            )
            wait = model.addVars(physical_route_nodes, lb=0.0)

            y_keys = []
            for d in range(num_drones):
                capacity = battery_capacities[d]
                for i in physical_route_nodes:
                    for j in customers:
                        if i == j:
                            continue
                        if float(battery_consumption[d][i][j]) <= capacity + 1e-9:
                            y_keys.append((d, i, j))

            y = model.addVars(y_keys, vtype=GRB.BINARY)

            # Start and virtual-sink degree constraints.
            model.addConstr(
                gp.quicksum(x[depot, j] for j in customers + [sink]) == 1
            )
            model.addConstr(
                gp.quicksum(x[i, sink] for i in physical_route_nodes) == 1
            )

            # Truck visit degrees.
            for j in customers:
                incoming = gp.quicksum(
                    x[i, j] for i in physical_route_nodes if i != j
                )
                outgoing = gp.quicksum(
                    x[j, k] for k in customers + [sink] if k != j
                )
                model.addConstr(incoming == z[j])
                model.addConstr(outgoing == z[j])

            # MTZ subtour elimination.
            n_customers = max(1, len(customers))
            for j in customers:
                model.addConstr(u[j] >= z[j])
                model.addConstr(u[j] <= n_customers * z[j])
                model.addConstr(u[j] >= x[depot, j])

            for i in customers:
                for j in customers:
                    if i != j:
                        model.addConstr(
                            u[j]
                            >= u[i] + 1
                            - n_customers * (1 - x[i, j])
                        )

            y_by_customer = {j: [] for j in customers}
            y_by_drone = {d: [] for d in range(num_drones)}
            y_by_launch_drone = {
                (d, i): [] for d in range(num_drones)
                for i in physical_route_nodes
            }

            for key in y_keys:
                d, i, j = key
                y_by_customer[j].append(y[key])
                y_by_drone[d].append((key, y[key]))
                y_by_launch_drone[d, i].append((key, y[key]))

            # Every customer is served exactly once.
            for j in customers:
                model.addConstr(
                    z[j] + gp.quicksum(y_by_customer[j]) == 1
                )

            # Drone launches are allowed only from truck-visited nodes.
            for d in range(num_drones):
                for i in physical_route_nodes:
                    launch_vars = y_by_launch_drone[d, i]
                    if not launch_vars:
                        continue
                    visit_expression = 1 if i == depot else z[i]
                    model.addConstr(
                        gp.quicksum(var for _, var in launch_vars)
                        <= len(customers) * visit_expression
                    )

            # Global battery budget of each drone.
            for d in range(num_drones):
                model.addConstr(
                    gp.quicksum(
                        float(battery_consumption[key[0]][key[1]][key[2]]) * var
                        for key, var in y_by_drone[d]
                    )
                    <= battery_capacities[d]
                )

            # Waiting is the maximum workload of any single drone at a node.
            for i in physical_route_nodes:
                for d in range(num_drones):
                    launch_vars = y_by_launch_drone[d, i]
                    if launch_vars:
                        model.addConstr(
                            wait[i]
                            >= gp.quicksum(
                                float(drone_times[key[0]][key[1]][key[2]]) * var
                                for key, var in launch_vars
                            )
                        )

            truck_objective = gp.quicksum(
                float(
                    truck_times[i][depot if j == sink else j]
                ) * x[i, j]
                for i, j in arc_keys
            )
            waiting_objective = gp.quicksum(
                wait[i] for i in physical_route_nodes
            )
            model.setObjective(
                truck_objective + waiting_objective,
                GRB.MINIMIZE,
            )

            # Warm-start from the best heuristic solution.
            selected_start_arcs = {
                tuple(arc) for arc in best_solution["truck_route"]
            }
            selected_start_assignments = {
                (
                    int(item["drone"]),
                    int(item["dispatch_node"]),
                    int(item["delivery_node"]),
                )
                for item in best_solution["drone_assignments"]
            }
            start_visited = {depot}
            order_position = {}
            current = depot
            position = 1
            successor = {i: j for i, j in selected_start_arcs}
            seen = set()

            while current in successor and current not in seen:
                seen.add(current)
                nxt = successor[current]
                if nxt == sink:
                    break
                start_visited.add(nxt)
                order_position[nxt] = position
                position += 1
                current = nxt

            for key in arc_keys:
                x[key].Start = 1.0 if key in selected_start_arcs else 0.0
            for j in customers:
                z[j].Start = 1.0 if j in start_visited else 0.0
                u[j].Start = float(order_position.get(j, 0))
            for key in y_keys:
                y[key].Start = (
                    1.0 if key in selected_start_assignments else 0.0
                )
            for i in physical_route_nodes:
                wait[i].Start = float(
                    best_solution["waiting_times"].get(str(i), 0.0)
                )

            x_keys_list = list(arc_keys)
            x_vars_list = [x[key] for key in x_keys_list]
            y_keys_list = list(y_keys)
            y_vars_list = [y[key] for key in y_keys_list]

            def callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    x_values = cb_model.cbGetSolution(x_vars_list)
                    y_values = cb_model.cbGetSolution(y_vars_list)

                    selected_arcs = [
                        key
                        for key, value in zip(x_keys_list, x_values)
                        if value > 0.5
                    ]
                    selected_assignments = [
                        key
                        for key, value in zip(y_keys_list, y_values)
                        if value > 0.5
                    ]

                    solution = make_solution_from_arcs(
                        selected_arcs,
                        selected_assignments,
                        depot,
                        sink,
                        truck_times,
                        drone_times,
                    )
                    record_solution(solution)
                except Exception:
                    # Logging must never interrupt optimization.
                    pass

            model.optimize(callback)

            if model.SolCount > 0:
                selected_arcs = [
                    key for key in x_keys_list if x[key].X > 0.5
                ]
                selected_assignments = [
                    key for key in y_keys_list if y[key].X > 0.5
                ]
                final_mip_solution = make_solution_from_arcs(
                    selected_arcs,
                    selected_assignments,
                    depot,
                    sink,
                    truck_times,
                    drone_times,
                )
                record_solution(final_mip_solution)

        except gp.GurobiError:
            # The heuristic incumbent remains a valid fallback.
            pass
        except Exception:
            # Preserve a valid output even if model construction is interrupted.
            pass

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()