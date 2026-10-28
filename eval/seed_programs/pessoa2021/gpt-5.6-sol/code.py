import argparse
import json
import math
import os
import random
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


def configure_gurobi(model, time_limit):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0.01, float(time_limit))


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    customers_data = data["customers"]

    # Some benchmark files count the depot in num_customers. The customer list
    # is therefore the authoritative source for the actual number of customers.
    n = len(customers_data)

    num_vehicles = int(data["num_vehicles"])
    capacity = int(data["vehicle_capacity"])
    gamma = max(0, int(data["uncertainty_set"]["Gamma"]))
    depot_id = int(data["depot"]["id"])
    distance_matrix = data["distance_matrix"]

    if num_vehicles <= 0:
        raise ValueError("num_vehicles must be positive")
    if n == 0:
        raise ValueError("The instance contains no customers")
    if n < num_vehicles:
        raise ValueError(
            "The instance has fewer customers than vehicles, so exactly "
            "num_vehicles nonempty routes cannot be constructed"
        )

    output_id = [depot_id] + [int(c["id"]) for c in customers_data]
    nominal = [0] + [int(c["nominal_demand"]) for c in customers_data]
    deviation = [0] + [int(c["demand_deviation"]) for c in customers_data]

    matrix_size = len(distance_matrix)
    direct_ids_valid = (
        0 <= depot_id < matrix_size
        and all(0 <= output_id[i] < matrix_size for i in range(1, n + 1))
        and all(len(row) >= matrix_size for row in distance_matrix)
    )

    if direct_ids_valid:
        matrix_index = output_id[:]
    elif matrix_size >= n + 1:
        # Fallback for files whose IDs are labels rather than matrix indices.
        matrix_index = list(range(n + 1))
    else:
        raise ValueError("Distance matrix dimensions are inconsistent with the node data")

    def dist(i, j):
        return int(distance_matrix[matrix_index[i]][matrix_index[j]])

    def route_cost(route):
        if not route:
            return 0
        value = dist(0, route[0])
        for i, j in zip(route, route[1:]):
            value += dist(i, j)
        value += dist(route[-1], 0)
        return int(value)

    def worst_case_demand(route):
        total = sum(nominal[i] for i in route)
        if gamma > 0 and route:
            devs = sorted((deviation[i] for i in route), reverse=True)
            total += sum(devs[:min(gamma, len(devs))])
        return int(total)

    def feasible_route(route):
        return worst_case_demand(route) <= capacity

    def make_solution(routes):
        route_results = []
        total_cost = 0

        for route in routes:
            cost = route_cost(route)
            total_cost += cost
            route_results.append({
                "customers": [output_id[i] for i in route],
                "cost": int(cost),
                "worst_case_demand": int(worst_case_demand(route))
            })

        return {
            "objective_value": float(total_cost),
            "routes": route_results
        }

    best_value = math.inf
    best_solution = None
    best_routes = None

    def consider(routes):
        nonlocal best_value, best_solution, best_routes

        if routes is None or len(routes) != num_vehicles:
            return False
        if any(len(route) == 0 for route in routes):
            return False

        flattened = [i for route in routes for i in route]
        if len(flattened) != n or set(flattened) != set(range(1, n + 1)):
            return False
        if any(not feasible_route(route) for route in routes):
            return False

        solution = make_solution(routes)
        value = solution["objective_value"]

        if value + 1e-9 < best_value:
            best_value = value
            best_solution = solution
            best_routes = [list(route) for route in routes]
            if logger:
                logger.log_solution(value, solution)
            return True

        return False

    def insertion_delta(route, customer, position):
        left = 0 if position == 0 else route[position - 1]
        right = 0 if position == len(route) else route[position]
        return dist(left, customer) + dist(customer, right) - dist(left, right)

    def cheapest_insertion_order(members):
        route = []
        if not members:
            return route

        remaining = set(members)
        first = min(
            remaining,
            key=lambda i: dist(0, i) + dist(i, 0)
        )
        route.append(first)
        remaining.remove(first)

        while remaining:
            best = None
            for customer in remaining:
                for position in range(len(route) + 1):
                    delta = insertion_delta(route, customer, position)
                    key = (delta, customer, position)
                    if best is None or key < best[0]:
                        best = (key, customer, position)

            _, customer, position = best
            route.insert(position, customer)
            remaining.remove(customer)

        return route

    def construct_routes(order, mode):
        routes = [[] for _ in range(num_vehicles)]

        for index, customer in enumerate(order):
            remaining_count = len(order) - index
            empty_routes = sum(1 for route in routes if not route)
            force_empty = remaining_count == empty_routes

            choices = []

            for vehicle, route in enumerate(routes):
                if force_empty and route:
                    continue
                if worst_case_demand(route + [customer]) > capacity:
                    continue

                if route:
                    position = min(
                        range(len(route) + 1),
                        key=lambda p: insertion_delta(route, customer, p)
                    )
                    delta = insertion_delta(route, customer, position)
                else:
                    position = 0
                    delta = dist(0, customer) + dist(customer, 0)

                slack = capacity - worst_case_demand(route + [customer])

                if mode == 0:
                    score = (slack, delta, len(route), vehicle)
                elif mode == 1:
                    score = (delta, slack, len(route), vehicle)
                elif mode == 2:
                    score = (delta + 0.03 * slack, slack, vehicle)
                else:
                    score = (len(route), delta, slack, vehicle)

                choices.append((score, vehicle, position))

            if not choices:
                return None

            _, vehicle, position = min(choices, key=lambda item: item[0])
            routes[vehicle].insert(position, customer)

        if any(not route for route in routes):
            return None

        return routes

    def improve_route(route, local_deadline):
        route = list(route)

        if len(route) <= 2:
            if len(route) == 2:
                reversed_route = list(reversed(route))
                if route_cost(reversed_route) < route_cost(route):
                    return reversed_route
            return route

        passes = 0
        while passes < 5 and time.monotonic() < local_deadline:
            passes += 1
            current_cost = route_cost(route)
            best_cost = current_cost
            best_candidate = None

            # Directed relocate.
            for old_position in range(len(route)):
                if time.monotonic() >= local_deadline:
                    break

                customer = route[old_position]
                reduced = route[:old_position] + route[old_position + 1:]

                for new_position in range(len(reduced) + 1):
                    candidate = (
                        reduced[:new_position]
                        + [customer]
                        + reduced[new_position:]
                    )
                    cost = route_cost(candidate)

                    if cost < best_cost:
                        best_cost = cost
                        best_candidate = candidate

            # Exact directed segment reversal.
            if len(route) <= 45:
                for left in range(len(route) - 1):
                    if time.monotonic() >= local_deadline:
                        break

                    for right in range(left + 1, len(route)):
                        candidate = (
                            route[:left]
                            + list(reversed(route[left:right + 1]))
                            + route[right + 1:]
                        )
                        cost = route_cost(candidate)

                        if cost < best_cost:
                            best_cost = cost
                            best_candidate = candidate

            if best_candidate is None:
                break

            route = best_candidate

        return route

    def improve_route_orders(routes, local_deadline):
        result = [list(route) for route in routes]

        for k in range(len(result)):
            if time.monotonic() >= local_deadline:
                break
            result[k] = improve_route(result[k], local_deadline)

        return result

    def improve_between_routes(routes, local_deadline):
        routes = [list(route) for route in routes]
        current_total = sum(route_cost(route) for route in routes)
        rounds = 0

        while rounds < 6 and time.monotonic() < local_deadline:
            rounds += 1
            best_delta = 0
            best_move = None

            # Customer relocation between routes.
            for source in range(num_vehicles):
                if len(routes[source]) <= 1:
                    continue

                for source_pos, customer in enumerate(routes[source]):
                    if time.monotonic() >= local_deadline:
                        break

                    new_source = (
                        routes[source][:source_pos]
                        + routes[source][source_pos + 1:]
                    )
                    old_source_cost = route_cost(routes[source])
                    new_source_cost = route_cost(new_source)

                    for target in range(num_vehicles):
                        if target == source:
                            continue
                        if not feasible_route(routes[target] + [customer]):
                            continue

                        old_target_cost = route_cost(routes[target])

                        for target_pos in range(len(routes[target]) + 1):
                            new_target = (
                                routes[target][:target_pos]
                                + [customer]
                                + routes[target][target_pos:]
                            )
                            delta = (
                                new_source_cost
                                + route_cost(new_target)
                                - old_source_cost
                                - old_target_cost
                            )

                            if delta < best_delta:
                                best_delta = delta
                                best_move = (
                                    "relocate",
                                    source,
                                    target,
                                    new_source,
                                    new_target
                                )

            # Pairwise swaps, restricted for large instances.
            if n <= 100 and time.monotonic() < local_deadline:
                for first in range(num_vehicles):
                    for second in range(first + 1, num_vehicles):
                        old_cost = route_cost(routes[first]) + route_cost(routes[second])

                        for p, a in enumerate(routes[first]):
                            if time.monotonic() >= local_deadline:
                                break

                            for q, b in enumerate(routes[second]):
                                candidate_first = list(routes[first])
                                candidate_second = list(routes[second])
                                candidate_first[p] = b
                                candidate_second[q] = a

                                if not feasible_route(candidate_first):
                                    continue
                                if not feasible_route(candidate_second):
                                    continue

                                delta = (
                                    route_cost(candidate_first)
                                    + route_cost(candidate_second)
                                    - old_cost
                                )

                                if delta < best_delta:
                                    best_delta = delta
                                    best_move = (
                                        "swap",
                                        first,
                                        second,
                                        candidate_first,
                                        candidate_second
                                    )

            if best_move is None:
                break

            _, first, second, new_first, new_second = best_move
            routes[first] = new_first
            routes[second] = new_second
            current_total += best_delta
            consider(routes)

        return routes

    robust_size = {
        i: nominal[i] + (deviation[i] if gamma > 0 else 0)
        for i in range(1, n + 1)
    }

    depot_x = float(data["depot"]["x"])
    depot_y = float(data["depot"]["y"])
    coordinates = {
        i: (
            float(customers_data[i - 1]["x"]),
            float(customers_data[i - 1]["y"])
        )
        for i in range(1, n + 1)
    }

    sweep = sorted(
        range(1, n + 1),
        key=lambda i: math.atan2(
            coordinates[i][1] - depot_y,
            coordinates[i][0] - depot_x
        )
    )

    deterministic_orders = [
        sorted(range(1, n + 1), key=lambda i: (-robust_size[i], -nominal[i], i)),
        sorted(range(1, n + 1), key=lambda i: (-nominal[i], -deviation[i], i)),
        sorted(range(1, n + 1), key=lambda i: (-deviation[i], -nominal[i], i)),
        sweep,
        list(reversed(sweep))
    ]

    rng = random.Random(0)
    heuristic_duration = min(
        max(0.10, 0.15 * max(1, args.time_limit)),
        5.0
    )
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_duration)

    trial = 0
    while time.monotonic() < heuristic_deadline:
        if trial < len(deterministic_orders):
            order = list(deterministic_orders[trial])
        else:
            order = list(range(1, n + 1))
            order.sort(
                key=lambda i: -robust_size[i] * (0.75 + 0.5 * rng.random())
            )

        routes = construct_routes(order, trial % 4)

        if routes is not None:
            consider(routes)

            if time.monotonic() < heuristic_deadline:
                improved = improve_route_orders(routes, heuristic_deadline)
                consider(improved)

        trial += 1

        if trial >= 150:
            break

    # If construction failed, find a feasible robust assignment by MIP.
    if best_solution is None and deadline - time.monotonic() > 0.10:
        remaining = deadline - time.monotonic()
        assignment_limit = min(max(0.05, remaining - 0.05), max(2.0, 0.30 * remaining))

        assignment_model = gp.Model("robust_assignment")
        configure_gurobi(assignment_model, assignment_limit)

        vehicles = range(num_vehicles)
        customers = range(1, n + 1)

        assign = assignment_model.addVars(
            vehicles, customers, vtype=GRB.BINARY, name="assign"
        )

        assignment_model.addConstrs(
            (
                gp.quicksum(assign[k, i] for k in vehicles) == 1
                for i in customers
            ),
            name="customer_assignment"
        )
        assignment_model.addConstrs(
            (
                gp.quicksum(assign[k, i] for i in customers) >= 1
                for k in vehicles
            ),
            name="nonempty"
        )

        if gamma == 0:
            assignment_model.addConstrs(
                (
                    gp.quicksum(nominal[i] * assign[k, i] for i in customers)
                    <= capacity
                    for k in vehicles
                ),
                name="capacity"
            )
        elif gamma >= n:
            assignment_model.addConstrs(
                (
                    gp.quicksum(
                        (nominal[i] + deviation[i]) * assign[k, i]
                        for i in customers
                    ) <= capacity
                    for k in vehicles
                ),
                name="capacity"
            )
        else:
            max_deviation = max(deviation)
            z = assignment_model.addVars(
                vehicles,
                lb=0.0,
                ub=max_deviation,
                vtype=GRB.CONTINUOUS,
                name="z"
            )
            p = assignment_model.addVars(
                vehicles,
                customers,
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name="p"
            )

            assignment_model.addConstrs(
                (
                    z[k] + p[k, i] >= deviation[i] * assign[k, i]
                    for k in vehicles
                    for i in customers
                ),
                name="robust_dual"
            )
            assignment_model.addConstrs(
                (
                    gp.quicksum(nominal[i] * assign[k, i] for i in customers)
                    + gamma * z[k]
                    + gp.quicksum(p[k, i] for i in customers)
                    <= capacity
                    for k in vehicles
                ),
                name="capacity"
            )

        assignment_model.addConstrs(
            (
                gp.quicksum(i * assign[k, i] for i in customers)
                <= gp.quicksum(i * assign[k + 1, i] for i in customers)
                for k in range(num_vehicles - 1)
            ),
            name="symmetry"
        )

        # A weak geographic objective is preferable to a pure feasibility model.
        assignment_model.setObjective(
            gp.quicksum(
                (dist(0, i) + dist(i, 0)) * assign[k, i]
                for k in vehicles
                for i in customers
            ),
            GRB.MINIMIZE
        )

        assignment_model.optimize()

        if assignment_model.SolCount > 0:
            routes = []
            for k in vehicles:
                members = [
                    i for i in customers
                    if assign[k, i].X > 0.5
                ]
                routes.append(cheapest_insertion_order(members))

            consider(routes)

            if time.monotonic() < deadline:
                local_deadline = min(deadline, time.monotonic() + 1.0)
                routes = improve_route_orders(routes, local_deadline)
                consider(routes)

        assignment_model.dispose()

    # Inter-route local search before the exact phase.
    if best_routes is not None and deadline - time.monotonic() > 0.10:
        local_duration = min(
            3.0,
            max(0.10, 0.12 * max(1, args.time_limit)),
            deadline - time.monotonic()
        )
        local_deadline = time.monotonic() + local_duration
        improved = improve_between_routes(best_routes, local_deadline)
        improved = improve_route_orders(improved, local_deadline)
        consider(improved)

    # Exact branch-and-cut is useful for small and medium instances. Avoid
    # creating a very large vehicle-indexed model on large benchmarks.
    remaining = deadline - time.monotonic()
    use_exact_model = (
        best_solution is not None
        and remaining > 0.20
        and n <= 75
        and num_vehicles * n * n <= 300000
    )

    if use_exact_model:
        model = gp.Model("robust_cvrp")
        configure_gurobi(model, max(0.05, remaining - 0.05))
        model.Params.LazyConstraints = 1

        vehicles = range(num_vehicles)
        customers = range(1, n + 1)
        nodes = range(n + 1)

        arc_keys = [
            (k, i, j)
            for k in vehicles
            for i in nodes
            for j in nodes
            if i != j
        ]

        x = model.addVars(arc_keys, vtype=GRB.BINARY, name="x")
        y = model.addVars(vehicles, customers, vtype=GRB.BINARY, name="y")

        model.addConstrs(
            (
                gp.quicksum(y[k, i] for k in vehicles) == 1
                for i in customers
            ),
            name="assignment"
        )

        for k in vehicles:
            model.addConstr(
                gp.quicksum(x[k, 0, j] for j in customers) == 1
            )
            model.addConstr(
                gp.quicksum(x[k, i, 0] for i in customers) == 1
            )

            for i in customers:
                model.addConstr(
                    gp.quicksum(x[k, i, j] for j in nodes if j != i) == y[k, i]
                )
                model.addConstr(
                    gp.quicksum(x[k, j, i] for j in nodes if j != i) == y[k, i]
                )

        if gamma == 0:
            model.addConstrs(
                (
                    gp.quicksum(nominal[i] * y[k, i] for i in customers)
                    <= capacity
                    for k in vehicles
                ),
                name="capacity"
            )
        elif gamma >= n:
            model.addConstrs(
                (
                    gp.quicksum(
                        (nominal[i] + deviation[i]) * y[k, i]
                        for i in customers
                    ) <= capacity
                    for k in vehicles
                ),
                name="capacity"
            )
        else:
            max_deviation = max(deviation)
            z = model.addVars(
                vehicles,
                lb=0.0,
                ub=max_deviation,
                vtype=GRB.CONTINUOUS,
                name="z"
            )
            p = model.addVars(
                vehicles,
                customers,
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name="p"
            )

            model.addConstrs(
                (
                    z[k] + p[k, i] >= deviation[i] * y[k, i]
                    for k in vehicles
                    for i in customers
                ),
                name="robust_dual"
            )
            model.addConstrs(
                (
                    gp.quicksum(nominal[i] * y[k, i] for i in customers)
                    + gamma * z[k]
                    + gp.quicksum(p[k, i] for i in customers)
                    <= capacity
                    for k in vehicles
                ),
                name="capacity"
            )

        model.addConstrs(
            (
                gp.quicksum(i * y[k, i] for i in customers)
                <= gp.quicksum(i * y[k + 1, i] for i in customers)
                for k in range(num_vehicles - 1)
            ),
            name="vehicle_symmetry"
        )

        model.setObjective(
            gp.quicksum(
                dist(i, j) * x[k, i, j]
                for k, i, j in arc_keys
            ),
            GRB.MINIMIZE
        )

        # Sort the incumbent consistently with the symmetry constraints.
        start_routes = sorted(
            [list(route) for route in best_routes],
            key=lambda route: sum(route)
        )

        start_arcs = set()
        start_assignments = set()

        for k, route in enumerate(start_routes):
            for i in route:
                start_assignments.add((k, i))

            sequence = [0] + route + [0]
            for i, j in zip(sequence, sequence[1:]):
                start_arcs.add((k, i, j))

        for key in arc_keys:
            x[key].Start = 1.0 if key in start_arcs else 0.0

        for k in vehicles:
            for i in customers:
                y[k, i].Start = 1.0 if (k, i) in start_assignments else 0.0

        x_variable_list = [x[key] for key in arc_keys]

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            values = cb_model.cbGetSolution(x_variable_list)
            selected = {
                key
                for key, value in zip(arc_keys, values)
                if value > 0.5
            }

            outgoing_by_vehicle = []
            subtour_found = False

            for k in vehicles:
                outgoing = {}
                for kk, i, j in selected:
                    if kk == k:
                        outgoing[i] = j
                outgoing_by_vehicle.append(outgoing)

                remaining_nodes = set(outgoing)
                while remaining_nodes:
                    start = next(iter(remaining_nodes))
                    walk = []
                    position = {}
                    current = start

                    while current in outgoing and current not in position:
                        position[current] = len(walk)
                        walk.append(current)
                        current = outgoing[current]

                    for node in walk:
                        remaining_nodes.discard(node)

                    if current in position:
                        cycle = walk[position[current]:]
                        if 0 not in cycle:
                            subtour_found = True
                            cycle_set = set(cycle)
                            cb_model.cbLazy(
                                gp.quicksum(
                                    x[k, i, j]
                                    for i in cycle_set
                                    for j in cycle_set
                                    if i != j
                                ) <= len(cycle_set) - 1
                            )

            if subtour_found:
                return

            routes = []
            all_seen = set()

            for k in vehicles:
                outgoing = outgoing_by_vehicle[k]
                route = []
                seen = set()
                current = 0

                while current in outgoing:
                    nxt = outgoing[current]
                    if nxt == 0:
                        break
                    if nxt in seen:
                        return

                    seen.add(nxt)
                    route.append(nxt)
                    current = nxt

                routes.append(route)
                all_seen.update(route)

            if len(all_seen) == n:
                consider(routes)

        model.optimize(callback)

        if model.SolCount > 0:
            selected = {
                key for key in arc_keys
                if x[key].X > 0.5
            }

            routes = []
            valid = True
            all_seen = set()

            for k in vehicles:
                outgoing = {}
                for kk, i, j in selected:
                    if kk == k:
                        outgoing[i] = j

                route = []
                seen = set()
                current = 0

                while current in outgoing:
                    nxt = outgoing[current]
                    if nxt == 0:
                        break
                    if nxt in seen:
                        valid = False
                        break

                    seen.add(nxt)
                    route.append(nxt)
                    current = nxt

                routes.append(route)
                all_seen.update(route)

            if valid and len(all_seen) == n:
                consider(routes)

        model.dispose()

    if best_solution is None:
        raise RuntimeError("No feasible solution was found within the time limit")

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()