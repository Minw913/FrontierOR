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


def configure_gurobi(model, time_limit=None):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = max(0.01, float(time_limit))


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["num_customers"])
    num_vehicles = int(data["num_vehicles"])
    capacity = int(data["vehicle_capacity"])
    gamma = int(data["uncertainty_set"]["Gamma"])
    depot_id = int(data["depot"]["id"])
    customers_data = data["customers"]
    distance_matrix = data["distance_matrix"]

    if n != len(customers_data):
        raise ValueError("num_customers does not match the customer list")
    if num_vehicles <= 0:
        raise ValueError("num_vehicles must be positive")
    if n < num_vehicles:
        raise ValueError("Exactly one nonempty route per vehicle is impossible when customers < vehicles")

    # Internal customer indices are 1..n; index 0 denotes the depot.
    actual_id = [depot_id] + [int(c["id"]) for c in customers_data]
    nominal = [0] + [int(c["nominal_demand"]) for c in customers_data]
    deviation = [0] + [int(c["demand_deviation"]) for c in customers_data]

    def dist(i, j):
        return int(distance_matrix[actual_id[i]][actual_id[j]])

    def route_cost(route):
        if not route:
            return 0
        total = dist(0, route[0])
        for a, b in zip(route, route[1:]):
            total += dist(a, b)
        total += dist(route[-1], 0)
        return int(total)

    def worst_case_demand(route):
        base = sum(nominal[i] for i in route)
        if gamma <= 0 or not route:
            return int(base)
        largest = sorted((deviation[i] for i in route), reverse=True)
        return int(base + sum(largest[:min(gamma, len(largest))]))

    def feasible_route(route):
        return worst_case_demand(route) <= capacity

    def solution_from_routes(routes):
        result_routes = []
        objective = 0
        for route in routes:
            cost = route_cost(route)
            objective += cost
            result_routes.append({
                "customers": [actual_id[i] for i in route],
                "cost": int(cost),
                "worst_case_demand": int(worst_case_demand(route))
            })
        return {
            "objective_value": float(objective),
            "routes": result_routes
        }

    best_solution = None
    best_value = math.inf
    best_internal_routes = None

    def consider(routes):
        nonlocal best_solution, best_value, best_internal_routes

        if routes is None or len(routes) != num_vehicles:
            return False
        if any(len(r) == 0 for r in routes):
            return False

        flattened = [i for r in routes for i in r]
        if len(flattened) != n or set(flattened) != set(range(1, n + 1)):
            return False
        if any(not feasible_route(r) for r in routes):
            return False

        solution = solution_from_routes(routes)
        value = float(solution["objective_value"])
        if value + 1e-9 < best_value:
            best_value = value
            best_solution = solution
            best_internal_routes = [list(r) for r in routes]
            if logger:
                logger.log_solution(value, solution)
            return True
        return False

    def insertion_delta(route, customer, position):
        left = 0 if position == 0 else route[position - 1]
        right = 0 if position == len(route) else route[position]
        return dist(left, customer) + dist(customer, right) - dist(left, right)

    def construct_routes(order, mode):
        routes = [[] for _ in range(num_vehicles)]

        for index, customer in enumerate(order):
            empty_count = sum(1 for r in routes if not r)
            remaining_including_current = len(order) - index
            force_empty = remaining_including_current == empty_count

            choices = []
            for k, route in enumerate(routes):
                if force_empty and route:
                    continue

                candidate_members = route + [customer]
                load = worst_case_demand(candidate_members)
                if load > capacity:
                    continue

                if route:
                    best_pos = min(
                        range(len(route) + 1),
                        key=lambda pos: insertion_delta(route, customer, pos)
                    )
                    delta = insertion_delta(route, customer, best_pos)
                else:
                    best_pos = 0
                    delta = dist(0, customer) + dist(customer, 0)

                slack = capacity - load
                if mode == 0:
                    score = (slack, delta, k)
                elif mode == 1:
                    score = (delta, slack, k)
                else:
                    # A compromise between geographic compactness and bin packing.
                    score = (delta + 0.05 * slack, delta, k)

                choices.append((score, k, best_pos))

            if not choices:
                return None

            _, chosen_k, chosen_pos = min(choices, key=lambda x: x[0])
            routes[chosen_k].insert(chosen_pos, customer)

        if any(not r for r in routes):
            return None
        return routes

    def improve_route_order(route, local_deadline):
        route = list(route)
        if len(route) <= 2:
            return route

        passes = 0
        while passes < 4 and time.monotonic() < local_deadline:
            passes += 1
            current_cost = route_cost(route)
            best_cost_for_pass = current_cost
            best_route_for_pass = route

            # Relocate one customer to any other position.
            for old_pos in range(len(route)):
                if time.monotonic() >= local_deadline:
                    break
                reduced = route[:old_pos] + route[old_pos + 1:]
                customer = route[old_pos]
                for new_pos in range(len(reduced) + 1):
                    candidate = reduced[:new_pos] + [customer] + reduced[new_pos:]
                    cost = route_cost(candidate)
                    if cost < best_cost_for_pass:
                        best_cost_for_pass = cost
                        best_route_for_pass = candidate

            # Directed 2-opt: reversal is evaluated exactly because costs may be asymmetric.
            if len(route) <= 35:
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
                        if cost < best_cost_for_pass:
                            best_cost_for_pass = cost
                            best_route_for_pass = candidate

            if best_cost_for_pass >= current_cost:
                break
            route = best_route_for_pass

        return route

    def improve_all_routes(routes, local_deadline):
        improved = []
        for route in routes:
            if time.monotonic() >= local_deadline:
                improved.extend([list(r) for r in routes[len(improved):]])
                break
            improved.append(improve_route_order(route, local_deadline))
        return improved

    # Fast constructive phase.
    heuristic_budget = min(2.0, max(0.05, 0.10 * max(1, args.time_limit)))
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)
    rng = random.Random(0)

    robust_size = {
        i: nominal[i] + (deviation[i] if gamma > 0 else 0)
        for i in range(1, n + 1)
    }

    deterministic_orders = [
        sorted(range(1, n + 1), key=lambda i: (-robust_size[i], -nominal[i], i)),
        sorted(range(1, n + 1), key=lambda i: (-nominal[i], -deviation[i], i)),
        sorted(range(1, n + 1), key=lambda i: (-deviation[i], -nominal[i], i))
    ]

    depot_x = float(data["depot"]["x"])
    depot_y = float(data["depot"]["y"])
    coordinates = {
        i: (float(customers_data[i - 1]["x"]), float(customers_data[i - 1]["y"]))
        for i in range(1, n + 1)
    }
    sweep_order = sorted(
        range(1, n + 1),
        key=lambda i: math.atan2(
            coordinates[i][1] - depot_y,
            coordinates[i][0] - depot_x
        )
    )
    deterministic_orders.extend([sweep_order, list(reversed(sweep_order))])

    trial = 0
    while time.monotonic() < heuristic_deadline:
        if trial < len(deterministic_orders):
            order = list(deterministic_orders[trial])
        else:
            # Randomized demand-biased ordering.
            order = list(range(1, n + 1))
            noise_scale = 0.35
            order.sort(
                key=lambda i: -(
                    robust_size[i] * (1.0 + noise_scale * rng.random())
                )
            )

        routes = construct_routes(order, trial % 3)
        if routes is not None:
            consider(routes)
            if time.monotonic() < heuristic_deadline:
                improved = improve_all_routes(routes, heuristic_deadline)
                consider(improved)

        trial += 1
        if trial >= 100 and best_solution is not None:
            break

    # If greedy packing failed, solve a short robust assignment feasibility MIP.
    if best_solution is None and deadline - time.monotonic() > 0.25:
        assignment_time = min(3.0, 0.20 * max(1, args.time_limit),
                              max(0.05, deadline - time.monotonic() - 0.1))
        am = gp.Model("robust_assignment")
        configure_gurobi(am, assignment_time)

        ay = am.addVars(num_vehicles, range(1, n + 1), vtype=GRB.BINARY, name="y")
        am.addConstrs(
            (gp.quicksum(ay[k, i] for k in range(num_vehicles)) == 1
             for i in range(1, n + 1)),
            name="assign"
        )
        am.addConstrs(
            (gp.quicksum(ay[k, i] for i in range(1, n + 1)) >= 1
             for k in range(num_vehicles)),
            name="nonempty"
        )

        if gamma <= 0:
            am.addConstrs(
                (gp.quicksum(nominal[i] * ay[k, i] for i in range(1, n + 1))
                 <= capacity for k in range(num_vehicles)),
                name="capacity"
            )
        elif gamma >= n:
            am.addConstrs(
                (gp.quicksum((nominal[i] + deviation[i]) * ay[k, i]
                             for i in range(1, n + 1)) <= capacity
                 for k in range(num_vehicles)),
                name="capacity"
            )
        else:
            max_dev = max(deviation) if deviation else 0
            az = am.addVars(num_vehicles, lb=0.0, ub=max_dev,
                            vtype=GRB.CONTINUOUS, name="z")
            ap = am.addVars(num_vehicles, range(1, n + 1), lb=0.0,
                            vtype=GRB.CONTINUOUS, name="p")
            am.addConstrs(
                (az[k] + ap[k, i] >= deviation[i] * ay[k, i]
                 for k in range(num_vehicles) for i in range(1, n + 1)),
                name="robust_dual"
            )
            am.addConstrs(
                (gp.quicksum(nominal[i] * ay[k, i] for i in range(1, n + 1))
                 + gamma * az[k]
                 + gp.quicksum(ap[k, i] for i in range(1, n + 1))
                 <= capacity for k in range(num_vehicles)),
                name="capacity"
            )

        # Vehicle-label symmetry breaking.
        am.addConstrs(
            (gp.quicksum(i * ay[k, i] for i in range(1, n + 1))
             <= gp.quicksum(i * ay[k + 1, i] for i in range(1, n + 1))
             for k in range(num_vehicles - 1)),
            name="symmetry"
        )

        am.setObjective(0.0, GRB.MINIMIZE)
        am.optimize()

        if am.SolCount > 0:
            groups = [[] for _ in range(num_vehicles)]
            for k in range(num_vehicles):
                members = [i for i in range(1, n + 1) if ay[k, i].X > 0.5]
                # Cheapest insertion constructs a route for each assigned group.
                route = []
                for customer in sorted(members, key=lambda i: -robust_size[i]):
                    pos = min(
                        range(len(route) + 1),
                        key=lambda p: insertion_delta(route, customer, p)
                    )
                    route.insert(pos, customer)
                groups[k] = route
            consider(groups)
            if time.monotonic() < deadline:
                consider(improve_all_routes(groups, min(deadline, time.monotonic() + 0.5)))

        am.dispose()

    remaining = deadline - time.monotonic()

    # Exact vehicle-indexed branch-and-cut model.
    if remaining > 0.05:
        model = gp.Model("robust_cvrp")
        configure_gurobi(model, max(0.01, remaining - 0.02))
        model.Params.LazyConstraints = 1

        vehicles = range(num_vehicles)
        customers = range(1, n + 1)
        nodes = range(0, n + 1)
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
            (gp.quicksum(y[k, i] for k in vehicles) == 1 for i in customers),
            name="assignment"
        )

        for k in vehicles:
            model.addConstr(
                gp.quicksum(x[k, 0, j] for j in customers) == 1,
                name=f"depot_out_{k}"
            )
            model.addConstr(
                gp.quicksum(x[k, i, 0] for i in customers) == 1,
                name=f"depot_in_{k}"
            )
            for i in customers:
                model.addConstr(
                    gp.quicksum(x[k, i, j] for j in nodes if j != i) == y[k, i],
                    name=f"out_{k}_{i}"
                )
                model.addConstr(
                    gp.quicksum(x[k, j, i] for j in nodes if j != i) == y[k, i],
                    name=f"in_{k}_{i}"
                )

        if gamma <= 0:
            model.addConstrs(
                (gp.quicksum(nominal[i] * y[k, i] for i in customers)
                 <= capacity for k in vehicles),
                name="robust_capacity"
            )
        elif gamma >= n:
            model.addConstrs(
                (gp.quicksum((nominal[i] + deviation[i]) * y[k, i]
                             for i in customers) <= capacity
                 for k in vehicles),
                name="robust_capacity"
            )
        else:
            max_dev = max(deviation) if deviation else 0
            z = model.addVars(vehicles, lb=0.0, ub=max_dev,
                              vtype=GRB.CONTINUOUS, name="z")
            p = model.addVars(vehicles, customers, lb=0.0,
                              vtype=GRB.CONTINUOUS, name="p")
            model.addConstrs(
                (z[k] + p[k, i] >= deviation[i] * y[k, i]
                 for k in vehicles for i in customers),
                name="robust_dual"
            )
            model.addConstrs(
                (gp.quicksum(nominal[i] * y[k, i] for i in customers)
                 + gamma * z[k]
                 + gp.quicksum(p[k, i] for i in customers)
                 <= capacity for k in vehicles),
                name="robust_capacity"
            )

        # Any unlabeled route set can be relabeled to satisfy these constraints.
        model.addConstrs(
            (gp.quicksum(i * y[k, i] for i in customers)
             <= gp.quicksum(i * y[k + 1, i] for i in customers)
             for k in range(num_vehicles - 1)),
            name="vehicle_symmetry"
        )

        model.setObjective(
            gp.quicksum(dist(i, j) * x[k, i, j] for k, i, j in arc_keys),
            GRB.MINIMIZE
        )

        # Install a MIP start from the heuristic incumbent.
        if best_internal_routes is not None:
            start_routes = sorted(
                [list(r) for r in best_internal_routes],
                key=lambda r: sum(r)
            )
            selected_arcs = set()
            selected_assignments = set()
            for k, route in enumerate(start_routes):
                for i in route:
                    selected_assignments.add((k, i))
                sequence = [0] + route + [0]
                for i, j in zip(sequence, sequence[1:]):
                    selected_arcs.add((k, i, j))

            for key in arc_keys:
                x[key].Start = 1.0 if key in selected_arcs else 0.0
            for k in vehicles:
                for i in customers:
                    y[k, i].Start = 1.0 if (k, i) in selected_assignments else 0.0

        xvars = [x[key] for key in arc_keys]

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            values = cb_model.cbGetSolution(xvars)
            chosen = {
                key for key, value in zip(arc_keys, values)
                if value > 0.5
            }

            subtour_found = False
            outgoing_by_vehicle = []
            for k in vehicles:
                outgoing = {}
                for kk, i, j in chosen:
                    if kk == k:
                        outgoing[i] = j
                outgoing_by_vehicle.append(outgoing)

                unvisited = set(outgoing.keys())
                while unvisited:
                    first = next(iter(unvisited))
                    walk = []
                    position = {}
                    current = first

                    while current in outgoing and current not in position:
                        position[current] = len(walk)
                        walk.append(current)
                        current = outgoing[current]

                    for node in walk:
                        unvisited.discard(node)

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
            seen_all = set()
            for k in vehicles:
                outgoing = outgoing_by_vehicle[k]
                route = []
                current = 0
                seen = set()
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
                seen_all.update(route)

            if len(seen_all) == n:
                consider(routes)

        model.optimize(callback)

        if model.SolCount > 0:
            chosen = {
                key for key in arc_keys
                if x[key].X > 0.5
            }
            final_routes = []
            valid = True
            all_seen = set()

            for k in vehicles:
                outgoing = {}
                for kk, i, j in chosen:
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
                final_routes.append(route)
                all_seen.update(route)

            if valid and len(all_seen) == n:
                consider(final_routes)

        model.dispose()

    if best_solution is None:
        raise RuntimeError("No feasible solution was found within the time limit")

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()