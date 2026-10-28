import argparse
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


EPS = 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    hard_deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    depot = instance["depot"]
    customers = list(instance["customers"])
    n = len(customers)
    capacity = int(instance["vehicle_capacity"])

    depot_id = int(depot["id"])
    depot_ready = float(depot["ready_time"])
    depot_due = float(depot["due_date"])

    ids = [int(c["id"]) for c in customers]
    xs = [float(depot["x"])] + [float(c["x"]) for c in customers]
    ys = [float(depot["y"])] + [float(c["y"]) for c in customers]
    demands = [int(c["demand"]) for c in customers]
    ready = [float(c["ready_time"]) for c in customers]
    due = [float(c["due_date"]) for c in customers]
    service = [float(c["service_time"]) for c in customers]

    def write_solution(solution):
        directory = os.path.dirname(os.path.abspath(args.solution_path))
        os.makedirs(directory, exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f, indent=2)

    if n == 0:
        solution = {"objective_value": 0.0, "routes": []}
        if logger:
            logger.log_solution(0.0, solution)
        write_solution(solution)
        return

    # Matrix index 0 is the depot; customer i uses matrix index i + 1.
    dist = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            dist[i][j] = d
            dist[j][i] = d

    eval_cache = {}

    def evaluate_route(route):
        key = tuple(route)
        cached = eval_cache.get(key)
        if cached is not None:
            return cached

        load = 0
        current_time = depot_ready
        total_distance = 0.0
        previous = 0

        for customer_index in route:
            load += demands[customer_index]
            if load > capacity:
                result = (False, math.inf, load)
                eval_cache[key] = result
                return result

            node = customer_index + 1
            travel = dist[previous][node]
            total_distance += travel
            arrival = current_time + travel

            if arrival > due[customer_index] + EPS:
                result = (False, math.inf, load)
                eval_cache[key] = result
                return result

            current_time = (
                max(arrival, ready[customer_index])
                + service[customer_index]
            )
            previous = node

        return_distance = dist[previous][0]
        total_distance += return_distance

        if current_time + return_distance > depot_due + EPS:
            result = (False, math.inf, load)
        else:
            result = (True, total_distance, load)

        if len(eval_cache) > 150000:
            eval_cache.clear()
        eval_cache[key] = result
        return result

    def route_cost(route):
        return evaluate_route(route)[1]

    def solution_cost(routes):
        return sum(route_cost(route) for route in routes if route)

    def make_solution(routes):
        output_routes = []
        objective = 0.0

        for route in routes:
            if not route:
                continue
            objective += route_cost(route)
            customer_ids = [ids[i] for i in route]
            output_routes.append({
                "customer_ids": customer_ids,
                "path": [depot_id] + customer_ids + [depot_id],
            })

        return {
            "objective_value": float(objective),
            "routes": output_routes,
        }

    singleton_routes = [[i] for i in range(n)]
    for route in singleton_routes:
        if not evaluate_route(route)[0]:
            raise RuntimeError(
                "The instance contains a customer that cannot be served "
                "even on a dedicated route."
            )

    best_routes = [route[:] for route in singleton_routes]
    best_cost = solution_cost(best_routes)
    best_solution = make_solution(best_routes)

    if logger:
        logger.log_solution(best_cost, best_solution)

    route_pool = {}
    max_pool_size = 30000

    def add_pool_route(route):
        if not route or len(route_pool) >= max_pool_size:
            return

        key = tuple(route)
        if key in route_pool:
            return

        feasible, cost, _ = evaluate_route(route)
        if feasible:
            route_pool[key] = cost

    for route in singleton_routes:
        add_pool_route(route)

    def add_solution_to_pool(routes, add_fragments=False):
        for route in routes:
            if not route:
                continue

            add_pool_route(route)
            add_pool_route(list(reversed(route)))

            if not add_fragments or len(route) <= 1:
                continue
            if len(route_pool) >= max_pool_size:
                return

            length = len(route)
            if length <= 12:
                for begin in range(length):
                    for end in range(begin + 1, length + 1):
                        add_pool_route(route[begin:end])
                        if len(route_pool) >= max_pool_size:
                            return
            else:
                for cut in range(1, length):
                    add_pool_route(route[:cut])
                    add_pool_route(route[cut:])
                    if len(route_pool) >= max_pool_size:
                        return

                for position in range(length):
                    add_pool_route(route[:position] + route[position + 1:])
                    if len(route_pool) >= max_pool_size:
                        return

    def register_solution(routes, pool_fragments=False):
        nonlocal best_routes, best_cost, best_solution

        routes = [list(route) for route in routes if route]
        cost = solution_cost(routes)
        add_solution_to_pool(routes, add_fragments=pool_fragments)

        if cost < best_cost - 1e-7:
            best_cost = cost
            best_routes = [route[:] for route in routes]
            best_solution = make_solution(best_routes)

            if logger:
                logger.log_solution(best_cost, best_solution)
            return True

        return False

    nearest = []
    for i in range(n):
        order = [j for j in range(n) if j != i]
        order.sort(key=lambda j: dist[i + 1][j + 1])
        nearest.append(order)

    savings_edges = []

    if n <= 500:
        for i in range(n):
            for j in range(i + 1, n):
                saving = (
                    dist[0][i + 1]
                    + dist[0][j + 1]
                    - dist[i + 1][j + 1]
                )
                savings_edges.append((saving, i, j))
    else:
        edge_set = set()
        neighbor_limit = min(40, n - 1)

        for i in range(n):
            for j in nearest[i][:neighbor_limit]:
                a, b = (i, j) if i < j else (j, i)
                edge_set.add((a, b))

        for i, j in edge_set:
            saving = (
                dist[0][i + 1]
                + dist[0][j + 1]
                - dist[i + 1][j + 1]
            )
            savings_edges.append((saving, i, j))

    savings_edges.sort(reverse=True, key=lambda item: item[0])

    def savings_construct(edge_order, deadline):
        routes = {i: [i] for i in range(n)}
        owner = list(range(n))

        # This must be a dictionary. Deleting an element from a list would
        # shift all subsequent indices and corrupt route-to-load mappings.
        loads = {i: demands[i] for i in range(n)}

        for edge_number, (_, i, j) in enumerate(edge_order):
            if edge_number % 256 == 0 and time.monotonic() >= deadline:
                break

            route_id_i = owner[i]
            route_id_j = owner[j]

            if route_id_i == route_id_j:
                continue
            if route_id_i not in routes or route_id_j not in routes:
                continue

            route_i = routes[route_id_i]
            route_j = routes[route_id_j]

            if i != route_i[0] and i != route_i[-1]:
                continue
            if j != route_j[0] and j != route_j[-1]:
                continue

            if loads[route_id_i] + loads[route_id_j] > capacity:
                continue

            left = route_i if route_i[-1] == i else list(reversed(route_i))
            right = route_j if route_j[0] == j else list(reversed(route_j))
            candidate = left + right

            feasible = evaluate_route(candidate)[0]
            if not feasible:
                reversed_candidate = list(reversed(candidate))
                if evaluate_route(reversed_candidate)[0]:
                    candidate = reversed_candidate
                    feasible = True

            if not feasible:
                continue

            routes[route_id_i] = candidate
            loads[route_id_i] += loads[route_id_j]

            del routes[route_id_j]
            del loads[route_id_j]

            for customer_index in candidate:
                owner[customer_index] = route_id_i

        return [list(route) for route in routes.values()]

    def local_search(initial_routes, deadline):
        routes = [list(route) for route in initial_routes if route]
        rounds = 0

        while rounds < 40 and time.monotonic() < deadline:
            rounds += 1
            best_delta = -1e-8
            best_move = None

            # Intra-route 2-opt.
            stop = False
            for route_index, route in enumerate(routes):
                if time.monotonic() >= deadline:
                    stop = True
                    break
                if len(route) < 3:
                    continue

                old_cost = route_cost(route)

                for begin in range(len(route) - 1):
                    for end in range(begin + 1, len(route)):
                        if time.monotonic() >= deadline:
                            stop = True
                            break

                        candidate = (
                            route[:begin]
                            + list(reversed(route[begin:end + 1]))
                            + route[end + 1:]
                        )
                        feasible, new_cost, _ = evaluate_route(candidate)

                        if feasible and new_cost - old_cost < best_delta:
                            best_delta = new_cost - old_cost
                            best_move = (
                                "replace",
                                route_index,
                                candidate,
                            )
                    if stop:
                        break
                if stop:
                    break

            if time.monotonic() >= deadline:
                break

            customer_route = {}
            customer_position = {}
            for route_index, route in enumerate(routes):
                for position, customer_index in enumerate(route):
                    customer_route[customer_index] = route_index
                    customer_position[customer_index] = position

            # Customer relocation between or within routes.
            stop = False
            for source_index, source in enumerate(routes):
                if time.monotonic() >= deadline:
                    stop = True
                    break

                source_old_cost = route_cost(source)

                for source_position, customer_index in enumerate(source):
                    if time.monotonic() >= deadline:
                        stop = True
                        break

                    reduced = (
                        source[:source_position]
                        + source[source_position + 1:]
                    )
                    reduced_cost = route_cost(reduced) if reduced else 0.0

                    if len(routes) <= 25:
                        targets = range(len(routes))
                    else:
                        target_set = {
                            customer_route[j]
                            for j in nearest[customer_index][:min(18, n - 1)]
                        }
                        target_set.add(source_index)
                        targets = target_set

                    for target_index in targets:
                        if time.monotonic() >= deadline:
                            stop = True
                            break

                        if target_index == source_index:
                            for insertion in range(len(reduced) + 1):
                                candidate = (
                                    reduced[:insertion]
                                    + [customer_index]
                                    + reduced[insertion:]
                                )
                                if candidate == source:
                                    continue

                                feasible, new_cost, _ = evaluate_route(candidate)
                                delta = new_cost - source_old_cost

                                if feasible and delta < best_delta:
                                    best_delta = delta
                                    best_move = (
                                        "replace",
                                        source_index,
                                        candidate,
                                    )
                            continue

                        target = routes[target_index]
                        target_old_cost = route_cost(target)

                        for insertion in range(len(target) + 1):
                            new_target = (
                                target[:insertion]
                                + [customer_index]
                                + target[insertion:]
                            )

                            feasible, new_target_cost, _ = evaluate_route(
                                new_target
                            )
                            if not feasible:
                                continue

                            delta = (
                                reduced_cost
                                + new_target_cost
                                - source_old_cost
                                - target_old_cost
                            )

                            if delta < best_delta:
                                best_delta = delta
                                best_move = (
                                    "relocate",
                                    source_index,
                                    target_index,
                                    reduced,
                                    new_target,
                                )

                    if stop:
                        break
                if stop:
                    break

            if time.monotonic() >= deadline:
                break

            # Restricted cross-route swaps.
            seen_pairs = set()
            neighbor_count = min(16, max(0, n - 1))
            stop = False

            for customer_x in range(n):
                if time.monotonic() >= deadline:
                    stop = True
                    break

                route_x_index = customer_route[customer_x]
                position_x = customer_position[customer_x]

                for customer_y in nearest[customer_x][:neighbor_count]:
                    if time.monotonic() >= deadline:
                        stop = True
                        break

                    route_y_index = customer_route[customer_y]
                    if route_x_index == route_y_index:
                        continue

                    pair = (
                        (customer_x, customer_y)
                        if customer_x < customer_y
                        else (customer_y, customer_x)
                    )
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    position_y = customer_position[customer_y]
                    route_x = routes[route_x_index]
                    route_y = routes[route_y_index]

                    new_x = route_x[:]
                    new_y = route_y[:]
                    new_x[position_x] = customer_y
                    new_y[position_y] = customer_x

                    feasible_x, cost_x, _ = evaluate_route(new_x)
                    if not feasible_x:
                        continue

                    feasible_y, cost_y, _ = evaluate_route(new_y)
                    if not feasible_y:
                        continue

                    delta = (
                        cost_x
                        + cost_y
                        - route_cost(route_x)
                        - route_cost(route_y)
                    )

                    if delta < best_delta:
                        best_delta = delta
                        best_move = (
                            "swap",
                            route_x_index,
                            route_y_index,
                            new_x,
                            new_y,
                        )

                if stop:
                    break

            if best_move is None:
                break

            move_type = best_move[0]

            if move_type == "replace":
                routes[best_move[1]] = best_move[2]

            elif move_type == "relocate":
                (
                    _,
                    source_index,
                    target_index,
                    new_source,
                    new_target,
                ) = best_move
                routes[source_index] = new_source
                routes[target_index] = new_target
                routes = [route for route in routes if route]

            else:
                _, index_x, index_y, new_x, new_y = best_move
                routes[index_x] = new_x
                routes[index_y] = new_y

            register_solution(routes, pool_fragments=False)

        return routes

    rng = random.Random(0)
    heuristic_deadline = min(
        start_time + max(0.0, args.time_limit * 0.72),
        hard_deadline - 0.25,
    )

    if time.monotonic() < heuristic_deadline:
        routes = savings_construct(savings_edges, heuristic_deadline)
        register_solution(routes, pool_fragments=True)

        local_deadline = min(
            heuristic_deadline,
            time.monotonic()
            + max(0.05, min(2.5, args.time_limit * 0.14)),
        )
        routes = local_search(routes, local_deadline)
        register_solution(routes, pool_fragments=True)

    iteration = 0

    while time.monotonic() < heuristic_deadline:
        iteration += 1
        randomized_edges = list(savings_edges)

        block_size = 8 + (iteration * 13) % 75
        for begin in range(0, len(randomized_edges), block_size):
            if time.monotonic() >= heuristic_deadline:
                break

            end = min(len(randomized_edges), begin + block_size)
            block = randomized_edges[begin:end]
            rng.shuffle(block)
            randomized_edges[begin:end] = block

        routes = savings_construct(randomized_edges, heuristic_deadline)
        register_solution(routes, pool_fragments=(iteration <= 6))

        remaining = heuristic_deadline - time.monotonic()
        if remaining <= 0:
            break

        local_deadline = min(
            heuristic_deadline,
            time.monotonic() + min(0.8, max(0.04, remaining * 0.25)),
        )
        routes = local_search(routes, local_deadline)
        register_solution(routes, pool_fragments=(iteration <= 6))

    # Optimize over the generated route pool with set partitioning.
    remaining = hard_deadline - time.monotonic()

    if remaining > 0.5 and route_pool:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            pool_items = list(route_pool.items())
            model = gp.Model("vrptw_route_pool")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, remaining - 0.15)

            variables = []
            containing = [[] for _ in range(n)]

            for route_index, (route, cost) in enumerate(pool_items):
                variable = model.addVar(
                    vtype=GRB.BINARY,
                    obj=cost,
                    name=f"route_{route_index}",
                )
                variables.append(variable)

                for customer_index in route:
                    containing[customer_index].append(variable)

            model.ModelSense = GRB.MINIMIZE
            model.update()

            for customer_index in range(n):
                model.addConstr(
                    gp.quicksum(containing[customer_index]) == 1,
                    name=f"serve_{customer_index}",
                )

            incumbent_keys = {tuple(route) for route in best_routes}
            for index, (route, _) in enumerate(pool_items):
                variables[index].Start = (
                    1.0 if route in incumbent_keys else 0.0
                )

            callback_best = [best_cost]

            def callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                try:
                    objective = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if objective >= callback_best[0] - 1e-7:
                        return

                    values = cb_model.cbGetSolution(variables)
                    selected = [
                        list(pool_items[i][0])
                        for i, value in enumerate(values)
                        if value > 0.5
                    ]
                    solution = make_solution(selected)
                    actual_objective = solution["objective_value"]

                    if actual_objective < callback_best[0] - 1e-7:
                        callback_best[0] = actual_objective
                        if logger:
                            logger.log_solution(
                                actual_objective,
                                solution,
                            )
                except Exception:
                    pass

            model.optimize(callback)

            if model.SolCount > 0:
                selected_routes = [
                    list(pool_items[i][0])
                    for i, variable in enumerate(variables)
                    if variable.X > 0.5
                ]
                register_solution(
                    selected_routes,
                    pool_fragments=False,
                )

        except Exception:
            # The heuristic incumbent remains available if Gurobi cannot be
            # initialized or the route-pool model cannot be completed.
            pass

    final_solution = make_solution(best_routes)
    write_solution(final_solution)


if __name__ == "__main__":
    main()