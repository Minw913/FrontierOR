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

    if n == 0:
        solution = {"objective_value": 0.0, "routes": []}
        if logger:
            logger.log_solution(0.0, solution)
        directory = os.path.dirname(os.path.abspath(args.solution_path))
        os.makedirs(directory, exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f, indent=2)
        return

    # Internal customer indices are 0..n-1. Distance-matrix index 0 is the
    # depot and customer i has matrix index i+1.
    dist = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        xi, yi = xs[i], ys[i]
        for j in range(i + 1, n + 1):
            d = math.hypot(xi - xs[j], yi - ys[j])
            dist[i][j] = d
            dist[j][i] = d

    eval_cache = {}

    def evaluate_route(route):
        """Return (feasible, distance, load) for an internal-index route."""
        key = tuple(route)
        cached = eval_cache.get(key)
        if cached is not None:
            return cached

        load = 0
        t = depot_ready
        total_distance = 0.0
        previous_matrix_index = 0

        for customer_index in route:
            load += demands[customer_index]
            if load > capacity:
                result = (False, math.inf, load)
                eval_cache[key] = result
                return result

            matrix_index = customer_index + 1
            travel = dist[previous_matrix_index][matrix_index]
            total_distance += travel
            arrival = t + travel
            if arrival > due[customer_index] + EPS:
                result = (False, math.inf, load)
                eval_cache[key] = result
                return result

            t = max(arrival, ready[customer_index]) + service[customer_index]
            previous_matrix_index = matrix_index

        return_distance = dist[previous_matrix_index][0]
        total_distance += return_distance
        if t + return_distance > depot_due + EPS:
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
        return sum(route_cost(route) for route in routes)

    def make_solution(routes):
        output_routes = []
        objective = 0.0
        for route in routes:
            if not route:
                continue
            cost = route_cost(route)
            objective += cost
            customer_ids = [ids[i] for i in route]
            output_routes.append({
                "customer_ids": customer_ids,
                "path": [depot_id] + customer_ids + [depot_id],
            })
        return {
            "objective_value": float(objective),
            "routes": output_routes,
        }

    # Singleton routes provide an immediate feasible incumbent for every
    # conventionally feasible instance.
    singleton_routes = [[i] for i in range(n)]
    for route in singleton_routes:
        if not evaluate_route(route)[0]:
            raise RuntimeError(
                "Instance has a customer that cannot be served even by a "
                "dedicated vehicle."
            )

    best_routes = [r[:] for r in singleton_routes]
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
        if key not in route_pool:
            feasible, cost, _ = evaluate_route(route)
            if feasible:
                route_pool[key] = cost

    for route in singleton_routes:
        add_pool_route(route)

    def add_solution_to_pool(routes, add_fragments=False):
        for route in routes:
            add_pool_route(route)
            rev = list(reversed(route))
            add_pool_route(rev)

            if add_fragments and len(route) > 1 and len(route_pool) < max_pool_size:
                length = len(route)
                if length <= 12:
                    for a in range(length):
                        for b in range(a + 1, length + 1):
                            add_pool_route(route[a:b])
                else:
                    # Cheap fragments that make the final set-partitioning
                    # model more flexible without creating a quadratic pool.
                    for cut in range(1, length):
                        add_pool_route(route[:cut])
                        add_pool_route(route[cut:])
                    for p in range(length):
                        add_pool_route(route[:p] + route[p + 1:])

    def register_solution(routes, pool_fragments=False):
        nonlocal best_routes, best_cost, best_solution
        routes = [list(r) for r in routes if r]
        cost = solution_cost(routes)
        add_solution_to_pool(routes, add_fragments=pool_fragments)
        if cost < best_cost - 1e-7:
            best_cost = cost
            best_routes = [r[:] for r in routes]
            best_solution = make_solution(best_routes)
            if logger:
                logger.log_solution(best_cost, best_solution)
            return True
        return False

    add_solution_to_pool(best_routes, add_fragments=False)

    # Nearest-customer lists are used to restrict expensive local search.
    nearest = []
    for i in range(n):
        order = [j for j in range(n) if j != i]
        order.sort(key=lambda j: dist[i + 1][j + 1])
        nearest.append(order)

    # Generate a savings edge list. For larger instances, retaining nearest
    # neighbors avoids repeatedly processing a prohibitively large O(n^2)
    # list while still preserving the most useful merges.
    savings_edges = []
    if n <= 500:
        for i in range(n):
            for j in range(i + 1, n):
                saving = dist[0][i + 1] + dist[0][j + 1] - dist[i + 1][j + 1]
                savings_edges.append((saving, i, j))
    else:
        edge_set = set()
        k = min(40, n - 1)
        for i in range(n):
            for j in nearest[i][:k]:
                a, b = (i, j) if i < j else (j, i)
                edge_set.add((a, b))
        for i, j in edge_set:
            saving = dist[0][i + 1] + dist[0][j + 1] - dist[i + 1][j + 1]
            savings_edges.append((saving, i, j))

    savings_edges.sort(reverse=True, key=lambda item: item[0])

    def savings_construct(edge_order, deadline):
        routes = {i: [i] for i in range(n)}
        owner = list(range(n))
        loads = demands[:]

        for edge_number, (_, i, j) in enumerate(edge_order):
            if edge_number % 512 == 0 and time.monotonic() >= deadline:
                break

            ri = owner[i]
            rj = owner[j]
            if ri == rj or ri not in routes or rj not in routes:
                continue

            route_i = routes[ri]
            route_j = routes[rj]
            if not (i == route_i[0] or i == route_i[-1]):
                continue
            if not (j == route_j[0] or j == route_j[-1]):
                continue
            if loads[ri] + loads[rj] > capacity:
                continue

            left = route_i if route_i[-1] == i else list(reversed(route_i))
            right = route_j if route_j[0] == j else list(reversed(route_j))
            candidate = left + right

            feasible = evaluate_route(candidate)[0]
            if not feasible:
                reverse_candidate = list(reversed(candidate))
                if evaluate_route(reverse_candidate)[0]:
                    candidate = reverse_candidate
                    feasible = True

            if not feasible:
                continue

            routes[ri] = candidate
            loads[ri] += loads[rj]
            del routes[rj]
            del loads[rj]
            for customer_index in candidate:
                owner[customer_index] = ri

        return [list(route) for route in routes.values()]

    def local_search(initial_routes, deadline):
        routes = [list(r) for r in initial_routes if r]
        rounds = 0

        while rounds < 40 and time.monotonic() < deadline:
            rounds += 1
            best_delta = -1e-8
            best_move = None

            # Intra-route 2-opt.
            for r_index, route in enumerate(routes):
                if time.monotonic() >= deadline:
                    break
                if len(route) < 3:
                    continue
                old_cost = route_cost(route)
                for a in range(len(route) - 1):
                    for b in range(a + 1, len(route)):
                        candidate = route[:a] + list(reversed(route[a:b + 1])) + route[b + 1:]
                        feasible, new_cost, _ = evaluate_route(candidate)
                        if feasible:
                            delta = new_cost - old_cost
                            if delta < best_delta:
                                best_delta = delta
                                best_move = ("replace", r_index, candidate)

            # Map customers to their current routes.
            customer_route = {}
            customer_position = {}
            for r_index, route in enumerate(routes):
                for pos, customer_index in enumerate(route):
                    customer_route[customer_index] = r_index
                    customer_position[customer_index] = pos

            # Cross-route relocation, including deletion of a singleton route.
            for source_index, source in enumerate(routes):
                if time.monotonic() >= deadline:
                    break
                source_old_cost = route_cost(source)

                for source_pos, customer_index in enumerate(source):
                    reduced = source[:source_pos] + source[source_pos + 1:]
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
                        if target_index == source_index:
                            # Or-opt relocation within one route.
                            base = reduced
                            for insertion in range(len(base) + 1):
                                candidate = base[:insertion] + [customer_index] + base[insertion:]
                                if candidate == source:
                                    continue
                                feasible, new_cost, _ = evaluate_route(candidate)
                                if feasible:
                                    delta = new_cost - source_old_cost
                                    if delta < best_delta:
                                        best_delta = delta
                                        best_move = (
                                            "replace", source_index, candidate
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
                            feasible, new_target_cost, _ = evaluate_route(new_target)
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

            # Restricted cross-route swaps.
            seen_pairs = set()
            neighbor_count = min(16, max(0, n - 1))
            for x in range(n):
                if time.monotonic() >= deadline:
                    break
                rx = customer_route[x]
                px = customer_position[x]
                for y in nearest[x][:neighbor_count]:
                    ry = customer_route[y]
                    if rx == ry:
                        continue
                    pair = (x, y) if x < y else (y, x)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    py = customer_position[y]
                    route_x = routes[rx]
                    route_y = routes[ry]
                    new_x = route_x[:]
                    new_y = route_y[:]
                    new_x[px] = y
                    new_y[py] = x

                    feasible_x, cost_x, _ = evaluate_route(new_x)
                    if not feasible_x:
                        continue
                    feasible_y, cost_y, _ = evaluate_route(new_y)
                    if not feasible_y:
                        continue

                    delta = (
                        cost_x + cost_y
                        - route_cost(route_x)
                        - route_cost(route_y)
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_move = ("swap", rx, ry, new_x, new_y)

            if best_move is None:
                break

            kind = best_move[0]
            if kind == "replace":
                routes[best_move[1]] = best_move[2]
            elif kind == "relocate":
                _, source_index, target_index, new_source, new_target = best_move
                routes[source_index] = new_source
                routes[target_index] = new_target
                routes = [r for r in routes if r]
            else:
                _, rx, ry, new_x, new_y = best_move
                routes[rx] = new_x
                routes[ry] = new_y

            register_solution(routes, pool_fragments=False)

        return routes

    rng = random.Random(0)
    heuristic_deadline = start_time + max(0.0, args.time_limit * 0.72)
    heuristic_deadline = min(heuristic_deadline, hard_deadline - 0.25)

    # Deterministic Clarke-Wright construction.
    if time.monotonic() < heuristic_deadline:
        routes = savings_construct(savings_edges, heuristic_deadline)
        register_solution(routes, pool_fragments=True)

        local_budget = min(
            heuristic_deadline,
            time.monotonic() + max(0.05, min(2.5, args.time_limit * 0.14)),
        )
        routes = local_search(routes, local_budget)
        register_solution(routes, pool_fragments=True)

    # Randomized savings orders diversify the route pool and incumbents.
    iteration = 0
    while time.monotonic() < heuristic_deadline:
        iteration += 1
        randomized_edges = list(savings_edges)

        # Shuffle within rank blocks, preserving the broad savings ordering.
        block_size = 8 + (iteration * 13) % 75
        for begin in range(0, len(randomized_edges), block_size):
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

    # Solve a set-partitioning problem over all generated feasible routes.
    # This can combine routes originating from different heuristic solutions.
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
            pool_index = {route: index for index, (route, _) in enumerate(pool_items)}

            for index, (route, cost) in enumerate(pool_items):
                variable = model.addVar(
                    vtype=GRB.BINARY,
                    obj=cost,
                    name=f"route_{index}",
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

            best_keys = {tuple(route) for route in best_routes}
            for index, (route, _) in enumerate(pool_items):
                variables[index].Start = 1.0 if route in best_keys else 0.0

            callback_best = [best_cost]

            def mip_callback(cb_model, where):
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
                    actual = solution["objective_value"]
                    if actual < callback_best[0] - 1e-7:
                        callback_best[0] = actual
                        if logger:
                            logger.log_solution(actual, solution)
                except Exception:
                    pass

            model.optimize(mip_callback)

            if model.SolCount > 0:
                selected_routes = [
                    list(pool_items[i][0])
                    for i, variable in enumerate(variables)
                    if variable.X > 0.5
                ]
                register_solution(selected_routes, pool_fragments=False)

        except Exception:
            # The heuristic incumbent remains valid if Gurobi is unavailable,
            # its license cannot be initialized, or insufficient time remains.
            pass

    final_solution = make_solution(best_routes)
    best_cost = final_solution["objective_value"]

    directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(directory, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()