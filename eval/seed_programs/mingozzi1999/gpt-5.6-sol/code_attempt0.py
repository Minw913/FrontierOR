import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def route_cost(route, distance):
    return float(sum(distance[route[i]][route[i + 1]]
                     for i in range(len(route) - 1)))


def solution_from_routes(routes, distance):
    objective = float(sum(route_cost(route, distance) for route in routes))
    return {
        "objective_value": objective,
        "routes": routes,
    }


def first_fit_decreasing(customers, demands, capacity, max_bins):
    bins = []
    loads = []
    ordered = sorted(customers, key=lambda i: (-demands[i], i))

    for customer in ordered:
        demand = demands[customer]
        candidates = [
            k for k in range(len(bins))
            if loads[k] + demand <= capacity
        ]
        if candidates:
            # Best fit: leave the least residual capacity.
            k = min(candidates,
                    key=lambda b: capacity - loads[b] - demand)
            bins[k].append(customer)
            loads[k] += demand
        else:
            bins.append([customer])
            loads.append(demand)
            if len(bins) > max_bins:
                return None
    return bins


def backtracking_pack(customers, demands, capacity, max_bins, deadline):
    ordered = sorted(customers, key=lambda i: (-demands[i], i))
    n = len(ordered)
    bins = [[] for _ in range(max_bins)]
    loads = [0] * max_bins

    suffix = [0] * (n + 1)
    for p in range(n - 1, -1, -1):
        suffix[p] = suffix[p + 1] + demands[ordered[p]]

    calls = 0

    def search(pos):
        nonlocal calls
        calls += 1
        if calls % 1024 == 0 and time.monotonic() >= deadline:
            return False
        if pos == n:
            return True

        if suffix[pos] > sum(capacity - load for load in loads):
            return False

        customer = ordered[pos]
        demand = demands[customer]
        seen_loads = set()

        # Trying fuller bins first often gives a good bin-packing search.
        candidates = sorted(range(max_bins), key=lambda k: (-loads[k], k))
        for k in candidates:
            if loads[k] in seen_loads:
                continue
            seen_loads.add(loads[k])

            if loads[k] + demand > capacity:
                continue

            was_empty = not bins[k]
            bins[k].append(customer)
            loads[k] += demand

            if search(pos + 1):
                return True

            loads[k] -= demand
            bins[k].pop()

            # Empty bins are symmetric.
            if was_empty:
                break

        return False

    if search(0):
        return [list(b) for b in bins if b]
    return None


def ensure_exact_number_of_linehaul_bins(bins, number):
    bins = [list(b) for b in bins]
    while len(bins) < number:
        source = max(
            (k for k in range(len(bins)) if len(bins[k]) > 1),
            key=lambda k: len(bins[k]),
            default=None,
        )
        if source is None:
            return None
        bins.append([bins[source].pop()])
    return bins


def pack_with_gurobi(customers, demands, capacity, number_bins,
                     require_nonempty, time_limit):
    if not customers:
        return [[] for _ in range(number_bins)] if not require_nonempty else None
    if time_limit <= 0.02:
        return None

    model = gp.Model("packing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.01, time_limit)

    z = model.addVars(customers, range(number_bins), vtype=GRB.BINARY,
                      name="z")

    for customer in customers:
        model.addConstr(
            gp.quicksum(z[customer, k] for k in range(number_bins)) == 1
        )

    for k in range(number_bins):
        model.addConstr(
            gp.quicksum(demands[i] * z[i, k] for i in customers) <= capacity
        )
        if require_nonempty:
            model.addConstr(
                gp.quicksum(z[i, k] for i in customers) >= 1
            )

    # A tiny deterministic objective encourages low-index bins to be filled.
    model.setObjective(
        gp.quicksum(k * z[i, k] for i in customers
                    for k in range(number_bins)),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.SolCount == 0:
        return None

    bins = [[] for _ in range(number_bins)]
    for i in customers:
        selected = max(range(number_bins), key=lambda k: z[i, k].X)
        bins[selected].append(i)
    return bins


def make_packing(customers, demands, capacity, number_bins,
                 require_exact, deadline):
    if any(demands[i] > capacity for i in customers):
        return None
    if require_exact and len(customers) < number_bins:
        return None

    bins = first_fit_decreasing(customers, demands, capacity, number_bins)

    if bins is None:
        dfs_deadline = min(deadline, time.monotonic() + 1.0)
        bins = backtracking_pack(
            customers, demands, capacity, number_bins, dfs_deadline
        )

    if bins is not None:
        if require_exact:
            bins = ensure_exact_number_of_linehaul_bins(bins, number_bins)
            if bins is None:
                return None
        while len(bins) < number_bins:
            bins.append([])
        return bins

    remaining = deadline - time.monotonic()
    if remaining <= 0.02:
        return None
    return pack_with_gurobi(
        customers,
        demands,
        capacity,
        number_bins,
        require_exact,
        min(3.0, remaining),
    )


def nearest_neighbor(nodes, start, depot, distance):
    remaining = set(nodes)
    result = []
    current = start
    while remaining:
        nxt = min(
            remaining,
            key=lambda j: (distance[current][j], distance[j][depot], j)
        )
        result.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return result


def hungarian(cost):
    """Minimum-cost assignment. Returns column assigned to each row."""
    n = len(cost)
    if n == 0:
        return []

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        minv = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        j0 = 0

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0

            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = float(cost[i0 - 1][j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1

            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def improve_route_order(route, num_linehaul_nodes, distance, deadline):
    # route is [depot, customers..., depot]. Reversals remain within a phase.
    if len(route) <= 3:
        return route

    best = list(route)
    best_cost = route_cost(best, distance)
    customer_count = len(best) - 2
    phase_ranges = []

    if num_linehaul_nodes >= 2:
        phase_ranges.append((1, num_linehaul_nodes))
    backhaul_count = customer_count - num_linehaul_nodes
    if backhaul_count >= 2:
        phase_ranges.append(
            (1 + num_linehaul_nodes, customer_count)
        )

    # Limit expensive full reversal search on very large routes.
    exhaustive = customer_count <= 60
    max_passes = 5 if exhaustive else 2

    for _ in range(max_passes):
        if time.monotonic() >= deadline:
            break
        improved = False

        for left_bound, right_bound in phase_ranges:
            for left in range(left_bound, right_bound):
                max_right = right_bound if exhaustive else min(
                    right_bound, left + 1
                )
                for right in range(left + 1, max_right + 1):
                    candidate = (
                        best[:left]
                        + list(reversed(best[left:right + 1]))
                        + best[right + 1:]
                    )
                    candidate_cost = route_cost(candidate, distance)
                    if candidate_cost + 1e-9 < best_cost:
                        best = candidate
                        best_cost = candidate_cost
                        improved = True
                        break
                if improved or time.monotonic() >= deadline:
                    break
            if improved or time.monotonic() >= deadline:
                break

        if not improved:
            break

    return best


def construct_heuristic_routes(linehaul_bins, backhaul_bins, depot,
                               distance, deadline):
    k_routes = len(linehaul_bins)

    line_orders = [
        nearest_neighbor(nodes, depot, depot, distance)
        for nodes in linehaul_bins
    ]

    pair_cost = [[0.0] * k_routes for _ in range(k_routes)]
    candidate_back_orders = [[None] * k_routes for _ in range(k_routes)]

    for r in range(k_routes):
        end_linehaul = line_orders[r][-1]
        for b in range(k_routes):
            order = nearest_neighbor(
                backhaul_bins[b], end_linehaul, depot, distance
            )
            candidate_back_orders[r][b] = order
            sequence = [end_linehaul] + order + [depot]
            pair_cost[r][b] = sum(
                distance[sequence[t]][sequence[t + 1]]
                for t in range(len(sequence) - 1)
            )

    assignment = hungarian(pair_cost)
    routes = []

    for r in range(k_routes):
        b = assignment[r]
        back_order = candidate_back_orders[r][b]
        route = [depot] + line_orders[r] + back_order + [depot]
        route = improve_route_order(
            route, len(line_orders[r]), distance, deadline
        )
        routes.append(route)

    return routes


def extract_routes_from_values(x_keys, values, depot, number_routes):
    successor = {}
    outgoing_depot = []

    for (i, j), value in zip(x_keys, values):
        if value > 0.5:
            successor[i] = j
            if i == depot:
                outgoing_depot.append(j)

    outgoing_depot.sort()
    if len(outgoing_depot) != number_routes:
        return None

    routes = []
    globally_seen = set()

    for first in outgoing_depot:
        route = [depot, first]
        current = first
        local_seen = set()

        while current != depot:
            if current in local_seen:
                return None
            local_seen.add(current)
            globally_seen.add(current)

            if current not in successor:
                return None
            current = successor[current]
            route.append(current)

            if len(route) > len(x_keys) + 2:
                return None

        routes.append(route)

    return routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    depot = int(instance["depot"]["index"])
    linehauls = [int(c["index"]) for c in instance["linehaul_customers"]]
    backhauls = [int(c["index"]) for c in instance["backhaul_customers"]]
    all_customers = linehauls + backhauls
    linehaul_set = set(linehauls)
    backhaul_set = set(backhauls)

    number_routes = int(instance["num_vehicles"])
    capacity = int(instance["vehicle_capacity"])
    distance = instance["distance_matrix"]

    linehaul_demand = {
        int(c["index"]): int(c["demand"])
        for c in instance["linehaul_customers"]
    }
    backhaul_demand = {
        int(c["index"]): int(c["demand"])
        for c in instance["backhaul_customers"]
    }

    if number_routes <= 0:
        if all_customers:
            raise ValueError("A positive number of vehicles is required.")
        solution = {"objective_value": 0.0, "routes": []}
        if logger:
            logger.log_solution(0.0, solution)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    if len(linehauls) < number_routes:
        raise ValueError(
            "Infeasible instance: each route needs a linehaul customer."
        )

    # Construct a feasible incumbent before starting the routing MIP.
    packing_deadline = min(
        deadline,
        time.monotonic() + max(0.5, min(4.0, 0.15 * args.time_limit)),
    )

    linehaul_bins = make_packing(
        linehauls,
        linehaul_demand,
        capacity,
        number_routes,
        True,
        packing_deadline,
    )
    backhaul_bins = make_packing(
        backhauls,
        backhaul_demand,
        capacity,
        number_routes,
        False,
        packing_deadline,
    )

    best_solution = None
    best_objective = math.inf

    if linehaul_bins is not None and backhaul_bins is not None:
        heuristic_deadline = min(
            deadline,
            time.monotonic() + max(0.1, min(2.0, 0.05 * args.time_limit)),
        )
        heuristic_routes = construct_heuristic_routes(
            linehaul_bins, backhaul_bins, depot, distance, heuristic_deadline
        )
        best_solution = solution_from_routes(heuristic_routes, distance)
        best_objective = best_solution["objective_value"]
        if logger:
            logger.log_solution(best_objective, best_solution)

    remaining = deadline - time.monotonic()

    if remaining > 0.05:
        model = gp.Model("vehicle_routing_backhauls")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.01, remaining)

        allowed_arcs = []

        # Depot can only depart toward a linehaul customer.
        for j in linehauls:
            allowed_arcs.append((depot, j))

        # A linehaul can lead to another linehaul, a backhaul, or the depot.
        for i in linehauls:
            for j in linehauls:
                if i != j:
                    allowed_arcs.append((i, j))
            for j in backhauls:
                allowed_arcs.append((i, j))
            allowed_arcs.append((i, depot))

        # A backhaul can only lead to another backhaul or the depot.
        for i in backhauls:
            for j in backhauls:
                if i != j:
                    allowed_arcs.append((i, j))
            allowed_arcs.append((i, depot))

        x = model.addVars(allowed_arcs, vtype=GRB.BINARY, name="x")

        outgoing = {i: [] for i in [depot] + all_customers}
        incoming = {i: [] for i in [depot] + all_customers}
        for i, j in allowed_arcs:
            outgoing[i].append(x[i, j])
            incoming[j].append(x[i, j])

        for i in all_customers:
            model.addConstr(gp.quicksum(outgoing[i]) == 1,
                            name=f"out_{i}")
            model.addConstr(gp.quicksum(incoming[i]) == 1,
                            name=f"in_{i}")

        model.addConstr(
            gp.quicksum(outgoing[depot]) == number_routes,
            name="depot_departures",
        )
        model.addConstr(
            gp.quicksum(incoming[depot]) == number_routes,
            name="depot_returns",
        )

        # Generic order variables eliminate customer-only subtours, including
        # subtours consisting entirely of zero-demand customers.
        n_customers = len(all_customers)
        order = model.addVars(
            all_customers,
            lb=1.0,
            ub=float(max(1, n_customers)),
            vtype=GRB.CONTINUOUS,
            name="order",
        )

        for i, j in allowed_arcs:
            if i != depot and j != depot:
                model.addConstr(
                    order[j] >= order[i] + 1
                    - n_customers * (1 - x[i, j]),
                    name=f"order_{i}_{j}",
                )

        # Cumulative linehaul load. It resets at the depot and no linehaul is
        # permitted after entering the backhaul phase.
        line_load = model.addVars(
            linehauls,
            lb={i: linehaul_demand[i] for i in linehauls},
            ub=capacity,
            vtype=GRB.CONTINUOUS,
            name="line_load",
        )
        for i in linehauls:
            for j in linehauls:
                if i == j:
                    continue
                model.addConstr(
                    line_load[j] >= line_load[i] + linehaul_demand[j]
                    - capacity * (1 - x[i, j]),
                    name=f"line_capacity_{i}_{j}",
                )

        # Backhaul collection load starts independently when the route enters
        # the backhaul phase.
        back_load = model.addVars(
            backhauls,
            lb={i: backhaul_demand[i] for i in backhauls},
            ub=capacity,
            vtype=GRB.CONTINUOUS,
            name="back_load",
        )
        for i in backhauls:
            for j in backhauls:
                if i == j:
                    continue
                model.addConstr(
                    back_load[j] >= back_load[i] + backhaul_demand[j]
                    - capacity * (1 - x[i, j]),
                    name=f"back_capacity_{i}_{j}",
                )

        model.setObjective(
            gp.quicksum(distance[i][j] * x[i, j]
                        for i, j in allowed_arcs),
            GRB.MINIMIZE,
        )

        # Supply the heuristic solution as a complete arc MIP start.
        if best_solution is not None:
            selected = set()
            position = {}
            line_cumulative = {}
            back_cumulative = {}

            for route in best_solution["routes"]:
                for t in range(len(route) - 1):
                    selected.add((route[t], route[t + 1]))

                cumulative_l = 0
                cumulative_b = 0
                for pos, node in enumerate(route[1:-1], start=1):
                    position[node] = pos
                    if node in linehaul_set:
                        cumulative_l += linehaul_demand[node]
                        line_cumulative[node] = cumulative_l
                    else:
                        cumulative_b += backhaul_demand[node]
                        back_cumulative[node] = cumulative_b

            for arc in allowed_arcs:
                x[arc].Start = 1.0 if arc in selected else 0.0
            for i in all_customers:
                order[i].Start = float(position[i])
            for i in linehauls:
                line_load[i].Start = float(line_cumulative[i])
            for i in backhauls:
                back_load[i].Start = float(back_cumulative[i])

        x_keys = list(allowed_arcs)
        x_vars = [x[arc] for arc in x_keys]
        callback_best = [best_objective]

        def incumbent_callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                values = cb_model.cbGetSolution(x_vars)
                routes = extract_routes_from_values(
                    x_keys, values, depot, number_routes
                )
                if routes is None:
                    return

                solution = solution_from_routes(routes, distance)
                objective = solution["objective_value"]

                if objective + 1e-8 < callback_best[0]:
                    callback_best[0] = objective
                    if logger:
                        logger.log_solution(objective, solution)
            except Exception:
                # Logging must never interrupt optimization.
                return

        model.optimize(incumbent_callback)

        if model.SolCount > 0:
            values = [var.X for var in x_vars]
            routes = extract_routes_from_values(
                x_keys, values, depot, number_routes
            )
            if routes is not None:
                mip_solution = solution_from_routes(routes, distance)
                if mip_solution["objective_value"] + 1e-8 < best_objective:
                    best_solution = mip_solution
                    best_objective = mip_solution["objective_value"]
                    if logger and best_objective + 1e-8 < callback_best[0]:
                        logger.log_solution(best_objective, best_solution)

    if best_solution is None:
        raise RuntimeError(
            "No feasible solution was found within the time limit."
        )

    # Recompute the reported objective from the actual output routes.
    best_solution = solution_from_routes(best_solution["routes"], distance)

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f)


if __name__ == "__main__":
    main()