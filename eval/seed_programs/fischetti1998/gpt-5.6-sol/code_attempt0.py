import argparse
import heapq
import json
import os
import re
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


def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n"])
    depot = int(data["depot"])
    budget = int(data["t0"])
    nodes = list(range(1, n + 1))

    prizes = {i: 0 for i in nodes}
    for key, value in data["prizes"].items():
        prizes[int(key)] = int(value)

    travel = [[0] * (n + 1) for _ in range(n + 1)]
    for key, value in data["travel_times"].items():
        endpoints = re.findall(r"-?\d+", key)
        if len(endpoints) != 2:
            raise ValueError(f"Invalid travel-time key: {key}")
        i, j = map(int, endpoints)
        t = int(value)
        travel[i][j] = t
        travel[j][i] = t

    return n, depot, budget, nodes, prizes, travel


def route_cost(route, travel):
    return sum(travel[route[i]][route[i + 1]] for i in range(len(route) - 1))


def make_solution(route, prizes):
    visited = route[:-1]
    edges = [[route[i], route[i + 1]] for i in range(len(route) - 1)]
    objective = float(sum(prizes[i] for i in visited))
    return {
        "objective_value": objective,
        "visited_nodes": visited,
        "edges": edges,
        "tour": route,
    }


def extract_route(depot, selected, chosen_edges):
    if depot not in selected or len(selected) < 3:
        return None

    adjacency = {i: [] for i in selected}
    for i, j in chosen_edges:
        if i in adjacency and j in adjacency:
            adjacency[i].append(j)
            adjacency[j].append(i)

    if any(len(adjacency[i]) != 2 for i in selected):
        return None

    route = [depot]
    previous = None
    current = depot

    for _ in range(len(selected)):
        neighbors = adjacency[current]
        nxt = neighbors[0] if neighbors[0] != previous else neighbors[1]
        route.append(nxt)
        previous, current = current, nxt
        if current == depot:
            break

    if route[-1] != depot or len(route) != len(selected) + 1:
        return None
    if len(set(route[:-1])) != len(selected):
        return None

    return route


def push_top(heap, key, i, j, limit=6):
    entry = (key, i, j)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def improve_route(initial_route, budget, prizes, travel, deadline, mode):
    route = initial_route[:]
    current_cost = route_cost(route, travel)
    all_nodes = range(1, len(travel))
    used = set(route[:-1])

    def insertion_phase():
        nonlocal route, current_cost, used
        while time.monotonic() < deadline:
            best = None
            best_key = None

            for city in all_nodes:
                if city in used:
                    continue
                prize = prizes[city]

                for pos in range(len(route) - 1):
                    a, b = route[pos], route[pos + 1]
                    delta = travel[a][city] + travel[city][b] - travel[a][b]
                    if current_cost + delta > budget:
                        continue
                    if prize == 0 and delta >= 0:
                        continue

                    if mode == "ratio":
                        if delta <= 0:
                            key = (2, prize, -delta)
                        else:
                            key = (1, prize / float(delta), prize)
                    elif mode == "prize":
                        key = (prize, -max(delta, 0), -delta)
                    else:
                        key = (-delta, prize, -travel[a][city] - travel[city][b])

                    if best_key is None or key > best_key:
                        best_key = key
                        best = (city, pos, delta)

            if best is None:
                break

            city, pos, delta = best
            route.insert(pos + 1, city)
            used.add(city)
            current_cost += delta

    insertion_phase()

    # Reduce travel time using 2-opt, potentially creating room for more cities.
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        k = len(route) - 1
        best_delta = 0
        best_move = None

        for i in range(k):
            if time.monotonic() >= deadline:
                break
            for j in range(i + 2, k):
                if i == 0 and j == k - 1:
                    continue
                a, b = route[i], route[i + 1]
                c, d = route[j], route[j + 1]
                delta = travel[a][c] + travel[b][d] - travel[a][b] - travel[c][d]
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)

        if best_move is not None:
            i, j = best_move
            route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
            current_cost += best_delta
            improved = True

    insertion_phase()
    return route


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    n, depot, budget, nodes, prizes, travel = read_instance(args.instance_path)
    nondepot = [i for i in nodes if i != depot]

    best_solution = None
    best_objective = float("-inf")
    best_route = None

    objective_heap = []
    ratio_heap = []
    cheap_heap = []

    # Find feasible triangle seeds. Every legal tour must contain at least
    # the depot and two other cities.
    for idx, i in enumerate(nondepot):
        for j in nondepot[idx + 1:]:
            cost = travel[depot][i] + travel[i][j] + travel[j][depot]
            if cost > budget:
                continue

            value = prizes[i] + prizes[j]
            push_top(objective_heap, (value, -cost), i, j)
            push_top(ratio_heap, ((value + 1.0) / (cost + 1.0), value), i, j)
            push_top(cheap_heap, (-cost, value), i, j)

            if (best_solution is None or value > best_objective or
                    (value == best_objective and
                     cost < route_cost(best_route, travel))):
                best_route = [depot, i, j, depot]
                best_solution = make_solution(best_route, prizes)
                best_objective = best_solution["objective_value"]

    if best_solution is not None and logger:
        logger.log_solution(best_objective, best_solution)

    # Spend a small portion of the limit generating a good MIP start.
    heuristic_deadline = min(
        deadline,
        start_time + min(2.0, max(0.05, 0.15 * max(args.time_limit, 0)))
    )

    seed_pairs = []
    seen_pairs = set()
    for heap in (objective_heap, ratio_heap, cheap_heap):
        for _, i, j in sorted(heap, reverse=True):
            pair = tuple(sorted((i, j)))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                seed_pairs.append(pair)

    for i, j in seed_pairs:
        if time.monotonic() >= heuristic_deadline:
            break
        seed = [depot, i, j, depot]
        for mode in ("ratio", "prize", "cheap"):
            if time.monotonic() >= heuristic_deadline:
                break
            candidate_route = improve_route(
                seed, budget, prizes, travel, heuristic_deadline, mode
            )
            candidate = make_solution(candidate_route, prizes)
            candidate_obj = candidate["objective_value"]
            if candidate_obj > best_objective + 1e-9:
                best_objective = candidate_obj
                best_solution = candidate
                best_route = candidate_route
                if logger:
                    logger.log_solution(best_objective, best_solution)

    # If there is enough time, solve the exact undirected prize-collecting
    # tour formulation with lazy connectivity cuts.
    if n >= 3 and time.monotonic() < deadline:
        try:
            model = gp.Model("prize_collecting_tour")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.LazyConstraints = 1

            pairs = [(i, j) for i in nodes for j in nodes if i < j]
            x = model.addVars(pairs, vtype=GRB.BINARY, name="x")
            y = model.addVars(nodes, vtype=GRB.BINARY, name="y")

            model.addConstr(y[depot] == 1, name="visit_depot")
            model.addConstr(
                gp.quicksum(y[i] for i in nodes) >= 3,
                name="minimum_tour_size"
            )

            for i in nodes:
                model.addConstr(
                    gp.quicksum(
                        x[min(i, j), max(i, j)]
                        for j in nodes if j != i
                    ) == 2 * y[i],
                    name=f"degree_{i}"
                )

            model.addConstr(
                gp.quicksum(travel[i][j] * x[i, j] for i, j in pairs)
                <= budget,
                name="time_budget"
            )

            model.setObjective(
                gp.quicksum(prizes[i] * y[i] for i in nodes),
                GRB.MAXIMIZE
            )

            if best_route is not None:
                route_edges = {
                    tuple(sorted((best_route[k], best_route[k + 1])))
                    for k in range(len(best_route) - 1)
                }
                route_nodes = set(best_route[:-1])
                for i in nodes:
                    y[i].Start = 1.0 if i in route_nodes else 0.0
                for i, j in pairs:
                    x[i, j].Start = 1.0 if (i, j) in route_edges else 0.0

            state = {
                "best_objective": best_objective,
                "best_solution": best_solution,
                "best_route": best_route,
            }

            def callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                y_values = cb_model.cbGetSolution(y)
                x_values = cb_model.cbGetSolution(x)
                selected = {i for i in nodes if y_values[i] > 0.5}

                adjacency = {i: [] for i in selected}
                chosen_edges = []
                for i, j in pairs:
                    if x_values[i, j] > 0.5:
                        chosen_edges.append((i, j))
                        if i in adjacency and j in adjacency:
                            adjacency[i].append(j)
                            adjacency[j].append(i)

                # Identify connected components in the incumbent.
                components = []
                unseen = set(selected)
                while unseen:
                    root = next(iter(unseen))
                    stack = [root]
                    unseen.remove(root)
                    component = {root}
                    while stack:
                        u = stack.pop()
                        for v in adjacency.get(u, []):
                            if v in unseen:
                                unseen.remove(v)
                                component.add(v)
                                stack.append(v)
                    components.append(component)

                disconnected = False
                for component in components:
                    if depot in component:
                        continue
                    disconnected = True
                    outside = set(nodes) - component
                    cut_expr = gp.quicksum(
                        x[min(i, j), max(i, j)]
                        for i in component
                        for j in outside
                    )
                    representative = min(component)
                    cb_model.cbLazy(cut_expr >= 2 * y[representative])

                if disconnected:
                    return

                route = extract_route(depot, selected, chosen_edges)
                if route is None:
                    return
                if route_cost(route, travel) > budget:
                    return

                solution = make_solution(route, prizes)
                objective = solution["objective_value"]
                if objective > state["best_objective"] + 1e-9:
                    state["best_objective"] = objective
                    state["best_solution"] = solution
                    state["best_route"] = route
                    if logger:
                        logger.log_solution(objective, solution)

            remaining = deadline - time.monotonic()
            if remaining > 0:
                model.Params.TimeLimit = max(0.001, remaining)
                model.optimize(callback)

                best_objective = state["best_objective"]
                best_solution = state["best_solution"]
                best_route = state["best_route"]

                if model.SolCount > 0:
                    selected = {i for i in nodes if y[i].X > 0.5}
                    chosen_edges = [
                        (i, j) for i, j in pairs if x[i, j].X > 0.5
                    ]
                    final_route = extract_route(depot, selected, chosen_edges)
                    if final_route is not None and route_cost(final_route, travel) <= budget:
                        final_solution = make_solution(final_route, prizes)
                        final_obj = final_solution["objective_value"]
                        if final_obj > best_objective + 1e-9:
                            best_objective = final_obj
                            best_solution = final_solution
                            best_route = final_route
                            if logger:
                                logger.log_solution(final_obj, final_solution)

        except gp.GurobiError:
            # Return the best heuristic or incumbent solution obtained so far.
            pass

    # No legal tour exists if no feasible solution was found. The schema has
    # no infeasibility field, so emit an empty decision assignment.
    if best_solution is None:
        best_solution = {
            "objective_value": 0.0,
            "visited_nodes": [],
            "edges": [],
            "tour": [],
        }

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()