import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


EPS = 1e-7


def parse_pair_key(key):
    a, b = key.split(",")
    return int(a.strip()), int(b.strip())


def clean_number(value):
    value = float(value)
    nearest = round(value)
    if abs(value - nearest) <= 1e-7:
        return int(nearest)
    return float(round(value, 10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_vertices = int(data["num_vertices"])
    num_products = int(data["num_products"])
    depot = int(data["depot"])

    vertices = list(range(num_vertices))
    markets = [v for v in vertices if v != depot]
    products = list(range(num_products))

    travel_cost = {}
    for key, value in data["travel_costs"].items():
        i, j = parse_pair_key(key)
        if i > j:
            i, j = j, i
        travel_cost[i, j] = float(value)

    def edge_cost(i, j):
        if i == j:
            return 0.0
        return travel_cost[(min(i, j), max(i, j))]

    demands = [float(v) for v in data["product_demands"]]

    product_markets = {
        p: [int(v) for v in data["product_markets"].get(str(p), [])]
        for p in products
    }

    purchase_cost = {}
    for key, value in data["purchase_costs"].items():
        p, j = parse_pair_key(key)
        purchase_cost[p, j] = float(value)

    supplies = {}
    for key, value in data["supplies"].items():
        p, j = parse_pair_key(key)
        supplies[p, j] = float(value)

    available_pairs = [
        (p, j)
        for p in products
        for j in product_markets[p]
        if (p, j) in supplies
    ]

    edges = [(i, j) for i in vertices for j in vertices if i < j]

    def allocate_purchases(selected_markets):
        selected = set(selected_markets)
        allocation = {}

        for p in products:
            candidates = [
                j for j in product_markets[p]
                if j in selected and (p, j) in supplies
            ]
            total_supply = sum(supplies[p, j] for j in candidates)
            if total_supply + 1e-8 < demands[p]:
                return None

            candidates.sort(
                key=lambda j: (
                    purchase_cost[p, j],
                    edge_cost(depot, j),
                    j,
                )
            )

            remaining = demands[p]
            for j in candidates:
                if remaining <= EPS:
                    break
                amount = min(remaining, supplies[p, j])
                if amount > EPS:
                    allocation[p, j] = amount
                    remaining -= amount

            if remaining > 1e-6:
                return None

        return allocation

    def nearest_neighbor_tour(selected_markets):
        remaining = set(selected_markets)
        tour = [depot]
        current = depot

        while remaining:
            nxt = min(
                remaining,
                key=lambda j: (edge_cost(current, j), edge_cost(depot, j), j)
            )
            tour.append(nxt)
            remaining.remove(nxt)
            current = nxt

        tour.append(depot)
        return tour

    def improve_tour_2opt(tour):
        if len(tour) <= 4:
            return tour

        tour = list(tour)
        max_passes = 8
        max_checks = 50000
        checks = 0

        for _ in range(max_passes):
            improved = False
            n = len(tour) - 1

            for i in range(1, n - 1):
                if time.monotonic() >= deadline:
                    return tour
                a, b = tour[i - 1], tour[i]

                for k in range(i + 1, n):
                    checks += 1
                    if checks > max_checks or time.monotonic() >= deadline:
                        return tour

                    c, d = tour[k], tour[k + 1]
                    old_cost = edge_cost(a, b) + edge_cost(c, d)
                    new_cost = edge_cost(a, c) + edge_cost(b, d)

                    if new_cost + 1e-9 < old_cost:
                        tour[i:k + 1] = reversed(tour[i:k + 1])
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return tour

    def repair_purchases(raw_values, selected_markets):
        selected = set(selected_markets)
        repaired = {}

        for p in products:
            candidates = [
                j for j in product_markets[p]
                if j in selected and (p, j) in supplies
            ]

            for j in candidates:
                value = float(raw_values.get((p, j), 0.0))
                value = min(max(value, 0.0), supplies[p, j])
                if abs(value - round(value)) <= 1e-7:
                    value = float(round(value))
                if value > EPS:
                    repaired[p, j] = value

            residual = demands[p] - sum(
                repaired.get((p, j), 0.0) for j in candidates
            )

            if residual > 1e-8:
                ordered = sorted(
                    candidates,
                    key=lambda j: (purchase_cost[p, j], j)
                )
                for j in ordered:
                    current = repaired.get((p, j), 0.0)
                    room = supplies[p, j] - current
                    addition = min(room, residual)
                    if addition > 0:
                        repaired[p, j] = current + addition
                        residual -= addition
                    if residual <= 1e-8:
                        break

            elif residual < -1e-8:
                excess = -residual
                ordered = sorted(
                    candidates,
                    key=lambda j: (-purchase_cost[p, j], j)
                )
                for j in ordered:
                    current = repaired.get((p, j), 0.0)
                    reduction = min(current, excess)
                    new_value = current - reduction
                    if new_value <= EPS:
                        repaired.pop((p, j), None)
                    else:
                        repaired[p, j] = new_value
                    excess -= reduction
                    if excess <= 1e-8:
                        break

            final_total = sum(repaired.get((p, j), 0.0) for j in candidates)
            final_residual = demands[p] - final_total
            if abs(final_residual) > 1e-7 and candidates:
                if final_residual > 0:
                    for j in candidates:
                        current = repaired.get((p, j), 0.0)
                        if current + final_residual <= supplies[p, j] + 1e-7:
                            repaired[p, j] = current + final_residual
                            break
                else:
                    for j in candidates:
                        current = repaired.get((p, j), 0.0)
                        if current + final_residual >= -1e-7:
                            new_value = current + final_residual
                            if new_value <= EPS:
                                repaired.pop((p, j), None)
                            else:
                                repaired[p, j] = new_value
                            break

        return repaired

    def build_solution(tour, raw_purchases):
        visited = list(tour[1:-1])
        purchases = repair_purchases(raw_purchases, visited)

        tour_edges = []
        travel_total = 0.0
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            travel_total += edge_cost(a, b)
            tour_edges.append([min(a, b), max(a, b)])

        purchase_total = sum(
            purchase_cost[p, j] * amount
            for (p, j), amount in purchases.items()
        )
        objective = travel_total + purchase_total

        purchase_output = {str(p): {} for p in products}
        for p in products:
            for j in visited:
                amount = purchases.get((p, j), 0.0)
                if amount > EPS:
                    purchase_output[str(p)][str(j)] = clean_number(amount)

        return {
            "objective_value": float(objective),
            "visited_markets": [int(j) for j in visited],
            "tour": [int(v) for v in tour],
            "tour_edges": tour_edges,
            "purchases": purchase_output,
        }

    def tour_from_edge_values(selected_markets, edge_values):
        active = set(selected_markets)
        active.add(depot)
        adjacency = {v: [] for v in active}

        for (i, j), value in edge_values.items():
            if value > 0.5 and i in active and j in active:
                adjacency[i].append(j)
                adjacency[j].append(i)

        if any(len(adjacency[v]) != 2 for v in active):
            return None

        tour = [depot]
        previous = None
        current = depot

        while True:
            neighbors = adjacency[current]
            nxt = neighbors[0] if neighbors[0] != previous else neighbors[1]

            if nxt == depot:
                tour.append(depot)
                break

            if nxt in tour:
                return None

            tour.append(nxt)
            previous, current = current, nxt

            if len(tour) > len(active) + 1:
                return None

        if set(tour[:-1]) != active or len(tour) != len(active) + 1:
            return None

        return tour

    best_solution = None
    best_objective = math.inf

    def accept_solution(solution):
        nonlocal best_solution, best_objective
        objective = float(solution["objective_value"])
        if objective + 1e-7 < best_objective:
            best_objective = objective
            best_solution = solution
            if logger:
                logger.log_solution(objective, solution)
            return True
        return False

    # Heuristic 1: use markets selected by globally cheapest product allocations.
    global_allocation = allocate_purchases(markets)
    selected_by_purchase = set()
    if global_allocation is not None:
        selected_by_purchase.update(j for (_, j), qv in global_allocation.items() if qv > EPS)

    # An undirected simple depot cycle requires at least two markets.
    if len(selected_by_purchase) < 2:
        additions = sorted(
            (j for j in markets if j not in selected_by_purchase),
            key=lambda j: (edge_cost(depot, j), j)
        )
        while len(selected_by_purchase) < 2 and additions:
            selected_by_purchase.add(additions.pop(0))

    heuristic_market_sets = []
    if selected_by_purchase:
        heuristic_market_sets.append(sorted(selected_by_purchase))
    heuristic_market_sets.append(list(markets))

    seen_sets = set()
    for selected in heuristic_market_sets:
        signature = tuple(sorted(selected))
        if signature in seen_sets:
            continue
        seen_sets.add(signature)

        allocation = allocate_purchases(selected)
        if allocation is None or len(selected) < 2:
            continue

        tour = nearest_neighbor_tour(selected)
        tour = improve_tour_2opt(tour)
        accept_solution(build_solution(tour, allocation))

    if best_solution is None:
        raise RuntimeError("Could not construct an initial feasible solution")

    remaining_time = deadline - time.monotonic()

    if remaining_time > 0.05:
        model = gp.Model("purchasing_tour")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.LazyConstraints = 1
        model.Params.TimeLimit = max(0.01, remaining_time)

        y = model.addVars(markets, vtype=GRB.BINARY, name="visit")
        x = model.addVars(edges, vtype=GRB.BINARY, name="edge")
        q = model.addVars(available_pairs, lb=0.0, vtype=GRB.CONTINUOUS, name="buy")

        incident_edges = {v: [] for v in vertices}
        for i, j in edges:
            incident_edges[i].append((i, j))
            incident_edges[j].append((i, j))

        model.addConstr(
            gp.quicksum(x[e] for e in incident_edges[depot]) == 2
        )

        for j in markets:
            model.addConstr(
                gp.quicksum(x[e] for e in incident_edges[j]) == 2 * y[j]
            )

        for p in products:
            model.addConstr(
                gp.quicksum(q[p, j] for j in product_markets[p]) == demands[p]
            )
            for j in product_markets[p]:
                model.addConstr(q[p, j] <= supplies[p, j] * y[j])

        # Fix markets that are indispensable for at least one product.
        for j in markets:
            mandatory = False
            for p in products:
                if j not in product_markets[p]:
                    continue
                other_supply = sum(
                    supplies[p, k]
                    for k in product_markets[p]
                    if k != j
                )
                if other_supply + 1e-9 < demands[p]:
                    mandatory = True
                    break
            if mandatory:
                y[j].LB = 1.0

        model.setObjective(
            gp.quicksum(edge_cost(i, j) * x[i, j] for i, j in edges)
            + gp.quicksum(
                purchase_cost[p, j] * q[p, j] for p, j in available_pairs
            ),
            GRB.MINIMIZE,
        )

        # Supply the best heuristic solution as a MIP start.
        start_visited = set(best_solution["visited_markets"])
        start_edge_set = {
            (min(a, b), max(a, b))
            for a, b in best_solution["tour_edges"]
        }
        start_purchase = {}
        for p_str, entries in best_solution["purchases"].items():
            p = int(p_str)
            for j_str, amount in entries.items():
                start_purchase[p, int(j_str)] = float(amount)

        for j in markets:
            y[j].Start = 1.0 if j in start_visited else 0.0
        for e in edges:
            x[e].Start = 1.0 if e in start_edge_set else 0.0
        for pair in available_pairs:
            q[pair].Start = start_purchase.get(pair, 0.0)

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                y_values = cb_model.cbGetSolution(y)
                x_values = cb_model.cbGetSolution(x)

                selected = [j for j in markets if y_values[j] > 0.5]
                active = set(selected)
                active.add(depot)

                adjacency = {v: [] for v in active}
                for (i, j), value in x_values.items():
                    if value > 0.5 and i in active and j in active:
                        adjacency[i].append(j)
                        adjacency[j].append(i)

                # Find connected components of the incumbent edge set.
                components = []
                unseen = set(active)
                while unseen:
                    root = next(iter(unseen))
                    stack = [root]
                    unseen.remove(root)
                    component = {root}

                    while stack:
                        v = stack.pop()
                        for w in adjacency[v]:
                            if w in unseen:
                                unseen.remove(w)
                                component.add(w)
                                stack.append(w)

                    components.append(component)

                if len(components) > 1:
                    for component in components:
                        if depot in component:
                            continue
                        representative = next(iter(component))
                        crossing = [
                            x[min(i, j), max(i, j)]
                            for i in component
                            for j in vertices
                            if j not in component
                        ]
                        cb_model.cbLazy(
                            gp.quicksum(crossing) >= 2 * y[representative]
                        )
                    return

                tour = tour_from_edge_values(selected, x_values)
                if tour is None:
                    return

                q_values = cb_model.cbGetSolution(q)
                raw_purchases = {
                    pair: value for pair, value in q_values.items()
                    if value > EPS
                }
                accept_solution(build_solution(tour, raw_purchases))

            except gp.GurobiError:
                return

        model.optimize(callback)

        if model.SolCount > 0:
            selected = [j for j in markets if y[j].X > 0.5]
            final_edge_values = {e: x[e].X for e in edges}
            tour = tour_from_edge_values(selected, final_edge_values)

            if tour is not None:
                raw_purchases = {
                    pair: q[pair].X
                    for pair in available_pairs
                    if q[pair].X > EPS
                }
                accept_solution(build_solution(tour, raw_purchases))

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()