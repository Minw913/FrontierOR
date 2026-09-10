import argparse
import heapq
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


EPS = 1e-8


def shortest_path(num_nodes, arcs, adjacency, source, target, allowed, weight):
    """Return a list of arc indices forming a source-target path."""
    if source == target:
        return []

    allowed_indices = range(len(arcs)) if allowed is None else allowed
    has_negative = any(weight[a] < 0 for a in allowed_indices)

    if not has_negative:
        dist = [math.inf] * num_nodes
        pred_arc = [-1] * num_nodes
        dist[source] = 0.0
        heap = [(0.0, source)]

        while heap:
            current_dist, node = heapq.heappop(heap)
            if current_dist > dist[node] + 1e-12:
                continue
            if node == target:
                break

            for arc_idx in adjacency[node]:
                if allowed is not None and arc_idx not in allowed:
                    continue
                _, nxt = arcs[arc_idx]
                candidate = current_dist + weight[arc_idx]
                if candidate < dist[nxt] - 1e-12:
                    dist[nxt] = candidate
                    pred_arc[nxt] = arc_idx
                    heapq.heappush(heap, (candidate, nxt))

        if not math.isfinite(dist[target]):
            return None

        path = []
        node = target
        while node != source:
            arc_idx = pred_arc[node]
            if arc_idx < 0:
                return None
            path.append(arc_idx)
            node = arcs[arc_idx][0]
        path.reverse()
        return path

    # Bellman-Ford fallback for unusual instances with negative costs.
    dist = [math.inf] * num_nodes
    pred_arc = [-1] * num_nodes
    dist[source] = 0.0
    arc_list = list(range(len(arcs))) if allowed is None else list(allowed)

    for _ in range(num_nodes - 1):
        changed = False
        for arc_idx in arc_list:
            u, v = arcs[arc_idx]
            if math.isfinite(dist[u]):
                candidate = dist[u] + weight[arc_idx]
                if candidate < dist[v] - 1e-12:
                    dist[v] = candidate
                    pred_arc[v] = arc_idx
                    changed = True
        if not changed:
            break

    if not math.isfinite(dist[target]):
        return None

    path = []
    seen = set()
    node = target
    while node != source:
        if node in seen:
            return bfs_path(arcs, adjacency, source, target, allowed)
        seen.add(node)
        arc_idx = pred_arc[node]
        if arc_idx < 0:
            return bfs_path(arcs, adjacency, source, target, allowed)
        path.append(arc_idx)
        node = arcs[arc_idx][0]

    path.reverse()
    return path


def bfs_path(arcs, adjacency, source, target, allowed):
    if source == target:
        return []

    queue = [source]
    head = 0
    pred_arc = {source: -1}

    while head < len(queue):
        node = queue[head]
        head += 1
        for arc_idx in adjacency[node]:
            if allowed is not None and arc_idx not in allowed:
                continue
            _, nxt = arcs[arc_idx]
            if nxt in pred_arc:
                continue
            pred_arc[nxt] = arc_idx
            if nxt == target:
                path = []
                current = target
                while current != source:
                    a = pred_arc[current]
                    path.append(a)
                    current = arcs[a][0]
                path.reverse()
                return path
            queue.append(nxt)
    return None


def solution_from_paths(arcs, fixed_costs, variable_costs, commodities,
                        selected, paths):
    objective = sum(fixed_costs[a] for a in selected)
    for k, path in enumerate(paths):
        demand = commodities[k]["demand"]
        objective += sum(demand * variable_costs[a][k] for a in path)

    open_arcs = {
        f"{arcs[a][0]}_{arcs[a][1]}": 1
        for a in sorted(selected)
    }
    routings = {}
    for k, path in enumerate(paths):
        route = {}
        for a in path:
            key = f"{arcs[a][0]}_{arcs[a][1]}"
            route[key] = route.get(key, 0.0) + 1.0
        routings[str(k)] = route

    return {
        "objective_value": float(objective),
        "open_arcs": open_arcs,
        "routings": routings,
    }


def improve_selected_network(num_nodes, arcs, adjacency, fixed_costs,
                             variable_costs, commodities, selected):
    """Reroute commodities on the selected network and remove unused arcs."""
    selected = set(selected)
    mandatory_negative = {a for a, cost in enumerate(fixed_costs) if cost < 0}

    for _ in range(3):
        paths = []
        used = set()
        feasible = True

        for k, commodity in enumerate(commodities):
            weights = [
                commodity["demand"] * variable_costs[a][k]
                for a in range(len(arcs))
            ]
            path = shortest_path(
                num_nodes, arcs, adjacency,
                commodity["origin"], commodity["destination"],
                selected, weights
            )
            if path is None:
                feasible = False
                break
            paths.append(path)
            used.update(path)

        if not feasible:
            return None

        new_selected = used | mandatory_negative
        if new_selected == selected:
            return paths, selected
        selected = new_selected

    return paths, selected


def build_heuristics(data, arcs, adjacency, fixed_costs, variable_costs,
                     commodities, logger, start_time, time_limit):
    num_nodes = data["num_nodes"]
    num_arcs = len(arcs)
    num_commodities = len(commodities)

    orders = [
        list(range(num_commodities)),
        list(reversed(range(num_commodities))),
        sorted(range(num_commodities),
               key=lambda k: commodities[k]["demand"], reverse=True),
        sorted(range(num_commodities),
               key=lambda k: commodities[k]["demand"]),
    ]

    best_solution = None
    best_paths = None
    best_selected = None
    seen_orders = set()

    for order in orders:
        order_tuple = tuple(order)
        if order_tuple in seen_orders:
            continue
        seen_orders.add(order_tuple)

        if time.time() - start_time >= max(0.05, 0.20 * time_limit):
            break

        selected = {a for a, cost in enumerate(fixed_costs) if cost < 0}
        original_paths = [None] * num_commodities
        feasible = True

        for k in order:
            commodity = commodities[k]
            weights = [
                commodity["demand"] * variable_costs[a][k]
                + (0.0 if a in selected else fixed_costs[a])
                for a in range(num_arcs)
            ]
            path = shortest_path(
                num_nodes, arcs, adjacency,
                commodity["origin"], commodity["destination"],
                None, weights
            )
            if path is None:
                feasible = False
                break
            original_paths[k] = path
            selected.update(path)

        if not feasible:
            continue

        improved = improve_selected_network(
            num_nodes, arcs, adjacency, fixed_costs,
            variable_costs, commodities, selected
        )
        if improved is None:
            paths = original_paths
        else:
            paths, selected = improved

        candidate = solution_from_paths(
            arcs, fixed_costs, variable_costs, commodities, selected, paths
        )

        if (best_solution is None or
                candidate["objective_value"] <
                best_solution["objective_value"] - 1e-7):
            best_solution = candidate
            best_paths = paths
            best_selected = set(selected)
            if logger:
                logger.log_solution(candidate["objective_value"], candidate)

    return best_solution, best_paths, best_selected


def vector_solution(arcs, fixed_costs, variable_costs, commodities,
                    y_values, x_values):
    selected = {
        a for a, value in enumerate(y_values)
        if value >= 0.5
    }

    objective = sum(fixed_costs[a] for a in selected)
    routings = {}

    for k, commodity in enumerate(commodities):
        route = {}
        demand = commodity["demand"]
        for a in range(len(arcs)):
            value = x_values[k][a]
            if abs(value) > EPS:
                key = f"{arcs[a][0]}_{arcs[a][1]}"
                route[key] = float(value)
                objective += demand * variable_costs[a][k] * value
        routings[str(k)] = route

    open_arcs = {
        f"{arcs[a][0]}_{arcs[a][1]}": 1
        for a in sorted(selected)
    }

    return {
        "objective_value": float(objective),
        "open_arcs": open_arcs,
        "routings": routings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_nodes = int(data["num_nodes"])
    arcs = [tuple(map(int, arc)) for arc in data["arcs"]]
    num_arcs = len(arcs)
    commodities = data["commodities"]
    num_commodities = len(commodities)

    fixed_costs = []
    variable_costs = []
    for u, v in arcs:
        key = f"{u}_{v}"
        fixed_costs.append(float(data["fixed_costs"][key]))
        variable_costs.append(
            [float(value) for value in data["variable_costs"][key]]
        )

    adjacency = [[] for _ in range(num_nodes)]
    incoming = [[] for _ in range(num_nodes)]
    outgoing = [[] for _ in range(num_nodes)]
    for a, (u, v) in enumerate(arcs):
        adjacency[u].append(a)
        outgoing[u].append(a)
        incoming[v].append(a)

    best_solution, heuristic_paths, heuristic_selected = build_heuristics(
        data, arcs, adjacency, fixed_costs, variable_costs,
        commodities, logger, start_time, args.time_limit
    )

    if best_solution is None:
        raise RuntimeError("The instance has no directed path for at least one commodity.")

    remaining = args.time_limit - (time.time() - start_time)

    if remaining > 0.05:
        try:
            model = gp.Model("uncapacitated_fixed_charge_network_design")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, remaining)

            y = [
                model.addVar(vtype=GRB.BINARY, obj=fixed_costs[a])
                for a in range(num_arcs)
            ]
            x = [
                [
                    model.addVar(
                        lb=0.0,
                        ub=1.0,
                        vtype=GRB.CONTINUOUS,
                        obj=commodities[k]["demand"] * variable_costs[a][k]
                    )
                    for a in range(num_arcs)
                ]
                for k in range(num_commodities)
            ]

            model.ModelSense = GRB.MINIMIZE

            for k, commodity in enumerate(commodities):
                origin = int(commodity["origin"])
                destination = int(commodity["destination"])

                for node in range(num_nodes):
                    rhs = -1.0 if node == origin else (
                        1.0 if node == destination else 0.0
                    )
                    model.addConstr(
                        gp.quicksum(x[k][a] for a in incoming[node])
                        - gp.quicksum(x[k][a] for a in outgoing[node])
                        == rhs
                    )

                for a in range(num_arcs):
                    model.addConstr(x[k][a] <= y[a])

            model.update()

            for a in range(num_arcs):
                y[a].Start = 1.0 if a in heuristic_selected else 0.0

            path_sets = [set(path) for path in heuristic_paths]
            for k in range(num_commodities):
                for a in range(num_arcs):
                    x[k][a].Start = 1.0 if a in path_sets[k] else 0.0

            callback_best = [best_solution["objective_value"]]

            def incumbent_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                incumbent_obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if incumbent_obj >= callback_best[0] - 1e-7:
                    return

                y_values = cb_model.cbGetSolution(y)
                x_values = [
                    cb_model.cbGetSolution(x[k])
                    for k in range(num_commodities)
                ]
                solution = vector_solution(
                    arcs, fixed_costs, variable_costs, commodities,
                    y_values, x_values
                )
                callback_best[0] = solution["objective_value"]

                if logger:
                    logger.log_solution(solution["objective_value"], solution)

            model.optimize(incumbent_callback)

            if model.SolCount > 0:
                y_values = [var.X for var in y]
                x_values = [
                    [x[k][a].X for a in range(num_arcs)]
                    for k in range(num_commodities)
                ]
                mip_solution = vector_solution(
                    arcs, fixed_costs, variable_costs, commodities,
                    y_values, x_values
                )

                if (mip_solution["objective_value"] <
                        best_solution["objective_value"] - 1e-7):
                    best_solution = mip_solution
                    if logger and (
                        mip_solution["objective_value"] <
                        callback_best[0] - 1e-7
                    ):
                        logger.log_solution(
                            mip_solution["objective_value"], mip_solution
                        )

        except gp.GurobiError:
            # The constructive solution remains available if optimization
            # cannot start or is interrupted by an environment-level error.
            pass

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()