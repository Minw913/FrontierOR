import argparse
import heapq
import json
import math
import time
from collections import deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def strongly_connected_components(num_nodes, arcs_uv):
    out_adj = [[] for _ in range(num_nodes)]
    rev_adj = [[] for _ in range(num_nodes)]

    for u, v in arcs_uv:
        out_adj[u].append(v)
        rev_adj[v].append(u)

    visited = [False] * num_nodes
    order = []

    for root in range(num_nodes):
        if visited[root]:
            continue

        visited[root] = True
        stack = [(root, 0)]

        while stack:
            u, next_idx = stack[-1]
            if next_idx < len(out_adj[u]):
                v = out_adj[u][next_idx]
                stack[-1] = (u, next_idx + 1)
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, 0))
            else:
                order.append(u)
                stack.pop()

    component = [-1] * num_nodes
    component_id = 0

    for root in reversed(order):
        if component[root] != -1:
            continue

        component[root] = component_id
        stack = [root]

        while stack:
            u = stack.pop()
            for v in rev_adj[u]:
                if component[v] == -1:
                    component[v] = component_id
                    stack.append(v)

        component_id += 1

    return component


def shortest_arc_path(source, target, adjacency, arc_weights):
    if source == target:
        return []

    n = len(adjacency)
    distance = [math.inf] * n
    predecessor_arc = [-1] * n
    distance[source] = 0.0
    heap = [(0.0, source)]

    while heap:
        dist_u, u = heapq.heappop(heap)
        if dist_u != distance[u]:
            continue
        if u == target:
            break

        for arc_idx, v in adjacency[u]:
            candidate = dist_u + arc_weights[arc_idx]
            if candidate < distance[v] - 1e-12:
                distance[v] = candidate
                predecessor_arc[v] = arc_idx
                heapq.heappush(heap, (candidate, v))

    if not math.isfinite(distance[target]):
        return None

    path = []
    current = target
    while current != source:
        arc_idx = predecessor_arc[current]
        if arc_idx < 0:
            return None
        path.append(arc_idx)
        current = adjacency.arc_from[arc_idx]

    path.reverse()
    return path


class ArcAdjacency(list):
    pass


def make_solution(instance, x_values, y_values, objective_value):
    arcs = instance["arcs"]
    commodities = instance["commodities"]
    fleet_types = instance["fleet_types"]

    flows = {}
    for k, commodity in enumerate(commodities):
        commodity_id = commodity["id"]
        for a, arc in enumerate(arcs):
            key = f"x_{commodity_id}_{arc['from']}_{arc['to']}"
            value = float(x_values[k][a])
            if abs(value) < 1e-10:
                value = 0.0
            flows[key] = max(0.0, value)

    vehicles = {}
    for t, fleet_type in enumerate(fleet_types):
        fleet_id = fleet_type["id"]
        for a, arc in enumerate(arcs):
            key = f"y_{fleet_id}_{arc['from']}_{arc['to']}"
            value = int(round(y_values[t][a]))
            vehicles[key] = max(0, value)

    return {
        "objective_value": float(objective_value),
        "flows": flows,
        "vehicles": vehicles,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    nodes = instance["nodes"]
    arcs = instance["arcs"]
    commodities = instance["commodities"]
    fleet_types = instance["fleet_types"]

    num_nodes = len(nodes)
    num_arcs = len(arcs)
    num_commodities = len(commodities)
    num_fleet_types = len(fleet_types)

    node_index = {node_id: i for i, node_id in enumerate(nodes)}

    arc_from = [node_index[arc["from"]] for arc in arcs]
    arc_to = [node_index[arc["to"]] for arc in arcs]
    arc_flow_cost = [float(arc["flow_cost"]) for arc in arcs]
    arcs_uv = list(zip(arc_from, arc_to))

    components = strongly_connected_components(num_nodes, arcs_uv)

    # A nonnegative conserved vehicle circulation can only use arcs whose
    # endpoints belong to the same strongly connected component.
    usable_arc = [
        components[arc_from[a]] == components[arc_to[a]]
        for a in range(num_arcs)
    ]

    outgoing_arcs = [[] for _ in range(num_nodes)]
    incoming_arcs = [[] for _ in range(num_nodes)]
    usable_adjacency = ArcAdjacency([[] for _ in range(num_nodes)])
    usable_adjacency.arc_from = arc_from

    for a in range(num_arcs):
        u = arc_from[a]
        v = arc_to[a]
        outgoing_arcs[u].append(a)
        incoming_arcs[v].append(a)
        if usable_arc[a]:
            usable_adjacency[u].append((a, v))

    capacities = [int(ft["capacity"]) for ft in fleet_types]
    fixed_costs = [float(ft["fixed_cost"]) for ft in fleet_types]

    if any(capacity <= 0 for capacity in capacities):
        raise ValueError("All fleet capacities must be positive.")

    # Remove vehicle types that are unambiguously dominated one-for-one.
    dominated = [False] * num_fleet_types
    for t in range(num_fleet_types):
        for s in range(num_fleet_types):
            if s == t:
                continue
            if capacities[s] >= capacities[t] and fixed_costs[s] <= fixed_costs[t]:
                if capacities[s] > capacities[t] or fixed_costs[s] < fixed_costs[t] or s < t:
                    dominated[t] = True
                    break

    # Cache shortest return paths used to turn loaded arc traversals into
    # feasible vehicle cycles.
    return_path_cache = {}

    def get_return_path(arc_idx):
        if arc_idx in return_path_cache:
            return return_path_cache[arc_idx]

        source = arc_to[arc_idx]
        target = arc_from[arc_idx]

        if source == target:
            return_path_cache[arc_idx] = []
            return []

        predecessor_node = [-1] * num_nodes
        predecessor_arc = [-1] * num_nodes
        predecessor_node[source] = source
        queue = deque([source])

        while queue and predecessor_node[target] == -1:
            u = queue.popleft()
            for a, v in usable_adjacency[u]:
                if predecessor_node[v] == -1:
                    predecessor_node[v] = u
                    predecessor_arc[v] = a
                    queue.append(v)

        if predecessor_node[target] == -1:
            return_path_cache[arc_idx] = None
            return None

        path = []
        current = target
        while current != source:
            a = predecessor_arc[current]
            path.append(a)
            current = predecessor_node[current]
        path.reverse()

        return_path_cache[arc_idx] = path
        return path

    def route_all_commodities(weight_builder):
        routes = []
        for commodity in commodities:
            origin = node_index[commodity["origin"]]
            destination = node_index[commodity["destination"]]

            if origin == destination:
                routes.append([])
                continue

            weights = weight_builder(commodity)
            path = shortest_arc_path(
                origin, destination, usable_adjacency, weights
            )
            if path is None:
                return None
            routes.append(path)
        return routes

    def build_candidate(routes, fleet_type_idx):
        x_values = [
            [0.0] * num_arcs for _ in range(num_commodities)
        ]
        arc_load = [0] * num_arcs

        flow_objective = 0.0
        for k, path in enumerate(routes):
            demand = int(commodities[k]["demand"])
            for a in path:
                x_values[k][a] += demand
                arc_load[a] += demand
                flow_objective += demand * arc_flow_cost[a]

        y_values = [
            [0] * num_arcs for _ in range(num_fleet_types)
        ]

        capacity = capacities[fleet_type_idx]
        for a, load in enumerate(arc_load):
            if load <= 0:
                continue

            vehicle_count = (load + capacity - 1) // capacity
            return_path = get_return_path(a)
            if return_path is None:
                return None

            # The loaded arc plus a return path is a directed cycle, so adding
            # this vector preserves vehicle balance at every node.
            y_values[fleet_type_idx][a] += vehicle_count
            for return_arc in return_path:
                y_values[fleet_type_idx][return_arc] += vehicle_count

        vehicle_objective = sum(
            fixed_costs[t] * sum(y_values[t])
            for t in range(num_fleet_types)
        )
        objective = flow_objective + vehicle_objective
        return objective, x_values, y_values

    best_heuristic = None

    # First route commodities by transportation cost.
    base_routes = route_all_commodities(
        lambda commodity: [
            arc_flow_cost[a] if usable_arc[a] else math.inf
            for a in range(num_arcs)
        ]
    )

    candidate_types = [
        t for t in range(num_fleet_types) if not dominated[t]
    ]
    candidate_types.sort(
        key=lambda t: (
            fixed_costs[t] / capacities[t],
            fixed_costs[t],
            -capacities[t],
        )
    )

    if base_routes is not None:
        for t in candidate_types:
            candidate = build_candidate(base_routes, t)
            if candidate is not None and (
                best_heuristic is None
                or candidate[0] < best_heuristic[0] - 1e-9
            ):
                best_heuristic = candidate

    # Try additional fleet-aware routes while reserving most of the budget
    # for the exact MIP search.
    heuristic_cutoff = start_time + min(
        5.0, max(0.1, 0.10 * max(1, args.time_limit))
    )

    for t in candidate_types:
        if time.monotonic() >= heuristic_cutoff:
            break

        capacity = capacities[t]
        fixed_cost = fixed_costs[t]

        def fleet_aware_weights(commodity, cap=capacity, fc=fixed_cost):
            demand = int(commodity["demand"])
            vehicles_needed = (demand + cap - 1) // cap
            return [
                (
                    demand * arc_flow_cost[a] + vehicles_needed * fc
                    if usable_arc[a]
                    else math.inf
                )
                for a in range(num_arcs)
            ]

        routes = route_all_commodities(fleet_aware_weights)
        if routes is None:
            continue

        candidate = build_candidate(routes, t)
        if candidate is not None and (
            best_heuristic is None
            or candidate[0] < best_heuristic[0] - 1e-9
        ):
            best_heuristic = candidate

    best_logged_objective = math.inf

    if best_heuristic is not None:
        heuristic_solution = make_solution(
            instance,
            best_heuristic[1],
            best_heuristic[2],
            best_heuristic[0],
        )
        best_logged_objective = best_heuristic[0]
        if logger:
            logger.log_solution(best_heuristic[0], heuristic_solution)

    model = gp.Model("fleet_conserving_multicommodity_network_design")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    remaining_time = max(0.01, deadline - time.monotonic())
    model.Params.TimeLimit = remaining_time

    x_vars = []
    for k, commodity in enumerate(commodities):
        demand = float(commodity["demand"])
        row = []
        for a in range(num_arcs):
            upper_bound = demand if usable_arc[a] else 0.0
            row.append(
                model.addVar(
                    lb=0.0,
                    ub=upper_bound,
                    vtype=GRB.CONTINUOUS,
                    obj=arc_flow_cost[a],
                    name=f"x_{k}_{a}",
                )
            )
        x_vars.append(row)

    y_vars = []
    for t in range(num_fleet_types):
        row = []
        for a in range(num_arcs):
            upper_bound = GRB.INFINITY
            if not usable_arc[a] or dominated[t]:
                upper_bound = 0.0

            row.append(
                model.addVar(
                    lb=0.0,
                    ub=upper_bound,
                    vtype=GRB.INTEGER,
                    obj=fixed_costs[t],
                    name=f"y_{t}_{a}",
                )
            )
        y_vars.append(row)

    model.ModelSense = GRB.MINIMIZE

    # Commodity flow conservation.
    for k, commodity in enumerate(commodities):
        origin = node_index[commodity["origin"]]
        destination = node_index[commodity["destination"]]
        demand = float(commodity["demand"])

        for n in range(num_nodes):
            rhs = 0.0
            if n == origin:
                rhs += demand
            if n == destination:
                rhs -= demand

            model.addConstr(
                gp.quicksum(x_vars[k][a] for a in outgoing_arcs[n])
                - gp.quicksum(x_vars[k][a] for a in incoming_arcs[n])
                == rhs,
                name=f"flow_balance_{k}_{n}",
            )

    # Capacity supplied by all vehicle types on each arc.
    for a in range(num_arcs):
        model.addConstr(
            gp.quicksum(x_vars[k][a] for k in range(num_commodities))
            <= gp.quicksum(
                capacities[t] * y_vars[t][a]
                for t in range(num_fleet_types)
            ),
            name=f"capacity_{a}",
        )

    # Vehicle conservation by type.
    for t in range(num_fleet_types):
        for n in range(num_nodes):
            model.addConstr(
                gp.quicksum(y_vars[t][a] for a in outgoing_arcs[n])
                - gp.quicksum(y_vars[t][a] for a in incoming_arcs[n])
                == 0,
                name=f"vehicle_balance_{t}_{n}",
            )

    # Warm start from the explicitly constructed feasible circulation.
    if best_heuristic is not None:
        _, start_x, start_y = best_heuristic
        for k in range(num_commodities):
            for a in range(num_arcs):
                x_vars[k][a].Start = start_x[k][a]
        for t in range(num_fleet_types):
            for a in range(num_arcs):
                y_vars[t][a].Start = start_y[t][a]

    flat_x_vars = [
        x_vars[k][a]
        for k in range(num_commodities)
        for a in range(num_arcs)
    ]
    flat_y_vars = [
        y_vars[t][a]
        for t in range(num_fleet_types)
        for a in range(num_arcs)
    ]

    callback_state = {"best_logged": best_logged_objective}

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        objective = float(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ))
        tolerance = 1e-7 * max(1.0, abs(objective))
        if objective >= callback_state["best_logged"] - tolerance:
            return

        callback_state["best_logged"] = objective
        if not logger:
            return

        flat_x_values = cb_model.cbGetSolution(flat_x_vars)
        flat_y_values = cb_model.cbGetSolution(flat_y_vars)

        x_values = []
        offset = 0
        for _ in range(num_commodities):
            x_values.append(
                flat_x_values[offset:offset + num_arcs]
            )
            offset += num_arcs

        y_values = []
        offset = 0
        for _ in range(num_fleet_types):
            y_values.append(
                flat_y_values[offset:offset + num_arcs]
            )
            offset += num_arcs

        solution = make_solution(
            instance, x_values, y_values, objective
        )
        logger.log_solution(objective, solution)

    model.optimize(incumbent_callback)

    if model.SolCount > 0:
        final_x = [
            [x_vars[k][a].X for a in range(num_arcs)]
            for k in range(num_commodities)
        ]
        final_y = [
            [y_vars[t][a].X for a in range(num_arcs)]
            for t in range(num_fleet_types)
        ]
        final_objective = float(model.ObjVal)
    elif best_heuristic is not None:
        final_objective, final_x, final_y = best_heuristic
    else:
        raise RuntimeError(
            "No feasible solution was found within the time limit."
        )

    final_solution = make_solution(
        instance, final_x, final_y, final_objective
    )

    tolerance = 1e-7 * max(1.0, abs(final_objective))
    if logger and final_objective < callback_state["best_logged"] - tolerance:
        logger.log_solution(final_objective, final_solution)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()