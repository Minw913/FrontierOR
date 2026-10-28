import argparse
import heapq
import json
import os
import time
from collections import defaultdict, deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    arc_data = instance["network"]["arcs"]
    commodity_data = instance["commodities"]["commodity_list"]

    arcs = []
    for a in arc_data:
        arcs.append({
            "id": int(a["arc_id"]),
            "u": int(a["from_node"]),
            "v": int(a["to_node"]),
            "capacity": int(a["capacity"]),
        })

    commodities = []
    for c in commodity_data:
        commodities.append({
            "id": int(c["commodity_id"]),
            "s": int(c["origin"]),
            "t": int(c["destination"]),
            "demand": int(c["demand"]),
            "reject_cost": float(c["artificial_arc_cost"]),
        })

    num_arcs = len(arcs)
    num_commodities = len(commodities)

    def make_solution(routes):
        entries = []
        objective = 0.0

        for k, c in enumerate(commodities):
            if c["s"] == c["t"]:
                rejected = False
                path_indices = []
            elif k in routes and routes[k] is not None:
                rejected = False
                path_indices = routes[k]
            else:
                rejected = True
                path_indices = []
                objective += c["reject_cost"]

            path_arcs = [
                {
                    "from": arcs[ai]["u"],
                    "to": arcs[ai]["v"],
                    "arc_id": arcs[ai]["id"],
                }
                for ai in path_indices
            ]

            entries.append({
                "commodity_id": c["id"],
                "rejected": rejected,
                "path_arcs": path_arcs,
            })

        return {
            "objective_value": float(objective),
            "commodities": entries,
        }

    # The all-rejected solution is always feasible. Commodities with identical
    # origin and destination use an empty, zero-cost physical path.
    best_routes = {}
    best_solution = make_solution(best_routes)
    best_objective = best_solution["objective_value"]

    if logger:
        logger.log_solution(best_objective, best_solution)

    if args.time_limit <= 0:
        write_solution(args.solution_path, best_solution)
        return

    # For each commodity, retain only arcs that can lie on an s-t route and
    # whose capacities can individually support its unsplittable demand.
    commodity_edges = [[] for _ in range(num_commodities)]
    commodity_adjacency = [None] * num_commodities

    preprocessing_completed = True

    for k, c in enumerate(commodities):
        if time.monotonic() - start_time >= args.time_limit:
            preprocessing_completed = False
            break

        s, t, demand = c["s"], c["t"], c["demand"]
        if s == t:
            commodity_adjacency[k] = {}
            continue

        eligible = [ai for ai, a in enumerate(arcs) if a["capacity"] >= demand]
        forward_adj = defaultdict(list)
        reverse_adj = defaultdict(list)

        for ai in eligible:
            a = arcs[ai]
            forward_adj[a["u"]].append(a["v"])
            reverse_adj[a["v"]].append(a["u"])

        forward_reachable = {s}
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in forward_adj.get(u, ()):
                if v not in forward_reachable:
                    forward_reachable.add(v)
                    queue.append(v)

        if t not in forward_reachable:
            commodity_adjacency[k] = {}
            continue

        backward_reachable = {t}
        queue = deque([t])
        while queue:
            v = queue.popleft()
            for u in reverse_adj.get(v, ()):
                if u not in backward_reachable:
                    backward_reachable.add(u)
                    queue.append(u)

        kept = []
        adjacency = defaultdict(list)
        for ai in eligible:
            a = arcs[ai]
            if a["u"] in forward_reachable and a["v"] in backward_reachable:
                kept.append(ai)
                adjacency[a["u"]].append(ai)

        commodity_edges[k] = kept
        commodity_adjacency[k] = dict(adjacency)

    if not preprocessing_completed:
        write_solution(args.solution_path, best_solution)
        return

    def shortest_residual_path(k, residual):
        c = commodities[k]
        s, t, demand = c["s"], c["t"], c["demand"]
        adjacency = commodity_adjacency[k]

        if s == t:
            return []
        if not adjacency:
            return None

        distances = {s: 0.0}
        parent = {}
        heap = [(0.0, s)]

        while heap:
            dist_u, u = heapq.heappop(heap)
            if dist_u != distances.get(u):
                continue
            if u == t:
                break

            for ai in adjacency.get(u, ()):
                if residual[ai] < demand:
                    continue

                a = arcs[ai]
                v = a["v"]

                # Prefer short paths while mildly avoiding scarce residual arcs.
                weight = 1.0 + demand / max(float(residual[ai]), 1.0)
                new_dist = dist_u + weight

                if new_dist + 1e-12 < distances.get(v, float("inf")):
                    distances[v] = new_dist
                    parent[v] = (u, ai)
                    heapq.heappush(heap, (new_dist, v))

        if t not in distances:
            return None

        path = []
        node = t
        while node != s:
            previous, ai = parent[node]
            path.append(ai)
            node = previous
        path.reverse()
        return path

    def run_greedy(order):
        residual = [a["capacity"] for a in arcs]
        routes = {}

        for k in order:
            c = commodities[k]
            if c["s"] == c["t"]:
                routes[k] = []
                continue
            if c["reject_cost"] <= 0 or not commodity_edges[k]:
                continue

            path = shortest_residual_path(k, residual)
            if path is None:
                continue

            routes[k] = path
            for ai in path:
                residual[ai] -= c["demand"]

        return routes

    routable = [
        k for k, c in enumerate(commodities)
        if c["s"] == c["t"] or commodity_edges[k]
    ]

    greedy_orders = [
        sorted(
            routable,
            key=lambda k: (
                -commodities[k]["reject_cost"] / max(commodities[k]["demand"], 1),
                -commodities[k]["reject_cost"],
                commodities[k]["demand"],
            ),
        ),
        sorted(
            routable,
            key=lambda k: (
                -commodities[k]["reject_cost"],
                commodities[k]["demand"],
            ),
        ),
        sorted(
            routable,
            key=lambda k: (
                commodities[k]["demand"],
                -commodities[k]["reject_cost"],
            ),
        ),
    ]

    # Avoid spending excessive preprocessing time on repeated greedy passes.
    estimated_size = sum(len(x) for x in commodity_edges)
    max_greedy_passes = 3 if estimated_size <= 2_000_000 else 1

    for order in greedy_orders[:max_greedy_passes]:
        if time.monotonic() - start_time >= args.time_limit:
            break

        routes = run_greedy(order)
        solution = make_solution(routes)
        objective = solution["objective_value"]

        if objective + 1e-9 < best_objective:
            best_objective = objective
            best_routes = routes
            best_solution = solution
            if logger:
                logger.log_solution(best_objective, best_solution)

    remaining = args.time_limit - (time.monotonic() - start_time)
    if remaining <= 0.01:
        write_solution(args.solution_path, best_solution)
        return

    try:
        model = gp.Model("unsplittable_multicommodity_flow")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        rejection_vars = {}
        flow_vars = {}
        flat_flow_vars = []
        capacity_terms = defaultdict(list)

        # Create rejection and physical-arc variables.
        for k, c in enumerate(commodities):
            if c["s"] == c["t"]:
                continue

            y = model.addVar(
                vtype=GRB.BINARY,
                obj=c["reject_cost"],
                name=f"reject_{k}",
            )
            rejection_vars[k] = y
            y.Start = 0.0 if k in best_routes else 1.0

            best_path_set = set(best_routes.get(k, []))

            for ai in commodity_edges[k]:
                x = model.addVar(
                    vtype=GRB.BINARY,
                    obj=0.0,
                    name=f"x_{k}_{ai}",
                )
                flow_vars[k, ai] = x
                flat_flow_vars.append((k, ai, x))
                capacity_terms[ai].append((c["demand"], x))
                x.Start = 1.0 if ai in best_path_set else 0.0

        model.update()

        # Per-commodity flow conservation.
        for k, c in enumerate(commodities):
            if c["s"] == c["t"]:
                continue

            outgoing = defaultdict(list)
            incoming = defaultdict(list)

            for ai in commodity_edges[k]:
                a = arcs[ai]
                x = flow_vars[k, ai]
                outgoing[a["u"]].append(x)
                incoming[a["v"]].append(x)

            relevant_nodes = set(outgoing) | set(incoming) | {c["s"], c["t"]}
            y = rejection_vars[k]

            for node in relevant_nodes:
                expr = gp.LinExpr()
                if outgoing.get(node):
                    expr += gp.quicksum(outgoing[node])
                if incoming.get(node):
                    expr -= gp.quicksum(incoming[node])

                if node == c["s"]:
                    expr += y
                    model.addConstr(expr == 1.0, name=f"flow_s_{k}")
                elif node == c["t"]:
                    expr -= y
                    model.addConstr(expr == -1.0, name=f"flow_t_{k}")
                else:
                    model.addConstr(expr == 0.0, name=f"flow_{k}_{node}")

        # Shared physical capacities.
        for ai, terms in capacity_terms.items():
            model.addConstr(
                gp.quicksum(demand * var for demand, var in terms)
                <= arcs[ai]["capacity"],
                name=f"capacity_{ai}",
            )

        model.ModelSense = GRB.MINIMIZE
        model.update()

        remaining = args.time_limit - (time.monotonic() - start_time)
        if remaining <= 0.01:
            write_solution(args.solution_path, best_solution)
            return

        model.Params.TimeLimit = max(0.01, remaining)

        flow_var_list = [item[2] for item in flat_flow_vars]
        rejection_keys = list(rejection_vars.keys())
        rejection_var_list = [rejection_vars[k] for k in rejection_keys]

        def extract_path(k, selected_arc_indices):
            c = commodities[k]
            s, t = c["s"], c["t"]

            if s == t:
                return []

            adjacency = defaultdict(list)
            for ai in selected_arc_indices:
                adjacency[arcs[ai]["u"]].append(ai)

            parent = {}
            visited = {s}
            queue = deque([s])

            while queue and t not in visited:
                u = queue.popleft()
                for ai in adjacency.get(u, ()):
                    v = arcs[ai]["v"]
                    if v not in visited:
                        visited.add(v)
                        parent[v] = (u, ai)
                        queue.append(v)

            if t not in visited:
                return None

            path = []
            node = t
            while node != s:
                previous, ai = parent[node]
                path.append(ai)
                node = previous
            path.reverse()
            return path

        def solution_from_values(flow_values, rejection_values):
            selected = defaultdict(list)
            for (k, ai, _), value in zip(flat_flow_vars, flow_values):
                if value > 0.5:
                    selected[k].append(ai)

            rejection_by_k = {
                k: value for k, value in zip(rejection_keys, rejection_values)
            }

            routes = {}
            for k, c in enumerate(commodities):
                if c["s"] == c["t"]:
                    routes[k] = []
                    continue

                if rejection_by_k.get(k, 1.0) > 0.5:
                    continue

                path = extract_path(k, selected.get(k, ()))
                if path is not None:
                    routes[k] = path

            return routes, make_solution(routes)

        callback_state = {
            "best_objective": best_objective,
            "best_routes": best_routes,
            "best_solution": best_solution,
        }

        def incumbent_callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                flow_values = (
                    cb_model.cbGetSolution(flow_var_list)
                    if flow_var_list else []
                )
                rejection_values = (
                    cb_model.cbGetSolution(rejection_var_list)
                    if rejection_var_list else []
                )

                routes, solution = solution_from_values(
                    flow_values, rejection_values
                )
                objective = solution["objective_value"]

                if objective + 1e-9 < callback_state["best_objective"]:
                    callback_state["best_objective"] = objective
                    callback_state["best_routes"] = routes
                    callback_state["best_solution"] = solution
                    if logger:
                        logger.log_solution(objective, solution)
            except Exception:
                # A logging or extraction failure must not terminate the MIP.
                return

        model.optimize(incumbent_callback)

        best_objective = callback_state["best_objective"]
        best_routes = callback_state["best_routes"]
        best_solution = callback_state["best_solution"]

        if model.SolCount > 0:
            flow_values = [var.X for var in flow_var_list]
            rejection_values = [var.X for var in rejection_var_list]
            routes, final_mip_solution = solution_from_values(
                flow_values, rejection_values
            )
            final_mip_objective = final_mip_solution["objective_value"]

            if final_mip_objective + 1e-9 < best_objective:
                best_objective = final_mip_objective
                best_routes = routes
                best_solution = final_mip_solution
                if logger:
                    logger.log_solution(best_objective, best_solution)

    except gp.GurobiError:
        # Preserve and return the best feasible heuristic solution.
        pass
    except Exception:
        # Preserve feasibility even if model construction or extraction fails.
        pass

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()