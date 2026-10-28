import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def empty_solution():
    return {
        "objective_value": 0.0,
        "served_arcs": [],
        "tour_arcs": []
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["num_nodes"])
    depot = int(data["depot"])

    raw_arcs = data["arcs"]
    arcs = [
        (int(a[0]), int(a[1]), float(a[2]), a[2])
        for a in raw_arcs
    ]

    raw_profitable = data["profitable_arcs"]
    profitable = [
        (int(a[0]), int(a[1]), float(a[2]), a[2])
        for a in raw_profitable
    ]

    vi_nodes = [int(v) for v in data.get("VI_nodes", [])]
    strong_edges = [
        (int(e[0]), int(e[1]))
        for e in data.get("strong_incompatibilities", [])
    ]
    weak_edges = [
        (int(e[0]), int(e[1]), float(e[2]))
        for e in data.get("weak_incompatibilities", [])
    ]

    initial_solution = empty_solution()
    best_solution = initial_solution
    best_objective = 0.0

    if logger:
        logger.log_solution(0.0, initial_solution)

    # Map each profitable arc to a road arc with the same ordered endpoints.
    # Under the intended schema, profitable arcs are members of the road-arc set.
    arc_indices_by_pair = {}
    for a, (u, v, _, _) in enumerate(arcs):
        arc_indices_by_pair.setdefault((u, v), []).append(a)

    profitable_to_arc = []
    for u, v, _, _ in profitable:
        candidates = arc_indices_by_pair.get((u, v), [])
        if not candidates:
            raise ValueError(
                f"Profitable arc ({u}, {v}) does not occur in the road network."
            )
        # If parallel arcs occur without explicit identifiers, use the
        # least-cost matching road arc as the profitable one.
        chosen = min(candidates, key=lambda idx: arcs[idx][2])
        profitable_to_arc.append(chosen)

    incoming = [[] for _ in range(n)]
    outgoing = [[] for _ in range(n)]
    for a, (u, v, _, _) in enumerate(arcs):
        outgoing[u].append(a)
        incoming[v].append(a)

    profitable_by_tail = {v: [] for v in vi_nodes}
    for p, (u, _, _, _) in enumerate(profitable):
        if u in profitable_by_tail:
            profitable_by_tail[u].append(p)

    def make_solution(x_values, y_values):
        """Convert an integral candidate into the required output format."""
        counts = [max(0, int(round(value))) for value in x_values]
        selected = [float(value) > 0.5 for value in y_values]

        # Determine the undirected support component containing the depot.
        adjacency = [[] for _ in range(n)]
        for a, count in enumerate(counts):
            if count <= 0:
                continue
            u, v, _, _ = arcs[a]
            adjacency[u].append(v)
            adjacency[v].append(u)

        reachable = [False] * n
        reachable[depot] = True
        stack = [depot]
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if not reachable[v]:
                    reachable[v] = True
                    stack.append(v)

        # A selected profitable arc must belong to the depot component.
        for p, is_selected in enumerate(selected):
            if not is_selected:
                continue
            u, v, _, _ = profitable[p]
            if not reachable[u] or not reachable[v]:
                return None

        # Disconnected components with no service can be removed. They are
        # balanced components and therefore are not needed by the tour.
        cleaned_counts = counts[:]
        for a, count in enumerate(cleaned_counts):
            if count <= 0:
                continue
            u, v, _, _ = arcs[a]
            if not reachable[u] or not reachable[v]:
                cleaned_counts[a] = 0

        served_arcs = []
        total_profit = 0.0
        for p, is_selected in enumerate(selected):
            if not is_selected:
                continue
            u, v, profit, original_profit = profitable[p]
            road_index = profitable_to_arc[p]
            road_cost = arcs[road_index][3]
            served_arcs.append({
                "from": u,
                "to": v,
                "profit": original_profit,
                "cost": road_cost
            })
            total_profit += profit

        tour_arcs = []
        total_travel_cost = 0.0
        for a, count in enumerate(cleaned_counts):
            if count <= 0:
                continue
            u, v, cost, original_cost = arcs[a]
            tour_arcs.append({
                "from": u,
                "to": v,
                "count": count,
                "cost": original_cost
            })
            total_travel_cost += count * cost

        active = set()
        for p, is_selected in enumerate(selected):
            if is_selected:
                active.add(profitable[p][0])

        total_penalty = 0.0
        for u, v, penalty in weak_edges:
            if u in active and v in active:
                total_penalty += penalty

        objective = total_profit - total_travel_cost - total_penalty
        if abs(objective) < 1e-9:
            objective = 0.0

        return {
            "objective_value": float(objective),
            "served_arcs": served_arcs,
            "tour_arcs": tour_arcs
        }

    remaining = args.time_limit - (time.monotonic() - start_time)
    if remaining > 0:
        try:
            model = gp.Model("profitable_arc_tour")

            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.LazyConstraints = 1
            model.Params.TimeLimit = max(0.001, remaining)

            x = [
                model.addVar(vtype=GRB.INTEGER, lb=0.0, name=f"x_{a}")
                for a in range(len(arcs))
            ]
            y = [
                model.addVar(vtype=GRB.BINARY, name=f"y_{p}")
                for p in range(len(profitable))
            ]
            z = {
                v: model.addVar(vtype=GRB.BINARY, name=f"active_{v}")
                for v in vi_nodes
            }
            w = [
                model.addVar(vtype=GRB.BINARY, name=f"override_{e}")
                for e in range(len(weak_edges))
            ]

            model.update()

            # Directed flow balance ensures that the selected support is Eulerian.
            for v in range(n):
                model.addConstr(
                    gp.quicksum(x[a] for a in outgoing[v])
                    == gp.quicksum(x[a] for a in incoming[v]),
                    name=f"balance_{v}"
                )

            # A profitable arc can be served only when it is traversed.
            for p, a in enumerate(profitable_to_arc):
                model.addConstr(y[p] <= x[a], name=f"serve_requires_traverse_{p}")

            # Exact activation indicators for incompatibility-graph nodes.
            for v in vi_nodes:
                plist = profitable_by_tail.get(v, [])
                if plist:
                    for p in plist:
                        model.addConstr(y[p] <= z[v])
                    model.addConstr(
                        z[v] <= gp.quicksum(y[p] for p in plist),
                        name=f"active_if_served_{v}"
                    )
                else:
                    model.addConstr(z[v] == 0, name=f"inactive_{v}")

            # Strong incompatibilities.
            for e, (u, v) in enumerate(strong_edges):
                if u not in z or v not in z:
                    continue
                model.addConstr(z[u] + z[v] <= 1, name=f"strong_{e}")

            # Weak incompatibilities and their paid overrides.
            for e, (u, v, _) in enumerate(weak_edges):
                if u not in z or v not in z:
                    model.addConstr(w[e] == 0, name=f"unused_override_{e}")
                    continue
                model.addConstr(
                    z[u] + z[v] <= 1 + w[e],
                    name=f"weak_{e}"
                )
                model.addConstr(w[e] <= z[u])
                model.addConstr(w[e] <= z[v])

            objective = (
                gp.quicksum(profitable[p][2] * y[p]
                            for p in range(len(profitable)))
                - gp.quicksum(arcs[a][2] * x[a]
                              for a in range(len(arcs)))
                - gp.quicksum(weak_edges[e][2] * w[e]
                              for e in range(len(weak_edges)))
            )
            model.setObjective(objective, GRB.MAXIMIZE)

            # The empty tour is always a feasible initial solution.
            for var in x:
                var.Start = 0.0
            for var in y:
                var.Start = 0.0
            for var in z.values():
                var.Start = 0.0
            for var in w:
                var.Start = 0.0

            model._x_vars = x
            model._y_vars = y
            model._best_objective = best_objective
            model._best_solution = best_solution
            model._logger = logger

            def callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                x_values = cb_model.cbGetSolution(cb_model._x_vars)
                y_values = cb_model.cbGetSolution(cb_model._y_vars)

                # Build undirected connected components of the incumbent support.
                adjacency = [[] for _ in range(n)]
                for a, value in enumerate(x_values):
                    if value > 0.5:
                        u, v, _, _ = arcs[a]
                        adjacency[u].append(v)
                        adjacency[v].append(u)

                component_id = [-1] * n
                components = []
                for root in range(n):
                    if component_id[root] != -1:
                        continue
                    cid = len(components)
                    component = []
                    stack = [root]
                    component_id[root] = cid
                    while stack:
                        u = stack.pop()
                        component.append(u)
                        for v in adjacency[u]:
                            if component_id[v] == -1:
                                component_id[v] = cid
                                stack.append(v)
                    components.append(component)

                depot_component = component_id[depot]
                cuts_added = False

                for cid, component in enumerate(components):
                    if cid == depot_component:
                        continue

                    component_set = set(component)
                    representative_p = None
                    for p, value in enumerate(y_values):
                        if value <= 0.5:
                            continue
                        u, v, _, _ = profitable[p]
                        if u in component_set and v in component_set:
                            representative_p = p
                            break

                    if representative_p is None:
                        continue

                    leaving = [
                        a
                        for u in component
                        for a in outgoing[u]
                        if arcs[a][1] not in component_set
                    ]

                    cb_model.cbLazy(
                        gp.quicksum(cb_model._x_vars[a] for a in leaving)
                        >= cb_model._y_vars[representative_p]
                    )
                    cuts_added = True

                # This candidate is not yet feasible for the lazy connectivity
                # constraints, so it must not be logged as an incumbent.
                if cuts_added:
                    return

                solution = make_solution(x_values, y_values)
                if solution is None:
                    return

                value = solution["objective_value"]
                if value > cb_model._best_objective + 1e-7:
                    cb_model._best_objective = value
                    cb_model._best_solution = solution
                    if cb_model._logger:
                        cb_model._logger.log_solution(value, solution)

            model.optimize(callback)

            best_objective = model._best_objective
            best_solution = model._best_solution

            if model.SolCount > 0:
                x_values = [var.X for var in x]
                y_values = [var.X for var in y]
                final_candidate = make_solution(x_values, y_values)
                if (
                    final_candidate is not None
                    and final_candidate["objective_value"] > best_objective + 1e-7
                ):
                    best_objective = final_candidate["objective_value"]
                    best_solution = final_candidate
                    if logger:
                        logger.log_solution(best_objective, best_solution)

        except gp.GurobiError:
            # The empty tour remains a valid fallback.
            pass

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()