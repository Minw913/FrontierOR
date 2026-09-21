import argparse
import heapq
import json
import math
import os
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    params = instance["problem_parameters"]
    nodes = instance["nodes"]
    n = len(nodes)
    capacity = float(params["vehicle_capacity"])
    horizon = float(params["planning_horizon"])

    source_candidates = [i for i, node in enumerate(nodes) if node["type"] == "depot_source"]
    sink_candidates = [i for i, node in enumerate(nodes) if node["type"] == "depot_sink"]
    customers = [i for i, node in enumerate(nodes) if node["type"] == "customer"]

    if len(source_candidates) != 1 or len(sink_candidates) != 1:
        raise ValueError("The instance must contain exactly one depot_source and one depot_sink.")

    source = source_candidates[0]
    sink = sink_candidates[0]

    node_ids = [int(node["id"]) for node in nodes]
    xcoord = [float(node["x"]) for node in nodes]
    ycoord = [float(node["y"]) for node in nodes]
    demand = [float(node["demand"]) for node in nodes]
    service = [float(node["service_time"]) for node in nodes]
    tw_open = [float(node["time_window_open"]) for node in nodes]
    tw_close = [float(node["time_window_close"]) for node in nodes]

    # Euclidean travel costs and travel times.
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = xcoord[i], ycoord[i]
        for j in range(i + 1, n):
            d = math.hypot(xi - xcoord[j], yi - ycoord[j])
            dist[i][j] = d
            dist[j][i] = d

    def evaluate_route(route, with_schedule=False):
        """Return (feasible, cost, schedule, cumulative_loads)."""
        if not route:
            return False, math.inf, None, None

        total_load = sum(demand[i] for i in route)
        if total_load > capacity + EPS:
            return False, math.inf, None, None

        schedule = {}
        loads = {}
        current_start = tw_open[source]
        previous = source
        cumulative = 0.0
        cost = 0.0

        for customer in route:
            # The direct arc must exist under the definition in the problem.
            if demand[previous] + demand[customer] > capacity + EPS:
                return False, math.inf, None, None
            if (tw_open[previous] + service[previous] + dist[previous][customer]
                    > tw_close[customer] + EPS):
                return False, math.inf, None, None

            arrival = current_start + service[previous] + dist[previous][customer]
            current_start = max(tw_open[customer], arrival)
            if current_start > tw_close[customer] + EPS:
                return False, math.inf, None, None

            cumulative += demand[customer]
            schedule[customer] = current_start
            loads[customer] = cumulative
            cost += dist[previous][customer]
            previous = customer

        if demand[previous] + demand[sink] > capacity + EPS:
            return False, math.inf, None, None
        if tw_open[previous] + service[previous] + dist[previous][sink] > horizon + EPS:
            return False, math.inf, None, None

        return_time = current_start + service[previous] + dist[previous][sink]
        if return_time > horizon + EPS:
            return False, math.inf, None, None

        cost += dist[previous][sink]
        return True, cost, schedule if with_schedule else None, loads if with_schedule else None

    def route_set_cost(routes):
        total = 0.0
        for route in routes:
            feasible, cost, _, _ = evaluate_route(route)
            if not feasible:
                return math.inf
            total += cost
        return total

    def make_solution(routes):
        clean_routes = [route for route in routes if route]
        obj = route_set_cost(clean_routes)
        output_routes = [
            [node_ids[source]] + [node_ids[i] for i in route] + [node_ids[sink]]
            for route in clean_routes
        ]
        return {
            "objective_value": float(obj),
            "routes": output_routes,
        }

    best_routes = None
    best_objective = math.inf

    def register(routes):
        nonlocal best_routes, best_objective
        candidate_routes = [list(r) for r in routes if r]
        candidate_obj = route_set_cost(candidate_routes)
        if not math.isfinite(candidate_obj):
            return False

        flattened = [c for route in candidate_routes for c in route]
        if len(flattened) != len(customers) or set(flattened) != set(customers):
            return False
        if len(flattened) != len(set(flattened)):
            return False

        if candidate_obj + 1e-7 < best_objective:
            best_objective = candidate_obj
            best_routes = candidate_routes
            solution = make_solution(candidate_routes)
            if logger:
                logger.log_solution(candidate_obj, solution)
            return True
        return False

    # Always establish a simple feasible incumbent first.
    singleton_routes = []
    for c in customers:
        feasible, _, _, _ = evaluate_route([c])
        if not feasible:
            raise RuntimeError(
                f"Customer node {node_ids[c]} cannot be served even by a dedicated vehicle."
            )
        singleton_routes.append([c])

    register(singleton_routes)

    # Clarke-Wright-style directed savings construction.
    active = {k: [c] for k, c in enumerate(customers)}
    owner = {c: k for k, c in enumerate(customers)}
    savings_heap = []

    heuristic_budget = min(8.0, max(0.0, 0.18 * max(0, args.time_limit)))
    heuristic_deadline = min(deadline, start_time + heuristic_budget)

    if time.monotonic() < heuristic_deadline:
        stop_building = False
        for i in customers:
            if stop_building:
                break
            for j in customers:
                if i == j:
                    continue
                if demand[i] + demand[j] > capacity + EPS:
                    continue
                if tw_open[i] + service[i] + dist[i][j] > tw_close[j] + EPS:
                    continue

                saving = dist[i][sink] + dist[source][j] - dist[i][j]
                if saving > EPS:
                    heapq.heappush(savings_heap, (-saving, i, j))

            if time.monotonic() >= heuristic_deadline:
                stop_building = True

        while savings_heap and time.monotonic() < heuristic_deadline:
            _, i, j = heapq.heappop(savings_heap)
            ri = owner[i]
            rj = owner[j]
            if ri == rj or ri not in active or rj not in active:
                continue

            route_i = active[ri]
            route_j = active[rj]
            if route_i[-1] != i or route_j[0] != j:
                continue

            merged = route_i + route_j
            feasible, _, _, _ = evaluate_route(merged)
            if not feasible:
                continue

            active[ri] = merged
            del active[rj]
            for c in merged:
                owner[c] = ri

            register(list(active.values()))

    # Limited first-improvement 2-opt.
    improvement = True
    while improvement and time.monotonic() < heuristic_deadline:
        improvement = False
        route_keys = list(active.keys())
        for rid in route_keys:
            if rid not in active:
                continue
            route = active[rid]
            if len(route) < 3:
                continue

            _, old_cost, _, _ = evaluate_route(route)
            accepted = False
            for left in range(len(route) - 1):
                for right in range(left + 1, len(route)):
                    candidate = route[:left] + list(reversed(route[left:right + 1])) + route[right + 1:]
                    feasible, new_cost, _, _ = evaluate_route(candidate)
                    if feasible and new_cost + 1e-7 < old_cost:
                        active[rid] = candidate
                        for c in candidate:
                            owner[c] = rid
                        register(list(active.values()))
                        improvement = True
                        accepted = True
                        break
                if accepted or time.monotonic() >= heuristic_deadline:
                    break
            if improvement or time.monotonic() >= heuristic_deadline:
                break

    # Limited inter-route relocate search, including route elimination.
    improvement = True
    while improvement and time.monotonic() < heuristic_deadline:
        improvement = False
        route_keys = list(active.keys())

        for ra in route_keys:
            if ra not in active:
                continue
            route_a = active[ra]
            _, cost_a, _, _ = evaluate_route(route_a)

            for pos_a, customer in enumerate(route_a):
                reduced_a = route_a[:pos_a] + route_a[pos_a + 1:]
                if reduced_a:
                    feasible_a, new_cost_a, _, _ = evaluate_route(reduced_a)
                    if not feasible_a:
                        continue
                else:
                    new_cost_a = 0.0

                for rb in route_keys:
                    if rb == ra or rb not in active:
                        continue
                    route_b = active[rb]
                    _, cost_b, _, _ = evaluate_route(route_b)

                    for pos_b in range(len(route_b) + 1):
                        new_b = route_b[:pos_b] + [customer] + route_b[pos_b:]
                        feasible_b, new_cost_b, _, _ = evaluate_route(new_b)
                        if not feasible_b:
                            continue

                        if new_cost_a + new_cost_b + 1e-7 < cost_a + cost_b:
                            if reduced_a:
                                active[ra] = reduced_a
                                for c in reduced_a:
                                    owner[c] = ra
                            else:
                                del active[ra]

                            active[rb] = new_b
                            for c in new_b:
                                owner[c] = rb

                            register(list(active.values()))
                            improvement = True
                            break

                    if improvement or time.monotonic() >= heuristic_deadline:
                        break
                if improvement or time.monotonic() >= heuristic_deadline:
                    break
            if improvement or time.monotonic() >= heuristic_deadline:
                break

    # Exact arc-flow MIP, warm-started from the heuristic incumbent.
    remaining = deadline - time.monotonic()
    if gp is not None and remaining > 0.25 and customers:
        try:
            arcs = []

            for j in customers:
                if demand[source] + demand[j] <= capacity + EPS:
                    if tw_open[source] + service[source] + dist[source][j] <= tw_close[j] + EPS:
                        arcs.append((source, j))

            for i in customers:
                for j in customers:
                    if i == j:
                        continue
                    if demand[i] + demand[j] > capacity + EPS:
                        continue
                    if tw_open[i] + service[i] + dist[i][j] <= tw_close[j] + EPS:
                        arcs.append((i, j))

                if demand[i] + demand[sink] <= capacity + EPS:
                    if tw_open[i] + service[i] + dist[i][sink] <= horizon + EPS:
                        arcs.append((i, sink))

            incoming = {i: [] for i in customers}
            outgoing = {i: [] for i in customers}
            source_arcs = []
            sink_arcs = []

            for arc in arcs:
                i, j = arc
                if i == source:
                    source_arcs.append(arc)
                if j == sink:
                    sink_arcs.append(arc)
                if j in incoming:
                    incoming[j].append(arc)
                if i in outgoing:
                    outgoing[i].append(arc)

            if all(incoming[c] and outgoing[c] for c in customers):
                model = gp.Model("CVRPTW")
                model.Params.OutputFlag = 0
                model.Params.Seed = 0
                model.Params.MIPGap = 1e-4
                model.Params.NumericFocus = 0
                model.Params.Threads = 1

                xvars = {
                    arc: model.addVar(vtype=GRB.BINARY, name=f"x_{arc[0]}_{arc[1]}")
                    for arc in arcs
                }
                bvars = {
                    i: model.addVar(
                        lb=tw_open[i],
                        ub=tw_close[i],
                        vtype=GRB.CONTINUOUS,
                        name=f"time_{i}",
                    )
                    for i in customers
                }
                uvars = {
                    i: model.addVar(
                        lb=demand[i],
                        ub=capacity,
                        vtype=GRB.CONTINUOUS,
                        name=f"load_{i}",
                    )
                    for i in customers
                }

                model.setObjective(
                    gp.quicksum(dist[i][j] * xvars[i, j] for i, j in arcs),
                    GRB.MINIMIZE,
                )

                for c in customers:
                    model.addConstr(
                        gp.quicksum(xvars[a] for a in incoming[c]) == 1,
                        name=f"indegree_{c}",
                    )
                    model.addConstr(
                        gp.quicksum(xvars[a] for a in outgoing[c]) == 1,
                        name=f"outdegree_{c}",
                    )

                model.addConstr(
                    gp.quicksum(xvars[a] for a in source_arcs)
                    == gp.quicksum(xvars[a] for a in sink_arcs),
                    name="vehicle_balance",
                )

                for i, j in arcs:
                    if i == source and j in bvars:
                        model.addGenConstrIndicator(
                            xvars[i, j],
                            True,
                            bvars[j] >= tw_open[source] + service[source] + dist[i][j],
                            name=f"source_time_{j}",
                        )
                    elif i in bvars and j in bvars:
                        model.addGenConstrIndicator(
                            xvars[i, j],
                            True,
                            bvars[j] >= bvars[i] + service[i] + dist[i][j],
                            name=f"time_{i}_{j}",
                        )
                        model.addGenConstrIndicator(
                            xvars[i, j],
                            True,
                            uvars[j] >= uvars[i] + demand[j],
                            name=f"load_{i}_{j}",
                        )
                    elif i in bvars and j == sink:
                        model.addGenConstrIndicator(
                            xvars[i, j],
                            True,
                            bvars[i] + service[i] + dist[i][j] <= horizon,
                            name=f"return_{i}",
                        )

                # Warm start.
                for var in xvars.values():
                    var.Start = 0.0

                for route in best_routes:
                    feasible, _, schedule, loads = evaluate_route(route, with_schedule=True)
                    if not feasible:
                        continue
                    previous = source
                    for c in route:
                        if (previous, c) in xvars:
                            xvars[previous, c].Start = 1.0
                        bvars[c].Start = schedule[c]
                        uvars[c].Start = loads[c]
                        previous = c
                    if (previous, sink) in xvars:
                        xvars[previous, sink].Start = 1.0

                arc_list = list(xvars.keys())
                var_list = [xvars[a] for a in arc_list]

                def extract_routes(values):
                    selected = [
                        arc_list[k] for k, value in enumerate(values)
                        if value > 0.5
                    ]
                    successors = {}
                    starts = []
                    for i, j in selected:
                        if i == source:
                            starts.append(j)
                        else:
                            if i in successors:
                                return None
                            successors[i] = j

                    routes = []
                    visited = set()
                    for first in starts:
                        route = []
                        current = first
                        while current != sink:
                            if current not in customers or current in visited:
                                return None
                            visited.add(current)
                            route.append(current)
                            if current not in successors:
                                return None
                            current = successors[current]
                        routes.append(route)

                    if visited != set(customers):
                        return None
                    return routes

                def callback(cb_model, where):
                    if where != GRB.Callback.MIPSOL:
                        return
                    try:
                        values = cb_model.cbGetSolution(var_list)
                        routes = extract_routes(values)
                        if routes is not None:
                            register(routes)
                    except Exception:
                        pass

                remaining = deadline - time.monotonic()
                if remaining > 0.01:
                    model.Params.TimeLimit = max(0.01, remaining)
                    model.optimize(callback)

                    if model.SolCount > 0:
                        values = [var.X for var in var_list]
                        final_mip_routes = extract_routes(values)
                        if final_mip_routes is not None:
                            register(final_mip_routes)

        except Exception:
            # Preserve and output the best heuristic incumbent if model
            # construction, licensing, or optimization fails.
            pass

    if best_routes is None:
        raise RuntimeError("No feasible solution was found.")

    final_solution = make_solution(best_routes)
    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()