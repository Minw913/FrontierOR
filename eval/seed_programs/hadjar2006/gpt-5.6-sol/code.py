import argparse
import json
import os
import sys
import time
from collections import deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def hopcroft_karp(task_ids, adjacency):
    """Maximum-cardinality matching in the task precedence bipartite graph."""
    pair_u = {u: None for u in task_ids}
    pair_v = {v: None for v in task_ids}
    dist = {}

    def bfs():
        queue = deque()
        found = False
        for u in task_ids:
            if pair_u[u] is None:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1

        while queue:
            u = queue.popleft()
            for v in adjacency.get(u, ()):
                matched_u = pair_v[v]
                if matched_u is None:
                    found = True
                elif dist[matched_u] < 0:
                    dist[matched_u] = dist[u] + 1
                    queue.append(matched_u)
        return found

    def dfs(u):
        for v in adjacency.get(u, ()):
            matched_u = pair_v[v]
            if matched_u is None or (
                dist.get(matched_u, -1) == dist[u] + 1 and dfs(matched_u)
            ):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    while bfs():
        for u in task_ids:
            if pair_u[u] is None:
                dfs(u)

    return pair_u, pair_v


def matching_to_routes(task_ids, pair_u, pair_v):
    """Convert a task matching into vertex-disjoint paths, breaking any malformed cycles."""
    visited = set()
    routes = []

    for start in task_ids:
        if pair_v[start] is not None or start in visited:
            continue
        route = []
        current = start
        local = set()
        while current is not None and current not in visited and current not in local:
            route.append(current)
            visited.add(current)
            local.add(current)
            current = pair_u.get(current)
        if route:
            routes.append(route)

    # Compatibility should define a DAG, but defensively handle cycles or malformed input.
    for start in task_ids:
        if start in visited:
            continue
        route = []
        current = start
        local = set()
        while current is not None and current not in visited and current not in local:
            route.append(current)
            visited.add(current)
            local.add(current)
            current = pair_u.get(current)
        if route:
            routes.append(route)

    return routes


class MinCostFlow:
    def __init__(self, n):
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u, v, capacity, cost):
        forward = [v, len(self.graph[v]), capacity, float(cost)]
        reverse = [u, len(self.graph[u]), 0, -float(cost)]
        self.graph[u].append(forward)
        self.graph[v].append(reverse)
        return forward

    def send_flow(self, source, sink, required_flow):
        total_flow = 0
        total_cost = 0.0
        n = len(self.graph)
        eps = 1e-12

        while total_flow < required_flow:
            dist = [float("inf")] * n
            parent_node = [-1] * n
            parent_edge = [-1] * n
            in_queue = [False] * n

            dist[source] = 0.0
            queue = deque([source])
            in_queue[source] = True

            while queue:
                u = queue.popleft()
                in_queue[u] = False
                for edge_index, edge in enumerate(self.graph[u]):
                    v, _, capacity, cost = edge
                    if capacity <= 0:
                        continue
                    candidate = dist[u] + cost
                    if candidate + eps < dist[v]:
                        dist[v] = candidate
                        parent_node[v] = u
                        parent_edge[v] = edge_index
                        if not in_queue[v]:
                            in_queue[v] = True
                            queue.append(v)

            if parent_node[sink] < 0:
                break

            augment = required_flow - total_flow
            v = sink
            while v != source:
                u = parent_node[v]
                edge = self.graph[u][parent_edge[v]]
                augment = min(augment, edge[2])
                v = u

            v = sink
            while v != source:
                u = parent_node[v]
                edge_index = parent_edge[v]
                edge = self.graph[u][edge_index]
                reverse_index = edge[1]
                edge[2] -= augment
                self.graph[v][reverse_index][2] += augment
                v = u

            total_flow += augment
            total_cost += augment * dist[sink]

        return total_flow, total_cost


def assign_routes_to_depots(routes, depot_ids, capacities, out_cost, in_cost):
    """Min-cost capacitated assignment of complete routes to depots."""
    route_count = len(routes)
    depot_count = len(depot_ids)

    source = 0
    route_offset = 1
    depot_offset = route_offset + route_count
    sink = depot_offset + depot_count

    mcf = MinCostFlow(sink + 1)
    assignment_edges = {}

    for r in range(route_count):
        mcf.add_edge(source, route_offset + r, 1, 0.0)

    for depot_index, depot_id in enumerate(depot_ids):
        mcf.add_edge(
            depot_offset + depot_index,
            sink,
            int(capacities[depot_id]),
            0.0,
        )

    for r, route in enumerate(routes):
        first = route[0]
        last = route[-1]
        for depot_index, depot_id in enumerate(depot_ids):
            if (depot_id, first) not in out_cost:
                continue
            if (last, depot_id) not in in_cost:
                continue
            cost = out_cost[(depot_id, first)] + in_cost[(last, depot_id)]
            edge = mcf.add_edge(
                route_offset + r,
                depot_offset + depot_index,
                1,
                cost,
            )
            assignment_edges[(r, depot_id)] = edge

    flow, _ = mcf.send_flow(source, sink, route_count)
    if flow != route_count:
        return None

    assigned = []
    for r in range(route_count):
        chosen = None
        for depot_id in depot_ids:
            edge = assignment_edges.get((r, depot_id))
            if edge is not None and edge[2] == 0:
                chosen = depot_id
                break
        if chosen is None:
            return None
        assigned.append(chosen)

    return assigned


def make_solution(routes, assigned_depots, out_cost, in_cost, pair_cost):
    output_routes = []
    objective = 0.0

    for tasks, depot_id in zip(routes, assigned_depots):
        route_cost = out_cost[(depot_id, tasks[0])]
        route_cost += in_cost[(tasks[-1], depot_id)]
        for i, j in zip(tasks, tasks[1:]):
            route_cost += pair_cost[(i, j)]

        route_cost = float(route_cost)
        objective += route_cost
        output_routes.append(
            {
                "depot": int(depot_id),
                "tasks": [int(task_id) for task_id in tasks],
                "cost": route_cost,
            }
        )

    output_routes.sort(
        key=lambda route: (
            route["depot"],
            route["tasks"][0] if route["tasks"] else -1,
            tuple(route["tasks"]),
        )
    )

    return {
        "objective_value": float(objective),
        "routes": output_routes,
    }


def extract_model_solution(
    value_getter,
    active_depots,
    task_ids,
    successors,
    start_vars,
    end_vars,
    connection_vars,
    out_cost,
    in_cost,
    pair_cost,
):
    selected_start = {}
    selected_end = set()
    selected_next = {}

    for (d, i), variable in start_vars.items():
        if value_getter(variable) > 0.5:
            selected_start[i] = d

    for (d, i), variable in end_vars.items():
        if value_getter(variable) > 0.5:
            selected_end.add((d, i))

    for (d, i, j), variable in connection_vars.items():
        if value_getter(variable) > 0.5:
            selected_next[(d, i)] = j

    routes = []
    depots = []
    covered = set()

    starts = sorted(selected_start.items(), key=lambda item: (item[1], item[0]))
    for first, depot_id in starts:
        route = []
        current = first
        local = set()

        while current not in local and current not in covered:
            route.append(current)
            local.add(current)
            covered.add(current)

            if (depot_id, current) in selected_end:
                break
            next_task = selected_next.get((depot_id, current))
            if next_task is None:
                break
            current = next_task

        if route:
            routes.append(route)
            depots.append(depot_id)

    # A valid incumbent must be entirely decomposable from selected pull-outs.
    if len(covered) != len(task_ids):
        return None

    return make_solution(routes, depots, out_cost, in_cost, pair_cost)


def write_solution(path, solution):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(solution, file, indent=2, allow_nan=False)


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

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    tasks_data = instance.get("tasks", [])
    depots_data = instance.get("depots", [])

    task_ids = [int(task["id"]) for task in tasks_data]
    depot_ids = [int(depot["id"]) for depot in depots_data]
    capacities = {
        int(depot["id"]): int(depot["capacity"]) for depot in depots_data
    }
    active_depots = [d for d in depot_ids if capacities.get(d, 0) > 0]

    if not task_ids:
        solution = {"objective_value": 0.0, "routes": []}
        if logger:
            logger.log_solution(0.0, solution)
        write_solution(args.solution_path, solution)
        return

    pair_cost = {}
    successors = {i: [] for i in task_ids}
    predecessors = {i: [] for i in task_ids}

    for record in instance.get("compatible_pairs", []):
        i = int(record["task_i"])
        j = int(record["task_j"])
        cost = float(record["cost"])
        key = (i, j)
        if key not in pair_cost or cost < pair_cost[key]:
            pair_cost[key] = cost

    for i, j in pair_cost:
        if i in successors and j in predecessors:
            successors[i].append(j)
            predecessors[j].append(i)

    for i in task_ids:
        successors[i].sort()
        predecessors[i].sort()

    out_cost = {}
    for record in instance.get("depot_to_task_costs", []):
        d = int(record["depot_id"])
        i = int(record["task_id"])
        cost = float(record["cost"])
        key = (d, i)
        if key not in out_cost or cost < out_cost[key]:
            out_cost[key] = cost

    in_cost = {}
    for record in instance.get("task_to_depot_costs", []):
        i = int(record["task_id"])
        d = int(record["depot_id"])
        cost = float(record["cost"])
        key = (i, d)
        if key not in in_cost or cost < in_cost[key]:
            in_cost[key] = cost

    # Construct a fast feasible incumbent via minimum path cover and depot assignment.
    pair_u, pair_v = hopcroft_karp(task_ids, successors)
    initial_routes = matching_to_routes(task_ids, pair_u, pair_v)
    initial_depots = assign_routes_to_depots(
        initial_routes,
        active_depots,
        capacities,
        out_cost,
        in_cost,
    )

    initial_solution = None
    best_solution = None
    best_objective = float("inf")

    if initial_depots is not None:
        initial_solution = make_solution(
            initial_routes,
            initial_depots,
            out_cost,
            in_cost,
            pair_cost,
        )
        best_solution = initial_solution
        best_objective = initial_solution["objective_value"]
        if logger:
            logger.log_solution(best_objective, initial_solution)

    remaining = deadline - time.monotonic()
    if remaining <= 0.05:
        if best_solution is None:
            raise RuntimeError("No feasible solution could be constructed within the time limit.")
        write_solution(args.solution_path, best_solution)
        return

    model = gp.Model("multi_depot_vehicle_scheduling")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.01, deadline - time.monotonic())

    start_vars = {}
    end_vars = {}
    connection_vars = {}

    for d in active_depots:
        for i in task_ids:
            if (d, i) in out_cost:
                start_vars[(d, i)] = model.addVar(
                    vtype=GRB.BINARY,
                    obj=out_cost[(d, i)],
                    name=f"start_{d}_{i}",
                )
            if (i, d) in in_cost:
                end_vars[(d, i)] = model.addVar(
                    vtype=GRB.BINARY,
                    obj=in_cost[(i, d)],
                    name=f"end_{d}_{i}",
                )

        for i, j in pair_cost:
            connection_vars[(d, i, j)] = model.addVar(
                vtype=GRB.BINARY,
                obj=pair_cost[(i, j)],
                name=f"connect_{d}_{i}_{j}",
            )

    model.ModelSense = GRB.MINIMIZE
    model.update()

    for d in active_depots:
        capacity_expr = gp.quicksum(
            start_vars[(d, i)]
            for i in task_ids
            if (d, i) in start_vars
        )
        model.addConstr(
            capacity_expr <= capacities[d],
            name=f"capacity_{d}",
        )

        # Explicit depot balance.
        model.addConstr(
            gp.quicksum(
                start_vars[(d, i)]
                for i in task_ids
                if (d, i) in start_vars
            )
            == gp.quicksum(
                end_vars[(d, i)]
                for i in task_ids
                if (d, i) in end_vars
            ),
            name=f"depot_balance_{d}",
        )

        for i in task_ids:
            incoming = gp.LinExpr()
            outgoing = gp.LinExpr()

            if (d, i) in start_vars:
                incoming += start_vars[(d, i)]
            if (d, i) in end_vars:
                outgoing += end_vars[(d, i)]

            for predecessor in predecessors[i]:
                variable = connection_vars.get((d, predecessor, i))
                if variable is not None:
                    incoming += variable

            for successor in successors[i]:
                variable = connection_vars.get((d, i, successor))
                if variable is not None:
                    outgoing += variable

            model.addConstr(
                incoming == outgoing,
                name=f"flow_{d}_{i}",
            )

    for i in task_ids:
        coverage = gp.LinExpr()
        for d in active_depots:
            if (d, i) in start_vars:
                coverage += start_vars[(d, i)]
            for predecessor in predecessors[i]:
                variable = connection_vars.get((d, predecessor, i))
                if variable is not None:
                    coverage += variable

        model.addConstr(coverage == 1, name=f"cover_{i}")

    # Supply the heuristic incumbent as a complete MIP start.
    if initial_solution is not None:
        for variable in start_vars.values():
            variable.Start = 0.0
        for variable in end_vars.values():
            variable.Start = 0.0
        for variable in connection_vars.values():
            variable.Start = 0.0

        for route, depot_id in zip(initial_routes, initial_depots):
            first = route[0]
            last = route[-1]
            if (depot_id, first) in start_vars:
                start_vars[(depot_id, first)].Start = 1.0
            if (depot_id, last) in end_vars:
                end_vars[(depot_id, last)].Start = 1.0
            for i, j in zip(route, route[1:]):
                connection_vars[(depot_id, i, j)].Start = 1.0

    incumbent_state = {
        "objective": best_objective,
        "solution": best_solution,
    }

    def callback(cb_model, where):
        if time.monotonic() >= deadline:
            cb_model.terminate()
            return

        if where != GRB.Callback.MIPSOL:
            return

        try:
            callback_objective = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if callback_objective >= incumbent_state["objective"] - 1e-7:
                return

            solution = extract_model_solution(
                lambda variable: cb_model.cbGetSolution(variable),
                active_depots,
                task_ids,
                successors,
                start_vars,
                end_vars,
                connection_vars,
                out_cost,
                in_cost,
                pair_cost,
            )
            if solution is None:
                return

            objective = solution["objective_value"]
            if objective < incumbent_state["objective"] - 1e-7:
                incumbent_state["objective"] = objective
                incumbent_state["solution"] = solution
                if logger:
                    logger.log_solution(objective, solution)
        except Exception:
            # Logging or extraction must not abort optimization.
            pass

    model.Params.TimeLimit = max(0.01, deadline - time.monotonic())
    model.optimize(callback)

    if model.SolCount > 0:
        final_solution = extract_model_solution(
            lambda variable: variable.X,
            active_depots,
            task_ids,
            successors,
            start_vars,
            end_vars,
            connection_vars,
            out_cost,
            in_cost,
            pair_cost,
        )
        if (
            final_solution is not None
            and final_solution["objective_value"]
            < incumbent_state["objective"] - 1e-7
        ):
            incumbent_state["objective"] = final_solution["objective_value"]
            incumbent_state["solution"] = final_solution
            if logger:
                logger.log_solution(
                    final_solution["objective_value"],
                    final_solution,
                )

    best_solution = incumbent_state["solution"]
    if best_solution is None:
        raise RuntimeError("The instance is infeasible or no feasible solution was found.")

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    sys.setrecursionlimit(1_000_000)
    main()