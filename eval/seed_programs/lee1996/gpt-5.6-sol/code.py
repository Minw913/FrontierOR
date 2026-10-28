import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def edge_cost(costs, i, j):
    return float(costs[i][j])


def minimum_spanning_tree(active, costs):
    active = sorted(active)
    if len(active) <= 1:
        return [], 0.0

    in_tree = {active[0]}
    remaining = set(active[1:])
    best_cost = {v: edge_cost(costs, active[0], v) for v in remaining}
    best_parent = {v: active[0] for v in remaining}

    edges = []
    total = 0.0

    while remaining:
        v = min(remaining, key=lambda node: (best_cost[node], node))
        parent = best_parent[v]
        a, b = sorted((parent, v))
        edges.append([a, b])
        total += edge_cost(costs, parent, v)

        remaining.remove(v)
        in_tree.add(v)

        for u in remaining:
            c = edge_cost(costs, v, u)
            if c < best_cost[u] - 1e-12 or (
                abs(c - best_cost[u]) <= 1e-12 and v < best_parent[u]
            ):
                best_cost[u] = c
                best_parent[u] = v

    edges.sort()
    return edges, total


def build_solution(active, tree_edges, assignment_costs, interconnection_costs,
                   node_weights):
    active = sorted(set(active))
    if not active:
        raise ValueError("A feasible solution must activate at least one Steiner node.")

    assignments = {}
    assignment_total = 0.0
    for i, row in enumerate(assignment_costs):
        facility = min(active, key=lambda j: (float(row[j]), j))
        assignments[str(i)] = facility
        assignment_total += float(row[facility])

    normalized_edges = []
    tree_total = 0.0
    for edge in tree_edges:
        a, b = sorted((int(edge[0]), int(edge[1])))
        normalized_edges.append([a, b])
        tree_total += edge_cost(interconnection_costs, a, b)
    normalized_edges.sort()

    activation_total = sum(float(node_weights[j]) for j in active)
    objective = assignment_total + tree_total + activation_total

    return {
        "objective_value": float(objective),
        "active_steiner_nodes": active,
        "assignments": assignments,
        "steiner_tree_edges": normalized_edges,
    }


def is_connected_tree(active, edges):
    active = sorted(set(active))
    if not active:
        return False
    if len(edges) != len(active) - 1:
        return False

    active_set = set(active)
    adjacency = {v: [] for v in active}
    for a, b in edges:
        if a not in active_set or b not in active_set or a == b:
            return False
        adjacency[a].append(b)
        adjacency[b].append(a)

    reached = {active[0]}
    stack = [active[0]]
    while stack:
        v = stack.pop()
        for u in adjacency[v]:
            if u not in reached:
                reached.add(u)
                stack.append(u)
    return len(reached) == len(active)


def heuristic_search(m, n, assignment_costs, interconnection_costs,
                     node_weights, deadline, consider):
    # Best one-facility solution.
    single_scores = []
    for j in range(n):
        score = float(node_weights[j])
        for i in range(m):
            score += float(assignment_costs[i][j])
        single_scores.append((score, j))

    single_scores.sort()
    best_single = single_scores[0][1]
    consider([best_single], [])

    # Restrict expensive forward searches for very large instances.
    if n <= 400:
        pool = [j for _, j in single_scores]
    else:
        pool_set = {j for _, j in single_scores[:300]}
        pool_set.update(
            j for _, j in sorted(
                (float(node_weights[j]), j) for j in range(n)
            )[:100]
        )
        pool = sorted(pool_set)

    seed_count = min(3, len(single_scores))
    for seed_pos in range(seed_count):
        if time.monotonic() >= deadline:
            break

        seed = single_scores[seed_pos][1]
        selected = {seed}
        current_assignment = [
            float(assignment_costs[i][seed]) for i in range(m)
        ]
        nearest_connection = {
            j: edge_cost(interconnection_costs, seed, j)
            for j in pool if j != seed
        }

        edges, _ = minimum_spanning_tree(selected, interconnection_costs)
        consider(selected, edges)

        for _ in range(min(40, len(pool) - 1)):
            if time.monotonic() >= deadline:
                break

            best_candidate = None
            best_delta = 0.0

            for j in pool:
                if j in selected:
                    continue

                savings = 0.0
                for i in range(m):
                    improvement = (
                        current_assignment[i] -
                        float(assignment_costs[i][j])
                    )
                    if improvement > 0.0:
                        savings += improvement

                delta = (
                    float(node_weights[j])
                    + nearest_connection[j]
                    - savings
                )
                if (
                    best_candidate is None
                    or delta < best_delta - 1e-12
                    or (abs(delta - best_delta) <= 1e-12 and j < best_candidate)
                ):
                    best_delta = delta
                    best_candidate = j

            if best_candidate is None or best_delta >= -1e-10:
                break

            new_node = best_candidate
            selected.add(new_node)

            for i in range(m):
                c = float(assignment_costs[i][new_node])
                if c < current_assignment[i]:
                    current_assignment[i] = c

            for j in pool:
                if j not in selected:
                    c = edge_cost(interconnection_costs, new_node, j)
                    if c < nearest_connection[j]:
                        nearest_connection[j] = c

            edges, _ = minimum_spanning_tree(selected, interconnection_costs)
            consider(selected, edges)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    overall_deadline = start_time + max(0, args.time_limit)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    m = int(data["parameters"]["m"])
    n = int(data["parameters"]["n"])
    assignment_costs = data["cost_matrices"]["assignment_costs"]["matrix"]
    interconnection_costs = data["cost_matrices"]["interconnection_costs"]["matrix"]
    node_weights = data["steiner_nodes"]["node_weights"]

    if n <= 0:
        raise ValueError("The instance has no Steiner nodes.")
    if len(assignment_costs) != m or len(node_weights) != n:
        raise ValueError("Instance dimensions do not match parameters m and n.")

    best_solution = None
    best_objective = math.inf

    def consider(active, edges):
        nonlocal best_solution, best_objective
        candidate = build_solution(
            active,
            edges,
            assignment_costs,
            interconnection_costs,
            node_weights,
        )
        value = candidate["objective_value"]
        tolerance = 1e-8 * max(1.0, abs(best_objective)) if math.isfinite(best_objective) else 0.0
        if value < best_objective - tolerance:
            best_objective = value
            best_solution = candidate
            if logger:
                logger.log_solution(value, candidate)

    # Spend a modest part of the available time on a constructive warm start.
    now = time.monotonic()
    remaining_initial = max(0.0, overall_deadline - now)
    heuristic_budget = min(3.0, max(0.0, 0.12 * remaining_initial))
    heuristic_deadline = min(overall_deadline, now + heuristic_budget)

    heuristic_search(
        m,
        n,
        assignment_costs,
        interconnection_costs,
        node_weights,
        heuristic_deadline,
        consider,
    )

    # The singleton scan always supplies a feasible fallback.
    if best_solution is None:
        j = min(
            range(n),
            key=lambda v: (
                float(node_weights[v]) +
                sum(float(assignment_costs[i][v]) for i in range(m)),
                v,
            ),
        )
        consider([j], [])

    remaining = overall_deadline - time.monotonic()

    if remaining > 0.05:
        try:
            model = gp.Model("connected_facility_tree")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, remaining)
            model.Params.LazyConstraints = 1

            y = model.addVars(n, vtype=GRB.BINARY, name="activate")
            x = model.addVars(
                m, n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="assign"
            )
            edge_keys = [(i, j) for i in range(n) for j in range(i + 1, n)]
            e = model.addVars(edge_keys, vtype=GRB.BINARY, name="edge")

            for i in range(m):
                model.addConstr(
                    gp.quicksum(x[i, j] for j in range(n)) == 1.0,
                    name=f"assign_once_{i}",
                )
                for j in range(n):
                    model.addConstr(x[i, j] <= y[j])

            for i, j in edge_keys:
                model.addConstr(e[i, j] <= y[i])
                model.addConstr(e[i, j] <= y[j])

            model.addConstr(
                gp.quicksum(e[i, j] for i, j in edge_keys)
                == gp.quicksum(y[j] for j in range(n)) - 1,
                name="tree_edge_count",
            )

            objective = (
                gp.quicksum(
                    float(assignment_costs[i][j]) * x[i, j]
                    for i in range(m) for j in range(n)
                )
                + gp.quicksum(
                    edge_cost(interconnection_costs, i, j) * e[i, j]
                    for i, j in edge_keys
                )
                + gp.quicksum(float(node_weights[j]) * y[j] for j in range(n))
            )
            model.setObjective(objective, GRB.MINIMIZE)

            # Install the heuristic incumbent as a MIP start.
            warm_active = set(best_solution["active_steiner_nodes"])
            warm_edges = {
                tuple(sorted(edge))
                for edge in best_solution["steiner_tree_edges"]
            }
            warm_assignments = best_solution["assignments"]

            for j in range(n):
                y[j].Start = 1.0 if j in warm_active else 0.0
            for i, j in edge_keys:
                e[i, j].Start = 1.0 if (i, j) in warm_edges else 0.0
            for i in range(m):
                assigned = int(warm_assignments[str(i)])
                for j in range(n):
                    x[i, j].Start = 1.0 if j == assigned else 0.0

            model._y = y
            model._e = e
            model._n = n
            model._assignment_costs = assignment_costs
            model._interconnection_costs = interconnection_costs
            model._node_weights = node_weights
            model._consider = consider

            def callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return

                y_values = cb_model.cbGetSolution(cb_model._y)
                e_values = cb_model.cbGetSolution(cb_model._e)

                active = [j for j in range(cb_model._n) if y_values[j] > 0.5]
                if not active:
                    return

                active_set = set(active)
                adjacency = {j: [] for j in active}
                selected_edges = []

                for (i, j), value in e_values.items():
                    if value > 0.5:
                        selected_edges.append([i, j])
                        if i in active_set and j in active_set:
                            adjacency[i].append(j)
                            adjacency[j].append(i)

                components = []
                unseen = set(active)
                while unseen:
                    root = next(iter(unseen))
                    component = {root}
                    stack = [root]
                    unseen.remove(root)
                    while stack:
                        v = stack.pop()
                        for u in adjacency[v]:
                            if u in unseen:
                                unseen.remove(u)
                                component.add(u)
                                stack.append(u)
                    components.append(component)

                if len(components) > 1:
                    # Add a globally valid conditional connectivity cut:
                    # if one selected node lies on each side, an edge must cross.
                    component = min(
                        components,
                        key=lambda comp: len(comp) * (cb_model._n - len(comp)),
                    )
                    outside_active = active_set - component
                    i_rep = min(component)
                    j_rep = min(outside_active)

                    crossing = []
                    for a in component:
                        for b in range(cb_model._n):
                            if b not in component:
                                p, q = (a, b) if a < b else (b, a)
                                crossing.append(cb_model._e[p, q])

                    cb_model.cbLazy(
                        gp.quicksum(crossing)
                        >= cb_model._y[i_rep] + cb_model._y[j_rep] - 1
                    )
                    return

                if is_connected_tree(active, selected_edges):
                    cb_model._consider(active, selected_edges)

            model.optimize(callback)

            if model.SolCount > 0:
                active = [j for j in range(n) if y[j].X > 0.5]
                selected_edges = [
                    [i, j] for i, j in edge_keys if e[i, j].X > 0.5
                ]
                if is_connected_tree(active, selected_edges):
                    consider(active, selected_edges)

        except gp.GurobiError:
            # Preserve and output the constructive feasible solution if the
            # optimizer is unavailable or terminates unexpectedly.
            pass

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()