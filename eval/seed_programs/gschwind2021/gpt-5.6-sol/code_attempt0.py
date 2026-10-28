import argparse
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


def make_solution(groups, vertices):
    return {
        "objective_value": float(len(groups)),
        "solution": [[vertices[i] for i in group] for group in groups],
    }


def is_valid_group(group, adj, s, connectivity_required):
    if not group:
        return False
    if len(group) == 1:
        return True

    group_set = set(group)

    for v in group:
        nonneighbors = 0
        for u in group:
            if u != v and u not in adj[v]:
                nonneighbors += 1
                if nonneighbors > s - 1:
                    return False

    if connectivity_required:
        seen = {group[0]}
        stack = [group[0]]
        while stack:
            v = stack.pop()
            for u in adj[v]:
                if u in group_set and u not in seen:
                    seen.add(u)
                    stack.append(u)
        if len(seen) != len(group):
            return False

    return True


def induced_components(group, adj):
    remaining = set(group)
    components = []

    while remaining:
        root = next(iter(remaining))
        remaining.remove(root)
        component = {root}
        stack = [root]

        while stack:
            v = stack.pop()
            for u in adj[v]:
                if u in remaining:
                    remaining.remove(u)
                    component.add(u)
                    stack.append(u)

        components.append(component)

    return components


def greedy_partition(order, adj, s, connectivity_required):
    groups = []
    missing_counts = []

    for v in order:
        best_group = -1
        best_size = -1

        for k, group in enumerate(groups):
            if connectivity_required and group:
                if not any(u in adj[v] for u in group):
                    continue

            new_vertex_missing = 0
            feasible = True

            for u in group:
                if u not in adj[v]:
                    new_vertex_missing += 1
                    if new_vertex_missing > s - 1:
                        feasible = False
                        break
                    if missing_counts[k][u] >= s - 1:
                        feasible = False
                        break

            if feasible and len(group) > best_size:
                best_group = k
                best_size = len(group)

        if best_group < 0:
            groups.append([v])
            missing_counts.append({v: 0})
        else:
            group = groups[best_group]
            misses = missing_counts[best_group]
            v_missing = 0
            for u in group:
                if u not in adj[v]:
                    misses[u] += 1
                    v_missing += 1
            group.append(v)
            misses[v] = v_missing

    return groups


def merge_groups(groups, adj, s, connectivity_required, deadline):
    groups = [list(g) for g in groups]
    improved = True

    while improved and time.monotonic() < deadline:
        improved = False
        order = sorted(range(len(groups)), key=lambda k: len(groups[k]))
        merged_pair = None

        for pos_i, i in enumerate(order):
            if time.monotonic() >= deadline:
                break
            for j in order[pos_i + 1:]:
                if i == j:
                    continue
                union = groups[i] + groups[j]
                if is_valid_group(union, adj, s, connectivity_required):
                    merged_pair = (min(i, j), max(i, j), union)
                    break
            if merged_pair is not None:
                break

        if merged_pair is not None:
            i, j, union = merged_pair
            groups[i] = union
            del groups[j]
            improved = True

    return groups


def solve_with_gurobi(
    n,
    vertices,
    adj,
    s,
    decomposition_type,
    connectivity_required,
    initial_groups,
    deadline,
    logger,
    best_holder,
):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return

    group_limit = len(initial_groups)
    if group_limit <= 1:
        return

    remaining = deadline - time.monotonic()
    if remaining <= 0.05:
        return

    nonneighbors = []
    nonedge_count = 0
    all_vertices = set(range(n))

    for v in range(n):
        nv = list(all_vertices - adj[v] - {v})
        nonneighbors.append(nv)
        nonedge_count += len(nv)

    nonedge_count //= 2

    if s == 1:
        estimated_terms = nonedge_count * group_limit
    else:
        estimated_terms = sum(len(x) for x in nonneighbors) * group_limit

    estimated_variables = n * group_limit
    operation_budget = max(300000, min(8000000, int(120000 * remaining)))

    if estimated_variables > 350000 or estimated_terms > operation_budget:
        return

    try:
        model = gp.Model("minimum_splex_decomposition")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.01, deadline - time.monotonic())

        use_lazy_connectivity = connectivity_required and s > 1
        if use_lazy_connectivity:
            model.Params.LazyConstraints = 1
            model.Params.PreCrush = 1

        x = model.addVars(n, group_limit, vtype=GRB.BINARY, name="x")
        y = model.addVars(group_limit, vtype=GRB.BINARY, name="y")

        for v in range(n):
            if decomposition_type == "partitioning":
                model.addConstr(
                    gp.quicksum(x[v, k] for k in range(group_limit)) == 1
                )
            else:
                model.addConstr(
                    gp.quicksum(x[v, k] for k in range(group_limit)) >= 1
                )

        for k in range(group_limit):
            model.addConstr(
                gp.quicksum(x[v, k] for v in range(n)) >= y[k]
            )
            for v in range(n):
                model.addConstr(x[v, k] <= y[k])

        for k in range(group_limit - 1):
            model.addConstr(y[k] >= y[k + 1])

        if s == 1:
            for u in range(n):
                for v in nonneighbors[u]:
                    if u < v:
                        for k in range(group_limit):
                            model.addConstr(x[u, k] + x[v, k] <= 1)
        else:
            allowed_missing = s - 1
            for v in range(n):
                d_non = len(nonneighbors[v])
                if d_non <= allowed_missing:
                    continue
                coefficient = d_non - allowed_missing
                for k in range(group_limit):
                    model.addConstr(
                        gp.quicksum(x[u, k] for u in nonneighbors[v])
                        <= d_non - coefficient * x[v, k]
                    )

        maximum_group_size = min(
            n, max((len(adj[v]) + s for v in range(n)), default=1)
        )
        if maximum_group_size > 0:
            lower_bound = int(math.ceil(n / maximum_group_size))
            model.addConstr(
                gp.quicksum(y[k] for k in range(group_limit)) >= lower_bound
            )

        model.setObjective(
            gp.quicksum(y[k] for k in range(group_limit)), GRB.MINIMIZE
        )

        # Warm start from the heuristic partition.
        group_of_vertex = {}
        for k, group in enumerate(initial_groups):
            y[k].Start = 1.0
            for v in group:
                group_of_vertex[v] = k

        for k in range(len(initial_groups), group_limit):
            y[k].Start = 0.0

        for v in range(n):
            assigned = group_of_vertex.get(v, -1)
            for k in range(group_limit):
                x[v, k].Start = 1.0 if k == assigned else 0.0

        flat_x = [x[v, k] for k in range(group_limit) for v in range(n)]
        flat_y = [y[k] for k in range(group_limit)]

        model._x = x
        model._y = y
        model._flat_x = flat_x
        model._flat_y = flat_y
        model._n = n
        model._group_limit = group_limit
        model._adj = adj
        model._vertices = vertices
        model._connectivity = use_lazy_connectivity
        model._logger = logger
        model._best_holder = best_holder

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                x_values = cb_model.cbGetSolution(cb_model._flat_x)
                y_values = cb_model.cbGetSolution(cb_model._flat_y)

                selected_groups = []
                disconnected = False

                for k in range(cb_model._group_limit):
                    if y_values[k] <= 0.5:
                        continue

                    group = [
                        v
                        for v in range(cb_model._n)
                        if x_values[k * cb_model._n + v] > 0.5
                    ]

                    if not group:
                        continue

                    if cb_model._connectivity and len(group) > 1:
                        components = induced_components(group, cb_model._adj)
                        if len(components) > 1:
                            disconnected = True
                            group_set = set(group)

                            for component in components:
                                outside_selected = group_set - component
                                if not outside_selected:
                                    continue

                                v_rep = next(iter(component))
                                w_rep = next(iter(outside_selected))

                                boundary = set()
                                for u in component:
                                    boundary.update(cb_model._adj[u])
                                boundary.difference_update(component)

                                cb_model.cbLazy(
                                    gp.quicksum(
                                        cb_model._x[u, k] for u in boundary
                                    )
                                    >= cb_model._x[v_rep, k]
                                    + cb_model._x[w_rep, k]
                                    - 1
                                )

                    selected_groups.append(group)

                if disconnected:
                    return

                objective = len(selected_groups)
                if objective < cb_model._best_holder["objective"]:
                    solution = make_solution(
                        selected_groups, cb_model._vertices
                    )
                    cb_model._best_holder["objective"] = objective
                    cb_model._best_holder["groups"] = [
                        list(g) for g in selected_groups
                    ]
                    cb_model._best_holder["solution"] = solution
                    if cb_model._logger:
                        cb_model._logger.log_solution(
                            float(objective), solution
                        )
            except Exception:
                # A callback failure must not prevent returning the best
                # heuristic incumbent.
                return

        model.optimize(callback)

        if model.SolCount > 0:
            groups = []
            for k in range(group_limit):
                if y[k].X <= 0.5:
                    continue
                group = [v for v in range(n) if x[v, k].X > 0.5]
                if group:
                    groups.append(group)

            valid = True
            covered = [0] * n
            for group in groups:
                if not is_valid_group(group, adj, s, connectivity_required):
                    valid = False
                    break
                for v in group:
                    covered[v] += 1

            if decomposition_type == "partitioning":
                valid = valid and all(count == 1 for count in covered)
            else:
                valid = valid and all(count >= 1 for count in covered)

            if valid and len(groups) < best_holder["objective"]:
                solution = make_solution(groups, vertices)
                best_holder["objective"] = len(groups)
                best_holder["groups"] = [list(g) for g in groups]
                best_holder["solution"] = solution
                if logger:
                    logger.log_solution(float(len(groups)), solution)

    except Exception:
        return


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
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    graph = instance["graph"]
    settings = instance["problem_settings"]

    vertices = list(graph["vertices"])
    n = len(vertices)
    vertex_index = {vertex: i for i, vertex in enumerate(vertices)}

    adj = [set() for _ in range(n)]
    for edge in graph["edges"]:
        if len(edge) != 2:
            continue
        a, b = edge
        if a not in vertex_index or b not in vertex_index:
            continue
        u = vertex_index[a]
        v = vertex_index[b]
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)

    s = max(1, int(settings["parameter_s"]))
    decomposition_type = settings["decomposition_type"].lower()
    connectivity_required = bool(settings["connectivity_required"])

    if n == 0:
        final_solution = {"objective_value": 0.0, "solution": []}
        if logger:
            logger.log_solution(0.0, final_solution)
        parent = os.path.dirname(os.path.abspath(args.solution_path))
        os.makedirs(parent, exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(final_solution, f)
        return

    # Singleton groups are always a valid initial incumbent.
    singleton_groups = [[v] for v in range(n)]
    best_holder = {
        "objective": n,
        "groups": singleton_groups,
        "solution": make_solution(singleton_groups, vertices),
    }

    if logger:
        logger.log_solution(float(n), best_holder["solution"])

    heuristic_time = min(1.5, max(0.05, args.time_limit * 0.10))
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_time)

    degrees = [len(adj[v]) for v in range(n)]
    orders = [
        sorted(range(n), key=lambda v: (-degrees[v], v)),
        sorted(range(n), key=lambda v: (degrees[v], v)),
    ]

    rng = random.Random(0)
    random_trials = 4 if n > 1000 else 20
    for _ in range(random_trials):
        order = list(range(n))
        rng.shuffle(order)
        orders.append(order)

    for order in orders:
        if time.monotonic() >= heuristic_deadline:
            break

        candidate = greedy_partition(
            order, adj, s, connectivity_required
        )

        merge_deadline = min(
            heuristic_deadline, time.monotonic() + 0.08
        )
        candidate = merge_groups(
            candidate, adj, s, connectivity_required, merge_deadline
        )

        if len(candidate) < best_holder["objective"]:
            solution = make_solution(candidate, vertices)
            best_holder["objective"] = len(candidate)
            best_holder["groups"] = [list(g) for g in candidate]
            best_holder["solution"] = solution
            if logger:
                logger.log_solution(float(len(candidate)), solution)

        if best_holder["objective"] == 1:
            break

    if best_holder["objective"] > 1 and time.monotonic() < deadline:
        solve_with_gurobi(
            n=n,
            vertices=vertices,
            adj=adj,
            s=s,
            decomposition_type=decomposition_type,
            connectivity_required=connectivity_required,
            initial_groups=best_holder["groups"],
            deadline=deadline,
            logger=logger,
            best_holder=best_holder,
        )

    final_solution = best_holder["solution"]
    final_solution["objective_value"] = float(len(final_solution["solution"]))

    parent = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(parent, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f)


if __name__ == "__main__":
    main()