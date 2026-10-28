import argparse
import json
import os
import random
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def make_solution(selected, colors, num_clusters):
    return {
        "objective_value": float(len(set(colors.values()))),
        "selected_vertices": {
            str(i): int(selected[i]) for i in range(num_clusters)
        },
        "coloring": {
            str(int(selected[i])): int(colors[selected[i]])
            for i in range(num_clusters)
        },
    }


def canonicalize_colors(selected, colors, cluster_order):
    """Relabel colors by first appearance in cluster_order."""
    mapping = {}
    next_color = 0
    result = {}
    for ci in cluster_order:
        v = selected[ci]
        old = colors[v]
        if old not in mapping:
            mapping[old] = next_color
            next_color += 1
        result[v] = mapping[old]
    return result


def dsatur_coloring(selected, adjacency, cluster_order):
    vertices = list(selected.values())
    vertex_position = {selected[ci]: pos for pos, ci in enumerate(cluster_order)}
    selected_set = set(vertices)
    induced_degree = {
        v: sum(1 for u in adjacency[v] if u in selected_set) for v in vertices
    }

    colors = {}
    neighbor_colors = {v: set() for v in vertices}

    while len(colors) < len(vertices):
        uncolored = (v for v in vertices if v not in colors)
        v = max(
            uncolored,
            key=lambda z: (
                len(neighbor_colors[z]),
                induced_degree[z],
                -vertex_position[z],
            ),
        )

        forbidden = neighbor_colors[v]
        color = 0
        while color in forbidden:
            color += 1
        colors[v] = color

        for u in adjacency[v]:
            if u in neighbor_colors and u not in colors:
                neighbor_colors[u].add(color)

    return canonicalize_colors(selected, colors, cluster_order)


def initial_heuristics(clusters, adjacency, vertex_cluster, cluster_order,
                       start_time, time_limit):
    num_clusters = len(clusters)
    rng = random.Random(0)

    cross_degree = {}
    neighboring_clusters = {}
    for v in range(len(adjacency)):
        nc = {vertex_cluster[u] for u in adjacency[v]
              if vertex_cluster[u] != vertex_cluster[v]}
        neighboring_clusters[v] = nc
        cross_degree[v] = sum(
            1 for u in adjacency[v] if vertex_cluster[u] != vertex_cluster[v]
        )

    # Reliable deterministic baseline.
    selected = {}
    for ci in range(num_clusters):
        selected[ci] = min(
            clusters[ci],
            key=lambda v: (
                len(neighboring_clusters[v]),
                cross_degree[v],
                len(adjacency[v]),
                v,
            ),
        )
    colors = dsatur_coloring(selected, adjacency, cluster_order)
    best_selected = dict(selected)
    best_colors = dict(colors)
    best_value = len(set(colors.values()))

    heuristic_budget = min(2.0, max(0.02, time_limit * 0.10))
    deadline = min(start_time + max(0, time_limit), time.monotonic() + heuristic_budget)

    # Jointly choose representatives and colors using several cluster orders.
    iteration = 0
    while time.monotonic() < deadline and iteration < 150:
        if iteration == 0:
            order = list(cluster_order)
        elif iteration == 1:
            order = list(reversed(cluster_order))
        else:
            order = list(range(num_clusters))
            rng.shuffle(order)

        chosen = {}
        color_groups = []

        for step, ci in enumerate(order):
            best_choice = None
            best_key = None

            for v in clusters[ci]:
                future_cost = len(neighboring_clusters[v])
                degree_cost = cross_degree[v]

                for color, group in enumerate(color_groups):
                    if all(u not in adjacency[v] for u in group):
                        key = (
                            0,                       # Existing color preferred.
                            color,
                            future_cost,
                            degree_cost,
                            rng.random(),
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best_choice = (v, color)

                # A new color is always feasible.
                key = (
                    1,
                    len(color_groups),
                    future_cost,
                    degree_cost,
                    rng.random(),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_choice = (v, len(color_groups))

            v, color = best_choice
            chosen[ci] = v
            if color == len(color_groups):
                color_groups.append(set())
            color_groups[color].add(v)

        candidate_colors = {}
        for color, group in enumerate(color_groups):
            for v in group:
                candidate_colors[v] = color

        # DSATUR can improve the coloring of the representatives just chosen.
        ds_colors = dsatur_coloring(chosen, adjacency, cluster_order)
        if len(set(ds_colors.values())) <= len(set(candidate_colors.values())):
            candidate_colors = ds_colors
        else:
            candidate_colors = canonicalize_colors(
                chosen, candidate_colors, cluster_order
            )

        value = len(set(candidate_colors.values()))
        if value < best_value:
            best_value = value
            best_selected = dict(chosen)
            best_colors = dict(candidate_colors)
            if best_value == 1:
                break

        # Randomized representative selection followed by DSATUR.
        if time.monotonic() < deadline:
            random_selected = {}
            for ci in range(num_clusters):
                candidates = sorted(
                    clusters[ci],
                    key=lambda v: (
                        len(neighboring_clusters[v]),
                        cross_degree[v],
                        len(adjacency[v]),
                        v,
                    ),
                )
                pool_size = min(len(candidates), max(1, 1 + len(candidates) // 3))
                random_selected[ci] = candidates[rng.randrange(pool_size)]

            random_colors = dsatur_coloring(
                random_selected, adjacency, cluster_order
            )
            value = len(set(random_colors.values()))
            if value < best_value:
                best_value = value
                best_selected = random_selected
                best_colors = random_colors
                if best_value == 1:
                    break

        iteration += 1

    return best_selected, best_colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    graph = instance["graph"]
    partition = instance["partition"]
    num_vertices = int(graph["num_vertices"])
    clusters = [[int(v) for v in c] for c in partition["clusters"]]
    num_clusters = int(partition["num_clusters"])
    edges = [(int(e[0]), int(e[1])) for e in graph["edges"]]

    adjacency = [set() for _ in range(num_vertices)]
    vertex_cluster = [-1] * num_vertices
    for ci, cluster in enumerate(clusters):
        for v in cluster:
            vertex_cluster[v] = ci

    relevant_edges = []
    for u, v in edges:
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)
        if vertex_cluster[u] != vertex_cluster[v]:
            relevant_edges.append((u, v))

    # Put small, highly constrained clusters early. This strengthens the
    # color-symmetry rule that cluster at position i may only use colors <= i.
    cluster_order = sorted(
        range(num_clusters),
        key=lambda ci: (
            len(clusters[ci]),
            -sum(
                len({
                    vertex_cluster[u] for u in adjacency[v]
                    if vertex_cluster[u] != ci
                })
                for v in clusters[ci]
            ),
            ci,
        ),
    )
    cluster_position = {
        ci: pos for pos, ci in enumerate(cluster_order)
    }

    best_selected, best_colors = initial_heuristics(
        clusters,
        adjacency,
        vertex_cluster,
        cluster_order,
        start_time,
        max(0, args.time_limit),
    )
    best_value = len(set(best_colors.values()))
    best_solution = make_solution(
        best_selected, best_colors, num_clusters
    )

    if logger:
        logger.log_solution(float(best_value), best_solution)

    # One color is a global lower bound, so the heuristic is already optimal.
    remaining = args.time_limit - (time.monotonic() - start_time)
    if best_value > 1 and remaining > 0.01:
        max_colors = best_value
        model = gp.Model("select_and_color")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.01, remaining)

        x = {}
        variable_meta = []

        # Canonical color symmetry: after colors are ordered by first use,
        # the cluster in position i can only use colors 0,...,i.
        for ci in cluster_order:
            limit = min(max_colors, cluster_position[ci] + 1)
            for v in clusters[ci]:
                for color in range(limit):
                    var = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{v}_{color}",
                    )
                    x[v, color] = var
                    variable_meta.append((v, color, ci, var))

        y = model.addVars(
            max_colors, vtype=GRB.BINARY, name="used"
        )
        model.update()

        for ci in range(num_clusters):
            model.addConstr(
                gp.quicksum(
                    x[v, color]
                    for v in clusters[ci]
                    for color in range(min(max_colors,
                                           cluster_position[ci] + 1))
                ) == 1,
                name=f"choose_{ci}",
            )

        for ci in range(num_clusters):
            limit = min(max_colors, cluster_position[ci] + 1)
            for color in range(limit):
                model.addConstr(
                    gp.quicksum(x[v, color] for v in clusters[ci])
                    <= y[color]
                )

        model.addConstr(y[0] == 1)
        for color in range(max_colors - 1):
            model.addConstr(y[color] >= y[color + 1])

        for color in range(max_colors):
            usage = [
                x[v, color]
                for ci in range(num_clusters)
                if color < min(max_colors, cluster_position[ci] + 1)
                for v in clusters[ci]
            ]
            if usage:
                model.addConstr(y[color] <= gp.quicksum(usage))
            else:
                model.addConstr(y[color] == 0)

        conflict_count = len(relevant_edges) * max_colors
        use_lazy_conflicts = conflict_count > 1_500_000

        if not use_lazy_conflicts:
            for u, v in relevant_edges:
                common_limit = min(
                    max_colors,
                    cluster_position[vertex_cluster[u]] + 1,
                    cluster_position[vertex_cluster[v]] + 1,
                )
                for color in range(common_limit):
                    model.addConstr(x[u, color] + x[v, color] <= 1)
        else:
            model.Params.LazyConstraints = 1

        model.setObjective(
            gp.quicksum(y[color] for color in range(max_colors)),
            GRB.MINIMIZE,
        )

        # Supply the heuristic as a complete MIP start.
        for _, _, _, var in variable_meta:
            var.Start = 0.0
        for ci in range(num_clusters):
            v = best_selected[ci]
            color = best_colors[v]
            x[v, color].Start = 1.0
        for color in range(max_colors):
            y[color].Start = 1.0 if color < best_value else 0.0

        incumbent = {
            "value": best_value,
            "selected": dict(best_selected),
            "colors": dict(best_colors),
            "solution": best_solution,
        }
        vars_only = [item[3] for item in variable_meta]

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            try:
                values = cb_model.cbGetSolution(vars_only)
                chosen = {}
                chosen_colors = {}
                for (v, color, ci, _), value in zip(variable_meta, values):
                    if value > 0.5:
                        chosen[ci] = v
                        chosen_colors[v] = color

                if len(chosen) != num_clusters:
                    return

                if use_lazy_conflicts:
                    violated = False
                    for u, v in relevant_edges:
                        if (vertex_cluster[u] in chosen
                                and vertex_cluster[v] in chosen
                                and chosen[vertex_cluster[u]] == u
                                and chosen[vertex_cluster[v]] == v
                                and chosen_colors[u] == chosen_colors[v]):
                            color = chosen_colors[u]
                            cb_model.cbLazy(
                                x[u, color] + x[v, color] <= 1
                            )
                            violated = True
                    if violated:
                        return

                value = len(set(chosen_colors.values()))
                if value < incumbent["value"]:
                    canonical = canonicalize_colors(
                        chosen, chosen_colors, cluster_order
                    )
                    solution = make_solution(
                        chosen, canonical, num_clusters
                    )
                    incumbent["value"] = value
                    incumbent["selected"] = dict(chosen)
                    incumbent["colors"] = canonical
                    incumbent["solution"] = solution
                    if logger:
                        logger.log_solution(float(value), solution)
            except gp.GurobiError:
                return

        model.optimize(callback)

        # Recover the final incumbent directly in case no callback update was
        # recorded (for example, if presolve immediately solved the model).
        if model.SolCount > 0:
            chosen = {}
            chosen_colors = {}
            for v, color, ci, var in variable_meta:
                if var.X > 0.5:
                    chosen[ci] = v
                    chosen_colors[v] = color

            if len(chosen) == num_clusters:
                chosen_colors = canonicalize_colors(
                    chosen, chosen_colors, cluster_order
                )
                value = len(set(chosen_colors.values()))
                if value < incumbent["value"]:
                    solution = make_solution(
                        chosen, chosen_colors, num_clusters
                    )
                    incumbent["value"] = value
                    incumbent["selected"] = chosen
                    incumbent["colors"] = chosen_colors
                    incumbent["solution"] = solution
                    if logger:
                        logger.log_solution(float(value), solution)

        best_solution = incumbent["solution"]

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()