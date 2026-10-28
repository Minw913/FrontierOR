import argparse
import heapq
import json
import os
import time
from collections import deque

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def make_solution(path, arcs, objective_value):
    return {
        "objective_value": float(objective_value),
        "solution_arcs": [
            {
                "id": int(arcs[i]["id"]),
                "from_node": int(arcs[i]["from_node"]),
                "to_node": int(arcs[i]["to_node"]),
            }
            for i in path
        ],
    }


def evaluate_path(path, linear_costs, quadratic_costs):
    value = sum(linear_costs[i] for i in path)
    for i in path:
        row = quadratic_costs[i]
        for j in path:
            value += float(row[j])
    return float(value)


def shortest_path(
    num_nodes,
    source,
    target,
    outgoing,
    arc_from,
    weights,
):
    """
    Find a feasible source-target path with Dijkstra.

    A uniform shift is applied if some heuristic arc weights are negative.
    This changes the heuristic ranking between paths of different lengths,
    but guarantees nonnegative weights and still produces a feasible path.
    """
    if source == target:
        return []

    if not weights:
        return None

    minimum_weight = min(weights)
    shift = -minimum_weight if minimum_weight < 0.0 else 0.0

    distances = [float("inf")] * num_nodes
    predecessor_arc = [-1] * num_nodes
    settled = [False] * num_nodes

    distances[source] = 0.0
    queue = [(0.0, source)]

    while queue:
        current_distance, node = heapq.heappop(queue)
        if settled[node]:
            continue

        settled[node] = True
        if node == target:
            break

        for arc_index, next_node in outgoing[node]:
            candidate = (
                current_distance
                + float(weights[arc_index])
                + shift
            )
            if candidate < distances[next_node] - 1e-12:
                distances[next_node] = candidate
                predecessor_arc[next_node] = arc_index
                heapq.heappush(queue, (candidate, next_node))

    if predecessor_arc[target] < 0:
        return None

    reversed_path = []
    node = target
    seen_nodes = set()

    while node != source:
        if node in seen_nodes:
            return None
        seen_nodes.add(node)

        arc_index = predecessor_arc[node]
        if arc_index < 0:
            return None

        reversed_path.append(arc_index)
        node = arc_from[arc_index]

    reversed_path.reverse()
    return reversed_path


def graph_is_acyclic(num_nodes, arcs):
    indegree = [0] * num_nodes
    adjacency = [[] for _ in range(num_nodes)]

    for arc in arcs:
        u = int(arc["from_node"])
        v = int(arc["to_node"])
        adjacency[u].append(v)
        indegree[v] += 1

    queue = deque(node for node in range(num_nodes) if indegree[node] == 0)
    visited = 0

    while queue:
        u = queue.popleft()
        visited += 1

        for v in adjacency[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    return visited == num_nodes


def write_solution(path, arcs, objective, solution_path):
    solution = make_solution(path, arcs, objective)

    parent = os.path.dirname(os.path.abspath(solution_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(solution_path, "w", encoding="utf-8") as output_file:
        json.dump(solution, output_file, indent=2)

    return solution


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

    with open(args.instance_path, "r", encoding="utf-8") as input_file:
        instance = json.load(input_file)

    num_nodes = int(instance["num_nodes"])
    arcs = instance["arcs"]
    num_arcs = len(arcs)
    source = int(instance["source_node"])
    target = int(instance["target_node"])
    linear_costs = [float(value) for value in instance["linear_costs"]]
    quadratic_costs = instance["quadratic_costs"]

    outgoing = [[] for _ in range(num_nodes)]
    incoming = [[] for _ in range(num_nodes)]
    arc_from = [0] * num_arcs
    arc_to = [0] * num_arcs

    for i, arc in enumerate(arcs):
        u = int(arc["from_node"])
        v = int(arc["to_node"])

        arc_from[i] = u
        arc_to[i] = v
        outgoing[u].append((i, v))
        incoming[v].append((i, u))

    best_path = None
    best_objective = float("inf")
    best_solution = None

    def consider(path):
        nonlocal best_path, best_objective, best_solution

        if path is None:
            return False

        objective = evaluate_path(
            path,
            linear_costs,
            quadratic_costs,
        )

        if objective < best_objective - 1e-8:
            best_path = list(path)
            best_objective = objective
            best_solution = make_solution(
                best_path,
                arcs,
                best_objective,
            )

            if logger:
                logger.log_solution(best_objective, best_solution)

            return True

        return False

    # First obtain a minimum-arc-count feasible path.
    consider(
        shortest_path(
            num_nodes,
            source,
            target,
            outgoing,
            arc_from,
            [1.0] * num_arcs,
        )
    )

    # Then use the linear costs and diagonal quadratic terms.
    additive_costs = [
        linear_costs[i] + float(quadratic_costs[i][i])
        for i in range(num_arcs)
    ]
    consider(
        shortest_path(
            num_nodes,
            source,
            target,
            outgoing,
            arc_from,
            additive_costs,
        )
    )

    # Iteratively construct shortest paths using marginal costs relative to
    # the current incumbent.
    for _ in range(8):
        if best_path is None or time.monotonic() >= deadline:
            break

        selected = best_path
        marginal_costs = [0.0] * num_arcs

        for i in range(num_arcs):
            value = linear_costs[i] + float(quadratic_costs[i][i])
            row_i = quadratic_costs[i]

            for j in selected:
                if j != i:
                    value += (
                        float(row_i[j])
                        + float(quadratic_costs[j][i])
                    )

            marginal_costs[i] = value

        candidate = shortest_path(
            num_nodes,
            source,
            target,
            outgoing,
            arc_from,
            marginal_costs,
        )

        previous_path = tuple(best_path)
        improved = consider(candidate)

        if not improved or tuple(best_path) == previous_path:
            break

    if best_path is None:
        raise RuntimeError(
            "The instance contains no directed source-target path."
        )

    if time.monotonic() >= deadline:
        write_solution(
            best_path,
            arcs,
            best_objective,
            args.solution_path,
        )
        return

    model = None

    try:
        model = gp.Model("quadratic_shortest_path")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.NonConvex = 2

        x = model.addVars(
            num_arcs,
            vtype=GRB.BINARY,
            name="x",
        )
        x_list = [x[i] for i in range(num_arcs)]

        for node in range(num_nodes):
            incoming_expression = gp.quicksum(
                x[i] for i, _ in incoming[node]
            )
            outgoing_expression = gp.quicksum(
                x[i] for i, _ in outgoing[node]
            )

            if node == source:
                model.addConstr(
                    outgoing_expression == 1,
                    name=f"source_out_{node}",
                )
                model.addConstr(
                    incoming_expression == 0,
                    name=f"source_in_{node}",
                )
            elif node == target:
                model.addConstr(
                    incoming_expression == 1,
                    name=f"target_in_{node}",
                )
                model.addConstr(
                    outgoing_expression == 0,
                    name=f"target_out_{node}",
                )
            else:
                model.addConstr(
                    incoming_expression == outgoing_expression,
                    name=f"flow_{node}",
                )
                model.addConstr(
                    incoming_expression <= 1,
                    name=f"in_degree_{node}",
                )
                model.addConstr(
                    outgoing_expression <= 1,
                    name=f"out_degree_{node}",
                )

        # Exclude cycles when the input graph itself is not acyclic.
        order = None
        if not graph_is_acyclic(num_nodes, arcs):
            order = model.addVars(
                num_nodes,
                lb=0.0,
                ub=float(max(0, num_nodes - 1)),
                vtype=GRB.CONTINUOUS,
                name="order",
            )

            model.addConstr(
                order[source] == 0.0,
                name="source_order",
            )

            big_m = float(max(1, num_nodes))
            for i in range(num_arcs):
                u = arc_from[i]
                v = arc_to[i]

                model.addConstr(
                    order[v]
                    >= order[u] + 1.0 - big_m * (1.0 - x[i]),
                    name=f"acyclic_{i}",
                )

        # For binary x_i, the diagonal term Q_ii*x_i*x_i equals Q_ii*x_i.
        linear_coefficients = [
            linear_costs[i] + float(quadratic_costs[i][i])
            for i in range(num_arcs)
        ]

        linear_expression = gp.LinExpr(linear_coefficients, x_list)
        objective = gp.QuadExpr(linear_expression)

        for i in range(num_arcs):
            coefficients = []
            first_variables = []
            second_variables = []
            row_i = quadratic_costs[i]

            for j in range(i + 1, num_arcs):
                coefficient = (
                    float(row_i[j])
                    + float(quadratic_costs[j][i])
                )

                if coefficient != 0.0:
                    coefficients.append(coefficient)
                    first_variables.append(x[i])
                    second_variables.append(x[j])

            if coefficients:
                objective.addTerms(
                    coefficients,
                    first_variables,
                    second_variables,
                )

            if (i & 31) == 0 and time.monotonic() >= deadline:
                write_solution(
                    best_path,
                    arcs,
                    best_objective,
                    args.solution_path,
                )
                model.dispose()
                model = None
                return

        model.setObjective(objective, GRB.MINIMIZE)

        selected_set = set(best_path)
        for i in range(num_arcs):
            x[i].Start = 1.0 if i in selected_set else 0.0

        if order is not None:
            node_positions = {source: 0}
            position = 0
            current_node = source

            for arc_index in best_path:
                if arc_from[arc_index] != current_node:
                    break

                position += 1
                current_node = arc_to[arc_index]
                node_positions[current_node] = position

            for node in range(num_nodes):
                order[node].Start = float(node_positions.get(node, 0))

        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0.0:
            write_solution(
                best_path,
                arcs,
                best_objective,
                args.solution_path,
            )
            model.dispose()
            model = None
            return

        model.Params.TimeLimit = remaining_time

        callback_state = {
            "best_objective": best_objective,
            "best_path": list(best_path),
            "best_solution": best_solution,
        }

        def incumbent_callback(callback_model, where):
            if time.monotonic() >= deadline:
                callback_model.terminate()
                return

            if where != GRB.Callback.MIPSOL:
                return

            try:
                values = callback_model.cbGetSolution(x_list)
                candidate_path = [
                    i for i, value in enumerate(values)
                    if value > 0.5
                ]

                objective_value = evaluate_path(
                    candidate_path,
                    linear_costs,
                    quadratic_costs,
                )

                if (
                    objective_value
                    < callback_state["best_objective"] - 1e-8
                ):
                    solution = make_solution(
                        candidate_path,
                        arcs,
                        objective_value,
                    )

                    callback_state["best_objective"] = objective_value
                    callback_state["best_path"] = candidate_path
                    callback_state["best_solution"] = solution

                    if logger:
                        logger.log_solution(objective_value, solution)

            except Exception:
                # Logging or incumbent extraction must not interrupt search.
                pass

        model.optimize(incumbent_callback)

        if model.SolCount > 0:
            model_path = [
                i for i in range(num_arcs)
                if x[i].X > 0.5
            ]

            model_objective = evaluate_path(
                model_path,
                linear_costs,
                quadratic_costs,
            )

            if (
                model_objective
                < callback_state["best_objective"] - 1e-8
            ):
                model_solution = make_solution(
                    model_path,
                    arcs,
                    model_objective,
                )

                callback_state["best_objective"] = model_objective
                callback_state["best_path"] = model_path
                callback_state["best_solution"] = model_solution

                if logger:
                    logger.log_solution(
                        model_objective,
                        model_solution,
                    )

        best_path = callback_state["best_path"]
        best_objective = callback_state["best_objective"]

    except gp.GurobiError:
        # Return the best heuristic incumbent if Gurobi cannot finish.
        pass
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass

    write_solution(
        best_path,
        arcs,
        best_objective,
        args.solution_path,
    )


if __name__ == "__main__":
    main()