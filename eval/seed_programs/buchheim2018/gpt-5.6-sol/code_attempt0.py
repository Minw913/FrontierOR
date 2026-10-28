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
    value = 0
    for i in path:
        value += linear_costs[i]
    for i in path:
        row = quadratic_costs[i]
        for j in path:
            value += row[j]
    return float(value)


def shortest_path(num_nodes, source, target, outgoing, weights):
    """Dijkstra after a uniform shift that makes all arc weights nonnegative."""
    if source == target:
        return []

    minimum = min(weights) if weights else 0.0
    shift = -minimum if minimum < 0.0 else 0.0

    dist = [float("inf")] * num_nodes
    predecessor_arc = [-1] * num_nodes
    settled = [False] * num_nodes
    dist[source] = 0.0
    queue = [(0.0, source)]

    while queue:
        current_dist, node = heapq.heappop(queue)
        if settled[node]:
            continue
        settled[node] = True

        if node == target:
            break

        for arc_index, next_node in outgoing[node]:
            candidate = current_dist + float(weights[arc_index]) + shift
            if candidate < dist[next_node] - 1e-12:
                dist[next_node] = candidate
                predecessor_arc[next_node] = arc_index
                heapq.heappush(queue, (candidate, next_node))

    if predecessor_arc[target] < 0:
        return None

    reversed_path = []
    node = target
    seen = set()
    while node != source:
        if node in seen:
            return None
        seen.add(node)

        arc_index = predecessor_arc[node]
        if arc_index < 0:
            return None
        reversed_path.append(arc_index)
        node = outgoing.arc_from[arc_index]

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

    queue = deque(v for v in range(num_nodes) if indegree[v] == 0)
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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as input_file:
        instance = json.load(input_file)

    num_nodes = int(instance["num_nodes"])
    arcs = instance["arcs"]
    num_arcs = len(arcs)
    source = int(instance["source_node"])
    target = int(instance["target_node"])
    linear_costs = [float(v) for v in instance["linear_costs"]]
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

    # Attach the origin array for inexpensive path reconstruction.
    outgoing.arc_from = arc_from if hasattr(outgoing, "__dict__") else None
    # Built-in lists cannot carry attributes, so use a small wrapper.
    class OutgoingList(list):
        pass

    wrapped_outgoing = OutgoingList(outgoing)
    wrapped_outgoing.arc_from = arc_from
    outgoing = wrapped_outgoing

    best_path = None
    best_objective = float("inf")
    best_solution = None

    def consider(path):
        nonlocal best_path, best_objective, best_solution
        if path is None:
            return False

        objective = evaluate_path(path, linear_costs, quadratic_costs)
        if objective < best_objective - 1e-8:
            best_path = list(path)
            best_objective = objective
            best_solution = make_solution(best_path, arcs, best_objective)
            if logger:
                logger.log_solution(best_objective, best_solution)
            return True
        return False

    # Quickly establish feasible incumbents.
    consider(shortest_path(
        num_nodes, source, target, outgoing, [1.0] * num_arcs
    ))

    additive_costs = [
        linear_costs[i] + float(quadratic_costs[i][i])
        for i in range(num_arcs)
    ]
    consider(shortest_path(
        num_nodes, source, target, outgoing, additive_costs
    ))

    # Fixed-point marginal-cost shortest paths often improve the initial path.
    for _ in range(8):
        if best_path is None or time.monotonic() >= deadline:
            break

        selected = best_path
        marginal = [0.0] * num_arcs
        for i in range(num_arcs):
            value = linear_costs[i] + float(quadratic_costs[i][i])
            row_i = quadratic_costs[i]
            for j in selected:
                if j != i:
                    value += float(row_i[j]) + float(quadratic_costs[j][i])
            marginal[i] = value

        candidate = shortest_path(
            num_nodes, source, target, outgoing, marginal
        )
        old_path = tuple(best_path)
        improved = consider(candidate)
        if not improved or tuple(best_path) == old_path:
            break

    if best_path is None:
        raise RuntimeError("The instance contains no directed source-target path.")

    if time.monotonic() >= deadline:
        write_solution(best_path, arcs, best_objective, args.solution_path)
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

        x = model.addVars(num_arcs, vtype=GRB.BINARY, name="x")
        x_list = [x[i] for i in range(num_arcs)]

        for node in range(num_nodes):
            in_expr = gp.quicksum(x[i] for i, _ in incoming[node])
            out_expr = gp.quicksum(x[i] for i, _ in outgoing[node])

            if node == source:
                model.addConstr(out_expr == 1, name=f"source_out_{node}")
                model.addConstr(in_expr == 0, name=f"source_in_{node}")
            elif node == target:
                model.addConstr(in_expr == 1, name=f"target_in_{node}")
                model.addConstr(out_expr == 0, name=f"target_out_{node}")
            else:
                model.addConstr(in_expr == out_expr, name=f"flow_{node}")
                model.addConstr(in_expr <= 1, name=f"in_degree_{node}")
                model.addConstr(out_expr <= 1, name=f"out_degree_{node}")

        # A cyclic network can otherwise contain disconnected selected cycles.
        order = None
        if not graph_is_acyclic(num_nodes, arcs):
            order = model.addVars(
                num_nodes,
                lb=0.0,
                ub=float(max(0, num_nodes - 1)),
                vtype=GRB.CONTINUOUS,
                name="order",
            )
            model.addConstr(order[source] == 0.0, name="source_order")

            big_m = float(max(1, num_nodes))
            for i in range(num_arcs):
                u = arc_from[i]
                v = arc_to[i]
                model.addConstr(
                    order[v] >= order[u] + 1.0 - big_m * (1.0 - x[i]),
                    name=f"acyclic_{i}",
                )

        # Since x_i is binary, Q_ii*x_i^2 is included in the linear part.
        linear_coefficients = [
            linear_costs[i] + float(quadratic_costs[i][i])
            for i in range(num_arcs)
        ]
        objective = gp.QuadExpr(
            gp.LinExpr(linear_coefficients, x_list)
        )

        for i in range(num_arcs):
            coefficients = []
            first_vars = []
            second_vars = []
            row_i = quadratic_costs[i]

            for j in range(i + 1, num_arcs):
                coefficient = float(row_i[j]) + float(quadratic_costs[j][i])
                if coefficient != 0.0:
                    coefficients.append(coefficient)
                    first_vars.append(x[i])
                    second_vars.append(x[j])

            if coefficients:
                objective.addTerms(coefficients, first_vars, second_vars)

            if (i & 31) == 0 and time.monotonic() >= deadline:
                write_solution(best_path, arcs, best_objective, args.solution_path)
                model.dispose()
                return

        model.setObjective(objective, GRB.MINIMIZE)

        selected_set = set(best_path)
        for i in range(num_arcs):
            x[i].Start = 1.0 if i in selected_set else 0.0

        if order is not None:
            node_positions = {source: 0}
            position = 0
            current = source
            for arc_index in best_path:
                if arc_from[arc_index] != current:
                    break
                position += 1
                current = arc_to[arc_index]
                node_positions[current] = position

            for node in range(num_nodes):
                order[node].Start = float(node_positions.get(node, 0))

        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            write_solution(best_path, arcs, best_objective, args.solution_path)
            model.dispose()
            return

        model.Params.TimeLimit = remaining

        callback_state = {
            "best_objective": best_objective,
            "best_path": list(best_path),
            "best_solution": best_solution,
        }

        def incumbent_callback(cb_model, where):
            if time.monotonic() >= deadline:
                cb_model.terminate()
                return

            if where != GRB.Callback.MIPSOL:
                return

            try:
                values = cb_model.cbGetSolution(x_list)
                path = [i for i, value in enumerate(values) if value > 0.5]
                objective_value = evaluate_path(
                    path, linear_costs, quadratic_costs
                )

                if objective_value < callback_state["best_objective"] - 1e-8:
                    solution = make_solution(path, arcs, objective_value)
                    callback_state["best_objective"] = objective_value
                    callback_state["best_path"] = path
                    callback_state["best_solution"] = solution
                    if logger:
                        logger.log_solution(objective_value, solution)
            except Exception:
                # Logging must not interrupt the optimization search.
                pass

        model.optimize(incumbent_callback)

        if model.SolCount > 0:
            model_path = [
                i for i in range(num_arcs) if x[i].X > 0.5
            ]
            model_objective = evaluate_path(
                model_path, linear_costs, quadratic_costs
            )
            if model_objective < callback_state["best_objective"] - 1e-8:
                model_solution = make_solution(
                    model_path, arcs, model_objective
                )
                callback_state["best_objective"] = model_objective
                callback_state["best_path"] = model_path
                callback_state["best_solution"] = model_solution
                if logger:
                    logger.log_solution(model_objective, model_solution)

        best_path = callback_state["best_path"]
        best_objective = callback_state["best_objective"]

    except gp.GurobiError:
        # Preserve the best heuristic incumbent if model construction or
        # optimization cannot be completed.
        pass
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass

    write_solution(best_path, arcs, best_objective, args.solution_path)


if __name__ == "__main__":
    main()