import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def make_solution(profits, selected_items):
    selected_items = sorted(selected_items)
    objective = sum(profits[i] for i in selected_items)
    return {
        "objective_value": float(objective),
        "selected_items": selected_items,
    }


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(solution, output_file)
    os.replace(temporary_path, path)


def greedy_solution(order, profits, weights, adjacency, forbidden, capacity):
    n = len(profits)
    blocked = [False] * n
    selected = []
    total_weight = 0

    for i in order:
        if forbidden[i] or blocked[i]:
            continue
        if total_weight + weights[i] > capacity:
            continue

        selected.append(i)
        total_weight += weights[i]
        blocked[i] = True
        for neighbor in adjacency[i]:
            blocked[neighbor] = True

    return selected


def improve_solution(
    initial_selected,
    order,
    profits,
    weights,
    adjacency,
    forbidden,
    capacity,
    deadline,
):
    n = len(profits)
    selected = [False] * n
    total_weight = 0

    for i in initial_selected:
        selected[i] = True
        total_weight += weights[i]

    # Profitable one-for-many replacements. Removing all currently selected
    # conflicts makes adding the candidate item feasible with respect to the
    # conflict graph.
    for position, i in enumerate(order):
        if position % 1024 == 0 and time.monotonic() >= deadline:
            break
        if selected[i] or forbidden[i]:
            continue

        blockers = [j for j in adjacency[i] if selected[j]]
        removed_profit = sum(profits[j] for j in blockers)
        removed_weight = sum(weights[j] for j in blockers)

        if profits[i] <= removed_profit:
            continue
        if total_weight - removed_weight + weights[i] > capacity:
            continue

        for j in blockers:
            selected[j] = False
        selected[i] = True
        total_weight = total_weight - removed_weight + weights[i]

    # Refill any capacity and conflict-free space exposed by replacements.
    for position, i in enumerate(order):
        if position % 1024 == 0 and time.monotonic() >= deadline:
            break
        if selected[i] or forbidden[i]:
            continue
        if total_weight + weights[i] > capacity:
            continue
        if any(selected[j] for j in adjacency[i]):
            continue

        selected[i] = True
        total_weight += weights[i]

    return [i for i in range(n) if selected[i]]


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
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path
        else None
    )

    with open(args.instance_path, "r", encoding="utf-8") as instance_file:
        instance = json.load(instance_file)

    n = int(instance["num_items"])
    capacity = int(instance["capacity"])
    profits = [int(v) for v in instance["items"]["profits"]]
    weights = [int(v) for v in instance["items"]["weights"]]
    raw_edges = instance["conflict_graph"]["edges"]

    if len(profits) != n or len(weights) != n:
        raise ValueError("Profit and weight arrays must have num_items entries")

    best_solution = make_solution(profits, [])
    best_objective = 0

    if logger:
        logger.log_solution(best_objective, best_solution)

    # Build a deduplicated graph for heuristics and the MIP.
    adjacency = [set() for _ in range(n)]
    forbidden = [False] * n
    edges = []

    for edge in raw_edges:
        u, v = int(edge[0]), int(edge[1])
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Invalid conflict edge ({u}, {v})")

        if u == v:
            if not forbidden[u]:
                forbidden[u] = True
                edges.append((u, u))
            continue

        if u > v:
            u, v = v, u

        if v not in adjacency[u]:
            adjacency[u].add(v)
            adjacency[v].add(u)
            edges.append((u, v))

    def consider(selected_items):
        nonlocal best_solution, best_objective
        candidate = make_solution(profits, selected_items)
        candidate_objective = int(candidate["objective_value"])
        if candidate_objective > best_objective:
            best_objective = candidate_objective
            best_solution = candidate
            if logger:
                logger.log_solution(candidate_objective, candidate)

    # A good feasible incumbent helps both short runs and MIP performance.
    if time.monotonic() < deadline and n > 0:
        degrees = [len(adjacency[i]) for i in range(n)]

        ordering_keys = [
            lambda i: (
                profits[i] / weights[i],
                profits[i],
                -weights[i],
            ),
            lambda i: (
                profits[i],
                profits[i] / weights[i],
                -weights[i],
            ),
            lambda i: (
                profits[i] / (weights[i] * ((degrees[i] + 1) ** 0.30)),
                profits[i] / weights[i],
                profits[i],
            ),
            lambda i: (
                profits[i] / (degrees[i] + 1),
                profits[i] / weights[i],
                profits[i],
            ),
        ]

        ratio_order = None
        for key_function in ordering_keys:
            if time.monotonic() >= deadline:
                break
            order = sorted(range(n), key=key_function, reverse=True)
            if ratio_order is None:
                ratio_order = order
            consider(
                greedy_solution(
                    order,
                    profits,
                    weights,
                    adjacency,
                    forbidden,
                    capacity,
                )
            )

        if ratio_order is not None and time.monotonic() < deadline:
            improved = improve_solution(
                best_solution["selected_items"],
                ratio_order,
                profits,
                weights,
                adjacency,
                forbidden,
                capacity,
                deadline,
            )
            consider(improved)

    # If preprocessing and heuristics consumed the budget, return immediately.
    if time.monotonic() >= deadline:
        write_solution(args.solution_path, best_solution)
        return

    model = None
    try:
        model = gp.Model("knapsack_with_conflicts")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        model.setObjective(
            gp.quicksum(profits[i] * x[i] for i in range(n)),
            GRB.MAXIMIZE,
        )
        model.addConstr(
            gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity,
            name="capacity",
        )

        completed_model = True
        for edge_index, (u, v) in enumerate(edges):
            if edge_index % 4096 == 0 and time.monotonic() >= deadline:
                completed_model = False
                break
            if u == v:
                model.addLConstr(x[u] <= 0, name=f"self_conflict_{u}")
            else:
                model.addLConstr(
                    x[u] + x[v] <= 1,
                    name=f"conflict_{edge_index}",
                )

        if completed_model and time.monotonic() < deadline:
            selected_set = set(best_solution["selected_items"])
            for i in range(n):
                x[i].Start = 1.0 if i in selected_set else 0.0

            remaining_time = deadline - time.monotonic()
            if remaining_time > 0:
                model.Params.TimeLimit = max(0.001, remaining_time)

                variables = [x[i] for i in range(n)]

                def incumbent_callback(callback_model, where):
                    nonlocal best_solution, best_objective
                    if where != GRB.Callback.MIPSOL:
                        return

                    values = callback_model.cbGetSolution(variables)
                    selected = [
                        i for i, value in enumerate(values) if value > 0.5
                    ]
                    objective = sum(profits[i] for i in selected)

                    if objective > best_objective:
                        solution = {
                            "objective_value": float(objective),
                            "selected_items": selected,
                        }
                        best_objective = objective
                        best_solution = solution
                        if logger:
                            logger.log_solution(objective, solution)

                model.optimize(incumbent_callback)

                # Defensive final extraction in case the final incumbent was
                # available without producing a new callback-visible value.
                if model.SolCount > 0:
                    selected = [
                        i for i in range(n) if x[i].X > 0.5
                    ]
                    consider(selected)

    except gp.GurobiError:
        # The heuristic incumbent remains a valid solution if model creation
        # or optimization fails unexpectedly.
        pass
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()