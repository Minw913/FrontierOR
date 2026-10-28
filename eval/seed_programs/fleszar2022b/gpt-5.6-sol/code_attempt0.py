import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def objective_value(assignment, values, pairwise_values, m):
    bags = [[] for _ in range(m)]
    total = 0
    for i, k in enumerate(assignment):
        if k >= 0:
            bags[k].append(i)
            total += values[i]

    for bag in bags:
        for p in range(len(bag)):
            i = bag[p]
            for r in range(p + 1, len(bag)):
                total += pairwise_values[i][bag[r]]
    return int(total)


def canonicalize(assignment, weights, m):
    """Relabel identical knapsacks by non-increasing load."""
    loads = [0] * m
    for i, k in enumerate(assignment):
        if k >= 0:
            loads[k] += weights[i]

    order = sorted(range(m), key=lambda k: (-loads[k], k))
    new_label = {old: new for new, old in enumerate(order)}
    return [new_label[k] if k >= 0 else -1 for k in assignment]


def make_solution(assignment, value):
    return {
        "objective_value": float(value),
        "assignment": {
            str(i): int(k)
            for i, k in enumerate(assignment)
            if k >= 0
        },
    }


def greedy_solution(
    n,
    m,
    capacity,
    weights,
    values,
    pairwise_values,
    deadline,
    seed_pair=None,
):
    assignment = [-1] * n
    loads = [0] * m
    gains = [[values[i] for _ in range(m)] for i in range(n)]

    def place(i, k):
        assignment[i] = k
        loads[k] += weights[i]
        for j in range(n):
            if assignment[j] < 0 and j != i:
                gains[j][k] += pairwise_values[j][i]

    if seed_pair is not None:
        i, j = seed_pair
        if i != j and weights[i] + weights[j] <= capacity:
            place(i, 0)
            place(j, 0)

    while time.monotonic() < deadline:
        best_gain = 0
        best_i = -1
        best_k = -1

        for i in range(n):
            if assignment[i] >= 0:
                continue
            wi = weights[i]
            for k in range(m):
                if loads[k] + wi <= capacity and gains[i][k] > best_gain:
                    best_gain = gains[i][k]
                    best_i = i
                    best_k = k

        if best_i < 0:
            break
        place(best_i, best_k)

    return assignment


def local_search(
    assignment,
    n,
    m,
    capacity,
    weights,
    values,
    pairwise_values,
    deadline,
    incumbent_callback,
):
    assignment = assignment[:]
    loads = [0] * m
    bags = [set() for _ in range(m)]

    for i, k in enumerate(assignment):
        if k >= 0:
            loads[k] += weights[i]
            bags[k].add(i)

    max_iterations = max(50, min(500, 4 * n))

    for _ in range(max_iterations):
        if time.monotonic() >= deadline:
            break

        best_delta = 0
        best_operation = None

        for i in range(n):
            if time.monotonic() >= deadline:
                break

            source = assignment[i]

            if source < 0:
                for target in range(m):
                    if loads[target] + weights[i] > capacity:
                        continue
                    delta = values[i]
                    for j in bags[target]:
                        delta += pairwise_values[i][j]
                    if delta > best_delta:
                        best_delta = delta
                        best_operation = ("insert", i, target)
            else:
                source_interaction = 0
                for j in bags[source]:
                    if j != i:
                        source_interaction += pairwise_values[i][j]

                removal_delta = -(values[i] + source_interaction)
                if removal_delta > best_delta:
                    best_delta = removal_delta
                    best_operation = ("remove", i, source)

                for target in range(m):
                    if target == source:
                        continue
                    if loads[target] + weights[i] > capacity:
                        continue

                    target_interaction = 0
                    for j in bags[target]:
                        target_interaction += pairwise_values[i][j]

                    delta = target_interaction - source_interaction
                    if delta > best_delta:
                        best_delta = delta
                        best_operation = ("move", i, source, target)

        if best_operation is None or best_delta <= 0:
            break

        kind = best_operation[0]
        i = best_operation[1]

        if kind == "insert":
            target = best_operation[2]
            assignment[i] = target
            bags[target].add(i)
            loads[target] += weights[i]

        elif kind == "remove":
            source = best_operation[2]
            bags[source].remove(i)
            loads[source] -= weights[i]
            assignment[i] = -1

        else:
            source = best_operation[2]
            target = best_operation[3]
            bags[source].remove(i)
            loads[source] -= weights[i]
            bags[target].add(i)
            loads[target] += weights[i]
            assignment[i] = target

        incumbent_callback(assignment)

    return assignment


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

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    n = int(instance["n"])
    m = int(instance["m"])
    capacity = int(instance["capacity"])
    weights = [int(x) for x in instance["weights"]]
    values = [int(x) for x in instance["values"]]
    pairwise_values = [
        [int(x) for x in row] for row in instance["pairwise_values"]
    ]

    best_assignment = [-1] * n
    best_value = 0

    def consider(candidate):
        nonlocal best_assignment, best_value
        candidate = canonicalize(candidate, weights, m)
        value = objective_value(candidate, values, pairwise_values, m)
        if value > best_value:
            best_value = value
            best_assignment = candidate[:]
            solution = make_solution(best_assignment, best_value)
            if logger:
                logger.log_solution(float(best_value), solution)
            return True
        return False

    # The empty assignment is always feasible.
    if logger:
        logger.log_solution(0.0, make_solution(best_assignment, best_value))

    # Reserve most of the time for Gurobi while constructing useful starts.
    total_limit = max(0.0, float(args.time_limit))
    heuristic_budget = min(3.0, max(0.05, 0.10 * total_limit))
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)

    if time.monotonic() < heuristic_deadline:
        candidate = greedy_solution(
            n,
            m,
            capacity,
            weights,
            values,
            pairwise_values,
            heuristic_deadline,
        )
        consider(candidate)

    # A seeded pair can discover profitable combinations whose members are
    # individually unattractive.
    best_pair = None
    best_pair_score = 0
    if time.monotonic() < heuristic_deadline:
        for i in range(n):
            if time.monotonic() >= heuristic_deadline:
                break
            for j in range(i + 1, n):
                if weights[i] + weights[j] <= capacity:
                    score = values[i] + values[j] + pairwise_values[i][j]
                    if score > best_pair_score:
                        best_pair_score = score
                        best_pair = (i, j)

    if best_pair is not None and time.monotonic() < heuristic_deadline:
        candidate = greedy_solution(
            n,
            m,
            capacity,
            weights,
            values,
            pairwise_values,
            heuristic_deadline,
            seed_pair=best_pair,
        )
        consider(candidate)

    if time.monotonic() < heuristic_deadline:
        local_search(
            best_assignment,
            n,
            m,
            capacity,
            weights,
            values,
            pairwise_values,
            heuristic_deadline,
            consider,
        )

    # Build and solve the exact binary quadratic formulation when time remains.
    if time.monotonic() + 0.05 < deadline:
        try:
            model = gp.Model("quadratic_multiple_knapsack")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.NonConvex = 2

            x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

            for i in range(n):
                model.addConstr(
                    gp.quicksum(x[i, k] for k in range(m)) <= 1
                )

            loads = []
            for k in range(m):
                load = gp.quicksum(weights[i] * x[i, k] for i in range(n))
                model.addConstr(load <= capacity)
                loads.append(load)

            # Valid symmetry breaking for identical knapsacks.
            for k in range(m - 1):
                model.addConstr(loads[k] >= loads[k + 1])

            obj = gp.QuadExpr()
            for i in range(n):
                if values[i] != 0:
                    for k in range(m):
                        obj.add(values[i] * x[i, k])

            objective_complete = True
            for i in range(n):
                if time.monotonic() + 0.03 >= deadline:
                    objective_complete = False
                    break
                for j in range(i + 1, n):
                    q = pairwise_values[i][j]
                    if q != 0:
                        for k in range(m):
                            obj.add(q, x[i, k], x[j, k])

            if objective_complete:
                model.setObjective(obj, GRB.MAXIMIZE)

                # Supply the best heuristic incumbent as a complete MIP start.
                for i in range(n):
                    assigned_k = best_assignment[i]
                    for k in range(m):
                        x[i, k].Start = 1.0 if assigned_k == k else 0.0

                flat_x = [x[i, k] for i in range(n) for k in range(m)]

                def callback(cb_model, where):
                    if where != GRB.Callback.MIPSOL:
                        return
                    try:
                        vals = cb_model.cbGetSolution(flat_x)
                        candidate_assignment = [-1] * n
                        pos = 0
                        for i in range(n):
                            selected = -1
                            selected_value = 0.5
                            for k in range(m):
                                val = vals[pos]
                                pos += 1
                                if val > selected_value:
                                    selected_value = val
                                    selected = k
                            candidate_assignment[i] = selected
                        consider(candidate_assignment)
                    except Exception:
                        # Logging must not interrupt optimization.
                        pass

                remaining = deadline - time.monotonic() - 0.02
                if remaining > 0:
                    model.Params.TimeLimit = max(0.001, remaining)
                    model.optimize(callback)

                    if model.SolCount > 0:
                        candidate_assignment = [-1] * n
                        for i in range(n):
                            for k in range(m):
                                if x[i, k].X > 0.5:
                                    candidate_assignment[i] = k
                                    break
                        consider(candidate_assignment)

        except gp.GurobiError:
            # Return the best heuristic solution if model construction or
            # optimization cannot proceed.
            pass

    final_solution = make_solution(best_assignment, best_value)
    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()