import argparse
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


def read_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["N"])
    m = int(data["m"])
    capacities = data["group_capacities"]
    lower = [int(capacities[g]["L_g"]) for g in range(m)]
    upper = [int(capacities[g]["U_g"]) for g in range(m)]

    if len(lower) != m or len(upper) != m:
        raise ValueError("group_capacities must contain exactly m entries")
    if any(lower[g] < 0 or upper[g] < lower[g] for g in range(m)):
        raise ValueError("Invalid group capacity bounds")
    if sum(lower) > n or sum(upper) < n:
        raise ValueError("The capacity bounds make the instance infeasible")

    values = data["distances_upper_triangular"]
    expected = n * (n - 1) // 2
    if len(values) != expected:
        raise ValueError(
            f"Expected {expected} upper-triangular distances, got {len(values)}"
        )

    distance = [[0.0] * n for _ in range(n)]
    k = 0
    for i in range(n - 1):
        row_i = distance[i]
        for j in range(i + 1, n):
            value = float(values[k])
            k += 1
            row_i[j] = value
            distance[j][i] = value

    return n, m, lower, upper, distance


def objective_value(assignment, distance):
    n = len(assignment)
    total = 0.0
    for i in range(n - 1):
        gi = assignment[i]
        row = distance[i]
        for j in range(i + 1, n):
            if gi == assignment[j]:
                total += row[j]
    return total


def make_solution(assignment, objective, m):
    groups = [[] for _ in range(m)]
    for element, group in enumerate(assignment):
        groups[group].append(element)

    return {
        "objective_value": float(objective),
        "assignment": {
            str(element): int(group)
            for element, group in enumerate(assignment)
        },
        "groups": {
            str(group): groups[group]
            for group in range(m)
        },
    }


def concentrated_sizes(n, m, lower, upper):
    sizes = lower[:]
    remaining = n - sum(sizes)

    while remaining > 0:
        candidates = [g for g in range(m) if sizes[g] < upper[g]]
        if not candidates:
            raise ValueError("Unable to construct feasible group sizes")
        g = max(candidates, key=lambda x: (sizes[x], upper[x], -x))
        sizes[g] += 1
        remaining -= 1

    return sizes


def randomized_sizes(n, m, lower, upper, rng, mode):
    sizes = lower[:]
    remaining = n - sum(sizes)

    while remaining > 0:
        candidates = [g for g in range(m) if sizes[g] < upper[g]]
        if not candidates:
            raise ValueError("Unable to construct feasible group sizes")

        if mode == 0:
            g = max(
                candidates,
                key=lambda x: (
                    sizes[x] + rng.random() * 0.35,
                    upper[x] - sizes[x],
                ),
            )
        elif mode == 1:
            g = min(
                candidates,
                key=lambda x: (
                    sizes[x] / max(1, upper[x]),
                    rng.random(),
                ),
            )
        else:
            weights = [
                max(1.0, float(sizes[g] + 1)) *
                max(1.0, float(upper[g] - sizes[g]))
                for g in candidates
            ]
            g = rng.choices(candidates, weights=weights, k=1)[0]

        sizes[g] += 1
        remaining -= 1

    return sizes


def sequential_assignment(sizes):
    assignment = []
    for g, size in enumerate(sizes):
        assignment.extend([g] * size)
    return assignment


def greedy_assignment(n, m, sizes, distance, rng, trial):
    if n == 0:
        return []

    if trial == 0:
        order = list(range(n))
        degree = [sum(distance[i]) for i in range(n)]
        order.sort(key=lambda i: (-degree[i], i))
    else:
        order = list(range(n))
        rng.shuffle(order)

    assignment = [-1] * n
    remaining_slots = sizes[:]
    affinity = [[0.0] * m for _ in range(n)]

    positive_groups = [g for g in range(m) if remaining_slots[g] > 0]

    # Give each nonempty group one seed so that later affinity scores are useful.
    cursor = 0
    for g in positive_groups:
        if cursor >= n:
            break
        element = order[cursor]
        cursor += 1
        assignment[element] = g
        remaining_slots[g] -= 1
        for i in range(n):
            affinity[i][g] += distance[i][element]

    remaining_elements = order[cursor:]

    for element in remaining_elements:
        candidates = [g for g in range(m) if remaining_slots[g] > 0]
        if trial == 0:
            group = max(
                candidates,
                key=lambda g: (
                    affinity[element][g],
                    remaining_slots[g],
                    -g,
                ),
            )
        else:
            best_score = max(affinity[element][g] for g in candidates)
            scale = max(1.0, abs(best_score))
            near_best = [
                g for g in candidates
                if affinity[element][g] >= best_score - 0.03 * scale
            ]
            group = rng.choice(near_best)

        assignment[element] = group
        remaining_slots[group] -= 1
        for i in range(n):
            affinity[i][group] += distance[i][element]

    return assignment


def perturb_assignment(assignment, m, lower, upper, rng, strength):
    result = assignment[:]
    n = len(result)
    groups = [[] for _ in range(m)]
    for i, g in enumerate(result):
        groups[g].append(i)

    for _ in range(strength):
        if n >= 2 and rng.random() < 0.75:
            a = rng.randrange(n)
            b = rng.randrange(n)
            if result[a] != result[b]:
                ga, gb = result[a], result[b]
                result[a], result[b] = gb, ga
                groups[ga].remove(a)
                groups[gb].remove(b)
                groups[ga].append(b)
                groups[gb].append(a)
        else:
            sources = [g for g in range(m) if len(groups[g]) > lower[g]]
            targets = [g for g in range(m) if len(groups[g]) < upper[g]]
            if sources and targets:
                ga = rng.choice(sources)
                valid_targets = [g for g in targets if g != ga]
                if valid_targets:
                    gb = rng.choice(valid_targets)
                    a = rng.choice(groups[ga])
                    groups[ga].remove(a)
                    groups[gb].append(a)
                    result[a] = gb

    return result


def local_search(
    initial_assignment,
    initial_objective,
    distance,
    lower,
    upper,
    deadline,
    report_candidate,
):
    n = len(initial_assignment)
    m = len(lower)
    assignment = initial_assignment[:]

    groups = [set() for _ in range(m)]
    for i, g in enumerate(assignment):
        groups[g].add(i)

    affinity = [[0.0] * m for _ in range(n)]
    for i in range(n - 1):
        gi = assignment[i]
        row_i = distance[i]
        for j in range(i + 1, n):
            value = row_i[j]
            gj = assignment[j]
            affinity[i][gj] += value
            affinity[j][gi] += value

    current_objective = float(initial_objective)
    iteration = 0
    epsilon = 1e-11

    while time.monotonic() < deadline:
        best_gain = epsilon
        best_move = None
        timed_out = False

        # Cardinality-changing single-element relocations.
        for a in range(n):
            if (a & 31) == 0 and time.monotonic() >= deadline:
                timed_out = True
                break
            ga = assignment[a]
            if len(groups[ga]) <= lower[ga]:
                continue
            remove_value = affinity[a][ga]
            for gb in range(m):
                if gb == ga or len(groups[gb]) >= upper[gb]:
                    continue
                gain = affinity[a][gb] - remove_value
                if gain > best_gain:
                    best_gain = gain
                    best_move = ("move", a, ga, gb)

        if timed_out:
            break

        # Capacity-preserving pair swaps.
        for a in range(n - 1):
            if (a & 15) == 0 and time.monotonic() >= deadline:
                timed_out = True
                break
            ga = assignment[a]
            row_a = distance[a]
            for b in range(a + 1, n):
                gb = assignment[b]
                if ga == gb:
                    continue
                dab = row_a[b]
                gain = (
                    affinity[a][gb] - dab - affinity[a][ga]
                    + affinity[b][ga] - dab - affinity[b][gb]
                )
                if gain > best_gain:
                    best_gain = gain
                    best_move = ("swap", a, b, ga, gb)

        if timed_out or best_move is None:
            break

        if best_move[0] == "move":
            _, a, ga, gb = best_move
            for i in range(n):
                value = distance[i][a]
                affinity[i][ga] -= value
                affinity[i][gb] += value
            groups[ga].remove(a)
            groups[gb].add(a)
            assignment[a] = gb
        else:
            _, a, b, ga, gb = best_move
            for i in range(n):
                dia = distance[i][a]
                dib = distance[i][b]
                affinity[i][ga] += dib - dia
                affinity[i][gb] += dia - dib

            groups[ga].remove(a)
            groups[gb].remove(b)
            groups[ga].add(b)
            groups[gb].add(a)
            assignment[a] = gb
            assignment[b] = ga

        current_objective += best_gain
        iteration += 1

        # Periodically remove accumulated floating-point drift.
        if iteration % 30 == 0:
            current_objective = objective_value(assignment, distance)

        report_candidate(assignment, current_objective)

    current_objective = objective_value(assignment, distance)
    report_candidate(assignment, current_objective)
    return assignment, current_objective


def try_gurobi(
    n,
    m,
    lower,
    upper,
    distance,
    warm_assignment,
    deadline,
    report_candidate,
):
    remaining = deadline - time.monotonic()
    if remaining <= 0.25:
        return

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return

    try:
        model = gp.Model("maximum_diversity_grouping")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.NonConvex = 2
        model.Params.TimeLimit = max(0.05, deadline - time.monotonic())

        x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

        for i in range(n):
            model.addConstr(gp.quicksum(x[i, g] for g in range(m)) == 1)

        for g in range(m):
            size_expr = gp.quicksum(x[i, g] for i in range(n))
            model.addConstr(size_expr >= lower[g])
            model.addConstr(size_expr <= upper[g])

        objective = gp.QuadExpr()
        for i in range(n - 1):
            row_i = distance[i]
            for j in range(i + 1, n):
                dij = row_i[j]
                if dij != 0.0:
                    for g in range(m):
                        objective.add(dij * x[i, g] * x[j, g])

        model.setObjective(objective, GRB.MAXIMIZE)

        for i in range(n):
            assigned_group = warm_assignment[i]
            for g in range(m):
                x[i, g].Start = 1.0 if g == assigned_group else 0.0

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                values = cb_model.cbGetSolution(x)
                candidate = []
                for i in range(n):
                    group = max(range(m), key=lambda g: values[i, g])
                    candidate.append(group)

                counts = [0] * m
                for g in candidate:
                    counts[g] += 1
                if all(lower[g] <= counts[g] <= upper[g] for g in range(m)):
                    obj = objective_value(candidate, distance)
                    report_candidate(candidate, obj)
            except Exception:
                pass

        model.optimize(callback)

        if model.SolCount > 0:
            candidate = []
            for i in range(n):
                candidate.append(max(range(m), key=lambda g: x[i, g].X))
            counts = [0] * m
            for g in candidate:
                counts[g] += 1
            if all(lower[g] <= counts[g] <= upper[g] for g in range(m)):
                report_candidate(candidate, objective_value(candidate, distance))

    except Exception:
        # The heuristic incumbent remains valid if Gurobi is unavailable,
        # the license is restricted, or model construction is interrupted.
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
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path
        else None
    )

    n, m, lower, upper, distance = read_instance(args.instance_path)
    rng = random.Random(0)

    best_assignment = None
    best_objective = -math.inf

    def report_candidate(candidate_assignment, candidate_objective):
        nonlocal best_assignment, best_objective
        tolerance = 1e-9 * max(1.0, abs(best_objective)) if math.isfinite(best_objective) else 0.0
        if best_assignment is None or candidate_objective > best_objective + tolerance:
            best_assignment = candidate_assignment[:]
            best_objective = float(candidate_objective)
            if logger:
                solution = make_solution(best_assignment, best_objective, m)
                logger.log_solution(best_objective, solution)

    # Obtain and log a feasible solution immediately.
    base_sizes = concentrated_sizes(n, m, lower, upper)
    base_assignment = sequential_assignment(base_sizes)
    base_objective = objective_value(base_assignment, distance)
    report_candidate(base_assignment, base_objective)

    # Reserve substantial time for exact MIP search on modest-size instances.
    quadratic_terms = m * n * max(0, n - 1) // 2
    mip_eligible = (
        n <= 100
        and quadratic_terms <= 50000
        and args.time_limit >= 2
    )

    if mip_eligible:
        heuristic_deadline = min(
            deadline,
            start_time + max(0.25, 0.28 * max(0, args.time_limit)),
        )
    else:
        heuristic_deadline = deadline

    trial = 0
    while time.monotonic() < heuristic_deadline:
        if trial == 0:
            sizes = base_sizes
        else:
            sizes = randomized_sizes(
                n, m, lower, upper, rng, (trial - 1) % 3
            )

        candidate = greedy_assignment(
            n, m, sizes, distance, rng, trial
        )
        candidate_objective = objective_value(candidate, distance)
        report_candidate(candidate, candidate_objective)

        candidate, candidate_objective = local_search(
            candidate,
            candidate_objective,
            distance,
            lower,
            upper,
            heuristic_deadline,
            report_candidate,
        )

        trial += 1
        if time.monotonic() >= heuristic_deadline:
            break

        # Alternate fresh constructions with perturbations of the incumbent.
        strength = min(max(2, n // 12), 12)
        perturbed = perturb_assignment(
            best_assignment, m, lower, upper, rng, strength
        )
        perturbed_objective = objective_value(perturbed, distance)
        perturbed, perturbed_objective = local_search(
            perturbed,
            perturbed_objective,
            distance,
            lower,
            upper,
            heuristic_deadline,
            report_candidate,
        )
        trial += 1

    if mip_eligible and time.monotonic() < deadline:
        try_gurobi(
            n,
            m,
            lower,
            upper,
            distance,
            best_assignment,
            deadline,
            report_candidate,
        )

    # Recompute exactly from the selected assignment before serialization.
    best_objective = objective_value(best_assignment, distance)
    final_solution = make_solution(best_assignment, best_objective, m)

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()