import argparse
import json
import math
import os
import random
import time

from solution_logger import SolutionLogger


def make_solution(objective, assignment):
    return {
        "objective_value": float(objective),
        "assignment": {str(i): int(assignment[i]) for i in range(len(assignment))}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["n_clients"])
    m = int(data["n_servers"])
    variable_count = n * m

    base_costs = [
        [float(data["linear_costs"][i][s]) for s in range(m)]
        for i in range(n)
    ]

    # Effective linear costs include any unusual self-interaction entries.
    effective_costs = [row[:] for row in base_costs]
    self_adjustment = [[0.0] * m for _ in range(n)]

    # Aggregate duplicate/reversed quadratic entries.
    edge_costs = {}
    for entry in data.get("quadratic_costs", []):
        i, s, j, t = map(int, entry[:4])
        q = float(entry[4])

        if not (0 <= i < n and 0 <= j < n and 0 <= s < m and 0 <= t < m):
            continue
        if q == 0.0:
            continue

        a = i * m + s
        b = j * m + t

        if a == b:
            effective_costs[i][s] += q
            self_adjustment[i][s] += q
            continue

        # Two different assignments of the same client can never be active.
        if i == j:
            continue

        if a > b:
            a, b = b, a
        edge_costs[(a, b)] = edge_costs.get((a, b), 0.0) + q

    edge_costs = {
        key: value for key, value in edge_costs.items()
        if value != 0.0
    }
    edges = [(a, b, q) for (a, b), q in edge_costs.items()]

    adjacency = [[] for _ in range(variable_count)]
    for a, b, q in edges:
        adjacency[a].append((b, q))
        adjacency[b].append((a, q))

    def exact_objective(assignment):
        active = [False] * variable_count
        total = 0.0
        for i, s in enumerate(assignment):
            v = i * m + s
            active[v] = True
            total += effective_costs[i][s]
        for a, b, q in edges:
            if active[a] and active[b]:
                total += q
        return float(total)

    best_assignment = [
        min(range(m), key=lambda s: effective_costs[i][s])
        for i in range(n)
    ]
    best_objective = exact_objective(best_assignment)

    if logger:
        logger.log_solution(
            best_objective,
            make_solution(best_objective, best_assignment)
        )

    def update_best(assignment, objective=None):
        nonlocal best_assignment, best_objective
        if objective is None:
            objective = exact_objective(assignment)
        tolerance = 1e-9 * max(1.0, abs(best_objective))
        if objective < best_objective - tolerance:
            best_assignment = list(assignment)
            best_objective = float(objective)
            if logger:
                logger.log_solution(
                    best_objective,
                    make_solution(best_objective, best_assignment)
                )
            return True
        return False

    def active_neighbor_sums(assignment):
        active = [False] * variable_count
        for i, s in enumerate(assignment):
            active[i * m + s] = True

        sums = [0.0] * variable_count
        for a, b, q in edges:
            if active[a]:
                sums[b] += q
            if active[b]:
                sums[a] += q
        return active, sums

    def local_search(initial_assignment, local_deadline):
        assignment = list(initial_assignment)
        objective = exact_objective(assignment)
        active, neighbor_sums = active_neighbor_sums(assignment)

        improved = True
        while improved and time.monotonic() < local_deadline:
            improved = False
            for i in range(n):
                if time.monotonic() >= local_deadline:
                    break

                old_s = assignment[i]
                old_v = i * m + old_s
                best_s = old_s
                best_delta = 0.0

                for new_s in range(m):
                    if new_s == old_s:
                        continue
                    new_v = i * m + new_s
                    delta = (
                        effective_costs[i][new_s]
                        - effective_costs[i][old_s]
                        - neighbor_sums[old_v]
                        + neighbor_sums[new_v]
                    )
                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_s = new_s

                if best_s == old_s:
                    continue

                new_v = i * m + best_s

                active[old_v] = False
                for w, q in adjacency[old_v]:
                    neighbor_sums[w] -= q

                active[new_v] = True
                for w, q in adjacency[new_v]:
                    neighbor_sums[w] += q

                assignment[i] = best_s
                objective += best_delta
                improved = True
                update_best(assignment, objective)

        # Eliminate accumulated floating-point drift.
        objective = exact_objective(assignment)
        update_best(assignment, objective)
        return assignment, objective

    def greedy_assignment(order, randomized=False, rng=None):
        assignment = [-1] * n
        active = [False] * variable_count
        neighbor_sums = [0.0] * variable_count

        for i in order:
            values = [
                effective_costs[i][s] + neighbor_sums[i * m + s]
                for s in range(m)
            ]
            if randomized and m > 1:
                ranked = sorted(range(m), key=lambda s: values[s])
                restricted = ranked[:min(3, m)]
                chosen_s = rng.choice(restricted)
            else:
                chosen_s = min(range(m), key=lambda s: values[s])

            assignment[i] = chosen_s
            v = i * m + chosen_s
            active[v] = True
            for w, q in adjacency[v]:
                neighbor_sums[w] += q

        return assignment

    # Spend a small portion of the budget constructing good warm starts.
    remaining_initial = max(0.0, deadline - time.monotonic())
    heuristic_budget = min(3.0, max(0.05, 0.08 * remaining_initial))
    heuristic_deadline = min(deadline, time.monotonic() + heuristic_budget)

    if time.monotonic() < heuristic_deadline:
        local_search(best_assignment, heuristic_deadline)

    rng = random.Random(0)
    order_by_spread = sorted(
        range(n),
        key=lambda i: max(effective_costs[i]) - min(effective_costs[i]),
        reverse=True
    )

    if time.monotonic() < heuristic_deadline:
        candidate = greedy_assignment(order_by_spread)
        update_best(candidate)
        local_search(candidate, heuristic_deadline)

    restart = 0
    while time.monotonic() < heuristic_deadline:
        order = list(range(n))
        rng.shuffle(order)
        candidate = greedy_assignment(
            order,
            randomized=(restart % 2 == 1),
            rng=rng
        )
        update_best(candidate)
        local_search(candidate, heuristic_deadline)
        restart += 1

    # Detect complete same-server interaction stars. For such a server:
    #
    # linear + adjacent quadratic cost
    #   = k_s * sum_i c[i,s] x[i,s]
    #
    # where k_s is the number of clients assigned to server s. This replaces
    # O(n^2) pair-product variables with O(n) product variables.
    same_server_count = [0] * m
    same_server_valid = [True] * m

    for a, b, q in edges:
        i, s = divmod(a, m)
        j, t = divmod(b, m)
        if s != t:
            continue
        same_server_count[s] += 1
        expected = base_costs[i][s] + base_costs[j][s]
        tolerance = 1e-8 * max(1.0, abs(expected), abs(q))
        if abs(q - expected) > tolerance:
            same_server_valid[s] = False

    expected_pairs = n * (n - 1) // 2
    compressed_server = [
        same_server_valid[s] and same_server_count[s] == expected_pairs
        for s in range(m)
    ]

    try:
        remaining = deadline - time.monotonic()
        if remaining > 0.02:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("client_server_quadratic_assignment")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1

            x = model.addVars(
                variable_count,
                vtype=GRB.BINARY,
                name="x"
            )

            for i in range(n):
                model.addConstr(
                    gp.quicksum(x[i * m + s] for s in range(m)) == 1,
                    name=f"assign_{i}"
                )

            objective = gp.LinExpr()
            z_variables = {}

            for s in range(m):
                if compressed_server[s]:
                    load = gp.quicksum(x[i * m + s] for i in range(n))
                    for i in range(n):
                        v = i * m + s
                        z = model.addVar(
                            lb=0.0,
                            ub=float(n),
                            vtype=GRB.CONTINUOUS,
                            name=f"z_{i}_{s}"
                        )
                        z_variables[(i, s)] = z
                        model.addConstr(z <= load)
                        model.addConstr(z <= n * x[v])
                        model.addConstr(z >= load - n * (1 - x[v]))
                        objective += base_costs[i][s] * z
                        if self_adjustment[i][s] != 0.0:
                            objective += self_adjustment[i][s] * x[v]
                else:
                    for i in range(n):
                        objective += effective_costs[i][s] * x[i * m + s]

            y_records = []
            for edge_index, (a, b, q) in enumerate(edges):
                _, s = divmod(a, m)
                _, t = divmod(b, m)

                if s == t and compressed_server[s]:
                    continue

                y = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"y_{edge_index}"
                )

                # Sign-specific McCormick constraints are sufficient for
                # minimization and reduce the model size.
                if q >= 0.0:
                    model.addConstr(y >= x[a] + x[b] - 1)
                else:
                    model.addConstr(y <= x[a])
                    model.addConstr(y <= x[b])

                objective += q * y
                y_records.append((y, a, b))

            model.setObjective(objective, GRB.MINIMIZE)

            # Warm-start from the best heuristic assignment.
            active_start = [False] * variable_count
            server_load = [0] * m
            for i, s in enumerate(best_assignment):
                active_start[i * m + s] = True
                server_load[s] += 1

            for v in range(variable_count):
                x[v].Start = 1.0 if active_start[v] else 0.0
            for y, a, b in y_records:
                y.Start = 1.0 if active_start[a] and active_start[b] else 0.0
            for (i, s), z in z_variables.items():
                z.Start = float(server_load[s] if best_assignment[i] == s else 0)

            model.update()
            model.Params.TimeLimit = max(
                0.01, deadline - time.monotonic()
            )

            def incumbent_callback(cb_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                try:
                    values = cb_model.cbGetSolution(
                        [x[v] for v in range(variable_count)]
                    )
                    assignment = []
                    for i in range(n):
                        chosen = max(
                            range(m),
                            key=lambda s: values[i * m + s]
                        )
                        assignment.append(chosen)
                    objective_value = exact_objective(assignment)
                    update_best(assignment, objective_value)
                except Exception:
                    # Logging or callback extraction should never abort search.
                    pass

            model.optimize(incumbent_callback)

            if model.SolCount > 0:
                assignment = []
                for i in range(n):
                    chosen = max(
                        range(m),
                        key=lambda s: x[i * m + s].X
                    )
                    assignment.append(chosen)
                update_best(assignment, exact_objective(assignment))

    except Exception:
        # Preserve and return the best heuristic/incumbent solution if Gurobi
        # is unavailable, model construction fails, or the budget expires.
        pass

    best_objective = exact_objective(best_assignment)
    final_solution = make_solution(best_objective, best_assignment)

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()