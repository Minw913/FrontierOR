import argparse
import heapq
import json
import math
import os
import time

from solution_logger import SolutionLogger


def make_solution(value, selected):
    return {
        "objective_value": float(value),
        "selected_elements": sorted(int(j) for j in selected),
    }


def solve(instance, deadline, logger):
    instance_type = instance["instance_type"].upper()
    n = int(instance["n"])
    m = int(instance["m"])
    k = max(0, min(int(instance["k"]), n))
    data = instance["data_matrix"]
    node_weights = instance.get("node_weights", [0.0] * m)
    element_weights = instance.get("element_weights", [0.0] * n)

    best_value = 0.0
    best_selected = []

    def evaluate(selected):
        if not selected:
            return 0.0

        if instance_type == "LOC":
            total = 0.0
            for i in range(m):
                row = data[i]
                value = 0.0
                for j in selected:
                    a = float(row[j])
                    if a > value:
                        value = a
                total += value
            return total

        if instance_type == "COV":
            total = 0.0
            for i in range(m):
                row = data[i]
                for j in selected:
                    if float(row[j]) > 0.5:
                        total += float(node_weights[i])
                        break
            return total

        if instance_type == "INF":
            total = 0.0
            for i in range(m):
                row = data[i]
                failure = 1.0
                for j in selected:
                    if float(row[j]) > 0.5:
                        p = min(1.0, max(0.0, float(element_weights[j])))
                        failure *= 1.0 - p
                total += 1.0 - failure
            return total

        raise ValueError(f"Unsupported instance type: {instance_type}")

    def update_best(selected, value=None):
        nonlocal best_value, best_selected
        if value is None:
            value = evaluate(selected)
        value = float(value)
        tolerance = 1e-10 * max(1.0, abs(best_value))
        if value > best_value + tolerance:
            best_value = value
            best_selected = sorted(selected)
            solution = make_solution(best_value, best_selected)
            if logger:
                logger.log_solution(best_value, solution)
            return True
        return False

    # The empty set is the first known feasible solution.
    if logger:
        logger.log_solution(0.0, make_solution(0.0, []))

    if k == 0 or n == 0 or m == 0 or time.monotonic() >= deadline:
        return make_solution(best_value, best_selected)

    # Lazy greedy provides a strong feasible starting solution. All three
    # functions in the schema are monotone submodular, so stale marginal
    # gains are valid upper bounds.
    selected = []
    selected_flag = [False] * n
    heap = []

    if instance_type == "LOC":
        current = [0.0] * m
        for j in range(n):
            gain = 0.0
            for i in range(m):
                gain += max(0.0, float(data[i][j]))
            heapq.heappush(heap, (-gain, j))

        current_value = 0.0
        while len(selected) < k and heap and time.monotonic() < deadline:
            _, j = heapq.heappop(heap)
            if selected_flag[j]:
                continue

            gain = 0.0
            for i in range(m):
                a = float(data[i][j])
                if a > current[i]:
                    gain += a - current[i]

            next_bound = -heap[0][0] if heap else -math.inf
            if gain + 1e-12 >= next_bound:
                if gain <= 1e-12:
                    break
                selected_flag[j] = True
                selected.append(j)
                for i in range(m):
                    a = float(data[i][j])
                    if a > current[i]:
                        current[i] = a
                current_value = float(sum(current))
                update_best(selected, current_value)
            else:
                heapq.heappush(heap, (-gain, j))

    elif instance_type == "COV":
        covered_by = [[] for _ in range(n)]
        for i in range(m):
            row = data[i]
            for j in range(n):
                if float(row[j]) > 0.5:
                    covered_by[j].append(i)

        covered = [False] * m
        for j in range(n):
            gain = sum(float(node_weights[i]) for i in covered_by[j])
            heapq.heappush(heap, (-gain, j))

        current_value = 0.0
        while len(selected) < k and heap and time.monotonic() < deadline:
            _, j = heapq.heappop(heap)
            if selected_flag[j]:
                continue

            gain = 0.0
            for i in covered_by[j]:
                if not covered[i]:
                    gain += float(node_weights[i])

            next_bound = -heap[0][0] if heap else -math.inf
            if gain + 1e-12 >= next_bound:
                if gain <= 1e-12:
                    break
                selected_flag[j] = True
                selected.append(j)
                for i in covered_by[j]:
                    if not covered[i]:
                        covered[i] = True
                        current_value += float(node_weights[i])
                update_best(selected, current_value)
            else:
                heapq.heappush(heap, (-gain, j))

    elif instance_type == "INF":
        targets_by_source = [[] for _ in range(n)]
        probabilities = [
            min(1.0, max(0.0, float(element_weights[j]))) for j in range(n)
        ]

        for i in range(m):
            row = data[i]
            for j in range(n):
                if float(row[j]) > 0.5:
                    targets_by_source[j].append(i)

        failure = [1.0] * m
        for j in range(n):
            gain = probabilities[j] * len(targets_by_source[j])
            heapq.heappush(heap, (-gain, j))

        current_value = 0.0
        while len(selected) < k and heap and time.monotonic() < deadline:
            _, j = heapq.heappop(heap)
            if selected_flag[j]:
                continue

            p = probabilities[j]
            gain = p * sum(failure[i] for i in targets_by_source[j])
            next_bound = -heap[0][0] if heap else -math.inf

            if gain + 1e-12 >= next_bound:
                if gain <= 1e-12:
                    break
                selected_flag[j] = True
                selected.append(j)
                q = 1.0 - p
                for i in targets_by_source[j]:
                    old_failure = failure[i]
                    failure[i] *= q
                    current_value += old_failure - failure[i]
                update_best(selected, current_value)
            else:
                heapq.heappush(heap, (-gain, j))
    else:
        raise ValueError(f"Unsupported instance type: {instance_type}")

    remaining = deadline - time.monotonic()
    if remaining <= 0.05:
        return make_solution(best_value, best_selected)

    # Improve and, when time permits, prove optimality using Gurobi.
    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("submodular_cardinality")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(0.01, remaining)
        model.Params.LazyConstraints = 1

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        y = model.addVars(m, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
        model.addConstr(gp.quicksum(x[j] for j in range(n)) <= k, name="cardinality")

        start_set = set(best_selected)
        for j in range(n):
            x[j].Start = 1.0 if j in start_set else 0.0

        if instance_type == "LOC":
            for i in range(m):
                row = data[i]
                upper = max((float(v) for v in row), default=0.0)
                model.addConstr(y[i] <= upper)
                model.addConstr(
                    y[i] <= gp.quicksum(
                        float(row[j]) * x[j]
                        for j in range(n)
                        if float(row[j]) > 0.0
                    )
                )

                start_y = 0.0
                for j in best_selected:
                    start_y = max(start_y, float(row[j]))
                y[i].Start = start_y

            model.setObjective(gp.quicksum(y[i] for i in range(m)), GRB.MAXIMIZE)

        elif instance_type == "COV":
            for i in range(m):
                row = data[i]
                model.addConstr(
                    y[i] <= gp.quicksum(
                        x[j] for j in range(n) if float(row[j]) > 0.5
                    )
                )
                y[i].Start = 1.0 if any(
                    float(row[j]) > 0.5 for j in best_selected
                ) else 0.0

            model.setObjective(
                gp.quicksum(float(node_weights[i]) * y[i] for i in range(m)),
                GRB.MAXIMIZE,
            )

        else:  # INF
            probabilities = [
                min(1.0, max(0.0, float(element_weights[j]))) for j in range(n)
            ]
            # A probability of one is represented to essentially machine
            # precision while keeping logarithmic coefficients finite.
            log_coeff = [
                -math.log(max(1e-15, 1.0 - probabilities[j])) for j in range(n)
            ]

            for i in range(m):
                row = data[i]
                terms = [
                    log_coeff[j] * x[j]
                    for j in range(n)
                    if float(row[j]) > 0.5 and log_coeff[j] > 0.0
                ]
                s_expr = gp.quicksum(terms)
                model.addConstr(y[i] <= 1.0)
                model.addConstr(y[i] <= s_expr)

                failure_start = 1.0
                for j in best_selected:
                    if float(row[j]) > 0.5:
                        failure_start *= 1.0 - probabilities[j]
                y[i].Start = 1.0 - failure_start

            model.setObjective(gp.quicksum(y[i] for i in range(m)), GRB.MAXIMIZE)

        def callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return

            xv = cb_model.cbGetSolution([x[j] for j in range(n)])
            candidate = [j for j, value in enumerate(xv) if value > 0.5]
            candidate_value = evaluate(candidate)
            update_best(candidate, candidate_value)

            if instance_type == "LOC":
                yv = cb_model.cbGetSolution([y[i] for i in range(m)])
                for i in range(m):
                    row = data[i]
                    actual = 0.0
                    for j in candidate:
                        actual = max(actual, float(row[j]))

                    if yv[i] > actual + 1e-7:
                        # Exact threshold inequality for max_j a_ij x_j.
                        expr = gp.LinExpr(actual)
                        for j in range(n):
                            a = float(row[j])
                            if a > actual + 1e-15:
                                expr += (a - actual) * x[j]
                        cb_model.cbLazy(y[i] <= expr)

            elif instance_type == "INF":
                yv = cb_model.cbGetSolution([y[i] for i in range(m)])
                candidate_flag = [False] * n
                for j in candidate:
                    candidate_flag[j] = True

                for i in range(m):
                    row = data[i]
                    s0 = 0.0
                    active_js = []
                    for j in range(n):
                        if float(row[j]) > 0.5 and log_coeff[j] > 0.0:
                            active_js.append(j)
                            if candidate_flag[j]:
                                s0 += log_coeff[j]

                    exp_term = math.exp(-min(s0, 745.0))
                    surrogate_value = 1.0 - exp_term
                    if yv[i] > surrogate_value + 1e-7:
                        # Tangent upper bound for 1-exp(-s), a concave function.
                        expr = gp.LinExpr(surrogate_value - exp_term * s0)
                        for j in active_js:
                            expr += exp_term * log_coeff[j] * x[j]
                        cb_model.cbLazy(y[i] <= expr)

        # Reset the optimization-only limit after model construction.
        remaining = deadline - time.monotonic()
        if remaining > 0.01:
            model.Params.TimeLimit = remaining
            model.optimize(callback)

            if model.SolCount > 0:
                final_candidate = [j for j in range(n) if x[j].X > 0.5]
                update_best(final_candidate)

    except Exception:
        # The greedy incumbent remains valid if Gurobi is unavailable,
        # construction fails, or optimization is interrupted.
        pass

    return make_solution(best_value, best_selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    solution = solve(instance, deadline, logger)

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()