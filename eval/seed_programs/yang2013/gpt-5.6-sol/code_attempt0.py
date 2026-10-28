import argparse
import json
import time

from solution_logger import SolutionLogger


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
        data = json.load(f)

    n = int(data["n"])
    capacity = int(data["capacity"])
    weights = [int(v) for v in data["weights"]]
    profit = data["profit_matrix"]

    best_obj = None
    best_solution = None

    def write_solution():
        if best_solution is None:
            solution = [0] * n
            objective = 0
        else:
            solution = best_solution
            objective = best_obj

        result = {
            "objective_value": float(objective),
            "solution": [int(v) for v in solution],
        }
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(result, f)

    def record_solution(objective, solution):
        nonlocal best_obj, best_solution
        objective = int(objective)
        if best_obj is None or objective > best_obj:
            best_obj = objective
            best_solution = [int(v) for v in solution]
            solution_dict = {
                "objective_value": float(best_obj),
                "solution": best_solution.copy(),
            }
            if logger:
                logger.log_solution(float(best_obj), solution_dict)
            return True
        return False

    def exact_objective(solution):
        selected = [i for i, value in enumerate(solution) if value]
        total = 0
        for pos, i in enumerate(selected):
            total += int(profit[i][i])
            row = profit[i]
            for j in selected[pos + 1:]:
                total += int(row[j])
        return total

    # Zero-weight items can always be selected because every profit is
    # nonnegative. Items heavier than the capacity can never be selected.
    zero_items = [i for i in range(n) if weights[i] == 0]
    active_items = [
        i for i in range(n)
        if 0 < weights[i] <= capacity
    ]

    base_solution = [0] * n
    for i in zero_items:
        base_solution[i] = 1

    base_obj = exact_objective(base_solution)
    record_solution(base_obj, base_solution)

    # If every individually feasible positive-weight item fits, selecting all
    # of them is optimal by monotonicity of the nonnegative objective.
    if sum(weights[i] for i in active_items) <= capacity:
        all_solution = base_solution.copy()
        for i in active_items:
            all_solution[i] = 1
        record_solution(exact_objective(all_solution), all_solution)
        write_solution()
        return

    if not active_items or time.monotonic() >= deadline:
        write_solution()
        return

    k = len(active_items)

    # Fold profits involving forced zero-weight items into linear coefficients.
    constant_profit = base_obj
    linear_profit = []
    for item in active_items:
        value = int(profit[item][item])
        row = profit[item]
        for z in zero_items:
            value += int(row[z])
        linear_profit.append(value)

    def evaluate_active_selection(active_selected):
        total = constant_profit
        selected_positions = [
            p for p, selected in enumerate(active_selected) if selected
        ]
        for pos, p in enumerate(selected_positions):
            total += linear_profit[p]
            original_p = active_items[p]
            row = profit[original_p]
            for q in selected_positions[pos + 1:]:
                total += int(row[active_items[q]])
        return total

    def active_to_full(active_selected):
        full = base_solution.copy()
        for p, selected in enumerate(active_selected):
            if selected:
                full[active_items[p]] = 1
        return full

    # Produce several inexpensive feasible starts before invoking the solver.
    if time.monotonic() < deadline:
        orders = []

        # Standalone/currently forced profit density.
        orders.append(sorted(
            range(k),
            key=lambda p: (
                linear_profit[p] / weights[active_items[p]],
                linear_profit[p],
                -weights[active_items[p]],
            ),
            reverse=True,
        ))

        # Total interaction-potential density. Python's built-in sum performs
        # the row aggregation efficiently and this ordering often identifies
        # profitable clusters missed by standalone density.
        if time.monotonic() < deadline:
            potential = [
                linear_profit[p] + sum(int(v) for v in profit[active_items[p]])
                for p in range(k)
            ]
            orders.append(sorted(
                range(k),
                key=lambda p: (
                    potential[p] / weights[active_items[p]],
                    potential[p],
                    -weights[active_items[p]],
                ),
                reverse=True,
            ))

            # Absolute potential can help when high-profit clusters contain
            # relatively heavy items.
            orders.append(sorted(
                range(k),
                key=lambda p: (
                    potential[p],
                    linear_profit[p],
                    -weights[active_items[p]],
                ),
                reverse=True,
            ))

        seen_orders = set()
        for order in orders:
            if time.monotonic() >= deadline:
                break

            signature = tuple(order)
            if signature in seen_orders:
                continue
            seen_orders.add(signature)

            selected = [0] * k
            used_weight = 0
            for p in order:
                w = weights[active_items[p]]
                if used_weight + w <= capacity:
                    selected[p] = 1
                    used_weight += w

            objective = evaluate_active_selection(selected)
            record_solution(objective, active_to_full(selected))

    if time.monotonic() >= deadline:
        write_solution()
        return

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        write_solution()
        return

    model = None
    try:
        model = gp.Model("quadratic_knapsack")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.NonConvex = 2

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            write_solution()
            return
        model.Params.TimeLimit = max(0.001, remaining)

        x = model.addVars(k, vtype=GRB.BINARY, name="x")
        x_vars = [x[p] for p in range(k)]

        model.addConstr(
            gp.quicksum(weights[active_items[p]] * x[p] for p in range(k))
            <= capacity,
            name="capacity",
        )

        objective = gp.QuadExpr()
        objective.addConstant(float(constant_profit))
        objective.addTerms(
            [float(v) for v in linear_profit],
            x_vars,
        )

        construction_expired = False
        for p in range(k):
            coeffs = []
            vars1 = []
            vars2 = []
            original_p = active_items[p]
            row = profit[original_p]

            for q in range(p + 1, k):
                value = int(row[active_items[q]])
                if value:
                    coeffs.append(float(value))
                    vars1.append(x[p])
                    vars2.append(x[q])

            if coeffs:
                objective.addTerms(coeffs, vars1, vars2)

            if (p & 31) == 0 and time.monotonic() >= deadline:
                construction_expired = True
                break

        if construction_expired:
            model.dispose()
            write_solution()
            return

        model.setObjective(objective, GRB.MAXIMIZE)

        # Supply the best heuristic incumbent as a MIP start.
        for p in range(k):
            x[p].Start = float(best_solution[active_items[p]])

        model.update()

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            model.dispose()
            write_solution()
            return
        model.Params.TimeLimit = max(0.001, remaining)

        def callback(cb_model, where):
            if time.monotonic() >= deadline:
                cb_model.terminate()
                return

            if where != GRB.Callback.MIPSOL:
                return

            values = cb_model.cbGetSolution(x_vars)
            selected = [1 if value > 0.5 else 0 for value in values]

            used_weight = sum(
                weights[active_items[p]]
                for p, value in enumerate(selected)
                if value
            )
            if used_weight > capacity:
                return

            incumbent_obj = evaluate_active_selection(selected)
            record_solution(incumbent_obj, active_to_full(selected))

        model.optimize(callback)

        if model.SolCount > 0:
            selected = [1 if x[p].X > 0.5 else 0 for p in range(k)]
            used_weight = sum(
                weights[active_items[p]]
                for p, value in enumerate(selected)
                if value
            )
            if used_weight <= capacity:
                final_obj = evaluate_active_selection(selected)
                record_solution(final_obj, active_to_full(selected))

    except Exception:
        # The heuristic incumbent remains valid if model construction,
        # optimization, licensing, or callback processing fails.
        pass
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass

    write_solution()


if __name__ == "__main__":
    main()