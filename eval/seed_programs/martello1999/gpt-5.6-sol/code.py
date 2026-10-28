import argparse
import json
import math
import os
import time


def main():
    parser = argparse.ArgumentParser(description="Solve a 0/1 knapsack instance.")
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None, type=str)
    args = parser.parse_args()

    from solution_logger import SolutionLogger

    logger = (
        SolutionLogger(args.log_path, sense="maximize")
        if args.log_path
        else None
    )

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    capacity = int(instance["capacity"])
    raw_items = instance["items"]

    ids = [int(item["id"]) for item in raw_items]
    profits = [int(item["profit"]) for item in raw_items]
    weights = [int(item["weight"]) for item in raw_items]
    n = len(raw_items)

    state = {
        "profit": 0,
        "selected": [],
    }

    def make_solution(profit, selected_indices):
        return {
            "objective_value": int(profit),
            "selected_items": [ids[i] for i in selected_indices],
        }

    def record_if_better(selected_indices):
        total_weight = sum(weights[i] for i in selected_indices)
        if total_weight > capacity:
            return False

        total_profit = sum(profits[i] for i in selected_indices)
        if total_profit <= state["profit"]:
            return False

        selected_indices = sorted(selected_indices)
        state["profit"] = int(total_profit)
        state["selected"] = selected_indices

        solution = make_solution(total_profit, selected_indices)
        if logger:
            logger.log_solution(total_profit, solution)
        return True

    def greedy(order):
        selected = []
        used_weight = 0
        for i in order:
            new_weight = used_weight + weights[i]
            if new_weight <= capacity:
                selected.append(i)
                used_weight = new_weight
        return selected

    if n > 0:
        # Density greedy.
        density_order = sorted(
            range(n),
            key=lambda i: (
                -(profits[i] / weights[i]),
                -profits[i],
                weights[i],
                i,
            ),
        )
        record_if_better(greedy(density_order))

        # High-profit-first greedy.
        profit_order = sorted(
            range(n),
            key=lambda i: (-profits[i], weights[i], i),
        )
        record_if_better(greedy(profit_order))

        # Light-item-first greedy can be useful on instances with many items.
        weight_order = sorted(
            range(n),
            key=lambda i: (weights[i], -profits[i], i),
        )
        record_if_better(greedy(weight_order))

        # Explicitly consider the best singleton.
        best_singleton = max(range(n), key=lambda i: (profits[i], -weights[i]))
        record_if_better([best_singleton])

    # If all items fit, the positive-profit assumption makes this optimal.
    all_items_fit = sum(weights) <= capacity
    if all_items_fit:
        record_if_better(list(range(n)))

    # Use Gurobi for exact branch-and-bound when time remains.
    if (
        n > 0
        and not all_items_fit
        and time.monotonic() < deadline
    ):
        model = None
        env = None
        try:
            import gurobipy as gp
            from gurobipy import GRB

            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()

            model = gp.Model("knapsack", env=env)
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1

            # Scaling by common divisors preserves the feasible set and
            # objective ordering while reducing coefficient magnitudes.
            weight_gcd = 0
            profit_gcd = 0
            for w in weights:
                weight_gcd = math.gcd(weight_gcd, w)
            for p in profits:
                profit_gcd = math.gcd(profit_gcd, p)
            weight_gcd = max(1, weight_gcd)
            profit_gcd = max(1, profit_gcd)

            scaled_weights = [w // weight_gcd for w in weights]
            scaled_capacity = capacity // weight_gcd
            scaled_profits = [p // profit_gcd for p in profits]

            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            variables = [x[i] for i in range(n)]

            weight_expr = gp.LinExpr(scaled_weights, variables)
            profit_expr = gp.LinExpr(scaled_profits, variables)

            model.addConstr(weight_expr <= scaled_capacity, name="capacity")
            model.setObjective(profit_expr, GRB.MAXIMIZE)

            # The incumbent is known feasible, so this bound is safe.
            if state["profit"] > 0:
                model.addConstr(
                    profit_expr >= state["profit"] // profit_gcd,
                    name="incumbent_bound",
                )

            incumbent_set = set(state["selected"])
            starts = [1.0 if i in incumbent_set else 0.0 for i in range(n)]
            model.setAttr(GRB.Attr.Start, variables, starts)

            model.update()

            remaining = deadline - time.monotonic()
            if remaining > 0:
                model.Params.TimeLimit = max(0.001, remaining)

                model._variables = variables

                def incumbent_callback(cb_model, where):
                    if where != GRB.Callback.MIPSOL:
                        return

                    scaled_obj = cb_model.cbGet(
                        GRB.Callback.MIPSOL_OBJ
                    )
                    if (
                        scaled_obj * profit_gcd
                        <= state["profit"] + 1e-6
                    ):
                        return

                    values = cb_model.cbGetSolution(
                        cb_model._variables
                    )
                    selected = [
                        i for i, value in enumerate(values)
                        if value > 0.5
                    ]
                    record_if_better(selected)

                model.optimize(incumbent_callback)

                # Defensive final extraction in case a solver incumbent was
                # not delivered through the callback.
                if model.SolCount > 0:
                    values = model.getAttr(GRB.Attr.X, variables)
                    selected = [
                        i for i, value in enumerate(values)
                        if value > 0.5
                    ]
                    record_if_better(selected)

        except Exception:
            # The heuristic incumbent remains a valid fallback if Gurobi
            # cannot initialize or terminates unexpectedly.
            pass
        finally:
            if model is not None:
                try:
                    model.dispose()
                except Exception:
                    pass
            if env is not None:
                try:
                    env.dispose()
                except Exception:
                    pass

    final_solution = make_solution(
        state["profit"],
        state["selected"],
    )

    solution_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"))


if __name__ == "__main__":
    main()