import argparse
import json
import os
import time

from solution_logger import SolutionLogger


def empty_solution():
    return {
        "objective_value": 0.0,
        "families_selected": [],
        "items_selected": []
    }


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(solution, f, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.monotonic()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    capacity = int(instance["knapsack_capacity"])

    best = {
        "objective": 0,
        "solution": empty_solution()
    }

    if logger:
        logger.log_solution(0.0, best["solution"])

    if args.time_limit <= 0 or time.monotonic() - start_time >= args.time_limit:
        write_solution(args.solution_path, best["solution"])
        return

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        write_solution(args.solution_path, best["solution"])
        return

    # Remove variables that cannot belong to a strictly improving solution.
    family_records = []
    for family in instance.get("families", []):
        family_id = int(family["family_id"])
        setup_cost = int(family["setup_cost"])
        setup_capacity = int(family["setup_capacity"])

        if setup_capacity > capacity:
            continue

        feasible_items = []
        total_available_profit = 0

        for item in family.get("items", []):
            weight = int(item["weight"])
            profit = int(item["profit"])

            # Zero-profit items are never needed because all costs and weights
            # are nonnegative.
            if profit <= 0:
                continue
            if setup_capacity + weight > capacity:
                continue

            record = {
                "family_id": family_id,
                "item_id": int(item["item_id"]),
                "weight": weight,
                "profit": profit
            }
            feasible_items.append(record)
            total_available_profit += profit

        # Such a family cannot yield positive net profit even if every
        # individually feasible item is selected.
        if not feasible_items or total_available_profit <= setup_cost:
            continue

        family_records.append({
            "family_id": family_id,
            "setup_cost": setup_cost,
            "setup_capacity": setup_capacity,
            "items": feasible_items
        })

    if not family_records or time.monotonic() - start_time >= args.time_limit:
        write_solution(args.solution_path, best["solution"])
        return

    model = None
    try:
        model = gp.Model("family_setup_knapsack")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0

        y_vars = []
        x_vars = []
        item_records = []
        family_item_vars = []

        capacity_expr = gp.LinExpr()
        objective_expr = gp.LinExpr()

        for family in family_records:
            y = model.addVar(
                vtype=GRB.BINARY,
                obj=-family["setup_cost"],
                name=f"family_{family['family_id']}"
            )
            y_vars.append(y)
            capacity_expr += family["setup_capacity"] * y
            objective_expr += -family["setup_cost"] * y

            local_x = []
            for item in family["items"]:
                x = model.addVar(
                    vtype=GRB.BINARY,
                    obj=item["profit"],
                    name=f"item_{family['family_id']}_{item['item_id']}"
                )
                x_vars.append(x)
                item_records.append(item)
                local_x.append(x)

                capacity_expr += item["weight"] * x
                objective_expr += item["profit"] * x

                # An item can only be selected after activating its family.
                model.addConstr(x <= y)

            # Avoid paying for or reporting an activated family with no item.
            model.addConstr(y <= gp.quicksum(local_x))
            family_item_vars.append(local_x)

        model.addConstr(capacity_expr <= capacity, name="capacity")
        model.setObjective(objective_expr, GRB.MAXIMIZE)

        # The empty solution is always feasible.
        for y in y_vars:
            y.Start = 0.0
        for x in x_vars:
            x.Start = 0.0

        all_vars = y_vars + x_vars
        number_of_families = len(y_vars)

        def make_solution(values):
            y_values = values[:number_of_families]
            x_values = values[number_of_families:]

            selected_families = []
            setup_total = 0
            for family, value in zip(family_records, y_values):
                if value > 0.5:
                    selected_families.append(family["family_id"])
                    setup_total += family["setup_cost"]

            selected_items = []
            profit_total = 0
            for item, value in zip(item_records, x_values):
                if value > 0.5:
                    selected_items.append({
                        "family": item["family_id"],
                        "item": item["item_id"]
                    })
                    profit_total += item["profit"]

            objective = profit_total - setup_total
            solution = {
                "objective_value": float(objective),
                "families_selected": selected_families,
                "items_selected": selected_items
            }
            return objective, solution

        def accept_solution(values):
            objective, solution = make_solution(values)
            if objective > best["objective"]:
                best["objective"] = objective
                best["solution"] = solution
                if logger:
                    logger.log_solution(float(objective), solution)

        def incumbent_callback(cb_model, where):
            if where != GRB.Callback.MIPSOL:
                return
            try:
                values = cb_model.cbGetSolution(all_vars)
                accept_solution(values)
            except Exception:
                # Logging or callback extraction errors should not discard the
                # best solution already obtained.
                return

        remaining = args.time_limit - (time.monotonic() - start_time)
        if remaining > 0:
            model.Params.TimeLimit = max(0.001, remaining)
            model.optimize(incumbent_callback)

            # Also inspect the final incumbent in case optimization ended
            # before its callback snapshot was processed.
            if model.SolCount > 0:
                final_values = [var.X for var in all_vars]
                accept_solution(final_values)

    except Exception:
        # The maintained empty or incumbent solution remains feasible.
        pass
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass

    write_solution(args.solution_path, best["solution"])


if __name__ == "__main__":
    main()