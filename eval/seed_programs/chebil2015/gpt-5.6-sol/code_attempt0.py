import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def make_zero_solution(instance):
    return {
        "classes": [
            {
                "class_id": cls["class_id"],
                "y_i": 0,
                "items": [
                    {
                        "item_id": item["item_id"],
                        "x_ij": 0
                    }
                    for item in cls["items"]
                ]
            }
            for cls in instance["classes"]
        ],
        "objective_value": 0.0
    }


def write_solution(path, solution):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(solution, file, separators=(",", ":"))


def main():
    program_start = time.monotonic()

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    classes = instance["classes"]
    capacity = int(instance["b"])
    deadline = program_start + max(0, args.time_limit)

    best_solution = make_zero_solution(instance)
    best_objective = 0

    if logger:
        logger.log_solution(0.0, best_solution)

    # If no optimization time remains, the empty solution is always feasible.
    if time.monotonic() >= deadline:
        write_solution(args.solution_path, best_solution)
        return

    try:
        model = gp.Model("multiple_class_setup_knapsack")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        y_vars = []
        x_vars = []
        capacity_expr = gp.LinExpr()

        for cls in classes:
            setup_cost = int(cls["f_i"])
            setup_capacity = int(cls["d_i"])

            y = model.addVar(
                vtype=GRB.BINARY,
                obj=setup_cost,
                name=f"y_{cls['class_id']}"
            )
            y_vars.append(y)
            capacity_expr.addTerms(setup_capacity, y)

            class_x = []
            for item in cls["items"]:
                x = model.addVar(
                    vtype=GRB.BINARY,
                    obj=int(item["c_ij"]),
                    name=f"x_{cls['class_id']}_{item['item_id']}"
                )
                class_x.append(x)
                capacity_expr.addTerms(int(item["a_ij"]), x)

            x_vars.append(class_x)

        model.ModelSense = GRB.MAXIMIZE
        model.addConstr(capacity_expr <= capacity, name="capacity")

        # Selecting any item requires activating its class.
        for class_index, class_x in enumerate(x_vars):
            y = y_vars[class_index]
            for x in class_x:
                model.addConstr(x <= y)

        model.update()

        # Supply the guaranteed-feasible empty solution as a MIP start.
        for y in y_vars:
            y.Start = 0.0
        for class_x in x_vars:
            for x in class_x:
                x.Start = 0.0

        query_vars = list(y_vars)
        for class_x in x_vars:
            query_vars.extend(class_x)

        def solution_from_values(values):
            position = 0
            y_values = []
            for _ in classes:
                y_values.append(1 if values[position] > 0.5 else 0)
                position += 1

            result_classes = []
            objective = 0

            for class_index, cls in enumerate(classes):
                y_value = y_values[class_index]
                objective += int(cls["f_i"]) * y_value

                output_items = []
                for item in cls["items"]:
                    x_value = 1 if values[position] > 0.5 else 0
                    position += 1
                    objective += int(item["c_ij"]) * x_value
                    output_items.append({
                        "item_id": item["item_id"],
                        "x_ij": x_value
                    })

                result_classes.append({
                    "class_id": cls["class_id"],
                    "y_i": y_value,
                    "items": output_items
                })

            solution = {
                "classes": result_classes,
                "objective_value": float(objective)
            }
            return objective, solution

        def incumbent_callback(cb_model, where):
            nonlocal best_objective, best_solution

            if where == GRB.Callback.MIPSOL:
                callback_objective = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)

                if callback_objective > best_objective + 1e-6:
                    values = cb_model.cbGetSolution(query_vars)
                    objective, solution = solution_from_values(values)

                    if objective > best_objective:
                        best_objective = objective
                        best_solution = solution
                        if logger:
                            logger.log_solution(float(objective), solution)

                if time.monotonic() >= deadline:
                    cb_model.terminate()

            elif where == GRB.Callback.MIP:
                if time.monotonic() >= deadline:
                    cb_model.terminate()

        remaining_time = deadline - time.monotonic()
        if remaining_time > 0:
            model.Params.TimeLimit = max(0.001, remaining_time)
            model.optimize(incumbent_callback)

            # Retrieve the final incumbent as a safeguard in case it was not
            # delivered through the callback.
            if model.SolCount > 0:
                values = [var.X for var in query_vars]
                objective, solution = solution_from_values(values)
                if objective > best_objective:
                    best_objective = objective
                    best_solution = solution
                    if logger:
                        logger.log_solution(float(objective), solution)

    except (gp.GurobiError, RuntimeError, ValueError):
        # The empty solution remains a valid fallback.
        pass

    best_solution["objective_value"] = float(best_objective)
    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()