import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve the crew rotation exact-cover problem with Gurobi."
    )
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON.")
    parser.add_argument("--solution_path", required=True, help="Path for solution JSON.")
    parser.add_argument("--time_limit", required=True, type=int, help="Runtime limit in seconds.")
    parser.add_argument("--log_path", default=None, help="Optional incumbent JSONL log path.")
    return parser.parse_args()


def make_solution(values, costs):
    selected = [j for j, value in enumerate(values) if value > 0.5]
    objective = float(sum(costs[j] for j in selected))
    variable_values = {
        str(j): (1.0 if values[j] > 0.5 else 0.0)
        for j in range(len(values))
    }
    return objective, {
        "objective_value": objective,
        "selected_rotations": selected,
        "variable_values": variable_values,
    }


def load_and_validate(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dimensions = data["dimensions"]
    num_rows = int(dimensions["num_rows"])
    num_cols = int(dimensions["num_cols"])

    if num_rows < 0 or num_cols < 0:
        raise ValueError("num_rows and num_cols must be nonnegative")

    costs = data["cost_vector"]
    columns = data["constraint_matrix_A"]["columns"]

    if len(costs) != num_cols:
        raise ValueError("cost_vector length does not equal num_cols")
    if len(columns) != num_cols:
        raise ValueError("constraint_matrix_A.columns length does not equal num_cols")

    costs = [int(value) for value in costs]
    normalized_columns = []

    for j, column in enumerate(columns):
        seen = set()
        normalized = []
        for raw_index in column:
            row = int(raw_index)
            if row < 0 or row >= num_rows:
                raise ValueError(f"Invalid flight-leg index {row} in rotation {j}")
            if row in seen:
                raise ValueError(f"Duplicate flight-leg index {row} in rotation {j}")
            seen.add(row)
            normalized.append(row)
        normalized_columns.append(normalized)

    base_rows = []
    if bool(data.get("has_base_constraints", False)):
        base_data = data["base_constraints"]
        num_bases = int(base_data["num_bases"])
        d_rows = base_data["D_matrix"]["rows"]
        lower_bounds = base_data["lower_bounds_d1"]
        upper_bounds = base_data["upper_bounds_d2"]

        if len(d_rows) != num_bases:
            raise ValueError("D_matrix row count does not equal num_bases")
        if len(lower_bounds) != num_bases:
            raise ValueError("lower_bounds_d1 length does not equal num_bases")
        if len(upper_bounds) != num_bases:
            raise ValueError("upper_bounds_d2 length does not equal num_bases")

        for b in range(num_bases):
            if len(d_rows[b]) != num_cols:
                raise ValueError(f"D_matrix row {b} length does not equal num_cols")

            lower = float(lower_bounds[b])
            upper = float(upper_bounds[b])
            if lower > upper:
                raise ValueError(f"Base constraint {b} has lower bound above upper bound")

            coefficients = [float(value) for value in d_rows[b]]
            base_rows.append((coefficients, lower, upper))

    return num_rows, num_cols, costs, normalized_columns, base_rows


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    num_rows, num_cols, costs, columns, base_rows = load_and_validate(
        args.instance_path
    )

    model = gp.Model("crew_rotation_set_partitioning")
    model.ModelSense = GRB.MINIMIZE

    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    # Constraints are created first so each variable can be inserted efficiently
    # with a sparse Gurobi Column.
    cover_constraints = [
        model.addLConstr(gp.LinExpr(), GRB.EQUAL, 1.0)
        for _ in range(num_rows)
    ]

    base_constraint_pairs = []
    for _, lower, upper in base_rows:
        lower_constraint = model.addLConstr(
            gp.LinExpr(), GRB.GREATER_EQUAL, lower
        )
        upper_constraint = model.addLConstr(
            gp.LinExpr(), GRB.LESS_EQUAL, upper
        )
        base_constraint_pairs.append((lower_constraint, upper_constraint))

    model.update()

    variables = []
    for j in range(num_cols):
        coefficients = []
        constraints = []

        for row in columns[j]:
            coefficients.append(1.0)
            constraints.append(cover_constraints[row])

        for b, (d_row, _, _) in enumerate(base_rows):
            coefficient = d_row[j]
            if coefficient != 0.0:
                lower_constraint, upper_constraint = base_constraint_pairs[b]
                coefficients.extend((coefficient, coefficient))
                constraints.extend((lower_constraint, upper_constraint))

        column = gp.Column(coefficients, constraints)
        variable = model.addVar(
            lb=0.0,
            ub=1.0,
            obj=float(costs[j]),
            vtype=GRB.BINARY,
            column=column,
        )
        variables.append(variable)

    model.update()

    elapsed = time.monotonic() - start_time
    model.Params.TimeLimit = max(0.0, float(args.time_limit) - elapsed)

    incumbent_state = {
        "objective": float("inf"),
        "solution": None,
    }

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return

        try:
            raw_values = cb_model.cbGetSolution(variables)
            objective, solution = make_solution(raw_values, costs)

            if objective < incumbent_state["objective"] - 1e-9:
                incumbent_state["objective"] = objective
                incumbent_state["solution"] = solution
                if logger:
                    logger.log_solution(objective, solution)
        except Exception:
            # Logging or incumbent extraction must not interrupt optimization.
            pass

    model.optimize(incumbent_callback)

    if model.SolCount <= 0:
        raise RuntimeError(
            "No feasible solution was found within the supplied time limit."
        )

    final_values = [variable.X for variable in variables]
    final_objective, final_solution = make_solution(final_values, costs)

    # Normally the callback has already logged this incumbent. This also covers
    # solver outcomes where a final incumbent was not exposed through MIPSOL.
    if final_objective < incumbent_state["objective"] - 1e-9:
        incumbent_state["objective"] = final_objective
        incumbent_state["solution"] = final_solution
        if logger:
            logger.log_solution(final_objective, final_solution)

    solution_directory = os.path.dirname(os.path.abspath(args.solution_path))
    if solution_directory:
        os.makedirs(solution_directory, exist_ok=True)

    temporary_path = args.solution_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f, separators=(",", ":"), allow_nan=False)
    os.replace(temporary_path, args.solution_path)


if __name__ == "__main__":
    main()