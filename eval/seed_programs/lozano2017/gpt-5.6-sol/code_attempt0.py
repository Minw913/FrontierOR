import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def configure_model(model, time_limit):
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.MIPGapAbs = 1e-7
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.FeasibilityTol = 1e-8
    model.Params.IntFeasTol = 1e-8
    model.Params.TimeLimit = max(0.01, float(time_limit))


def dot(coefficients, values):
    return sum(float(a) * float(v) for a, v in zip(coefficients, values))


def remaining_time(deadline):
    return max(0.0, deadline - time.monotonic())


def clean_integer(value):
    return float(round(value))


def clean_continuous(value):
    if abs(value) <= 1e-9:
        return 0.0
    return float(value)


def make_solution(data, x_values, y_values):
    integer_y = {int(i) - 1 for i in data["integer_follower_vars"]}

    x_clean = [clean_integer(v) for v in x_values]
    y_clean = []
    for j, value in enumerate(y_values):
        if j in integer_y:
            y_clean.append(clean_integer(value))
        else:
            y_clean.append(clean_continuous(value))

    leader_value = dot(data["c"], x_clean)
    follower_value = dot(data["d2"], y_clean)

    # Both are emitted as JSON floats, as required by the output schema.
    return {
        "objective_value": float(leader_value),
        "x": x_clean,
        "y": y_clean,
        "follower_objective_value": float(follower_value),
    }


def compute_variable_upper_bounds(data):
    n1 = data["n1"]
    n2 = data["n2"]

    x_ub = [GRB.INFINITY] * n1
    y_ub = [GRB.INFINITY] * n2

    for A, B, b in (
        (data["A1"], data["B1"], data["b1"]),
        (data["A2"], data["B2"], data["b2"]),
    ):
        for i, rhs in enumerate(b):
            for j in range(n1):
                coefficient = A[i][j]
                if coefficient > 0:
                    x_ub[j] = min(x_ub[j], math.floor(rhs / coefficient))
            for j in range(n2):
                coefficient = B[i][j]
                if coefficient > 0:
                    y_ub[j] = min(y_ub[j], float(rhs) / coefficient)

    return x_ub, y_ub


def build_master(data, time_limit):
    n1 = data["n1"]
    n2 = data["n2"]
    integer_y = {int(i) - 1 for i in data["integer_follower_vars"]}
    x_ub, y_ub = compute_variable_upper_bounds(data)

    model = gp.Model("optimistic_bilevel_master")
    configure_model(model, time_limit)

    x = []
    for j in range(n1):
        ub = x_ub[j]
        x.append(
            model.addVar(
                lb=0.0,
                ub=ub,
                vtype=GRB.INTEGER,
                name=f"x_{j}",
            )
        )

    y = []
    for j in range(n2):
        vtype = GRB.INTEGER if j in integer_y else GRB.CONTINUOUS
        y.append(
            model.addVar(
                lb=0.0,
                ub=y_ub[j],
                vtype=vtype,
                name=f"y_{j}",
            )
        )

    for i in range(data["m1"]):
        model.addConstr(
            gp.quicksum(data["A1"][i][j] * x[j] for j in range(n1))
            + gp.quicksum(data["B1"][i][j] * y[j] for j in range(n2))
            <= data["b1"][i],
            name=f"upper_{i}",
        )

    for i in range(data["m2"]):
        model.addConstr(
            gp.quicksum(data["A2"][i][j] * x[j] for j in range(n1))
            + gp.quicksum(data["B2"][i][j] * y[j] for j in range(n2))
            <= data["b2"][i],
            name=f"lower_{i}",
        )

    leader_expr = gp.quicksum(data["c"][j] * x[j] for j in range(n1))
    follower_expr = gp.quicksum(data["d2"][j] * y[j] for j in range(n2))
    model.setObjective(leader_expr, GRB.MAXIMIZE)
    model.update()

    return model, x, y, leader_expr, follower_expr


def solve_follower(data, x_values, y_start, time_limit):
    """
    Solve the follower problem for fixed x.

    Returns a dictionary containing:
      - incumbent follower vector and value, if one was found;
      - an upper bound on the follower optimum;
      - whether optimality was certified to numerical tolerance.
    """
    n2 = data["n2"]
    integer_y = {int(i) - 1 for i in data["integer_follower_vars"]}

    model = gp.Model("follower")
    configure_model(model, time_limit)

    y = []
    for j in range(n2):
        vtype = GRB.INTEGER if j in integer_y else GRB.CONTINUOUS
        var = model.addVar(lb=0.0, vtype=vtype, name=f"fy_{j}")
        if y_start is not None:
            var.Start = max(0.0, float(y_start[j]))
        y.append(var)

    for i in range(data["m2"]):
        residual = float(data["b2"][i]) - sum(
            data["A2"][i][j] * x_values[j] for j in range(data["n1"])
        )
        model.addConstr(
            gp.quicksum(data["B2"][i][j] * y[j] for j in range(n2))
            <= residual,
            name=f"follower_constraint_{i}",
        )

    objective = gp.quicksum(data["d2"][j] * y[j] for j in range(n2))
    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()

    result = {
        "has_solution": model.SolCount > 0,
        "certified": False,
        "status": model.Status,
        "y": None,
        "value": None,
        "upper_bound": None,
    }

    if model.SolCount > 0:
        result["y"] = [var.X for var in y]
        result["value"] = float(model.ObjVal)

    if model.Status == GRB.OPTIMAL:
        result["certified"] = True
        result["upper_bound"] = float(model.ObjVal)
    elif model.SolCount > 0:
        try:
            result["upper_bound"] = float(model.ObjBound)
            tolerance = 1e-6 * (1.0 + abs(result["value"]))
            if result["upper_bound"] <= result["value"] + tolerance:
                result["certified"] = True
        except gp.GurobiError:
            pass

    model.dispose()
    return result


def add_follower_witness_cut(
    master,
    data,
    x_vars,
    y_vars,
    follower_expr,
    witness_y,
    witness_value,
    cut_number,
):
    """
    Add the globally valid disjunction

        d2*y >= d2*witness
        OR
        witness is infeasible for the selected x.

    Since A2*x is integral, infeasibility of a witness in row i can be
    represented by an integral threshold on A2[i]*x.
    """
    alternatives = []

    value_branch = master.addVar(vtype=GRB.BINARY, name=f"cut_{cut_number}_value")
    alternatives.append(value_branch)
    master.addGenConstrIndicator(
        value_branch,
        True,
        follower_expr >= float(witness_value) - 1e-7,
        name=f"cut_{cut_number}_value_indicator",
    )

    for i in range(data["m2"]):
        threshold = float(data["b2"][i]) - sum(
            data["B2"][i][j] * witness_y[j] for j in range(data["n2"])
        )

        nearest = round(threshold)
        if abs(threshold - nearest) <= 1e-6:
            violation_lhs = int(nearest) + 1
        else:
            violation_lhs = math.floor(threshold) + 1

        row_branch = master.addVar(
            vtype=GRB.BINARY,
            name=f"cut_{cut_number}_row_{i}",
        )
        alternatives.append(row_branch)

        lhs = gp.quicksum(
            data["A2"][i][j] * x_vars[j] for j in range(data["n1"])
        )
        master.addGenConstrIndicator(
            row_branch,
            True,
            lhs >= violation_lhs,
            name=f"cut_{cut_number}_row_indicator_{i}",
        )

    master.addConstr(
        gp.quicksum(alternatives) >= 1,
        name=f"cut_{cut_number}_disjunction",
    )
    master.update()


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(solution, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not data.get("leader_follower_objective_coefficients_fixed_zero", False):
        raise ValueError(
            "This solver expects leader_follower_objective_coefficients_fixed_zero=true"
        )

    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    build_budget = max(0.05, remaining_time(deadline))
    master, x_vars, y_vars, leader_expr, follower_expr = build_master(
        data, build_budget
    )

    best_solution = None
    best_objective = -math.inf
    cut_number = 0
    stalled_iterations = 0

    # Leave a small margin for serialization and cleanup.
    while remaining_time(deadline) > 0.15:
        remaining = remaining_time(deadline)

        # Short repeated master solves usually produce useful candidates much
        # sooner than spending the entire budget proving one relaxation.
        master_slice = min(30.0, max(0.10, remaining * 0.45))
        configure_model(master, master_slice)
        master.optimize()

        if master.SolCount == 0:
            if master.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                break
            stalled_iterations += 1
            if stalled_iterations >= 3:
                break
            continue

        x_candidate = [float(round(var.X)) for var in x_vars]
        y_candidate = [float(var.X) for var in y_vars]
        master_was_optimal = master.Status == GRB.OPTIMAL

        remaining = remaining_time(deadline)
        if remaining <= 0.10:
            break

        follower_slice = min(30.0, max(0.08, remaining * 0.55))
        follower_result = solve_follower(
            data,
            x_candidate,
            y_candidate,
            follower_slice,
        )

        candidate_follower_value = dot(data["d2"], y_candidate)
        tolerance = 1e-5 * (1.0 + abs(candidate_follower_value))

        candidate_is_optimal_response = False
        if follower_result["certified"]:
            upper_bound = follower_result["upper_bound"]
            if upper_bound is not None:
                candidate_is_optimal_response = (
                    upper_bound <= candidate_follower_value + tolerance
                )

        if candidate_is_optimal_response:
            solution = make_solution(data, x_candidate, y_candidate)
            objective = solution["objective_value"]

            if objective > best_objective + 1e-7:
                best_objective = objective
                best_solution = solution
                if logger:
                    logger.log_solution(objective, solution)

            # An optimal master solution that is also bilevel feasible proves
            # global optimality because the master is a relaxation.
            if master_was_optimal:
                break

            # Leader objective values are integral. Search only for a strict
            # improvement over the incumbent.
            master.addConstr(
                leader_expr >= int(round(best_objective)) + 1,
                name=f"incumbent_cutoff_{cut_number}",
            )
            master.update()
            stalled_iterations = 0
            continue

        # Any feasible follower witness yields a globally valid value-function
        # cut, even if the follower solve timed out before proving optimality.
        if (
            follower_result["has_solution"]
            and follower_result["value"] is not None
            and follower_result["value"] > candidate_follower_value + tolerance
        ):
            add_follower_witness_cut(
                master,
                data,
                x_vars,
                y_vars,
                follower_expr,
                follower_result["y"],
                follower_result["value"],
                cut_number,
            )
            cut_number += 1
            stalled_iterations = 0
            continue

        # The follower solve did not prove the submitted y optimal and did not
        # find a strictly better witness. Retry while time remains, but avoid
        # looping indefinitely under severe time pressure.
        stalled_iterations += 1
        if stalled_iterations >= 3:
            break

    # Extremely short limits may expire before a certified incumbent is found.
    # Emit a schema-valid fallback rather than failing to create an output file.
    if best_solution is None:
        x_fallback = [0.0] * data["n1"]
        y_fallback = [0.0] * data["n2"]
        best_solution = make_solution(data, x_fallback, y_fallback)

    write_solution(args.solution_path, best_solution)
    master.dispose()


if __name__ == "__main__":
    main()