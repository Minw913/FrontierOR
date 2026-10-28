import argparse
import json
import time
from typing import List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def set_gurobi_params(model: gp.Model, time_limit: Optional[float] = None):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = max(0.01, float(time_limit))


def compute_x_upper_bounds(
    n1: int, A1: List[List[int]], b1: List[int], A2: List[List[int]], b2: List[int]
) -> List[int]:
    # Since y >= 0 and B coefficients are nonnegative, necessary conditions include:
    # A1 x <= b1 and A2 x <= b2 (taking y=0 as best case for feasibility).
    # Use these to derive variable-wise upper bounds.
    ub_default = max([130] + b1 + b2)
    ubs = [ub_default for _ in range(n1)]

    def tighten_with(A, b):
        m = len(A)
        for i in range(n1):
            best = None
            for r in range(m):
                a = A[r][i]
                if a > 0:
                    val = b[r] // a
                    if best is None or val < best:
                        best = val
            if best is not None:
                ubs[i] = min(ubs[i], int(best))

    tighten_with(A1, b1)
    tighten_with(A2, b2)

    # Ensure nonnegative and at least 0
    ubs = [max(0, int(u)) for u in ubs]
    return ubs


def build_follower_model(
    n2: int,
    int_follower_zero_based: set,
    B_rows: List[List[int]],
    rhs_base: List[int],
    d2: List[int],
):
    model = gp.Model("follower")
    y = []
    for j in range(n2):
        vtype = GRB.INTEGER if j in int_follower_zero_based else GRB.CONTINUOUS
        yj = model.addVar(lb=0.0, vtype=vtype, name=f"y_{j}")
        y.append(yj)

    constrs = []
    for r in range(len(B_rows)):
        expr = gp.quicksum(B_rows[r][j] * y[j] for j in range(n2))
        c = model.addConstr(expr <= rhs_base[r], name=f"row_{r}")
        constrs.append(c)

    obj = gp.quicksum(d2[j] * y[j] for j in range(n2))
    model.setObjective(obj, GRB.MAXIMIZE)
    return model, y, constrs


def update_rhs_from_x(constrs, A_rows, b_vec, x_vals):
    for r, c in enumerate(constrs):
        rhs = b_vec[r] - sum(A_rows[r][i] * x_vals[i] for i in range(len(x_vals)))
        c.RHS = rhs


def evaluate_x(
    x_vals: List[int],
    low_model: gp.Model,
    low_constrs,
    both_model: gp.Model,
    both_constrs_low,
    both_constrs_up,
    A2,
    b2,
    A1,
    b1,
    d2,
    low_y_vars,
    both_y_vars,
    remaining_time_fn,
    tol=1e-5,
):
    # Solve follower lower-level optimal value z_low(x)
    update_rhs_from_x(low_constrs, A2, b2, x_vals)
    rem = remaining_time_fn()
    if rem <= 0:
        return None
    set_gurobi_params(low_model, rem)
    low_model.optimize()

    if low_model.SolCount == 0:
        return {
            "status": "infeasible_lower",
        }

    z_low = low_model.ObjVal
    low_status = low_model.Status

    # If not proven optimal (time-limited), treat as uncertain
    if low_status not in (GRB.OPTIMAL,):
        return {
            "status": "uncertain",
        }

    # Solve max follower objective with both upper+lower constraints
    update_rhs_from_x(both_constrs_low, A2, b2, x_vals)
    update_rhs_from_x(both_constrs_up, A1, b1, x_vals)
    rem = remaining_time_fn()
    if rem <= 0:
        return None
    set_gurobi_params(both_model, rem)
    both_model.optimize()

    if both_model.SolCount == 0:
        return {
            "status": "invalid_no_upper_feasible",
            "z_low": z_low,
        }

    z_both = both_model.ObjVal
    both_status = both_model.Status
    if both_status not in (GRB.OPTIMAL,):
        return {
            "status": "uncertain",
            "z_low": z_low,
        }

    if z_both >= z_low - tol:
        y_vals = [both_y_vars[j].X for j in range(len(both_y_vars))]
        return {
            "status": "feasible",
            "z_low": z_low,
            "z_both": z_both,
            "y": y_vals,
        }
    else:
        return {
            "status": "invalid_not_follower_optimal_for_upper",
            "z_low": z_low,
            "z_both": z_both,
        }


def add_no_good_cut_for_x(master: gp.Model, x_vars, x_star: List[int], cut_id: int):
    # Exclude exact integer vector x = x_star
    n1 = len(x_vars)
    eq_bins = []
    for i in range(n1):
        b = master.addVar(vtype=GRB.BINARY, name=f"ng_{cut_id}_{i}")
        # if b=1 => x_i == x_star_i
        master.addGenConstrIndicator(b, True, x_vars[i] <= x_star[i], name=f"ngi_le_{cut_id}_{i}")
        master.addGenConstrIndicator(b, True, x_vars[i] >= x_star[i], name=f"ngi_ge_{cut_id}_{i}")
        eq_bins.append(b)
    master.addConstr(gp.quicksum(eq_bins) <= n1 - 1, name=f"nogood_{cut_id}")
    master.update()


def build_master(
    data,
    x_ubs: List[int],
    int_follower_zero_based: set,
):
    n1 = data["n1"]
    n2 = data["n2"]
    m1 = data["m1"]
    m2 = data["m2"]

    c = data["c"]
    A1 = data["A1"]
    B1 = data["B1"]
    b1 = data["b1"]
    A2 = data["A2"]
    B2 = data["B2"]
    b2 = data["b2"]

    model = gp.Model("master")
    x_vars = []
    for i in range(n1):
        if data.get("all_leader_vars_integer", True):
            xi = model.addVar(lb=0.0, ub=x_ubs[i], vtype=GRB.INTEGER, name=f"x_{i}")
        else:
            xi = model.addVar(lb=0.0, ub=x_ubs[i], vtype=GRB.CONTINUOUS, name=f"x_{i}")
        x_vars.append(xi)

    y_vars = []
    for j in range(n2):
        vtype = GRB.INTEGER if j in int_follower_zero_based else GRB.CONTINUOUS
        yj = model.addVar(lb=0.0, vtype=vtype, name=f"y_{j}")
        y_vars.append(yj)

    # Upper-level constraints
    for r in range(m1):
        expr = gp.quicksum(A1[r][i] * x_vars[i] for i in range(n1)) + gp.quicksum(B1[r][j] * y_vars[j] for j in range(n2))
        model.addConstr(expr <= b1[r], name=f"up_{r}")

    # Lower-level constraints
    for r in range(m2):
        expr = gp.quicksum(A2[r][i] * x_vars[i] for i in range(n1)) + gp.quicksum(B2[r][j] * y_vars[j] for j in range(n2))
        model.addConstr(expr <= b2[r], name=f"low_{r}")

    obj = gp.quicksum(c[i] * x_vars[i] for i in range(n1))
    model.setObjective(obj, GRB.MAXIMIZE)
    model.update()
    return model, x_vars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False, default=None)
    args = parser.parse_args()

    start_time = time.time()

    def remaining_time():
        return args.time_limit - (time.time() - start_time)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n1 = data["n1"]
    n2 = data["n2"]
    c = data["c"]
    d2 = data["d2"]
    A1 = data["A1"]
    B1 = data["B1"]
    b1 = data["b1"]
    A2 = data["A2"]
    B2 = data["B2"]
    b2 = data["b2"]

    int_followers = set(idx - 1 for idx in data["integer_follower_vars"])  # one-based -> zero-based

    # Bounds for x to keep master stable and finite
    x_ubs = compute_x_upper_bounds(n1, A1, b1, A2, b2)

    # Build models
    master, x_vars = build_master(data, x_ubs, int_followers)
    set_gurobi_params(master)

    low_model, low_y, low_constrs = build_follower_model(n2, int_followers, B2, b2, d2)
    both_B = B2  # lower constraints in both-model
    both_model = gp.Model("follower_with_upper")
    both_y = []
    for j in range(n2):
        vtype = GRB.INTEGER if j in int_followers else GRB.CONTINUOUS
        both_y.append(both_model.addVar(lb=0.0, vtype=vtype, name=f"y_{j}"))
    both_constrs_low = []
    for r in range(len(B2)):
        expr = gp.quicksum(B2[r][j] * both_y[j] for j in range(n2))
        both_constrs_low.append(both_model.addConstr(expr <= b2[r], name=f"low_{r}"))
    both_constrs_up = []
    for r in range(len(B1)):
        expr = gp.quicksum(B1[r][j] * both_y[j] for j in range(n2))
        both_constrs_up.append(both_model.addConstr(expr <= b1[r], name=f"up_{r}"))
    both_model.setObjective(gp.quicksum(d2[j] * both_y[j] for j in range(n2)), GRB.MAXIMIZE)

    set_gurobi_params(low_model)
    set_gurobi_params(both_model)

    best_solution = None
    best_obj = -float("inf")

    cut_count = 0

    # Main loop
    while remaining_time() > 0.05:
        rem = remaining_time()
        set_gurobi_params(master, rem)
        master.optimize()

        if master.SolCount == 0:
            break

        x_star = [int(round(x_vars[i].X)) for i in range(n1)]
        x_obj = float(sum(c[i] * x_star[i] for i in range(n1)))

        eval_res = evaluate_x(
            x_star,
            low_model,
            low_constrs,
            both_model,
            both_constrs_low,
            both_constrs_up,
            A2,
            b2,
            A1,
            b1,
            d2,
            low_y,
            both_y,
            remaining_time,
        )

        if eval_res is None:
            break

        if eval_res["status"] == "feasible":
            if x_obj > best_obj + 1e-9:
                best_obj = x_obj
                y_star = eval_res["y"]
                follower_obj = float(sum(d2[j] * y_star[j] for j in range(n2)))
                best_solution = {
                    "objective_value": float(best_obj),
                    "x": [float(v) for v in x_star],
                    "y": [float(v) for v in y_star],
                    "follower_objective_value": follower_obj,
                }
                if logger:
                    logger.log_solution(best_obj, best_solution)

            # If master is solved to optimality and this top candidate is bilevel-feasible,
            # then this is globally optimal.
            if master.Status == GRB.OPTIMAL:
                break

            # Otherwise keep searching for possibly better (if master was time-limited).
            add_no_good_cut_for_x(master, x_vars, x_star, cut_count)
            cut_count += 1

        elif eval_res["status"] in (
            "invalid_no_upper_feasible",
            "invalid_not_follower_optimal_for_upper",
            "infeasible_lower",
        ):
            add_no_good_cut_for_x(master, x_vars, x_star, cut_count)
            cut_count += 1
        else:
            # uncertain (typically follower solve time-limited)
            break

        # Bound-based early stop if possible
        if best_solution is not None and master.SolCount > 0:
            try:
                if master.ObjBound <= best_obj + 1e-6:
                    break
            except Exception:
                pass

    # Fallback if nothing found: try x=0
    if best_solution is None and remaining_time() > 0.05:
        x_zero = [0 for _ in range(n1)]
        eval_res = evaluate_x(
            x_zero,
            low_model,
            low_constrs,
            both_model,
            both_constrs_low,
            both_constrs_up,
            A2,
            b2,
            A1,
            b1,
            d2,
            low_y,
            both_y,
            remaining_time,
        )
        if eval_res is not None and eval_res.get("status") == "feasible":
            y_star = eval_res["y"]
            obj = float(sum(c[i] * x_zero[i] for i in range(n1)))
            fobj = float(sum(d2[j] * y_star[j] for j in range(n2)))
            best_solution = {
                "objective_value": obj,
                "x": [float(v) for v in x_zero],
                "y": [float(v) for v in y_star],
                "follower_objective_value": fobj,
            }
            if logger:
                logger.log_solution(obj, best_solution)

    # Absolute fallback to schema-valid output
    if best_solution is None:
        best_solution = {
            "objective_value": float(-1e18),
            "x": [0.0 for _ in range(n1)],
            "y": [0.0 for _ in range(n2)],
            "follower_objective_value": 0.0,
        }

    with open(args.solution_path, "w") as f:
        json.dump(best_solution, f)


if __name__ == "__main__":
    main()