import argparse
import json
import os
import sys

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_cumulative_breakpoints(T, NT, theta, lambdas):
    """Breakpoints of the cumulative arrival function A(t)."""
    xpts = []
    ypts = []
    cum = 0.0
    xpts.append(0.0)
    ypts.append(0.0)
    for p in range(NT):
        end = min((p + 1) * theta, T)
        if end <= xpts[-1]:
            continue
        cum += lambdas[p] * (end - p * theta) if end < (p + 1) * theta else lambdas[p] * theta
        xpts.append(float(end))
        ypts.append(cum)
    if xpts[-1] < T:
        xpts.append(float(T))
        ypts.append(cum)
    return xpts, ypts, cum


def interpolate(xpts, ypts, t):
    """Piecewise-linear interpolation of cumulative arrivals A(t)."""
    if t <= xpts[0]:
        return ypts[0]
    if t >= xpts[-1]:
        return ypts[-1]
    lo, hi = 0, len(xpts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xpts[mid] <= t:
            lo = mid
        else:
            hi = mid
    x0, x1 = xpts[lo], xpts[hi]
    y0, y1 = ypts[lo], ypts[hi]
    if x1 == x0:
        return y1
    return y0 + (y1 - y0) * (t - x0) / (x1 - x0)


def fallback_schedule(T, D, h_min, h_max):
    """A simple feasible schedule: all interior headways = h_min, slack dumped
    into the (unbounded) initial/final intervals."""
    if D <= 1:
        return [T / 2.0] if D == 1 else []
    total_int = (D - 1) * h_min
    if total_int <= T:
        # spread the free slack evenly between initial and final intervals
        slack = T - total_int
        start = slack / 2.0
        return [start + (i) * h_min for i in range(D)]
    # infeasible with h_min; fall back to equal spacing
    return [T * (i + 1) / float(D + 1) for i in range(D)]


def simulate_objective(schedule, T, D, C, xpts, ypts):
    """Exact objective for a given departure schedule (used for fallback)."""
    t = [0.0] + list(schedule) + [float(T)]
    alphas = [0.0] + [interpolate(xpts, ypts, ti) for ti in schedule] + [ypts[-1]]
    total = 0.0
    w = 0.0  # waiting after last departure
    for j in range(D + 1):
        h = t[j + 1] - t[j]
        a = alphas[j + 1] - alphas[j]
        w_start = w if j >= 1 else 0.0
        total += h * (w_start + 0.5 * a)
        if j < D:
            avail = w + a
            b = min(C, avail)
            w = avail - b
    return total


def solve(instance, time_limit, logger):
    params = instance["parameters"]
    T = params["T_planning_horizon_minutes"]
    D = params["D_num_train_services"]
    NT = params["NT_num_demand_periods"]
    theta = params["theta_period_length_minutes"]
    h_min = params["h_min_minutes"]
    h_max = params["h_max_minutes"]
    C = params["C_train_capacity_passengers"]

    demand = sorted(instance["demand_periods"], key=lambda d: d["period"])
    lambdas = [float(d["lambda_pax_per_min"]) for d in demand]
    while len(lambdas) < NT:
        lambdas.append(0.0)

    xpts, ypts, total_arrivals = build_cumulative_breakpoints(T, NT, theta, lambdas)

    model = gp.Model("train_scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.NonConvex = 2
    if time_limit and time_limit > 0:
        model.Params.TimeLimit = time_limit

    # Departure time variables t_1..t_D (t_0 = 0, t_{D+1} = T are constants)
    t_vars = model.addVars(D, lb=0.0, ub=float(T), name="t")
    t_full = [0.0] + [t_vars[i] for i in range(D)] + [float(T)]

    # Headway variables h_0..h_D
    h_vars = []
    for j in range(D + 1):
        if j == 0 or j == D:
            lb, ub = 0.0, float(T)
        else:
            lb, ub = float(h_min), float(h_max)
        h_vars.append(model.addVar(lb=lb, ub=ub, name=f"h_{j}"))
    for j in range(D + 1):
        model.addConstr(h_vars[j] == t_full[j + 1] - t_full[j], name=f"headway_{j}")

    # Cumulative arrivals alpha_i = A(t_i) via piecewise-linear constraints
    alpha_vars = model.addVars(D, lb=0.0, ub=float(total_arrivals), name="alpha")
    for i in range(D):
        model.addGenConstrPWL(t_vars[i], alpha_vars[i], xpts, ypts,
                              name=f"pwl_alpha_{i}")
    alpha_full = [0.0] + [alpha_vars[i] for i in range(D)] + [float(total_arrivals)]

    # Arrivals per interval
    a_vars = model.addVars(D + 1, lb=0.0, ub=float(total_arrivals), name="a")
    for j in range(D + 1):
        model.addConstr(a_vars[j] == alpha_full[j + 1] - alpha_full[j],
                        name=f"arrivals_{j}")

    # Waiting after departure j, and boarding variables
    w_vars = model.addVars(D, lb=0.0, ub=float(total_arrivals), name="w")
    b_vars = model.addVars(D, lb=0.0, ub=float(C), name="b")
    for j in range(1, D + 1):
        avail = (w_vars[j - 1] if j >= 2 else 0.0) + a_vars[j - 1]
        model.addConstr(b_vars[j - 1] <= avail, name=f"board_cap_{j}")
        model.addConstr(w_vars[j - 1] == avail - b_vars[j - 1],
                        name=f"wait_{j}")

    # Objective: sum_j h_j * (w_start_j + 0.5 * a_j)
    obj = gp.QuadExpr()
    for j in range(D + 1):
        w_start = w_vars[j - 1] if j >= 1 else 0.0
        obj += h_vars[j] * (w_start + 0.5 * a_vars[j])
    model.setObjective(obj, GRB.MINIMIZE)

    # Provide a feasible MIP start (equal / min-headway schedule)
    fallback = fallback_schedule(T, D, h_min, h_max)
    for i in range(D):
        t_vars[i].Start = fallback[i]

    t_list = [t_vars[i] for i in range(D)]

    def make_solution_dict(obj_val, t_values):
        return {
            "objective_value": float(obj_val),
            "departure_schedule_minutes": {
                f"train_{i + 1}": float(t_values[i]) for i in range(D)
            },
        }

    def callback(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                vals = m.cbGetSolution(t_list)
                sol = make_solution_dict(obj_val, vals)
                logger.log_solution(obj_val, sol)
            except Exception:
                pass

    model.optimize(callback if logger is not None else None)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        t_values = [t_vars[i].X for i in range(D)]
        solution = make_solution_dict(obj_val, t_values)
    else:
        # Fallback heuristic if the solver produced nothing
        t_values = fallback
        obj_val = simulate_objective(t_values, T, D, C, xpts, ypts)
        solution = make_solution_dict(obj_val, t_values)
        if logger is not None:
            logger.log_solution(obj_val, solution)

    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()