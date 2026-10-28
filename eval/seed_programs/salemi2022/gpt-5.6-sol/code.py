import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def write_json(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), allow_nan=False)
    os.replace(tmp_path, path)


def clean_number(v):
    v = float(v)
    if abs(v) < 1e-9:
        return 0.0
    return v


def main():
    start_time = time.monotonic()
    args = parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = int(data["n_generators"])
    S = int(data["n_scenarios"])
    T = int(data["T"])
    gens = data["generators"]
    demand = data["demand_scenarios"]
    reserve = data["reserve_scenarios"]

    M = [float(g["M"]) for g in gens]
    m_min = [float(g["m"]) for g in gens]
    min_up = [int(g["L"]) for g in gens]
    min_down = [int(g["ell"]) for g in gens]
    RU = [float(g["RU"]) for g in gens]
    SU = [float(g["SU"]) for g in gens]
    RD = [float(g["RD"]) for g in gens]
    SD = [float(g["SD"]) for g in gens]
    prod_cost = [float(g["c_g"]) for g in gens]
    fixed_cost = [float(g["c_f"]) for g in gens]
    startup_costs = [
        [float(v) for v in g["startup_costs_K"]] for g in gens
    ]

    def minimal_startup_costs(x_values):
        q_values = [[0.0 for _ in range(T)] for _ in range(G)]
        for i in range(G):
            for t in range(T):
                best = 0.0
                # k is the number of preceding periods included.
                for k in range(1, t + 1):
                    expression = (
                        x_values[i][t]
                        - sum(x_values[i][r] for r in range(t - k, t))
                    )
                    best = max(best, startup_costs[i][k - 1] * expression)
                q_values[i][t] = max(0.0, best)
        return q_values

    def make_solution(x_values, y_values, z_values, p_values):
        x_out = [
            [int(round(x_values[i][t])) for t in range(T)] for i in range(G)
        ]
        y_out = [
            [int(round(y_values[i][t])) for t in range(T)] for i in range(G)
        ]
        z_out = [
            [int(round(z_values[i][t])) for t in range(T)] for i in range(G)
        ]

        p_out = []
        for i in range(G):
            generator_scenarios = []
            for s in range(S):
                trajectory = []
                for t in range(T):
                    if x_out[i][t] == 0:
                        trajectory.append(0.0)
                    else:
                        trajectory.append(clean_number(p_values[i][s][t]))
                generator_scenarios.append(trajectory)
            p_out.append(generator_scenarios)

        q_out = minimal_startup_costs(x_out)

        # Availability is not costly and is not subject to a separate ramp
        # constraint, so choosing the full committed capacity is feasible and
        # dominates every smaller availability assignment.
        p_bar_out = [
            [
                [float(M[i] * x_out[i][t]) for t in range(T)]
                for _ in range(S)
            ]
            for i in range(G)
        ]

        fixed_and_startup = 0.0
        for i in range(G):
            for t in range(T):
                fixed_and_startup += fixed_cost[i] * x_out[i][t]
                fixed_and_startup += q_out[i][t]

        expected_production = 0.0
        if S > 0:
            for i in range(G):
                for s in range(S):
                    for t in range(T):
                        expected_production += prod_cost[i] * p_out[i][s][t]
            expected_production /= S

        objective = clean_number(fixed_and_startup + expected_production)

        return {
            "objective_value": objective,
            "x": x_out,
            "y": y_out,
            "y_bar": z_out,
            "q": q_out,
            "p": p_out,
            "p_bar": p_bar_out,
        }

    # A robust initial solution for feasible instances: commit every generator
    # that can physically start, and operate it at its maximum ramp-reachable
    # output. Because over-generation is allowed and SU <= RU, this trajectory
    # dominates the output capability of later starts.
    warm_x = [[0 for _ in range(T)] for _ in range(G)]
    warm_y = [[0 for _ in range(T)] for _ in range(G)]
    warm_z = [[0 for _ in range(T)] for _ in range(G)]
    warm_p = [[[0.0 for _ in range(T)] for _ in range(S)] for _ in range(G)]
    warm_max_output = [[0.0 for _ in range(T)] for _ in range(G)]

    for i in range(G):
        usable = SU[i] + 1e-9 >= m_min[i]
        if usable:
            for t in range(T):
                warm_x[i][t] = 1
            if T > 0:
                warm_y[i][0] = 1
                warm_max_output[i][0] = min(M[i], SU[i])
                for t in range(1, T):
                    warm_max_output[i][t] = min(
                        M[i], warm_max_output[i][t - 1] + RU[i]
                    )
            for s in range(S):
                for t in range(T):
                    warm_p[i][s][t] = warm_max_output[i][t]

    warm_feasible = True
    for s in range(S):
        for t in range(T):
            total_p = sum(warm_p[i][s][t] for i in range(G))
            total_capacity = sum(M[i] * warm_x[i][t] for i in range(G))
            if total_p + 1e-7 < float(demand[s][t]):
                warm_feasible = False
            if total_capacity + 1e-7 < float(demand[s][t]) + float(reserve[s][t]):
                warm_feasible = False

    fallback_solution = None
    best_logged_objective = math.inf
    if warm_feasible:
        fallback_solution = make_solution(warm_x, warm_y, warm_z, warm_p)
        best_logged_objective = fallback_solution["objective_value"]
        if logger:
            logger.log_solution(
                fallback_solution["objective_value"], fallback_solution
            )

    model = gp.Model("stochastic_unit_commitment")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x = model.addVars(G, T, vtype=GRB.BINARY, name="x")
    y = model.addVars(G, T, vtype=GRB.BINARY, name="y")
    z = model.addVars(G, T, vtype=GRB.BINARY, name="z")
    q = model.addVars(G, T, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    p = model.addVars(G, S, T, lb=0.0, vtype=GRB.CONTINUOUS, name="p")

    # Commitment transitions and mutually exclusive transition indicators.
    for i in range(G):
        for t in range(T):
            previous_x = x[i, t - 1] if t > 0 else 0.0
            model.addConstr(y[i, t] - z[i, t] == x[i, t] - previous_x)
            model.addConstr(y[i, t] + z[i, t] <= 1)

    # Minimum up- and down-time constraints.
    for i in range(G):
        L = min_up[i]
        ell = min_down[i]
        for t in range(T):
            if L > 0:
                model.addConstr(
                    gp.quicksum(
                        y[i, r] for r in range(max(0, t - L + 1), t + 1)
                    )
                    <= x[i, t]
                )
            if ell > 0:
                model.addConstr(
                    gp.quicksum(
                        z[i, r] for r in range(max(0, t - ell + 1), t + 1)
                    )
                    <= 1 - x[i, t]
                )

    # Start-up cost epigraph.
    for i in range(G):
        for t in range(T):
            for k in range(1, t + 1):
                model.addConstr(
                    q[i, t]
                    >= startup_costs[i][k - 1]
                    * (
                        x[i, t]
                        - gp.quicksum(x[i, r] for r in range(t - k, t))
                    )
                )

    # Scenario-dependent production limits and ramping.
    for i in range(G):
        for s in range(S):
            for t in range(T):
                model.addConstr(p[i, s, t] >= m_min[i] * x[i, t])
                model.addConstr(p[i, s, t] <= M[i] * x[i, t])

                previous_p = p[i, s, t - 1] if t > 0 else 0.0
                previous_x = x[i, t - 1] if t > 0 else 0.0

                model.addConstr(
                    p[i, s, t] - previous_p
                    <= RU[i] * previous_x + SU[i] * y[i, t]
                )
                model.addConstr(
                    previous_p - p[i, s, t]
                    <= RD[i] * x[i, t] + SD[i] * z[i, t]
                )

    # Demand and reserve. Full committed capacity may be selected as p_bar, so
    # the reserve constraint can be imposed directly on sum(M_i x_it).
    for s in range(S):
        for t in range(T):
            model.addConstr(
                gp.quicksum(p[i, s, t] for i in range(G))
                >= float(demand[s][t])
            )
            model.addConstr(
                gp.quicksum(M[i] * x[i, t] for i in range(G))
                >= float(demand[s][t]) + float(reserve[s][t])
            )

    fixed_term = gp.quicksum(
        fixed_cost[i] * x[i, t] + q[i, t]
        for i in range(G)
        for t in range(T)
    )
    production_term = gp.quicksum(
        (prod_cost[i] / S) * p[i, s, t]
        for i in range(G)
        for s in range(S)
        for t in range(T)
    ) if S > 0 else 0.0

    model.setObjective(fixed_term + production_term, GRB.MINIMIZE)

    # Supply a complete feasible MIP start whenever the dominance construction
    # succeeded.
    if warm_feasible:
        for i in range(G):
            for t in range(T):
                x[i, t].Start = warm_x[i][t]
                y[i, t].Start = warm_y[i][t]
                z[i, t].Start = warm_z[i][t]
                q[i, t].Start = 0.0
                for s in range(S):
                    p[i, s, t].Start = warm_p[i][s][t]

    x_vars = [x[i, t] for i in range(G) for t in range(T)]
    y_vars = [y[i, t] for i in range(G) for t in range(T)]
    z_vars = [z[i, t] for i in range(G) for t in range(T)]
    p_vars = [
        p[i, s, t]
        for i in range(G)
        for s in range(S)
        for t in range(T)
    ]
    callback_vars = x_vars + y_vars + z_vars + p_vars

    callback_state = {
        "best": best_logged_objective,
    }

    def unpack_solution(values):
        pos = 0
        xv = [[0 for _ in range(T)] for _ in range(G)]
        yv = [[0 for _ in range(T)] for _ in range(G)]
        zv = [[0 for _ in range(T)] for _ in range(G)]
        pv = [[[0.0 for _ in range(T)] for _ in range(S)] for _ in range(G)]

        for i in range(G):
            for t in range(T):
                xv[i][t] = int(round(values[pos]))
                pos += 1
        for i in range(G):
            for t in range(T):
                yv[i][t] = int(round(values[pos]))
                pos += 1
        for i in range(G):
            for t in range(T):
                zv[i][t] = int(round(values[pos]))
                pos += 1
        for i in range(G):
            for s in range(S):
                for t in range(T):
                    pv[i][s][t] = float(values[pos])
                    pos += 1

        return make_solution(xv, yv, zv, pv)

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            values = cb_model.cbGetSolution(callback_vars)
            solution = unpack_solution(values)
            objective = solution["objective_value"]
            tolerance = 1e-7 * max(1.0, abs(callback_state["best"]))
            if objective < callback_state["best"] - tolerance:
                callback_state["best"] = objective
                logger.log_solution(objective, solution)
        except Exception:
            # Logging must not terminate the optimization run.
            pass

    elapsed = time.monotonic() - start_time
    remaining = max(0.0, float(args.time_limit) - elapsed - 0.05)

    final_solution = fallback_solution
    if remaining > 0.0:
        model.Params.TimeLimit = remaining
        try:
            model.optimize(incumbent_callback)
            if model.SolCount > 0:
                final_values = [v.X for v in callback_vars]
                final_solution = unpack_solution(final_values)
        except gp.GurobiError:
            pass

    # The input format has no way to represent infeasibility. This branch only
    # protects schema production for an infeasible instance or an immediate
    # solver failure.
    if final_solution is None:
        zero_x = [[0 for _ in range(T)] for _ in range(G)]
        zero_y = [[0 for _ in range(T)] for _ in range(G)]
        zero_z = [[0 for _ in range(T)] for _ in range(G)]
        zero_p = [[[0.0 for _ in range(T)] for _ in range(S)] for _ in range(G)]
        final_solution = make_solution(zero_x, zero_y, zero_z, zero_p)

    if logger:
        tolerance = 1e-7 * max(1.0, abs(callback_state["best"]))
        if final_solution["objective_value"] < callback_state["best"] - tolerance:
            callback_state["best"] = final_solution["objective_value"]
            logger.log_solution(
                final_solution["objective_value"], final_solution
            )

    write_json(args.solution_path, final_solution)


if __name__ == "__main__":
    main()