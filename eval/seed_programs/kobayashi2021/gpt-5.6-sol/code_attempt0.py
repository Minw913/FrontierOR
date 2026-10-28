import argparse
import json
import math
import os
import time

import numpy as np
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


def optimal_cvar_terms(x, scenarios, probabilities, beta):
    """Return an optimal (a, v) for a fixed portfolio x."""
    losses = -(scenarios @ x)
    order = np.argsort(losses)
    sorted_losses = losses[order]
    sorted_prob = probabilities[order]

    weighted_losses = sorted_prob * sorted_losses
    suffix_prob = np.cumsum(sorted_prob[::-1])[::-1]
    suffix_weighted_losses = np.cumsum(weighted_losses[::-1])[::-1]

    tail_prob_after = np.empty_like(suffix_prob)
    tail_loss_after = np.empty_like(suffix_weighted_losses)
    tail_prob_after[:-1] = suffix_prob[1:]
    tail_loss_after[:-1] = suffix_weighted_losses[1:]
    tail_prob_after[-1] = 0.0
    tail_loss_after[-1] = 0.0

    factor = 1.0 / (1.0 - beta)
    candidate_values = (
        sorted_losses
        + factor
        * (tail_loss_after - sorted_losses * tail_prob_after)
    )
    best_index = int(np.argmin(candidate_values))
    a = float(sorted_losses[best_index])
    v = float(
        factor
        * np.dot(probabilities, np.maximum(losses - a, 0.0))
    )
    v = max(0.0, v)
    return a, v


def make_solution(
    raw_x,
    raw_z,
    mu,
    mu_bar,
    scenarios,
    probabilities,
    beta,
    gamma,
    k,
    fallback=None,
):
    """Clean solver tolerances and create a complete feasible solution."""
    n = len(mu)
    z = np.array([1 if float(value) >= 0.5 else 0 for value in raw_z], dtype=int)

    if int(z.sum()) > k or int(z.sum()) == 0:
        return fallback

    x = np.asarray(raw_x, dtype=float).copy()
    x[~np.isfinite(x)] = 0.0
    x = np.maximum(x, 0.0)
    x[z == 0] = 0.0

    total = float(x.sum())
    if total <= 1e-14:
        return fallback
    x /= total

    expected_return = float(np.dot(mu, x))
    if expected_return < mu_bar:
        selected = np.flatnonzero(z)
        best_asset = int(selected[np.argmax(mu[selected])])
        best_return = float(mu[best_asset])

        if best_return + 1e-10 < mu_bar:
            return fallback

        if best_return > expected_return + 1e-15:
            target = min(best_return, mu_bar + 1e-11)
            blend = (target - expected_return) / (best_return - expected_return)
            blend = min(1.0, max(0.0, blend))
            x *= 1.0 - blend
            x[best_asset] += blend

    # Restore exact budget after any floating-point adjustment.
    x[z == 0] = 0.0
    total = float(x.sum())
    if total <= 0.0:
        return fallback
    x /= total

    if float(np.dot(mu, x)) < mu_bar - 1e-8:
        return fallback

    a, v = optimal_cvar_terms(x, scenarios, probabilities, beta)
    regularization = float(np.dot(x, x)) / (2.0 * gamma)
    objective = regularization + a + v

    if not all(math.isfinite(q) for q in (objective, a, v)):
        return fallback

    return {
        "objective_value": float(objective),
        "x": [float(value) for value in x],
        "z": [int(value) for value in z],
        "a": float(a),
        "v": float(v),
    }


def main():
    args = parse_args()
    start_time = time.monotonic()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["N"])
    s_count = int(data["S"])
    k = min(int(data["k"]), n)
    beta = float(data["beta"])
    gamma = float(data["gamma"])
    mu_bar = float(data["mu_bar"])
    mu = np.asarray(data["mu"], dtype=float)
    scenarios = np.asarray(data["scenarios"], dtype=float)

    p_data = data["p_s"]
    if isinstance(p_data, (list, tuple)):
        probabilities = np.asarray(p_data, dtype=float)
    else:
        probabilities = np.full(s_count, float(p_data), dtype=float)

    if mu.shape != (n,):
        raise ValueError("mu must have length N")
    if scenarios.shape != (s_count, n):
        raise ValueError("scenarios must have shape [S][N]")
    if probabilities.shape != (s_count,):
        raise ValueError("Scenario probabilities must have length S")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be strictly between zero and one")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if k < 1:
        raise ValueError("No feasible portfolio exists when k < 1")

    # A one-asset portfolio in the highest-return asset is feasible whenever
    # the instance itself is feasible under the simplex return constraint.
    warm_asset = int(np.argmax(mu))
    if float(mu[warm_asset]) < mu_bar - 1e-10:
        raise ValueError("The expected-return constraint is infeasible")

    warm_x = np.zeros(n, dtype=float)
    warm_z = np.zeros(n, dtype=int)
    warm_x[warm_asset] = 1.0
    warm_z[warm_asset] = 1

    warm_solution = make_solution(
        warm_x,
        warm_z,
        mu,
        mu_bar,
        scenarios,
        probabilities,
        beta,
        gamma,
        k,
    )
    if warm_solution is None:
        raise RuntimeError("Failed to construct an initial feasible solution")

    best_holder = {
        "objective": warm_solution["objective_value"],
        "solution": warm_solution,
    }

    if logger:
        logger.log_solution(warm_solution["objective_value"], warm_solution)

    model = gp.Model("cardinality_regularized_cvar")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    elapsed = time.monotonic() - start_time
    remaining = max(0.0, float(args.time_limit) - elapsed)
    model.Params.TimeLimit = remaining

    x_vars = model.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    z_vars = model.addVars(n, vtype=GRB.BINARY, name="z")

    # For normalized probabilities, an optimal VaR threshold lies within the
    # attainable scenario-loss range.
    all_asset_losses = -scenarios
    a_lb = float(np.min(all_asset_losses))
    a_ub = float(np.max(all_asset_losses))
    a_var = model.addVar(lb=a_lb, ub=a_ub, vtype=GRB.CONTINUOUS, name="a")
    v_var = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")
    u_vars = model.addVars(
        s_count, lb=0.0, vtype=GRB.CONTINUOUS, name="tail_excess"
    )

    model.addConstr(gp.quicksum(x_vars[i] for i in range(n)) == 1.0, name="budget")
    model.addConstr(
        gp.quicksum(float(mu[i]) * x_vars[i] for i in range(n)) >= mu_bar,
        name="required_return",
    )
    model.addConstr(
        gp.quicksum(z_vars[i] for i in range(n)) <= k,
        name="cardinality",
    )

    for i in range(n):
        model.addConstr(x_vars[i] <= z_vars[i], name=f"selection_link[{i}]")

    for scenario_index in range(s_count):
        loss_expr = -gp.quicksum(
            float(scenarios[scenario_index, i]) * x_vars[i]
            for i in range(n)
        )
        model.addConstr(
            u_vars[scenario_index] >= loss_expr - a_var,
            name=f"excess_definition[{scenario_index}]",
        )

    factor = 1.0 / (1.0 - beta)
    model.addConstr(
        v_var
        >= factor
        * gp.quicksum(
            float(probabilities[j]) * u_vars[j] for j in range(s_count)
        ),
        name="cvar_contribution",
    )

    regularization = gp.quicksum(
        x_vars[i] * x_vars[i] for i in range(n)
    ) / (2.0 * gamma)
    model.setObjective(regularization + a_var + v_var, GRB.MINIMIZE)

    # Supply the guaranteed feasible portfolio as a MIP start.
    warm_a = warm_solution["a"]
    warm_v = warm_solution["v"]
    warm_losses = -(scenarios @ warm_x)

    for i in range(n):
        x_vars[i].Start = float(warm_x[i])
        z_vars[i].Start = int(warm_z[i])
    a_var.Start = warm_a
    v_var.Start = warm_v
    for j in range(s_count):
        u_vars[j].Start = max(0.0, float(warm_losses[j] - warm_a))

    def incumbent_callback(cb_model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            raw_x = cb_model.cbGetSolution([x_vars[i] for i in range(n)])
            raw_z = cb_model.cbGetSolution([z_vars[i] for i in range(n)])

            candidate = make_solution(
                raw_x,
                raw_z,
                mu,
                mu_bar,
                scenarios,
                probabilities,
                beta,
                gamma,
                k,
                fallback=None,
            )
            if candidate is None:
                return

            candidate_obj = candidate["objective_value"]
            tolerance = 1e-10 * max(1.0, abs(best_holder["objective"]))
            if candidate_obj < best_holder["objective"] - tolerance:
                best_holder["objective"] = candidate_obj
                best_holder["solution"] = candidate
                if logger:
                    logger.log_solution(candidate_obj, candidate)
        except Exception:
            # Logging or projection must not interrupt the optimization.
            return

    if remaining > 0.0:
        model.optimize(incumbent_callback)

    if model.SolCount > 0:
        final_raw_x = [x_vars[i].X for i in range(n)]
        final_raw_z = [z_vars[i].X for i in range(n)]
        final_candidate = make_solution(
            final_raw_x,
            final_raw_z,
            mu,
            mu_bar,
            scenarios,
            probabilities,
            beta,
            gamma,
            k,
            fallback=None,
        )
        if (
            final_candidate is not None
            and final_candidate["objective_value"]
            < best_holder["objective"] - 1e-10
        ):
            best_holder["objective"] = final_candidate["objective_value"]
            best_holder["solution"] = final_candidate
            if logger:
                logger.log_solution(
                    final_candidate["objective_value"], final_candidate
                )

    solution = best_holder["solution"]

    output_dir = os.path.dirname(os.path.abspath(args.solution_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()