import argparse
import itertools
import json
import math
import os
import random
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from solution_logger import SolutionLogger


class TimeLimitReached(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def read_linear_rules(instance, n):
    constraints = instance.get("constraints", {})
    rules = (
        constraints.get("linear_constraints")
        or constraints.get("rules")
        or instance.get("linear_constraints")
        or []
    )

    parsed = []
    for rule in rules:
        coefficients = rule.get("coefficients", rule.get("a"))
        if coefficients is None:
            continue
        if len(coefficients) != n:
            raise ValueError("Linear-constraint coefficient vector has wrong length")

        lower = finite_or_none(rule.get("lower_bound", rule.get("lb")))
        upper = finite_or_none(rule.get("upper_bound", rule.get("ub")))
        parsed.append(
            (
                np.asarray(coefficients, dtype=float),
                lower,
                upper,
            )
        )

    return parsed


def build_covariance(instance, n):
    direct = instance.get("covariance_matrix")
    if direct is None:
        direct = instance.get("covariance")

    if direct is not None:
        covariance = np.asarray(direct, dtype=float)
        if covariance.shape != (n, n):
            raise ValueError("Covariance matrix has wrong shape")
        return 0.5 * (covariance + covariance.T)

    r = int(instance.get("r", 0))
    loadings = np.asarray(instance.get("factor_loadings", []), dtype=float)

    if loadings.shape == (n, r):
        pass
    elif loadings.shape == (r, n):
        loadings = loadings.T
    elif r == 0 and loadings.size == 0:
        loadings = np.zeros((n, 0), dtype=float)
    else:
        raise ValueError(
            f"factor_loadings must have shape ({n}, {r}) "
            f"or transposed shape ({r}, {n})"
        )

    idiosyncratic = np.asarray(
        instance["idiosyncratic_variance"], dtype=float
    )
    if idiosyncratic.shape != (n,):
        raise ValueError("idiosyncratic_variance has wrong length")

    return loadings @ loadings.T + np.diag(idiosyncratic)


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(0.01, float(args.time_limit))

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    n = int(instance["n"])
    k = min(n, max(0, int(instance["k"])))
    gamma = float(instance["gamma"])
    kappa = float(instance.get("kappa", 1))

    if n <= 0:
        raise ValueError("n must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")

    mu = np.asarray(instance["mu"], dtype=float)
    if mu.shape != (n,):
        raise ValueError("mu has wrong length")

    covariance = build_covariance(instance, n)
    # Q includes both covariance risk and the ridge penalty.
    q_matrix = covariance + np.eye(n, dtype=float) / gamma
    q_matrix = 0.5 * (q_matrix + q_matrix.T)
    linear_cost = -kappa * mu

    constraints_data = instance.get("constraints", {})

    lower_raw = constraints_data.get("l_min_investment")
    upper_raw = constraints_data.get("u_max_investment")

    if lower_raw is None:
        lower = np.zeros(n, dtype=float)
    else:
        lower = np.maximum(0.0, np.asarray(lower_raw, dtype=float))

    if upper_raw is None:
        upper = np.ones(n, dtype=float)
    else:
        upper = np.minimum(
            1.0, np.maximum(0.0, np.asarray(upper_raw, dtype=float))
        )

    if lower.shape != (n,) or upper.shape != (n,):
        raise ValueError("Investment bound vector has wrong length")
    if np.any(lower > upper + 1e-12):
        raise ValueError("An investment lower bound exceeds its upper bound")

    has_min_return = bool(
        constraints_data.get("has_min_return_constraint", False)
    )
    r_bar = constraints_data.get("r_bar")
    if has_min_return:
        if r_bar is None:
            raise ValueError("Minimum-return constraint enabled with null r_bar")
        r_bar = float(r_bar)

    rules = read_linear_rules(instance, n)

    # Rows common to every support. The budget row is an equality.
    common_rows = [np.ones(n, dtype=float)]
    common_lbs = [1.0]
    common_ubs = [1.0]

    if has_min_return:
        common_rows.append(mu.copy())
        common_lbs.append(r_bar)
        common_ubs.append(np.inf)

    for coefficients, rule_lb, rule_ub in rules:
        common_rows.append(coefficients)
        common_lbs.append(-np.inf if rule_lb is None else rule_lb)
        common_ubs.append(np.inf if rule_ub is None else rule_ub)

    common_a = np.vstack(common_rows)
    common_lb = np.asarray(common_lbs, dtype=float)
    common_ub = np.asarray(common_ubs, dtype=float)

    def objective_full(x):
        return float(0.5 * x @ q_matrix @ x + linear_cost @ x)

    def feasible_full(x, tolerance=3e-6):
        if x.shape != (n,) or not np.all(np.isfinite(x)):
            return False
        if np.min(x) < -tolerance:
            return False
        if abs(float(np.sum(x)) - 1.0) > tolerance:
            return False
        if np.count_nonzero(x > tolerance) > k:
            return False

        for i in range(n):
            if x[i] > tolerance:
                if x[i] < lower[i] - tolerance:
                    return False
                if x[i] > upper[i] + tolerance:
                    return False

        activities = common_a @ x
        for j in range(len(common_lb)):
            if (
                math.isfinite(common_lb[j])
                and activities[j] < common_lb[j] - tolerance
            ):
                return False
            if (
                math.isfinite(common_ub[j])
                and activities[j] > common_ub[j] + tolerance
            ):
                return False
        return True

    best_value = math.inf
    best_x = None

    def record_candidate(x):
        nonlocal best_value, best_x

        x = np.asarray(x, dtype=float).copy()
        x[np.abs(x) < 1e-10] = 0.0

        # Correct tiny budget residuals without materially changing feasibility.
        residual = 1.0 - float(np.sum(x))
        if abs(residual) <= 1e-7:
            adjustable = [
                i
                for i in range(n)
                if x[i] > 1e-9
                and lower[i] - 1e-8 <= x[i] + residual <= upper[i] + 1e-8
            ]
            if adjustable:
                x[adjustable[0]] += residual

        if not feasible_full(x):
            return

        value = objective_full(x)
        if value < best_value - 1e-10:
            best_value = value
            best_x = x.copy()
            solution = {
                "objective_value": float(value),
                "x": [float(v) for v in x],
            }
            if logger:
                logger.log_solution(float(value), solution)

    def initial_point(support, mode):
        support = list(support)
        values = lower[support].copy()
        remaining = 1.0 - float(np.sum(values))

        if remaining < -1e-10:
            return None

        capacities = upper[support] - values

        if mode == 0:
            # Water filling tends to give a good ridge-regularized start.
            active = list(range(len(support)))
            while remaining > 1e-12 and active:
                share = remaining / len(active)
                used = 0.0
                next_active = []
                for p in active:
                    increment = min(share, capacities[p])
                    if increment > 0.0:
                        values[p] += increment
                        capacities[p] -= increment
                        remaining -= increment
                        used += increment
                    if capacities[p] > 1e-12:
                        next_active.append(p)
                active = next_active
                if used <= 1e-14:
                    break
        else:
            # Greedy return allocation is useful for minimum-return instances.
            order = sorted(
                range(len(support)),
                key=lambda p: (-mu[support[p]], support[p]),
            )
            for p in order:
                increment = min(remaining, capacities[p])
                values[p] += increment
                remaining -= increment
                if remaining <= 1e-12:
                    break

        if remaining > 1e-8:
            return None
        return values

    def solve_support(support):
        if time.monotonic() >= deadline:
            raise TimeLimitReached

        support = tuple(sorted(support))
        m = len(support)
        if m == 0 or m > k:
            return

        support_array = np.asarray(support, dtype=int)
        local_lower = lower[support_array]
        local_upper = upper[support_array]

        if float(np.sum(local_lower)) > 1.0 + 1e-10:
            return
        if float(np.sum(local_upper)) < 1.0 - 1e-10:
            return

        local_a = common_a[:, support_array]
        local_q = q_matrix[np.ix_(support_array, support_array)]
        local_c = linear_cost[support_array]

        def local_objective(y):
            return float(0.5 * y @ local_q @ y + local_c @ y)

        def local_gradient(y):
            return local_q @ y + local_c

        def timeout_callback(_):
            if time.monotonic() >= deadline:
                raise TimeLimitReached

        linear_constraint = LinearConstraint(local_a, common_lb, common_ub)
        bounds = Bounds(local_lower, local_upper)

        starts = []
        for mode in (0, 1):
            candidate = initial_point(support, mode)
            if candidate is not None:
                starts.append(candidate)

        if not starts:
            return

        # First improve feasibility using a smooth violation objective. This
        # makes SLSQP much more reliable for narrow exposure constraints.
        def violation_objective(y):
            activity = local_a @ y
            value = 0.0
            gradient_activity = np.zeros_like(activity)

            for row in range(len(activity)):
                if (
                    math.isfinite(common_lb[row])
                    and activity[row] < common_lb[row]
                ):
                    difference = activity[row] - common_lb[row]
                    value += difference * difference
                    gradient_activity[row] += 2.0 * difference

                if (
                    math.isfinite(common_ub[row])
                    and activity[row] > common_ub[row]
                ):
                    difference = activity[row] - common_ub[row]
                    value += difference * difference
                    gradient_activity[row] += 2.0 * difference

            return float(value), local_a.T @ gradient_activity

        for starting_point in starts:
            if time.monotonic() >= deadline:
                raise TimeLimitReached

            phase_start = starting_point
            activity = local_a @ phase_start
            initially_feasible = np.all(
                activity >= common_lb - 1e-7
            ) and np.all(activity <= common_ub + 1e-7)

            if not initially_feasible:
                try:
                    phase_result = minimize(
                        fun=lambda y: violation_objective(y)[0],
                        x0=phase_start,
                        jac=lambda y: violation_objective(y)[1],
                        method="L-BFGS-B",
                        bounds=list(zip(local_lower, local_upper)),
                        callback=timeout_callback,
                        options={
                            "maxiter": 150,
                            "ftol": 1e-14,
                            "gtol": 1e-10,
                            "maxls": 30,
                        },
                    )
                    phase_start = phase_result.x
                except TimeLimitReached:
                    raise
                except Exception:
                    phase_start = starting_point

            try:
                result = minimize(
                    fun=local_objective,
                    x0=phase_start,
                    jac=local_gradient,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=[linear_constraint],
                    callback=timeout_callback,
                    options={
                        "maxiter": 300,
                        "ftol": 1e-11,
                        "disp": False,
                    },
                )
            except TimeLimitReached:
                raise
            except Exception:
                continue

            full_x = np.zeros(n, dtype=float)
            full_x[support_array] = result.x
            record_candidate(full_x)

            # Convex fixed-support problems need only one successful start.
            if result.success and feasible_full(full_x):
                break

    diagonal = np.diag(q_matrix)
    utility_score = kappa * mu - 0.5 * diagonal
    ratio_score = kappa * mu / np.maximum(diagonal, 1e-12)

    rankings = [
        list(np.argsort(-utility_score)),
        list(np.argsort(-mu)),
        list(np.argsort(diagonal)),
        list(np.argsort(-ratio_score)),
        list(np.argsort(-upper)),
    ]

    seen_supports = set()

    def try_support(support):
        support = tuple(sorted(set(int(i) for i in support)))
        if not support or len(support) > k or support in seen_supports:
            return
        seen_supports.add(support)
        solve_support(support)

    try:
        # Quickly test high-quality structured supports.
        for ranking in rankings:
            available = [i for i in ranking if upper[i] > 1e-14]
            for size in range(1, min(k, len(available)) + 1):
                try_support(available[:size])

        # Test small variations of the best utility support.
        base_ranking = [
            i for i in rankings[0] if upper[i] > 1e-14
        ]
        for size in range(1, min(k, len(base_ranking)) + 1):
            base = base_ranking[:size]
            try_support(base)
            for position in range(size):
                for replacement in base_ranking[size:min(n, size + 12)]:
                    modified = base.copy()
                    modified[position] = replacement
                    try_support(modified)

        eligible = [i for i in range(n) if upper[i] > 1e-14]

        # Exhaustive enumeration is exact when the support search is small.
        combination_count = 0
        for size in range(1, min(k, len(eligible)) + 1):
            combination_count += math.comb(len(eligible), size)
            if combination_count > 150000:
                break

        if combination_count <= 150000:
            for size in range(1, min(k, len(eligible)) + 1):
                for support in itertools.combinations(eligible, size):
                    try_support(support)
        else:
            # Deterministic randomized support search for large instances.
            rng = random.Random(0)
            weights = np.exp(
                np.clip(
                    utility_score - np.max(utility_score),
                    -30.0,
                    0.0,
                )
            )
            weights = weights + 0.05
            eligible_weights = [float(weights[i]) for i in eligible]

            while time.monotonic() < deadline:
                size = rng.randint(1, min(k, len(eligible)))

                remaining = eligible.copy()
                remaining_weights = eligible_weights.copy()
                support = []

                for _ in range(size):
                    total_weight = sum(remaining_weights)
                    draw = rng.random() * total_weight
                    cumulative = 0.0
                    selected_position = len(remaining) - 1
                    for p, weight in enumerate(remaining_weights):
                        cumulative += weight
                        if cumulative >= draw:
                            selected_position = p
                            break

                    support.append(remaining.pop(selected_position))
                    remaining_weights.pop(selected_position)

                try_support(support)

    except TimeLimitReached:
        pass

    if best_x is None:
        # Last-resort singleton checks, primarily for extremely short limits.
        for i in range(n):
            candidate = np.zeros(n, dtype=float)
            candidate[i] = 1.0
            record_candidate(candidate)

    if best_x is None:
        raise RuntimeError("No feasible solution found within the time limit")

    solution = {
        "objective_value": float(best_value),
        "x": [float(v) for v in best_x],
    }

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(solution, file, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()