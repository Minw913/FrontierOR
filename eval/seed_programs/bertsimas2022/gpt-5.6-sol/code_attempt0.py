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


def get_linear_rules(instance):
    """Read optional linear rules if an extended instance includes them."""
    constraints = instance.get("constraints", {})
    rules = (
        constraints.get("linear_constraints")
        or constraints.get("rules")
        or instance.get("linear_constraints")
        or []
    )

    parsed = []
    for rule in rules:
        coefficients = (
            rule.get("coefficients")
            if "coefficients" in rule
            else rule.get("a")
        )
        if coefficients is None:
            continue
        lower = rule.get("lower_bound", rule.get("lb"))
        upper = rule.get("upper_bound", rule.get("ub"))
        parsed.append((coefficients, lower, upper))
    return parsed


def normalize_factor_loadings(raw, n, r):
    if raw is None:
        return None

    # Schema orientation: n securities by r factors.
    if len(raw) == n and all(len(row) == r for row in raw):
        return [[float(v) for v in row] for row in raw]

    # Also accept the transposed orientation for robustness.
    if len(raw) == r and all(len(row) == n for row in raw):
        return [
            [float(raw[f][i]) for f in range(r)]
            for i in range(n)
        ]

    raise ValueError(
        f"factor_loadings must have shape [{n}][{r}] "
        f"(or transposed [{r}][{n}])"
    )


def finite_bound(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    n = int(instance["n"])
    k = max(0, min(int(instance["k"]), n))
    r = int(instance.get("r", 0))
    gamma = float(instance["gamma"])
    kappa = float(instance.get("kappa", 1))
    mu = [float(v) for v in instance["mu"]]

    if len(mu) != n:
        raise ValueError("mu has the wrong length")
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive")

    constraints_data = instance.get("constraints", {})

    lower_raw = constraints_data.get("l_min_investment")
    upper_raw = constraints_data.get("u_max_investment")

    lower = (
        [max(0.0, float(v)) for v in lower_raw]
        if lower_raw is not None
        else [0.0] * n
    )
    upper = (
        [min(1.0, max(0.0, float(v))) for v in upper_raw]
        if upper_raw is not None
        else [1.0] * n
    )

    if len(lower) != n or len(upper) != n:
        raise ValueError("Investment-bound vectors have the wrong length")

    for i in range(n):
        if lower[i] > upper[i] + 1e-12:
            raise ValueError(f"Invalid investment bounds for security {i}")

    has_min_return = bool(
        constraints_data.get("has_min_return_constraint", False)
    )
    r_bar = constraints_data.get("r_bar")
    if has_min_return:
        if r_bar is None:
            raise ValueError("Minimum-return constraint is enabled but r_bar is null")
        r_bar = float(r_bar)

    linear_rules = get_linear_rules(instance)
    for coefficients, _, _ in linear_rules:
        if len(coefficients) != n:
            raise ValueError("A linear-rule coefficient vector has the wrong length")

    direct_covariance = (
        instance.get("covariance_matrix")
        if instance.get("covariance_matrix") is not None
        else instance.get("covariance")
    )

    factor_loadings = None
    idiosyncratic = None
    covariance = None

    if direct_covariance is not None:
        if len(direct_covariance) != n or any(
            len(row) != n for row in direct_covariance
        ):
            raise ValueError("Covariance matrix has the wrong shape")
        covariance = [
            [float(direct_covariance[i][j]) for j in range(n)]
            for i in range(n)
        ]
    else:
        factor_loadings = normalize_factor_loadings(
            instance.get("factor_loadings"), n, r
        )
        idiosyncratic = [
            float(v) for v in instance["idiosyncratic_variance"]
        ]
        if len(idiosyncratic) != n:
            raise ValueError("idiosyncratic_variance has the wrong length")

    def objective_value(x_values):
        ridge = sum(v * v for v in x_values) / (2.0 * gamma)
        expected_return = sum(mu[i] * x_values[i] for i in range(n))

        if covariance is not None:
            variance = 0.0
            for i in range(n):
                xi = x_values[i]
                if xi == 0.0:
                    continue
                variance += covariance[i][i] * xi * xi
                for j in range(i + 1, n):
                    variance += (
                        2.0
                        * covariance[i][j]
                        * xi
                        * x_values[j]
                    )
            variance *= 0.5
        else:
            factor_exposure = [0.0] * r
            for i in range(n):
                xi = x_values[i]
                if xi == 0.0:
                    continue
                for f in range(r):
                    factor_exposure[f] += factor_loadings[i][f] * xi

            systematic = sum(v * v for v in factor_exposure)
            specific = sum(
                idiosyncratic[i] * x_values[i] * x_values[i]
                for i in range(n)
            )
            variance = 0.5 * (systematic + specific)

        return float(variance + ridge - kappa * expected_return)

    def is_feasible(x_values, tolerance=1e-7):
        if len(x_values) != n:
            return False
        if abs(sum(x_values) - 1.0) > tolerance:
            return False

        selected = 0
        for i, value in enumerate(x_values):
            if value < -tolerance:
                return False
            if value > tolerance:
                selected += 1
                if value < lower[i] - tolerance:
                    return False
                if value > upper[i] + tolerance:
                    return False

        if selected > k:
            return False

        if has_min_return:
            portfolio_return = sum(mu[i] * x_values[i] for i in range(n))
            if portfolio_return < r_bar - tolerance:
                return False

        for coefficients, lb, ub in linear_rules:
            activity = sum(float(coefficients[i]) * x_values[i] for i in range(n))
            lb = finite_bound(lb)
            ub = finite_bound(ub)
            if lb is not None and activity < lb - tolerance:
                return False
            if ub is not None and activity > ub + tolerance:
                return False

        return True

    # Construct a few inexpensive feasible starts. These are especially useful
    # when the time limit is short.
    def build_support(order):
        support = []
        total_lower = 0.0
        total_upper = 0.0

        for i in order:
            if len(support) >= k:
                break
            if upper[i] <= 1e-14:
                continue
            if total_lower + lower[i] > 1.0 + 1e-12:
                continue

            support.append(i)
            total_lower += lower[i]
            total_upper += upper[i]

            if total_upper >= 1.0 - 1e-12:
                break

        if total_upper < 1.0 - 1e-12:
            return None
        return support

    def allocate_on_support(support, priority, equalize=False):
        if support is None:
            return None

        x_values = [0.0] * n
        for i in support:
            x_values[i] = lower[i]

        remaining = 1.0 - sum(x_values)
        if remaining < -1e-10:
            return None

        if equalize:
            active = set(support)
            while remaining > 1e-12 and active:
                share = remaining / len(active)
                used = 0.0
                saturated = []
                for i in active:
                    increment = min(share, upper[i] - x_values[i])
                    if increment > 0.0:
                        x_values[i] += increment
                        used += increment
                    if upper[i] - x_values[i] <= 1e-12:
                        saturated.append(i)
                remaining -= used
                for i in saturated:
                    active.discard(i)
                if used <= 1e-14:
                    break
        else:
            support_set = set(support)
            for i in priority:
                if i not in support_set:
                    continue
                increment = min(remaining, upper[i] - x_values[i])
                if increment > 0.0:
                    x_values[i] += increment
                    remaining -= increment
                if remaining <= 1e-12:
                    break

        if abs(remaining) <= 1e-10:
            remaining = 0.0

        return x_values if remaining <= 1e-8 else None

    if covariance is not None:
        diagonal_risk = [
            covariance[i][i] + 1.0 / gamma for i in range(n)
        ]
    else:
        diagonal_risk = [
            idiosyncratic[i]
            + sum(factor_loadings[i][f] ** 2 for f in range(r))
            + 1.0 / gamma
            for i in range(n)
        ]

    return_order = sorted(range(n), key=lambda i: (-mu[i], i))
    utility_order = sorted(
        range(n),
        key=lambda i: (
            -(kappa * mu[i] - 0.5 * diagonal_risk[i]),
            i,
        ),
    )
    low_risk_order = sorted(range(n), key=lambda i: (diagonal_risk[i], i))
    capacity_order = sorted(range(n), key=lambda i: (-upper[i], i))

    candidate_orders = [
        return_order,
        utility_order,
        low_risk_order,
        capacity_order,
    ]

    best = {"objective": math.inf, "solution": None}

    for order in candidate_orders:
        support = build_support(order)
        if support is None:
            continue

        for equalize in (False, True):
            allocation_priority = return_order if has_min_return else order
            candidate = allocate_on_support(
                support, allocation_priority, equalize=equalize
            )
            if candidate is None or not is_feasible(candidate):
                continue

            value = objective_value(candidate)
            if value < best["objective"] - 1e-12:
                solution = {
                    "objective_value": value,
                    "x": [float(v) for v in candidate],
                }
                best["objective"] = value
                best["solution"] = solution
                if logger:
                    logger.log_solution(value, solution)

    model = gp.Model("sparse_portfolio")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    remaining_time = max(0.01, deadline - time.monotonic())
    model.Params.TimeLimit = remaining_time

    x = model.addVars(
        n,
        lb=0.0,
        ub={i: upper[i] for i in range(n)},
        vtype=GRB.CONTINUOUS,
        name="x",
    )
    z = model.addVars(n, vtype=GRB.BINARY, name="z")

    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1.0, name="budget")
    model.addConstr(gp.quicksum(z[i] for i in range(n)) <= k, name="cardinality")

    for i in range(n):
        model.addConstr(x[i] <= upper[i] * z[i], name=f"upper_link_{i}")
        if lower[i] > 0.0:
            model.addConstr(x[i] >= lower[i] * z[i], name=f"lower_link_{i}")
        if upper[i] <= 1e-14:
            z[i].UB = 0.0

    if has_min_return:
        model.addConstr(
            gp.quicksum(mu[i] * x[i] for i in range(n)) >= r_bar,
            name="minimum_return",
        )

    for rule_index, (coefficients, lb, ub) in enumerate(linear_rules):
        expression = gp.quicksum(
            float(coefficients[i]) * x[i] for i in range(n)
        )
        lb = finite_bound(lb)
        ub = finite_bound(ub)
        if lb is not None:
            model.addConstr(expression >= lb, name=f"rule_lb_{rule_index}")
        if ub is not None:
            model.addConstr(expression <= ub, name=f"rule_ub_{rule_index}")

    if covariance is not None:
        quadratic = gp.QuadExpr()
        for i in range(n):
            diagonal = covariance[i][i] + 1.0 / gamma
            if diagonal != 0.0:
                quadratic.add(0.5 * diagonal * x[i] * x[i])
            for j in range(i + 1, n):
                coefficient = covariance[i][j]
                if coefficient != 0.0:
                    quadratic.add(coefficient * x[i] * x[j])
    else:
        factor_exposure = model.addVars(
            r,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="factor_exposure",
        )

        for f in range(r):
            model.addConstr(
                factor_exposure[f]
                == gp.quicksum(
                    factor_loadings[i][f] * x[i] for i in range(n)
                ),
                name=f"factor_definition_{f}",
            )

        quadratic = gp.QuadExpr()
        for f in range(r):
            quadratic.add(
                0.5 * factor_exposure[f] * factor_exposure[f]
            )
        for i in range(n):
            coefficient = idiosyncratic[i] + 1.0 / gamma
            quadratic.add(0.5 * coefficient * x[i] * x[i])

    linear_return = gp.quicksum(
        kappa * mu[i] * x[i] for i in range(n)
    )
    model.setObjective(quadratic - linear_return, GRB.MINIMIZE)

    if best["solution"] is not None:
        warm_x = best["solution"]["x"]
        for i in range(n):
            x[i].Start = warm_x[i]
            z[i].Start = 1.0 if warm_x[i] > 1e-10 else 0.0

    def incumbent_callback(callback_model, where):
        if time.monotonic() >= deadline:
            callback_model.terminate()
            return

        if where != GRB.Callback.MIPSOL:
            return

        try:
            values = callback_model.cbGetSolution(
                [x[i] for i in range(n)]
            )
            values = [float(v) for v in values]
            value = objective_value(values)

            if value < best["objective"] - 1e-10:
                solution = {
                    "objective_value": value,
                    "x": values,
                }
                best["objective"] = value
                best["solution"] = solution
                if logger:
                    logger.log_solution(value, solution)
        except Exception:
            # Logging or callback extraction should not abort the optimization.
            pass

    try:
        model.optimize(incumbent_callback)
    except gp.GurobiError:
        # Preserve and return any feasible incumbent already obtained.
        pass

    if model.SolCount > 0:
        final_x = [float(x[i].X) for i in range(n)]
        final_value = objective_value(final_x)
        if final_value < best["objective"] - 1e-10 or best["solution"] is None:
            final_solution = {
                "objective_value": final_value,
                "x": final_x,
            }
            best["objective"] = final_value
            best["solution"] = final_solution
            if logger:
                logger.log_solution(final_value, final_solution)

    if best["solution"] is None:
        raise RuntimeError(
            "No feasible portfolio was found within the time limit"
        )

    output_directory = os.path.dirname(os.path.abspath(args.solution_path))
    os.makedirs(output_directory, exist_ok=True)

    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(best["solution"], f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()