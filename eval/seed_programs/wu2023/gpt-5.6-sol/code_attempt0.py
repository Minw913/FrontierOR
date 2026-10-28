import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    return parser.parse_args()


def configure_model(model, time_limit):
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.01, float(time_limit))


def compute_quality_matrix(costs):
    """
    quality[i,j] is the largest capped scaling factor t for which scenario j
    componentwise dominates t times scenario i:
        t * costs[i,d] <= costs[j,d] for every d.
    """
    n_scenarios, n_dimensions = costs.shape
    quality = np.ones((n_scenarios, n_scenarios), dtype=np.float64)
    for d in range(n_dimensions):
        ratios = costs[:, d][None, :] / costs[:, d][:, None]
        np.minimum(quality, ratios, out=quality)
    np.minimum(quality, 1.0, out=quality)
    return quality


def evaluate_selected(quality, selected):
    selected = list(selected)
    represented = quality[:, selected]
    best_positions = np.argmax(represented, axis=1)
    values = represented[np.arange(quality.shape[0]), best_positions]
    assignment = [int(best_positions[i]) for i in range(quality.shape[0])]

    # Force each selected scenario into its own aggregate. This is needed for
    # the corresponding one-stage convex-combination witness.
    for k, scenario in enumerate(selected):
        assignment[scenario] = k

    t = 1.0
    for i, k in enumerate(assignment):
        t = min(t, float(quality[i, selected[k]]))
    return max(0.0, min(1.0, t)), assignment


def greedy_selection_sets(quality, requested_k):
    n = quality.shape[0]
    maximum_k = max(requested_k)

    single_scores = np.min(quality, axis=0)
    first = int(np.argmax(single_scores))

    selected = [first]
    chosen = np.zeros(n, dtype=bool)
    chosen[first] = True
    current = quality[:, first].copy()
    snapshots = {1: list(selected)}

    for size in range(2, maximum_k + 1):
        best_candidate = None
        best_score = -1.0

        remaining = np.flatnonzero(~chosen)
        for candidate in remaining:
            score = float(np.min(np.maximum(current, quality[:, candidate])))
            if score > best_score + 1e-15:
                best_score = score
                best_candidate = int(candidate)

        if best_candidate is None:
            best_candidate = int(remaining[0])

        selected.append(best_candidate)
        chosen[best_candidate] = True
        current = np.maximum(current, quality[:, best_candidate])
        snapshots[size] = list(selected)

    # Small deterministic one-swap improvement.
    if n <= 250 and len(requested_k) <= 100:
        for k in requested_k:
            selected_k = list(snapshots[k])
            best_t, _ = evaluate_selected(quality, selected_k)
            selected_set = set(selected_k)

            improved = True
            passes = 0
            while improved and passes < 2:
                improved = False
                passes += 1
                best_swap = None
                for position in range(k):
                    for candidate in range(n):
                        if candidate in selected_set:
                            continue
                        trial = list(selected_k)
                        trial[position] = candidate
                        trial_t, _ = evaluate_selected(quality, trial)
                        if trial_t > best_t + 1e-12:
                            best_t = trial_t
                            best_swap = (position, candidate)
                if best_swap is not None:
                    position, candidate = best_swap
                    selected_set.remove(selected_k[position])
                    selected_k[position] = candidate
                    selected_set.add(candidate)
                    improved = True
            snapshots[k] = selected_k

    return snapshots


def one_stage_witness_from_selected(costs, quality, selected):
    n = costs.shape[0]
    k_count = len(selected)
    t, assignment = evaluate_selected(quality, selected)

    lambda_weights = {}
    for k, scenario in enumerate(selected):
        lambda_weights[str(k)] = {str(int(scenario)): 1.0}

    return {
        "t": float(t),
        "approximation_ratio": float(1.0 / max(t, 1e-300)),
        "assignment": {str(i): int(assignment[i]) for i in range(n)},
        "lambda_weights": lambda_weights,
        "solver_status": int(GRB.TIME_LIMIT),
        "best_bound": 1.0,
        "mip_gap": float((1.0 - t) / max(t, 1e-300)),
    }


def two_stage_result(quality, selected):
    t, _ = evaluate_selected(quality, selected)
    return {
        "t": float(t),
        "approximation_ratio": float(1.0 / max(t, 1e-300)),
        "selected_scenarios": [int(v) for v in selected],
    }


def solve_two_stage_mip(quality, k_count, initial_selected, time_budget):
    n = quality.shape[0]
    model = gp.Model("two_stage_scenario_selection")
    configure_model(model, time_budget)

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    mu = model.addVars(n, n, vtype=GRB.BINARY, name="mu")
    t = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="t")

    model.addConstr(gp.quicksum(x[j] for j in range(n)) == k_count)
    for i in range(n):
        model.addConstr(gp.quicksum(mu[i, j] for j in range(n)) == 1)
        for j in range(n):
            model.addConstr(mu[i, j] <= x[j])
            model.addConstr(t <= float(quality[i, j]) + (1.0 - mu[i, j]))

    model.setObjective(t, GRB.MAXIMIZE)

    initial_t, initial_assignment = evaluate_selected(quality, initial_selected)
    selected_position = {scenario: pos for pos, scenario in enumerate(initial_selected)}

    for j in range(n):
        x[j].Start = 1.0 if j in selected_position else 0.0
    for i in range(n):
        assigned_scenario = initial_selected[initial_assignment[i]]
        for j in range(n):
            mu[i, j].Start = 1.0 if j == assigned_scenario else 0.0
    t.Start = initial_t

    model.optimize()

    selected = list(initial_selected)
    if model.SolCount > 0:
        ranked = sorted(range(n), key=lambda j: x[j].X, reverse=True)
        selected = ranked[:k_count]

    result = two_stage_result(quality, selected)
    model.dispose()
    return result


def calculate_one_stage_witness(costs, assignment, weights):
    n, dimensions = costs.shape
    k_count = len(weights)

    aggregate_costs = np.zeros((k_count, dimensions), dtype=np.float64)
    for k in range(k_count):
        for scenario, value in weights[k].items():
            aggregate_costs[k, scenario * 0:scenario * 0 + dimensions] += (
                float(value) * costs[scenario]
            )

    t = 1.0
    for i in range(n):
        k = assignment[i]
        ratio = float(np.min(aggregate_costs[k] / costs[i]))
        t = min(t, ratio)

    return max(0.0, min(1.0, t))


def solve_one_stage_mip(costs, initial_selected, quality, time_budget):
    n, dimensions = costs.shape
    k_count = len(initial_selected)

    model = gp.Model("one_stage_IP_mu")
    configure_model(model, time_budget)

    mu = model.addVars(n, k_count, vtype=GRB.BINARY, name="mu")
    lam = model.addVars(n, k_count, lb=0.0, ub=1.0,
                        vtype=GRB.CONTINUOUS, name="lambda")
    t = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="t")

    for i in range(n):
        model.addConstr(gp.quicksum(mu[i, k] for k in range(k_count)) == 1)

    for k in range(k_count):
        model.addConstr(gp.quicksum(lam[j, k] for j in range(n)) == 1.0)
        for j in range(n):
            model.addConstr(lam[j, k] <= mu[j, k])

    # If mu[i,k] is one, aggregate k must componentwise dominate t*c_i.
    # Because t <= 1, costs[i,d] is a valid deactivation constant.
    for i in range(n):
        for k in range(k_count):
            for d in range(dimensions):
                aggregate_component = gp.quicksum(
                    float(costs[j, d]) * lam[j, k] for j in range(n)
                )
                model.addConstr(
                    t * float(costs[i, d])
                    <= aggregate_component
                    + float(costs[i, d]) * (1.0 - mu[i, k])
                )

    model.setObjective(t, GRB.MAXIMIZE)

    initial_t, initial_assignment = evaluate_selected(quality, initial_selected)
    for i in range(n):
        for k in range(k_count):
            mu[i, k].Start = 1.0 if initial_assignment[i] == k else 0.0
            lam[i, k].Start = 1.0 if initial_selected[k] == i else 0.0
    t.Start = initial_t

    model.optimize()

    if model.SolCount <= 0:
        fallback = one_stage_witness_from_selected(
            costs, quality, initial_selected
        )
        fallback["solver_status"] = int(model.Status)
        model.dispose()
        return fallback

    assignment = []
    for i in range(n):
        assigned = max(range(k_count), key=lambda k: mu[i, k].X)
        assignment.append(int(assigned))

    # Retain only weights supported by the rounded assignment and normalize.
    weights = []
    for k in range(k_count):
        cluster_members = [i for i in range(n) if assignment[i] == k]
        raw = {
            i: max(0.0, float(lam[i, k].X))
            for i in cluster_members
            if lam[i, k].X > 1e-12
        }
        total = sum(raw.values())

        if total <= 1e-14:
            # This should not occur in an integer incumbent, but gives a
            # numerically safe witness if the returned solution is imprecise.
            if cluster_members:
                raw = {cluster_members[0]: 1.0}
            else:
                # Repair an empty rounded cluster by moving the strongest
                # lambda-support scenario into it.
                scenario = max(range(n), key=lambda i: lam[i, k].X)
                old_k = assignment[scenario]
                assignment[scenario] = k
                raw = {scenario: 1.0}

                # If the old aggregate became empty, assign one of its
                # positive-lambda scenarios back to it.
                if not any(a == old_k for a in assignment):
                    replacement = max(range(n), key=lambda i: lam[i, old_k].X)
                    assignment[replacement] = old_k
        else:
            raw = {i: value / total for i, value in raw.items()}

        weights.append(raw)

    # Rebuild weights once more after any rare empty-cluster repair.
    for k in range(k_count):
        members = [i for i in range(n) if assignment[i] == k]
        supported = {
            i: max(0.0, float(lam[i, k].X))
            for i in members
            if lam[i, k].X > 1e-12
        }
        total = sum(supported.values())
        if total > 1e-14:
            weights[k] = {i: value / total for i, value in supported.items()}
        elif members:
            weights[k] = {members[0]: 1.0}

    witness_t = calculate_one_stage_witness(costs, assignment, weights)

    try:
        best_bound = float(model.ObjBound)
        if not math.isfinite(best_bound):
            best_bound = 1.0
        best_bound = min(1.0, max(witness_t, best_bound))
    except Exception:
        best_bound = 1.0

    try:
        mip_gap = float(model.MIPGap)
        if not math.isfinite(mip_gap):
            mip_gap = (best_bound - witness_t) / max(witness_t, 1e-300)
    except Exception:
        mip_gap = (best_bound - witness_t) / max(witness_t, 1e-300)

    result = {
        "t": float(witness_t),
        "approximation_ratio": float(1.0 / max(witness_t, 1e-300)),
        "assignment": {str(i): int(assignment[i]) for i in range(n)},
        "lambda_weights": {
            str(k): {
                str(i): float(value)
                for i, value in sorted(weights[k].items())
                if value > 0.0
            }
            for k in range(k_count)
        },
        "solver_status": int(model.Status),
        "best_bound": float(best_bound),
        "mip_gap": float(max(0.0, mip_gap)),
    }

    model.dispose()
    return result


def update_objective(solution):
    values = []
    values.extend(float(v["t"]) for v in solution["one_stage_IP_mu"].values())
    values.extend(float(v["t"]) for v in solution["two_stage_IP"].values())
    solution["objective_value"] = float(max(values) if values else 0.0)


def write_solution(path, solution):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(solution, handle, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r", encoding="utf-8") as handle:
        instance = json.load(handle)

    costs = np.asarray(instance["scenarios"], dtype=np.float64)
    n = int(instance["num_scenarios_N"])
    dimensions = int(instance["problem_dimension_n"])
    if costs.shape != (n, dimensions):
        raise ValueError("The scenarios matrix does not match N and problem_dimension_n")
    if np.any(costs <= 0):
        raise ValueError("All scenario coefficients must be positive")

    requested_k = []
    seen = set()
    for value in instance["K_values"]:
        k = int(value)
        if not 1 <= k <= n:
            raise ValueError(f"Invalid K value: {k}")
        if k not in seen:
            seen.add(k)
            requested_k.append(k)

    quality = compute_quality_matrix(costs)
    greedy_sets = greedy_selection_sets(quality, requested_k)

    solution = {
        "objective_value": 0.0,
        "one_stage_IP_mu": {},
        "two_stage_IP": {},
    }

    for k in requested_k:
        selected = greedy_sets[k]
        solution["two_stage_IP"][str(k)] = two_stage_result(quality, selected)
        solution["one_stage_IP_mu"][str(k)] = one_stage_witness_from_selected(
            costs, quality, selected
        )

    update_objective(solution)
    best_logged_objective = solution["objective_value"]
    if logger:
        logger.log_solution(best_logged_objective, solution)

    # Improve the discrete selections first. Reserve most of the remaining
    # time for the stronger one-stage convex aggregation models.
    total_jobs = 2 * len(requested_k)
    completed_jobs = 0

    for k in sorted(requested_k, reverse=True):
        remaining = deadline - time.monotonic()
        jobs_left = max(1, total_jobs - completed_jobs)
        completed_jobs += 1
        if remaining <= 0.15:
            continue

        fair_share = remaining / jobs_left
        budget = min(fair_share, max(0.05, 0.30 * remaining / max(1, len(requested_k))))
        if budget < 0.05:
            continue

        initial = solution["two_stage_IP"][str(k)]["selected_scenarios"]
        try:
            improved = solve_two_stage_mip(quality, k, initial, budget)
            solution["two_stage_IP"][str(k)] = improved

            # A selected-scenario solution is also a valid one-stage witness.
            old_one = solution["one_stage_IP_mu"][str(k)]
            selected_witness = one_stage_witness_from_selected(
                costs, quality, improved["selected_scenarios"]
            )
            if selected_witness["t"] > old_one["t"] + 1e-12:
                solution["one_stage_IP_mu"][str(k)] = selected_witness
        except gp.GurobiError:
            pass

        update_objective(solution)
        if solution["objective_value"] > best_logged_objective + 1e-12:
            best_logged_objective = solution["objective_value"]
            if logger:
                logger.log_solution(best_logged_objective, solution)

    # Solve the one-stage IP-mu formulations with the remaining time.
    one_stage_order = sorted(requested_k, reverse=True)
    for index, k in enumerate(one_stage_order):
        remaining = deadline - time.monotonic()
        jobs_left = len(one_stage_order) - index
        completed_jobs += 1
        if remaining <= 0.15:
            break

        budget = max(0.05, remaining / max(1, jobs_left))
        budget = min(budget, max(0.05, remaining - 0.05))
        if budget < 0.05:
            break

        selected = solution["two_stage_IP"][str(k)]["selected_scenarios"]
        try:
            result = solve_one_stage_mip(costs, selected, quality, budget)
            if result["t"] + 1e-12 >= solution["one_stage_IP_mu"][str(k)]["t"]:
                solution["one_stage_IP_mu"][str(k)] = result
            else:
                # Preserve the stronger witness but retain meaningful solver
                # metadata from the attempted solve.
                old = solution["one_stage_IP_mu"][str(k)]
                old["solver_status"] = result["solver_status"]
                old["best_bound"] = max(old["t"], result["best_bound"])
                old["mip_gap"] = (
                    old["best_bound"] - old["t"]
                ) / max(old["t"], 1e-300)
        except gp.GurobiError:
            pass

        update_objective(solution)
        if solution["objective_value"] > best_logged_objective + 1e-12:
            best_logged_objective = solution["objective_value"]
            if logger:
                logger.log_solution(best_logged_objective, solution)

    update_objective(solution)
    write_solution(args.solution_path, solution)


if __name__ == "__main__":
    main()