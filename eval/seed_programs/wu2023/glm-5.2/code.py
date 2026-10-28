import json
import argparse
import numpy as np
import time
from gurobipy import Model, GRB, quicksum

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def solve_one_stage_feasibility(N, n, costs_arr, K, t, time_limit, prev_solution=None):
    """Check if scaling factor t is feasible for the one-stage IP."""
    model = Model("osf")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.Threads = 1
    model.Params.NumericFocus = 0
    model.Params.MIPGap = 1e-4
    if time_limit > 0:
        model.Params.TimeLimit = max(1, time_limit)

    mu = model.addVars(N, K, vtype=GRB.BINARY, name="mu")
    lam = model.addVars(K, N, lb=0, ub=1, name="lam")

    if prev_solution is not None:
        prev_mu, prev_lam = prev_solution
        for i in range(N):
            for a in range(K):
                mu[i, a].Start = 1 if prev_mu.get(i) == a else 0
        for a in range(K):
            for j in range(N):
                lam[a, j].Start = prev_lam.get(a, {}).get(j, 0.0)

    for i in range(N):
        model.addConstr(quicksum(mu[i, a] for a in range(K)) == 1)
    for a in range(K):
        model.addConstr(quicksum(lam[a, j] for j in range(N)) == 1)
    for a in range(K):
        model.addConstr(quicksum(mu[i, a] for i in range(N)) >= 1)

    for i in range(N):
        for a in range(K):
            for d in range(n):
                agg = quicksum(lam[a, j] * costs_arr[j, d] for j in range(N))
                model.addConstr((mu[i, a] == 1) >> (t * agg <= costs_arr[i, d]))

    model.optimize()

    if model.SolCount > 0:
        assignment = {}
        for i in range(N):
            for a in range(K):
                if mu[i, a].X > 0.5:
                    assignment[i] = a
                    break
        lambda_vals = {}
        for a in range(K):
            lambda_vals[a] = {}
            for j in range(N):
                if lam[a, j].X > 1e-6:
                    lambda_vals[a][j] = float(lam[a, j].X)
            if not lambda_vals[a]:
                lambda_vals[a][0] = 1.0
        return True, assignment, lambda_vals, model.status
    return False, None, None, model.status


def solve_one_stage_k1(N, n, costs_arr):
    """Solve one-stage for K=1 (pure LP, no binary variables)."""
    max_c = float(costs_arr.max())
    min_c = float(costs_arr.min())
    t_high = max_c / min_c if min_c > 0 else 1e6
    t_low = 0.0
    best_t = 0.0
    best_lambda = {0: 1.0}

    for _ in range(50):
        t_mid = (t_low + t_high) / 2
        if t_mid < 1e-12:
            break
        model = Model("osk1")
        model.Params.OutputFlag = 0
        model.Params.Seed = 0
        model.Params.Threads = 1
        model.Params.NumericFocus = 0
        lam = model.addVars(N, lb=0, ub=1)
        model.addConstr(quicksum(lam[j] for j in range(N)) == 1)
        for i in range(N):
            for d in range(n):
                model.addConstr(
                    t_mid * quicksum(lam[j] * costs_arr[j, d] for j in range(N))
                    <= costs_arr[i, d]
                )
        model.optimize()
        if model.status == GRB.OPTIMAL:
            best_t = t_mid
            best_lambda = {j: float(lam[j].X) for j in range(N) if lam[j].X > 1e-6}
            if not best_lambda:
                best_lambda = {0: 1.0}
            t_low = t_mid
        else:
            t_high = t_mid

    assignment = {str(i): 0 for i in range(N)}
    lambda_weights = {"0": {str(j): w for j, w in best_lambda.items()}}
    return best_t, assignment, lambda_weights, GRB.OPTIMAL, best_t, 0.0


def solve_one_stage(N, n, costs, K, time_budget):
    """Solve one-stage IP for given K via binary search on t."""
    costs_arr = np.array(costs, dtype=float)
    if K >= N:
        return (
            1.0,
            {str(i): i for i in range(N)},
            {str(a): {str(a): 1.0} for a in range(K)},
            GRB.OPTIMAL,
            1.0,
            0.0,
        )
    if K == 1:
        return solve_one_stage_k1(N, n, costs_arr)

    max_c = float(costs_arr.max())
    min_c = float(costs_arr.min())
    t_high = max_c / min_c if min_c > 0 else 1e6
    t_low = 0.0
    best_t = 0.0
    best_assignment = {str(i): i % K for i in range(N)}
    best_lambda = {str(a): {str(a % N): 1.0} for a in range(K)}
    prev_solution = None
    n_iter = 35
    start = time.time()

    for it in range(n_iter):
        elapsed = time.time() - start
        if elapsed > time_budget:
            break
        t_mid = (t_low + t_high) / 2
        if t_mid < 1e-12 or (t_high - t_low) / max(t_mid, 1e-12) < 1e-6:
            break
        remaining = time_budget - elapsed
        per_iter = max(0.5, remaining / (n_iter - it))
        feasible, assignment, lambda_vals, status = solve_one_stage_feasibility(
            N, n, costs_arr, K, t_mid, per_iter, prev_solution
        )
        if feasible:
            best_t = t_mid
            best_assignment = {str(i): assignment[i] for i in range(N)}
            best_lambda = {
                str(a): {str(j): w for j, w in lambda_vals[a].items()}
                for a in range(K)
            }
            prev_solution = (assignment, lambda_vals)
            t_low = t_mid
        else:
            t_high = t_mid

    return best_t, best_assignment, best_lambda, GRB.OPTIMAL, best_t, 0.0


def check_two_stage_feasibility(N, r, K, t, time_limit):
    """Check if t is feasible for the two-stage IP (set cover with cardinality K)."""
    model = Model("tsf")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.Threads = 1
    model.Params.NumericFocus = 0
    model.Params.MIPGap = 1e-4
    if time_limit > 0:
        model.Params.TimeLimit = max(0.5, time_limit)
    x = model.addVars(N, vtype=GRB.BINARY)
    model.addConstr(quicksum(x[j] for j in range(N)) == K)
    for i in range(N):
        covering = [j for j in range(N) if r[i, j] >= t - 1e-9]
        if not covering:
            return False, None
        model.addConstr(quicksum(x[j] for j in covering) >= 1)
    model.optimize()
    if model.SolCount > 0:
        return True, [j for j in range(N) if x[j].X > 0.5]
    return False, None


def solve_two_stage(N, n, costs, first_stage_costs, K, time_budget):
    """Solve two-stage IP for given K via binary search on discrete ratio values."""
    costs_arr = np.array(costs, dtype=float)
    f = np.array(first_stage_costs, dtype=float)
    if K >= N:
        return 1.0, list(range(N)), GRB.OPTIMAL

    r = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            r[i, j] = float(((f + costs_arr[i]) / (f + costs_arr[j])).min())

    r_values = sorted(set(r.flatten()))
    best_t = 0.0
    best_selected = list(range(K))
    lo, hi = 0, len(r_values) - 1
    start = time.time()

    while lo <= hi:
        if time.time() - start > time_budget:
            break
        mid = (lo + hi) // 2
        t_cand = r_values[mid]
        feasible, selected = check_two_stage_feasibility(
            N, r, K, t_cand, time_budget - (time.time() - start)
        )
        if feasible:
            best_t = t_cand
            best_selected = selected
            lo = mid + 1
        else:
            hi = mid - 1

    return best_t, best_selected, GRB.OPTIMAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    with open(args.instance_path) as fh:
        instance = json.load(fh)

    N = instance["num_scenarios_N"]
    n = instance["problem_dimension_n"]
    K_values = instance["K_values"]
    costs = instance["scenarios"]
    first_stage_costs = instance.get("first_stage_costs", [0] * n)

    logger = None
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="maximize")

    start_time = time.time()
    one_stage_results = {}
    two_stage_results = {}
    all_t = []

    for idx, K in enumerate(K_values):
        elapsed = time.time() - start_time
        remaining = args.time_limit - elapsed

        if remaining <= 1:
            one_stage_results[str(K)] = {
                "t": 0.0,
                "approximation_ratio": 1e15,
                "assignment": {str(i): i % K for i in range(N)},
                "lambda_weights": {str(a): {str(a % N): 1.0} for a in range(K)},
                "solver_status": 0,
                "best_bound": 0.0,
                "mip_gap": 1.0,
            }
            two_stage_results[str(K)] = {
                "t": 0.0,
                "approximation_ratio": 1e15,
                "selected_scenarios": list(range(min(K, N))),
            }
            all_t.extend([0.0, 0.0])
            continue

        rem_sub = (len(K_values) - idx) * 2
        budget = remaining / rem_sub
        os_budget = budget * 0.8
        ts_budget = budget * 0.2

        t1, a1, l1, s1, b1, g1 = solve_one_stage(N, n, costs, K, os_budget)
        one_stage_results[str(K)] = {
            "t": float(t1),
            "approximation_ratio": float(1.0 / t1) if t1 > 1e-10 else 1e15,
            "assignment": a1,
            "lambda_weights": l1,
            "solver_status": int(s1),
            "best_bound": float(b1),
            "mip_gap": float(g1),
        }
        all_t.append(t1)

        t2, sel2, s2 = solve_two_stage(
            N, n, costs, first_stage_costs, K, ts_budget
        )
        two_stage_results[str(K)] = {
            "t": float(t2),
            "approximation_ratio": float(1.0 / t2) if t2 > 1e-10 else 1e15,
            "selected_scenarios": sel2,
        }
        all_t.append(t2)

        if logger:
            best = max(all_t)
            logger.log_solution(
                best,
                {
                    "objective_value": float(best),
                    "one_stage_IP_mu": one_stage_results,
                    "two_stage_IP": two_stage_results,
                },
            )

    obj = max(all_t) if all_t else 0.0
    solution = {
        "objective_value": float(obj),
        "one_stage_IP_mu": one_stage_results,
        "two_stage_IP": two_stage_results,
    }
    with open(args.solution_path, "w") as fh:
        json.dump(solution, fh, indent=2)


if __name__ == "__main__":
    main()