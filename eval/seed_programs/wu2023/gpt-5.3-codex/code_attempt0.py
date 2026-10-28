import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


EPS = 1e-9
BIG_RATIO = 1e12


def approx_ratio_from_t(t: float) -> float:
    return 1.0 / t if t > 1e-12 else BIG_RATIO


def cap_t(t: float) -> float:
    if t is None or math.isnan(t):
        return 0.0
    return max(0.0, min(1.0, float(t)))


def compute_ratio_matrix(scenarios: List[List[float]]) -> List[List[float]]:
    n_scen = len(scenarios)
    n_dim = len(scenarios[0]) if n_scen > 0 else 0
    r = [[0.0] * n_scen for _ in range(n_scen)]
    for i in range(n_scen):
        ci = scenarios[i]
        for j in range(n_scen):
            cj = scenarios[j]
            m = float("inf")
            for d in range(n_dim):
                val = ci[d] / cj[d]
                if val < m:
                    m = val
            r[i][j] = m
    return r


def greedy_sequence(r: List[List[float]], max_k: int) -> List[List[int]]:
    n = len(r)
    if n == 0 or max_k <= 0:
        return [[] for _ in range(max_k + 1)]
    max_k_eff = min(max_k, n)
    seq = [[] for _ in range(max_k + 1)]

    # first center
    best_j = 0
    best_val = -1.0
    for j in range(n):
        val = min(r[i][j] for i in range(n))
        if val > best_val + 1e-15:
            best_val = val
            best_j = j

    selected = [best_j]
    covered = [r[i][best_j] for i in range(n)]
    seq[1] = selected.copy()

    selected_set = {best_j}
    for k in range(2, max_k_eff + 1):
        best_cand = None
        best_score = -1.0
        for j in range(n):
            if j in selected_set:
                continue
            score = float("inf")
            for i in range(n):
                v = covered[i]
                if r[i][j] > v:
                    v = r[i][j]
                if v < score:
                    score = v
                if score <= best_score:
                    break
            if score > best_score + 1e-15:
                best_score = score
                best_cand = j
        if best_cand is None:
            break
        selected.append(best_cand)
        selected_set.add(best_cand)
        for i in range(n):
            if r[i][best_cand] > covered[i]:
                covered[i] = r[i][best_cand]
        seq[k] = selected.copy()

    # fill possibly missing levels
    last = []
    for k in range(1, max_k + 1):
        if k <= max_k_eff and seq[k]:
            last = seq[k]
        seq[k] = last.copy()
    return seq


def build_selected_for_k(seq: List[List[int]], k: int, n: int) -> List[int]:
    if k <= 0:
        return []
    base = seq[min(k, len(seq) - 1)].copy() if len(seq) > 1 else []
    if not base and n > 0:
        base = [0]
    if k <= n:
        return base[:k]
    # k > n: repeat first index to satisfy required length
    out = base[:]
    while len(out) < k:
        out.append(out[0] if out else 0)
    return out[:k]


def evaluate_two_stage_t(selected: List[int], r: List[List[float]]) -> float:
    n = len(r)
    if n == 0 or not selected:
        return 0.0
    uniq = list(dict.fromkeys(selected))
    t = float("inf")
    for i in range(n):
        best = -1.0
        for j in uniq:
            v = r[i][j]
            if v > best:
                best = v
        if best < t:
            t = best
    return cap_t(t)


def two_stage_assignment_from_selected(selected: List[int], r: List[List[float]]) -> List[int]:
    n = len(r)
    k = len(selected)
    if n == 0 or k == 0:
        return []
    assign = [0] * n
    for i in range(n):
        best_k = 0
        best_v = -1.0
        for kk in range(k):
            j = selected[kk]
            v = r[i][j]
            if v > best_v + 1e-15:
                best_v = v
                best_k = kk
        assign[i] = best_k
    return assign


def one_stage_heuristic_from_selected(
    selected_for_agg: List[int], scenarios: List[List[float]], r: List[List[float]]
) -> Tuple[float, Dict[int, int], Dict[int, Dict[int, float]]]:
    n = len(scenarios)
    k = len(selected_for_agg)
    if n == 0 or k == 0:
        return 0.0, {}, {}

    assignment_list = two_stage_assignment_from_selected(selected_for_agg, r)
    assignment = {i: assignment_list[i] for i in range(n)}

    lambda_weights = {}
    for kk, sc in enumerate(selected_for_agg):
        lambda_weights[kk] = {int(sc): 1.0}

    t = evaluate_one_stage_witness(scenarios, assignment, lambda_weights, k)
    return t, assignment, lambda_weights


def evaluate_one_stage_witness(
    scenarios: List[List[float]],
    assignment: Dict[int, int],
    lambda_weights: Dict[int, Dict[int, float]],
    k: int,
) -> float:
    n_scen = len(scenarios)
    if n_scen == 0 or k <= 0:
        return 0.0
    n_dim = len(scenarios[0])
    agg = [[0.0] * n_dim for _ in range(k)]

    for kk in range(k):
        lw = lambda_weights.get(kk, {})
        s = sum(max(0.0, float(v)) for v in lw.values())
        if s <= 1e-15:
            # fallback one-hot at scenario 0
            lw = {0: 1.0}
            s = 1.0
        for j_str, val in lw.items():
            j = int(j_str)
            lam = max(0.0, float(val)) / s
            cj = scenarios[j]
            for d in range(n_dim):
                agg[kk][d] += lam * cj[d]

    t = float("inf")
    for i in range(n_scen):
        kk = int(assignment[i])
        ci = scenarios[i]
        ak = agg[kk]
        for d in range(n_dim):
            denom = ak[d]
            if denom <= 0:
                return 0.0
            val = ci[d] / denom
            if val < t:
                t = val
    return cap_t(t)


def solve_two_stage_mip(
    scenarios: List[List[float]],
    r: List[List[float]],
    K: int,
    time_limit: float,
    heuristic_selected: List[int],
) -> Tuple[float, List[int], int]:
    n = len(scenarios)
    if n == 0 or K <= 0:
        return 0.0, [], 2

    K_eff = min(K, n)
    if time_limit <= 1e-3:
        return evaluate_two_stage_t(heuristic_selected, r), heuristic_selected[:K], 9

    model = gp.Model("two_stage_ip")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.1, time_limit)

    t = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="t")
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    model.addConstr(gp.quicksum(y[j] for j in range(n)) == K_eff, name="selectK")
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f"assign_{i}")
        for j in range(n):
            model.addConstr(x[i, j] <= y[j], name=f"link_{i}_{j}")

    for j in range(n):
        model.addConstr(x[j, j] >= y[j], name=f"self_{j}")

    # condensed ratio constraints
    for i in range(n):
        for j in range(n):
            rij = r[i][j]
            rij_eff = 1.0 if rij >= 1.0 else rij
            coef = 1.0 - rij_eff
            if coef > 1e-12:
                model.addConstr(t + coef * x[i, j] <= 1.0, name=f"ratio_{i}_{j}")

    model.setObjective(t, GRB.MAXIMIZE)

    # warm start
    hs = heuristic_selected[:K_eff]
    hs_set = set(hs)
    for j in range(n):
        y[j].Start = 1.0 if j in hs_set else 0.0
    if hs:
        assign0 = two_stage_assignment_from_selected(hs, r)
        for i in range(n):
            jsel = hs[assign0[i]]
            for j in range(n):
                x[i, j].Start = 1.0 if j == jsel else 0.0
        for j in hs:
            x[j, j].Start = 1.0
    t.Start = evaluate_two_stage_t(hs, r)

    model.optimize()
    status = int(model.Status)

    if model.SolCount > 0:
        yvals = [(j, y[j].X) for j in range(n)]
        chosen = [j for j, v in yvals if v > 0.5]
        if len(chosen) > K_eff:
            chosen.sort(key=lambda jj: y[jj].X, reverse=True)
            chosen = chosen[:K_eff]
        elif len(chosen) < K_eff:
            remain = [j for j, _ in sorted(yvals, key=lambda p: p[1], reverse=True) if j not in chosen]
            chosen.extend(remain[: K_eff - len(chosen)])
        if not chosen:
            chosen = hs[:K_eff] if hs else list(range(K_eff))
    else:
        chosen = hs[:K_eff] if hs else list(range(K_eff))

    # match requested length K
    out_selected = chosen.copy()
    while len(out_selected) < K:
        out_selected.append(out_selected[0] if out_selected else 0)
    out_selected = out_selected[:K]

    t_val = evaluate_two_stage_t(out_selected, r)
    return t_val, out_selected, status


def solve_one_stage_mip(
    scenarios: List[List[float]],
    K: int,
    time_limit: float,
    heuristic_assignment: Dict[int, int],
    heuristic_lambda: Dict[int, Dict[int, float]],
    heuristic_t: float,
) -> Tuple[float, Dict[int, int], Dict[int, Dict[int, float]], int, float, float]:
    n = len(scenarios)
    n_dim = len(scenarios[0]) if n > 0 else 0
    if n == 0 or K <= 0:
        return 0.0, {}, {}, 2, 0.0, 0.0

    if time_limit <= 1e-3:
        return heuristic_t, heuristic_assignment, heuristic_lambda, 9, heuristic_t, 0.0

    M = [max(scenarios[i][d] for i in range(n)) for d in range(n_dim)]

    model = gp.Model("one_stage_ip_mu")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0.1, time_limit)

    t = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="t")
    x = model.addVars(n, K, vtype=GRB.BINARY, name="x")
    w = model.addVars(K, n, lb=0.0, vtype=GRB.CONTINUOUS, name="w")
    v = model.addVars(K, n_dim, lb=0.0, vtype=GRB.CONTINUOUS, name="v")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(K)) == 1, name=f"assign_{i}")
    if K <= n:
        for k in range(K):
            model.addConstr(gp.quicksum(x[i, k] for i in range(n)) >= 1, name=f"use_{k}")

    for k in range(K):
        model.addConstr(gp.quicksum(w[k, j] for j in range(n)) == t, name=f"sumw_{k}")
        for d in range(n_dim):
            model.addConstr(
                v[k, d] == gp.quicksum(w[k, j] * scenarios[j][d] for j in range(n)),
                name=f"vdef_{k}_{d}",
            )

    for i in range(n):
        for k in range(K):
            for d in range(n_dim):
                model.addConstr(
                    v[k, d] <= scenarios[i][d] + M[d] * (1 - x[i, k]),
                    name=f"rep_{i}_{k}_{d}",
                )

    model.setObjective(t, GRB.MAXIMIZE)

    # warm start
    for i in range(n):
        ai = heuristic_assignment.get(i, 0)
        for k in range(K):
            x[i, k].Start = 1.0 if k == ai else 0.0

    t.Start = cap_t(heuristic_t)
    for k in range(K):
        lw = heuristic_lambda.get(k, {})
        s = sum(max(0.0, float(val)) for val in lw.values())
        if s <= 1e-15:
            lw = {0: 1.0}
            s = 1.0
        for j in range(n):
            lam = max(0.0, float(lw.get(j, 0.0))) / s
            w[k, j].Start = t.Start * lam
        for d in range(n_dim):
            v[k, d].Start = sum(
                (w[k, j].Start if w[k, j].Start is not None else 0.0) * scenarios[j][d]
                for j in range(n)
            )

    model.optimize()
    status = int(model.Status)
    best_bound = float(model.ObjBound) if model.Status != GRB.LOADED else 0.0
    mip_gap = float(model.MIPGap) if model.SolCount > 0 else 1.0

    if model.SolCount > 0 and t.X > 1e-12:
        t_sol = float(t.X)
        assign = {}
        for i in range(n):
            bestk = 0
            bestv = -1.0
            for k in range(K):
                xv = x[i, k].X
                if xv > bestv:
                    bestv = xv
                    bestk = k
            assign[i] = int(bestk)

        lambdas: Dict[int, Dict[int, float]] = {}
        for k in range(K):
            vals = []
            s = 0.0
            for j in range(n):
                val = w[k, j].X
                if val > 1e-10:
                    vals.append((j, val))
                    s += val
            if s <= 1e-12:
                lambdas[k] = {0: 1.0}
            else:
                # normalize by sum (numerically safer than divide by t_sol)
                inner = {}
                for j, val in vals:
                    lam = val / s
                    if lam > 1e-12:
                        inner[int(j)] = float(lam)
                if not inner:
                    inner = {0: 1.0}
                lambdas[k] = inner

        t_eval = evaluate_one_stage_witness(scenarios, assign, lambdas, K)
        return t_eval, assign, lambdas, status, best_bound, mip_gap

    # fallback heuristic
    t_eval = evaluate_one_stage_witness(scenarios, heuristic_assignment, heuristic_lambda, K)
    return t_eval, heuristic_assignment, heuristic_lambda, status, best_bound, mip_gap


def build_solution_template(K_values: List[int]) -> Dict:
    return {
        "objective_value": 0.0,
        "one_stage_IP_mu": {str(K): {} for K in K_values},
        "two_stage_IP": {str(K): {} for K in K_values},
    }


def recompute_objective(solution: Dict, K_values: List[int]) -> float:
    best = 0.0
    for K in K_values:
        k = str(K)
        if solution["one_stage_IP_mu"].get(k) and "t" in solution["one_stage_IP_mu"][k]:
            best = max(best, float(solution["one_stage_IP_mu"][k]["t"]))
        if solution["two_stage_IP"].get(k) and "t" in solution["two_stage_IP"][k]:
            best = max(best, float(solution["two_stage_IP"][k]["t"]))
    return float(best)


def maybe_log(logger: SolutionLogger, solution: Dict, best_logged_metric: List[float]):
    if logger is None:
        return
    obj = float(solution.get("objective_value", 0.0))
    metric = approx_ratio_from_t(obj)  # minimize
    if metric + 1e-12 < best_logged_metric[0]:
        best_logged_metric[0] = metric
        logger.log_solution(metric, solution)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_logged_metric = [float("inf")]

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    scenarios = inst["scenarios"]
    N = int(inst["num_scenarios_N"])
    n_dim = int(inst["problem_dimension_n"])
    K_values = [int(k) for k in inst["K_values"]]

    # basic consistency fallback
    if len(scenarios) != N:
        N = len(scenarios)
    if N > 0 and len(scenarios[0]) != n_dim:
        n_dim = len(scenarios[0])

    r = compute_ratio_matrix(scenarios)

    solution = build_solution_template(K_values)

    # Heuristic initialization for all K
    maxK = max(K_values) if K_values else 0
    seq = greedy_sequence(r, min(maxK, N) if N > 0 else 0)

    heuristic_cache = {}
    for K in K_values:
        kstr = str(K)
        selected = build_selected_for_k(seq, K, N)
        t2 = evaluate_two_stage_t(selected, r)
        solution["two_stage_IP"][kstr] = {
            "t": float(t2),
            "approximation_ratio": float(approx_ratio_from_t(t2)),
            "selected_scenarios": [int(x) for x in selected],
        }

        t1, assignment, lambdas = one_stage_heuristic_from_selected(selected, scenarios, r)
        solution["one_stage_IP_mu"][kstr] = {
            "t": float(t1),
            "approximation_ratio": float(approx_ratio_from_t(t1)),
            "assignment": {str(i): int(assignment[i]) for i in range(N)},
            "lambda_weights": {
                str(k): {str(j): float(v) for j, v in lambdas.get(k, {}).items()}
                for k in range(K)
            },
            "solver_status": 9,
            "best_bound": float(t1),
            "mip_gap": 0.0,
        }
        heuristic_cache[K] = (selected, t1, assignment, lambdas)

    solution["objective_value"] = recompute_objective(solution, K_values)
    maybe_log(logger, solution, best_logged_metric)

    # Exact improvement passes
    tasks = [("two_stage", K) for K in K_values] + [("one_stage", K) for K in K_values]
    for idx, (kind, K) in enumerate(tasks):
        elapsed = time.time() - start
        remaining = args.time_limit - elapsed
        if remaining <= 0.05:
            break

        tasks_left = len(tasks) - idx
        alloc = max(0.1, remaining / max(1, tasks_left))

        kstr = str(K)
        selected_h, t1_h, assign_h, lambda_h = heuristic_cache[K]

        if kind == "two_stage":
            t2, selected, status = solve_two_stage_mip(scenarios, r, K, alloc, selected_h)
            solution["two_stage_IP"][kstr] = {
                "t": float(t2),
                "approximation_ratio": float(approx_ratio_from_t(t2)),
                "selected_scenarios": [int(x) for x in selected],
            }

            # refresh one-stage heuristic from improved selected (quick local uplift)
            t1_new, assign_new, lambda_new = one_stage_heuristic_from_selected(selected, scenarios, r)
            old_t1 = solution["one_stage_IP_mu"][kstr]["t"]
            if t1_new > old_t1 + 1e-12:
                solution["one_stage_IP_mu"][kstr].update(
                    {
                        "t": float(t1_new),
                        "approximation_ratio": float(approx_ratio_from_t(t1_new)),
                        "assignment": {str(i): int(assign_new[i]) for i in range(N)},
                        "lambda_weights": {
                            str(k): {str(j): float(v) for j, v in lambda_new.get(k, {}).items()}
                            for k in range(K)
                        },
                        "solver_status": int(status),
                        "best_bound": float(max(solution["one_stage_IP_mu"][kstr].get("best_bound", 0.0), t1_new)),
                        "mip_gap": float(solution["one_stage_IP_mu"][kstr].get("mip_gap", 0.0)),
                    }
                )
                heuristic_cache[K] = (selected, t1_new, assign_new, lambda_new)

        else:  # one_stage
            # Skip very large one-stage models if likely too heavy
            complexity = N * max(1, K) * max(1, N) + N * max(1, K) * max(1, n_dim)
            if complexity > 8_000_000 and alloc < 5.0:
                continue

            t1, assign, lambdas, status, bound, gap = solve_one_stage_mip(
                scenarios=scenarios,
                K=K,
                time_limit=alloc,
                heuristic_assignment=assign_h,
                heuristic_lambda=lambda_h,
                heuristic_t=t1_h,
            )
            solution["one_stage_IP_mu"][kstr] = {
                "t": float(t1),
                "approximation_ratio": float(approx_ratio_from_t(t1)),
                "assignment": {str(i): int(assign[i]) for i in range(N)},
                "lambda_weights": {
                    str(k): {str(j): float(v) for j, v in lambdas.get(k, {}).items()}
                    for k in range(K)
                },
                "solver_status": int(status),
                "best_bound": float(bound),
                "mip_gap": float(gap),
            }

        new_obj = recompute_objective(solution, K_values)
        if new_obj > solution["objective_value"] + 1e-12:
            solution["objective_value"] = float(new_obj)
            maybe_log(logger, solution, best_logged_metric)
        else:
            solution["objective_value"] = float(new_obj)

    # final write
    os.makedirs(os.path.dirname(args.solution_path), exist_ok=True) if os.path.dirname(args.solution_path) else None
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()