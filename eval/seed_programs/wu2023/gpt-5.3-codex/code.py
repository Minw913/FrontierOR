import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


BIG_RATIO = 1e12


def cap_t(t: float) -> float:
    if t is None:
        return 0.0
    try:
        if math.isnan(float(t)):
            return 0.0
    except Exception:
        return 0.0
    return max(0.0, min(1.0, float(t)))


def approx_ratio_from_t(t: float) -> float:
    t = cap_t(t)
    return (1.0 / t) if t > 1e-12 else BIG_RATIO


def compute_ratio_matrix(scenarios: List[List[float]]) -> List[List[float]]:
    n = len(scenarios)
    d = len(scenarios[0]) if n > 0 else 0
    r = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            m = float("inf")
            for k in range(d):
                val = scenarios[i][k] / scenarios[j][k]
                if val < m:
                    m = val
            r[i][j] = m
    return r


def greedy_sequence(r: List[List[float]], max_k: int) -> List[List[int]]:
    n = len(r)
    seq = [[] for _ in range(max_k + 1)]
    if n == 0 or max_k <= 0:
        return seq

    # first selected
    best_j, best_val = 0, -1.0
    for j in range(n):
        v = min(r[i][j] for i in range(n))
        if v > best_val:
            best_val = v
            best_j = j

    selected = [best_j]
    selected_set = {best_j}
    covered = [r[i][best_j] for i in range(n)]
    seq[1] = selected.copy()

    for k in range(2, min(max_k, n) + 1):
        cand, cand_score = None, -1.0
        for j in range(n):
            if j in selected_set:
                continue
            score = float("inf")
            for i in range(n):
                val = max(covered[i], r[i][j])
                if val < score:
                    score = val
                if score <= cand_score:
                    break
            if score > cand_score:
                cand_score = score
                cand = j
        if cand is None:
            break
        selected.append(cand)
        selected_set.add(cand)
        for i in range(n):
            covered[i] = max(covered[i], r[i][cand])
        seq[k] = selected.copy()

    last = []
    for k in range(1, max_k + 1):
        if seq[k]:
            last = seq[k]
        seq[k] = last.copy()
    return seq


def selected_for_k(seq: List[List[int]], K: int, n: int) -> List[int]:
    if K <= 0:
        return []
    base = seq[min(K, len(seq) - 1)].copy() if len(seq) > 1 else []
    if not base and n > 0:
        base = [0]
    out = base[:K]
    while len(out) < K:
        out.append(out[0] if out else 0)
    return out


def evaluate_two_stage_t(selected: List[int], r: List[List[float]]) -> float:
    if not r or not selected:
        return 0.0
    uniq = list(dict.fromkeys(selected))
    best_min = float("inf")
    n = len(r)
    for i in range(n):
        best = -1.0
        for j in uniq:
            best = max(best, r[i][j])
        best_min = min(best_min, best)
    return cap_t(best_min)


def assignment_from_selected(selected: List[int], r: List[List[float]]) -> List[int]:
    n = len(r)
    k = len(selected)
    a = [0] * n
    if n == 0 or k == 0:
        return a
    for i in range(n):
        bi, bv = 0, -1.0
        for kk in range(k):
            v = r[i][selected[kk]]
            if v > bv:
                bv = v
                bi = kk
        a[i] = bi
    return a


def sanitize_lambda(lambda_weights: Dict, K: int, n: int) -> Dict[int, Dict[int, float]]:
    out: Dict[int, Dict[int, float]] = {}
    for k in range(K):
        raw = lambda_weights.get(k, lambda_weights.get(str(k), {}))
        inner: Dict[int, float] = {}
        if isinstance(raw, dict):
            for jk, val in raw.items():
                try:
                    j = int(jk)
                    v = float(val)
                    if 0 <= j < n and v > 0:
                        inner[j] = inner.get(j, 0.0) + v
                except Exception:
                    pass
        if not inner:
            inner = {0: 1.0} if n > 0 else {}
        s = sum(inner.values())
        if s <= 1e-15 and n > 0:
            inner = {0: 1.0}
            s = 1.0
        if s > 1e-15:
            inner = {j: v / s for j, v in inner.items() if v / s > 1e-15}
        out[k] = inner if inner else ({0: 1.0} if n > 0 else {})
    return out


def evaluate_one_stage_witness(
    scenarios: List[List[float]],
    assignment: Dict[int, int],
    lambda_weights: Dict[int, Dict[int, float]],
    K: int,
) -> float:
    n = len(scenarios)
    if n == 0 or K <= 0:
        return 0.0
    d = len(scenarios[0])
    lambdas = sanitize_lambda(lambda_weights, K, n)

    agg = [[0.0] * d for _ in range(K)]
    for k in range(K):
        for j, w in lambdas[k].items():
            for dd in range(d):
                agg[k][dd] += w * scenarios[j][dd]

    t = float("inf")
    for i in range(n):
        k = int(assignment.get(i, assignment.get(str(i), 0)))
        if not (0 <= k < K):
            k = 0
        for dd in range(d):
            den = agg[k][dd]
            if den <= 0:
                return 0.0
            t = min(t, scenarios[i][dd] / den)
    return cap_t(t)


def one_stage_heuristic(selected: List[int], scenarios: List[List[float]], r: List[List[float]]):
    n = len(scenarios)
    K = len(selected)
    a_list = assignment_from_selected(selected, r)
    assignment = {i: int(a_list[i]) for i in range(n)}
    lambdas = {k: {int(selected[k]): 1.0} for k in range(K)}
    t = evaluate_one_stage_witness(scenarios, assignment, lambdas, K)
    return t, assignment, lambdas


def solve_two_stage_mip(
    scenarios: List[List[float]],
    r: List[List[float]],
    K: int,
    time_limit: float,
    heuristic_selected: List[int],
):
    n = len(scenarios)
    if n == 0 or K <= 0:
        return 0.0, [], 2
    K_eff = min(K, n)
    if time_limit <= 0.05:
        return evaluate_two_stage_t(heuristic_selected, r), heuristic_selected[:K], 9

    try:
        m = gp.Model("two_stage")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = max(0.1, time_limit)

        t = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="t")
        y = m.addVars(n, vtype=GRB.BINARY, name="y")
        x = m.addVars(n, n, vtype=GRB.BINARY, name="x")

        m.addConstr(gp.quicksum(y[j] for j in range(n)) == K_eff)
        for i in range(n):
            m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
            for j in range(n):
                m.addConstr(x[i, j] <= y[j])
        for j in range(n):
            m.addConstr(x[j, j] >= y[j])

        for i in range(n):
            for j in range(n):
                rij = min(1.0, r[i][j])
                coef = 1.0 - rij
                if coef > 1e-12:
                    m.addConstr(t + coef * x[i, j] <= 1.0)

        hs = heuristic_selected[:K_eff] if heuristic_selected else list(range(K_eff))
        hs_set = set(hs)
        for j in range(n):
            y[j].Start = 1.0 if j in hs_set else 0.0
        if hs:
            a0 = assignment_from_selected(hs, r)
            for i in range(n):
                j0 = hs[a0[i]]
                x[i, j0].Start = 1.0
            for j0 in hs:
                x[j0, j0].Start = 1.0
        t.Start = evaluate_two_stage_t(hs, r)

        m.setObjective(t, GRB.MAXIMIZE)
        m.optimize()

        status = int(m.Status)
        if m.SolCount > 0:
            yv = [(j, float(y[j].X)) for j in range(n)]
            chosen = [j for j, v in yv if v > 0.5]
            if len(chosen) < K_eff:
                rem = [j for j, _ in sorted(yv, key=lambda z: z[1], reverse=True) if j not in chosen]
                chosen.extend(rem[: K_eff - len(chosen)])
            elif len(chosen) > K_eff:
                chosen = sorted(chosen, key=lambda j: y[j].X, reverse=True)[:K_eff]
        else:
            chosen = hs

        out = chosen[:]
        while len(out) < K:
            out.append(out[0] if out else 0)
        out = out[:K]
        return evaluate_two_stage_t(out, r), out, status
    except Exception:
        hs = heuristic_selected[:]
        while len(hs) < K:
            hs.append(hs[0] if hs else 0)
        hs = hs[:K]
        return evaluate_two_stage_t(hs, r), hs, 9


def solve_one_stage_mip(
    scenarios: List[List[float]],
    K: int,
    time_limit: float,
    heuristic_assignment: Dict[int, int],
    heuristic_lambda: Dict[int, Dict[int, float]],
    heuristic_t: float,
):
    n = len(scenarios)
    d = len(scenarios[0]) if n > 0 else 0
    if n == 0 or K <= 0:
        return 0.0, {}, {}, 2, 0.0, 0.0
    if time_limit <= 0.05:
        lam = sanitize_lambda(heuristic_lambda, K, n)
        t = evaluate_one_stage_witness(scenarios, heuristic_assignment, lam, K)
        return t, heuristic_assignment, lam, 9, t, 0.0

    lam_h = sanitize_lambda(heuristic_lambda, K, n)

    try:
        M = [max(scenarios[i][dd] for i in range(n)) for dd in range(d)]

        m = gp.Model("one_stage_ip_mu")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = max(0.1, time_limit)

        t = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="t")
        x = m.addVars(n, K, vtype=GRB.BINARY, name="x")
        w = m.addVars(K, n, lb=0, vtype=GRB.CONTINUOUS, name="w")
        v = m.addVars(K, d, lb=0, vtype=GRB.CONTINUOUS, name="v")

        for i in range(n):
            m.addConstr(gp.quicksum(x[i, k] for k in range(K)) == 1)
        if K <= n:
            for k in range(K):
                m.addConstr(gp.quicksum(x[i, k] for i in range(n)) >= 1)

        for k in range(K):
            m.addConstr(gp.quicksum(w[k, j] for j in range(n)) == t)
            for dd in range(d):
                m.addConstr(v[k, dd] == gp.quicksum(w[k, j] * scenarios[j][dd] for j in range(n)))

        for i in range(n):
            for k in range(K):
                for dd in range(d):
                    m.addConstr(v[k, dd] <= scenarios[i][dd] + M[dd] * (1 - x[i, k]))

        # warm start (safe)
        t_start = cap_t(heuristic_t)
        t.Start = t_start
        for i in range(n):
            ai = heuristic_assignment.get(i, heuristic_assignment.get(str(i), 0))
            try:
                ai = int(ai)
            except Exception:
                ai = 0
            if ai < 0 or ai >= K:
                ai = 0
            x[i, ai].Start = 1.0

        for k in range(K):
            lk = lam_h.get(k, {})
            for j in range(n):
                w[k, j].Start = t_start * float(lk.get(j, 0.0))
            for dd in range(d):
                v[k, dd].Start = sum((w[k, j].Start or 0.0) * scenarios[j][dd] for j in range(n))

        m.setObjective(t, GRB.MAXIMIZE)
        m.optimize()

        status = int(m.Status)
        best_bound = float(m.ObjBound) if m.Status != GRB.LOADED else 0.0
        mip_gap = float(m.MIPGap) if m.SolCount > 0 else 1.0

        if m.SolCount > 0:
            assignment = {}
            for i in range(n):
                bk, bv = 0, -1.0
                for k in range(K):
                    xv = float(x[i, k].X)
                    if xv > bv:
                        bv = xv
                        bk = k
                assignment[i] = bk

            lambdas = {}
            for k in range(K):
                vals = {}
                s = 0.0
                for j in range(n):
                    val = float(w[k, j].X)
                    if val > 1e-12:
                        vals[j] = val
                        s += val
                if s <= 1e-15:
                    lambdas[k] = {0: 1.0}
                else:
                    lambdas[k] = {j: vals[j] / s for j in vals if vals[j] / s > 1e-15}
                    if not lambdas[k]:
                        lambdas[k] = {0: 1.0}

            t_eval = evaluate_one_stage_witness(scenarios, assignment, lambdas, K)
            return t_eval, assignment, lambdas, status, best_bound, mip_gap

        t_fallback = evaluate_one_stage_witness(scenarios, heuristic_assignment, lam_h, K)
        return t_fallback, heuristic_assignment, lam_h, status, best_bound, mip_gap

    except Exception:
        t_fallback = evaluate_one_stage_witness(scenarios, heuristic_assignment, lam_h, K)
        return t_fallback, heuristic_assignment, lam_h, 9, t_fallback, 0.0


def build_solution_template(K_values: List[int]) -> Dict:
    return {
        "objective_value": 0.0,
        "one_stage_IP_mu": {str(K): {} for K in K_values},
        "two_stage_IP": {str(K): {} for K in K_values},
    }


def recompute_objective(sol: Dict, K_values: List[int]) -> float:
    best = 0.0
    for K in K_values:
        k = str(K)
        if "t" in sol["one_stage_IP_mu"].get(k, {}):
            best = max(best, cap_t(sol["one_stage_IP_mu"][k]["t"]))
        if "t" in sol["two_stage_IP"].get(k, {}):
            best = max(best, cap_t(sol["two_stage_IP"][k]["t"]))
    return best


def maybe_log(logger: SolutionLogger, sol: Dict, best_metric_box: List[float]):
    if logger is None:
        return
    metric = approx_ratio_from_t(sol.get("objective_value", 0.0))  # minimize
    if metric + 1e-12 < best_metric_box[0]:
        best_metric_box[0] = metric
        logger.log_solution(metric, sol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_metric_box = [float("inf")]
    start = time.time()

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    scenarios = inst["scenarios"]
    N = int(inst["num_scenarios_N"])
    K_values = [int(k) for k in inst["K_values"]]

    if len(scenarios) != N:
        N = len(scenarios)

    r = compute_ratio_matrix(scenarios)
    sol = build_solution_template(K_values)

    maxK = max(K_values) if K_values else 0
    seq = greedy_sequence(r, min(maxK, N) if N > 0 else 0)

    heuristic_cache = {}

    # Initial feasible solution for all K
    for K in K_values:
        k = str(K)
        selected = selected_for_k(seq, K, N)
        t2 = evaluate_two_stage_t(selected, r)
        sol["two_stage_IP"][k] = {
            "t": float(t2),
            "approximation_ratio": float(approx_ratio_from_t(t2)),
            "selected_scenarios": [int(x) for x in selected],
        }

        t1, assign, lambdas = one_stage_heuristic(selected, scenarios, r)
        heuristic_cache[K] = (selected, t1, assign, lambdas)
        sol["one_stage_IP_mu"][k] = {
            "t": float(t1),
            "approximation_ratio": float(approx_ratio_from_t(t1)),
            "assignment": {str(i): int(assign[i]) for i in range(N)},
            "lambda_weights": {
                str(kk): {str(j): float(v) for j, v in lambdas.get(kk, {}).items()}
                for kk in range(K)
            },
            "solver_status": 9,
            "best_bound": float(t1),
            "mip_gap": 0.0,
        }

    sol["objective_value"] = float(recompute_objective(sol, K_values))
    maybe_log(logger, sol, best_metric_box)

    # Improvement pass
    tasks = [("two", K) for K in K_values] + [("one", K) for K in K_values]

    for idx, (kind, K) in enumerate(tasks):
        elapsed = time.time() - start
        remaining = args.time_limit - elapsed
        if remaining <= 0.05:
            break

        budget = max(0.1, remaining / max(1, len(tasks) - idx))
        selected_h, t1_h, assign_h, lambdas_h = heuristic_cache[K]
        k = str(K)

        if kind == "two":
            t2, selected, status = solve_two_stage_mip(scenarios, r, K, budget, selected_h)
            sol["two_stage_IP"][k] = {
                "t": float(t2),
                "approximation_ratio": float(approx_ratio_from_t(t2)),
                "selected_scenarios": [int(x) for x in selected],
            }

            t1_new, a_new, l_new = one_stage_heuristic(selected, scenarios, r)
            if t1_new > cap_t(sol["one_stage_IP_mu"][k]["t"]) + 1e-12:
                heuristic_cache[K] = (selected, t1_new, a_new, l_new)
                sol["one_stage_IP_mu"][k] = {
                    "t": float(t1_new),
                    "approximation_ratio": float(approx_ratio_from_t(t1_new)),
                    "assignment": {str(i): int(a_new[i]) for i in range(N)},
                    "lambda_weights": {
                        str(kk): {str(j): float(v) for j, v in l_new.get(kk, {}).items()}
                        for kk in range(K)
                    },
                    "solver_status": int(status),
                    "best_bound": float(t1_new),
                    "mip_gap": 0.0,
                }

        else:
            t1, assign, lambdas, status, bound, gap = solve_one_stage_mip(
                scenarios=scenarios,
                K=K,
                time_limit=budget,
                heuristic_assignment=assign_h,
                heuristic_lambda=lambdas_h,
                heuristic_t=t1_h,
            )
            heuristic_cache[K] = (selected_h, t1, assign, lambdas)
            sol["one_stage_IP_mu"][k] = {
                "t": float(t1),
                "approximation_ratio": float(approx_ratio_from_t(t1)),
                "assignment": {str(i): int(assign.get(i, 0)) for i in range(N)},
                "lambda_weights": {
                    str(kk): {str(j): float(v) for j, v in lambdas.get(kk, {}).items()}
                    for kk in range(K)
                },
                "solver_status": int(status),
                "best_bound": float(bound),
                "mip_gap": float(gap),
            }

        sol["objective_value"] = float(recompute_objective(sol, K_values))
        maybe_log(logger, sol, best_metric_box)

    out_dir = os.path.dirname(args.solution_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()