import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB, quicksum

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def normalize_probs(probs):
    total = float(sum(probs))
    if total <= 0.0:
        n = len(probs)
        return [1.0 / n for _ in probs] if n else []
    return [p / total for p in probs]


def weighted_quantile(values, probs, q):
    order = sorted(zip(values, probs), key=lambda x: x[0])
    if not order:
        return 0.0
    if q <= 0.0:
        return order[0][0]
    if q >= 1.0:
        return order[-1][0]
    cum = 0.0
    for value, prob in order:
        cum += prob
        if cum >= q:
            return value
    return order[-1][0]


def clean(value):
    return 0.0 if -1e-9 < value < 1e-9 else value


# ---------------------------------------------------------------------------
# Heuristic first-stage solution (per-retailer newsvendor, no transshipment)
# ---------------------------------------------------------------------------
def heuristic_stock(D, P, h, pen, lb):
    N = len(h)
    s = [0.0] * N
    for i in range(N):
        denom = h[i] + pen[i]
        cr = pen[i] / denom if denom > 1e-12 else 1.0
        idx_vals = [row[i] for row in D]
        q = weighted_quantile(idx_vals, P, cr) if idx_vals else 0.0
        s[i] = max(lb[i], 0.0, q)
    return s


# ---------------------------------------------------------------------------
# Build a full solution dictionary from (s, per-scenario recourse decisions)
# recourse[k] is None (closed form, t = 0) or (f_list, t_matrix)
# ---------------------------------------------------------------------------
def assemble_solution(s_val, recourse, h, pen, cost, D, P):
    N = len(s_val)
    scenarios = []
    obj = 0.0
    for k, d in enumerate(D):
        pk = P[k]
        rd = recourse[k]
        if rd is None:
            f = [min(s_val[i], d[i]) for i in range(N)]
            t = [[0.0] * N for _ in range(N)]
        else:
            f = [clean(v) for v in rd[0]]
            t = rd[1]

        e = [0.0] * N
        r = [0.0] * N
        q = [0.0] * N
        hold_cost = 0.0
        penalty_cost = 0.0
        for i in range(N):
            fi = f[i]
            t_out = math.fsum(t[i])
            t_in = 0.0
            for j in range(N):
                t_in += t[j][i]
            ei = s_val[i] - fi - t_out
            ri = d[i] - fi - t_in
            if -1e-7 < ei < 0.0:
                ei = 0.0
            if -1e-7 < ri < 0.0:
                ri = 0.0
            qi = s_val[i] - ei  # = fi + t_out (constraint 3)
            if -1e-7 < qi < 0.0:
                qi = 0.0
            e[i] = ei
            r[i] = ri
            q[i] = qi
            hold_cost += h[i] * ei
            penalty_cost += pen[i] * ri

        trans_cost = 0.0
        for i in range(N):
            ci = cost[i]
            row = t[i]
            for j in range(N):
                tij = row[j]
                if tij:
                    trans_cost += ci[j] * tij

        obj += pk * (hold_cost + penalty_cost + trans_cost)
        scenarios.append({"e": e, "f": f, "q": q, "r": r, "t": t})

    solution = {
        "objective_value": obj,
        "s": [clean(v) for v in s_val],
        "scenarios": scenarios,
    }
    return obj, solution


# ---------------------------------------------------------------------------
# Deterministic-equivalent LP.
#
# Eliminating e and r (e_i = s_i - f_i - t_out_i >= 0,
# r_i = d_i - f_i - t_in_i >= 0), the two-stage program becomes:
#
#   min  sum_k p_k { sum_i h_i (s_i - f_ki - sum_j t_kij)
#                  + sum_i pen_i (d_ki - f_ki - sum_j t_kji)
#                  + sum_{i!=j} c_ij t_kij }
#   s.t. f_ki + sum_{j!=i} t_kij <= s_i        (stock balance / e >= 0)
#        f_ki + sum_{j!=i} t_kji <= d_ki       (demand balance / r >= 0)
#        s >= lb, f,t >= 0
#
# Constraint family 4 of the original description is linearly implied
# by the other balance equations, hence it is automatically satisfied.
# Transshipment costs enter through their means: since recourse quantities
# are reported per demand scenario, the expectation over independent cost
# realizations of c * t equals E[c] * t.
# ---------------------------------------------------------------------------
def solve_lp(N, lb, h, pen, cost, D, P, active, time_budget):
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    out_pairs = [[(i, j) for j in range(N) if j != i] for i in range(N)]
    in_pairs = [[(j, i) for j in range(N) if j != i] for i in range(N)]

    model = gp.Model("two_stage_transshipment")
    model.setParam("OutputFlag", 0)
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", max(time_budget, 1e-6))

    s = model.addVars(N, name="s")
    # No retailer ever needs more stock than the largest total-demand
    # realization; this cap keeps the LP bounded in degenerate cost cases.
    if D:
        cap = max(math.fsum(row) for row in D)
    else:
        cap = max(max(lb), 0.0)
    for i in range(N):
        s[i].LB = max(lb[i], 0.0)
        s[i].UB = max(s[i].LB, cap)

    fvars = {}
    tvars = {}
    for k in active:
        fvars[k] = model.addVars(N, name="f_%d" % k)
        tvars[k] = model.addVars(pairs, name="t_%d" % k)

    for k in active:
        fk = fvars[k]
        tk = tvars[k]
        dk = D[k]
        for i in range(N):
            model.addConstr(
                fk[i] + quicksum(tk[p] for p in out_pairs[i]) <= s[i],
                name="stock_%d_%d" % (k, i),
            )
            model.addConstr(
                fk[i] + quicksum(tk[p] for p in in_pairs[i]) <= dk[i],
                name="dem_%d_%d" % (k, i),
            )

    obj = gp.LinExpr()
    total_p = math.fsum(P[k] for k in active)
    if total_p <= 0.0:
        total_p = 1.0
    obj.addTerms([h[i] * total_p for i in range(N)],
                 [s[i] for i in range(N)])
    const = 0.0
    for k in active:
        pk = P[k]
        dk = D[k]
        fk = fvars[k]
        tk = tvars[k]
        obj.addTerms([-pk * (h[i] + pen[i]) for i in range(N)],
                     [fk[i] for i in range(N)])
        obj.addTerms([pk * (cost[i][j] - h[i] - pen[j]) for (i, j) in pairs],
                     [tk[p] for p in pairs])
        const += pk * math.fsum(pen[i] * dk[i] for i in range(N))
    obj += const
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    status = model.Status
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED):
        return None
    try:
        s_val = [s[i].X for i in range(N)]
    except (gp.GurobiError, AttributeError, TypeError):
        return None
    if any((v is None) or (not math.isfinite(v)) for v in s_val):
        return None

    recourse = [None] * len(D)
    for k in active:
        fk = fvars[k]
        tk = tvars[k]
        f_lst = [fk[i].X for i in range(N)]
        t_mat = [[0.0] * N for _ in range(N)]
        for (i, j) in pairs:
            v = tk[i, j].X
            if -1e-9 < v < 1e-9:
                v = 0.0
            t_mat[i][j] = v
        recourse[k] = (f_lst, t_mat)

    try:
        model.dispose()
    except Exception:
        pass
    return s_val, recourse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    args = parse_args()

    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as fh:
        data = json.load(fh)

    N = int(data["N"])
    lb = list(data.get("first_stage", {}).get("variable_lower_bounds", [0.0] * N))
    if len(lb) != N:
        lb = (list(lb) + [0.0] * N)[:N]

    sec = data.get("second_stage", {}) or {}
    h = list(sec.get("holding_costs", [0.0] * N))
    pen = list(sec.get("penalty_costs", [0.0] * N))
    cost = sec.get("transshipment_costs_mean", [[0.0] * N for _ in range(N)])

    dd = sec.get("demand_distribution", {}) or {}
    D = [list(map(float, row)) for row in dd.get("scenarios", []) or []]
    P_raw = list(dd.get("probabilities", []) or [])
    if len(P_raw) != len(D):
        P_raw = ([1.0 / len(D)] * len(D)) if D else []
    P = normalize_probs(P_raw)
    active = [k for k in range(len(D)) if P[k] > 0.0]

    # --- Feasible incumbent: closed-form newsvendor solution with t = 0 ----
    if D:
        s_h = heuristic_stock(D, P, h, pen, lb)
    else:
        s_h = [max(lb[i], 0.0) for i in range(N)]
    rec_h = [None] * len(D)
    best_obj, best_sol = assemble_solution(s_h, rec_h, h, pen, cost, D, P)
    if logger:
        logger.log_solution(best_obj, best_sol)

    # --- Deterministic-equivalent LP over all positive-probability scenarios
    if active:
        est_size = len(active) * (N + N * (N - 1)) + N
        remaining = float(args.time_limit) - (time.time() - start_time) - 2.0
        if remaining > 1.0 and est_size <= 8_000_000 and (
            remaining >= 20.0 or est_size <= 1_500_000
        ):
            try:
                res = solve_lp(N, lb, h, pen, cost, D, P, active, remaining)
            except Exception:
                res = None
            if res is not None:
                s_val, recourse = res
                obj, sol = assemble_solution(s_val, recourse, h, pen, cost, D, P)
                if obj < best_obj - 1e-9:
                    best_obj, best_sol = obj, sol
                    if logger:
                        logger.log_solution(best_obj, best_sol)

    # --- Write final solution ------------------------------------------------
    with open(args.solution_path, "w") as fh:
        json.dump(
            {
                "objective_value": best_sol["objective_value"],
                "s": best_sol["s"],
                "scenarios": best_sol["scenarios"],
            },
            fh,
        )


if __name__ == "__main__":
    main()