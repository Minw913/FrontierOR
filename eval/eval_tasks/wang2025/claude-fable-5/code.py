import argparse
import json
import time
import random
import numpy as np

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", type=str, required=True)
    ap.add_argument("--solution_path", type=str, required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", type=str, default=None)
    args = ap.parse_args()

    start_time = time.time()
    deadline = start_time + max(1.0, args.time_limit - 2.0)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    typ = inst["instance_type"]
    n = int(inst["n"])
    m = int(inst["m"])
    k = int(inst["k"])
    k = max(0, min(k, n))

    data = np.array(inst["data_matrix"], dtype=float).reshape(m, n)
    node_w = np.array(inst.get("node_weights", [0.0] * m), dtype=float)
    elem_w = np.array(inst.get("element_weights", [0.0] * n), dtype=float)

    random.seed(0)
    np.random.seed(0)

    # ---------- Set up evaluation structures ----------
    if typ == "LOC":
        M = data.copy()
        is_maxtype = True
    elif typ == "COV":
        M = data * node_w[:, None]
        is_maxtype = True
    else:  # INF
        # Q[t][j] = 1 - p_j * a[t][j]  (survival factor)
        Q = 1.0 - data * elem_w[None, :]
        np.clip(Q, 0.0, 1.0, out=Q)
        is_maxtype = False

    def eval_set(S):
        if not S:
            return 0.0
        if is_maxtype:
            return float(M[:, S].max(axis=1).sum())
        else:
            return float((1.0 - Q[:, S].prod(axis=1)).sum())

    best_S = []
    best_val = 0.0

    def report(val, S):
        nonlocal best_S, best_val
        if val > best_val + 1e-12 or (not best_S and S):
            best_val = val
            best_S = list(S)
            if logger:
                logger.log_solution(
                    best_val,
                    {"objective_value": best_val,
                     "selected_elements": [int(j) for j in best_S]},
                )

    def timeleft():
        return deadline - time.time()

    # ---------- Handle trivial cases ----------
    if n == 0 or k == 0:
        write_solution(args.solution_path, 0.0, [])
        return
    if k >= n:
        S = list(range(n))
        v = eval_set(S)
        report(v, S)
        write_solution(args.solution_path, best_val, best_S)
        return

    # ---------- Greedy ----------
    def greedy(randomized=False, rng=None):
        S = []
        if is_maxtype:
            cur = np.zeros(m)
            for _ in range(k):
                gains = np.maximum(M, cur[:, None]).sum(axis=0) - cur.sum()
                if S:
                    gains[S] = -np.inf
                if randomized:
                    top = np.argsort(gains)[-3:]
                    top = [t for t in top if gains[t] > -np.inf]
                    j = int(rng.choice(top))
                else:
                    j = int(np.argmax(gains))
                S.append(j)
                cur = np.maximum(cur, M[:, j])
            return S, float(cur.sum())
        else:
            surv = np.ones(m)
            for _ in range(k):
                gains = surv @ (1.0 - Q)
                if S:
                    gains[S] = -np.inf
                if randomized:
                    top = np.argsort(gains)[-3:]
                    top = [t for t in top if gains[t] > -np.inf]
                    j = int(rng.choice(top))
                else:
                    j = int(np.argmax(gains))
                S.append(j)
                surv = surv * Q[:, j]
            return S, float(m - surv.sum())

    S0, v0 = greedy()
    report(v0, S0)

    # ---------- Local search (swap moves) ----------
    def local_search(S, val):
        S = list(S)
        kk = len(S)
        if kk == 0:
            return S, val
        improved = True
        while improved and timeleft() > 0.5:
            improved = False
            if is_maxtype:
                sub = M[:, S]
                mx = sub.max(axis=1)
                am = sub.argmax(axis=1)
                sub2 = sub.copy()
                sub2[np.arange(m), am] = -np.inf
                sec = sub2.max(axis=1)
                sec = np.maximum(sec, 0.0)
                for c in range(kk):
                    if timeleft() <= 0.5:
                        break
                    base = np.where(am == c, sec, mx)
                    vals = np.maximum(base[:, None], M).sum(axis=0)
                    vals[S] = -np.inf
                    j = int(np.argmax(vals))
                    if vals[j] > val + 1e-9 * max(1.0, abs(val)):
                        S[c] = j
                        val = float(vals[j])
                        report(val, S)
                        improved = True
                        break
            else:
                sub = Q[:, S]
                cp = np.cumprod(sub, axis=1)
                left = np.concatenate([np.ones((m, 1)), cp[:, :-1]], axis=1)
                cpr = np.cumprod(sub[:, ::-1], axis=1)[:, ::-1]
                right = np.concatenate([cpr[:, 1:], np.ones((m, 1))], axis=1)
                for c in range(kk):
                    if timeleft() <= 0.5:
                        break
                    surv_base = left[:, c] * right[:, c]
                    vals = m - surv_base @ Q
                    vals[S] = -np.inf
                    j = int(np.argmax(vals))
                    if vals[j] > val + 1e-9 * max(1.0, abs(val)):
                        S[c] = j
                        val = float(vals[j])
                        report(val, S)
                        improved = True
                        break
        return S, val

    S1, v1 = local_search(S0, v0)
    report(v1, S1)

    proved_optimal = False

    # ---------- Exact MIP for LOC / COV ----------
    if is_maxtype and timeleft() > 3.0:
        try:
            nnz = int((M > 1e-12).sum())
            if nnz <= 3_000_000:
                import gurobipy as gp
                from gurobipy import GRB

                model = gp.Model("maxfl")
                model.Params.OutputFlag = 0
                model.Params.Seed = 0
                model.Params.MIPGap = 1e-4
                model.Params.NumericFocus = 0
                model.Params.Threads = 1
                model.Params.TimeLimit = max(1.0, timeleft() - 1.0)

                x = model.addVars(n, vtype=GRB.BINARY, name="x")
                obj = gp.LinExpr()
                for i in range(m):
                    row = M[i]
                    idx = np.nonzero(row > 1e-12)[0]
                    if len(idx) == 0:
                        continue
                    ys = model.addVars(len(idx), lb=0.0, ub=1.0)
                    model.addConstr(gp.quicksum(ys[t] for t in range(len(idx))) <= 1)
                    for t, j in enumerate(idx):
                        model.addConstr(ys[t] <= x[int(j)])
                        obj += float(row[j]) * ys[t]
                model.addConstr(gp.quicksum(x[j] for j in range(n)) <= k)
                model.setObjective(obj, GRB.MAXIMIZE)

                # warm start
                sset = set(best_S)
                for j in range(n):
                    x[j].Start = 1.0 if j in sset else 0.0

                model._x = x
                model._n = n

                def cb(mdl, where):
                    if where == GRB.Callback.MIPSOL:
                        objv = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                        xv = mdl.cbGetSolution([mdl._x[j] for j in range(mdl._n)])
                        sel = [j for j in range(mdl._n) if xv[j] > 0.5]
                        if logger and objv > best_val + 1e-9:
                            logger.log_solution(
                                float(objv),
                                {"objective_value": float(objv),
                                 "selected_elements": [int(j) for j in sel]},
                            )

                model.optimize(cb)

                if model.SolCount > 0:
                    sel = [j for j in range(n) if x[j].X > 0.5]
                    if len(sel) > k:
                        sel = sel[:k]
                    v = eval_set(sel)
                    report(v, sel)
                if model.Status == GRB.OPTIMAL:
                    proved_optimal = True
        except Exception:
            pass

    # ---------- Randomized restarts with remaining time ----------
    if not proved_optimal:
        rng = np.random.RandomState(1)
        while timeleft() > 1.0:
            Sr, vr = greedy(randomized=True, rng=rng)
            report(vr, Sr)
            Sr, vr = local_search(Sr, vr)
            report(vr, Sr)

    # ---------- Fill to k elements (monotone => cannot hurt) ----------
    if len(best_S) < k:
        chosen = set(best_S)
        S = list(best_S)
        for j in range(n):
            if len(S) >= k:
                break
            if j not in chosen:
                S.append(j)
        v = eval_set(S)
        if v >= best_val - 1e-9:
            report(max(v, best_val), S)

    write_solution(args.solution_path, best_val, best_S)


def write_solution(path, val, S):
    with open(path, "w") as f:
        json.dump(
            {"objective_value": float(val),
             "selected_elements": [int(j) for j in S]},
            f,
        )


if __name__ == "__main__":
    main()