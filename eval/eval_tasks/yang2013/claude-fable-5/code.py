import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    start = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    n = int(data["n"])
    cap = int(data["capacity"])
    w = np.array(data["weights"], dtype=np.int64)
    P = np.array(data["profit_matrix"], dtype=np.int64).reshape(n, n) if n > 0 else np.zeros((0, 0), dtype=np.int64)

    end = start + max(1, args.time_limit) - 1.0  # safety buffer

    def write_output(obj, sel):
        out = {"objective_value": float(obj), "solution": [int(v) for v in sel]}
        with open(args.solution_path, "w") as f:
            json.dump(out, f)

    if n == 0:
        if logger:
            logger.log_solution(0.0, {"objective_value": 0.0, "solution": []})
        write_output(0.0, [])
        return

    diag = np.diag(P).copy()
    fit = w <= cap  # items that can fit alone

    best = {"obj": -1, "sel": np.zeros(n, dtype=bool)}

    def update_best(obj, sel):
        if obj > best["obj"]:
            best["obj"] = int(obj)
            best["sel"] = sel.copy()
            if logger:
                logger.log_solution(float(obj),
                                    {"objective_value": float(obj),
                                     "solution": [int(v) for v in sel]})
            return True
        return False

    def total_obj(sel):
        xv = sel.astype(np.int64)
        contrib = P @ xv
        return int((contrib[sel].sum() + diag[sel].sum()) // 2)

    # start with empty (always feasible)
    update_best(0, np.zeros(n, dtype=bool))

    rng = np.random.default_rng(0)

    # ---------------- construction ----------------
    def construct(rcl=1):
        sel = np.zeros(n, dtype=bool)
        contrib = np.zeros(n, dtype=np.int64)
        W = 0
        obj = 0
        while True:
            cand = fit & (~sel) & (w <= cap - W)
            idx = np.where(cand)[0]
            if len(idx) == 0:
                break
            gains = (diag[idx] + contrib[idx]).astype(np.float64)
            ratio = gains / np.maximum(w[idx], 1)
            if rcl <= 1:
                pick = idx[int(np.argmax(ratio))]
            else:
                k = min(rcl, len(idx))
                topk = np.argpartition(-ratio, k - 1)[:k]
                pick = idx[rng.choice(topk)]
            g = int(diag[pick] + contrib[pick])
            sel[pick] = True
            W += int(w[pick])
            obj += g
            contrib += P[pick]
        return sel, contrib, W, obj

    # ---------------- local search: adds + swaps ----------------
    def local_search(sel, contrib, W, obj, deadline):
        while time.time() < deadline:
            improved = False
            # greedy adds
            while True:
                rem = cap - W
                cand = (~sel) & (w <= rem)
                idx = np.where(cand)[0]
                if len(idx) == 0:
                    break
                gains = diag[idx] + contrib[idx]
                b = int(np.argmax(gains))
                if gains[b] <= 0:
                    break
                j = int(idx[b])
                sel[j] = True
                W += int(w[j])
                obj += int(gains[b])
                contrib += P[j]
                improved = True
            # best 1-1 swap
            S = np.where(sel)[0]
            U = np.where(~sel)[0]
            if len(S) > 0 and len(U) > 0:
                uval = diag[U] + contrib[U]
                wU = w[U]
                best_delta = 0
                bi = bj = -1
                cnt = 0
                for i in S:
                    cnt += 1
                    if cnt % 64 == 0 and time.time() > deadline:
                        break
                    rem_i = cap - W + int(w[i])
                    mask = wU <= rem_i
                    if not mask.any():
                        continue
                    Um = U[mask]
                    d = uval[mask] - P[i, Um] - contrib[i]
                    bb = int(np.argmax(d))
                    if d[bb] > best_delta:
                        best_delta = int(d[bb])
                        bi = int(i)
                        bj = int(Um[bb])
                if best_delta > 0:
                    sel[bi] = False
                    contrib -= P[bi]
                    W -= int(w[bi])
                    sel[bj] = True
                    contrib += P[bj]
                    W += int(w[bj])
                    obj += best_delta
                    improved = True
            if not improved:
                break
        return sel, contrib, W, obj

    # ---------------- decide whether to use Gurobi ----------------
    nnz_off = int(np.count_nonzero(np.triu(P, 1)))
    use_grb = (n <= 300) or (n <= 1000 and nnz_off <= 100000)

    if use_grb:
        heur_end = min(end, start + 0.30 * (end - start))
    else:
        heur_end = end

    # ---------------- heuristic phase ----------------
    try:
        sel, contrib, W, obj = construct(rcl=1)
        sel, contrib, W, obj = local_search(sel, contrib, W, obj, heur_end)
        update_best(obj, sel)

        it = 0
        while time.time() < heur_end:
            it += 1
            if it % 5 == 0:
                sel, contrib, W, obj = construct(rcl=3)
            else:
                sel = best["sel"].copy()
                S = np.where(sel)[0]
                if len(S) == 0:
                    sel, contrib, W, obj = construct(rcl=3)
                else:
                    k = max(1, int(len(S) * rng.uniform(0.1, 0.4)))
                    drop = rng.choice(S, size=min(k, len(S)), replace=False)
                    sel[drop] = False
                    contrib = P @ sel.astype(np.int64)
                    W = int(w[sel].sum())
                    obj = int((contrib[sel].sum() + diag[sel].sum()) // 2)
                    # a few random adds for diversification
                    U = np.where((~sel) & (w <= cap - W) & fit)[0]
                    if len(U) > 0:
                        rng.shuffle(U)
                        added = 0
                        for j in U:
                            if added >= 3:
                                break
                            if w[j] <= cap - W:
                                g = int(diag[j] + contrib[j])
                                sel[j] = True
                                W += int(w[j])
                                obj += g
                                contrib += P[j]
                                added += 1
            sel, contrib, W, obj = local_search(sel, contrib, W, obj, heur_end)
            update_best(obj, sel)
    except Exception:
        pass

    # ---------------- Gurobi exact phase ----------------
    if use_grb and time.time() < end - 3:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            try:
                import scipy.sparse as sp
                Q = sp.csr_matrix(np.triu(P, 1).astype(np.float64))
            except Exception:
                Q = np.triu(P, 1).astype(np.float64)

            m = gp.Model("qkp")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, end - time.time())

            x = m.addMVar(n, vtype=GRB.BINARY, name="x")
            m.setObjective(x @ Q @ x + diag.astype(np.float64) @ x, GRB.MAXIMIZE)
            m.addConstr(w.astype(np.float64) @ x <= float(cap))
            x.Start = best["sel"].astype(np.float64)
            xlist = x.tolist()

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if objv > best["obj"] + 1e-6:
                        vals = model.cbGetSolution(xlist)
                        selc = np.array([v > 0.5 for v in vals], dtype=bool)
                        if int(w[selc].sum()) <= cap:
                            update_best(total_obj(selc), selc)

            m.optimize(cb)

            if m.SolCount > 0:
                selg = np.array(x.X) > 0.5
                if int(w[selg].sum()) <= cap:
                    update_best(total_obj(selg), selg)
        except Exception:
            pass

    # continue heuristic with leftover time (if gurobi finished early or was skipped)
    try:
        while time.time() < end - 0.2:
            sel = best["sel"].copy()
            S = np.where(sel)[0]
            if len(S) == 0:
                sel, contrib, W, obj = construct(rcl=3)
            else:
                k = max(1, int(len(S) * rng.uniform(0.1, 0.4)))
                drop = rng.choice(S, size=min(k, len(S)), replace=False)
                sel[drop] = False
                contrib = P @ sel.astype(np.int64)
                W = int(w[sel].sum())
                obj = int((contrib[sel].sum() + diag[sel].sum()) // 2)
            sel, contrib, W, obj = local_search(sel, contrib, W, obj, end - 0.2)
            update_best(obj, sel)
    except Exception:
        pass

    write_output(best["obj"], best["sel"])


if __name__ == "__main__":
    main()