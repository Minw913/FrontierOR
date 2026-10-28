import argparse
import json
import math
import random
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    budget = float(max(int(args.time_limit), 1))

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n_jobs"])
    m = int(inst["n_machines"])
    PT = np.asarray(inst["processing_times"], dtype=np.float64).reshape(n, m)

    flags = inst.get("no_idle_flags")
    if flags is not None and len(flags) == m:
        NI = [bool(v) for v in flags]
    else:
        NI = [False] * m
        for k in (inst.get("no_idle_machines") or []):
            NI[int(k) - 1] = True

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    reserve = min(0.2 * budget, max(0.8, 4e-6 * n * m))
    deadline = t0 + budget - reserve
    rng = random.Random(0)
    INF = float("inf")

    # ---------------- exact evaluation ----------------
    def eval_machines(seq, collect=False):
        """Minimum feasible completion times for a permutation.

        For machine k with arrivals a_j (completion of position j on
        machine k-1), cumulative processing P_j and b_j = a_j - P_{j-1}:
            regular : C_j = P_j + max_{l<=j} b_l
            no-idle : C_j = P_j + max_{l}   b_l  (whole block shifted)
        """
        sub = PT[seq]
        a = np.zeros(len(seq))
        mats = [] if collect else None
        for k in range(m):
            pk = sub[:, k]
            P = np.cumsum(pk)
            b = a + pk - P
            if NI[k]:
                a = P + b.max()
            else:
                a = P + np.maximum.accumulate(b)
            if collect:
                mats.append(a)
        return mats if collect else a[-1]

    def makespan_of(seq):
        return float(eval_machines(seq))

    def solution_dict(seq):
        mats = eval_machines(seq, collect=True)
        obj = float(mats[-1][-1]) if n > 0 else 0.0
        return {
            "objective_value": obj,
            "sequence": [int(j) for j in seq],
            "completion_times": {
                "machine_%d" % k: [float(v) for v in mats[k]] for k in range(m)
            },
        }

    state = {"best_val": INF, "best_seq": None}
    BIG = (n * m) > 200000
    last_full_log = [0.0]

    def record(seq, val):
        best = state["best_val"]
        tol = 1e-7 * max(1.0, abs(best))
        if val < best - tol:
            state["best_val"] = float(val)
            state["best_seq"] = list(seq)
            if logger is not None:
                try:
                    now = time.perf_counter()
                    if BIG and now - last_full_log[0] < 2.0:
                        logger.log(float(val))
                    else:
                        logger.log_solution(float(val), solution_dict(seq))
                        last_full_log[0] = now
                except Exception:
                    pass
            return True
        return False

    # ---------------- trivial instance ----------------
    if n == 0 or m == 0:
        sol = {"objective_value": 0.0, "sequence": [],
               "completion_times": {"machine_%d" % k: [] for k in range(m)}}
        if logger is not None:
            try:
                logger.log_solution(0.0, sol)
            except Exception:
                pass
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # ---------------- fast static orders ----------------
    totals = PT.sum(axis=1)
    order_sum = sorted(range(n), key=lambda j: (-totals[j], j))
    record(order_sum, makespan_of(order_sum))

    wts = np.arange(m, dtype=np.float64) - 0.5 * (m - 1)
    order_slope = sorted(range(n), key=lambda j: (-PT[j].dot(wts), j))
    record(order_slope, makespan_of(order_slope))

    # ---------------- heuristic components ----------------
    POS_CAP = 400

    def insert_positions(L):
        if L + 1 <= POS_CAP:
            return range(L + 1)
        step = (L + 1) / float(POS_CAP)
        pts = {min(L, int(i * step)) for i in range(POS_CAP)}
        pts.add(L)
        return sorted(pts)

    def neh(ord_, dl):
        seq = []
        for idx, j in enumerate(ord_):
            best_pos = len(seq)
            best_val = INF
            for pos in insert_positions(len(seq)):
                cand = seq[:pos] + [j] + seq[pos:]
                v = eval_machines(cand)
                if v < best_val:
                    best_val = v
                    best_pos = pos
            seq = seq[:best_pos] + [j] + seq[best_pos:]
            if (idx & 3) == 0 and time.perf_counter() > dl:
                seq = seq + ord_[idx + 1:]
                break
        return seq

    def insertion_ls(seq, val, dl, max_passes):
        nn = len(seq)
        if nn < 2:
            return seq, val
        passes = 0
        improved = True
        while improved and passes < max_passes:
            improved = False
            passes += 1
            pos_order = list(range(nn))
            rng.shuffle(pos_order)
            for r in pos_order:
                if time.perf_counter() > dl:
                    return seq, val
                x = seq[r]
                base = seq[:r] + seq[r + 1:]
                best_v = val
                best_s = -1
                for s in range(nn):
                    if s == r:
                        continue
                    cand = base[:s] + [x] + base[s:]
                    v = eval_machines(cand)
                    if v < best_v - 1e-9:
                        best_v = v
                        best_s = s
                if best_s >= 0:
                    seq = base[:best_s] + [x] + base[best_s:]
                    val = best_v
                    improved = True
        return seq, val

    def iterated_greedy(seq, val, dl):
        cur = list(seq)
        cur_val = val
        best = list(seq)
        best_val = val
        total_p = float(PT.sum())
        temp = 0.4 * total_p / (10.0 * n * m)
        if temp <= 0.0:
            temp = 1.0
        d = 2 if n < 20 else 4
        if d > n:
            d = n
        since_imp = 0
        use_ls = n <= 250
        while time.perf_counter() < dl:
            idxs = rng.sample(range(n), d)
            idx_set = set(idxs)
            removed = [cur[i] for i in idxs]
            cand = [j for i, j in enumerate(cur) if i not in idx_set]
            for j in removed:
                bp = 0
                bv = INF
                for pos in insert_positions(len(cand)):
                    c2 = cand[:pos] + [j] + cand[pos:]
                    v = eval_machines(c2)
                    if v < bv:
                        bv = v
                        bp = pos
                cand = cand[:bp] + [j] + cand[bp:]
            cval = float(eval_machines(cand))
            if use_ls and cval < cur_val:
                cand, cval = insertion_ls(cand, cval, dl, max_passes=3)
            if cval < cur_val or rng.random() < math.exp(-(cval - cur_val) / temp):
                cur = cand
                cur_val = cval
            if cval < best_val - 1e-9:
                best = list(cand)
                best_val = cval
                since_imp = 0
                record(cand, cval)
            else:
                since_imp += 1
                if since_imp > 300:
                    cur = list(best)
                    cur_val = best_val
                    since_imp = 0
        return best, best_val

    # ---------------- exact MIP (small instances) ----------------
    def run_gurobi(time_left):
        import gurobipy as gp
        from gurobipy import GRB

        mdl = gp.Model("no_idle_pfsp")
        mdl.setParam("Seed", 0)
        mdl.setParam("MIPGap", 1e-4)
        mdl.setParam("NumericFocus", 0)
        mdl.setParam("Threads", 1)
        mdl.setParam("TimeLimit", max(0.1, float(time_left)))
        mdl.setParam("LogToConsole", 0)

        x = mdl.addVars(n, n, vtype=GRB.BINARY, name="x")
        C = mdl.addVars(n, m, lb=0.0, vtype=GRB.CONTINUOUS, name="C")

        for j in range(n):
            mdl.addConstr(gp.quicksum(x[j, i] for i in range(n)) == 1)
        for i in range(n):
            mdl.addConstr(gp.quicksum(x[j, i] for j in range(n)) == 1)

        ptl = PT.tolist()
        for i in range(n):
            for k in range(m):
                pte = gp.quicksum(ptl[j][k] * x[j, i] for j in range(n))
                if k > 0:
                    mdl.addConstr(C[i, k] >= C[i, k - 1] + pte)
                elif i == 0:
                    mdl.addConstr(C[i, k] >= pte)
                if i > 0:
                    if NI[k]:
                        mdl.addConstr(C[i, k] == C[i - 1, k] + pte)
                    else:
                        mdl.addConstr(C[i, k] >= C[i - 1, k] + pte)

        mdl.setObjective(C[n - 1, m - 1], GRB.MINIMIZE)

        bs = state["best_seq"]
        for i in range(n):
            x[bs[i], i].Start = 1.0
        mats = eval_machines(bs, collect=True)
        for k in range(m):
            mk = mats[k]
            for i in range(n):
                C[i, k].Start = float(mk[i])

        xflat = [x[j, i] for j in range(n) for i in range(n)]

        def _seq_from(vals):
            seq = [0] * n
            for i in range(n):
                for j in range(n):
                    if vals[j * n + i] > 0.5:
                        seq[i] = j
                        break
            return seq

        def _cb(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    vals = model.cbGetSolution(xflat)
                    seq = _seq_from(vals)
                    record(seq, float(eval_machines(seq)))
                except Exception:
                    pass

        mdl.optimize(_cb)

        if mdl.SolCount > 0:
            try:
                vals = mdl.getAttr("X", xflat)
                seq = _seq_from(vals)
                record(seq, float(eval_machines(seq)))
            except Exception:
                pass
        try:
            return mdl.Status == GRB.OPTIMAL
        except Exception:
            return False

    # ---------------- main pipeline ----------------
    GUROBI_MAX_N = 28

    if n > 3000:
        # Extremely large: bounded adjacent-swap descent on best static order.
        seq = list(state["best_seq"])
        val = state["best_val"]
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for i in range(n - 1):
                if time.perf_counter() > deadline:
                    break
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
                v = eval_machines(seq)
                if v < val - 1e-9:
                    val = float(v)
                    improved = True
                    record(seq, val)
                else:
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]
    else:
        if n > GUROBI_MAX_N:
            heur_end = deadline
        else:
            heur_end = min(deadline, t0 + min(0.15 * budget, 8.0))
        neh_dl = min(heur_end, t0 + 0.30 * budget)
        ls_dl = min(heur_end, t0 + 0.45 * budget)

        seq = neh(order_sum, neh_dl)
        val = makespan_of(seq)
        record(seq, val)

        seq, val = insertion_ls(seq, val, ls_dl, max_passes=100)
        record(seq, val)

        if time.perf_counter() < heur_end:
            seq, val = iterated_greedy(seq, val, heur_end)
            record(seq, val)

        if n <= GUROBI_MAX_N:
            remaining = deadline - time.perf_counter()
            optimal = False
            if remaining > 1.0:
                try:
                    optimal = run_gurobi(remaining)
                except Exception:
                    optimal = False
            if not optimal and time.perf_counter() < deadline - 0.05:
                try:
                    seq, val = iterated_greedy(state["best_seq"],
                                               state["best_val"], deadline)
                    record(seq, val)
                except Exception:
                    pass

    # ---------------- final output ----------------
    sol = solution_dict(state["best_seq"])
    if logger is not None:
        try:
            logger.log_solution(sol["objective_value"], sol)
        except Exception:
            pass
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()