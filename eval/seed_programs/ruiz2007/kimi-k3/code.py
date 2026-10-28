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

    # ---------------- robust instance loading ----------------
    def to_int(x, default=None):
        try:
            return int(x)
        except Exception:
            return default

    nj_f = to_int(inst.get("n_jobs"))
    nm_f = to_int(inst.get("n_machines"))
    flags_raw = inst.get("no_idle_flags")
    ni_machines = inst.get("no_idle_machines") or []
    raw = inst.get("processing_times", [])

    # Normalize processing times to a nested [rows][cols] structure.
    if isinstance(raw, list) and raw and not isinstance(raw[0], (list, tuple)):
        flat = [float(v) for v in raw]
        L = len(flat)
        if nj_f and nm_f and nj_f * nm_f == L:
            rows, cols = nj_f, nm_f
        elif flags_raw is not None and len(flags_raw) > 0 and L % len(flags_raw) == 0:
            cols = len(flags_raw)
            rows = L // cols
        else:
            rows, cols = 1, L
        data = [flat[i * cols:(i + 1) * cols] for i in range(rows)]
    else:
        data = [list(r) for r in raw]
        rows = len(data)
        cols = len(data[0]) if rows else 0

    # Decide orientation.  Schema says rows = jobs, cols = machines, but some
    # instance files carry inconsistent n_jobs/n_machines fields; use the data
    # shape together with no_idle_flags / no_idle_machines to disambiguate.
    transpose = False
    if rows and cols:
        if nj_f and nm_f and nj_f * nm_f == rows * cols:
            if rows == nj_f and cols == nm_f:
                transpose = False
            elif rows == nm_f and cols == nj_f:
                transpose = True
            elif flags_raw is not None and len(flags_raw) == cols:
                transpose = False
            elif flags_raw is not None and len(flags_raw) == rows:
                transpose = True
        elif flags_raw is not None:
            if len(flags_raw) == cols and len(flags_raw) != rows:
                transpose = False
            elif len(flags_raw) == rows and len(flags_raw) != cols:
                transpose = True
            elif ni_machines:
                try:
                    mx = max(int(v) for v in ni_machines)
                    if mx > cols and mx <= rows:
                        transpose = True
                except Exception:
                    pass

    arr = np.asarray(data, dtype=np.float64)
    if transpose:
        arr = arr.T
    arr = np.ascontiguousarray(arr)
    n, m = int(arr.shape[0]), int(arr.shape[1])
    PT = arr

    NI = [False] * m
    if flags_raw is not None and len(flags_raw) == m:
        NI = [bool(v) for v in flags_raw]
    else:
        for v in ni_machines:
            idx = to_int(v, 0) - 1
            if 0 <= idx < m:
                NI[idx] = True

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    reserve = min(0.2 * budget, max(1.0, 8e-6 * max(1, n * m)))
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
        L = len(seq)
        if L == 0:
            return [np.zeros(0) for _ in range(m)] if collect else 0.0
        idx = np.asarray(seq, dtype=np.int64)
        sub = PT[idx]
        a = np.zeros(L)
        mats = [] if collect else None
        for k in range(m):
            pk = sub[:, k]
            P = np.cumsum(pk)
            b = a + pk - P
            if NI[k]:
                a = P + float(b.max())
            else:
                a = P + np.maximum.accumulate(b)
            if collect:
                mats.append(a)
        return mats if collect else float(a[-1])

    def makespan_of(seq):
        return float(eval_machines(seq))

    def solution_dict(seq):
        mats = eval_machines(seq, collect=True)
        obj = float(mats[-1][-1]) if (n > 0 and m > 0) else 0.0
        return {
            "objective_value": obj,
            "sequence": [int(j) for j in seq],
            "completion_times": {
                "machine_%d" % k: [float(v) for v in mats[k]] for k in range(m)
            },
        }

    state = {"best_val": INF, "best_seq": None}
    THROTTLE = (n * m >= 15000) or (n >= 150)
    last_full_log = [0.0]

    def record(seq, val):
        if seq is None:
            return False
        val = float(val)
        best = state["best_val"]
        if state["best_seq"] is not None and math.isfinite(best):
            tol = 1e-7 * max(1.0, abs(best))
            if not (val < best - tol):
                return False
        elif not math.isfinite(val):
            return False
        state["best_val"] = val
        state["best_seq"] = [int(j) for j in seq]
        if logger is not None:
            try:
                now = time.perf_counter()
                if THROTTLE and now - last_full_log[0] < 2.0:
                    logger.log(val)
                else:
                    logger.log_solution(val, solution_dict(state["best_seq"]))
                    last_full_log[0] = now
            except Exception:
                pass
        return True

    # ---------------- trivial instances ----------------
    if n == 0 or m == 0:
        seq0 = list(range(n))
        if n > 0 and m > 0:
            sol = solution_dict(seq0)
        else:
            sol = {
                "objective_value": 0.0,
                "sequence": [int(j) for j in seq0],
                "completion_times": {"machine_%d" % k: [0.0] * n for k in range(m)},
            }
        if logger is not None:
            try:
                logger.log_solution(sol["objective_value"], sol)
            except Exception:
                pass
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # ---------------- measured evaluation cost (for adaptation) ----------------
    def measure_t_eval():
        seq = list(range(n))
        rng.shuffle(seq)
        reps = 3
        t = time.perf_counter()
        for _ in range(reps):
            eval_machines(seq)
        return max((time.perf_counter() - t) / reps, 1e-6)

    t_eval = measure_t_eval()

    def clip(v, lo, hi):
        return max(lo, min(hi, int(v)))

    neh_cap = clip(0.28 * budget / max(1, n) / t_eval, 6, 400)
    d_ig = 2 if n < 20 else 4
    d_ig = min(d_ig, n)
    ig_cap = clip(0.08 / max(1e-9, d_ig * t_eval), 6, 400)
    if n <= 120:
        ls_cap = None
        ls_passes = 100
        ig_use_ls = True
    elif n <= 250:
        ls_cap = 80
        ls_passes = 10
        ig_use_ls = True
    elif n <= 600:
        ls_cap = 60
        ls_passes = 3
        ig_use_ls = False
    else:
        ls_cap = None
        ls_passes = 0
        ig_use_ls = False

    # ---------------- fast static orders ----------------
    totals = PT.sum(axis=1)
    order_sum = sorted(range(n), key=lambda j: (-totals[j], j))
    record(order_sum, makespan_of(order_sum))

    wts = np.arange(m, dtype=np.float64) - 0.5 * (m - 1)
    order_slope = sorted(range(n), key=lambda j: (-float(PT[j].dot(wts)), j))
    record(order_slope, makespan_of(order_slope))

    # ---------------- heuristic components ----------------
    def insert_positions(L, cap):
        if cap is None or L + 1 <= cap:
            return range(L + 1)
        step = (L + 1) / float(cap)
        pts = {min(L, int(i * step)) for i in range(cap)}
        pts.add(L)
        return sorted(pts)

    def neh(ord_, dl, cap):
        seq = []
        for idx, j in enumerate(ord_):
            best_pos = len(seq)
            best_val = INF
            for pos in insert_positions(len(seq), cap):
                cand = seq[:pos] + [j] + seq[pos:]
                v = eval_machines(cand)
                if v < best_val:
                    best_val = v
                    best_pos = pos
            seq = seq[:best_pos] + [j] + seq[best_pos:]
            if time.perf_counter() > dl:
                seq = seq + ord_[idx + 1:]
                break
        return seq

    def insertion_ls(seq, val, dl, max_passes, cap):
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
                for s in insert_positions(nn - 1, cap):
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

    def swap_descent(seq, val, dl):
        improved = True
        while improved and time.perf_counter() < dl:
            improved = False
            for i in range(len(seq) - 1):
                if time.perf_counter() > dl:
                    return seq, val
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
                v = eval_machines(seq)
                if v < val - 1e-9:
                    val = float(v)
                    improved = True
                    record(seq, val)
                else:
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]
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
        since_imp = 0
        while time.perf_counter() < dl:
            idxs = rng.sample(range(n), d_ig)
            idx_set = set(idxs)
            removed = [cur[i] for i in idxs]
            cand = [j for i, j in enumerate(cur) if i not in idx_set]
            for j in removed:
                bp = 0
                bv = INF
                for pos in insert_positions(len(cand), ig_cap):
                    c2 = cand[:pos] + [j] + cand[pos:]
                    v = eval_machines(c2)
                    if v < bv:
                        bv = v
                        bp = pos
                cand = cand[:bp] + [j] + cand[bp:]
            cval = float(eval_machines(cand))
            if ig_use_ls and cval < cur_val:
                cand, cval = insertion_ls(cand, cval, dl, max_passes=3, cap=ls_cap)
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
        if bs is not None:
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

    try:
        if n > 3000:
            seq = list(state["best_seq"])
            val = state["best_val"]
            swap_descent(seq, val, deadline)
        else:
            if n > GUROBI_MAX_N:
                heur_end = deadline
            else:
                heur_end = min(deadline, t0 + min(0.15 * budget, 8.0))
            neh_dl = min(heur_end, t0 + 0.30 * budget)
            ls_dl = min(heur_end, t0 + 0.45 * budget)

            seq = neh(order_sum, neh_dl, neh_cap)
            val = makespan_of(seq)
            record(seq, val)

            if ls_passes > 0:
                seq, val = insertion_ls(seq, val, ls_dl, max_passes=ls_passes,
                                        cap=ls_cap)
                record(seq, val)
            elif time.perf_counter() < ls_dl:
                seq, val = swap_descent(seq, val, ls_dl)
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
    except Exception:
        pass

    # ---------------- final output ----------------
    if state["best_seq"] is None:
        seq0 = list(range(n))
        record(seq0, makespan_of(seq0))

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