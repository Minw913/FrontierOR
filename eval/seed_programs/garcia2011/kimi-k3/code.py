import argparse
import json
import random
import time

import numpy as np

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Instance / solution utilities
# ---------------------------------------------------------------------------

def load_instance(path):
    with open(path, "r") as fh:
        data = json.load(fh)
    n = int(data["n"])
    p = int(data["p"])
    cm = data.get("cost_matrix")
    if cm is not None:
        C = np.asarray(cm, dtype=np.int64)
    else:
        coords = np.asarray(data["node_coordinates"], dtype=np.float64)
        diff = coords[:, None, :] - coords[None, :, :]
        C = np.rint(np.sqrt((diff * diff).sum(axis=2))).astype(np.int64)
    return n, p, C


def build_solution(opened, C):
    """Given the set of opened facilities, assign every node to its nearest
    open median and build the output dictionary."""
    opened = sorted(int(j) for j in opened)
    S = np.asarray(opened, dtype=np.int64)
    D = C[:, S]
    idx = D.argmin(axis=1)
    costs = D[np.arange(C.shape[0]), idx]
    fac = S[idx]
    assignments = [
        {"customer": i, "facility": int(fac[i]), "cost": int(costs[i])}
        for i in range(C.shape[0])
    ]
    return {
        "objective_value": int(costs.sum()),
        "opened_facilities": opened,
        "assignments": assignments,
    }


# ---------------------------------------------------------------------------
# Heuristics: greedy construction + fast interchange local search + ILS
# ---------------------------------------------------------------------------

def greedy_construct(C, p, deadline):
    n = C.shape[0]
    BIG = np.iinfo(np.int64).max // 4
    totals = C.sum(axis=0)
    first = int(np.argmin(totals))
    S = [first]
    in_S = np.zeros(n, dtype=bool)
    in_S[first] = True
    d = C[:, first].copy()
    while len(S) < p:
        if time.time() > deadline:
            # Fast farthest-first fill if the greedy budget is exhausted.
            order = np.argsort(-d, kind="stable")
            for f in order:
                if len(S) >= p:
                    break
                f = int(f)
                if not in_S[f]:
                    S.append(f)
                    in_S[f] = True
                    d = np.minimum(d, C[:, f])
            break
        tot = np.empty(n, dtype=np.int64)
        for s0 in range(0, n, 256):
            blk = C[:, s0:s0 + 256]
            tot[s0:s0 + 256] = np.minimum(d[:, None], blk).sum(axis=0)
        tot[in_S] = BIG
        f = int(np.argmin(tot))
        S.append(f)
        in_S[f] = True
        d = np.minimum(d, C[:, f])
    return S


def nearest_two(C, S):
    """For each node: distance to nearest median, to 2nd nearest, and the id
    of the nearest median."""
    n = C.shape[0]
    S_arr = np.asarray(S, dtype=np.int64)
    D = C[:, S_arr]
    idx1 = D.argmin(axis=1)
    ar = np.arange(n)
    d1 = D[ar, idx1]
    INF = np.iinfo(np.int64).max // 4
    if len(S) >= 2:
        D2 = D.copy()
        D2[ar, idx1] = INF
        d2 = D2.min(axis=1)
    else:
        d2 = np.full(n, INF, dtype=np.int64)
    assign = S_arr[idx1]
    return d1, d2, assign


def local_search(C, p, S, deadline, improve_cb=None):
    """Best-improvement swap (interchange) local search with O(n^2) sweeps."""
    n = C.shape[0]
    S = [int(s) for s in S]
    in_S = np.zeros(n, dtype=bool)
    in_S[S] = True
    d1, d2, assign = nearest_two(C, S)
    cur = int(d1.sum())
    while time.time() < deadline:
        best_delta = -1e-9
        best_f = -1
        best_m = -1
        S_arr = np.asarray(S, dtype=np.int64)
        cand = np.nonzero(~in_S)[0]
        for pos in range(cand.size):
            if (pos & 63) == 0 and time.time() > deadline:
                return S, cur
            f = int(cand[pos])
            c = C[:, f]
            diff = d1 - c
            sav = float(diff[diff > 0].sum())
            if sav <= 0.0:
                continue
            extra = np.minimum(d2, c) - np.minimum(d1, c)
            per_m = np.bincount(assign, weights=extra.astype(np.float64),
                                minlength=n)
            dm = per_m[S_arr]
            j = int(np.argmin(dm))
            delta = float(dm[j]) - sav
            if delta < best_delta:
                best_delta = delta
                best_f = f
                best_m = int(S_arr[j])
        if best_f < 0:
            break
        in_S[best_m] = False
        in_S[best_f] = True
        S.remove(best_m)
        S.append(best_f)
        d1, d2, assign = nearest_two(C, S)
        cur = int(d1.sum())
        if improve_cb is not None:
            improve_cb(S, cur)
    return S, cur


def iterated_local_search(C, p, S_start, obj_start, deadline, record):
    n = C.shape[0]
    rng = random.Random(0)
    best_S = list(S_start)
    best_obj = obj_start
    cur_S = list(S_start)
    cur_obj = obj_start
    if p > 1:
        k_base = max(1, min(p - 1, max(2, p // 8)))
    else:
        k_base = 1
    while time.time() < deadline:
        in_cur = np.zeros(n, dtype=bool)
        in_cur[cur_S] = True
        non = np.nonzero(~in_cur)[0]
        if non.size == 0 or len(cur_S) == 0:
            break
        k = min(k_base, len(cur_S), int(non.size))
        if k <= 0:
            break
        rem = set(rng.sample(cur_S, k))
        add = rng.sample(non.tolist(), k)
        S1 = [s for s in cur_S if s not in rem] + add
        S1, obj1 = local_search(C, p, S1, deadline,
                                improve_cb=lambda S, o: record(S))
        if obj1 < best_obj:
            best_S = list(S1)
            best_obj = obj1
            record(S1, force=True)
        if obj1 <= cur_obj:
            cur_S = list(S1)
            cur_obj = obj1
        else:
            cur_S = list(best_S)
            cur_obj = best_obj
    return best_S, best_obj


# ---------------------------------------------------------------------------
# Exact MIP via Gurobi
# ---------------------------------------------------------------------------

def solve_mip(C, p, S0, deadline, record):
    import gurobipy as gp
    from gurobipy import GRB

    n = C.shape[0]
    m = gp.Model("p-median")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.OutputFlag = 0

    y = m.addMVar(n, vtype=GRB.BINARY, name="y")
    x = m.addMVar((n, n), vtype=GRB.BINARY, name="x")

    Cf = C.astype(np.float64)
    try:
        m.setObjective((Cf * x).sum(), GRB.MINIMIZE)
    except Exception:
        obj = None
        for j in range(n):
            term = Cf[:, j] @ x[:, j]
            obj = term if obj is None else obj + term
        m.setObjective(obj, GRB.MINIMIZE)

    m.addConstr(x.sum(axis=1) == 1)   # every node assigned to exactly one median
    m.addConstr(y.sum() == p)         # exactly p medians
    linked = False
    try:
        m.addConstr(x <= y)           # x[i,j] <= y[j] (numpy broadcasting)
        linked = True
    except Exception:
        linked = False
    if not linked:
        for j in range(n):
            m.addConstr(x[:, j] <= y[j])

    m.update()

    # MIP start from the heuristic incumbent.
    if S0 is not None and len(S0) == p:
        S_arr = np.asarray(S0, dtype=np.int64)
        ys = np.zeros(n)
        ys[S_arr] = 1.0
        xs = np.zeros((n, n))
        idx = C[:, S_arr].argmin(axis=1)
        xs[np.arange(n), S_arr[idx]] = 1.0
        y.Start = ys
        x.Start = xs

    tl = deadline - time.time() - 0.6
    if tl <= 0.5:
        return "no_time"
    m.Params.TimeLimit = tl

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                yv = np.asarray(model.cbGetSolution(y))
            except Exception:
                return
            opened = np.nonzero(yv > 0.5)[0]
            if opened.size != p:
                return
            record(opened.tolist(), force=True)

    m.optimize(cb)

    try:
        if m.SolCount > 0:
            yv = np.asarray(y.X)
            opened = np.nonzero(yv > 0.5)[0]
            if opened.size == p:
                record(opened.tolist(), force=True)
    except Exception:
        pass

    try:
        return "optimal" if m.Status == GRB.Status.OPTIMAL else "other"
    except Exception:
        return "other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MIP_MAX_N = 900


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n, p, C = load_instance(args.instance_path)

    t0 = time.time()
    T = max(int(args.time_limit), 1)
    deadline = t0 + T

    state = {"S": None, "obj": None, "sol": None}
    last_log = [0.0]

    def record(S, force=False):
        """Update the global incumbent if (opened set) S is better; log it."""
        if S is None:
            return None
        sol = build_solution(S, C)
        obj = sol["objective_value"]
        if state["obj"] is not None and obj >= state["obj"]:
            return state["obj"]
        state["S"] = list(sol["opened_facilities"])
        state["obj"] = obj
        state["sol"] = sol
        if logger:
            now = time.time()
            if force or (now - last_log[0] >= 0.25):
                last_log[0] = now
                logger.log_solution(obj, sol)
        return obj

    try:
        # Phase 1: greedy construction.
        g_end = min(t0 + min(max(0.10 * T, 0.5), 20.0), deadline)
        S0 = greedy_construct(C, p, g_end)
        record(S0, force=True)

        # Phase 2: interchange local search to a local optimum.
        ls_end = min(t0 + min(max(0.15 * T, 1.0), 25.0), deadline - 0.3)
        if time.time() < ls_end:
            S1, _ = local_search(C, p, state["S"], ls_end,
                                 improve_cb=lambda S, o: record(S))
            record(S1, force=True)

        # Write a provisional solution immediately (robustness).
        if state["sol"] is not None:
            with open(args.solution_path, "w") as fh:
                json.dump(state["sol"], fh)

        # Phase 3: exact MIP (if the instance is small enough).
        proven_optimal = False
        remaining = deadline - time.time() - 0.8
        if n <= MIP_MAX_N and remaining > 2.0:
            try:
                status = solve_mip(C, p, state["S"], deadline, record)
                proven_optimal = (status == "optimal")
            except Exception:
                proven_optimal = False

        # Phase 4: iterated local search with any remaining time
        # (fallback when the MIP is unavailable/inapplicable).
        if not proven_optimal:
            ils_end = deadline - 0.3
            if time.time() < ils_end and state["S"] is not None:
                iterated_local_search(C, p, state["S"], state["obj"],
                                      ils_end, record)
    except Exception:
        pass
    finally:
        sol = state["sol"]
        if sol is None:
            sol = build_solution(list(range(p)), C)
        with open(args.solution_path, "w") as fh:
            json.dump(sol, fh)


if __name__ == "__main__":
    main()