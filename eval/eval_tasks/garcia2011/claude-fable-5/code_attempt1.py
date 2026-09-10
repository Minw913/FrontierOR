import argparse
import json
import time
import numpy as np

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def nearest_two(sub):
    """Given n x p (float64) distance submatrix, return (assign_pos, d1, d2)."""
    n, p = sub.shape
    if p == 1:
        return (np.zeros(n, dtype=np.int64),
                sub[:, 0].copy(),
                np.full(n, np.inf))
    part = np.argpartition(sub, 1, axis=1)[:, :2]
    r = np.arange(n)
    v0 = sub[r, part[:, 0]]
    v1 = sub[r, part[:, 1]]
    swap = v0 > v1
    a = np.where(swap, part[:, 1], part[:, 0])
    d1 = np.where(swap, v1, v0)
    d2 = np.where(swap, v0, v1)
    return a.astype(np.int64), d1, d2


def solution_objective(D, M):
    Marr = np.asarray(M, dtype=np.int64)
    sub = D[:, Marr].astype(np.float64)
    return float(sub.min(axis=1).sum())


def build_solution(D, M):
    M = sorted(int(x) for x in M)
    Marr = np.array(M, dtype=np.int64)
    sub = D[:, Marr].astype(np.float64)
    n = D.shape[0]
    k = np.argmin(sub, axis=1)
    costs = sub[np.arange(n), k]
    facilities = Marr[k]
    assignments = []
    for i in range(n):
        assignments.append({
            "customer": int(i),
            "facility": int(facilities[i]),
            "cost": int(round(costs[i]))
        })
    return {
        "objective_value": int(round(costs.sum())),
        "opened_facilities": [int(x) for x in M],
        "assignments": assignments
    }


def greedy_construct(D, p, deadline, rng):
    n = D.shape[0]
    tot = D.sum(axis=0, dtype=np.float64)
    first = int(np.argmin(tot))
    chosen = np.zeros(n, dtype=bool)
    M = [first]
    chosen[first] = True
    d1 = D[:, first].astype(np.float64)

    full = (float(p) * n * n) <= 2e9
    col_chunk = max(1, int(4e6 // max(n, 1)))

    while len(M) < p:
        if time.time() > deadline:
            rest = np.flatnonzero(~chosen)
            need = p - len(M)
            extra = rng.choice(rest, size=need, replace=False)
            for j in extra:
                j = int(j)
                M.append(j)
                chosen[j] = True
                d1 = np.minimum(d1, D[:, j].astype(np.float64))
            break
        cand = np.flatnonzero(~chosen)
        if not full:
            c = max(64, int(2e9 / (float(p) * n + 1)))
            if len(cand) > c:
                cand = rng.choice(cand, size=c, replace=False)
        bestj = -1
        bestred = -1.0
        for s in range(0, len(cand), col_chunk):
            cb_ = cand[s:s + col_chunk]
            block = D[:, cb_].astype(np.float64)
            red = np.maximum(d1[:, None] - block, 0.0).sum(axis=0)
            jl = int(np.argmax(red))
            if red[jl] > bestred:
                bestred = float(red[jl])
                bestj = int(cb_[jl])
        M.append(bestj)
        chosen[bestj] = True
        d1 = np.minimum(d1, D[:, bestj].astype(np.float64))
    return M


def local_search(D, M_init, deadline):
    """Best-improvement swap (interchange) local search, chunked over candidates."""
    n = D.shape[0]
    M = np.array(sorted(set(int(x) for x in M_init)), dtype=np.int64)
    p = len(M)
    col_chunk = max(1, int(4e6 // max(n, 1)))

    while True:
        if time.time() > deadline:
            break
        sub = D[:, M].astype(np.float64)
        a, d1, d2 = nearest_two(sub)
        inS = np.zeros(n, dtype=bool)
        inS[M] = True
        F = np.flatnonzero(~inS)
        if len(F) == 0:
            break

        order = np.argsort(a, kind="stable")
        a_sorted = a[order]
        starts = np.flatnonzero(np.r_[True, a_sorted[1:] != a_sorted[:-1]])
        groups = a_sorted[starts]
        d1sum = d1.sum()

        best_delta = -0.5
        best_k = -1
        best_node = -1
        timed_out = False

        for s in range(0, len(F), col_chunk):
            if time.time() > deadline:
                timed_out = True
                break
            Fb = F[s:s + col_chunk]
            DF = D[:, Fb].astype(np.float64)
            base = np.minimum(DF, d1[:, None])
            total = base.sum(axis=0) - d1sum
            diff = np.minimum(DF, d2[:, None]) - base
            diff_sorted = diff[order]
            Sseg = np.add.reduceat(diff_sorted, starts, axis=0)
            Sfull = np.zeros((p, len(Fb)), dtype=np.float64)
            Sfull[groups] = Sseg
            delta = total[None, :] + Sfull
            k, f = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[k, f] < best_delta:
                best_delta = float(delta[k, f])
                best_k = int(k)
                best_node = int(Fb[f])

        if best_k >= 0:
            M[best_k] = best_node
            M.sort()
        else:
            break
        if timed_out:
            break

    obj = solution_objective(D, M)
    return [int(x) for x in M], obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 2.0

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    n = int(inst["n"])
    p = int(inst["p"])
    D = np.array(inst["cost_matrix"], dtype=np.float32)
    inst["cost_matrix"] = None
    del inst

    rng = np.random.default_rng(0)

    if p >= n:
        sol = build_solution(D, list(range(n)))
        if logger:
            logger.log_solution(sol["objective_value"], sol)
        with open(args.solution_path, "w") as fh:
            json.dump(sol, fh)
        return

    best = {"obj": float("inf"), "M": None}

    def record(M, obj):
        if obj < best["obj"] - 1e-9:
            best["obj"] = obj
            best["M"] = sorted(int(x) for x in M)
            if logger:
                try:
                    logger.log_solution(int(round(obj)), build_solution(D, best["M"]))
                except Exception:
                    try:
                        logger.log(int(round(obj)))
                    except Exception:
                        pass

    # 1) greedy construction (sampled for very large instances)
    try:
        M0 = greedy_construct(D, p, min(deadline, start + max(1, args.time_limit) * 0.4), rng)
    except MemoryError:
        M0 = list(rng.choice(n, size=p, replace=False))
    record(M0, solution_objective(D, M0))

    # 2) local search
    try:
        M1, obj1 = local_search(D, M0, deadline)
        record(M1, obj1)
    except MemoryError:
        pass

    # 3) exact phase with Gurobi if instance is small enough and time remains
    remaining = deadline - time.time()
    use_mip = (n <= 600) and (remaining > 10)
    mip_done = False

    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            Dd = D.astype(np.float64)
            m = gp.Model("pmedian")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, deadline - time.time())

            y = m.addMVar(n, vtype=GRB.BINARY, name="y")
            x = m.addMVar((n, n), lb=0.0, ub=1.0, name="x")

            m.addConstr(x.sum(axis=1) == 1)
            for j in range(n):
                m.addConstr(x[:, j] <= y[j])
            m.addConstr(y.sum() == p)
            m.setObjective((Dd * x).sum(), GRB.MINIMIZE)

            ystart = np.zeros(n)
            for j in best["M"]:
                ystart[j] = 1.0
            y.Start = ystart

            yvars = y.tolist()

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    vals = np.array(model.cbGetSolution(yvars))
                    med = sorted(int(z) for z in np.argsort(-vals)[:p])
                    record(med, solution_objective(D, med))

            m.optimize(cb)

            if m.SolCount > 0:
                vals = np.array(y.X)
                med = sorted(int(z) for z in np.argsort(-vals)[:p])
                record(med, solution_objective(D, med))
            mip_done = (m.Status == GRB.OPTIMAL)
        except Exception:
            mip_done = False

    # 4) perturbation / restart loop with remaining time
    if not mip_done:
        while time.time() < deadline - 0.5:
            M = list(best["M"])
            k = max(1, p // 5)
            inS = np.zeros(n, dtype=bool)
            inS[M] = True
            F = np.flatnonzero(~inS)
            if len(F) == 0:
                break
            out_idx = rng.choice(p, size=min(k, p), replace=False)
            in_nodes = rng.choice(F, size=min(k, len(F)), replace=False)
            for t in range(min(len(out_idx), len(in_nodes))):
                M[out_idx[t]] = int(in_nodes[t])
            M = sorted(set(M))
            while len(M) < p:
                inS2 = np.zeros(n, dtype=bool)
                inS2[M] = True
                cand = np.flatnonzero(~inS2)
                M.append(int(rng.choice(cand)))
            try:
                M2, obj2 = local_search(D, M, deadline)
                record(M2, obj2)
            except MemoryError:
                break

    sol = build_solution(D, best["M"])
    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh)


if __name__ == "__main__":
    main()