import argparse
import json
import time
import numpy as np

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def solution_objective(D, M):
    Marr = np.asarray(list(M), dtype=np.int64)
    return float(D[:, Marr].min(axis=1).sum(dtype=np.float64))


def build_solution(D, M):
    M = sorted(int(x) for x in M)
    Marr = np.array(M, dtype=np.int64)
    sub = D[:, Marr]
    n = D.shape[0]
    k = np.argmin(sub, axis=1)
    costs = sub[np.arange(n), k].astype(np.float64)
    facilities = Marr[k]
    assignments = []
    for i in range(n):
        assignments.append({
            "customer": int(i),
            "facility": int(facilities[i]),
            "cost": int(round(float(costs[i])))
        })
    return {
        "objective_value": int(round(float(costs.sum()))),
        "opened_facilities": [int(x) for x in M],
        "assignments": assignments
    }


def nearest_two(sub):
    """Given n x p float32 distance submatrix, return (assign_pos, d1, d2)."""
    n, p = sub.shape
    if p == 1:
        return (np.zeros(n, dtype=np.int64),
                sub[:, 0].copy(),
                np.full(n, np.inf, dtype=np.float32))
    part = np.argpartition(sub, 1, axis=1)[:, :2]
    r = np.arange(n)
    v0 = sub[r, part[:, 0]]
    v1 = sub[r, part[:, 1]]
    swap = v0 > v1
    a = np.where(swap, part[:, 1], part[:, 0]).astype(np.int64)
    d1 = np.where(swap, v1, v0)
    d2 = np.where(swap, v0, v1)
    return a, d1, d2


def greedy_construct(D, p, deadline, rng):
    n = D.shape[0]
    tot = D.sum(axis=0, dtype=np.float64)
    first = int(np.argmin(tot))
    chosen = np.zeros(n, dtype=bool)
    M = [first]
    chosen[first] = True
    d1 = D[:, first].astype(np.float32).copy()

    # cap on number of candidates evaluated per step (compute budget ~3e9 ops)
    cap = int(3e9 / (float(max(1, p)) * n + 1.0))
    cap = max(100, min(cap, n))
    col_chunk = max(1, int(2e6 // max(n, 1)))

    while len(M) < p:
        if time.time() > deadline:
            rest = np.flatnonzero(~chosen)
            need = p - len(M)
            if need > 0 and len(rest) > 0:
                take = min(need, len(rest))
                extra = rng.choice(rest, size=take, replace=False)
                for j in extra:
                    j = int(j)
                    M.append(j)
                    chosen[j] = True
                    np.minimum(d1, D[:, j], out=d1)
            break
        cand = np.flatnonzero(~chosen)
        if len(cand) == 0:
            break
        if len(cand) > cap:
            cand = rng.choice(cand, size=cap, replace=False)
        bestj = -1
        bestred = -1.0
        for s in range(0, len(cand), col_chunk):
            cb = cand[s:s + col_chunk]
            block = D[:, cb]
            red = np.maximum(d1[:, None] - block, 0.0).sum(axis=0, dtype=np.float64)
            jl = int(np.argmax(red))
            if float(red[jl]) > bestred:
                bestred = float(red[jl])
                bestj = int(cb[jl])
        if bestj < 0:
            bestj = int(cand[0])
        M.append(bestj)
        chosen[bestj] = True
        np.minimum(d1, D[:, bestj], out=d1)
    return M


def local_search(D, M_init, deadline):
    """Best-improvement swap (interchange) local search, chunked over candidates."""
    n = D.shape[0]
    M = np.array(sorted(set(int(x) for x in M_init)), dtype=np.int64)
    p = len(M)
    col_chunk = max(1, int(2e6 // max(n, 1)))

    while time.time() < deadline:
        sub = D[:, M]
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
        d1sum = float(d1.sum(dtype=np.float64))

        best_delta = -0.5
        best_k = -1
        best_node = -1
        timed_out = False

        for s in range(0, len(F), col_chunk):
            if time.time() > deadline:
                timed_out = True
                break
            Fb = F[s:s + col_chunk]
            DF = D[:, Fb]
            base = np.minimum(DF, d1[:, None])
            total = base.sum(axis=0, dtype=np.float64) - d1sum
            diff = np.minimum(DF, d2[:, None]) - base
            Sseg = np.add.reduceat(diff[order], starts, axis=0, dtype=np.float64)
            Sfull = np.zeros((p, len(Fb)), dtype=np.float64)
            Sfull[groups] = Sseg
            delta = total[None, :] + Sfull
            k, f = np.unravel_index(np.argmin(delta), delta.shape)
            if float(delta[k, f]) < best_delta:
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
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

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
            try:
                logger.log_solution(sol["objective_value"], sol)
            except Exception:
                pass
        with open(args.solution_path, "w") as fh:
            json.dump(sol, fh)
        return

    best = {"obj": float("inf"), "M": None}

    def record(M, obj=None):
        try:
            if obj is None:
                obj = solution_objective(D, M)
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
        except Exception:
            pass

    # 1) greedy construction (robust; falls back to random selection)
    try:
        greedy_deadline = min(deadline, start + max(1, args.time_limit) * 0.4)
        M0 = greedy_construct(D, p, greedy_deadline, rng)
        if len(set(M0)) != p:
            raise ValueError("bad greedy result")
    except Exception:
        M0 = [int(x) for x in rng.choice(n, size=p, replace=False)]
    record(M0)

    if best["M"] is None:
        best["M"] = sorted(M0)
        best["obj"] = solution_objective(D, M0)

    # 2) local search
    try:
        M1, obj1 = local_search(D, best["M"], deadline)
        record(M1, obj1)
    except Exception:
        pass

    # 3) exact phase with Gurobi if instance is small enough and time remains
    remaining = deadline - time.time()
    use_mip = (n <= 500) and (remaining > 10)
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
                    try:
                        vals = np.array(model.cbGetSolution(yvars))
                        med = sorted(int(z) for z in np.argsort(-vals)[:p])
                        record(med)
                    except Exception:
                        pass

            m.optimize(cb)

            if m.SolCount > 0:
                vals = np.array(y.X)
                med = sorted(int(z) for z in np.argsort(-vals)[:p])
                record(med)
            mip_done = (m.Status == GRB.OPTIMAL)
        except Exception:
            mip_done = False

    # 4) perturbation / restart loop with remaining time
    if not mip_done:
        while time.time() < deadline - 0.5:
            try:
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
                    M[int(out_idx[t])] = int(in_nodes[t])
                M = sorted(set(M))
                while len(M) < p:
                    inS2 = np.zeros(n, dtype=bool)
                    inS2[M] = True
                    cand2 = np.flatnonzero(~inS2)
                    if len(cand2) == 0:
                        break
                    M.append(int(rng.choice(cand2)))
                if len(M) != p:
                    break
                M2, obj2 = local_search(D, M, deadline)
                record(M2, obj2)
            except Exception:
                break

    if best["M"] is None or len(best["M"]) != p:
        best["M"] = sorted(int(x) for x in rng.choice(n, size=p, replace=False))

    sol = build_solution(D, best["M"])
    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh)


if __name__ == "__main__":
    main()