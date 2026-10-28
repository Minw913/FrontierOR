import argparse
import json
import time
import numpy as np

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def nearest_two(sub):
    """Given n x p distance submatrix, return (assign_pos, d1, d2)."""
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
    return a, d1, d2


def solution_objective(D, M):
    sub = D[:, M]
    return float(sub.min(axis=1).sum())


def build_solution(D, M):
    M = sorted(int(x) for x in M)
    Marr = np.array(M, dtype=np.int64)
    sub = D[:, Marr]
    k = np.argmin(sub, axis=1)
    n = D.shape[0]
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


def greedy_construct(D, p):
    n = D.shape[0]
    tot = D.sum(axis=0)
    first = int(np.argmin(tot))
    M = [first]
    d1 = D[:, first].copy()
    while len(M) < p:
        red = np.maximum(d1[:, None] - D, 0.0).sum(axis=0)
        red[M] = -1.0
        j = int(np.argmax(red))
        M.append(j)
        d1 = np.minimum(d1, D[:, j])
    return M


def local_search(D, M_init, deadline):
    """Best-improvement swap (interchange) local search, vectorized."""
    n = D.shape[0]
    M = np.array(sorted(M_init), dtype=np.int64)
    p = len(M)
    while True:
        sub = D[:, M]
        a, d1, d2 = nearest_two(sub)
        if time.time() > deadline:
            break
        inS = np.zeros(n, dtype=bool)
        inS[M] = True
        F = np.flatnonzero(~inS)
        if len(F) == 0:
            break
        DF = D[:, F]
        base = np.minimum(DF, d1[:, None])
        total = base.sum(axis=0) - d1.sum()
        diff = np.minimum(DF, d2[:, None]) - base
        # group rows of diff by assigned median via one-hot matmul (BLAS)
        onehot = np.zeros((p, n), dtype=D.dtype)
        onehot[a, np.arange(n)] = 1.0
        S = onehot @ diff  # p x |F|
        delta = total[None, :] + S
        k, f = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[k, f] < -1e-9:
            M[k] = F[f]
            M.sort()
        else:
            break
    obj = solution_objective(D, M)
    return list(int(x) for x in M), obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 1.5

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    n = int(inst["n"])
    p = int(inst["p"])
    D = np.array(inst["cost_matrix"], dtype=np.float64)

    # trivial case
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
                logger.log_solution(int(round(obj)), build_solution(D, best["M"]))

    # 1) greedy construction
    M0 = greedy_construct(D, p)
    record(M0, solution_objective(D, M0))

    # 2) local search
    M1, obj1 = local_search(D, M0, deadline)
    record(M1, obj1)

    # 3) exact phase with Gurobi if instance is small enough and time remains
    remaining = deadline - time.time()
    use_mip = (n <= 800) and (remaining > 10)
    mip_done = False

    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB

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
            m.setObjective((D * x).sum(), GRB.MINIMIZE)

            # warm start
            ystart = np.zeros(n)
            for j in best["M"]:
                ystart[j] = 1.0
            y.Start = ystart

            yvars = y.tolist()

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    vals = model.cbGetSolution(yvars)
                    vals = np.array(vals)
                    med = np.argsort(-vals)[:p]
                    med = sorted(int(x) for x in med)
                    obj = solution_objective(D, np.array(med, dtype=np.int64))
                    record(med, obj)

            m.optimize(cb)

            if m.SolCount > 0:
                vals = np.array(y.X)
                med = np.argsort(-vals)[:p]
                med = sorted(int(x) for x in med)
                obj = solution_objective(D, np.array(med, dtype=np.int64))
                record(med, obj)
            mip_done = True
        except Exception:
            mip_done = False

    # 4) perturbation / restart loop with remaining time (if MIP skipped or failed)
    if not mip_done:
        rng = np.random.default_rng(0)
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
            for t, oi in enumerate(out_idx[:len(in_nodes)]):
                M[oi] = int(in_nodes[t])
            M = sorted(set(M))
            # repair if duplicates removed
            while len(M) < p:
                inS2 = np.zeros(n, dtype=bool)
                inS2[M] = True
                cand = np.flatnonzero(~inS2)
                M.append(int(rng.choice(cand)))
            M2, obj2 = local_search(D, M, deadline)
            record(M2, obj2)

    sol = build_solution(D, best["M"])
    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh)


if __name__ == "__main__":
    main()