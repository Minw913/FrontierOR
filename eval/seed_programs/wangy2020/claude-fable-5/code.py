import argparse
import json
import time
import random
import numpy as np


def build_lap_solver():
    """Return a function lap(cost) -> permutation array (row i -> column perm[i])."""
    try:
        from scipy.optimize import linear_sum_assignment

        def lap(cost):
            cost = np.ascontiguousarray(cost, dtype=np.float64)
            r, c = linear_sum_assignment(cost)
            return np.asarray(c, dtype=int)

        return lap
    except Exception:
        pass

    def greedy_lap(cost):
        # Greedy fallback: assign each row its cheapest remaining column
        n = cost.shape[0]
        perm = np.full(n, -1, dtype=int)
        used = np.zeros(n, dtype=bool)
        # process rows in order of their min-cost spread
        order = np.argsort(cost.min(axis=1))
        for i in order:
            row = cost[i].copy()
            row[used] = np.inf
            j = int(row.argmin())
            perm[i] = j
            used[j] = True
        return perm

    def hungarian(cost):
        n = cost.shape[0]
        if n > 400:
            return greedy_lap(cost)
        INF = float("inf")
        u = [0.0] * (n + 1)
        v = [0.0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)
        cost_l = cost.tolist()
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [INF] * (n + 1)
            used = [False] * (n + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = INF
                j1 = 0
                row = cost_l[i0 - 1]
                ui = u[i0]
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = row[j - 1] - ui - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while j0:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
        ans = [0] * n
        for j in range(1, n + 1):
            ans[p[j] - 1] = j - 1
        return np.array(ans, dtype=int)

    return hungarian


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1.0, args.time_limit - 1.0)

    logger = None
    try:
        from solution_logger import SolutionLogger
        if args.log_path:
            logger = SolutionLogger(args.log_path, sense="minimize")
    except Exception:
        logger = None

    random.seed(0)
    np.random.seed(0)

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n = int(data["n"])
    A = np.array(data["A"], dtype=np.float64)
    B = np.array(data["B"], dtype=np.float64)
    C = np.array(data["C"], dtype=np.float64)
    idx = np.arange(n)

    lap = build_lap_solver()

    def evaluate(perm):
        a = A[idx, perm].sum()
        b = B[idx, perm].sum()
        c = C[idx, perm].sum()
        return a * b + c

    best = {"obj": float("inf"), "perm": None}

    def write_solution():
        perm = best["perm"] if best["perm"] is not None else np.arange(n)
        sol = {"objective_value": float(evaluate(perm)),
               "assignment": [int(v) for v in perm]}
        try:
            with open(args.solution_path, "w") as f:
                json.dump(sol, f)
        except Exception:
            pass

    def consider(perm):
        perm = np.asarray(perm, dtype=int)
        o = evaluate(perm)
        if o < best["obj"] - 1e-9:
            best["obj"] = o
            best["perm"] = perm.copy()
            if logger:
                try:
                    logger.log_solution(
                        float(o),
                        {"objective_value": float(o),
                         "assignment": [int(v) for v in perm]},
                    )
                except Exception:
                    pass
        return o

    # Trivial initial solution + safety write
    consider(np.arange(n))
    write_solution()

    if n == 1:
        write_solution()
        return

    # ------------- Memory-lean local search (row-wise 2-swap) -------------
    def local_search(perm, dl):
        perm = np.asarray(perm, dtype=int).copy()
        try:
            diagA = A[idx, perm].copy()
            diagB = B[idx, perm].copy()
            diagC = C[idx, perm].copy()
            a = diagA.sum()
            b = diagB.sum()
            c = diagC.sum()
            cur = a * b + c
            improved = True
            check = 0
            while improved:
                improved = False
                for i in range(n):
                    check += 1
                    if (check & 31) == 0 and time.time() > dl:
                        return perm
                    pi = perm[i]
                    dA = A[i, perm] + A[:, pi] - (A[i, pi] + diagA)
                    dB = B[i, perm] + B[:, pi] - (B[i, pi] + diagB)
                    dC = C[i, perm] + C[:, pi] - (C[i, pi] + diagC)
                    newv = (a + dA) * (b + dB) + (c + dC)
                    j = int(newv.argmin())
                    if j != i and newv[j] < cur - 1e-7:
                        a += dA[j]
                        b += dB[j]
                        c += dC[j]
                        perm[i], perm[j] = perm[j], perm[i]
                        diagA[i] = A[i, perm[i]]
                        diagA[j] = A[j, perm[j]]
                        diagB[i] = B[i, perm[i]]
                        diagB[j] = B[j, perm[j]]
                        diagC[i] = C[i, perm[i]]
                        diagC[j] = C[j, perm[j]]
                        cur = a * b + c
                        improved = True
        except Exception:
            pass
        return perm

    # ------------- Iterative linearization -------------
    cost_buf = None
    tmp_buf = None
    try:
        cost_buf = np.empty_like(A)
        tmp_buf = np.empty_like(A)
    except Exception:
        cost_buf = None
        tmp_buf = None

    def lin_iter(wa, wb, dl):
        prev = None
        for _ in range(20):
            if time.time() > dl:
                break
            try:
                if cost_buf is not None:
                    np.multiply(A, wa, out=cost_buf)
                    np.multiply(B, wb, out=tmp_buf)
                    cost_buf += tmp_buf
                    cost_buf += C
                    perm = lap(cost_buf)
                else:
                    perm = lap(wa * A + wb * B + C)
            except Exception:
                break
            consider(perm)
            a = A[idx, perm].sum()
            b = B[idx, perm].sum()
            key = (a, b)
            if key == prev:
                break
            prev = key
            wa, wb = b, a

    # Decide on Gurobi usage
    use_gurobi = n <= 200
    gp = None
    if use_gurobi:
        try:
            import gurobipy as _gp
            gp = _gp
        except Exception:
            use_gurobi = False

    if use_gurobi:
        heur_dl = min(deadline, start_time + max(3.0, 0.30 * args.time_limit))
    else:
        heur_dl = deadline

    # Multi-start linearization
    try:
        a_scale = float(np.abs(A).mean()) * n
        b_scale = float(np.abs(B).mean()) * n
        starts = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0),
                  (b_scale, a_scale), (b_scale, 0.0), (0.0, a_scale)]
        for wa, wb in starts:
            if time.time() > heur_dl:
                break
            lin_iter(wa, wb, heur_dl)
    except Exception:
        pass
    write_solution()

    # Local search on best
    try:
        if best["perm"] is not None and time.time() < heur_dl:
            p = local_search(best["perm"], heur_dl)
            consider(p)
    except Exception:
        pass
    write_solution()

    # Perturbation loop (kick + local search)
    try:
        rng = random.Random(0)
        kick = max(2, min(n // 15, 40))
        cur = best["perm"].copy()
        cur_obj = evaluate(cur)
        while time.time() < heur_dl:
            pert = cur.copy()
            for _ in range(kick):
                i = rng.randrange(n)
                j = rng.randrange(n)
                pert[i], pert[j] = pert[j], pert[i]
            pert = local_search(pert, heur_dl)
            o = consider(pert)
            if o <= cur_obj:
                cur = pert
                cur_obj = o
            else:
                cur = best["perm"].copy()
                cur_obj = best["obj"]
            if rng.random() < 0.2 and time.time() < heur_dl:
                a = A[idx, cur].sum()
                b = B[idx, cur].sum()
                lin_iter(b, a, heur_dl)
    except Exception:
        pass
    write_solution()

    # ------------- Exact phase with Gurobi (bilinear MIQP) -------------
    remaining = deadline - time.time()
    if use_gurobi and remaining > 3.0 and best["perm"] is not None:
        try:
            GRB = gp.GRB
            pa = lap(A)
            s_lb = float(A[idx, pa].sum())
            pa = lap(-A)
            s_ub = float(A[idx, pa].sum())
            pb = lap(B)
            t_lb = float(B[idx, pb].sum())
            pb = lap(-B)
            t_ub = float(B[idx, pb].sum())

            m = gp.Model("bilinear_assignment")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, deadline - time.time() - 0.5)
            try:
                m.Params.NonConvex = 2
            except Exception:
                pass

            xf = m.addMVar(n * n, vtype=GRB.BINARY, name="x")
            for i in range(n):
                m.addConstr(xf[i * n:(i + 1) * n].sum() == 1)
            for j in range(n):
                m.addConstr(xf[j::n].sum() == 1)

            s = m.addVar(lb=s_lb, ub=s_ub, name="s")
            t = m.addVar(lb=t_lb, ub=t_ub, name="t")
            Af = A.flatten()
            Bf = B.flatten()
            Cf = C.flatten()
            m.addConstr(s == Af @ xf)
            m.addConstr(t == Bf @ xf)
            m.setObjective(s * t + Cf @ xf, GRB.MINIMIZE)

            wp = best["perm"]
            start_vec = np.zeros(n * n)
            for i in range(n):
                start_vec[i * n + wp[i]] = 1.0
            xf.Start = start_vec
            s.Start = float(A[idx, wp].sum())
            t.Start = float(B[idx, wp].sum())

            try:
                x_list = xf.tolist()
            except Exception:
                x_list = [xf[k] for k in range(n * n)]

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        vals = model.cbGetSolution(x_list)
                        arr = np.array(vals).reshape(n, n)
                        perm = arr.argmax(axis=1)
                        if len(set(perm.tolist())) == n:
                            consider(perm)
                    except Exception:
                        pass

            m.optimize(cb)

            if m.SolCount > 0:
                try:
                    vals = np.array(xf.X).reshape(n, n)
                    perm = vals.argmax(axis=1)
                    if len(set(perm.tolist())) == n:
                        consider(perm)
                except Exception:
                    pass
        except Exception:
            pass

    # Use any leftover time on further heuristics
    try:
        rng2 = random.Random(1)
        kick2 = max(2, min(n // 15, 40))
        while time.time() < deadline - 0.3:
            pert = best["perm"].copy()
            for _ in range(kick2):
                i = rng2.randrange(n)
                j = rng2.randrange(n)
                pert[i], pert[j] = pert[j], pert[i]
            pert = local_search(pert, deadline - 0.2)
            consider(pert)
    except Exception:
        pass

    write_solution()


if __name__ == "__main__":
    main()