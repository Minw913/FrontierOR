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
            r, c = linear_sum_assignment(cost)
            return np.asarray(c, dtype=int)

        return lap
    except Exception:
        pass

    def hungarian(cost):
        n = cost.shape[0]
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
    deadline = start_time + max(1.0, args.time_limit - 0.7)

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
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

    # Trivial initial solution
    consider(np.arange(n))

    if n == 1:
        sol = {"objective_value": float(best["obj"]),
               "assignment": [int(v) for v in best["perm"]]}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    # ---------------- Local search: best-improvement 2-swap ----------------
    def local_search(perm, dl):
        perm = np.asarray(perm, dtype=int).copy()
        while time.time() < dl:
            MA = A[:, perm]
            MB = B[:, perm]
            MC = C[:, perm]
            da = np.diagonal(MA).copy()
            db = np.diagonal(MB).copy()
            dc = np.diagonal(MC).copy()
            a = da.sum()
            b = db.sum()
            c = dc.sum()
            cur = a * b + c
            DA = MA + MA.T - da[:, None] - da[None, :]
            DB = MB + MB.T - db[:, None] - db[None, :]
            DC = MC + MC.T - dc[:, None] - dc[None, :]
            NEW = (a + DA) * (b + DB) + (c + DC)
            k = int(NEW.argmin())
            i, j = divmod(k, n)
            if i != j and NEW.flat[k] < cur - 1e-7:
                perm[i], perm[j] = perm[j], perm[i]
            else:
                break
        return perm

    # ---------------- Iterative linearization ----------------
    def lin_iter(wa, wb, dl):
        prev = None
        for _ in range(25):
            if time.time() > dl:
                break
            cost = wa * A + wb * B + C
            perm = lap(cost)
            consider(perm)
            a = A[idx, perm].sum()
            b = B[idx, perm].sum()
            key = (a, b)
            if key == prev:
                break
            prev = key
            wa, wb = b, a

    # decide on Gurobi usage
    use_gurobi = n <= 300
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
    a_scale = float(np.abs(A).mean()) * n
    b_scale = float(np.abs(B).mean()) * n
    starts = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0),
              (b_scale, a_scale), (b_scale, 0.0), (0.0, a_scale)]
    for wa, wb in starts:
        if time.time() > heur_dl:
            break
        lin_iter(wa, wb, heur_dl)

    # Local search on best
    if best["perm"] is not None and time.time() < heur_dl:
        p = local_search(best["perm"], heur_dl)
        consider(p)

    # Perturbation loop (kick + local search)
    rng = random.Random(0)
    kick = max(2, n // 15)
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
        # occasionally re-linearize around current point
        if rng.random() < 0.25 and time.time() < heur_dl:
            a = A[idx, cur].sum()
            b = B[idx, cur].sum()
            lin_iter(b, a, heur_dl)

    # ---------------- Exact phase with Gurobi (bilinear MIQP) ----------------
    remaining = deadline - time.time()
    if use_gurobi and remaining > 3.0 and best["perm"] is not None:
        try:
            GRB = gp.GRB
            # bounds on aggregate A-cost and B-cost via LAPs
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

            # warm start
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
    while time.time() < deadline - 0.3:
        pert = best["perm"].copy()
        for _ in range(kick):
            i = rng.randrange(n)
            j = rng.randrange(n)
            pert[i], pert[j] = pert[j], pert[i]
        pert = local_search(pert, deadline - 0.2)
        consider(pert)

    sol = {"objective_value": float(best["obj"]),
           "assignment": [int(v) for v in best["perm"]]}
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()