import argparse
import json
import time
import random

import numpy as np

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def read_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    n = int(data["n"])
    m = int(data["m"])
    p = np.array(data["profits"], dtype=np.int64)
    P = np.array(data["pairwise_profits"], dtype=np.int64)
    w = np.array(data["weights"], dtype=np.int64)
    cap = np.array(data["capacities"], dtype=np.int64)
    # ensure symmetry & zero diagonal
    P = (P + P.T) // 2 if not np.array_equal(P, P.T) else P
    np.fill_diagonal(P, 0)
    return n, m, p, P, w, cap


class State:
    def __init__(self, n, m, p, P, w, cap):
        self.n, self.m = n, m
        self.p, self.P, self.w, self.cap = p, P, w, cap
        self.assign = np.full(n, -1, dtype=np.int64)
        self.load = np.zeros(m, dtype=np.int64)
        self.contrib = np.zeros((n, m), dtype=np.int64)
        self.obj = 0

    def add(self, i, k):
        d = int(self.p[i] + self.contrib[i, k])
        self.contrib[:, k] += self.P[i]
        self.load[k] += self.w[i]
        self.assign[i] = k
        self.obj += d

    def remove(self, i):
        k = self.assign[i]
        self.contrib[:, k] -= self.P[i]
        d = int(self.p[i] + self.contrib[i, k])
        self.load[k] -= self.w[i]
        self.assign[i] = -1
        self.obj -= d

    def snapshot(self):
        return (self.assign.copy(), self.load.copy(), self.contrib.copy(), self.obj)

    def restore(self, snap):
        self.assign = snap[0].copy()
        self.load = snap[1].copy()
        self.contrib = snap[2].copy()
        self.obj = snap[3]


def compute_objective(assign, p, P, m):
    tot = 0
    for k in range(m):
        idx = np.flatnonzero(assign == k)
        if idx.size:
            tot += int(p[idx].sum())
            tot += int(P[np.ix_(idx, idx)].sum()) // 2
    return tot


def local_search(st, deadline):
    EPS = 0.5
    n, m = st.n, st.m
    while time.time() < deadline:
        improved = False

        # ---- 1) insertions of unassigned items ----
        while time.time() < deadline:
            un = np.flatnonzero(st.assign < 0)
            if un.size == 0:
                break
            gains = st.p[un, None] + st.contrib[un]
            feas = (st.load[None, :] + st.w[un, None]) <= st.cap[None, :]
            g = np.where(feas, gains.astype(np.float64), -np.inf)
            flat = int(np.argmax(g))
            r, k = divmod(flat, m)
            if g[r, k] > EPS:
                st.add(int(un[r]), int(k))
                improved = True
            else:
                break

        # ---- 2) removals of items with negative contribution ----
        while time.time() < deadline:
            asg = np.flatnonzero(st.assign >= 0)
            if asg.size == 0:
                break
            vals = st.p[asg] + st.contrib[asg, st.assign[asg]]
            r = int(np.argmin(vals))
            if vals[r] < -EPS:
                st.remove(int(asg[r]))
                improved = True
            else:
                break

        # ---- 3) relocations between knapsacks ----
        if m >= 2:
            while time.time() < deadline:
                asg = np.flatnonzero(st.assign >= 0)
                if asg.size == 0:
                    break
                ka = st.assign[asg]
                cur = st.contrib[asg, ka]
                delta = (st.contrib[asg] - cur[:, None]).astype(np.float64)
                feas = (st.load[None, :] + st.w[asg, None]) <= st.cap[None, :]
                delta = np.where(feas, delta, -np.inf)
                delta[np.arange(asg.size), ka] = -np.inf
                flat = int(np.argmax(delta))
                r, k = divmod(flat, m)
                if delta[r, k] > EPS:
                    i = int(asg[r])
                    st.remove(i)
                    st.add(i, int(k))
                    improved = True
                else:
                    break

        # ---- 4) swaps between assigned items in different knapsacks ----
        if m >= 2:
            while time.time() < deadline:
                asg = np.flatnonzero(st.assign >= 0)
                if asg.size < 2:
                    break
                if asg.size > 1500:
                    sel = np.random.choice(asg, 1500, replace=False)
                else:
                    sel = asg
                ka = st.assign[sel]
                M = st.contrib[sel][:, ka].astype(np.float64)
                diag = np.diag(M).copy()
                D = M + M.T - diag[:, None] - diag[None, :] \
                    - 2.0 * st.P[np.ix_(sel, sel)]
                resid = st.cap - st.load
                dw = st.w[sel][:, None] - st.w[sel][None, :]
                feas = (dw <= resid[ka][None, :]) & (-dw <= resid[ka][:, None]) \
                       & (ka[:, None] != ka[None, :])
                D = np.where(feas, D, -np.inf)
                flat = int(np.argmax(D))
                a, b = divmod(flat, sel.size)
                if D[a, b] > EPS:
                    i, j = int(sel[a]), int(sel[b])
                    ki, kj = int(ka[a]), int(ka[b])
                    st.remove(i)
                    st.remove(j)
                    st.add(i, kj)
                    st.add(j, ki)
                    improved = True
                else:
                    break

        # ---- 5) swap assigned item with an unassigned item ----
        while time.time() < deadline:
            asg = np.flatnonzero(st.assign >= 0)
            un = np.flatnonzero(st.assign < 0)
            if asg.size == 0 or un.size == 0:
                break
            ka = st.assign[asg]
            cur = (st.p[asg] + st.contrib[asg, ka]).astype(np.float64)
            gain_j = (st.p[un][:, None] + st.contrib[un][:, ka]).astype(np.float64)
            D = gain_j - cur[None, :] - st.P[np.ix_(un, asg)]
            feas = (st.load[ka][None, :] - st.w[asg][None, :] + st.w[un][:, None]) \
                   <= st.cap[ka][None, :]
            D = np.where(feas, D, -np.inf)
            flat = int(np.argmax(D))
            a, b = divmod(flat, asg.size)
            if D[a, b] > EPS:
                j, i = int(un[a]), int(asg[b])
                k = int(ka[b])
                st.remove(i)
                st.add(j, k)
                improved = True
            else:
                break

        if not improved:
            break


def perturb(st, rng, strength):
    asg = np.flatnonzero(st.assign >= 0)
    if asg.size:
        k = max(1, int(strength * asg.size))
        sel = rng.choice(asg, size=min(k, asg.size), replace=False)
        for i in sel:
            st.remove(int(i))
    # random insertions of a few unassigned items
    un = np.flatnonzero(st.assign < 0)
    un = un.copy()
    rng.shuffle(un)
    cnt = 0
    limit = max(2, int(0.05 * st.n))
    for i in un:
        ks = np.flatnonzero(st.load + st.w[i] <= st.cap)
        if ks.size:
            st.add(int(i), int(rng.choice(ks)))
            cnt += 1
        if cnt >= limit:
            break


def make_solution(assign, obj):
    return {
        "objective_value": float(obj),
        "assignment": [[int(i), int(assign[i])]
                       for i in range(len(assign)) if assign[i] >= 0],
    }


def solve_gurobi(n, m, p, P, w, cap, warm_assign, warm_obj, deadline, logger):
    import gurobipy as gp
    from gurobipy import GRB

    remaining = deadline - time.time()
    if remaining < 3:
        return warm_assign, warm_obj

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("qmkp", env=env)
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1.0, remaining - 1.0)

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1)
    for k in range(m):
        model.addConstr(gp.quicksum(int(w[i]) * x[i, k] for i in range(n)) <= int(cap[k]))

    obj = gp.QuadExpr()
    for i in range(n):
        if p[i] != 0:
            for k in range(m):
                obj.add(x[i, k], float(p[i]))
    for i in range(n):
        Pi = P[i]
        for j in range(i + 1, n):
            if Pi[j] != 0:
                for k in range(m):
                    obj.add(x[i, k] * x[j, k], float(Pi[j]))
    model.setObjective(obj, GRB.MAXIMIZE)

    # warm start
    for i in range(n):
        for k in range(m):
            x[i, k].Start = 1.0 if warm_assign[i] == k else 0.0

    xlist = [x[i, k] for i in range(n) for k in range(m)]
    tracker = {"obj": warm_obj, "assign": warm_assign.copy()}

    def cb(mdl, where):
        if where == GRB.Callback.MIPSOL:
            o = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if o > tracker["obj"] + 1e-6:
                vals = mdl.cbGetSolution(xlist)
                a = np.full(n, -1, dtype=np.int64)
                for i in range(n):
                    for k in range(m):
                        if vals[i * m + k] > 0.5:
                            a[i] = k
                            break
                real = compute_objective(a, p, P, m)
                if real > tracker["obj"]:
                    tracker["obj"] = real
                    tracker["assign"] = a
                    if logger:
                        logger.log_solution(float(real), make_solution(a, real))

    try:
        model.optimize(cb)
    except Exception:
        pass

    if model.SolCount > 0:
        vals = model.getAttr("X", xlist)
        a = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            for k in range(m):
                if vals[i * m + k] > 0.5:
                    a[i] = k
                    break
        real = compute_objective(a, p, P, m)
        if real > tracker["obj"]:
            tracker["obj"] = real
            tracker["assign"] = a
            if logger:
                logger.log_solution(float(real), make_solution(a, real))
    return tracker["assign"], tracker["obj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", type=str, required=True)
    ap.add_argument("--solution_path", type=str, required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", type=str, default=None)
    args = ap.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 0.8

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")

    n, m, p, P, w, cap = read_instance(args.instance_path)
    random.seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)

    use_gurobi = (n <= 65) and (n * n * m <= 120000)
    if use_gurobi:
        heur_deadline = min(deadline, start + max(1.0, min(8.0, 0.15 * args.time_limit)))
    else:
        heur_deadline = deadline

    st = State(n, m, p, P, w, cap)
    local_search(st, heur_deadline)
    best_snap = st.snapshot()
    best_obj = st.obj
    if logger:
        logger.log_solution(float(best_obj), make_solution(best_snap[0], best_obj))

    # Iterated local search
    strength = 0.10
    while time.time() < heur_deadline:
        st.restore(best_snap)
        perturb(st, rng, strength)
        local_search(st, heur_deadline)
        if st.obj > best_obj:
            best_obj = st.obj
            best_snap = st.snapshot()
            strength = 0.10
            if logger:
                logger.log_solution(float(best_obj),
                                    make_solution(best_snap[0], best_obj))
        else:
            strength += 0.05
            if strength > 0.45:
                strength = 0.10

    best_assign = best_snap[0]

    if use_gurobi and time.time() < deadline - 3:
        try:
            a, o = solve_gurobi(n, m, p, P, w, cap, best_assign, best_obj,
                                deadline, logger)
            if o > best_obj:
                best_obj = o
                best_assign = a
        except Exception:
            pass

    # If time remains after Gurobi (or not used), continue ILS
    st.restore((best_assign, None, None, None)) if False else None
    if time.time() < deadline - 1:
        st = State(n, m, p, P, w, cap)
        for i in range(n):
            if best_assign[i] >= 0:
                st.add(i, int(best_assign[i]))
        local_search(st, deadline)
        if st.obj > best_obj:
            best_obj = st.obj
            best_assign = st.assign.copy()
            if logger:
                logger.log_solution(float(best_obj),
                                    make_solution(best_assign, best_obj))
        snap = st.snapshot()
        strength = 0.10
        while time.time() < deadline:
            st.restore(snap)
            perturb(st, rng, strength)
            local_search(st, deadline)
            if st.obj > best_obj:
                best_obj = st.obj
                best_assign = st.assign.copy()
                snap = st.snapshot()
                strength = 0.10
                if logger:
                    logger.log_solution(float(best_obj),
                                        make_solution(best_assign, best_obj))
            else:
                strength += 0.05
                if strength > 0.45:
                    strength = 0.10

    # Final verification and output
    final_obj = compute_objective(best_assign, p, P, m)
    sol = make_solution(best_assign, final_obj)
    if logger:
        logger.log_solution(float(final_obj), sol)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()