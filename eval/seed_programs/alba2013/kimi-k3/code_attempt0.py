import argparse
import json
import math
import random
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


# ======================================================================
# Problem context: holds instance data + helper functions
# ======================================================================
class Ctx:
    def __init__(self, n, s, cap_l, sizes, distP, distD, dp, dd, pv, dv):
        self.n = n
        self.s = s
        self.l = cap_l
        self.sizes = sizes
        self.distP = distP
        self.distD = distD
        self.dp = dp
        self.dd = dd
        self.pv = pv          # pv[r] = pickup vertex of request r
        self.dv = dv          # dv[r] = delivery vertex of request r

    def costP(self, pi):
        d = self.distP
        pv = self.pv
        c = d[self.dp][pv[pi[0]]]
        for a, b in zip(pi, pi[1:]):
            c += d[pv[a]][pv[b]]
        c += d[pv[pi[-1]]][self.dp]
        return c

    def costD(self, sigma):
        d = self.distD
        dv = self.dv
        c = d[self.dd][dv[sigma[0]]]
        for a, b in zip(sigma, sigma[1:]):
            c += d[dv[a]][dv[b]]
        c += d[dv[sigma[-1]]][self.dd]
        return c

    @staticmethod
    def positions(sigma):
        dpos = [0] * len(sigma)
        for idx, r in enumerate(sigma):
            dpos[r] = idx
        return dpos

    def greedy_assign(self, pi, dpos):
        """Assign items to stacks respecting LIFO + capacity.

        pi: request indices in pickup order; dpos[r]: delivery position of r.
        Returns list `assign` where assign[t] is the stack of request pi[t],
        or None if the (greedy) assignment fails.
        A new item with delivery rank `ar` may be placed on a stack only if
        the stack's top item has a strictly larger delivery rank (the top item
        is delivered later) and the stack has residual capacity.
        """
        s = self.s
        l = self.l
        sizes = self.sizes
        for mode in (0, 1):
            tops = []
            loads = []
            assign = []
            ok = True
            for r in pi:
                ar = dpos[r]
                sr = sizes[r]
                chosen = -1
                best_key = None
                for k in range(len(tops)):
                    if tops[k] > ar and loads[k] + sr <= l:
                        key = tops[k] if mode == 0 else -tops[k]
                        if best_key is None or key < best_key:
                            best_key = key
                            chosen = k
                if chosen < 0:
                    if len(tops) < s:
                        tops.append(ar)
                        loads.append(sr)
                        assign.append(len(tops) - 1)
                    else:
                        ok = False
                        break
                else:
                    tops[chosen] = ar
                    loads[chosen] += sr
                    assign.append(chosen)
            if ok:
                return assign
        return None

    def capacity_only_assign(self, pi):
        """First-fit assignment; valid when delivery order is exactly the
        reverse of the pickup order (any capacity-feasible stacking is then
        LIFO-feasible)."""
        loads = [0] * self.s
        asg = [0] * len(pi)
        for t, r in enumerate(pi):
            sr = self.sizes[r]
            placed = False
            for k in range(self.s):
                if loads[k] + sr <= self.l:
                    loads[k] += sr
                    asg[t] = k
                    placed = True
                    break
            if not placed:
                return None
        return asg


# ======================================================================
# Incumbent tracking + logging
# ======================================================================
class Incumbent:
    def __init__(self, ctx, logger):
        self.ctx = ctx
        self.logger = logger
        self.best_obj = math.inf
        self.best_pi = None
        self.best_sigma = None
        self.best_assign = None

    def make_solution(self, obj, pi, sigma):
        ctx = self.ctx
        return {
            "objective_value": float(obj),
            "pickup_tour": [ctx.dp] + [ctx.pv[r] for r in pi],
            "delivery_tour": [ctx.dd] + [ctx.dv[r] for r in sigma],
        }

    def update(self, pi, sigma, obj, assign=None):
        if obj < self.best_obj - 1e-6:
            self.best_obj = float(obj)
            self.best_pi = list(pi)
            self.best_sigma = list(sigma)
            self.best_assign = None if assign is None else list(assign)
            if self.logger:
                self.logger.log_solution(float(obj),
                                         self.make_solution(obj, pi, sigma))
            return True
        return False


# ======================================================================
# Standalone TSP routines (for initial solutions / lower bounds)
# ======================================================================
def nearest_neighbor(dist, depot, nodes):
    remaining = [v for v in nodes if v != depot]
    order = []
    cur = depot
    while remaining:
        nxt = min(remaining, key=lambda j: dist[cur][j])
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order


def two_opt(order, dist, depot, deadline):
    m = len(order)
    if m < 3:
        return order
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(m - 1):
            a_prev = depot if i == 0 else order[i - 1]
            oi = order[i]
            for j in range(i + 1, m):
                b_next = depot if j == m - 1 else order[j + 1]
                oj = order[j]
                delta = (dist[a_prev][oj] + dist[oi][b_next]
                         - dist[a_prev][oi] - dist[oj][b_next])
                if delta < -1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
                    break
            if improved or time.time() >= deadline:
                break
    return order


def tsp_gurobi(dist, depot, nodes, time_cap):
    m = gp.Model("tsp")
    p = m.Params
    p.Seed = 0
    p.MIPGap = 1e-4
    p.NumericFocus = 0
    p.Threads = 1
    p.TimeLimit = max(1.0, time_cap)
    p.OutputFlag = 0
    p.LogToConsole = 0
    N = len(nodes)
    cust = [v for v in nodes if v != depot]
    arcs = [(i, j) for i in nodes for j in nodes if i != j]
    x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
    u = m.addVars(cust, lb=1.0, ub=float(N - 1), name="u")
    for i in nodes:
        m.addConstr(gp.quicksum(x[i, j] for j in nodes if j != i) == 1.0)
        m.addConstr(gp.quicksum(x[j, i] for j in nodes if j != i) == 1.0)
    M = float(N - 1)
    for i in cust:
        for j in cust:
            if i != j:
                m.addConstr(u[i] - u[j] + M * x[i, j] <= M - 1.0)
    m.setObjective(gp.quicksum(dist[i][j] * x[i, j] for (i, j) in arcs),
                   GRB.MINIMIZE)
    status = None
    try:
        m.optimize()
        status = m.Status
        if m.SolCount == 0:
            return None, None, False
        xs = m.getAttr("X", x)
        succ = {}
        for (i, j), val in xs.items():
            if val > 0.5:
                succ[i] = j
        order = []
        cur = depot
        while True:
            cur = succ.get(cur, depot)
            if cur == depot:
                break
            order.append(cur)
        if len(order) != len(cust):
            return None, None, False
        return order, m.ObjVal, (status == GRB.OPTIMAL)
    finally:
        try:
            m.dispose()
        except Exception:
            pass


def solve_tsp(dist, depot, nodes, deadline, time_limit):
    ncust = len(nodes) - 1
    remaining = deadline - time.time()
    if ncust <= 26 and remaining > 3.0:
        cap = min(15.0, max(2.0, 0.08 * time_limit))
        try:
            order, obj, opt = tsp_gurobi(dist, depot, nodes,
                                         min(cap, remaining - 1.0))
            if order is not None:
                return order, obj, opt
        except Exception:
            pass
    order = nearest_neighbor(dist, depot, nodes)
    soft = min(deadline, time.time() + max(1.0, 0.07 * time_limit))
    order = two_opt(order, dist, depot, soft)
    c = dist[depot][order[0]]
    for a_, b_ in zip(order, order[1:]):
        c += dist[a_][b_]
    c += dist[order[-1]][depot]
    return order, c, False


# ======================================================================
# Local search on the coupled problem
# ======================================================================
def local_search(pi, sigma, inc, ctx, deadline):
    n = ctx.n
    pi = list(pi)
    sigma = list(sigma)
    dpos = ctx.positions(sigma)
    asg = ctx.greedy_assign(pi, dpos)
    if asg is None:
        return pi, sigma
    cp = ctx.costP(pi)
    cd = ctx.costD(sigma)
    inc.update(pi, sigma, cp + cd, asg)
    distP = ctx.distP
    distD = ctx.distD
    pv = ctx.pv
    dv = ctx.dv
    dp = ctx.dp
    dd = ctx.dd

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # ---------- pickup 2-opt ----------
        for i in range(n - 1):
            a_prev = dp if i == 0 else pv[pi[i - 1]]
            oi = pv[pi[i]]
            for j in range(i + 1, n):
                b_next = dp if j == n - 1 else pv[pi[j + 1]]
                oj = pv[pi[j]]
                delta = (distP[a_prev][oj] + distP[oi][b_next]
                         - distP[a_prev][oi] - distP[oj][b_next])
                if delta < -1e-9:
                    cand = pi[:i] + pi[i:j + 1][::-1] + pi[j + 1:]
                    ncost = ctx.costP(cand)
                    if ncost < cp - 1e-9:
                        na = ctx.greedy_assign(cand, dpos)
                        if na is not None:
                            pi = cand
                            cp = ncost
                            asg = na
                            improved = True
                            inc.update(pi, sigma, cp + cd, asg)
                            break
            if improved or time.time() >= deadline:
                break

        # ---------- pickup relocate ----------
        if not improved and time.time() < deadline:
            for i in range(n):
                xr = pi[i]
                xv = pv[xr]
                rest = pi[:i] + pi[i + 1:]
                a_v = dp if i == 0 else pv[pi[i - 1]]
                b_v = dp if i == n - 1 else pv[pi[i + 1]]
                base = distP[a_v][xv] + distP[xv][b_v] - distP[a_v][b_v]
                for k in range(n):
                    c_v = dp if k == 0 else pv[rest[k - 1]]
                    d_v = dp if k == n - 1 else pv[rest[k]]
                    delta = (distP[c_v][xv] + distP[xv][d_v]
                             - distP[c_v][d_v] - base)
                    if delta < -1e-9:
                        cand = rest[:k] + [xr] + rest[k:]
                        ncost = ctx.costP(cand)
                        if ncost < cp - 1e-9:
                            na = ctx.greedy_assign(cand, dpos)
                            if na is not None:
                                pi = cand
                                cp = ncost
                                asg = na
                                improved = True
                                inc.update(pi, sigma, cp + cd, asg)
                                break
                if improved or time.time() >= deadline:
                    break

        # ---------- delivery 2-opt ----------
        if not improved and time.time() < deadline:
            for i in range(n - 1):
                a_prev = dd if i == 0 else dv[sigma[i - 1]]
                oi = dv[sigma[i]]
                for j in range(i + 1, n):
                    b_next = dd if j == n - 1 else dv[sigma[j + 1]]
                    oj = dv[sigma[j]]
                    delta = (distD[a_prev][oj] + distD[oi][b_next]
                             - distD[a_prev][oi] - distD[oj][b_next])
                    if delta < -1e-9:
                        cand = sigma[:i] + sigma[i:j + 1][::-1] + sigma[j + 1:]
                        ncost = ctx.costD(cand)
                        if ncost < cd - 1e-9:
                            dpos_c = ctx.positions(cand)
                            na = ctx.greedy_assign(pi, dpos_c)
                            if na is not None:
                                sigma = cand
                                cd = ncost
                                dpos = dpos_c
                                asg = na
                                improved = True
                                inc.update(pi, sigma, cp + cd, asg)
                                break
                if improved or time.time() >= deadline:
                    break

        # ---------- delivery relocate ----------
        if not improved and time.time() < deadline:
            for i in range(n):
                xr = sigma[i]
                xv = dv[xr]
                rest = sigma[:i] + sigma[i + 1:]
                a_v = dd if i == 0 else dv[sigma[i - 1]]
                b_v = dd if i == n - 1 else dv[sigma[i + 1]]
                base = distD[a_v][xv] + distD[xv][b_v] - distD[a_v][b_v]
                for k in range(n):
                    c_v = dd if k == 0 else dv[rest[k - 1]]
                    d_v = dd if k == n - 1 else dv[rest[k]]
                    delta = (distD[c_v][xv] + distD[xv][d_v]
                             - distD[c_v][d_v] - base)
                    if delta < -1e-9:
                        cand = rest[:k] + [xr] + rest[k:]
                        ncost = ctx.costD(cand)
                        if ncost < cd - 1e-9:
                            dpos_c = ctx.positions(cand)
                            na = ctx.greedy_assign(pi, dpos_c)
                            if na is not None:
                                sigma = cand
                                cd = ncost
                                dpos = dpos_c
                                asg = na
                                improved = True
                                inc.update(pi, sigma, cp + cd, asg)
                                break
                if improved or time.time() >= deadline:
                    break

    return pi, sigma


def perturb(pi, sigma, ctx, rng, strength):
    n = ctx.n
    pi = list(pi)
    sigma = list(sigma)
    dpos = ctx.positions(sigma)
    for _ in range(strength):
        i = rng.randrange(n)
        k = rng.randrange(n)
        if i == k:
            continue
        cand = pi[:i] + pi[i + 1:]
        cand.insert(k, pi[i])
        na = ctx.greedy_assign(cand, dpos)
        if na is not None:
            pi = cand
    for _ in range(strength):
        i = rng.randrange(n)
        k = rng.randrange(n)
        if i == k:
            continue
        cand = sigma[:i] + sigma[i + 1:]
        cand.insert(k, sigma[i])
        dpos_c = ctx.positions(cand)
        na = ctx.greedy_assign(pi, dpos_c)
        if na is not None:
            sigma = cand
            dpos = dpos_c
    return pi, sigma


def ils(inc, ctx, deadline, rng):
    local_search(inc.best_pi, inc.best_sigma, inc, ctx, deadline)
    while time.time() < deadline:
        bp, bs = inc.best_pi, inc.best_sigma
        p2, s2 = perturb(bp, bs, ctx, rng, strength=2 + rng.randrange(4))
        local_search(p2, s2, inc, ctx, deadline)


# ======================================================================
# Exact MIP for the coupled problem (assignment-based formulation)
# ======================================================================
def solve_full_mip(ctx, inc, deadline):
    n = ctx.n
    s = ctx.s
    cap_l = ctx.l
    sizes = ctx.sizes
    distP = ctx.distP
    distD = ctx.distD
    dp = ctx.dp
    dd = ctx.dd
    pv = ctx.pv
    dv = ctx.dv

    m = gp.Model("double_tsp_multi_stack")
    prm = m.Params
    prm.Seed = 0
    prm.MIPGap = 1e-4
    prm.NumericFocus = 0
    prm.Threads = 1
    prm.TimeLimit = max(1.0, deadline - time.time())
    prm.OutputFlag = 0
    prm.LogToConsole = 0

    PV = [dp] + list(pv)
    DV = [dd] + list(dv)
    arcsP = [(i, j) for i in PV for j in PV if i != j]
    arcsD = [(i, j) for i in DV for j in DV if i != j]

    x = m.addVars(arcsP, vtype=GRB.BINARY, name="x")
    y = m.addVars(arcsD, vtype=GRB.BINARY, name="y")
    u = m.addVars(n, lb=1.0, ub=float(n), name="u")   # pickup positions
    v = m.addVars(n, lb=1.0, ub=float(n), name="v")   # delivery positions
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    a = m.addVars(pairs, vtype=GRB.BINARY, name="a")  # a[i,j]=1 iff i picked before j
    b = m.addVars(pairs, vtype=GRB.BINARY, name="b")  # b[i,j]=1 iff i delivered before j
    z = m.addVars([(r, k) for r in range(n) for k in range(s)],
                  vtype=GRB.BINARY, name="z")

    xkeys = list(x.keys())
    xvars = [x[k] for k in xkeys]
    ykeys = list(y.keys())
    yvars = [y[k] for k in ykeys]

    for i in PV:
        m.addConstr(gp.quicksum(x[i, j] for j in PV if j != i) == 1.0)
        m.addConstr(gp.quicksum(x[j, i] for j in PV if j != i) == 1.0)
    for i in DV:
        m.addConstr(gp.quicksum(y[i, j] for j in DV if j != i) == 1.0)
        m.addConstr(gp.quicksum(y[j, i] for j in DV if j != i) == 1.0)

    M = float(n)
    for r in range(n):
        pvr = pv[r]
        dvr = dv[r]
        for t in range(n):
            if t != r:
                m.addConstr(u[r] - u[t] + M * x[pvr, pv[t]] <= M - 1.0)
                m.addConstr(v[r] - v[t] + M * y[dvr, dv[t]] <= M - 1.0)

    for (i, j) in pairs:
        m.addConstr(u[j] - u[i] >= 1.0 - M * (1.0 - a[i, j]))
        m.addConstr(u[i] - u[j] >= 1.0 - M * a[i, j])
        m.addConstr(v[j] - v[i] >= 1.0 - M * (1.0 - b[i, j]))
        m.addConstr(v[i] - v[j] >= 1.0 - M * b[i, j])
        for k in range(s):
            # If i and j share stack k, their pickup/delivery orders must
            # be opposite (LIFO): a[i,j] + b[i,j] == 1.
            m.addConstr(a[i, j] + b[i, j] >= z[i, k] + z[j, k] - 1.0)
            m.addConstr(a[i, j] + b[i, j] <= 3.0 - z[i, k] - z[j, k])

    for r in range(n):
        m.addConstr(gp.quicksum(z[r, k] for k in range(s)) == 1.0)
    for k in range(s):
        m.addConstr(gp.quicksum(float(sizes[r]) * z[r, k] for r in range(n))
                    <= float(cap_l))

    m.setObjective(
        gp.quicksum(distP[i][j] * x[i, j] for (i, j) in arcsP)
        + gp.quicksum(distD[i][j] * y[i, j] for (i, j) in arcsD),
        GRB.MINIMIZE,
    )

    # ---- warm start from incumbent ----
    if inc.best_pi is not None and inc.best_assign is not None:
        pi = inc.best_pi
        sigma = inc.best_sigma
        asg = inc.best_assign
        for key in xkeys:
            x[key].Start = 0.0
        for key in ykeys:
            y[key].Start = 0.0
        tourP = [dp] + [pv[r] for r in pi] + [dp]
        for t in range(len(tourP) - 1):
            x[tourP[t], tourP[t + 1]].Start = 1.0
        tourD = [dd] + [dv[r] for r in sigma] + [dd]
        for t in range(len(tourD) - 1):
            y[tourD[t], tourD[t + 1]].Start = 1.0
        upos = [0] * n
        vpos = [0] * n
        for idx, r in enumerate(pi):
            upos[r] = idx + 1
        for idx, r in enumerate(sigma):
            vpos[r] = idx + 1
        for r in range(n):
            u[r].Start = float(upos[r])
            v[r].Start = float(vpos[r])
        for (i, j) in pairs:
            a[i, j].Start = 1.0 if upos[i] < upos[j] else 0.0
            b[i, j].Start = 1.0 if vpos[i] < vpos[j] else 0.0
        for r in range(n):
            for k in range(s):
                z[r, k].Start = 0.0
        for idx, r in enumerate(pi):
            z[r, asg[idx]].Start = 1.0

    v2pr = {pv[r]: r for r in range(n)}
    v2dr = {dv[r]: r for r in range(n)}

    def extract(vals, keys, depot, v2r):
        succ = {}
        for (i, j), val in zip(keys, vals):
            if val > 0.5:
                succ[i] = j
        order = []
        cur = depot
        for _ in range(n):
            cur = succ.get(cur, depot)
            if cur == depot or cur not in v2r:
                break
            order.append(v2r[cur])
        return order

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                pi = extract(model.cbGetSolution(xvars), xkeys, dp, v2pr)
                sigma = extract(model.cbGetSolution(yvars), ykeys, dd, v2dr)
                if len(pi) == n and len(sigma) == n:
                    inc.update(pi, sigma, float(objv), None)
            except Exception:
                pass

    status = None
    try:
        m.optimize(cb)
        status = m.Status
        if m.SolCount > 0:
            try:
                pi = extract(m.getAttr("X", xvars), xkeys, dp, v2pr)
                sigma = extract(m.getAttr("X", yvars), ykeys, dd, v2dr)
                if len(pi) == n and len(sigma) == n:
                    inc.update(pi, sigma, float(m.ObjVal), None)
            except Exception:
                pass
    finally:
        try:
            m.dispose()
        except Exception:
            pass
    return status


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (SolutionLogger(args.log_path, sense="minimize")
              if args.log_path else None)

    t0 = time.time()
    time_limit = max(1, int(args.time_limit))
    deadline = t0 + time_limit

    with open(args.instance_path) as f:
        inst = json.load(f)

    params = inst.get("parameters", {})
    s = int(params["num_stacks_s"])
    cap_l = int(params["stack_height_l"])
    n = int(params["num_requests_n"])
    P = inst["pickup_region"]
    D = inst["delivery_region"]
    dp = int(P["depot_index"])
    dd = int(D["depot_index"])
    distP = [[float(vv) for vv in row] for row in P["distance_matrix"]]
    distD = [[float(vv) for vv in row] for row in D["distance_matrix"]]
    requests = inst["requests"]
    pv = [int(r["pickup_vertex"]) for r in requests]
    dv = [int(r["delivery_vertex"]) for r in requests]
    sizes = [max(1, int(r.get("item_size", 1))) for r in requests]

    ctx = Ctx(n, s, cap_l, sizes, distP, distD, dp, dd, pv, dv)
    inc = Incumbent(ctx, logger)
    rng = random.Random(0)

    if n == 0:
        with open(args.solution_path, "w") as f:
            json.dump({"objective_value": 0.0,
                       "pickup_tour": [dp],
                       "delivery_tour": [dd]}, f)
        return

    v2pr = {pv[r]: r for r in range(n)}
    v2dr = {dv[r]: r for r in range(n)}
    nodesP = list(range(len(distP)))
    nodesD = list(range(len(distD)))

    # ---------------- Stage 1: independent TSPs + feasible construction ---
    tsp_budget_each = max(2.0, 0.12 * time_limit)

    orderP, objP, optP = solve_tsp(
        distP, dp, nodesP,
        min(deadline, time.time() + tsp_budget_each), time_limit)
    pi_t = [v2pr[v] for v in orderP]

    # Always-feasible construction: deliver in exact reverse pickup order.
    sigma_rev = list(reversed(pi_t))
    asg0 = ctx.capacity_only_assign(pi_t)
    inc.update(pi_t, sigma_rev, ctx.costP(pi_t) + ctx.costD(sigma_rev), asg0)

    orderD, objD, optD = solve_tsp(
        distD, dd, nodesD,
        min(deadline, time.time() + tsp_budget_each), time_limit)
    sigma_t = [v2dr[v] for v in orderD]

    dpos_t = ctx.positions(sigma_t)
    asg_t = ctx.greedy_assign(pi_t, dpos_t)
    if asg_t is not None:
        objt = ctx.costP(pi_t) + ctx.costD(sigma_t)
        inc.update(pi_t, sigma_t, objt, asg_t)
        if optP and optD and objt <= objP + objD + 1e-6:
            # Sum of independent TSP optima achieved -> globally optimal.
            sol = inc.make_solution(objt, inc.best_pi, inc.best_sigma)
            with open(args.solution_path, "w") as f:
                json.dump(sol, f)
            return

    # ---------------- Stage 2: improvement ------------------------------
    remaining = deadline - time.time()
    use_mip = ((n <= 28) or (n <= 36 and time_limit >= 120)) and remaining > 4.0

    if use_mip:
        # Brief local-search polish to strengthen the warm start.
        ls_end = min(deadline - 2.0, time.time() + min(10.0, 0.10 * remaining))
        if ls_end > time.time():
            local_search(inc.best_pi, inc.best_sigma, inc, ctx, ls_end)
        status = None
        try:
            status = solve_full_mip(ctx, inc, deadline - 0.5)
        except Exception:
            status = None
        if status != GRB.OPTIMAL and time.time() < deadline - 1.0:
            ils(inc, ctx, deadline - 0.5, rng)
    else:
        ils(inc, ctx, deadline - 0.5, rng)

    # ---------------- Final output ---------------------------------------
    cp = ctx.costP(inc.best_pi)
    cd = ctx.costD(inc.best_sigma)
    sol = inc.make_solution(cp + cd, inc.best_pi, inc.best_sigma)
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()