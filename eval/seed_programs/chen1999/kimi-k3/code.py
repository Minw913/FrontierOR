#!/usr/bin/env python3
"""
Parallel-machine scheduling solver.

Supports:
  * minimize total weighted completion time (identical / uniform / unrelated)
  * minimize weighted number of tardy jobs

Approach:
  * Fast constructive heuristics (WSPT-based greedy / weighted greedy insertion)
  * First-improvement local search (relocations, swaps, insertions)
  * Exact assignment MIP (Gurobi) with pairwise-sequencing variables, using the
    fact that for a fixed machine assignment the optimal order is WSPT
    (completion-time variant) resp. EDD (tardy-jobs variant).
  * Exact Lawler-Moore DP for the single-machine tardy-jobs case.
"""

import argparse
import json
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAVE_GUROBI = True
except Exception:  # pragma: no cover
    gp = None
    GRB = None
    HAVE_GUROBI = False

from solution_logger import SolutionLogger

EPS = 1e-9


# ---------------------------------------------------------------------------
# Instance handling
# ---------------------------------------------------------------------------

def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_data(inst):
    n = int(inst["num_jobs"])
    m = int(inst["num_machines"])
    jobs = inst["jobs"]
    w = [float(v) for v in jobs["weights"]]
    pt = jobs.get("processing_times")
    if pt is not None:
        p = [[float(v) for v in row] for row in pt]
    else:
        base = jobs.get("base_processing_times")
        speeds = (inst.get("machines") or {}).get("speeds")
        if base is not None and speeds:
            p = [[float(base[j]) / float(speeds[mm]) for mm in range(m)]
                 for j in range(n)]
        elif base is not None:
            p = [[float(base[j])] * m for j in range(n)]
        else:
            raise ValueError("instance contains no processing times")
    d = jobs.get("due_dates")
    if d is not None:
        d = [float(v) for v in d]
    return n, m, w, p, d


def compute_ratios(p, w, n, m):
    """WSPT ratio p_jm / w_j for every job/machine pair."""
    rat = []
    for j in range(n):
        wj = w[j]
        row = p[j]
        if wj > 0:
            rat.append([row[mm] / wj for mm in range(m)])
        else:
            rat.append([float("inf")] * m)
    return rat


# ---------------------------------------------------------------------------
# Weighted completion time helpers (machine lists kept in WSPT order)
# ---------------------------------------------------------------------------

def insert_wspt(jobs_sorted, j, mm, rat):
    rj = rat[j][mm]
    pos = len(jobs_sorted)
    for idx, jj in enumerate(jobs_sorted):
        r = rat[jj][mm]
        if r > rj or (r == rj and jj > j):
            pos = idx
            break
    return jobs_sorted[:pos] + [j] + jobs_sorted[pos:]


def cost_sorted(jobs_sorted, mm, p, w):
    t = 0.0
    c = 0.0
    for j in jobs_sorted:
        t += p[j][mm]
        c += w[j] * t
    return c


def build_solution_v1(mach_jobs, p, w, rat, n, m):
    schedule = {}
    completion = {}
    obj = 0.0
    for mm in range(m):
        ordered = sorted(mach_jobs[mm], key=lambda j: (rat[j][mm], j))
        schedule[str(mm)] = [int(j) for j in ordered]
        t = 0.0
        for j in ordered:
            t += p[j][mm]
            completion[str(j)] = t
            obj += w[j] * t
    sol = {
        "objective_value": float(obj),
        "schedule": schedule,
        "completion_times": completion,
        "tardy_jobs": [],
        "on_time_jobs": list(range(n)),
    }
    return obj, sol


def greedy_v1(n, m, p, w, rat):
    order = sorted(range(n), key=lambda j: (min(rat[j]), j))
    mach_jobs = [[] for _ in range(m)]
    if n > 3000:
        # very large: cheap least-completion-time insertion
        loads = [0.0] * m
        for j in order:
            best_mm = 0
            best_v = float("inf")
            for mm in range(m):
                v = loads[mm] + p[j][mm]
                if v < best_v:
                    best_v = v
                    best_mm = mm
            mach_jobs[best_mm].append(j)
            loads[best_mm] += p[j][mm]
        for mm in range(m):
            mach_jobs[mm].sort(key=lambda j: (rat[j][mm], j))
        return mach_jobs
    costs = [0.0] * m
    totw = [0.0] * m
    for j in order:
        best_mm = 0
        best_delta = float("inf")
        for mm in range(m):
            jobs_mm = mach_jobs[mm]
            rj = rat[j][mm]
            pre = 0.0
            prew = 0.0
            for jj in jobs_mm:
                r = rat[jj][mm]
                if r < rj or (r == rj and jj < j):
                    pre += p[jj][mm]
                    prew += w[jj]
                else:
                    break
            pj = p[j][mm]
            delta = w[j] * (pre + pj) + pj * (totw[mm] - prew)
            if delta < best_delta:
                best_delta = delta
                best_mm = mm
        mach_jobs[best_mm] = insert_wspt(mach_jobs[best_mm], j, best_mm, rat)
        costs[best_mm] += best_delta
        totw[best_mm] += w[j]
    return mach_jobs


def local_search_v1(mach_jobs, p, w, rat, n, m, deadline, improved_cb):
    costs = [cost_sorted(mach_jobs[mm], mm, p, w) for mm in range(m)]
    mach_of = [0] * n
    for mm in range(m):
        for j in mach_jobs[mm]:
            mach_of[j] = mm

    def relocation_pass():
        for j in range(n):
            if time.time() >= deadline:
                return None
            a = mach_of[j]
            la = mach_jobs[a]
            ia = la.index(j)
            na = la[:ia] + la[ia + 1:]
            ca = cost_sorted(na, a, p, w)
            da = ca - costs[a]
            for mm in range(m):
                if mm == a:
                    continue
                nb = insert_wspt(mach_jobs[mm], j, mm, rat)
                cb = cost_sorted(nb, mm, p, w)
                if da + cb - costs[mm] < -EPS:
                    mach_jobs[a] = na
                    mach_jobs[mm] = nb
                    costs[a] = ca
                    costs[mm] = cb
                    mach_of[j] = mm
                    improved_cb()
                    return True
        return False

    def swap_pass():
        for j in range(n):
            if time.time() >= deadline:
                return None
            a = mach_of[j]
            la = mach_jobs[a]
            ia = la.index(j)
            na_base = la[:ia] + la[ia + 1:]
            for k in range(j + 1, n):
                b = mach_of[k]
                if b == a:
                    continue
                na = insert_wspt(na_base, k, a, rat)
                ca = cost_sorted(na, a, p, w)
                lb = mach_jobs[b]
                ib = lb.index(k)
                nb0 = lb[:ib] + lb[ib + 1:]
                nb = insert_wspt(nb0, j, b, rat)
                cb = cost_sorted(nb, b, p, w)
                if ca + cb - costs[a] - costs[b] < -EPS:
                    mach_jobs[a] = na
                    mach_jobs[b] = nb
                    costs[a] = ca
                    costs[b] = cb
                    mach_of[j] = b
                    mach_of[k] = a
                    improved_cb()
                    return True
        return False

    improved = True
    while improved and time.time() < deadline:
        improved = False
        r = relocation_pass()
        while r:
            improved = True
            if time.time() >= deadline:
                return
            r = relocation_pass()
        if r is None:
            return
        s = swap_pass()
        if s is None:
            return
        if s:
            improved = True


# ---------------------------------------------------------------------------
# Weighted tardy jobs helpers (machine lists kept in EDD order)
# ---------------------------------------------------------------------------

def insert_edd(jobs_sorted, j, d):
    dj = d[j]
    pos = len(jobs_sorted)
    for idx, jj in enumerate(jobs_sorted):
        if dj < d[jj] or (dj == d[jj] and j < jj):
            pos = idx
            break
    return jobs_sorted[:pos] + [j] + jobs_sorted[pos:]


def feasible_insert_sorted(jobs_sorted, j, mm, p, d):
    """True iff inserting j (EDD position) keeps all jobs on time."""
    dj = d[j]
    pj = p[j][mm]
    t = 0.0
    inserted = False
    for jj in jobs_sorted:
        if not inserted and (dj < d[jj] or (dj == d[jj] and j < jj)):
            t += pj
            if t > dj + EPS:
                return False
            inserted = True
        t += p[jj][mm]
        if t > d[jj] + EPS:
            return False
    if not inserted:
        t += pj
        if t > dj + EPS:
            return False
    return True


def build_solution_v2(mach_jobs, p, w, d, n, m):
    schedule = {}
    completion = {}
    on_time = []
    for mm in range(m):
        jobs = sorted(mach_jobs[mm], key=lambda j: (d[j], j))
        # safety repair (Moore-Hodgson style): drop cheapest prefix job on violation
        feasible = False
        while not feasible:
            feasible = True
            t = 0.0
            for idx in range(len(jobs)):
                j = jobs[idx]
                t += p[j][mm]
                if t > d[j] + 1e-6:
                    drop_idx = min(range(idx + 1),
                                   key=lambda ii: (w[jobs[ii]], jobs[ii]))
                    jobs.pop(drop_idx)
                    feasible = False
                    break
        schedule[str(mm)] = [int(j) for j in jobs]
        t = 0.0
        for j in jobs:
            t += p[j][mm]
            completion[str(j)] = t
            on_time.append(j)
    on_set = set(on_time)
    tardy = [j for j in range(n) if j not in on_set]
    obj = float(sum(w[j] for j in tardy))
    sol = {
        "objective_value": obj,
        "schedule": schedule,
        "completion_times": completion,
        "tardy_jobs": tardy,
        "on_time_jobs": sorted(on_time),
    }
    return obj, sol


def greedy_v2(order, n, m, p, w, d, deadline):
    mach_jobs = [[] for _ in range(m)]
    for j in order:
        if time.time() >= deadline:
            break
        best_mm = -1
        best_pj = None
        for mm in range(m):
            if feasible_insert_sorted(mach_jobs[mm], j, mm, p, d):
                pj = p[j][mm]
                if best_pj is None or pj < best_pj:
                    best_pj = pj
                    best_mm = mm
        if best_mm >= 0:
            mach_jobs[best_mm] = insert_edd(mach_jobs[best_mm], j, d)
    return mach_jobs


def local_search_v2(mach_jobs, p, w, d, n, m, deadline, improved_cb):
    mach_of = [-1] * n
    for mm in range(m):
        for j in mach_jobs[mm]:
            mach_of[j] = mm

    def swap_pass():
        tardy = sorted((j for j in range(n) if mach_of[j] < 0),
                       key=lambda j: (-w[j], j))
        ontime = sorted((j for j in range(n) if mach_of[j] >= 0),
                        key=lambda j: (w[j], j))
        for j in tardy:
            if time.time() >= deadline:
                return None
            wj = w[j]
            for k in ontime:
                if w[k] >= wj - 1e-12:
                    break
                a = mach_of[k]
                la = mach_jobs[a]
                ia = la.index(k)
                cand0 = la[:ia] + la[ia + 1:]
                if feasible_insert_sorted(cand0, j, a, p, d):
                    mach_jobs[a] = insert_edd(cand0, j, d)
                    mach_of[k] = -1
                    mach_of[j] = a
                    improved_cb()
                    return True
        return False

    improved = True
    while improved and time.time() < deadline:
        improved = False
        tardy = sorted((j for j in range(n) if mach_of[j] < 0),
                       key=lambda j: (-w[j], j))
        for j in tardy:
            if time.time() >= deadline:
                return
            best_mm = -1
            best_pj = None
            for mm in range(m):
                if feasible_insert_sorted(mach_jobs[mm], j, mm, p, d):
                    pj = p[j][mm]
                    if best_pj is None or pj < best_pj:
                        best_pj = pj
                        best_mm = mm
            if best_mm >= 0:
                mach_jobs[best_mm] = insert_edd(mach_jobs[best_mm], j, d)
                mach_of[j] = best_mm
                improved = True
                improved_cb()
        if time.time() >= deadline:
            return
        r = swap_pass()
        if r is None:
            return
        if r:
            improved = True


def lawler_moore(n, p1, w, d):
    """Exact DP for 1 || sum w_j U_j (pseudo-poly, Pareto-pruned -> O(n^2))."""
    order = sorted(range(n), key=lambda j: (d[j], j))
    levels = [[(0.0, 0.0, -1, -1)]]  # (time, weight, prev_idx, job_taken)
    for j in order:
        prev = levels[-1]
        cur = [(t, wt, k, -1) for k, (t, wt, _, _) in enumerate(prev)]
        pj = p1[j]
        dj = d[j] + EPS
        wj = w[j]
        for k, (t, wt, _, _) in enumerate(prev):
            nt = t + pj
            if nt <= dj:
                cur.append((nt, wt + wj, k, j))
        cur.sort(key=lambda s: (s[0], -s[1]))
        pruned = []
        bw = -1.0
        for s in cur:
            if s[1] > bw + 1e-12:
                pruned.append(s)
                bw = s[1]
        levels.append(pruned)
    states = levels[-1]
    idx = len(states) - 1
    ontime = []
    L = len(levels) - 1
    while L > 0 and idx >= 0:
        st = levels[L][idx]
        if st[3] >= 0:
            ontime.append(st[3])
        idx = st[2]
        L -= 1
    return ontime


# ---------------------------------------------------------------------------
# Gurobi MIP models
# ---------------------------------------------------------------------------

def _set_params(model, time_remaining):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0.1, time_remaining)


def gurobi_v1(n, m, p, w, rat, init_mj, time_remaining, consider):
    model = gp.Model("wct")
    _set_params(model, time_remaining)

    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    keys = []
    for mm in range(m):
        order = sorted(range(n), key=lambda j: (rat[j][mm], j))
        for a in range(n - 1):
            i = order[a]
            for b in range(a + 1, n):
                keys.append((mm, i, order[b]))
    y = model.addVars(keys, lb=0.0, ub=1.0, name="y")

    model.addConstrs((x.sum(j, "*") == 1.0 for j in range(n)), name="assign")
    model.addConstrs((y[mm, i, j] >= x[i, mm] + x[j, mm] - 1.0
                      for (mm, i, j) in keys), name="link")

    coeffs = []
    varlist = []
    for j in range(n):
        wj = w[j]
        pj = p[j]
        for mm in range(m):
            coeffs.append(wj * pj[mm])
            varlist.append(x[j, mm])
    for (mm, i, j) in keys:
        coeffs.append(w[j] * p[i][mm])
        varlist.append(y[mm, i, j])
    model.setObjective(gp.LinExpr(coeffs, varlist), GRB.MINIMIZE)

    for mm in range(m):
        for j in init_mj[mm]:
            x[j, mm].Start = 1.0

    xvars = [x[j, mm] for j in range(n) for mm in range(m)]

    def _collect(vals):
        mj = [[] for _ in range(m)]
        for j in range(n):
            base = j * m
            best_mm = -1
            best_v = 0.5
            for mm in range(m):
                v = vals[base + mm]
                if v > best_v:
                    best_v = v
                    best_mm = mm
            if best_mm < 0:  # numerical safety: force assignment
                best_mm = 0
                bv = vals[base]
                for mm in range(1, m):
                    if vals[base + mm] > bv:
                        bv = vals[base + mm]
                        best_mm = mm
            mj[best_mm].append(j)
        return mj

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                vals = model.cbGetSolution(xvars)
                mj = _collect(vals)
                o, s = build_solution_v1(mj, p, w, rat, n, m)
                consider(o, s)
            except Exception:
                pass

    model.optimize(cb)

    if model.SolCount > 0:
        try:
            vals = [x[j, mm].X for j in range(n) for mm in range(m)]
            mj = _collect(vals)
            o, s = build_solution_v1(mj, p, w, rat, n, m)
            consider(o, s)
        except Exception:
            pass


def gurobi_v2(n, m, p, w, d, init_mj, time_remaining, consider):
    model = gp.Model("wtj")
    _set_params(model, time_remaining)

    ok = [[p[j][mm] <= d[j] + 1e-9 for mm in range(m)] for j in range(n)]
    xkeys = [(j, mm) for j in range(n) for mm in range(m) if ok[j][mm]]
    x = model.addVars(xkeys, vtype=GRB.BINARY, name="x")

    keys = []
    preds = [[] for _ in range(n)]
    for mm in range(m):
        js = sorted((j for j in range(n) if ok[j][mm]),
                    key=lambda j: (d[j], j))
        L = len(js)
        for b in range(L):
            j = js[b]
            for a in range(b):
                i = js[a]
                keys.append((mm, i, j))
                preds[j].append((mm, i))
    y = model.addVars(keys, lb=0.0, ub=1.0, name="y")

    model.addConstrs((x.sum(j, "*") <= 1.0
                      for j in range(n) if any(ok[j])), name="atmost1")
    model.addConstrs((y[mm, i, j] >= x[i, mm] + x[j, mm] - 1.0
                      for (mm, i, j) in keys), name="link")
    for j in range(n):
        coeffs = []
        vs = []
        for mm in range(m):
            if ok[j][mm]:
                coeffs.append(p[j][mm])
                vs.append(x[j, mm])
        for (mm, i) in preds[j]:
            coeffs.append(p[i][mm])
            vs.append(y[mm, i, j])
        if vs:
            model.addConstr(gp.LinExpr(coeffs, vs) <= d[j], name="due_%d" % j)

    coeffs = [w[j] for (j, mm) in xkeys]
    vs = [x[j, mm] for (j, mm) in xkeys]
    model.setObjective(gp.LinExpr(coeffs, vs), GRB.MAXIMIZE)

    for mm in range(m):
        for j in init_mj[mm]:
            if ok[j][mm]:
                x[j, mm].Start = 1.0

    xvars = [x[k] for k in xkeys]

    def _collect(vals):
        mj = [[] for _ in range(m)]
        for val, (j, mm) in zip(vals, xkeys):
            if val > 0.5:
                mj[mm].append(j)
        return mj

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                vals = model.cbGetSolution(xvars)
                mj = _collect(vals)
                o, s = build_solution_v2(mj, p, w, d, n, m)
                consider(o, s)
            except Exception:
                pass

    model.optimize(cb)

    if model.SolCount > 0:
        try:
            vals = [v.X for v in xvars]
            mj = _collect(vals)
            o, s = build_solution_v2(mj, p, w, d, n, m)
            consider(o, s)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main solve driver
# ---------------------------------------------------------------------------

def solve(inst, time_limit, logger):
    n, m, w, p, d = extract_data(inst)
    start = time.time()
    deadline = start + max(0.0, float(time_limit))
    buffer = min(1.0, max(0.1, 0.05 * max(1, time_limit)))

    best = {"obj": float("inf"), "sol": None}

    def consider(obj, sol):
        if obj is not None and obj < best["obj"] - EPS:
            best["obj"] = float(obj)
            best["sol"] = sol
            if logger:
                try:
                    logger.log_solution(float(obj), sol)
                except Exception:
                    pass

    ptype = str(inst.get("problem_type") or "").lower()
    ob = str(inst.get("objective") or "").lower()
    is_tardy = ("tardy" in ptype) or ("tardy" in ob)

    # degenerate instance
    if n == 0 or m == 0:
        schedule = {str(mm): [] for mm in range(m)}
        if is_tardy:
            sol = {"objective_value": float(sum(w)), "schedule": schedule,
                   "completion_times": {}, "tardy_jobs": list(range(n)),
                   "on_time_jobs": []}
        else:
            sol = {"objective_value": 0.0, "schedule": schedule,
                   "completion_times": {}, "tardy_jobs": [],
                   "on_time_jobs": list(range(n))}
        if logger:
            try:
                logger.log_solution(sol["objective_value"], sol)
            except Exception:
                pass
        return sol

    if time_limit >= 60:
        pair_limit = 250000
    elif time_limit >= 20:
        pair_limit = 120000
    else:
        pair_limit = 50000

    if not is_tardy:
        rat = compute_ratios(p, w, n, m)
        if m == 1:
            mj = [sorted(range(n), key=lambda j: (rat[j][0], j))]
            obj, sol = build_solution_v1(mj, p, w, rat, n, m)
            consider(obj, sol)
            return best["sol"]

        mj = greedy_v1(n, m, p, w, rat)
        obj, sol = build_solution_v1(mj, p, w, rat, n, m)
        consider(obj, sol)

        def improve_cb():
            o, s = build_solution_v1(mj, p, w, rat, n, m)
            consider(o, s)

        pair_count = m * n * (n - 1) // 2
        use_mip = HAVE_GUROBI and pair_count <= pair_limit
        if use_mip:
            ls_deadline = min(deadline - buffer,
                              time.time() + min(12.0, 0.2 * time_limit))
        else:
            ls_deadline = deadline - buffer
        if ls_deadline > time.time():
            local_search_v1(mj, p, w, rat, n, m, ls_deadline, improve_cb)
        rem = deadline - buffer - time.time()
        est_build = pair_count / 80000.0 + 0.5
        if use_mip and rem > est_build + 1.0:
            try:
                gurobi_v1(n, m, p, w, rat, mj, rem - est_build, consider)
            except Exception:
                pass
        return best["sol"]

    # ---- weighted tardy jobs ----
    if d is None:
        d = [0.0] * n

    if m == 1 and n <= 1000:
        ontime = lawler_moore(n, [p[j][0] for j in range(n)], w, d)
        mj = [sorted(ontime, key=lambda j: (d[j], j))]
        obj, sol = build_solution_v2(mj, p, w, d, n, m)
        consider(obj, sol)
        return best["sol"]

    # always-feasible fallback: everything tardy
    obj, sol = build_solution_v2([[] for _ in range(m)], p, w, d, n, m)
    consider(obj, sol)

    minp = [min(p[j]) for j in range(n)]
    orders = [
        sorted(range(n), key=lambda j: (-w[j], d[j], j)),
        sorted(range(n), key=lambda j: (d[j], -w[j], j)),
        sorted(range(n),
               key=lambda j: (-(w[j] / max(minp[j], 1e-12)), d[j], j)),
    ]
    best_mj = None
    best_on = -1.0
    for order in orders:
        if time.time() >= deadline - buffer:
            break
        cand = greedy_v2(order, n, m, p, w, d, deadline - buffer)
        onw = sum(w[j] for mm in range(m) for j in cand[mm])
        if onw > best_on:
            best_on = onw
            best_mj = cand
    if best_mj is None:
        best_mj = [[] for _ in range(m)]
    mj = best_mj
    obj, sol = build_solution_v2(mj, p, w, d, n, m)
    consider(obj, sol)

    def improve_cb2():
        o, s = build_solution_v2(mj, p, w, d, n, m)
        consider(o, s)

    # effective pair count (jobs must be able to meet their due date)
    pair_count = 0
    for mm in range(m):
        L = sum(1 for j in range(n) if p[j][mm] <= d[j] + EPS)
        pair_count += L * (L - 1) // 2
    use_mip = HAVE_GUROBI and pair_count <= pair_limit
    if use_mip:
        ls_deadline = min(deadline - buffer,
                          time.time() + min(12.0, 0.2 * time_limit))
    else:
        ls_deadline = deadline - buffer
    if ls_deadline > time.time():
        local_search_v2(mj, p, w, d, n, m, ls_deadline, improve_cb2)
    rem = deadline - buffer - time.time()
    est_build = pair_count / 80000.0 + 0.5
    if use_mip and rem > est_build + 1.0:
        try:
            gurobi_v2(n, m, p, w, d, mj, rem - est_build, consider)
        except Exception:
            pass
    return best["sol"]


def fallback_solution(inst):
    try:
        n, m, w, p, d = extract_data(inst)
    except Exception:
        return {"objective_value": 0.0, "schedule": {"0": []},
                "completion_times": {}, "tardy_jobs": [], "on_time_jobs": []}
    ptype = str(inst.get("problem_type") or "").lower()
    ob = str(inst.get("objective") or "").lower()
    is_tardy = ("tardy" in ptype) or ("tardy" in ob)
    if is_tardy:
        return {"objective_value": float(sum(w)),
                "schedule": {str(mm): [] for mm in range(m)},
                "completion_times": {},
                "tardy_jobs": list(range(n)), "on_time_jobs": []}
    schedule = {str(mm): [] for mm in range(m)}
    schedule["0"] = list(range(n))
    completion = {}
    t = 0.0
    obj = 0.0
    for j in range(n):
        t += p[j][0]
        completion[str(j)] = t
        obj += w[j] * t
    return {"objective_value": float(obj), "schedule": schedule,
            "completion_times": completion,
            "tardy_jobs": [], "on_time_jobs": list(range(n))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    inst = load_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    try:
        sol = solve(inst, args.time_limit, logger)
    except Exception:
        sol = fallback_solution(inst)

    if sol is None:
        sol = fallback_solution(inst)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()