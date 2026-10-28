import argparse
import json
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

GINF = GRB.INFINITY
BND = 1e7  # safety clamp for (effectively) infinite elongation bounds


# --------------------------------------------------------------------------- #
# Instance parsing
# --------------------------------------------------------------------------- #

def _axis_index(direction):
    s = str(direction).strip().lower()
    return {"x": 0, "y": 1, "z": 2}.get(s[0], 0)


def _parse_linking_groups(data, bar_index, n):
    raw = data.get("linking_groups") or []
    groups, assigned = [], set()
    for g in raw:
        members = None
        if isinstance(g, dict):
            for key in ("bars", "bar_ids", "members", "member_bars", "bar_list", "group"):
                v = g.get(key)
                if isinstance(v, list):
                    members = v
                    break
            if members is None:
                for v in g.values():
                    if isinstance(v, list):
                        members = v
                        break
        elif isinstance(g, (list, tuple)):
            members = list(g)
        if not members:
            continue
        idxs = []
        for b in members:
            if isinstance(b, dict):
                b = b.get("bar_id", b.get("bar"))
            try:
                e = bar_index[int(b)]
            except Exception:
                continue
            if e not in assigned:
                idxs.append(e)
        idxs = sorted(set(idxs))
        if idxs:
            groups.append(idxs)
            assigned.update(idxs)
    for e in range(n):
        if e not in assigned:
            groups.append([e])
    return groups


def prepare(data):
    bars = data.get("bars", []) or []
    n = len(bars)
    mp = data.get("material_properties", {}) or {}
    E = float(mp.get("modulus_of_elasticity", 1.0))
    rho = float(mp.get("cost_density", 1.0))
    areas = [float(a) for a in (data.get("discrete_areas") or [])]
    if not areas:
        areas = [0.01]
    m = len(areas)
    dim = int(data.get("dimension", 2) or 2)

    bar_ids = [b.get("bar_id", e) for e, b in enumerate(bars)]
    bar_index = {bid: e for e, bid in enumerate(bar_ids)}
    L = [max(float(b.get("length", 1.0)), 1e-12) for b in bars]
    EoverL = [E / L[e] for e in range(n)]

    # Degrees of freedom
    dofs = data.get("degrees_of_freedom")
    if not dofs:
        dofs = []
        did = 0
        for node in data.get("nodes", []) or []:
            if node.get("is_support", False):
                continue
            for a in "xyz"[:dim]:
                dofs.append({"dof_id": did, "node": node.get("node_id"), "direction": a})
                did += 1
    D = len(dofs)
    dof_ids = [d.get("dof_id", i) for i, d in enumerate(dofs)]
    dof_index = {did: i for i, did in enumerate(dof_ids)}
    node_dofs = {}
    for i, d in enumerate(dofs):
        ax = _axis_index(d.get("direction", "x"))
        node_dofs.setdefault(d.get("node"), []).append((i, ax))

    # Kinematic coefficients gamma[e] = [(dof_idx, coef), ...]
    gamma = []
    for e, b in enumerate(bars):
        cos = [float(v) for v in (b.get("direction_cosines") or [0.0] * dim)]
        entry = {}
        for (di, ax) in node_dofs.get(b.get("node_j"), []):
            if ax < len(cos):
                entry[di] = entry.get(di, 0.0) + cos[ax]
        for (di, ax) in node_dofs.get(b.get("node_i"), []):
            if ax < len(cos):
                entry[di] = entry.get(di, 0.0) - cos[ax]
        gamma.append([(di, cf) for di, cf in entry.items() if abs(cf) > 1e-14])

    dof_bars = [[] for _ in range(D)]
    for e in range(n):
        for (d, cf) in gamma[e]:
            dof_bars[d].append((e, cf))

    # Loading conditions
    conds = data.get("loading_conditions", []) or []
    C = len(conds)
    load_ids = [c.get("load_id", i) for i, c in enumerate(conds)]
    F = [[0.0] * C for _ in range(D)]
    for c, cond in enumerate(conds):
        for ld in (cond.get("loads", []) or []):
            i = dof_index.get(ld.get("dof_id"))
            if i is None:
                tgt_node = ld.get("node")
                tgt_dir = str(ld.get("direction", "")).strip().lower()[:1]
                for dd in dofs:
                    if dd.get("node") == tgt_node and \
                            str(dd.get("direction", "")).strip().lower()[:1] == tgt_dir:
                        i = dof_index.get(dd.get("dof_id"))
                        break
            if i is not None:
                F[i][c] += float(ld.get("force", 0.0))

    # Stress bounds -> elongation bounds  (delta = sigma * L / E)
    sb = data.get("stress_bounds") or {}
    slo_d = sb.get("lower")
    shi_d = sb.get("upper")
    slo_d = float(slo_d) if slo_d is not None else -1e30
    shi_d = float(shi_d) if shi_d is not None else 1e30
    sig_lo = [slo_d] * n
    sig_hi = [shi_d] * n
    for item in (data.get("bar_specific_stress_bounds") or []):
        e = bar_index.get(item.get("bar_id"))
        if e is None:
            continue
        if item.get("lower") is not None:
            sig_lo[e] = float(item["lower"])
        if item.get("upper") is not None:
            sig_hi[e] = float(item["upper"])
    lo_e = [sig_lo[e] * L[e] / E for e in range(n)]
    hi_e = [sig_hi[e] * L[e] / E for e in range(n)]

    # Optional explicit elongation bounds (not in schema, but handle if present)
    eb = data.get("elongation_bounds") or {}
    elo = eb.get("lower")
    ehi = eb.get("upper")
    bar_el = {}
    for key in ("bar_specific_elongation_bounds", "bar_elongation_bounds"):
        for item in (data.get(key) or []):
            bar_el[item.get("bar_id")] = item
    for e in range(n):
        lo2, hi2 = elo, ehi
        it = bar_el.get(bar_ids[e])
        if it:
            if it.get("lower") is not None:
                lo2 = it["lower"]
            if it.get("upper") is not None:
                hi2 = it["upper"]
        if lo2 is not None:
            lo_e[e] = max(lo_e[e], float(lo2))
        if hi2 is not None:
            hi_e[e] = min(hi_e[e], float(hi2))
    for e in range(n):
        lo_e[e] = max(lo_e[e], -BND)
        hi_e[e] = min(hi_e[e], BND)
        if lo_e[e] > hi_e[e]:
            lo_e[e], hi_e[e] = hi_e[e], lo_e[e]

    # Displacement bounds (global, may be null)
    db = data.get("displacement_bounds") or {}
    d_lo = db.get("lower")
    d_hi = db.get("upper")
    d_lo = float(d_lo) if d_lo is not None else None
    d_hi = float(d_hi) if d_hi is not None else None

    # Linking groups
    groups = _parse_linking_groups(data, bar_index, n)
    group_of_bar = [0] * n
    for g, members in enumerate(groups):
        for e in members:
            group_of_bar[e] = g
    group_len = [sum(L[e] for e in members) for members in groups]

    expected = data.get("expected_optimal_mass")
    expected = float(expected) if expected is not None else None

    return {
        "n": n, "m": m, "D": D, "C": C, "dim": dim,
        "E": E, "rho": rho, "areas": areas,
        "bar_ids": bar_ids, "bar_index": bar_index, "L": L, "EoverL": EoverL,
        "dof_ids": dof_ids, "dof_index": dof_index,
        "gamma": gamma, "dof_bars": dof_bars,
        "load_ids": load_ids, "F": F,
        "lo_e": lo_e, "hi_e": hi_e,
        "d_lo": d_lo, "d_hi": d_hi,
        "groups": groups, "group_of_bar": group_of_bar, "group_len": group_len,
        "expected": expected,
    }


# --------------------------------------------------------------------------- #
# Structural response (direct stiffness) and feasibility checking
# --------------------------------------------------------------------------- #

def compute_response(P, idx):
    n, D, C = P["n"], P["D"], P["C"]
    areas, E, L = P["areas"], P["E"], P["L"]
    K = np.zeros((D, D))
    for e in range(n):
        gd = P["gamma"][e]
        if not gd:
            continue
        ke = E * areas[idx[e]] / L[e]
        ds = [d for d, _ in gd]
        cs = np.array([cf for _, cf in gd])
        K[np.ix_(ds, ds)] += ke * np.outer(cs, cs)
    Fm = np.array(P["F"], dtype=float) if (D and C) else np.zeros((D, C))
    if D == 0 or C == 0:
        U = np.zeros((D, C))
        resid = 0.0
    else:
        try:
            U = np.linalg.solve(K, Fm)
        except np.linalg.LinAlgError:
            U = np.linalg.lstsq(K, Fm, rcond=None)[0]
        R = K.dot(U) - Fm
        fscale = max(1.0, float(np.abs(Fm).max()) if Fm.size else 1.0)
        resid = float(np.abs(R).max()) / fscale if R.size else 0.0
    DL = np.zeros((n, C))
    if C:
        for e in range(n):
            gd = P["gamma"][e]
            if gd:
                ds = [d for d, _ in gd]
                cs = np.array([cf for _, cf in gd])
                DL[e, :] = cs.dot(U[ds, :])
    return U, DL, resid


def check_feasible(P, U, DL, resid, tol=1e-6):
    if resid > 1e-5:
        return False
    D, C, n = P["D"], P["C"], P["n"]
    d_lo, d_hi = P["d_lo"], P["d_hi"]
    if d_lo is not None or d_hi is not None:
        for d in range(D):
            for c in range(C):
                v = U[d][c]
                if d_lo is not None and v < d_lo - tol * (1.0 + abs(d_lo)):
                    return False
                if d_hi is not None and v > d_hi + tol * (1.0 + abs(d_hi)):
                    return False
    lo_e, hi_e = P["lo_e"], P["hi_e"]
    for e in range(n):
        lo, hi = lo_e[e], hi_e[e]
        t = tol * (1.0 + max(abs(lo), abs(hi)))
        for c in range(C):
            v = DL[e][c]
            if v < lo - t or v > hi + t:
                return False
    return True


def cost_of(P, idx):
    return P["rho"] * sum(P["L"][e] * P["areas"][idx[e]] for e in range(P["n"]))


# --------------------------------------------------------------------------- #
# Heuristic: fully stressed design (+ displacement repair), max-area fallback
# --------------------------------------------------------------------------- #

def pick_area(P, a_req, imax):
    areas = P["areas"]
    thr = a_req * (1 - 1e-12)
    best = None
    for k, a in enumerate(areas):
        if a >= thr and (best is None or a < areas[best]):
            best = k
    return best if best is not None else imax


def fsd(P, deadline, imax):
    n, m, C = P["n"], P["m"], P["C"]
    areas = P["areas"]
    groups = P["groups"]
    G = len(groups)
    gob = P["group_of_bar"]
    lo_e, hi_e = P["lo_e"], P["hi_e"]
    d_lo, d_hi = P["d_lo"], P["d_hi"]
    cur = [imax] * G
    best = None
    for _ in range(60):
        if time.time() > deadline:
            break
        idx = [cur[gob[e]] for e in range(n)]
        U, DL, resid = compute_response(P, idx)
        if resid > 1e-3:
            break
        if check_feasible(P, U, DL, resid):
            cost = cost_of(P, idx)
            if best is None or cost < best[0] - 1e-12:
                best = (cost, cur[:], U, DL)
        new = cur[:]
        for g in range(G):
            worst = 0.0
            for e in groups[g]:
                hi, lo = hi_e[e], lo_e[e]
                for c in range(C):
                    dlt = DL[e][c]
                    if dlt >= 0.0:
                        r = dlt / hi if hi > 0 else (0.0 if dlt == 0.0 else float("inf"))
                    else:
                        r = dlt / lo if lo < 0 else float("inf")
                    if r > worst:
                        worst = r
            new[g] = pick_area(P, areas[cur[g]] * worst, imax)
        # displacement-bound repair: scale areas up by violation ratio
        viol = 1.0
        if d_lo is not None or d_hi is not None:
            for d in range(P["D"]):
                for c in range(C):
                    v = U[d][c]
                    if d_hi is not None and d_hi > 0 and v > d_hi:
                        viol = max(viol, v / d_hi)
                    if d_lo is not None and d_lo < 0 and v < d_lo:
                        viol = max(viol, v / d_lo)
        if viol > 1.0:
            for g in range(G):
                new[g] = pick_area(P, areas[new[g]] * viol * 1.02, imax)
        if new == cur:
            break
        cur = new
    return best


def heuristic(P, deadline):
    n, m = P["n"], P["m"]
    areas = P["areas"]
    imax = max(range(m), key=lambda k: areas[k])
    idx_max = [imax] * n
    U, DL, resid = compute_response(P, idx_max)
    feas = check_feasible(P, U, DL, resid)
    fallback = {
        "idx": idx_max, "U": U, "DL": DL, "feasible": feas,
        "group_sel": [imax for _ in P["groups"]],
    }
    best = fallback if feas else None
    try:
        fsdb = fsd(P, deadline, imax)
    except Exception:
        fsdb = None
    if fsdb is not None:
        cost, cur_g, U2, DL2 = fsdb
        idx2 = [cur_g[P["group_of_bar"][e]] for e in range(n)]
        if best is None or cost < cost_of(P, best["idx"]) - 1e-12:
            best = {"idx": idx2, "U": U2, "DL": DL2, "feasible": True,
                    "group_sel": list(cur_g)}
    return best if best is not None else fallback


# --------------------------------------------------------------------------- #
# Solution assembly
# --------------------------------------------------------------------------- #

def assemble_solution(P, idx, U, DL):
    n, D, C = P["n"], P["D"], P["C"]
    areas = P["areas"]
    obj = float(cost_of(P, idx))
    sol = {
        "objective_value": obj,
        "bar_areas": [
            {"bar_id": P["bar_ids"][e], "area": float(areas[idx[e]]),
             "area_index": int(idx[e])}
            for e in range(n)
        ],
        "displacements": [
            {"dof_id": P["dof_ids"][d], "load": P["load_ids"][c],
             "value": float(U[d][c])}
            for d in range(D) for c in range(C)
        ],
        "bar_forces": [
            {"bar_id": P["bar_ids"][e], "load": P["load_ids"][c],
             "force": float(P["EoverL"][e] * areas[idx[e]] * DL[e][c])}
            for e in range(n) for c in range(C)
        ],
    }
    return obj, sol


def assemble_from_flat(P, yv, uv, dv):
    n, m, D, C = P["n"], P["m"], P["D"], P["C"]
    gob = P["group_of_bar"]
    sel = []
    for g in range(len(P["groups"])):
        base = g * m
        row = yv[base:base + m]
        sel.append(max(range(m), key=lambda kk: row[kk]))
    idx = [sel[gob[e]] for e in range(n)]
    U = [[float(uv[d * C + c]) for c in range(C)] for d in range(D)]
    DL = [[float(dv[e * C + c]) for c in range(C)] for e in range(n)]
    return assemble_solution(P, idx, U, DL)


def last_resort(P):
    m = P["m"]
    imax = max(range(m), key=lambda k: P["areas"][k])
    idx = [imax] * P["n"]
    U = np.zeros((P["D"], P["C"]))
    DL = np.zeros((P["n"], P["C"]))
    return assemble_solution(P, idx, U, DL)


# --------------------------------------------------------------------------- #
# MILP (simultaneous analysis and design, big-M linearization of A*delta)
# --------------------------------------------------------------------------- #

def solve_mip(P, args, t0, heur, logger, state):
    n, m, D, C = P["n"], P["m"], P["D"], P["C"]
    groups = P["groups"]
    G = len(groups)
    gob = P["group_of_bar"]
    areas, EoverL, F = P["areas"], P["EoverL"], P["F"]
    lo_e, hi_e = P["lo_e"], P["hi_e"]
    d_lo_gp = P["d_lo"] if P["d_lo"] is not None else -GINF
    d_hi_gp = P["d_hi"] if P["d_hi"] is not None else GINF

    model = gp.Model("truss_discrete")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1
    model.Params.OutputFlag = 0
    remaining = args.time_limit - (time.time() - t0) - 1.0
    model.Params.TimeLimit = max(1.0, remaining)

    y = model.addVars(G, m, vtype=GRB.BINARY, name="y")
    u = model.addVars(D, C, lb=d_lo_gp, ub=d_hi_gp, name="u")
    dl = {}
    for e in range(n):
        lo, hi = lo_e[e], hi_e[e]
        for c in range(C):
            dl[e, c] = model.addVar(lb=lo, ub=hi, name="dl%d_%d" % (e, c))
    w = model.addVars(n, m, C, lb=-GINF, name="w")

    model.addConstrs((y.sum(g, "*") == 1 for g in range(G)), name="pick")

    gamma = P["gamma"]
    for e in range(n):
        ge = gamma[e]
        for c in range(C):
            model.addConstr(dl[e, c] == gp.quicksum(cf * u[d, c] for d, cf in ge),
                            name="compat%d_%d" % (e, c))

    # w[e,k,c] = y[g,k] * dl[e,c]  (exact big-M using elongation bounds)
    for e in range(n):
        g = gob[e]
        lo, hi = lo_e[e], hi_e[e]
        for k in range(m):
            yv = y[g, k]
            for c in range(C):
                wv = w[e, k, c]
                dv = dl[e, c]
                model.addConstr(wv <= hi * yv)
                model.addConstr(wv >= lo * yv)
                model.addConstr(wv <= dv - lo * (1 - yv))
                model.addConstr(wv >= dv - hi * (1 - yv))

    # Equilibrium: sum_e gamma_ed * (E/L_e) * sum_k a_k w[e,k,c] = F[d,c]
    dof_bars = P["dof_bars"]
    for d in range(D):
        incident = dof_bars[d]
        for c in range(C):
            expr = gp.LinExpr()
            for (e, cf) in incident:
                ke = cf * EoverL[e]
                for k in range(m):
                    expr.add(w[e, k, c], ke * areas[k])
            model.addConstr(expr == F[d][c], name="equil%d_%d" % (d, c))

    obj = gp.LinExpr()
    rho, group_len = P["rho"], P["group_len"]
    for g in range(G):
        cg = rho * group_len[g]
        for k in range(m):
            obj.add(y[g, k], cg * areas[k])
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()

    # MIP start from feasible heuristic solution
    if heur is not None and heur.get("feasible"):
        try:
            sk = heur["group_sel"]
            Uh, DLh = heur["U"], heur["DL"]
            for g in range(G):
                ks = sk[g]
                for k in range(m):
                    y[g, k].Start = 1.0 if k == ks else 0.0
            for d in range(D):
                for c in range(C):
                    u[d, c].Start = float(Uh[d][c])
            for e in range(n):
                ks = sk[gob[e]]
                for c in range(C):
                    val = float(DLh[e][c])
                    dl[e, c].Start = val
                    for k in range(m):
                        w[e, k, c].Start = val if k == ks else 0.0
        except Exception:
            pass

    ylist = [y[g, k] for g in range(G) for k in range(m)]
    ulist = [u[d, c] for d in range(D) for c in range(C)]
    dllist = [dl[e, c] for e in range(n) for c in range(C)]
    expected = P["expected"]

    def _cb(mdl, where):
        try:
            if time.time() - t0 > args.time_limit - 0.5:
                mdl.terminate()
                return
            if where != GRB.Callback.MIPSOL:
                return
            objv = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger is not None and objv < state["last"] - 1e-7:
                try:
                    yv = mdl.cbGetSolution(ylist)
                    uv = mdl.cbGetSolution(ulist)
                    dv = mdl.cbGetSolution(dllist)
                    obj2, sol = assemble_from_flat(P, yv, uv, dv)
                    if obj2 < state["last"] - 1e-9:
                        state["last"] = obj2
                        logger.log_solution(obj2, sol)
                except Exception:
                    try:
                        logger.log(float(objv))
                    except Exception:
                        pass
            if expected is not None and \
                    objv <= expected + 1e-6 * max(1.0, abs(expected)):
                mdl.terminate()
        except Exception:
            pass

    model.optimize(_cb)

    result = None
    try:
        if model.SolCount > 0:
            yv = model.getAttr("X", ylist)
            uv = model.getAttr("X", ulist)
            dv = model.getAttr("X", dllist)
            result = assemble_from_flat(P, yv, uv, dv)
    except Exception:
        result = None
    try:
        model.dispose()
    except Exception:
        pass
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    t0 = time.time()
    with open(args.instance_path) as f:
        data = json.load(f)
    P = prepare(data)

    heur = None
    try:
        heur = heuristic(P, t0 + min(15.0, max(1.0, 0.25 * args.time_limit)))
    except Exception:
        heur = None

    state = {"last": float("inf")}
    best = None
    if heur is not None and heur.get("feasible"):
        obj, sol = assemble_solution(P, heur["idx"], heur["U"], heur["DL"])
        best = (obj, sol)
        state["last"] = obj
        if logger:
            try:
                logger.log_solution(obj, sol)
            except Exception:
                pass

    res = None
    if time.time() - t0 < args.time_limit - 0.5:
        try:
            res = solve_mip(P, args, t0, heur, logger, state)
        except Exception:
            res = None

    if res is not None and (best is None or res[0] < best[0] - 1e-9):
        best = res
        if logger and res[0] < state["last"] - 1e-9:
            try:
                logger.log_solution(res[0], res[1])
            except Exception:
                pass
            state["last"] = res[0]

    if best is None:
        if heur is not None:
            best = assemble_solution(P, heur["idx"], heur["U"], heur["DL"])
        else:
            best = last_resort(P)

    with open(args.solution_path, "w") as f:
        json.dump(best[1], f)


if __name__ == "__main__":
    main()