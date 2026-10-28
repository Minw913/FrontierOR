import argparse
import json
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    dim = inst["dimension"]
    bars = inst["bars"]
    nb = len(bars)
    dofs = inst["degrees_of_freedom"]
    nd = len(dofs)
    loads = inst["loading_conditions"]
    nl = len(loads)
    areas = list(inst["discrete_areas"])
    K = len(areas)
    mat = inst["material_properties"]
    E_glob = float(mat["modulus_of_elasticity"])
    rho = float(mat["cost_density"])

    dof_idx = {d["dof_id"]: i for i, d in enumerate(dofs)}
    dof_map = {(d["node"], d["direction"]): i for i, d in enumerate(dofs)}

    # ---- stress bounds ----
    sb = inst.get("stress_bounds") or {}
    sL_def = sb.get("lower", -1e30)
    sU_def = sb.get("upper", 1e30)
    if sL_def is None:
        sL_def = -1e30
    if sU_def is None:
        sU_def = 1e30
    spec = {}
    for r in inst.get("bar_specific_stress_bounds") or []:
        lo = r.get("lower", sL_def)
        hi = r.get("upper", sU_def)
        spec[r["bar_id"]] = (sL_def if lo is None else lo, sU_def if hi is None else hi)

    db = inst.get("displacement_bounds") or {}
    dlo = db.get("lower", None)
    dup = db.get("upper", None)

    # ---- geometry: bar lengths, moduli, R matrix columns ----
    Ls = []
    Es = []
    rcols = []  # list of dict {dof_index: coeff}
    for b in bars:
        L = float(b["length"])
        Ls.append(L)
        Es.append(float(b.get("modulus_of_elasticity", E_glob)))
        c = b["direction_cosines"]
        col = {}
        for a in range(dim):
            nm = "xyz"[a]
            j = dof_map.get((b["node_j"], nm))
            if j is not None:
                col[j] = col.get(j, 0.0) + float(c[a])
            i = dof_map.get((b["node_i"], nm))
            if i is not None:
                col[i] = col.get(i, 0.0) - float(c[a])
        rcols.append(col)

    # ---- elongation bounds (from stress bounds, tightened by displacement bounds) ----
    eL = []
    eU = []
    for bi, b in enumerate(bars):
        sl, su = spec.get(b["bar_id"], (sL_def, sU_def))
        lo = sl * Ls[bi] / Es[bi]
        hi = su * Ls[bi] / Es[bi]
        if dlo is not None and dup is not None:
            emin = 0.0
            emax = 0.0
            for j, cc in rcols[bi].items():
                emin += min(cc * dlo, cc * dup)
                emax += max(cc * dlo, cc * dup)
            lo = max(lo, emin)
            hi = min(hi, emax)
        eL.append(lo)
        eU.append(hi)

    # ---- linking groups ----
    bar_id_to_idx = {b["bar_id"]: i for i, b in enumerate(bars)}
    grp_lists = []
    used = set()
    for g in inst.get("linking_groups") or []:
        ids = None
        if isinstance(g, dict):
            for v in g.values():
                if isinstance(v, list) and v and all(isinstance(x, int) for x in v):
                    ids = v
                    break
        elif isinstance(g, list):
            ids = g
        if ids:
            idxs = [bar_id_to_idx[i] for i in ids if i in bar_id_to_idx]
            idxs = [i for i in idxs if i not in used]
            if idxs:
                grp_lists.append(idxs)
                used.update(idxs)
    for i in range(nb):
        if i not in used:
            grp_lists.append([i])
    ngr = len(grp_lists)
    gof = [0] * nb
    for gi, lst in enumerate(grp_lists):
        for i in lst:
            gof[i] = gi

    # ---- external loads ----
    P = np.zeros((nl, nd))
    load_ids = []
    for li, lc in enumerate(loads):
        load_ids.append(lc["load_id"])
        for f in lc["loads"]:
            j = dof_idx.get(f.get("dof_id"))
            if j is None:
                j = dof_map[(f["node"], f["direction"])]
            P[li, j] += float(f["force"])

    # ---- helper to build solution dict ----
    def build_solution(kb, dvals, Fvals):
        obj = sum(rho * Ls[b] * areas[kb[b]] for b in range(nb))
        return {
            "objective_value": float(obj),
            "bar_areas": [
                {"bar_id": bars[b]["bar_id"], "area": float(areas[kb[b]]),
                 "area_index": int(kb[b])}
                for b in range(nb)
            ],
            "displacements": [
                {"dof_id": dofs[j]["dof_id"], "load": load_ids[l],
                 "value": float(dvals[l][j])}
                for l in range(nl) for j in range(nd)
            ],
            "bar_forces": [
                {"bar_id": bars[b]["bar_id"], "load": load_ids[l],
                 "force": float(Fvals[b][l])}
                for b in range(nb) for l in range(nl)
            ],
        }

    # ---- build MILP model ----
    m = gp.Model("truss")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    rem = args.time_limit - (time.time() - t0) - 3.0
    m.Params.TimeLimit = max(5.0, rem)

    y = m.addVars(ngr, K, vtype=GRB.BINARY, name="y")
    for g in range(ngr):
        m.addConstr(gp.quicksum(y[g, k] for k in range(K)) == 1)

    dlb = dlo if dlo is not None else -GRB.INFINITY
    dub = dup if dup is not None else GRB.INFINITY
    d = m.addVars(nl, nd, lb=dlb, ub=dub, name="d")

    elb = [eL[b] for b in range(nb) for l in range(nl)]
    eub = [eU[b] for b in range(nb) for l in range(nl)]
    e = m.addVars(nb, nl, lb=elb, ub=eub, name="e")

    slb = [min(eL[b], 0.0) for b in range(nb) for l in range(nl) for k in range(K)]
    sub = [max(eU[b], 0.0) for b in range(nb) for l in range(nl) for k in range(K)]
    s = m.addVars(nb, nl, K, lb=slb, ub=sub, name="s")

    F = m.addVars(nb, nl, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="F")

    for b in range(nb):
        g = gof[b]
        coef = Es[b] / Ls[b]
        for l in range(nl):
            m.addConstr(e[b, l] == gp.quicksum(s[b, l, k] for k in range(K)))
            m.addConstr(F[b, l] == coef * gp.quicksum(areas[k] * s[b, l, k]
                                                      for k in range(K)))
            for k in range(K):
                m.addConstr(s[b, l, k] <= eU[b] * y[g, k])
                m.addConstr(s[b, l, k] >= eL[b] * y[g, k])
            # compatibility
            m.addConstr(gp.quicksum(cc * d[l, j] for j, cc in rcols[b].items())
                        == e[b, l])

    # equilibrium
    dof_bars = [[] for _ in range(nd)]
    for b in range(nb):
        for j, cc in rcols[b].items():
            dof_bars[j].append((b, cc))
    for l in range(nl):
        for j in range(nd):
            m.addConstr(gp.quicksum(cc * F[b, l] for b, cc in dof_bars[j])
                        == P[l, j])

    m.setObjective(
        gp.quicksum(rho * Ls[b] * gp.quicksum(areas[k] * y[gof[b], k]
                                              for k in range(K))
                    for b in range(nb)),
        GRB.MINIMIZE)

    # ---- warm start: all bars at maximum area, linear elastic analysis ----
    kmax = max(range(K), key=lambda k: areas[k])
    fallback = None
    try:
        Kmat = np.zeros((nd, nd))
        for b in range(nb):
            coef = Es[b] / Ls[b] * areas[kmax]
            items = list(rcols[b].items())
            for j1, c1 in items:
                for j2, c2 in items:
                    Kmat[j1, j2] += coef * c1 * c2
        dstart = np.linalg.solve(Kmat, P.T).T  # nl x nd
        for g in range(ngr):
            for k in range(K):
                y[g, k].Start = 1.0 if k == kmax else 0.0
        Fst = np.zeros((nb, nl))
        for l in range(nl):
            for j in range(nd):
                d[l, j].Start = float(dstart[l, j])
        for b in range(nb):
            coef = Es[b] / Ls[b] * areas[kmax]
            for l in range(nl):
                ee = sum(cc * dstart[l, j] for j, cc in rcols[b].items())
                e[b, l].Start = ee
                for k in range(K):
                    s[b, l, k].Start = ee if k == kmax else 0.0
                Fst[b, l] = coef * ee
                F[b, l].Start = Fst[b, l]
        fallback = build_solution([kmax] * nb, dstart.tolist(), Fst.tolist())
    except Exception:
        fallback = None

    # ---- callback for incumbent logging ----
    ylist = [y[g, k] for g in range(ngr) for k in range(K)]
    dlist = [d[l, j] for l in range(nl) for j in range(nd)]
    Flist = [F[b, l] for b in range(nb) for l in range(nl)]
    m._best = float("inf")
    m._bestsol = None

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj >= m._best - 1e-9:
                return
            m._best = obj
            yv = model.cbGetSolution(ylist)
            dv = model.cbGetSolution(dlist)
            Fv = model.cbGetSolution(Flist)
            kg = []
            for g in range(ngr):
                vals = yv[g * K:(g + 1) * K]
                kg.append(max(range(K), key=lambda k: vals[k]))
            kb = [kg[gof[b]] for b in range(nb)]
            dvals = [[dv[l * nd + j] for j in range(nd)] for l in range(nl)]
            Fvals = [[Fv[b * nl + l] for l in range(nl)] for b in range(nb)]
            sol = build_solution(kb, dvals, Fvals)
            m._bestsol = sol
            if logger:
                logger.log_solution(sol["objective_value"], sol)

    try:
        m.optimize(cb)
    except gp.GurobiError:
        pass

    # ---- extract final solution ----
    sol = None
    if m.SolCount > 0:
        try:
            yv = [v.X for v in ylist]
            dv = [v.X for v in dlist]
            Fv = [v.X for v in Flist]
            kg = []
            for g in range(ngr):
                vals = yv[g * K:(g + 1) * K]
                kg.append(max(range(K), key=lambda k: vals[k]))
            kb = [kg[gof[b]] for b in range(nb)]
            dvals = [[dv[l * nd + j] for j in range(nd)] for l in range(nl)]
            Fvals = [[Fv[b * nl + l] for l in range(nl)] for b in range(nb)]
            sol = build_solution(kb, dvals, Fvals)
            if logger and sol["objective_value"] < m._best - 1e-9:
                logger.log_solution(sol["objective_value"], sol)
        except Exception:
            sol = None
    if sol is None:
        sol = m._bestsol
    if sol is None:
        sol = fallback
    if sol is None:
        # trivial last-resort output
        zero_d = [[0.0] * nd for _ in range(nl)]
        zero_F = [[0.0] * nl for _ in range(nb)]
        sol = build_solution([0] * nb, zero_d, zero_F)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()