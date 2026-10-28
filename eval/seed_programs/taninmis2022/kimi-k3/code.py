"""
Multi-stage stochastic generation / transmission / storage expansion planning
with embedded hourly unit commitment (FrontierOR / Taninmis-style benchmark).

Approach
--------
1. If the instance is small enough, solve the full deterministic-equivalent MIP
   over the whole scenario tree x representative days x operational scenarios.
2. Otherwise, solve an investment-planning model using one operational scenario
   per representative day, fix the investment decisions, and then solve
   independent per-(node, day, scenario) unit-commitment MIPs.
3. Greedy fallback ("build everything at the root") if the solver fails.
"""

import argparse
import json
import os
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:  # pragma: no cover
    SolutionLogger = None

VAR_NAMES = ("alpha", "gamma", "p", "s", "ff", "fb", "u", "v", "r")


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    return ap.parse_args()


def load_instance(path):
    with open(path, "r") as fh:
        return json.load(fh)


def set_params(m, timelimit=None):
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.OutputFlag = 0
    if timelimit is not None:
        m.Params.TimeLimit = max(1.0, float(timelimit))


def new_sol():
    return {k: {} for k in VAR_NAMES}


def node_multipliers(inst):
    """Demand-growth / investment-cost multiplier per scenario-tree node."""
    sp = inst.get("scenario_parameters", {}) or {}
    gh = float(sp.get("demand_growth_high_percent", 0.0) or 0.0)
    gl = float(sp.get("demand_growth_low_percent", 0.0) or 0.0)
    cv = float(sp.get("cost_variation_percent", 0.0) or 0.0)
    tree = inst.get("scenario_tree", []) or []
    nodes = {n["id"]: n for n in tree}
    dem, cost = {}, {}
    roots = [n["id"] for n in tree if n.get("parent_id") is None]
    if not roots:
        return dem, cost
    root = roots[0]
    dem[root] = 1.0
    cost[root] = 1.0
    order = [root]
    i = 0
    while i < len(order):
        nid = order[i]
        i += 1
        node = nodes.get(nid)
        if node is None:
            continue
        children = node.get("children") or []
        for j, cid in enumerate(children):
            if cid not in nodes:
                continue
            if len(children) <= 1:
                fd, fc = 1.0, 1.0
            else:
                fd = (1.0 + gh) if (j % 2 == 0) else (1.0 + gl)
                fc = (1.0 + cv) if (j % 2 == 0) else (1.0 - cv)
            dem[cid] = dem[nid] * fd
            cost[cid] = cost[nid] * fc
            order.append(cid)
    for nid in nodes:
        dem.setdefault(nid, 1.0)
        cost.setdefault(nid, 1.0)
    return dem, cost


class Problem:
    pass


def prepare(inst):
    p = Problem()
    sysd = inst.get("system", {}) or {}
    loads = {l["bus"]: float(l.get("annual_peak_load_MW", 0.0) or 0.0)
             for l in (sysd.get("loads") or [])}
    buses = list(sysd.get("buses") or [])
    gens = list(sysd.get("generators") or [])
    lines = list(sysd.get("transmission_lines") or [])
    stors = list(sysd.get("storage") or [])
    buset = set(buses)
    for b in loads:
        buset.add(b)
    for g in gens:
        buset.add(g.get("bus"))
    for l in lines:
        buset.add(l.get("from_bus"))
        buset.add(l.get("to_bus"))
    for k in stors:
        buset.add(k.get("bus"))
    preset = set(buses)
    p.buses = list(buses) + [b for b in buset if b not in preset]
    p.gens = gens
    p.lines = lines
    p.stors = stors
    p.loads = loads

    sp = inst.get("scenario_parameters", {}) or {}
    p.sp = sp
    p.T = int(sp.get("hours_per_day", 24) or 24)
    p.gp = inst.get("global_parameters", {}) or {}

    p.days = list(inst.get("representative_days") or [])
    p.day_by_id = {d["id"]: d for d in p.days}
    ll = inst.get("lower_level_scenarios", {}) or {}
    p.scens = {}
    p.sid = {}
    for d in p.days:
        arr = None
        for key in (str(d["id"]), d["id"]):
            if key in ll:
                arr = ll[key]
                break
        if not arr:
            arr = [{
                "scenario_id": 0,
                "demand_multipliers": [1.0] * p.T,
                "solar_multipliers": [1.0] * p.T,
                "wind_power_factors": [1.0] * p.T,
            }]
        p.scens[d["id"]] = arr
        p.sid[d["id"]] = [sc.get("scenario_id", i) for i, sc in enumerate(arr)]

    p.tree = list(inst.get("scenario_tree") or [])
    p.nodes = {n["id"]: n for n in p.tree}
    p.node_ids = [n["id"] for n in p.tree]
    roots = [n["id"] for n in p.tree if n.get("parent_id") is None]
    p.root = roots[0] if roots else (p.node_ids[0] if p.node_ids else None)
    p.prob = {n["id"]: float(n.get("probability", 0.0) or 0.0) for n in p.tree}

    # topological order (parents before children)
    order, seen = [], set()
    if p.root is not None:
        queue = [p.root]
        while queue:
            nid = queue.pop(0)
            if nid in seen or nid not in p.nodes:
                continue
            seen.add(nid)
            order.append(nid)
            for c in (p.nodes[nid].get("children") or []):
                queue.append(c)
    for nid in p.node_ids:
        if nid not in seen:
            order.append(nid)
    p.node_order = order

    p.gens_at_bus = {b: [] for b in p.buses}
    for g in gens:
        p.gens_at_bus.setdefault(g.get("bus"), []).append(g)
    p.lines_from = {b: [] for b in p.buses}
    p.lines_to = {b: [] for b in p.buses}
    for l in lines:
        p.lines_from.setdefault(l.get("from_bus"), []).append(l)
        p.lines_to.setdefault(l.get("to_bus"), []).append(l)
    p.stor_at_bus = {b: [] for b in p.buses}
    for k in stors:
        p.stor_at_bus.setdefault(k.get("bus"), []).append(k)

    df = p.gp.get("derating_factors", {}) or {}

    def derate(t):
        t = (t or "").lower()
        if t in df:
            return float(df[t])
        if t in ("coal", "natural_gas", "gas", "fossil", "nuclear", "oil", "biomass"):
            return float(df.get("fossil", 1.0) or 1.0)
        return 1.0

    p.derate = {g["id"]: derate(g.get("type")) for g in gens}
    p.is_solar = {g["id"]: (g.get("type") or "").lower() == "solar" for g in gens}
    p.is_wind = {g["id"]: (g.get("type") or "").lower() == "wind" for g in gens}
    p.is_ren = {g["id"]: p.is_solar[g["id"]] or p.is_wind[g["id"]] for g in gens}
    p.marg = {g["id"]: float(g.get("b_cost", 0.0) or 0.0)
              + float(g.get("c_cost", 0.0) or 0.0) * float(g.get("Pmax_MW", 0.0) or 0.0)
              for g in gens}
    p.capG = {g["id"]: float(g.get("capital_cost_per_kW", 0.0) or 0.0) * 1000.0
              * float(g.get("Pmax_MW", 0.0) or 0.0) for g in gens}
    p.capL = {l["id"]: float(l.get("capital_cost_per_kW", 0.0) or 0.0) * 1000.0
              * float(l.get("flow_limit_MW", 0.0) or 0.0) for l in lines}
    p.capK = {k["id"]: float(k.get("capital_cost_per_kW", 0.0) or 0.0) * 1000.0
              * float(k.get("capacity_MW", 0.0) or 0.0) for k in stors}
    p.candG = [g for g in gens if not g.get("existing", False)]
    p.candL = [l for l in lines if not l.get("existing", False)]
    p.candK = [k for k in stors if not k.get("existing", False)]
    return p


# --------------------------------------------------------------------------- #
# operational block
# --------------------------------------------------------------------------- #
def add_ops(m, p, demn, n, d, skey, sc, fracG, capL, capK, V, binarize):
    """Add hourly unit-commitment / dispatch block for one (node, day, scenario)."""
    T = p.T
    day = p.day_by_id[d]
    lp = list(day.get("load_profile") or [1.0] * T)
    solp = list(day.get("solar_profile") or [0.0] * T)
    dm = list(sc.get("demand_multipliers") or [1.0] * T)
    sm = list(sc.get("solar_multipliers") or [1.0] * T)
    wf = list(sc.get("wind_power_factors") or [1.0] * T)
    lp += [1.0] * max(0, T - len(lp))
    solp += [0.0] * max(0, T - len(solp))
    dm += [1.0] * max(0, T - len(dm))
    sm += [1.0] * max(0, T - len(sm))
    wf += [1.0] * max(0, T - len(wf))
    spin = float(p.gp.get("spinning_reserve_percent", 0.0) or 0.0)
    atype = GRB.BINARY if binarize else GRB.CONTINUOUS

    A, Gm, Pv, Sv = {}, {}, {}, {}
    for g in p.gens:
        gid = g["id"]
        Pmax = float(g.get("Pmax_MW", 0.0) or 0.0)
        Pmin = float(g.get("Pmin_MW", 0.0) or 0.0)
        R = float(g.get("ramp_MW_per_h", Pmax) or 0.0)
        UT = max(1, int(g.get("min_on_h", 1) or 1))
        DT = max(1, int(g.get("min_off_h", 1) or 1))
        ren = p.is_ren.get(gid, False)
        solar = p.is_solar.get(gid, False)
        is_cand = not g.get("existing", False)

        a, gm, pv, sv = [], [], [], []
        for t in range(1, T + 1):
            av = m.addVar(lb=0.0, ub=1.0, vtype=atype)
            gv = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
            pw = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
            sr = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
            a.append(av); gm.append(gv); pv.append(pw); sr.append(sr)
            A[(t, gid)] = av; Gm[(t, gid)] = gv; Pv[(t, gid)] = pw; Sv[(t, gid)] = sr
            V["alpha"][(n, d, skey, t, gid)] = av
            V["gamma"][(n, d, skey, t, gid)] = gv
            V["p"][(n, d, skey, t, gid)] = pw
            V["s"][(n, d, skey, t, gid)] = sr
        for t in range(1, T + 1):
            av = a[t - 1]; pw = pv[t - 1]; sr = sv[t - 1]; gv = gm[t - 1]
            m.addConstr(pw + sr <= Pmax * av)
            if ren:
                cf = (solp[t - 1] * sm[t - 1]) if solar else wf[t - 1]
                cf = max(0.0, float(cf))
                m.addConstr(pw <= cf * Pmax * av)
            else:
                m.addConstr(pw >= Pmin * av)
            if is_cand:
                m.addConstr(av <= fracG[gid])
            prev = a[t - 2] if t > 1 else 0.0
            m.addConstr(gv >= av - prev)
        # ramping
        for t in range(2, T + 1):
            if R >= Pmin - 1e-9:
                m.addConstr(pv[t - 1] - pv[t - 2] <= R)
                m.addConstr(pv[t - 2] - pv[t - 1] <= R)
            else:
                m.addConstr(pv[t - 1] - pv[t - 2] <= R + Pmax * (1.0 - a[t - 2]))
                m.addConstr(pv[t - 2] - pv[t - 1] <= R + Pmax * (1.0 - a[t - 1]))
        # minimum up / down time (truncated at end of day)
        for t in range(1, T + 1):
            prev = a[t - 2] if t > 1 else 0.0
            start = a[t - 1] - prev
            stop = prev - a[t - 1]
            for tau in range(t, min(t + UT - 1, T) + 1):
                m.addConstr(a[tau - 1] >= start)
            for tau in range(t, min(t + DT - 1, T) + 1):
                m.addConstr(1.0 - a[tau - 1] >= stop)

    # storage
    Us, Vs, Rs = {}, {}, {}
    for k in p.stors:
        kid = k["id"]
        eff = float(k.get("efficiency", p.gp.get("storage_efficiency", 1.0)) or 1.0)
        capexpr = capK[kid]
        u, v, r = [], [], []
        for t in range(1, T + 1):
            uv = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
            vv = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
            rv = m.addVar(lb=0.0, ub=(0.0 if t == 1 else GRB.INFINITY),
                          vtype=GRB.CONTINUOUS)
            u.append(uv); v.append(vv); r.append(rv)
            Us[(t, kid)] = uv; Vs[(t, kid)] = vv; Rs[(t, kid)] = rv
            V["u"][(n, d, skey, t, kid)] = uv
            V["v"][(n, d, skey, t, kid)] = vv
            V["r"][(n, d, skey, t, kid)] = rv
            m.addConstr(uv <= capexpr)
            m.addConstr(vv <= capexpr)
            m.addConstr(rv <= capexpr)
            m.addConstr(vv <= rv)
            if t > 1:
                m.addConstr(rv == r[t - 2] + eff * u[t - 2] - v[t - 2])

    # transmission lines
    Ff, Fb = {}, {}
    for l in p.lines:
        lid = l["id"]
        capexpr = capL[lid]
        for t in range(1, T + 1):
            f1 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
            f2 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS)
            Ff[(t, lid)] = f1; Fb[(t, lid)] = f2
            V["ff"][(n, d, skey, t, lid)] = f1
            V["fb"][(n, d, skey, t, lid)] = f2
            m.addConstr(f1 <= capexpr)
            m.addConstr(f2 <= capexpr)

    # bus-level constraints
    for b in p.buses:
        gb = p.gens_at_bus.get(b, [])
        kb = p.stor_at_bus.get(b, [])
        out_l = ([(l, True) for l in p.lines_from.get(b, [])]
                 + [(l, False) for l in p.lines_to.get(b, [])])
        in_l = ([(l, True) for l in p.lines_to.get(b, [])]
                + [(l, False) for l in p.lines_from.get(b, [])])
        peak = p.loads.get(b, 0.0)
        if not gb and not kb and not out_l and not in_l and peak <= 0.0:
            continue
        for t in range(1, T + 1):
            dmd = peak * demn * lp[t - 1] * dm[t - 1]
            if spin > 0.0 and gb:
                m.addConstr(gp.quicksum(Sv[(t, g["id"])] for g in gb) >= spin * dmd)
            bal = gp.LinExpr()
            for g in gb:
                bal.add(Pv[(t, g["id"])], 1.0)
            for k in kb:
                kid = k["id"]
                bal.add(Vs[(t, kid)], 1.0)
                bal.add(Us[(t, kid)], -1.0)
            for l, fwd in out_l:
                bal.add(Ff[(t, l["id"])] if fwd else Fb[(t, l["id"])], -1.0)
            for l, fwd in in_l:
                loss = float(l.get("loss_factor", 0.0) or 0.0)
                bal.add(Ff[(t, l["id"])] if fwd else Fb[(t, l["id"])], 1.0 - loss)
            m.addConstr(bal == dmd)

    # operating cost expression (secant-linearized cost)
    op = gp.LinExpr()
    for g in p.gens:
        gid = g["id"]
        su = float(g.get("startup_cost", 0.0) or 0.0)
        ac = float(g.get("a_cost", 0.0) or 0.0)
        mg = p.marg[gid]
        for t in range(1, T + 1):
            op.addTerms([su, ac, mg], [Gm[(t, gid)], A[(t, gid)], Pv[(t, gid)]])
    return op


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def build_planning(p, dem, cmult, scen_pairs, relax, timelimit):
    m = gp.Model("planning")
    set_params(m, timelimit)
    V = {k: {} for k in VAR_NAMES}
    xG, yG, xL, yL, xK, yK = {}, {}, {}, {}, {}, {}
    for n in p.node_order:
        par = p.nodes[n].get("parent_id")
        for g in p.candG:
            gid = g["id"]
            x = m.addVar(vtype=GRB.BINARY)
            y = m.addVar(vtype=GRB.BINARY)
            xG[(n, gid)] = x; yG[(n, gid)] = y
            if par is None or (par, gid) not in yG:
                m.addConstr(y == x)
            else:
                m.addConstr(y == yG[(par, gid)] + x)
        for l in p.candL:
            lid = l["id"]
            x = m.addVar(vtype=GRB.BINARY)
            y = m.addVar(vtype=GRB.BINARY)
            xL[(n, lid)] = x; yL[(n, lid)] = y
            if par is None or (par, lid) not in yL:
                m.addConstr(y == x)
            else:
                m.addConstr(y == yL[(par, lid)] + x)
        for k in p.candK:
            kid = k["id"]
            x = m.addVar(vtype=GRB.BINARY)
            y = m.addVar(vtype=GRB.BINARY)
            xK[(n, kid)] = x; yK[(n, kid)] = y
            if par is None or (par, kid) not in yK:
                m.addConstr(y == x)
            else:
                m.addConstr(y == yK[(par, kid)] + x)

    RM = float(p.gp.get("reserve_margin_requirement", 0.0) or 0.0)
    total_peak = sum(p.loads.values())
    for n in p.node_ids:
        expr = gp.LinExpr()
        for g in p.gens:
            if g.get("existing", False):
                expr.addConstant(p.derate[g["id"]] * float(g.get("Pmax_MW", 0.0) or 0.0))
        for g in p.candG:
            expr.add(yG[(n, g["id"])],
                     p.derate[g["id"]] * float(g.get("Pmax_MW", 0.0) or 0.0))
        m.addConstr(expr >= (1.0 + RM) * total_peak * dem.get(n, 1.0))

    D = max(1, len(p.days))
    obj = gp.LinExpr()
    for n in p.node_ids:
        pr = p.prob.get(n, 0.0)
        cm = cmult.get(n, 1.0)
        inv = gp.LinExpr()
        for g in p.candG:
            inv.add(xG[(n, g["id"])], p.capG[g["id"]] * cm)
        for l in p.candL:
            inv.add(xL[(n, l["id"])], p.capL[l["id"]] * cm)
        for k in p.candK:
            inv.add(xK[(n, k["id"])], p.capK[k["id"]] * cm)
        obj.add(inv, pr)
        fracG = {g["id"]: (1.0 if g.get("existing", False) else yG[(n, g["id"])])
                 for g in p.gens}
        capL = {l["id"]: (float(l.get("flow_limit_MW", 0.0) or 0.0)
                          if l.get("existing", False)
                          else float(l.get("flow_limit_MW", 0.0) or 0.0) * yL[(n, l["id"])])
                for l in p.lines}
        capK = {k["id"]: (float(k.get("capacity_MW", 0.0) or 0.0)
                          if k.get("existing", False)
                          else float(k.get("capacity_MW", 0.0) or 0.0) * yK[(n, k["id"])])
                for k in p.stors}
        for (d, si) in scen_pairs:
            sc = p.scens[d][si]
            skey = p.sid[d][si]
            op = add_ops(m, p, dem.get(n, 1.0), n, d, skey, sc,
                         fracG, capL, capK, V, not relax)
            w = pr * (1.0 / D) * (1.0 / max(1, len(p.scens[d])))
            obj.add(op, w)
    m.setObjective(obj, GRB.MINIMIZE)
    return m, V, (xG, yG, xL, yL, xK, yK)


def build_op_model(p, dem, n, d, si, inv_vals, timelimit):
    m = gp.Model("op")
    set_params(m, timelimit)
    V = {k: {} for k in VAR_NAMES}
    yGv, yLv, yKv = inv_vals
    fracG = {g["id"]: (1.0 if g.get("existing", False)
                       else float(yGv.get((n, g["id"]), 0.0)))
             for g in p.gens}
    capL = {l["id"]: (float(l.get("flow_limit_MW", 0.0) or 0.0)
                      if l.get("existing", False)
                      else float(l.get("flow_limit_MW", 0.0) or 0.0)
                      * float(yLv.get((n, l["id"]), 0.0)))
            for l in p.lines}
    capK = {k["id"]: (float(k.get("capacity_MW", 0.0) or 0.0)
                      if k.get("existing", False)
                      else float(k.get("capacity_MW", 0.0) or 0.0)
                      * float(yKv.get((n, k["id"]), 0.0)))
            for k in p.stors}
    sc = p.scens[d][si]
    skey = p.sid[d][si]
    op = add_ops(m, p, dem.get(n, 1.0), n, d, skey, sc,
                 fracG, capL, capK, V, True)
    m.setObjective(op, GRB.MINIMIZE)
    return m, V


# --------------------------------------------------------------------------- #
# extraction / fallbacks
# --------------------------------------------------------------------------- #
def extract_vals(m, V):
    out = {}
    for name in VAR_NAMES:
        dct = V.get(name, {})
        if not dct:
            out[name] = {}
            continue
        ks = list(dct.keys())
        vs = [dct[k] for k in ks]
        try:
            xv = m.getAttr("X", vs)
        except Exception:
            xv = [0.0] * len(ks)
        out[name] = dict(zip(ks, xv))
    return out


def extract_invest(p, m, invvars):
    xG, yG, xL, yL, xK, yK = invvars
    invest = {}
    yGv, yLv, yKv = {}, {}, {}
    for n in p.node_ids:
        gd, ld, kd = {}, {}, {}
        for g in p.candG:
            gid = g["id"]
            gd[gid] = 1 if xG[(n, gid)].X > 0.5 else 0
            yGv[(n, gid)] = 1.0 if yG[(n, gid)].X > 0.5 else 0.0
        for l in p.candL:
            lid = l["id"]
            ld[lid] = 1 if xL[(n, lid)].X > 0.5 else 0
            yLv[(n, lid)] = 1.0 if yL[(n, lid)].X > 0.5 else 0.0
        for k in p.candK:
            kid = k["id"]
            kd[kid] = 1 if xK[(n, kid)].X > 0.5 else 0
            yKv[(n, kid)] = 1.0 if yK[(n, kid)].X > 0.5 else 0.0
        invest[str(n)] = {"generators": gd, "lines": ld, "storage": kd}
    return invest, (yGv, yLv, yKv)


def heuristic_invest(p, dem, hint=None):
    hG = (hint or {}).get("G", {})
    hL = (hint or {}).get("L", {})
    hK = (hint or {}).get("K", {})
    if hint is None:
        buildG = {g["id"] for g in p.candG}
        buildL = {l["id"] for l in p.candL}
        buildK = {k["id"] for k in p.candK}
    else:
        buildG = {gid for (n, gid), v in hG.items() if v > 0.5}
        buildL = {lid for (n, lid), v in hL.items() if v > 0.5}
        buildK = {kid for (n, kid), v in hK.items() if v > 0.5}
    # top-up generation so the reserve margin holds in every node
    RM = float(p.gp.get("reserve_margin_requirement", 0.0) or 0.0)
    total_peak = sum(p.loads.values())
    need = max(((1.0 + RM) * total_peak * dem.get(n, 1.0))
               for n in p.node_ids) if p.node_ids else 0.0
    der = {g["id"]: p.derate[g["id"]] * float(g.get("Pmax_MW", 0.0) or 0.0)
           for g in p.candG}
    cur = sum(p.derate[g["id"]] * float(g.get("Pmax_MW", 0.0) or 0.0)
              for g in p.gens if g.get("existing", False))
    cur += sum(der[gid] for gid in buildG)
    rest = [g for g in p.candG if g["id"] not in buildG]
    rest.sort(key=lambda g: p.capG[g["id"]] / max(1e-6, der[g["id"]]))
    for g in rest:
        if cur >= need - 1e-9:
            break
        buildG.add(g["id"])
        cur += der[g["id"]]
    invest = {}
    yGv, yLv, yKv = {}, {}, {}
    for n in p.node_ids:
        gd, ld, kd = {}, {}, {}
        for g in p.candG:
            gid = g["id"]
            b = 1 if gid in buildG else 0
            gd[gid] = b if n == p.root else 0
            yGv[(n, gid)] = float(b)
        for l in p.candL:
            lid = l["id"]
            b = 1 if lid in buildL else 0
            ld[lid] = b if n == p.root else 0
            yLv[(n, lid)] = float(b)
        for k in p.candK:
            kid = k["id"]
            b = 1 if kid in buildK else 0
            kd[kid] = b if n == p.root else 0
            yKv[(n, kid)] = float(b)
        invest[str(n)] = {"generators": gd, "lines": ld, "storage": kd}
    return invest, (yGv, yLv, yKv)


def default_fill(p, n, d, si, inv_vals, sol):
    yGv, _, _ = inv_vals
    skey = p.sid[d][si]
    for g in p.gens:
        gid = g["id"]
        Pmin = float(g.get("Pmin_MW", 0.0) or 0.0)
        avail = 1.0 if g.get("existing", False) else float(yGv.get((n, gid), 0.0))
        ren = p.is_ren.get(gid, False)
        for t in range(1, p.T + 1):
            key = (n, d, skey, t, gid)
            sol["alpha"][key] = avail
            sol["gamma"][key] = avail if t == 1 else 0.0
            sol["p"][key] = (Pmin * avail) if not ren else 0.0
            sol["s"][key] = 0.0
    for l in p.lines:
        lid = l["id"]
        for t in range(1, p.T + 1):
            sol["ff"][(n, d, skey, t, lid)] = 0.0
            sol["fb"][(n, d, skey, t, lid)] = 0.0
    for k in p.stors:
        kid = k["id"]
        for t in range(1, p.T + 1):
            sol["u"][(n, d, skey, t, kid)] = 0.0
            sol["v"][(n, d, skey, t, kid)] = 0.0
            sol["r"][(n, d, skey, t, kid)] = 0.0


# --------------------------------------------------------------------------- #
# objective / output
# --------------------------------------------------------------------------- #
def compute_obj(p, dem, cmult, sol, invest):
    D = max(1, len(p.days))
    A = sol["alpha"]; Gm = sol["gamma"]; Pv = sol["p"]
    total = 0.0
    for n in p.node_ids:
        cm = cmult.get(n, 1.0)
        invd = invest.get(str(n), {}) or {}
        gd = invd.get("generators", {}) or {}
        ld = invd.get("lines", {}) or {}
        kd = invd.get("storage", {}) or {}
        inv = 0.0
        for g in p.candG:
            inv += (1 if gd.get(g["id"], 0) else 0) * p.capG[g["id"]] * cm
        for l in p.candL:
            inv += (1 if ld.get(l["id"], 0) else 0) * p.capL[l["id"]] * cm
        for k in p.candK:
            inv += (1 if kd.get(k["id"], 0) else 0) * p.capK[k["id"]] * cm
        opn = 0.0
        for d in p.days:
            did = d["id"]
            Sd = max(1, len(p.scens[did]))
            w = 1.0 / (D * Sd)
            for si in range(Sd):
                skey = p.sid[did][si]
                c = 0.0
                for g in p.gens:
                    gid = g["id"]
                    su = float(g.get("startup_cost", 0.0) or 0.0)
                    ac = float(g.get("a_cost", 0.0) or 0.0)
                    mg = p.marg[gid]
                    for t in range(1, p.T + 1):
                        key = (n, did, skey, t, gid)
                        c += (su * Gm.get(key, 0.0)
                              + ac * A.get(key, 0.0)
                              + mg * Pv.get(key, 0.0))
                opn += w * c
        total += p.prob.get(n, 0.0) * (inv + opn)
    return total


def to_output(p, sol, invest, obj):
    def conv(dct, rnd=False):
        out = {}
        for k, v in dct.items():
            key = "_".join(str(x) for x in k)
            if rnd:
                out[key] = 1.0 if (v or 0.0) >= 0.5 else 0.0
            else:
                v = float(v or 0.0)
                if -1e-9 < v < 0.0:
                    v = 0.0
                out[key] = v
        return out

    return {
        "objective_value": float(obj),
        "alpha": conv(sol["alpha"], True),
        "gamma": conv(sol["gamma"], True),
        "p_gen": conv(sol["p"]),
        "s_res": conv(sol["s"]),
        "f_fwd": conv(sol["ff"]),
        "f_bwd": conv(sol["fb"]),
        "u_stor": conv(sol["u"]),
        "v_stor": conv(sol["v"]),
        "r_stor": conv(sol["r"]),
        "investments": invest,
    }


# --------------------------------------------------------------------------- #
# incumbent logging callback
# --------------------------------------------------------------------------- #
def make_callback(p, dem, cmult, V, invvars, logger, state):
    names = list(VAR_NAMES)
    keys = {nm: list(V[nm].keys()) for nm in names}
    varlists = {nm: [V[nm][k] for k in keys[nm]] for nm in names}
    xG, yG, xL, yL, xK, yK = invvars
    xGk = list(xG.keys()); xGv = [xG[k] for k in xGk]
    xLk = list(xL.keys()); xLv = [xL[k] for k in xLk]
    xKk = list(xK.keys()); xKv = [xK[k] for k in xKk]

    def cb(m, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            objbst = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
        except Exception:
            return
        if objbst is None or objbst != objbst:
            return
        if objbst >= state.get("best", float("inf")) - 1e-6:
            return
        state["best"] = objbst
        if not state.get("full", False):
            try:
                logger.log(float(objbst))
            except Exception:
                pass
            return
        try:
            sol = {}
            for nm in names:
                if keys[nm]:
                    vals = m.cbGetSolution(varlists[nm])
                    sol[nm] = dict(zip(keys[nm], vals))
                else:
                    sol[nm] = {}
            xgv = m.cbGetSolution(xGv) if xGv else []
            xlv = m.cbGetSolution(xLv) if xLv else []
            xkv = m.cbGetSolution(xKv) if xKv else []
            gm_ = dict(zip(xGk, xgv))
            lm_ = dict(zip(xLk, xlv))
            km_ = dict(zip(xKk, xkv))
            invest = {}
            for n in p.node_ids:
                gd = {g["id"]: (1 if gm_.get((n, g["id"]), 0.0) > 0.5 else 0)
                      for g in p.candG}
                ld = {l["id"]: (1 if lm_.get((n, l["id"]), 0.0) > 0.5 else 0)
                      for l in p.candL}
                kd = {k["id"]: (1 if km_.get((n, k["id"]), 0.0) > 0.5 else 0)
                      for k in p.candK}
                invest[str(n)] = {"generators": gd, "lines": ld, "storage": kd}
            obj = compute_obj(p, dem, cmult, sol, invest)
            out = to_output(p, sol, invest, obj)
            logger.log_solution(float(obj), out)
        except Exception:
            try:
                logger.log(float(objbst))
            except Exception:
                pass

    return cb


# --------------------------------------------------------------------------- #
# main solution driver
# --------------------------------------------------------------------------- #
def solve_all(p, dem, cmult, deadline, logger, time_limit):
    sol = new_sol()
    invest, inv_vals = heuristic_invest(p, dem, None)
    scen_all = [(d["id"], si) for d in p.days
                for si in range(len(p.scens[d["id"]]))]
    if not p.node_ids or not scen_all:
        return sol, invest, 0.0
    n_op = len(p.node_ids) * len(scen_all)
    nalpha = n_op * p.T * max(1, len(p.gens))
    FULL = int(max(4000, min(40000, time_limit * 130)))
    reserve_time = 6.0

    def remaining():
        return deadline - time.time() - reserve_time

    solved = set()

    # ---- Tier 1: full deterministic-equivalent MIP ----
    if nalpha <= FULL and remaining() > 5.0:
        try:
            m, V, invvars = build_planning(p, dem, cmult, scen_all,
                                           relax=False, timelimit=remaining())
            if logger is not None:
                total_vars = sum(len(V[nm]) for nm in VAR_NAMES)
                state = {"best": float("inf"), "full": total_vars <= 250000}
                m.optimize(make_callback(p, dem, cmult, V, invvars, logger, state))
            else:
                m.optimize()
            if m.SolCount > 0:
                vals = extract_vals(m, V)
                for nm in VAR_NAMES:
                    sol[nm].update(vals[nm])
                invest, inv_vals = extract_invest(p, m, invvars)
                solved = set(scen_all)
        except Exception:
            pass

    # ---- Tier 2/3: reduced planning + per-scenario unit commitment ----
    if not solved:
        scen1 = [(d["id"], 0) for d in p.days]
        nalpha2 = len(p.node_ids) * len(p.days) * p.T * max(1, len(p.gens))
        relax = nalpha2 > FULL
        if remaining() > 5.0:
            try:
                m, V, invvars = build_planning(
                    p, dem, cmult, scen1, relax=relax,
                    timelimit=max(5.0, remaining() * 0.5))
                m.optimize()
                if m.SolCount > 0:
                    if not relax:
                        invest, inv_vals = extract_invest(p, m, invvars)
                        vals = extract_vals(m, V)
                        for nm in VAR_NAMES:
                            sol[nm].update(vals[nm])
                        solved = set(scen1)
                    else:
                        _, yG, _, yL, _, yK = invvars
                        hint = {"G": {}, "L": {}, "K": {}}
                        for kkey, var in yG.items():
                            try:
                                hint["G"][kkey] = var.X
                            except Exception:
                                pass
                        for kkey, var in yL.items():
                            try:
                                hint["L"][kkey] = var.X
                            except Exception:
                                pass
                        for kkey, var in yK.items():
                            try:
                                hint["K"][kkey] = var.X
                            except Exception:
                                pass
                        invest, inv_vals = heuristic_invest(p, dem, hint)
            except Exception:
                pass

        todo_pairs = [(d, si) for (d, si) in scen_all if (d, si) not in solved]
        subs = [(n, d, si) for n in p.node_ids for (d, si) in todo_pairs]
        subs.sort(key=lambda x: -p.prob.get(x[0], 0.0))
        i = 0
        while i < len(subs):
            rem = remaining()
            left = len(subs) - i
            if rem <= 1.5:
                for (n, d, si) in subs[i:]:
                    default_fill(p, n, d, si, inv_vals, sol)
                break
            tl = max(1.5, min(25.0, rem / max(1, left)))
            n, d, si = subs[i]
            try:
                mm, VV = build_op_model(p, dem, n, d, si, inv_vals, tl)
                mm.optimize()
                if mm.SolCount > 0:
                    vals = extract_vals(mm, VV)
                    for nm in VAR_NAMES:
                        sol[nm].update(vals[nm])
                else:
                    default_fill(p, n, d, si, inv_vals, sol)
            except Exception:
                default_fill(p, n, d, si, inv_vals, sol)
            i += 1

    obj = compute_obj(p, dem, cmult, sol, invest)
    return sol, invest, obj


def fallback_solution(p, dem, cmult):
    invest, inv_vals = heuristic_invest(p, dem, None)
    sol = new_sol()
    for n in p.node_ids:
        for d in p.days:
            for si in range(len(p.scens[d["id"]])):
                default_fill(p, n, d["id"], si, inv_vals, sol)
    obj = compute_obj(p, dem, cmult, sol, invest)
    return sol, invest, obj


def main():
    args = parse_args()
    t0 = time.time()
    time_limit = max(1, int(args.time_limit))
    deadline = t0 + time_limit
    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    p = None
    try:
        inst = load_instance(args.instance_path)
        p = prepare(inst)
        dem, cmult = node_multipliers(inst)
        sol, invest, obj = solve_all(p, dem, cmult, deadline, logger, time_limit)
    except Exception:
        try:
            inst = load_instance(args.instance_path)
            p = prepare(inst)
            dem, cmult = node_multipliers(inst)
            sol, invest, obj = fallback_solution(p, dem, cmult)
        except Exception:
            p = None
            sol, invest, obj = new_sol(), {}, 0.0

    if p is not None:
        out = to_output(p, sol, invest, obj)
    else:
        out = {"objective_value": 0.0, "alpha": {}, "gamma": {}, "p_gen": {},
               "s_res": {}, "f_fwd": {}, "f_bwd": {}, "u_stor": {},
               "v_stor": {}, "r_stor": {}, "investments": {}}

    if logger is not None:
        try:
            logger.log_solution(float(obj), out)
        except Exception:
            try:
                logger.log(float(obj))
            except Exception:
                pass

    spath = args.solution_path
    dname = os.path.dirname(os.path.abspath(spath))
    if dname:
        os.makedirs(dname, exist_ok=True)
    with open(spath, "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()