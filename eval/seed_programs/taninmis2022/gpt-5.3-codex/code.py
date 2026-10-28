import argparse
import json
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def pad(arr, n, fill):
    arr = list(arr) if arr is not None else []
    if len(arr) >= n:
        return arr[:n]
    if not arr:
        return [fill] * n
    return arr + [arr[-1]] * (n - len(arr))


def gen_class(gen_type: str) -> str:
    t = (gen_type or "").lower()
    if "wind" in t:
        return "wind"
    if "solar" in t:
        return "solar"
    return "fossil"


def compute_tree_multipliers(scenario_tree, scen_params):
    children = defaultdict(list)
    parent = {}
    roots = []
    for n in scenario_tree:
        nid = n["id"]
        pid = n.get("parent_id")
        parent[nid] = pid
        if pid is None:
            roots.append(nid)
        else:
            children[pid].append(nid)

    # Prefer explicit child ordering if provided
    for n in scenario_tree:
        if n.get("children"):
            children[n["id"]] = list(n["children"])

    low_g = float(scen_params.get("demand_growth_low_percent", 0.0))
    high_g = float(scen_params.get("demand_growth_high_percent", 0.0))
    cv = float(scen_params.get("cost_variation_percent", 0.0))

    demand_mult = {}
    cost_mult = {}

    def dfs(nid, dmul, cmul):
        demand_mult[nid] = dmul
        cost_mult[nid] = cmul
        ch = children.get(nid, [])
        if not ch:
            return
        k = len(ch)
        for i, cid in enumerate(ch):
            frac = 0.5 if k == 1 else i / (k - 1)
            g = low_g + frac * (high_g - low_g)
            cchg = -cv + frac * (2 * cv)
            dfs(cid, dmul * (1 + g), cmul * (1 + cchg))

    if roots:
        for r in roots:
            dfs(r, 1.0, 1.0)

    # Fallback for isolated nodes
    for n in scenario_tree:
        nid = n["id"]
        demand_mult.setdefault(nid, 1.0)
        cost_mult.setdefault(nid, 1.0)

    return demand_mult, cost_mult


def build_and_solve(instance, time_limit, logger=None):
    system = instance["system"]
    scen_params = instance["scenario_parameters"]
    scenario_tree = instance["scenario_tree"]
    rep_days = instance["representative_days"]
    lower_scen = instance.get("lower_level_scenarios", {})
    gpms = instance.get("global_parameters", {})

    H = int(scen_params["hours_per_day"])

    buses = list(system["buses"])
    bus_set = set(buses)

    gens = system["generators"]
    lines = system["transmission_lines"]
    stor = system["storage"]

    gen_ids = [g["id"] for g in gens]
    line_ids = [l["id"] for l in lines]
    stor_ids = [s["id"] for s in stor]

    G = {g["id"]: g for g in gens}
    L = {l["id"]: l for l in lines}
    S = {s["id"]: s for s in stor}

    cand_g = [g["id"] for g in gens if not g.get("existing", False)]
    cand_l = [l["id"] for l in lines if not l.get("existing", False)]
    cand_s = [s["id"] for s in stor if not s.get("existing", False)]

    g_by_bus = defaultdict(list)
    for g in gens:
        g_by_bus[g["bus"]].append(g["id"])

    s_by_bus = defaultdict(list)
    for s in stor:
        s_by_bus[s["bus"]].append(s["id"])

    # Network incidence
    outgoing = defaultdict(list)  # (bus -> [(line_id, "fwd"/"bwd")])
    incoming = defaultdict(list)
    for l in lines:
        lid = l["id"]
        i = l["from_bus"]
        j = l["to_bus"]
        # i -> j is fwd
        outgoing[i].append((lid, "fwd"))
        incoming[j].append((lid, "fwd"))
        # j -> i is bwd
        outgoing[j].append((lid, "bwd"))
        incoming[i].append((lid, "bwd"))

    # Loads
    peak = {b: 0.0 for b in buses}
    for ld in system.get("loads", []):
        b = ld["bus"]
        if b in bus_set:
            peak[b] = float(ld.get("annual_peak_load_MW", 0.0))

    reserve_margin_req = float(gpms.get("reserve_margin_requirement", 0.0))
    spin_pct = float(gpms.get("spinning_reserve_percent", 0.0))
    derating = gpms.get("derating_factors", {})
    der_fossil = float(derating.get("fossil", 1.0))
    der_wind = float(derating.get("wind", 0.0))
    der_solar = float(derating.get("solar", 0.0))

    # Scenario tree
    nodes = [n["id"] for n in scenario_tree]
    ndata = {n["id"]: n for n in scenario_tree}
    parent = {n["id"]: n.get("parent_id") for n in scenario_tree}
    prob = {n["id"]: float(n.get("probability", 0.0)) for n in scenario_tree}
    nodes_sorted = sorted(nodes, key=lambda x: (ndata[x].get("stage", 0), x))

    demand_mult, cost_mult = compute_tree_multipliers(scenario_tree, scen_params)

    # Representative days + lower-level scenarios normalized
    day_ids = [d["id"] for d in rep_days]
    day_info = {}
    for d in rep_days:
        did = d["id"]
        day_info[did] = {
            "load_profile": pad(d.get("load_profile", []), H, 1.0),
            "solar_profile": pad(d.get("solar_profile", []), H, 0.0),
        }

    scen_by_day = {}
    for did in day_ids:
        arr = lower_scen.get(str(did), lower_scen.get(did, []))
        if not arr:
            arr = [{
                "scenario_id": 0,
                "demand_multipliers": [1.0] * H,
                "solar_multipliers": [1.0] * H,
                "wind_power_factors": [0.0] * H
            }]
        out = []
        for idx, sc in enumerate(arr):
            out.append({
                "scenario_id": sc.get("scenario_id", idx),
                "demand_multipliers": pad(sc.get("demand_multipliers", []), H, 1.0),
                "solar_multipliers": pad(sc.get("solar_multipliers", []), H, 1.0),
                "wind_power_factors": pad(sc.get("wind_power_factors", []), H, 0.0),
            })
        scen_by_day[did] = out

    m = gp.Model("gtep_uc")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1, int(time_limit))
    m.Params.OutputFlag = 0

    # Investment variables
    y_g = {(n, g): m.addVar(vtype=GRB.BINARY) for n in nodes_sorted for g in cand_g}
    y_l = {(n, l): m.addVar(vtype=GRB.BINARY) for n in nodes_sorted for l in cand_l}
    y_s = {(n, s): m.addVar(vtype=GRB.BINARY) for n in nodes_sorted for s in cand_s}

    cap_g = {(n, g): m.addVar(lb=0.0, ub=float(G[g]["Pmax_MW"])) for n in nodes_sorted for g in cand_g}
    cap_l = {(n, l): m.addVar(lb=0.0, ub=float(L[l]["flow_limit_MW"])) for n in nodes_sorted for l in cand_l}
    cap_s = {(n, s): m.addVar(lb=0.0, ub=float(S[s]["capacity_MW"])) for n in nodes_sorted for s in cand_s}

    # Once-and-for-all along each scenario path
    for n in nodes_sorted:
        pid = parent[n]
        for g in cand_g:
            rated = float(G[g]["Pmax_MW"])
            if pid is None:
                m.addConstr(cap_g[(n, g)] == rated * y_g[(n, g)])
            else:
                m.addConstr(cap_g[(n, g)] == cap_g[(pid, g)] + rated * y_g[(n, g)])
            m.addConstr(cap_g[(n, g)] <= rated)
        for l in cand_l:
            rated = float(L[l]["flow_limit_MW"])
            if pid is None:
                m.addConstr(cap_l[(n, l)] == rated * y_l[(n, l)])
            else:
                m.addConstr(cap_l[(n, l)] == cap_l[(pid, l)] + rated * y_l[(n, l)])
            m.addConstr(cap_l[(n, l)] <= rated)
        for s in cand_s:
            rated = float(S[s]["capacity_MW"])
            if pid is None:
                m.addConstr(cap_s[(n, s)] == rated * y_s[(n, s)])
            else:
                m.addConstr(cap_s[(n, s)] == cap_s[(pid, s)] + rated * y_s[(n, s)])
            m.addConstr(cap_s[(n, s)] <= rated)

    # Operational variables
    alpha, gamma, p_gen, s_res = {}, {}, {}, {}
    f_fwd, f_bwd = {}, {}
    u_stor, v_stor, r_stor = {}, {}, {}
    load_shed, spill, reserve_slack = {}, {}, {}
    rm_slack = {n: m.addVar(lb=0.0) for n in nodes_sorted}

    for n in nodes_sorted:
        for d in day_ids:
            for sc in scen_by_day[d]:
                sid = sc["scenario_id"]
                for h in range(H):
                    for g in gen_ids:
                        alpha[(n, d, sid, h, g)] = m.addVar(lb=0.0, ub=1.0)
                        gamma[(n, d, sid, h, g)] = m.addVar(lb=0.0, ub=1.0)
                        p_gen[(n, d, sid, h, g)] = m.addVar(lb=0.0)
                        s_res[(n, d, sid, h, g)] = m.addVar(lb=0.0)
                    for l in line_ids:
                        f_fwd[(n, d, sid, h, l)] = m.addVar(lb=0.0)
                        f_bwd[(n, d, sid, h, l)] = m.addVar(lb=0.0)
                    for s in stor_ids:
                        u_stor[(n, d, sid, h, s)] = m.addVar(lb=0.0)
                        v_stor[(n, d, sid, h, s)] = m.addVar(lb=0.0)
                        r_stor[(n, d, sid, h, s)] = m.addVar(lb=0.0)
                    for b in buses:
                        load_shed[(n, d, sid, h, b)] = m.addVar(lb=0.0)
                        spill[(n, d, sid, h, b)] = m.addVar(lb=0.0)
                        reserve_slack[(n, d, sid, h, b)] = m.addVar(lb=0.0)

    # Constraints
    for n in nodes_sorted:
        dmul = demand_mult[n]

        # Reserve margin (global/system-wide)
        peak_total = sum(peak[b] for b in buses) * dmul
        cap_credit = gp.LinExpr()
        for g in gen_ids:
            gt = gen_class(G[g].get("type", ""))
            der = der_wind if gt == "wind" else der_solar if gt == "solar" else der_fossil
            if G[g].get("existing", False):
                cap_credit += der * float(G[g]["Pmax_MW"])
            else:
                cap_credit += der * cap_g[(n, g)]
        m.addConstr(cap_credit + rm_slack[n] >= (1.0 + reserve_margin_req) * peak_total)

        for d in day_ids:
            lp = day_info[d]["load_profile"]
            sp = day_info[d]["solar_profile"]

            for sc in scen_by_day[d]:
                sid = sc["scenario_id"]
                dm = sc["demand_multipliers"]
                sm = sc["solar_multipliers"]
                wf = sc["wind_power_factors"]

                # Generator constraints
                for g in gen_ids:
                    gd = G[g]
                    pmin = float(gd.get("Pmin_MW", 0.0))
                    pmax = float(gd.get("Pmax_MW", 0.0))
                    ramp = float(gd.get("ramp_MW_per_h", 1e9))
                    min_on = int(gd.get("min_on_h", 0))
                    min_off = int(gd.get("min_off_h", 0))
                    gtype = gen_class(gd.get("type", ""))

                    for h in range(H):
                        a = alpha[(n, d, sid, h, g)]
                        gu = gamma[(n, d, sid, h, g)]
                        p = p_gen[(n, d, sid, h, g)]
                        sr = s_res[(n, d, sid, h, g)]

                        # startup
                        if h == 0:
                            m.addConstr(gu >= a)
                        else:
                            m.addConstr(gu >= a - alpha[(n, d, sid, h - 1, g)])

                        # commitment and capacity
                        if g in cand_g:
                            rated = max(pmax, 1e-6)
                            m.addConstr(a <= cap_g[(n, g)] / rated)
                            m.addConstr(p + sr <= cap_g[(n, g)])
                        else:
                            m.addConstr(p + sr <= pmax * a)

                        m.addConstr(p >= pmin * a)

                        # Renewable availability
                        if gtype == "solar":
                            cf = max(0.0, sp[h] * sm[h])
                            if g in cand_g:
                                m.addConstr(p <= cap_g[(n, g)] * cf)
                            else:
                                m.addConstr(p <= pmax * cf)
                        elif gtype == "wind":
                            cf = max(0.0, wf[h])
                            if g in cand_g:
                                m.addConstr(p <= cap_g[(n, g)] * cf)
                            else:
                                m.addConstr(p <= pmax * cf)

                        # ramp
                        if h > 0:
                            pp = p_gen[(n, d, sid, h - 1, g)]
                            m.addConstr(p - pp <= ramp)
                            m.addConstr(pp - p <= ramp)

                    # Minimum up/down with end-of-day truncation
                    for h in range(H):
                        a_t = alpha[(n, d, sid, h, g)]
                        a_prev = 0.0 if h == 0 else alpha[(n, d, sid, h - 1, g)]

                        if min_on > 0:
                            end = min(H, h + min_on)
                            Lh = end - h
                            m.addConstr(
                                gp.quicksum(alpha[(n, d, sid, tau, g)] for tau in range(h, end))
                                >= Lh * (a_t - a_prev)
                            )
                        if min_off > 0:
                            end = min(H, h + min_off)
                            Lh = end - h
                            m.addConstr(
                                gp.quicksum(1.0 - alpha[(n, d, sid, tau, g)] for tau in range(h, end))
                                >= Lh * (a_prev - a_t)
                            )

                # Storage constraints
                for s in stor_ids:
                    sd = S[s]
                    eff = float(sd.get("efficiency", gpms.get("storage_efficiency", 1.0)))
                    eff = min(max(eff, 0.0), 1.0)
                    scmax = float(sd.get("capacity_MW", 0.0))

                    for h in range(H):
                        u = u_stor[(n, d, sid, h, s)]
                        v = v_stor[(n, d, sid, h, s)]
                        r = r_stor[(n, d, sid, h, s)]

                        if s in cand_s:
                            m.addConstr(u <= cap_s[(n, s)])
                            m.addConstr(v <= cap_s[(n, s)])
                            m.addConstr(r <= cap_s[(n, s)])
                        else:
                            m.addConstr(u <= scmax)
                            m.addConstr(v <= scmax)
                            m.addConstr(r <= scmax)

                        if h == 0:
                            m.addConstr(r == 0.0)
                        else:
                            rp = r_stor[(n, d, sid, h - 1, s)]
                            up = u_stor[(n, d, sid, h - 1, s)]
                            vp = v_stor[(n, d, sid, h - 1, s)]
                            m.addConstr(r == rp + eff * up - vp)

                        m.addConstr(v <= r)

                # Line capacity constraints (fixed bug: no generator expression in addConstr)
                for l in line_ids:
                    lim = float(L[l].get("flow_limit_MW", 0.0))
                    for h in range(H):
                        if l in cand_l:
                            m.addConstr(f_fwd[(n, d, sid, h, l)] <= cap_l[(n, l)])
                            m.addConstr(f_bwd[(n, d, sid, h, l)] <= cap_l[(n, l)])
                        else:
                            m.addConstr(f_fwd[(n, d, sid, h, l)] <= lim)
                            m.addConstr(f_bwd[(n, d, sid, h, l)] <= lim)

                # Reserve + nodal balance
                for h in range(H):
                    for b in buses:
                        demand = peak[b] * dmul * lp[h] * dm[h]

                        # Spinning reserve at bus
                        m.addConstr(
                            gp.quicksum(s_res[(n, d, sid, h, g)] for g in g_by_bus[b])
                            + reserve_slack[(n, d, sid, h, b)]
                            >= spin_pct * demand
                        )

                        sent = gp.LinExpr()
                        for lid, direc in outgoing[b]:
                            sent += f_fwd[(n, d, sid, h, lid)] if direc == "fwd" else f_bwd[(n, d, sid, h, lid)]

                        recv = gp.LinExpr()
                        for lid, direc in incoming[b]:
                            loss = float(L[lid].get("loss_factor", 0.0))
                            loss = min(max(loss, 0.0), 1.0)
                            if direc == "fwd":
                                recv += (1.0 - loss) * f_fwd[(n, d, sid, h, lid)]
                            else:
                                recv += (1.0 - loss) * f_bwd[(n, d, sid, h, lid)]

                        gen_supply = gp.quicksum(p_gen[(n, d, sid, h, g)] for g in g_by_bus[b])
                        stor_net = gp.quicksum(v_stor[(n, d, sid, h, s)] - u_stor[(n, d, sid, h, s)] for s in s_by_bus[b])

                        m.addConstr(
                            demand + sent - recv + spill[(n, d, sid, h, b)]
                            == gen_supply + stor_net + load_shed[(n, d, sid, h, b)]
                        )

    # Objective
    obj = gp.LinExpr()
    BIG = 1e6
    MED = 1e3

    day_w = 1.0 / max(1, len(day_ids))

    for n in nodes_sorted:
        pn = prob.get(n, 0.0)
        cm = cost_mult[n]

        # Investment costs
        for g in cand_g:
            obj += pn * cm * 1000.0 * float(G[g]["Pmax_MW"]) * float(G[g].get("capital_cost_per_kW", 0.0)) * y_g[(n, g)]
        for l in cand_l:
            obj += pn * cm * 1000.0 * float(L[l]["flow_limit_MW"]) * float(L[l].get("capital_cost_per_kW", 0.0)) * y_l[(n, l)]
        for s in cand_s:
            obj += pn * cm * 1000.0 * float(S[s]["capacity_MW"]) * float(S[s].get("capital_cost_per_kW", 0.0)) * y_s[(n, s)]

        obj += pn * BIG * rm_slack[n]

        # Operating costs
        for d in day_ids:
            scs = scen_by_day[d]
            sc_w = 1.0 / max(1, len(scs))
            w = pn * day_w * sc_w

            for sc in scs:
                sid = sc["scenario_id"]
                for h in range(H):
                    for g in gen_ids:
                        gd = G[g]
                        a = float(gd.get("a_cost", 0.0))
                        b = float(gd.get("b_cost", 0.0))
                        c = float(gd.get("c_cost", 0.0))
                        pmax = float(gd.get("Pmax_MW", 0.0))
                        su = float(gd.get("startup_cost", 0.0))
                        obj += w * su * gamma[(n, d, sid, h, g)]
                        obj += w * a * alpha[(n, d, sid, h, g)]
                        obj += w * (b + c * pmax) * p_gen[(n, d, sid, h, g)]

                    for b in buses:
                        obj += w * BIG * load_shed[(n, d, sid, h, b)]
                        obj += w * MED * spill[(n, d, sid, h, b)]
                        obj += w * BIG * reserve_slack[(n, d, sid, h, b)]

    m.setObjective(obj, GRB.MINIMIZE)

    best = [float("inf")]

    def cb(model, where):
        if logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if val + 1e-9 < best[0]:
                best[0] = val
                logger.log(float(val))

    m.optimize(cb if logger else None)

    if m.SolCount == 0:
        # Return schema-compliant empty-like solution
        out = {
            "objective_value": float("inf"),
            "alpha": {},
            "gamma": {},
            "p_gen": {},
            "s_res": {},
            "f_fwd": {},
            "f_bwd": {},
            "u_stor": {},
            "v_stor": {},
            "r_stor": {},
            "investments": {}
        }
        for n in nodes_sorted:
            out["investments"][str(n)] = {
                "generators": {g: 0 for g in cand_g},
                "lines": {l: 0 for l in cand_l},
                "storage": {s: 0 for s in cand_s}
            }
        return out

    def x(v):
        val = float(v.X)
        if abs(val) < 1e-9:
            return 0.0
        return val

    alpha_out, gamma_out, p_out, sr_out = {}, {}, {}, {}
    ff_out, fb_out = {}, {}
    u_out, v_out, r_out = {}, {}, {}

    for n in nodes_sorted:
        for d in day_ids:
            for sc in scen_by_day[d]:
                sid = sc["scenario_id"]
                for h in range(H):
                    for g in gen_ids:
                        key = f"{n}|{d}|{sid}|{h}|{g}"
                        alpha_out[key] = x(alpha[(n, d, sid, h, g)])
                        gamma_out[key] = x(gamma[(n, d, sid, h, g)])
                        p_out[key] = x(p_gen[(n, d, sid, h, g)])
                        sr_out[key] = x(s_res[(n, d, sid, h, g)])
                    for l in line_ids:
                        key = f"{n}|{d}|{sid}|{h}|{l}"
                        ff_out[key] = x(f_fwd[(n, d, sid, h, l)])
                        fb_out[key] = x(f_bwd[(n, d, sid, h, l)])
                    for s in stor_ids:
                        key = f"{n}|{d}|{sid}|{h}|{s}"
                        u_out[key] = x(u_stor[(n, d, sid, h, s)])
                        v_out[key] = x(v_stor[(n, d, sid, h, s)])
                        r_out[key] = x(r_stor[(n, d, sid, h, s)])

    inv = {}
    for n in nodes_sorted:
        inv[str(n)] = {
            "generators": {g: int(round(x(y_g[(n, g)]))) for g in cand_g},
            "lines": {l: int(round(x(y_l[(n, l)]))) for l in cand_l},
            "storage": {s: int(round(x(y_s[(n, s)]))) for s in cand_s},
        }

    solution = {
        "objective_value": float(m.ObjVal),
        "alpha": alpha_out,
        "gamma": gamma_out,
        "p_gen": p_out,
        "s_res": sr_out,
        "f_fwd": ff_out,
        "f_bwd": fb_out,
        "u_stor": u_out,
        "v_stor": v_out,
        "r_stor": r_out,
        "investments": inv
    }

    if logger:
        logger.log_solution(float(m.ObjVal), solution)

    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    sol = build_and_solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()