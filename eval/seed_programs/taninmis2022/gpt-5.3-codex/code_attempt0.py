import argparse
import json
import math
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def safe_array(arr, H, fill=1.0):
    arr = list(arr) if arr is not None else []
    if len(arr) >= H:
        return arr[:H]
    if len(arr) == 0:
        return [fill] * H
    return arr + [arr[-1]] * (H - len(arr))


def classify_gen_type(gtype: str):
    t = (gtype or "").lower()
    if "wind" in t:
        return "wind"
    if "solar" in t:
        return "solar"
    return "fossil"


def compute_node_multipliers(scenario_tree, scen_params):
    node_by_id = {n["id"]: n for n in scenario_tree}
    children = defaultdict(list)
    roots = []
    for n in scenario_tree:
        pid = n.get("parent_id")
        if pid is None:
            roots.append(n["id"])
        else:
            children[pid].append(n["id"])

    # Respect provided children ordering when present
    for n in scenario_tree:
        if "children" in n and n["children"] is not None and len(n["children"]) > 0:
            children[n["id"]] = list(n["children"])

    low_g = scen_params.get("demand_growth_low_percent", 0.0)
    high_g = scen_params.get("demand_growth_high_percent", 0.0)
    cv = scen_params.get("cost_variation_percent", 0.0)

    demand_mult = {}
    cost_mult = {}

    def dfs(node_id, dmul, cmul):
        demand_mult[node_id] = dmul
        cost_mult[node_id] = cmul
        ch = children.get(node_id, [])
        k = len(ch)
        if k == 0:
            return
        for r, c in enumerate(ch):
            frac = 0.5 if k == 1 else r / (k - 1)
            g = low_g + frac * (high_g - low_g)
            cchg = -cv + frac * (2.0 * cv)  # low-cost ... high-cost
            dfs(c, dmul * (1.0 + g), cmul * (1.0 + cchg))

    if not roots:
        # fallback
        for nid in node_by_id:
            demand_mult[nid] = 1.0
            cost_mult[nid] = 1.0
        return demand_mult, cost_mult

    for r in roots:
        dfs(r, 1.0, 1.0)

    # Fill unreachable (just in case malformed tree)
    for nid in node_by_id:
        demand_mult.setdefault(nid, 1.0)
        cost_mult.setdefault(nid, 1.0)

    return demand_mult, cost_mult


def build_and_solve(instance, time_limit, logger=None):
    t0 = time.time()

    system = instance["system"]
    scen_params = instance["scenario_parameters"]
    scenario_tree = instance["scenario_tree"]
    rep_days = instance["representative_days"]
    ll_scen_raw = instance.get("lower_level_scenarios", {})
    gparams = instance.get("global_parameters", {})

    H = int(scen_params["hours_per_day"])

    # Sets / maps
    buses = list(system["buses"])
    bus_set = set(buses)

    gens = system["generators"]
    lines = system["transmission_lines"]
    stors = system["storage"]

    gen_ids = [g["id"] for g in gens]
    line_ids = [l["id"] for l in lines]
    stor_ids = [s["id"] for s in stors]

    gen = {g["id"]: g for g in gens}
    line = {l["id"]: l for l in lines}
    stor = {s["id"]: s for s in stors}

    cand_g = [g["id"] for g in gens if not g.get("existing", False)]
    cand_l = [l["id"] for l in lines if not l.get("existing", False)]
    cand_s = [s["id"] for s in stors if not s.get("existing", False)]

    g_by_bus = defaultdict(list)
    for g in gens:
        g_by_bus[g["bus"]].append(g["id"])

    s_by_bus = defaultdict(list)
    for s in stors:
        s_by_bus[s["bus"]].append(s["id"])

    # For line-bus incidence with losses
    outgoing = defaultdict(list)  # (bus) -> list of (line_id, direction "fwd"/"bwd" as sent variable)
    incoming = defaultdict(list)  # (bus) -> list of (line_id, direction as received with loss)
    for l in lines:
        lid = l["id"]
        i = l["from_bus"]
        j = l["to_bus"]
        # At i: sent fwd, received bwd*(1-loss)
        outgoing[i].append((lid, "fwd"))
        incoming[i].append((lid, "bwd"))
        # At j: sent bwd, received fwd*(1-loss)
        outgoing[j].append((lid, "bwd"))
        incoming[j].append((lid, "fwd"))

    load_peak = {b: 0.0 for b in buses}
    for ld in system.get("loads", []):
        b = ld["bus"]
        if b in bus_set:
            load_peak[b] = float(ld.get("annual_peak_load_MW", 0.0))

    # Reserve and derating
    reserve_margin_req = float(gparams.get("reserve_margin_requirement", 0.0))
    spin_pct = float(gparams.get("spinning_reserve_percent", 0.0))
    der = gparams.get("derating_factors", {})
    der_fossil = float(der.get("fossil", 1.0))
    der_wind = float(der.get("wind", 0.0))
    der_solar = float(der.get("solar", 0.0))

    # Scenario tree
    nodes = [n["id"] for n in scenario_tree]
    node_data = {n["id"]: n for n in scenario_tree}
    parent = {n["id"]: n.get("parent_id") for n in scenario_tree}
    prob = {n["id"]: float(n.get("probability", 0.0)) for n in scenario_tree}

    # Topological by stage then id
    nodes_sorted = sorted(nodes, key=lambda nid: (node_data[nid].get("stage", 0), nid))

    demand_mult, cost_mult = compute_node_multipliers(scenario_tree, scen_params)

    # Representative days normalized
    day_ids = [d["id"] for d in rep_days]
    day_data = {}
    for d in rep_days:
        did = d["id"]
        day_data[did] = {
            "load_profile": safe_array(d.get("load_profile", []), H, fill=1.0),
            "solar_profile": safe_array(d.get("solar_profile", []), H, fill=0.0),
        }

    # Lower level scenarios normalized
    scen_by_day = {}
    for did in day_ids:
        arr = ll_scen_raw.get(str(did), ll_scen_raw.get(did, []))
        if not arr:
            arr = [{
                "scenario_id": 0,
                "demand_multipliers": [1.0] * H,
                "solar_multipliers": [1.0] * H,
                "wind_power_factors": [0.0] * H,
            }]
        norm = []
        for k, sc in enumerate(arr):
            sid = sc.get("scenario_id", k)
            norm.append({
                "scenario_id": sid,
                "demand_multipliers": safe_array(sc.get("demand_multipliers", []), H, fill=1.0),
                "solar_multipliers": safe_array(sc.get("solar_multipliers", []), H, fill=1.0),
                "wind_power_factors": safe_array(sc.get("wind_power_factors", []), H, fill=0.0),
            })
        scen_by_day[did] = norm

    # Model
    m = gp.Model("stochastic_gtep_uc")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1, int(time_limit))
    m.Params.OutputFlag = 0

    # Investment vars
    y_g = {(n, g): m.addVar(vtype=GRB.BINARY) for n in nodes_sorted for g in cand_g}
    y_l = {(n, l): m.addVar(vtype=GRB.BINARY) for n in nodes_sorted for l in cand_l}
    y_s = {(n, s): m.addVar(vtype=GRB.BINARY) for n in nodes_sorted for s in cand_s}

    cap_g = {(n, g): m.addVar(lb=0.0, ub=gen[g]["Pmax_MW"]) for n in nodes_sorted for g in cand_g}
    cap_l = {(n, l): m.addVar(lb=0.0, ub=line[l]["flow_limit_MW"]) for n in nodes_sorted for l in cand_l}
    cap_s = {(n, s): m.addVar(lb=0.0, ub=stor[s]["capacity_MW"]) for n in nodes_sorted for s in cand_s}

    # Recurrence once-and-for-all along scenario path
    for n in nodes_sorted:
        pid = parent[n]
        for g in cand_g:
            rated = float(gen[g]["Pmax_MW"])
            if pid is None:
                m.addConstr(cap_g[(n, g)] == rated * y_g[(n, g)])
            else:
                m.addConstr(cap_g[(n, g)] == cap_g[(pid, g)] + rated * y_g[(n, g)])
            m.addConstr(cap_g[(n, g)] <= rated)
        for l in cand_l:
            rated = float(line[l]["flow_limit_MW"])
            if pid is None:
                m.addConstr(cap_l[(n, l)] == rated * y_l[(n, l)])
            else:
                m.addConstr(cap_l[(n, l)] == cap_l[(pid, l)] + rated * y_l[(n, l)])
            m.addConstr(cap_l[(n, l)] <= rated)
        for s in cand_s:
            rated = float(stor[s]["capacity_MW"])
            if pid is None:
                m.addConstr(cap_s[(n, s)] == rated * y_s[(n, s)])
            else:
                m.addConstr(cap_s[(n, s)] == cap_s[(pid, s)] + rated * y_s[(n, s)])
            m.addConstr(cap_s[(n, s)] <= rated)

    # Operational vars
    alpha = {}
    gamma = {}
    p_gen = {}
    s_res = {}
    f_fwd = {}
    f_bwd = {}
    u_stor = {}
    v_stor = {}
    r_stor = {}

    load_shed = {}
    spill = {}
    reserve_slack = {}
    rm_slack = {n: m.addVar(lb=0.0) for n in nodes_sorted}

    # Build operation variables
    for n in nodes_sorted:
        for did in day_ids:
            for sc in scen_by_day[did]:
                sid = sc["scenario_id"]
                for h in range(H):
                    for g in gen_ids:
                        alpha[(n, did, sid, h, g)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
                        gamma[(n, did, sid, h, g)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
                        p_gen[(n, did, sid, h, g)] = m.addVar(lb=0.0)
                        s_res[(n, did, sid, h, g)] = m.addVar(lb=0.0)
                    for l in line_ids:
                        f_fwd[(n, did, sid, h, l)] = m.addVar(lb=0.0)
                        f_bwd[(n, did, sid, h, l)] = m.addVar(lb=0.0)
                    for s in stor_ids:
                        u_stor[(n, did, sid, h, s)] = m.addVar(lb=0.0)
                        v_stor[(n, did, sid, h, s)] = m.addVar(lb=0.0)
                        r_stor[(n, did, sid, h, s)] = m.addVar(lb=0.0)
                    for b in buses:
                        load_shed[(n, did, sid, h, b)] = m.addVar(lb=0.0)
                        spill[(n, did, sid, h, b)] = m.addVar(lb=0.0)
                        reserve_slack[(n, did, sid, h, b)] = m.addVar(lb=0.0)

    m.update()

    # Constraints
    for n in nodes_sorted:
        node_dmult = demand_mult[n]

        # Reserve margin (system-wide, schema has global requirement)
        peak_region = sum(load_peak[b] for b in buses) * node_dmult
        cap_expr = gp.LinExpr()
        for g in gen_ids:
            gt = classify_gen_type(gen[g].get("type", ""))
            if gt == "wind":
                dfg = der_wind
            elif gt == "solar":
                dfg = der_solar
            else:
                dfg = der_fossil

            if gen[g].get("existing", False):
                cap_expr += dfg * float(gen[g]["Pmax_MW"])
            else:
                cap_expr += dfg * cap_g[(n, g)]

        m.addConstr(cap_expr + rm_slack[n] >= (1.0 + reserve_margin_req) * peak_region)

        for did in day_ids:
            lp = day_data[did]["load_profile"]
            sp = day_data[did]["solar_profile"]

            for sc in scen_by_day[did]:
                sid = sc["scenario_id"]
                dm = sc["demand_multipliers"]
                sm = sc["solar_multipliers"]
                wf = sc["wind_power_factors"]

                # UC / dispatch / ramp / min up/down / storage dynamics
                for g in gen_ids:
                    gdat = gen[g]
                    g_bus = gdat["bus"]
                    pmin = float(gdat.get("Pmin_MW", 0.0))
                    pmax = float(gdat.get("Pmax_MW", 0.0))
                    ramp = float(gdat.get("ramp_MW_per_h", 1e6))
                    mu = int(gdat.get("min_on_h", 0))
                    md = int(gdat.get("min_off_h", 0))
                    gtype = classify_gen_type(gdat.get("type", ""))

                    for h in range(H):
                        a = alpha[(n, did, sid, h, g)]
                        gg = gamma[(n, did, sid, h, g)]
                        p = p_gen[(n, did, sid, h, g)]
                        sr = s_res[(n, did, sid, h, g)]

                        # startup logic
                        if h == 0:
                            m.addConstr(gg >= a)
                        else:
                            m.addConstr(gg >= a - alpha[(n, did, sid, h - 1, g)])

                        # candidate commitment only if built
                        if g in cand_g:
                            rated = max(1e-6, pmax)
                            avail = cap_g[(n, g)]
                            m.addConstr(a <= avail / rated)
                            m.addConstr(p + sr <= avail)
                        else:
                            m.addConstr(p + sr <= pmax * a)

                        # minimum output if committed
                        m.addConstr(p >= pmin * a)

                        # renewable availability
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
                            p_prev = p_gen[(n, did, sid, h - 1, g)]
                            m.addConstr(p - p_prev <= ramp)
                            m.addConstr(p_prev - p <= ramp)

                    # min up/down
                    for h in range(H):
                        a_t = alpha[(n, did, sid, h, g)]
                        a_prev = 0.0 if h == 0 else alpha[(n, did, sid, h - 1, g)]

                        if mu > 0:
                            end = min(H, h + mu)
                            m.addConstr(gp.quicksum(alpha[(n, did, sid, tau, g)] for tau in range(h, end))
                                        >= mu * (a_t - a_prev))

                        if md > 0:
                            end = min(H, h + md)
                            m.addConstr(gp.quicksum(1.0 - alpha[(n, did, sid, tau, g)] for tau in range(h, end))
                                        >= md * (a_prev - a_t))

                # Storage constraints
                for s in stor_ids:
                    sdat = stor[s]
                    sbus = sdat["bus"]
                    eff = float(sdat.get("efficiency", gparams.get("storage_efficiency", 1.0)))
                    eff = min(max(eff, 0.0), 1.0)
                    s_cap = float(sdat.get("capacity_MW", 0.0))

                    for h in range(H):
                        u = u_stor[(n, did, sid, h, s)]
                        v = v_stor[(n, did, sid, h, s)]
                        r = r_stor[(n, did, sid, h, s)]

                        if s in cand_s:
                            avail_s = cap_s[(n, s)]
                            m.addConstr(u <= avail_s)
                            m.addConstr(v <= avail_s)
                            m.addConstr(r <= avail_s)
                        else:
                            m.addConstr(u <= s_cap)
                            m.addConstr(v <= s_cap)
                            m.addConstr(r <= s_cap)

                        if h == 0:
                            m.addConstr(r == 0.0)
                        else:
                            r_prev = r_stor[(n, did, sid, h - 1, s)]
                            u_prev = u_stor[(n, did, sid, h - 1, s)]
                            v_prev = v_stor[(n, did, sid, h - 1, s)]
                            m.addConstr(r == r_prev + eff * u_prev - v_prev)

                        m.addConstr(v <= r)

                # Line capacities
                for l in line_ids:
                    ldat = line[l]
                    lim = float(ldat.get("flow_limit_MW", 0.0))
                    if l in cand_l:
                        m.addConstr(f_fwd[(n, did, sid, h, l)] <= cap_l[(n, l)] for h in range(H))
                        m.addConstr(f_bwd[(n, did, sid, h, l)] <= cap_l[(n, l)] for h in range(H))
                    else:
                        for h in range(H):
                            m.addConstr(f_fwd[(n, did, sid, h, l)] <= lim)
                            m.addConstr(f_bwd[(n, did, sid, h, l)] <= lim)

                # Bus-hour reserve and energy balance
                for h in range(H):
                    for b in buses:
                        demand_bh = load_peak[b] * node_dmult * lp[h] * dm[h]

                        # spinning reserve
                        m.addConstr(
                            gp.quicksum(s_res[(n, did, sid, h, g)] for g in g_by_bus[b])
                            + reserve_slack[(n, did, sid, h, b)]
                            >= spin_pct * demand_bh
                        )

                        sent = gp.LinExpr()
                        recv = gp.LinExpr()
                        for (lid, ddir) in outgoing[b]:
                            if ddir == "fwd":
                                sent += f_fwd[(n, did, sid, h, lid)]
                            else:
                                sent += f_bwd[(n, did, sid, h, lid)]
                        for (lid, ddir) in incoming[b]:
                            lf = float(line[lid].get("loss_factor", 0.0))
                            lf = min(max(lf, 0.0), 1.0)
                            if ddir == "fwd":
                                recv += (1.0 - lf) * f_fwd[(n, did, sid, h, lid)]
                            else:
                                recv += (1.0 - lf) * f_bwd[(n, did, sid, h, lid)]

                        gen_supply = gp.quicksum(p_gen[(n, did, sid, h, g)] for g in g_by_bus[b])
                        stor_net = gp.quicksum(v_stor[(n, did, sid, h, s)] - u_stor[(n, did, sid, h, s)] for s in s_by_bus[b])

                        m.addConstr(
                            demand_bh + sent - recv + spill[(n, did, sid, h, b)]
                            == gen_supply + stor_net + load_shed[(n, did, sid, h, b)]
                        )

    # Objective
    obj = gp.LinExpr()
    big_pen = 1e6
    med_pen = 1e3

    for n in nodes_sorted:
        pnode = prob.get(n, 0.0)
        cm = cost_mult[n]

        # investment
        for g in cand_g:
            rated_kw = 1000.0 * float(gen[g]["Pmax_MW"])
            capcost = float(gen[g].get("capital_cost_per_kW", 0.0))
            obj += pnode * cm * rated_kw * capcost * y_g[(n, g)]
        for l in cand_l:
            rated_kw = 1000.0 * float(line[l]["flow_limit_MW"])
            capcost = float(line[l].get("capital_cost_per_kW", 0.0))
            obj += pnode * cm * rated_kw * capcost * y_l[(n, l)]
        for s in cand_s:
            rated_kw = 1000.0 * float(stor[s]["capacity_MW"])
            capcost = float(stor[s].get("capital_cost_per_kW", 0.0))
            obj += pnode * cm * rated_kw * capcost * y_s[(n, s)]

        obj += pnode * big_pen * rm_slack[n]

        # operation
        day_w = 1.0 / max(1, len(day_ids))
        for did in day_ids:
            sc_list = scen_by_day[did]
            sc_w = 1.0 / max(1, len(sc_list))
            w = pnode * day_w * sc_w

            for sc in sc_list:
                sid = sc["scenario_id"]
                for h in range(H):
                    for g in gen_ids:
                        gd = gen[g]
                        a = float(gd.get("a_cost", 0.0))
                        b = float(gd.get("b_cost", 0.0))
                        c = float(gd.get("c_cost", 0.0))
                        pmax = float(gd.get("Pmax_MW", 0.0))
                        su = float(gd.get("startup_cost", 0.0))

                        obj += w * su * gamma[(n, did, sid, h, g)]
                        obj += w * a * alpha[(n, did, sid, h, g)]
                        obj += w * (b + c * pmax) * p_gen[(n, did, sid, h, g)]

                    for b in buses:
                        obj += w * big_pen * load_shed[(n, did, sid, h, b)]
                        obj += w * med_pen * spill[(n, did, sid, h, b)]
                        obj += w * big_pen * reserve_slack[(n, did, sid, h, b)]

    m.setObjective(obj, GRB.MINIMIZE)

    # Callback logger
    best_seen = [float("inf")]

    def cb(model, where):
        if logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if val + 1e-9 < best_seen[0]:
                best_seen[0] = val
                logger.log(float(val))

    m.optimize(cb if logger is not None else None)

    if m.SolCount == 0:
        # fallback all zeros
        objective_value = float("inf")
        sol = {
            "objective_value": objective_value,
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
            sol["investments"][str(n)] = {
                "generators": {g: 0 for g in cand_g},
                "lines": {l: 0 for l in cand_l},
                "storage": {s: 0 for s in cand_s},
            }
        return sol

    def vval(v):
        x = float(v.X)
        if abs(x) < 1e-9:
            return 0.0
        return x

    alpha_out = {}
    gamma_out = {}
    p_out = {}
    s_out = {}
    ff_out = {}
    fb_out = {}
    u_out = {}
    v_out = {}
    r_out = {}

    # complete projection
    for n in nodes_sorted:
        for did in day_ids:
            for sc in scen_by_day[did]:
                sid = sc["scenario_id"]
                for h in range(H):
                    for g in gen_ids:
                        k = f"{n}|{did}|{sid}|{h}|{g}"
                        alpha_out[k] = vval(alpha[(n, did, sid, h, g)])
                        gamma_out[k] = vval(gamma[(n, did, sid, h, g)])
                        p_out[k] = vval(p_gen[(n, did, sid, h, g)])
                        s_out[k] = vval(s_res[(n, did, sid, h, g)])
                    for l in line_ids:
                        k = f"{n}|{did}|{sid}|{h}|{l}"
                        ff_out[k] = vval(f_fwd[(n, did, sid, h, l)])
                        fb_out[k] = vval(f_bwd[(n, did, sid, h, l)])
                    for s in stor_ids:
                        k = f"{n}|{did}|{sid}|{h}|{s}"
                        u_out[k] = vval(u_stor[(n, did, sid, h, s)])
                        v_out[k] = vval(v_stor[(n, did, sid, h, s)])
                        r_out[k] = vval(r_stor[(n, did, sid, h, s)])

    inv_out = {}
    for n in nodes_sorted:
        inv_out[str(n)] = {
            "generators": {g: int(round(vval(y_g[(n, g)]))) for g in cand_g},
            "lines": {l: int(round(vval(y_l[(n, l)]))) for l in cand_l},
            "storage": {s: int(round(vval(y_s[(n, s)]))) for s in cand_s},
        }

    solution = {
        "objective_value": float(m.ObjVal),
        "alpha": alpha_out,
        "gamma": gamma_out,
        "p_gen": p_out,
        "s_res": s_out,
        "f_fwd": ff_out,
        "f_bwd": fb_out,
        "u_stor": u_out,
        "v_stor": v_out,
        "r_stor": r_out,
        "investments": inv_out,
    }

    if logger is not None:
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

    solution = build_and_solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()