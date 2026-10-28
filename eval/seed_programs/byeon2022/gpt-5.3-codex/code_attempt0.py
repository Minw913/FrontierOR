import argparse
import json
import math
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger


def get_bid_threshold(gen, bid, gas_price_min, gas_price_max):
    for k in [
        "profitability_threshold",
        "max_gas_price",
        "gas_price_threshold",
        "phi_max_bid",
        "bid_gas_price_cap",
    ]:
        if k in bid and bid[k] is not None:
            return float(bid[k])

    # Fallback heuristic if explicit bid threshold is absent in instance
    frac = gen.get("max_gas_price_fraction", None)
    if frac is not None:
        return max(gas_price_min, min(gas_price_max, float(frac) * gas_price_max))

    # Last fallback
    return gas_price_max


def build_and_solve(instance, time_limit, logger=None):
    beta = float(instance["objective_weight_beta"])
    T = int(instance["time_periods"])
    eta_p = float(instance.get("stress_parameters", {}).get("eta_p", 1.0))
    eta_g = float(instance.get("stress_parameters", {}).get("eta_g", 1.0))

    e_net = instance["electricity_network"]
    g_data = instance["generators"]
    gas_net = instance["gas_network"]

    buses = e_net["buses"]
    lines = e_net["lines"]
    gens = g_data["generators"]
    junctions = gas_net["junctions"]
    connections = gas_net["connections"]
    zones = gas_net["pricing_zones"]

    bus_ids = [b["id"] for b in buses]
    line_ids = [l["id"] for l in lines]
    gen_ids = [g["id"] for g in gens]
    junction_ids = [j["id"] for j in junctions]
    conn_ids = [c["id"] for c in connections]
    zone_ids = [z["id"] for z in zones]

    bus_by_id = {b["id"]: b for b in buses}
    line_by_id = {l["id"]: l for l in lines}
    gen_by_id = {g["id"]: g for g in gens}
    junc_by_id = {j["id"]: j for j in junctions}
    conn_by_id = {c["id"]: c for c in connections}
    zone_by_id = {z["id"]: z for z in zones}

    gas_price_min = float(gas_net["min_gas_price_mmBtu"])
    gas_price_max = float(gas_net["max_gas_price_mmBtu"])

    # Zone map: junction -> pricing zone
    junction_to_zone = {}
    for z in zones:
        zid = z["id"]
        for j in z["junctions"]:
            junction_to_zone[j] = zid

    # Incidence (electric)
    outgoing_lines_by_bus = {b: [] for b in bus_ids}
    incoming_lines_by_bus = {b: [] for b in bus_ids}
    for l in lines:
        outgoing_lines_by_bus[l["from_bus"]].append(l["id"])
        incoming_lines_by_bus[l["to_bus"]].append(l["id"])

    gens_at_bus = {b: [] for b in bus_ids}
    for g in gens:
        gens_at_bus[g["bus"]].append(g["id"])

    # Incidence (gas)
    outgoing_conn_by_j = {j: [] for j in junction_ids}
    incoming_conn_by_j = {j: [] for j in junction_ids}
    for c in connections:
        outgoing_conn_by_j[c["from_junction"]].append(c["id"])
        incoming_conn_by_j[c["to_junction"]].append(c["id"])

    gfpp_at_junction = {j: [] for j in junction_ids}
    for g in gens:
        if g["is_gfpp"] and g["gas_junction"] is not None:
            gfpp_at_junction[g["gas_junction"]].append(g["id"])

    # Bid ids per generator (preserve given order)
    bids_by_gen = {}
    bid_data = {}
    for g in gens:
        gid = g["id"]
        bids = g["bids"]
        bids_by_gen[gid] = [b["id"] for b in bids]
        for b in bids:
            bid_data[(gid, b["id"])] = b

    # Supply intervals per junction
    intervals_by_j = {}
    interval_data = {}
    for j in junctions:
        jid = j["id"]
        intervals = j.get("supply_intervals", [])
        intervals_by_j[jid] = [s["id"] for s in intervals]
        for s in intervals:
            interval_data[(jid, s["id"])] = s

    m = Model("integrated_primal_ucgna")

    # Required solver params
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = int(time_limit)

    # Decision variables
    o = {}       # on/off incl t=0
    p = {}       # generation incl t=0
    vp = {}      # startup t=1..T
    vm = {}      # shutdown t=1..T
    r = {}       # startup cost t=1..T
    phimax = {}  # max allowable gas price t=1..T

    w = {}       # bid selected
    se = {}      # bid generation

    f = {}       # line flow
    theta = {}   # voltage angles

    sg = {}
    sgs = {}
    pisq = {}
    phig = {}
    lg = {}
    qg = {}
    gamma = {}
    psi = {}

    # Aux for linearization psi * o (GFPP only)
    zpsi_on = {}

    # Generator vars
    for g in gens:
        gid = g["id"]
        init_o = int(g["initial_status"])
        init_p = float(g["initial_gen"])

        for t in range(0, T + 1):
            if t == 0:
                o[(gid, t)] = m.addVar(vtype=GRB.BINARY, lb=init_o, ub=init_o, name=f"o_{gid}_{t}")
                p[(gid, t)] = m.addVar(lb=init_p, ub=init_p, name=f"p_{gid}_{t}")
            else:
                o[(gid, t)] = m.addVar(vtype=GRB.BINARY, name=f"o_{gid}_{t}")
                p[(gid, t)] = m.addVar(lb=0.0, name=f"p_{gid}_{t}")
                vp[(gid, t)] = m.addVar(vtype=GRB.BINARY, name=f"vp_{gid}_{t}")
                vm[(gid, t)] = m.addVar(vtype=GRB.BINARY, name=f"vm_{gid}_{t}")
                r[(gid, t)] = m.addVar(lb=0.0, name=f"r_{gid}_{t}")
                phimax[(gid, t)] = m.addVar(lb=0.0, ub=gas_price_max, name=f"phimax_{gid}_{t}")

                for bid in bids_by_gen[gid]:
                    bmax = float(bid_data[(gid, bid)]["max_amount"])
                    w[(gid, bid, t)] = m.addVar(vtype=GRB.BINARY, name=f"w_{gid}_{bid}_{t}")
                    se[(gid, bid, t)] = m.addVar(lb=0.0, ub=bmax, name=f"se_{gid}_{bid}_{t}")

    # Electric network vars
    for l in lines:
        lid = l["id"]
        fmax = float(l["thermal_limit"])
        for t in range(1, T + 1):
            f[(lid, t)] = m.addVar(lb=-fmax, ub=fmax, name=f"f_{lid}_{t}")

    for b in buses:
        bid = b["id"]
        th_lb = float(b["voltage_angle_lb"])
        th_ub = float(b["voltage_angle_ub"])
        for t in range(1, T + 1):
            theta[(bid, t)] = m.addVar(lb=th_lb, ub=th_ub, name=f"theta_{bid}_{t}")

    # Gas vars
    for j in junctions:
        jid = j["id"]
        p_lb = float(j["pressure_lb_squared"])
        p_ub = float(j["pressure_ub_squared"])
        for t in range(1, T + 1):
            sg[(jid, t)] = m.addVar(lb=0.0, name=f"sg_{jid}_{t}")
            pisq[(jid, t)] = m.addVar(lb=p_lb, ub=p_ub, name=f"pisq_{jid}_{t}")
            lg[(jid, t)] = m.addVar(lb=0.0, name=f"lg_{jid}_{t}")
            qg[(jid, t)] = m.addVar(lb=0.0, name=f"qg_{jid}_{t}")
            gamma[(jid, t)] = m.addVar(lb=0.0, name=f"gamma_{jid}_{t}")
            for sid in intervals_by_j[jid]:
                interval = interval_data[(jid, sid)]
                sgs[(jid, sid, t)] = m.addVar(
                    lb=float(interval["interval_lb"]),
                    ub=float(interval["interval_ub"]),
                    name=f"sgs_{jid}_{sid}_{t}"
                )

    for c in connections:
        cid = c["id"]
        for t in range(1, T + 1):
            phig[(cid, t)] = m.addVar(lb=0.0, name=f"phig_{cid}_{t}")

    for z in zones:
        zid = z["id"]
        for t in range(1, T + 1):
            psi[(zid, t)] = m.addVar(lb=gas_price_min, ub=gas_price_max, name=f"psi_{zid}_{t}")

    for g in gens:
        if g["is_gfpp"] and g["gas_junction"] is not None:
            gid = g["id"]
            for t in range(1, T + 1):
                zpsi_on[(gid, t)] = m.addVar(lb=0.0, ub=gas_price_max, name=f"zpsi_on_{gid}_{t}")

    m.update()

    # Helper for pre-horizon status constants
    def pre_horizon_o_const(gen, tau):
        # tau <= 0
        init_status = int(gen["initial_status"])
        a = int(gen.get("initial_active_periods", 0))
        i = int(gen.get("initial_inactive_periods", 0))
        if init_status == 1:
            # on in [-a+1, 0], off before that
            return 1.0 if tau >= -a + 1 else 0.0
        else:
            # off in [-i+1, 0], on before that
            return 0.0 if tau >= -i + 1 else 1.0

    # Initial lock due to residual up/down requirement
    for g in gens:
        gid = g["id"]
        mu = int(g["min_up_time"])
        md = int(g["min_down_time"])
        init_status = int(g["initial_status"])
        a = int(g.get("initial_active_periods", 0))
        i = int(g.get("initial_inactive_periods", 0))

        if init_status == 1 and a < mu:
            lock = min(T, mu - a)
            for t in range(1, lock + 1):
                m.addConstr(o[(gid, t)] == 1, name=f"init_up_lock_{gid}_{t}")
        if init_status == 0 and i < md:
            lock = min(T, md - i)
            for t in range(1, lock + 1):
                m.addConstr(o[(gid, t)] == 0, name=f"init_down_lock_{gid}_{t}")

    # Unit commitment logic
    for g in gens:
        gid = g["id"]
        pmin = float(g["min_power"])
        pmax = float(g["max_power"])
        ru = float(g["ramp_up"])
        rd = float(g["ramp_down"])
        mu = int(g["min_up_time"])
        md = int(g["min_down_time"])

        # Transition
        for t in range(1, T + 1):
            m.addConstr(vp[(gid, t)] - vm[(gid, t)] == o[(gid, t)] - o[(gid, t - 1)],
                        name=f"trans_{gid}_{t}")

        # Min up-time
        for t in range(1, T + 1):
            start = max(1, t - mu + 1)
            m.addConstr(quicksum(vp[(gid, k)] for k in range(start, t + 1)) <= o[(gid, t)],
                        name=f"minup_{gid}_{t}")

        # Min down-time (as described)
        for t in range(md, T + 1):
            m.addConstr(quicksum(vp[(gid, k)] for k in range(t - md + 1, t + 1)) <= 1 - o[(gid, t - md)],
                        name=f"mindown_{gid}_{t}")

        # Startup cost tiers
        sc_params = g.get("startup_cost_params", [])
        for t in range(1, T + 1):
            for idx, pair in enumerate(sc_params):
                L = int(pair[0])
                C = float(pair[1])
                expr = o[(gid, t)]
                for n in range(1, L + 1):
                    tt = t - n
                    if tt >= 0:
                        expr = expr - o[(gid, tt)]
                    else:
                        expr = expr - pre_horizon_o_const(g, tt)
                m.addConstr(r[(gid, t)] >= C * expr, name=f"startup_cost_{gid}_{t}_{idx}")

        # Dispatch + bids
        for t in range(1, T + 1):
            m.addConstr(p[(gid, t)] == quicksum(se[(gid, b, t)] for b in bids_by_gen[gid]),
                        name=f"p_sum_se_{gid}_{t}")
            m.addConstr(p[(gid, t)] >= pmin * o[(gid, t)], name=f"pmin_{gid}_{t}")
            m.addConstr(p[(gid, t)] <= pmax * o[(gid, t)], name=f"pmax_{gid}_{t}")

            m.addConstr(p[(gid, t)] - p[(gid, t - 1)] <= ru * o[(gid, t - 1)] + pmax * vp[(gid, t)],
                        name=f"rampup_{gid}_{t}")
            m.addConstr(p[(gid, t - 1)] - p[(gid, t)] <= rd * o[(gid, t)] + pmin * vm[(gid, t)],
                        name=f"rampdown_{gid}_{t}")

            for b in bids_by_gen[gid]:
                bmax = float(bid_data[(gid, b)]["max_amount"])
                m.addConstr(se[(gid, b, t)] <= bmax * w[(gid, b, t)], name=f"se_w_{gid}_{b}_{t}")
                m.addConstr(w[(gid, b, t)] <= o[(gid, t)], name=f"w_on_{gid}_{b}_{t}")

        # Sequential bids for GFPP
        if g["is_gfpp"]:
            ordered = bids_by_gen[gid]
            for t in range(1, T + 1):
                for k in range(len(ordered) - 1):
                    bcur = ordered[k]
                    bnext = ordered[k + 1]
                    bcur_max = float(bid_data[(gid, bcur)]["max_amount"])
                    m.addConstr(w[(gid, bnext, t)] <= w[(gid, bcur, t)],
                                name=f"w_seq_{gid}_{k}_{t}")
                    m.addConstr(se[(gid, bcur, t)] >= bcur_max * w[(gid, bnext, t)],
                                name=f"se_seq_{gid}_{k}_{t}")

    # GFPP price-validity logic
    for g in gens:
        gid = g["id"]
        if g["is_gfpp"] and g["gas_junction"] is not None:
            ordered = bids_by_gen[gid]
            thresholds = [get_bid_threshold(g, bid_data[(gid, b)], gas_price_min, gas_price_max) for b in ordered]
            j = g["gas_junction"]
            zid = junction_to_zone.get(j, None)

            for t in range(1, T + 1):
                # phimax from bid-selection pattern
                expr = 0.0
                if len(ordered) >= 1:
                    for k in range(len(ordered) - 1):
                        expr += thresholds[k] * (w[(gid, ordered[k], t)] - w[(gid, ordered[k + 1], t)])
                    expr += thresholds[-1] * w[(gid, ordered[-1], t)]
                m.addConstr(phimax[(gid, t)] == expr, name=f"phimax_def_{gid}_{t}")

                # phimax >= psi * o via exact McCormick for binary o
                if zid is not None:
                    z = zpsi_on[(gid, t)]
                    ps = psi[(zid, t)]
                    on = o[(gid, t)]
                    L = gas_price_min
                    U = gas_price_max
                    m.addConstr(z <= U * on, name=f"z_u1_{gid}_{t}")
                    m.addConstr(z >= L * on, name=f"z_l1_{gid}_{t}")
                    m.addConstr(z <= ps - L * (1 - on), name=f"z_u2_{gid}_{t}")
                    m.addConstr(z >= ps - U * (1 - on), name=f"z_l2_{gid}_{t}")
                    m.addConstr(phimax[(gid, t)] >= z, name=f"phimax_ge_z_{gid}_{t}")
        else:
            for t in range(1, T + 1):
                m.addConstr(phimax[(gid, t)] == 0.0, name=f"phimax_zero_{gid}_{t}")

    # Power flow + nodal balance
    ref_bus = bus_ids[0] if bus_ids else None
    for t in range(1, T + 1):
        if ref_bus is not None:
            m.addConstr(theta[(ref_bus, t)] == 0.0, name=f"ref_theta_{t}")

        for l in lines:
            lid = l["id"]
            i = l["from_bus"]
            j = l["to_bus"]
            b = float(l["susceptance"])
            dmax = float(l["angle_diff_limit"])

            m.addConstr(f[(lid, t)] == b * (theta[(i, t)] - theta[(j, t)]), name=f"flowdc_{lid}_{t}")
            m.addConstr(theta[(i, t)] - theta[(j, t)] <= dmax, name=f"ang_up_{lid}_{t}")
            m.addConstr(theta[(i, t)] - theta[(j, t)] >= -dmax, name=f"ang_dn_{lid}_{t}")

        for b in buses:
            bid = b["id"]
            demand = eta_p * float(b["demand_profile"][t - 1])

            gen_sum = quicksum(p[(gid, t)] for gid in gens_at_bus[bid])
            out_sum = quicksum(f[(lid, t)] for lid in outgoing_lines_by_bus[bid])
            in_sum = quicksum(f[(lid, t)] for lid in incoming_lines_by_bus[bid])

            m.addConstr(gen_sum - demand == out_sum - in_sum, name=f"pbalance_{bid}_{t}")

    # Gas network constraints
    for t in range(1, T + 1):
        for j in junctions:
            jid = j["id"]
            demand = eta_g * float(j["gas_demand_profile"][t - 1])

            m.addConstr(lg[(jid, t)] + qg[(jid, t)] == demand, name=f"gas_dem_sat_{jid}_{t}")
            m.addConstr(qg[(jid, t)] <= demand, name=f"gas_shed_cap_{jid}_{t}")

            if j["is_source"]:
                m.addConstr(sg[(jid, t)] == quicksum(sgs[(jid, sid, t)] for sid in intervals_by_j[jid]),
                            name=f"sg_source_{jid}_{t}")
            else:
                m.addConstr(sg[(jid, t)] == 0.0, name=f"sg_nonsource_{jid}_{t}")

            out_sum = quicksum(phig[(cid, t)] for cid in outgoing_conn_by_j[jid])
            in_sum = quicksum(phig[(cid, t)] for cid in incoming_conn_by_j[jid])

            m.addConstr(sg[(jid, t)] - lg[(jid, t)] - gamma[(jid, t)] == out_sum - in_sum,
                        name=f"gas_balance_{jid}_{t}")

        for c in connections:
            cid = c["id"]
            jf = c["from_junction"]
            jt = c["to_junction"]
            typ = c["type"]

            # Pipeline Weymouth convex relaxation
            if typ == "pipeline" and c.get("weymouth_factor") is not None:
                k = float(c["weymouth_factor"])
                m.addQConstr(pisq[(jf, t)] - pisq[(jt, t)] >= k * phig[(cid, t)] * phig[(cid, t)],
                             name=f"weymouth_{cid}_{t}")

            # Compressor ratio
            if c.get("compression_ratio_lb") is not None and c.get("compression_ratio_ub") is not None:
                lb = float(c["compression_ratio_lb"])
                ub = float(c["compression_ratio_ub"])
                m.addConstr(pisq[(jt, t)] >= (lb ** 2) * pisq[(jf, t)], name=f"comp_lb_{cid}_{t}")
                m.addConstr(pisq[(jt, t)] <= (ub ** 2) * pisq[(jf, t)], name=f"comp_ub_{cid}_{t}")

            # Control valve ratio (if provided)
            if c.get("control_ratio_lb") is not None and c.get("control_ratio_ub") is not None:
                lb = float(c["control_ratio_lb"])
                ub = float(c["control_ratio_ub"])
                m.addConstr(pisq[(jt, t)] >= (lb ** 2) * pisq[(jf, t)], name=f"valve_lb_{cid}_{t}")
                m.addConstr(pisq[(jt, t)] <= (ub ** 2) * pisq[(jf, t)], name=f"valve_ub_{cid}_{t}")

        # GFPP gas consumption relaxation at each junction
        for jid in junction_ids:
            gfpps = gfpp_at_junction[jid]
            if gfpps:
                qexpr = 0.0
                for gid in gfpps:
                    gg = gen_by_id[gid]
                    H = gg.get("heat_rate_coefficients", {})
                    H2 = float(H.get("H_u2") or 0.0)
                    H1 = float(H.get("H_u1") or 0.0)
                    H0 = float(H.get("H_u0") or 0.0)
                    qexpr += H2 * p[(gid, t)] * p[(gid, t)] + H1 * p[(gid, t)] + H0 * o[(gid, t)]
                m.addQConstr(gamma[(jid, t)] >= qexpr, name=f"gamma_relax_{jid}_{t}")

    # Objective
    elec_cost = quicksum(
        float(gen_by_id[gid]["no_load_cost"]) * o[(gid, t)] + r[(gid, t)]
        for gid in gen_ids for t in range(1, T + 1)
    ) + quicksum(
        float(bid_data[(gid, b)]["price"]) * se[(gid, b, t)]
        for gid in gen_ids for b in bids_by_gen[gid] for t in range(1, T + 1)
    )

    gas_cost = quicksum(
        float(junc_by_id[jid]["demand_shedding_cost"]) * qg[(jid, t)]
        for jid in junction_ids for t in range(1, T + 1)
    ) + quicksum(
        float(interval_data[(jid, sid)]["slope"]) * sgs[(jid, sid, t)]
        for jid in junction_ids for sid in intervals_by_j[jid] for t in range(1, T + 1)
    )

    m.setObjective(beta * elec_cost + (1.0 - beta) * gas_cost, GRB.MINIMIZE)

    # Callback logging
    m._best_logged = float("inf")

    def cb(model, where):
        if logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < model._best_logged:
                model._best_logged = obj
                logger.log(float(obj))

    m.optimize(cb)

    # Extract best solution if available
    has_sol = m.SolCount > 0

    def val(v, default=0.0):
        if has_sol:
            return float(v.X)
        return float(default)

    primary = {}

    # o, p include period 0..T
    for gid in gen_ids:
        for t in range(0, T + 1):
            primary[f"o_{gid}_{t}"] = val(o[(gid, t)])
            primary[f"p_{gid}_{t}"] = val(p[(gid, t)])

    # period 1..T variables
    for gid in gen_ids:
        for t in range(1, T + 1):
            primary[f"vp_{gid}_{t}"] = val(vp[(gid, t)])
            primary[f"vm_{gid}_{t}"] = val(vm[(gid, t)])
            primary[f"r_{gid}_{t}"] = val(r[(gid, t)])
            primary[f"phimax_{gid}_{t}"] = val(phimax[(gid, t)])

            for b in bids_by_gen[gid]:
                primary[f"w_{gid}_{b}_{t}"] = val(w[(gid, b, t)])
                primary[f"se_{gid}_{b}_{t}"] = val(se[(gid, b, t)])

    for lid in line_ids:
        for t in range(1, T + 1):
            primary[f"f_{lid}_{t}"] = val(f[(lid, t)])

    for bid in bus_ids:
        for t in range(1, T + 1):
            primary[f"theta_{bid}_{t}"] = val(theta[(bid, t)])

    for jid in junction_ids:
        for t in range(1, T + 1):
            primary[f"sg_{jid}_{t}"] = val(sg[(jid, t)])
            primary[f"pisq_{jid}_{t}"] = val(pisq[(jid, t)])
            primary[f"lg_{jid}_{t}"] = val(lg[(jid, t)])
            primary[f"qg_{jid}_{t}"] = val(qg[(jid, t)])
            primary[f"gamma_{jid}_{t}"] = val(gamma[(jid, t)])
            for sid in intervals_by_j[jid]:
                primary[f"sgs_{jid}_{sid}_{t}"] = val(sgs[(jid, sid, t)])

    for cid in conn_ids:
        for t in range(1, T + 1):
            primary[f"phig_{cid}_{t}"] = val(phig[(cid, t)])

    for zid in zone_ids:
        for t in range(1, T + 1):
            primary[f"psi_{zid}_{t}"] = val(psi[(zid, t)])

    # Recompute objective from extracted primal values
    elec_obj = 0.0
    gas_obj = 0.0

    for g in gens:
        gid = g["id"]
        no_load = float(g["no_load_cost"])
        for t in range(1, T + 1):
            elec_obj += no_load * primary[f"o_{gid}_{t}"]
            elec_obj += primary[f"r_{gid}_{t}"]
            for b in bids_by_gen[gid]:
                price = float(bid_data[(gid, b)]["price"])
                elec_obj += price * primary[f"se_{gid}_{b}_{t}"]

    for j in junctions:
        jid = j["id"]
        shed_c = float(j["demand_shedding_cost"])
        for t in range(1, T + 1):
            gas_obj += shed_c * primary[f"qg_{jid}_{t}"]
            for sid in intervals_by_j[jid]:
                slope = float(interval_data[(jid, sid)]["slope"])
                gas_obj += slope * primary[f"sgs_{jid}_{sid}_{t}"]

    objective_value = beta * elec_obj + (1.0 - beta) * gas_obj

    solution = {
        "objective_value": float(objective_value),
        "released_problem_variant": "integrated_primal_ucgna",
        "objective_weight_beta": beta,
        "primary_variables": primary,
    }

    if logger and has_sol:
        logger.log_solution(float(objective_value), solution)

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