import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def get_profile_value(profile_obj, t):
    ptype = profile_obj.get("type", "constant")
    if ptype == "constant":
        return float(profile_obj.get("value", 1.0))
    vals = profile_obj.get("values", [])
    if t < len(vals):
        return float(vals[t])
    return 1.0


def build_and_solve(instance, time_limit, logger=None):
    t0 = time.time()

    dims = instance["problem_dimensions"]
    policy = instance["policy"]
    subp = instance["subperiods"]
    gens = instance["generators"]
    lines = instance["transmission_lines"]
    demand = instance["demand"]
    nse_segments = instance["nse_segments"]
    avail_profiles_raw = instance["availability_profiles"]

    # Dimensions
    G = len(gens)
    L = len(lines)
    S = len(nse_segments)

    W = int(subp["num_subperiods"])
    H = int(subp["hours_per_subperiod"])
    T = W * H

    subperiod_weights = subp["subperiod_weights"]
    hour_weight = [subperiod_weights[t // H] / H for t in range(T)]

    # Zone set from demand keys and assets
    zone_ids = set()
    for k in demand.keys():
        zone_ids.add(int(k))
    for g in gens:
        zone_ids.add(int(g["zone"]))
    for ln in lines:
        zone_ids.add(int(ln["from_zone"]))
        zone_ids.add(int(ln["to_zone"]))
    zone_ids = sorted(zone_ids)

    # Demand matrix
    dem = {}
    for z in zone_ids:
        arr = demand.get(str(z), demand.get(z, [0.0] * T))
        if len(arr) < T:
            arr = list(arr) + [0.0] * (T - len(arr))
        dem[z] = [float(v) for v in arr[:T]]

    # Availability lookup
    avail_profiles = {}
    for g in gens:
        gid = g["id"]
        key = str(gid)
        avail_profiles[gid] = avail_profiles_raw.get(key, {"type": "constant", "value": 1.0})

    def avail(i, t):
        gid = gens[i]["id"]
        return get_profile_value(avail_profiles[gid], t)

    # Sets
    uc_ids = [i for i, g in enumerate(gens) if g.get("is_UC", False)]
    storage_ids = [i for i, g in enumerate(gens) if g.get("is_storage", False)]
    hydro_ids = [i for i, g in enumerate(gens) if g.get("is_hydro", False)]

    storage_set = set(storage_ids)
    hydro_set = set(hydro_ids)
    uc_set = set(uc_ids)

    non_uc_ids = [i for i in range(G) if i not in uc_set]
    non_uc_non_storage_non_hydro = [i for i in range(G) if (i not in uc_set and i not in storage_set and i not in hydro_set)]

    rps_ids = [i for i, g in enumerate(gens) if g.get("is_RPS_qualifying", False)]

    # Grouping by zone
    gens_by_zone = defaultdict(list)
    storage_by_zone = defaultdict(list)
    for i, g in enumerate(gens):
        z = int(g["zone"])
        gens_by_zone[z].append(i)
        if i in storage_set:
            storage_by_zone[z].append(i)

    out_lines = defaultdict(list)
    in_lines = defaultdict(list)
    for l, ln in enumerate(lines):
        out_lines[int(ln["from_zone"])].append(l)
        in_lines[int(ln["to_zone"])].append(l)

    # Model
    m = gp.Model("gen_expansion_uc")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = max(1, int(time_limit))

    # Investment vars
    yPnew = {}
    yPret = {}
    yP = {}

    yEnew = {}
    yEret = {}
    yE = {}

    for i, g in enumerate(gens):
        cap_size = float(g.get("capacity_size_MW", 1.0))
        existing = float(g.get("existing_capacity_MW", 0.0))
        max_total = float(g.get("max_capacity_MW", existing))
        max_new = max(0.0, max_total - existing)

        yPnew[i] = m.addVar(lb=0.0, ub=max_new / cap_size if cap_size > 0 else 0.0, vtype=GRB.INTEGER, name=f"yPnew_{i}")
        can_retire = bool(g.get("can_retire", True))
        ret_ub = (existing / cap_size) if (cap_size > 0 and can_retire) else 0.0
        yPret[i] = m.addVar(lb=0.0, ub=ret_ub, vtype=GRB.INTEGER, name=f"yPret_{i}")
        yP[i] = m.addVar(lb=0.0, ub=max_total, vtype=GRB.CONTINUOUS, name=f"yP_{i}")

        m.addConstr(yP[i] == existing + cap_size * (yPnew[i] - yPret[i]), name=f"cap_balance_{i}")
        m.addConstr(cap_size * yPnew[i] <= max_new + 1e-9, name=f"new_cap_limit_{i}")
        m.addConstr(cap_size * yPret[i] <= existing + 1e-9, name=f"ret_cap_limit_{i}")

        if i in storage_set:
            e_size = float(g.get("storage_capacity_size_MWh", 1.0))
            e_existing = float(g.get("existing_storage_capacity_MWh", 0.0))
            e_max_total = float(g.get("max_storage_capacity_MWh", e_existing))
            e_max_new = max(0.0, e_max_total - e_existing)

            yEnew[i] = m.addVar(lb=0.0, ub=e_max_new / e_size if e_size > 0 else 0.0, vtype=GRB.INTEGER, name=f"yEnew_{i}")
            e_ret_ub = (e_existing / e_size) if (e_size > 0 and g.get("can_retire", True)) else 0.0
            yEret[i] = m.addVar(lb=0.0, ub=e_ret_ub, vtype=GRB.INTEGER, name=f"yEret_{i}")
            yE[i] = m.addVar(lb=0.0, ub=e_max_total, vtype=GRB.CONTINUOUS, name=f"yE_{i}")

            m.addConstr(yE[i] == e_existing + e_size * (yEnew[i] - yEret[i]), name=f"e_balance_{i}")
            m.addConstr(e_size * yEnew[i] <= e_max_new + 1e-9, name=f"e_new_limit_{i}")
            m.addConstr(e_size * yEret[i] <= e_existing + 1e-9, name=f"e_ret_limit_{i}")

            min_dur = float(g.get("min_duration_MWh_per_MW", 0.0))
            max_dur = float(g.get("max_duration_MWh_per_MW", 1e6))
            m.addConstr(yE[i] >= min_dur * yP[i], name=f"dur_min_{i}")
            m.addConstr(yE[i] <= max_dur * yP[i], name=f"dur_max_{i}")

    yFnew = {}
    for l, ln in enumerate(lines):
        ub = float(ln.get("max_new_capacity_MW", 0.0))
        yFnew[l] = m.addVar(lb=0.0, ub=ub, vtype=GRB.INTEGER, name=f"yFnew_{l}")

    # Operational vars
    xinj = {}
    xwdw = {}
    xsoc = {}
    xhydro = {}
    xspill = {}
    xflow = {}
    xnse = {}

    xcommit = {}
    xstart = {}
    xshut = {}

    for i in range(G):
        for t in range(T):
            xinj[i, t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"xinj_{i}_{t}")

    for i in storage_ids:
        for t in range(T):
            xwdw[i, t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"xwdw_{i}_{t}")
            xsoc[i, t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"xsoc_{i}_{t}")

    for i in hydro_ids:
        for t in range(T):
            xhydro[i, t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"xhydro_{i}_{t}")
            xspill[i, t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"xspill_{i}_{t}")

    for l in range(L):
        for t in range(T):
            xflow[l, t] = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"xflow_{l}_{t}")

    for s in range(S):
        for z in zone_ids:
            for t in range(T):
                xnse[s, z, t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"xnse_{s}_{z}_{t}")

    for i in uc_ids:
        for t in range(T):
            xcommit[i, t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"xcommit_{i}_{t}")
            xstart[i, t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"xstart_{i}_{t}")
            xshut[i, t] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"xshut_{i}_{t}")

    xRPS = {}
    xCO2 = {}
    if policy.get("RPS_enabled", False):
        for w in range(W):
            xRPS[w] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"xRPS_{w}")
    if policy.get("CO2_cap_enabled", False):
        for w in range(W):
            xCO2[w] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"xCO2_{w}")

    m.update()

    # Helpers
    def prev_t(t):
        h = t % H
        if h == 0:
            return t + H - 1
        return t - 1

    # Constraints

    # Power balance
    for z in zone_ids:
        for t in range(T):
            gen_sum = gp.quicksum(xinj[i, t] for i in gens_by_zone[z])
            wdw_sum = gp.quicksum(xwdw[i, t] for i in storage_by_zone[z]) if storage_by_zone[z] else 0.0
            out_sum = gp.quicksum(xflow[l, t] for l in out_lines[z]) if out_lines[z] else 0.0
            in_sum = gp.quicksum(xflow[l, t] for l in in_lines[z]) if in_lines[z] else 0.0
            nse_sum = gp.quicksum(xnse[s, z, t] for s in range(S))
            m.addConstr(gen_sum - wdw_sum - out_sum + in_sum + nse_sum == dem[z][t], name=f"balance_{z}_{t}")

    # NSE segment limits
    for s, seg in enumerate(nse_segments):
        max_frac = float(seg.get("max_frac", 0.0))
        for z in zone_ids:
            for t in range(T):
                m.addConstr(xnse[s, z, t] <= max_frac * dem[z][t], name=f"nse_max_{s}_{z}_{t}")

    # Transmission limits
    for l, ln in enumerate(lines):
        cap_total = float(ln.get("existing_capacity_MW", 0.0)) + yFnew[l]
        for t in range(T):
            m.addConstr(xflow[l, t] <= cap_total, name=f"flow_up_{l}_{t}")
            m.addConstr(-xflow[l, t] <= cap_total, name=f"flow_dn_{l}_{t}")

    # Generation/resource constraints
    for i, g in enumerate(gens):
        min_out = float(g.get("min_output_frac", 0.0))
        ru = float(g.get("ramp_up_frac_per_hr", 1e6))
        rd = float(g.get("ramp_dn_frac_per_hr", 1e6))
        cap_size = float(g.get("capacity_size_MW", 1.0))

        if i in uc_set:
            # UC logical and output
            for t in range(T):
                a = avail(i, t)
                m.addConstr(cap_size * xcommit[i, t] <= yP[i], name=f"uc_commit_cap_{i}_{t}")
                m.addConstr(cap_size * xstart[i, t] <= yP[i], name=f"uc_start_cap_{i}_{t}")
                m.addConstr(cap_size * xshut[i, t] <= yP[i], name=f"uc_shut_cap_{i}_{t}")

                m.addConstr(xinj[i, t] >= cap_size * min_out * xcommit[i, t], name=f"uc_min_gen_{i}_{t}")
                m.addConstr(xinj[i, t] <= cap_size * a * xcommit[i, t], name=f"uc_max_gen_{i}_{t}")

                tp = prev_t(t)
                m.addConstr(
                    xcommit[i, t] - xcommit[i, tp] == xstart[i, t] - xshut[i, t],
                    name=f"uc_trans_{i}_{t}",
                )

                # UC ramping
                acoef_up = min(a, max(min_out, ru))
                acoef_dn = min(a, max(min_out, rd))

                m.addConstr(
                    xinj[i, t] - xinj[i, tp]
                    <= cap_size * ru * (xcommit[i, t] - xstart[i, t])
                    + cap_size * acoef_up * xstart[i, t]
                    - cap_size * min_out * xshut[i, t],
                    name=f"uc_ramp_up_{i}_{t}",
                )
                m.addConstr(
                    xinj[i, tp] - xinj[i, t]
                    <= cap_size * rd * (xcommit[i, t] - xstart[i, t])
                    + cap_size * acoef_dn * xshut[i, t]
                    - cap_size * min_out * xstart[i, t],
                    name=f"uc_ramp_dn_{i}_{t}",
                )

            # Min up/down with circular indexing within each subperiod
            mut = int(g.get("min_up_time_hr", 0))
            mdt = int(g.get("min_down_time_hr", 0))
            for w in range(W):
                for h in range(H):
                    t = w * H + h
                    if mut > 0:
                        up_terms = [xstart[i, w * H + ((h - j) % H)] for j in range(mut)]
                        m.addConstr(xcommit[i, t] >= gp.quicksum(up_terms), name=f"uc_mut_{i}_{t}")
                    if mdt > 0:
                        dn_terms = [xshut[i, w * H + ((h - j) % H)] for j in range(mdt)]
                        m.addConstr(
                            xcommit[i, t] + gp.quicksum(dn_terms) <= yP[i] / cap_size,
                            name=f"uc_mdt_{i}_{t}",
                        )

        elif i in storage_set:
            ce = float(g.get("charge_efficiency", 1.0))
            de = float(g.get("discharge_efficiency", 1.0))
            selfd = float(g.get("self_discharge_rate", 0.0))
            for t in range(T):
                a = avail(i, t)
                m.addConstr(xwdw[i, t] <= a * yP[i], name=f"st_wdw_max_{i}_{t}")
                m.addConstr(xinj[i, t] + xwdw[i, t] <= yP[i], name=f"st_power_sum_{i}_{t}")
                m.addConstr(ce * xwdw[i, t] <= yE[i] - xsoc[i, t], name=f"st_room_{i}_{t}")
                m.addConstr(xinj[i, t] / max(de, 1e-6) <= xsoc[i, t], name=f"st_discharge_soc_{i}_{t}")
                m.addConstr(xsoc[i, t] <= yE[i], name=f"st_soc_cap_{i}_{t}")

                tp = prev_t(t)
                m.addConstr(
                    xsoc[i, t] - (1.0 - selfd) * xsoc[i, tp]
                    == ce * xwdw[i, t] - xinj[i, t] / max(de, 1e-6),
                    name=f"st_soc_bal_{i}_{t}",
                )

        elif i in hydro_set:
            dur = float(g.get("duration_MWh_per_MW", 0.0))
            for t in range(T):
                a = avail(i, t)
                # Hydro dispatch upper bound via availability
                m.addConstr(xinj[i, t] <= a * yP[i], name=f"hy_max_gen_{i}_{t}")
                m.addConstr(xhydro[i, t] <= dur * yP[i], name=f"hy_res_cap_{i}_{t}")
                m.addConstr(xinj[i, t] + xspill[i, t] >= min_out * yP[i], name=f"hy_min_out_{i}_{t}")

                tp = prev_t(t)
                inflow_frac = a  # using provided profile as normalized inflow proxy
                m.addConstr(
                    xhydro[i, t] - xhydro[i, tp] == inflow_frac * yP[i] - xinj[i, t] - xspill[i, t],
                    name=f"hy_bal_{i}_{t}",
                )

            # Non-UC ramping for hydro
            for t in range(T):
                tp = prev_t(t)
                m.addConstr(xinj[i, t] - xinj[i, tp] <= ru * yP[i], name=f"hy_rup_{i}_{t}")
                m.addConstr(xinj[i, tp] - xinj[i, t] <= rd * yP[i], name=f"hy_rdn_{i}_{t}")

        else:
            # Non-UC conventional/VRE
            for t in range(T):
                a = avail(i, t)
                m.addConstr(xinj[i, t] <= a * yP[i], name=f"gen_max_{i}_{t}")
                m.addConstr(xinj[i, t] >= min_out * yP[i], name=f"gen_min_{i}_{t}")

                tp = prev_t(t)
                m.addConstr(xinj[i, t] - xinj[i, tp] <= ru * yP[i], name=f"gen_rup_{i}_{t}")
                m.addConstr(xinj[i, tp] - xinj[i, t] <= rd * yP[i], name=f"gen_rdn_{i}_{t}")

    # Policy constraints
    total_weighted_demand = sum(hour_weight[t] * dem[z][t] for z in zone_ids for t in range(T))

    if policy.get("RPS_enabled", False):
        rps_share = float(policy.get("RPS_share", 0.0))
        lhs = gp.quicksum(hour_weight[t] * xinj[i, t] for i in rps_ids for t in range(T)) + gp.quicksum(xRPS[w] for w in range(W))
        m.addConstr(lhs >= rps_share * total_weighted_demand, name="RPS_req")

    if policy.get("CO2_cap_enabled", False):
        cap_rate = policy.get("CO2_cap_tons_per_MWh", None)
        if cap_rate is None:
            cap_rate = 0.05
        cap_tons = float(cap_rate) * total_weighted_demand

        emis_terms = []
        for i, g in enumerate(gens):
            ef = float(g.get("co2_tons_per_MWh", 0.0))
            if abs(ef) < 1e-12:
                continue
            for t in range(T):
                emis_terms.append(hour_weight[t] * ef * xinj[i, t])
                if i in storage_set:
                    emis_terms.append(hour_weight[t] * ef * xwdw[i, t])

        lhs = gp.quicksum(emis_terms) - gp.quicksum(xCO2[w] for w in range(W))
        m.addConstr(lhs <= cap_tons, name="CO2_cap")

    # Objective
    obj = gp.LinExpr()

    # Fixed costs
    for i, g in enumerate(gens):
        cap_size = float(g.get("capacity_size_MW", 1.0))
        inv = float(g.get("inv_cost_per_MW_yr", 0.0))
        fom = float(g.get("fom_cost_per_MW_yr", 0.0))

        obj += inv * cap_size * yPnew[i]
        obj += fom * yP[i]

        if i in hydro_set:
            dur = float(g.get("duration_MWh_per_MW", 0.0))
            inv_e = float(g.get("inv_cost_energy_per_MWh_yr", 0.0))
            fom_e = float(g.get("fom_cost_energy_per_MWh_yr", 0.0))
            obj += inv_e * dur * cap_size * yPnew[i]
            obj += fom_e * dur * yP[i]

        if i in storage_set:
            e_size = float(g.get("storage_capacity_size_MWh", 1.0))
            inv_e = float(g.get("inv_cost_energy_per_MWh_yr", 0.0))
            fom_e = float(g.get("fom_cost_energy_per_MWh_yr", 0.0))
            obj += inv_e * e_size * yEnew[i]
            obj += fom_e * yE[i]

    for l, ln in enumerate(lines):
        invf = float(ln.get("inv_cost_per_MW_yr", 0.0))
        obj += invf * yFnew[l]

    # Variable operating + startup + NSE + policy penalties
    for i, g in enumerate(gens):
        vc = float(g.get("var_cost_per_MWh", 0.0))
        if abs(vc) > 0:
            for t in range(T):
                obj += hour_weight[t] * vc * xinj[i, t]
            if i in storage_set:
                for t in range(T):
                    obj += hour_weight[t] * vc * xwdw[i, t]

    for i in uc_ids:
        sc = float(gens[i].get("start_cost_per_unit", 0.0))
        if abs(sc) > 0:
            for t in range(T):
                obj += hour_weight[t] * sc * xstart[i, t]

    for s, seg in enumerate(nse_segments):
        c = float(seg.get("cost_per_MWh", 0.0))
        if abs(c) > 0:
            for z in zone_ids:
                for t in range(T):
                    obj += hour_weight[t] * c * xnse[s, z, t]

    if policy.get("RPS_enabled", False):
        c = float(policy.get("RPS_noncompliance_cost_per_MWh", 0.0))
        obj += c * gp.quicksum(xRPS[w] for w in range(W))

    if policy.get("CO2_cap_enabled", False):
        c = float(policy.get("CO2_noncompliance_cost_per_ton", 0.0))
        obj += c * gp.quicksum(xCO2[w] for w in range(W))

    m.setObjective(obj, GRB.MINIMIZE)

    # Callback for incumbent logging
    if logger is not None:
        m._logger = logger
        m._best = float("inf")

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                objv = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if objv + 1e-9 < model._best:
                    model._best = objv
                    model._logger.log(float(objv))

        callback = cb
    else:
        callback = None

    # Solve
    remaining = max(1, int(time_limit - (time.time() - t0)))
    m.Params.TimeLimit = remaining

    if callback:
        m.optimize(callback)
    else:
        m.optimize()

    # Build solution
    feasible = m.SolCount > 0
    if not feasible:
        return {
            "objective_value": 1e30,
            "model_variables": {},
            "investment_decisions": [],
            "transmission_decisions": [],
            "_status": m.Status,
        }

    obj_val = float(m.ObjVal)

    model_vars = {}
    eps = 1e-9

    # Capacity variables
    for i in range(G):
        v = yPnew[i].X
        if abs(v) > eps:
            model_vars[f"yPnew_{i}"] = float(v)
        v = yPret[i].X
        if abs(v) > eps:
            model_vars[f"yPret_{i}"] = float(v)
        v = yP[i].X
        if abs(v) > eps:
            model_vars[f"yP_{i}"] = float(v)

    for i in storage_ids:
        v = yEnew[i].X
        if abs(v) > eps:
            model_vars[f"yEnew_{i}"] = float(v)
        v = yEret[i].X
        if abs(v) > eps:
            model_vars[f"yEret_{i}"] = float(v)
        v = yE[i].X
        if abs(v) > eps:
            model_vars[f"yE_{i}"] = float(v)

    for l in range(L):
        v = yFnew[l].X
        if abs(v) > eps:
            model_vars[f"yFnew_{l}"] = float(v)

    # Operational variables
    for i in range(G):
        for t in range(T):
            v = xinj[i, t].X
            if abs(v) > eps:
                model_vars[f"xinj_{i}_{t}"] = float(v)

    for i in storage_ids:
        for t in range(T):
            v = xwdw[i, t].X
            if abs(v) > eps:
                model_vars[f"xwdw_{i}_{t}"] = float(v)
            v = xsoc[i, t].X
            if abs(v) > eps:
                model_vars[f"xsoc_{i}_{t}"] = float(v)

    for i in hydro_ids:
        for t in range(T):
            v = xhydro[i, t].X
            if abs(v) > eps:
                model_vars[f"xhydro_{i}_{t}"] = float(v)

    for l in range(L):
        for t in range(T):
            v = xflow[l, t].X
            if abs(v) > eps:
                model_vars[f"xflow_{l}_{t}"] = float(v)

    for s in range(S):
        for z in zone_ids:
            for t in range(T):
                v = xnse[s, z, t].X
                if abs(v) > eps:
                    model_vars[f"xnse_{s}_{z}_{t}"] = float(v)

    for i in uc_ids:
        for t in range(T):
            v = xcommit[i, t].X
            if abs(v) > eps:
                model_vars[f"xcommit_{i}_{t}"] = float(v)
            v = xstart[i, t].X
            if abs(v) > eps:
                model_vars[f"xstart_{i}_{t}"] = float(v)
            v = xshut[i, t].X
            if abs(v) > eps:
                model_vars[f"xshut_{i}_{t}"] = float(v)

    for w in range(W):
        if w in xRPS:
            v = xRPS[w].X
            if abs(v) > eps:
                model_vars[f"xRPS_{w}"] = float(v)
        if w in xCO2:
            v = xCO2[w].X
            if abs(v) > eps:
                model_vars[f"xCO2_{w}"] = float(v)

    investment_decisions = []
    for i, g in enumerate(gens):
        rec = {
            "generator_id": int(g["id"]),
            "resource_type": g.get("resource_type", ""),
            "zone": int(g["zone"]),
            "y_P_new": float(yPnew[i].X),
            "y_P_ret": float(yPret[i].X),
            "y_P_total_MW": float(yP[i].X),
        }
        if i in storage_set:
            rec["y_E_new"] = float(yEnew[i].X)
            rec["y_E_ret"] = float(yEret[i].X)
            rec["y_E_total_MWh"] = float(yE[i].X)
        investment_decisions.append(rec)

    transmission_decisions = []
    for l, ln in enumerate(lines):
        total_cap = float(ln.get("existing_capacity_MW", 0.0)) + float(yFnew[l].X)
        transmission_decisions.append(
            {
                "line_id": int(ln["id"]),
                "y_F_new_MW": float(yFnew[l].X),
                "total_capacity_MW": float(total_cap),
            }
        )

    solution = {
        "objective_value": obj_val,
        "model_variables": model_vars,
        "investment_decisions": investment_decisions,
        "transmission_decisions": transmission_decisions,
    }

    if logger is not None:
        logger.log_solution(obj_val, solution)

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