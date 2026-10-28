import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB


def prev_hour(t, hps):
    h = t % hps
    if h == 0:
        return t + hps - 1
    return t - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    start = time.time()

    with open(args.instance_path) as f:
        inst = json.load(f)

    pp = inst['problem_parameters']
    num_zones = pp['num_zones']
    num_weeks = pp['num_weeks']
    hps = pp['hours_per_subperiod']
    T = pp['total_hours']
    num_clusters = pp['num_generator_clusters']

    subperiod_weights = inst['subperiod_weights']
    hw = [subperiod_weights[t // hps] / hps for t in range(T)]

    gens = inst['generators']
    gen_params = inst['generator_parameters']
    existing_cap = {int(k): float(v) for k, v in inst['existing_capacity'].items()}
    demand = {int(k): [float(x) for x in v] for k, v in inst['demand'].items()}
    consumer_segments = inst['consumer_segments']
    trans_lines = inst['transmission_lines']
    availability = {int(k): [float(x) for x in v] for k, v in inst['availability'].items()}
    hydro_inflow = {int(k): [float(x) for x in v] for k, v in inst.get('hydro_inflow', {}).items()}
    policy = inst['policy']

    info = []
    for g in gens:
        cid = g['cluster_id']
        rt = g['resource_type']
        tp = gen_params[rt]
        info.append({
            'cid': cid, 'zone': g['zone'], 'rt': rt, 'params': tp,
            'is_uc': g['is_UC'], 'is_vre': g['is_VRE'], 'is_storage': g['is_storage'],
            'is_hydro': g['is_hydro'], 'is_rps': g['is_RPS_qualifying'],
            'can_retire': g['can_retire'], 'cap_size': g['capacity_size_MW'],
            'existing': existing_cap.get(cid, 0.0),
            'avail': availability.get(cid, [1.0] * T),
        })

    storage_idx = [i for i, inf in enumerate(info) if inf['is_storage']]
    hydro_idx = [i for i, inf in enumerate(info) if inf['is_hydro']]
    uc_idx = [i for i, inf in enumerate(info) if inf['is_uc']]
    non_uc_nsh = [i for i, inf in enumerate(info)
                  if not inf['is_uc'] and not inf['is_storage'] and not inf['is_hydro']]

    num_lines = len(trans_lines)
    num_segs = len(consumer_segments)

    m = gp.Model("cem")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.OutputFlag = 0

    # ---- Investment variables ----
    y_P_new, y_P_ret, y_P = {}, {}, {}
    for i, inf in enumerate(info):
        cid = inf['cid']
        tp = inf['params']
        max_cap = tp.get('max_capacity_MW', 0) or 0
        cs = inf['cap_size']
        if inf['is_uc']:
            ub_new = max_cap / cs if max_cap > 0 else 0
            y_P_new[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub_new, name=f"y_P_new[{cid}]")
            ub_ret = inf['existing'] / cs if inf['existing'] > 0 else 0
            if not inf['can_retire']:
                ub_ret = 0
            y_P_ret[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub_ret, name=f"y_P_ret[{cid}]")
            y_P[i] = m.addVar(lb=0, name=f"y_P[{cid}]")
            m.addConstr(y_P[i] == inf['existing'] + y_P_new[i] * cs - y_P_ret[i] * cs)
        else:
            ub_new = max_cap if max_cap > 0 else 0
            y_P_new[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub_new, name=f"y_P_new[{cid}]")
            ub_ret = inf['existing'] if inf['existing'] > 0 else 0
            if not inf['can_retire']:
                ub_ret = 0
            y_P_ret[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub_ret, name=f"y_P_ret[{cid}]")
            y_P[i] = m.addVar(lb=0, name=f"y_P[{cid}]")
            m.addConstr(y_P[i] == inf['existing'] + y_P_new[i] - y_P_ret[i])

    # ---- Storage energy ----
    y_E_new, y_E = {}, {}
    for i in storage_idx:
        inf = info[i]
        cid = inf['cid']
        tp = inf['params']
        min_dur = tp['min_duration_MWh_per_MW']
        max_dur = tp['max_duration_MWh_per_MW']
        existing_energy = inf['existing'] * min_dur
        max_energy = tp.get('max_energy_capacity_MWh', 0) or 0
        if inf['is_uc']:
            ecs = tp['energy_capacity_size_MWh']
            ub = max_energy / ecs if max_energy > 0 else 0
            y_E_new[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"y_E_new[{cid}]")
            y_E[i] = m.addVar(lb=0, name=f"y_E[{cid}]")
            m.addConstr(y_E[i] == existing_energy + y_E_new[i] * ecs)
        else:
            ub = max_energy if max_energy > 0 else 0
            y_E_new[i] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"y_E_new[{cid}]")
            y_E[i] = m.addVar(lb=0, name=f"y_E[{cid}]")
            m.addConstr(y_E[i] == existing_energy + y_E_new[i])
        m.addConstr(y_E[i] >= min_dur * y_P[i])
        m.addConstr(y_E[i] <= max_dur * y_P[i])

    # ---- Transmission ----
    y_F_new, y_F = {}, {}
    for li, line in enumerate(trans_lines):
        lid = line['line_id']
        ex = line['existing_capacity_MW']
        mx = line['max_capacity_MW']
        ub = (mx - ex) if mx > ex else 0
        y_F_new[li] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"y_F_new[{lid}]")
        y_F[li] = m.addVar(lb=0, name=f"y_F[{lid}]")
        m.addConstr(y_F[li] == ex + y_F_new[li])

    # ---- Operations variables ----
    x_inj = m.addVars(num_clusters, T, lb=0, name="x_inj")

    if storage_idx:
        st_tup = [(i, t) for i in storage_idx for t in range(T)]
        x_wdw = m.addVars(st_tup, lb=0, name="x_wdw")
        x_soc = m.addVars(st_tup, lb=0, name="x_soc")
    else:
        x_wdw, x_soc = {}, {}

    if hydro_idx:
        hy_tup = [(i, t) for i in hydro_idx for t in range(T)]
        x_lvl = m.addVars(hy_tup, lb=0, name="x_lvl")
        x_spill = m.addVars(hy_tup, lb=0, name="x_spill")
    else:
        x_lvl, x_spill = {}, {}

    if uc_idx:
        uc_tup = [(i, t) for i in uc_idx for t in range(T)]
        x_commit = m.addVars(uc_tup, lb=0, name="x_commit")
        x_start = m.addVars(uc_tup, lb=0, name="x_start")
        x_shut = m.addVars(uc_tup, lb=0, name="x_shut")
    else:
        x_commit, x_start, x_shut = {}, {}, {}

    x_flow = m.addVars(num_lines, T, lb=-GRB.INFINITY, name="x_flow") if num_lines > 0 else {}
    x_nse = m.addVars(num_segs, T, lb=0, name="x_nse") if num_segs > 0 else {}

    x_RPS = m.addVars(num_weeks, lb=0, name="x_RPS") if policy['type'] == 'RPS' else None
    x_CO2 = m.addVars(num_weeks, lb=0, name="x_CO2") if policy['type'] == 'CO2' else None

    # ---- Generation limits ----
    for i in uc_idx:
        inf = info[i]
        cs = inf['cap_size']
        tp = inf['params']
        min_out = tp['min_output_frac']
        avail = inf['avail']
        nunits = y_P[i] / cs
        m.addConstrs((x_commit[i, t] <= nunits for t in range(T)), name=f"cc_{i}")
        m.addConstrs((x_inj[i, t] >= min_out * cs * x_commit[i, t] for t in range(T)), name=f"mo_{i}")
        m.addConstrs((x_inj[i, t] <= avail[t] * cs * x_commit[i, t] for t in range(T)), name=f"xo_{i}")

    for i in non_uc_nsh:
        inf = info[i]
        tp = inf['params']
        avail = inf['avail']
        m.addConstrs((x_inj[i, t] <= avail[t] * y_P[i] for t in range(T)), name=f"gl_{i}")
        min_out = tp.get('min_output_frac', 0) or 0
        if min_out > 0:
            m.addConstrs((x_inj[i, t] >= min_out * y_P[i] for t in range(T)), name=f"mr_{i}")
        ru = tp.get('ramp_up_rate_pct_per_hr', None)
        rd = tp.get('ramp_down_rate_pct_per_hr', None)
        if ru is not None:
            m.addConstrs((x_inj[i, t] - x_inj[i, prev_hour(t, hps)] <= ru * y_P[i]
                          for t in range(T)), name=f"ru_{i}")
        if rd is not None:
            m.addConstrs((x_inj[i, prev_hour(t, hps)] - x_inj[i, t] <= rd * y_P[i]
                          for t in range(T)), name=f"rd_{i}")

    for i in storage_idx:
        inf = info[i]
        avail = inf['avail']
        m.addConstrs((x_inj[i, t] <= avail[t] * y_P[i] for t in range(T)), name=f"si_{i}")
        m.addConstrs((x_wdw[i, t] <= avail[t] * y_P[i] for t in range(T)), name=f"sw_{i}")
        m.addConstrs((x_inj[i, t] + x_wdw[i, t] <= y_P[i] for t in range(T)), name=f"sc_{i}")

    for i in hydro_idx:
        inf = info[i]
        tp = inf['params']
        avail = inf['avail']
        min_out = tp['min_output_frac']
        m.addConstrs((x_inj[i, t] <= avail[t] * y_P[i] for t in range(T)), name=f"hl_{i}")
        m.addConstrs((x_inj[i, t] + x_spill[i, t] >= min_out * y_P[i] for t in range(T)), name=f"hm_{i}")

    # ---- UC commitment balance ----
    for i in uc_idx:
        m.addConstrs((x_commit[i, t] - x_commit[i, prev_hour(t, hps)] ==
                      x_start[i, t] - x_shut[i, t] for t in range(T)), name=f"cb_{i}")

    # ---- Min up/down time ----
    idx_cache = {}
    def get_prev_idx(t, L):
        key = (t, L)
        if key in idx_cache:
            return idx_cache[key]
        w = t // hps
        h = t % hps
        base = w * hps
        res = [base + (h - k) % hps for k in range(L)]
        idx_cache[key] = res
        return res

    for i in uc_idx:
        inf = info[i]
        tp = inf['params']
        mut = tp['min_up_time_hr']
        mdt = tp['min_down_time_hr']
        cs = inf['cap_size']
        nunits = y_P[i] / cs
        for t in range(T):
            up_idx = get_prev_idx(t, mut)
            dn_idx = get_prev_idx(t, mdt)
            if mut > 0:
                m.addConstr(x_commit[i, t] >= gp.quicksum(x_start[i, tt] for tt in up_idx))
            if mdt > 0:
                m.addConstr(x_commit[i, t] + gp.quicksum(x_shut[i, tt] for tt in dn_idx) <= nunits)

    # ---- UC ramping ----
    for i in uc_idx:
        inf = info[i]
        tp = inf['params']
        cs = inf['cap_size']
        ru = tp['ramp_up_rate_pct_per_hr']
        rd = tp['ramp_down_rate_pct_per_hr']
        min_out = tp['min_output_frac']
        avail = inf['avail']
        for t in range(T):
            p = prev_hour(t, hps)
            su_ramp = min(avail[t], max(min_out, ru))
            sd_ramp = min(avail[p], max(min_out, rd))
            m.addConstr(x_inj[i, t] - x_inj[i, p] <=
                        ru * cs * x_commit[i, p] + cs * su_ramp * x_start[i, t] - cs * min_out * x_shut[i, t])
            m.addConstr(x_inj[i, p] - x_inj[i, t] <=
                        rd * cs * x_commit[i, t] + cs * sd_ramp * x_shut[i, t] - cs * min_out * x_start[i, t])

    # ---- Storage dynamics ----
    for i in storage_idx:
        inf = info[i]
        tp = inf['params']
        ce = tp['charging_efficiency']
        de = tp['discharging_efficiency']
        sd = tp['self_discharge_rate_pct']
        for t in range(T):
            p = prev_hour(t, hps)
            m.addConstr(x_soc[i, t] == (1 - sd) * x_soc[i, p] + ce * x_wdw[i, t] - (1.0 / de) * x_inj[i, t])
        m.addConstrs((x_soc[i, t] <= y_E[i] for t in range(T)), name=f"soc_{i}")
        for t in range(T):
            p = prev_hour(t, hps)
            m.addConstr(ce * x_wdw[i, t] <= y_E[i] - x_soc[i, p])
            m.addConstr((1.0 / de) * x_inj[i, t] <= x_soc[i, p])

    # ---- Hydro dynamics ----
    for i in hydro_idx:
        inf = info[i]
        tp = inf['params']
        dur = tp['duration_MWh_per_MW']
        infl = hydro_inflow.get(inf['cid'], [0.0] * T)
        for t in range(T):
            p = prev_hour(t, hps)
            m.addConstr(x_lvl[i, t] == x_lvl[i, p] + infl[t] * y_P[i] - x_inj[i, t] - x_spill[i, t])
        m.addConstrs((x_lvl[i, t] <= dur * y_P[i] for t in range(T)), name=f"lvl_{i}")

    # ---- Transmission flow bounds ----
    for li in range(num_lines):
        m.addConstrs((x_flow[li, t] <= y_F[li] for t in range(T)), name=f"fp_{li}")
        m.addConstrs((x_flow[li, t] >= -y_F[li] for t in range(T)), name=f"fn_{li}")

    # ---- NSE bounds ----
    for si, seg in enumerate(consumer_segments):
        z = seg['zone']
        mf = seg['max_nse_fraction']
        dem = demand[z]
        m.addConstrs((x_nse[si, t] <= mf * dem[t] for t in range(T)), name=f"nse_{si}")

    # ---- Demand balance ----
    clusters_by_zone = {z: [] for z in range(num_zones)}
    storage_by_zone = {z: [] for z in range(num_zones)}
    for i, inf in enumerate(info):
        clusters_by_zone[inf['zone']].append(i)
        if inf['is_storage']:
            storage_by_zone[inf['zone']].append(i)
    lines_into = {z: [] for z in range(num_zones)}
    lines_outof = {z: [] for z in range(num_zones)}
    for li, line in enumerate(trans_lines):
        lines_into[line['to_zone']].append(li)
        lines_outof[line['from_zone']].append(li)
    segs_by_zone = {z: [] for z in range(num_zones)}
    for si, seg in enumerate(consumer_segments):
        segs_by_zone[seg['zone']].append(si)

    for z in range(num_zones):
        dem = demand[z]
        cb = clusters_by_zone[z]
        sb = storage_by_zone[z]
        lin = lines_into[z]
        lout = lines_outof[z]
        sg = segs_by_zone[z]
        for t in range(T):
            lhs = gp.LinExpr()
            for i in cb:
                lhs += x_inj[i, t]
            for i in sb:
                lhs -= x_wdw[i, t]
            for li in lin:
                lhs += x_flow[li, t]
            for li in lout:
                lhs -= x_flow[li, t]
            for si in sg:
                lhs += x_nse[si, t]
            m.addConstr(lhs == dem[t], name=f"bal_{z}_{t}")

    # ---- Policy ----
    total_wd = sum(hw[t] * demand[z][t] for t in range(T) for z in range(num_zones))
    if policy['type'] == 'RPS':
        share = policy['RPS_share']
        rps_gen = gp.LinExpr()
        for i, inf in enumerate(info):
            if inf['is_rps']:
                rps_gen += gp.quicksum(hw[t] * x_inj[i, t] for t in range(T))
        slack = gp.quicksum(x_RPS[w] for w in range(num_weeks))
        m.addConstr(rps_gen + slack >= share * total_wd)
    if policy['type'] == 'CO2':
        cap_rate = policy['CO2_cap_tons_per_MWh']
        emissions = gp.LinExpr()
        for i, inf in enumerate(info):
            tp = inf['params']
            co2 = tp.get('CO2_emission_tons_per_MWh', 0) or 0
            if co2 == 0:
                continue
            emissions += co2 * gp.quicksum(hw[t] * x_inj[i, t] for t in range(T))
            if inf['is_storage']:
                emissions += co2 * gp.quicksum(hw[t] * x_wdw[i, t] for t in range(T))
        slack = gp.quicksum(x_CO2[w] for w in range(num_weeks))
        m.addConstr(emissions - slack <= cap_rate * total_wd)

    # ---- Objective ----
    obj = gp.LinExpr()
    for i, inf in enumerate(info):
        tp = inf['params']
        cs = inf['cap_size']
        inv_cost = tp.get('investment_cost_per_MW_yr', 0) or 0
        if inf['is_uc']:
            obj += inv_cost * y_P_new[i] * cs
        else:
            obj += inv_cost * y_P_new[i]
        fom = tp.get('fixed_OM_cost_per_MW_yr', 0) or 0
        obj += fom * y_P[i]

    for i in storage_idx:
        inf = info[i]
        tp = inf['params']
        e_inv = tp.get('energy_investment_cost_per_MWh_yr', 0) or 0
        if inf['is_uc']:
            obj += e_inv * y_E_new[i] * tp['energy_capacity_size_MWh']
        else:
            obj += e_inv * y_E_new[i]
        e_fom = tp.get('energy_fixed_OM_cost_per_MWh_yr', 0) or 0
        obj += e_fom * y_E[i]

    for i in hydro_idx:
        inf = info[i]
        tp = inf['params']
        e_inv = tp.get('energy_investment_cost_per_MWh_yr', 0) or 0
        dur = tp['duration_MWh_per_MW']
        if inf['is_uc']:
            obj += e_inv * dur * y_P_new[i] * inf['cap_size']
        else:
            obj += e_inv * dur * y_P_new[i]
        e_fom = tp.get('energy_fixed_OM_cost_per_MWh_yr', 0) or 0
        obj += e_fom * dur * y_P[i]

    for li, line in enumerate(trans_lines):
        obj += line['investment_cost_per_MW_yr'] * y_F_new[li]

    for i, inf in enumerate(info):
        tp = inf['params']
        vc = tp.get('variable_cost_per_MWh', 0) or 0
        if vc == 0:
            continue
        obj += vc * gp.quicksum(hw[t] * x_inj[i, t] for t in range(T))
        if inf['is_storage']:
            obj += vc * gp.quicksum(hw[t] * x_wdw[i, t] for t in range(T))

    for si, seg in enumerate(consumer_segments):
        cost = seg['cost_per_MWh']
        if cost != 0:
            obj += cost * gp.quicksum(hw[t] * x_nse[si, t] for t in range(T))

    for i in uc_idx:
        tp = info[i]['params']
        suc = tp.get('start_up_cost', 0) or 0
        if suc != 0:
            obj += suc * gp.quicksum(hw[t] * x_start[i, t] for t in range(T))

    if policy['type'] == 'RPS':
        pen = policy['RPS_noncompliance_cost_per_MWh']
        obj += pen * gp.quicksum(x_RPS[w] for w in range(num_weeks))
    if policy['type'] == 'CO2':
        pen = policy['CO2_noncompliance_cost_per_ton']
        obj += pen * gp.quicksum(x_CO2[w] for w in range(num_weeks))

    m.setObjective(obj, GRB.MINIMIZE)

    # ---- Callback for logging incumbents ----
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                objb = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(float(objb))
            except Exception:
                pass
    m.setCallback(cb)

    # ---- Solve ----
    elapsed = time.time() - start
    remaining = max(1, args.time_limit - elapsed - 2)
    m.Params.TimeLimit = remaining
    m.optimize()

    # Fallback: relax integers if no solution found
    if m.SolCount == 0:
        try:
            for v in m.getVars():
                if v.VType in (GRB.INTEGER, GRB.BINARY):
                    v.VType = GRB.CONTINUOUS
            m.Params.TimeLimit = max(1, args.time_limit - (time.time() - start) - 1)
            m.optimize()
        except Exception:
            pass

    # ---- Extract solution ----
    def build_solution():
        sol = {}
        if m.SolCount > 0:
            sol['objective_value'] = float(m.ObjVal)
        else:
            sol['objective_value'] = 0.0

        inv = {}
        for i, inf in enumerate(info):
            cid = inf['cid']
            cs = inf['cap_size']
            nu = y_P_new[i].X if m.SolCount > 0 else 0.0
            ru_ = y_P_ret[i].X if m.SolCount > 0 else 0.0
            if inf['is_uc']:
                new_mw = nu * cs
                ret_mw = ru_ * cs
            else:
                new_mw = nu
                ret_mw = ru_
            total = y_P[i].X if m.SolCount > 0 else inf['existing']
            inv[str(cid)] = {
                'resource_type': inf['rt'],
                'zone': inf['zone'],
                'new_units': float(nu),
                'new_MW': float(new_mw),
                'retired_units': float(ru_),
                'retired_MW': float(ret_mw),
                'total_capacity_MW': float(total),
            }
        sol['investments'] = inv

        sei = {}
        for i in storage_idx:
            inf = info[i]
            cid = inf['cid']
            tp = inf['params']
            nu = y_E_new[i].X if m.SolCount > 0 else 0.0
            if inf['is_uc']:
                new_mh = nu * tp['energy_capacity_size_MWh']
            else:
                new_mh = nu
            te = y_E[i].X if m.SolCount > 0 else (inf['existing'] * tp['min_duration_MWh_per_MW'])
            sei[str(cid)] = {
                'new_units': float(nu),
                'new_MWh': float(new_mh),
                'total_energy_MWh': float(te),
            }
        sol['storage_energy_investments'] = sei

        ti_ = {}
        for li, line in enumerate(trans_lines):
            lid = line['line_id']
            nm = y_F_new[li].X if m.SolCount > 0 else 0.0
            tt = y_F[li].X if m.SolCount > 0 else line['existing_capacity_MW']
            ti_[str(lid)] = {
                'new_MW': float(nm),
                'total_capacity_MW': float(tt),
            }
        sol['transmission_investments'] = ti_

        mv = {}
        tol = 1e-6

        def addv(name, val):
            if val is not None and abs(val) > tol:
                mv[name] = float(val)

        if m.SolCount > 0:
            for i, inf in enumerate(info):
                cid = inf['cid']
                addv(f"y_P_new[{cid}]", y_P_new[i].X)
                addv(f"y_P_ret[{cid}]", y_P_ret[i].X)
                addv(f"y_P[{cid}]", y_P[i].X)
            for i in storage_idx:
                cid = info[i]['cid']
                addv(f"y_E_new[{cid}]", y_E_new[i].X)
                addv(f"y_E[{cid}]", y_E[i].X)
            for li, line in enumerate(trans_lines):
                lid = line['line_id']
                addv(f"y_F_new[{lid}]", y_F_new[li].X)
                addv(f"y_F[{lid}]", y_F[li].X)
            for i in range(num_clusters):
                cid = info[i]['cid']
                for t in range(T):
                    addv(f"x_inj[{cid},{t}]", x_inj[i, t].X)
            for i in storage_idx:
                cid = info[i]['cid']
                for t in range(T):
                    addv(f"x_wdw[{cid},{t}]", x_wdw[i, t].X)
                    addv(f"x_soc[{cid},{t}]", x_soc[i, t].X)
            for i in hydro_idx:
                cid = info[i]['cid']
                for t in range(T):
                    addv(f"x_lvl[{cid},{t}]", x_lvl[i, t].X)
                    addv(f"x_spill[{cid},{t}]", x_spill[i, t].X)
            for li, line in enumerate(trans_lines):
                lid = line['line_id']
                for t in range(T):
                    addv(f"x_flow[{lid},{t}]", x_flow[li, t].X)
            for si, seg in enumerate(consumer_segments):
                sid = seg['segment_id']
                z = seg['zone']
                for t in range(T):
                    addv(f"x_nse[{sid},{z},{t}]", x_nse[si, t].X)
            for i in uc_idx:
                cid = info[i]['cid']
                for t in range(T):
                    addv(f"x_commit[{cid},{t}]", x_commit[i, t].X)
                    addv(f"x_start[{cid},{t}]", x_start[i, t].X)
                    addv(f"x_shut[{cid},{t}]", x_shut[i, t].X)
            if x_RPS is not None:
                for w in range(num_weeks):
                    addv(f"x_RPS[{w}]", x_RPS[w].X)
            if x_CO2 is not None:
                for w in range(num_weeks):
                    addv(f"x_CO2[{w}]", x_CO2[w].X)

        sol['model_variables'] = mv
        sol['model_variable_name_patterns'] = {
            "y_P_new[i], y_P_ret[i], y_P[i]": "generator capacity variables",
            "y_E_new[i], y_E_ret[i], y_E[i]": "storage-energy variables",
            "y_F_new[l], y_F[l]": "transmission variables",
            "x_inj[i,t], x_wdw[i,t], x_soc[i,t]": "generation/storage operations",
            "x_nse[s,z,t]": "non-served energy",
            "x_lvl[i,t], x_spill[i,t]": "hydro operations",
            "x_flow[l,t]": "transmission flow",
            "x_commit[i,t], x_start[i,t], x_shut[i,t]": "unit commitment",
            "x_RPS[w], x_CO2[w]": "policy slack variables"
        }
        return sol

    sol = build_solution()

    if logger and m.SolCount > 0:
        logger.log_solution(float(m.ObjVal), sol)

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)


if __name__ == '__main__':
    main()