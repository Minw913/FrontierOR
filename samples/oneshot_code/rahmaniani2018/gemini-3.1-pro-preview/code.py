import json
import argparse
import sys
import subprocess

# Ensure gurobipy is installed in the execution environment
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gurobipy"])
    import gurobipy as gp
    from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def get_prev(t, hrs_per_sub, circular):
    sub_start = (t // hrs_per_sub) * hrs_per_sub
    raw_prev = t - 1
    if raw_prev >= sub_start:
        return raw_prev
    if circular:
        return sub_start + (raw_prev - sub_start) % hrs_per_sub
    return None

def get_next(t, hrs_per_sub, circular):
    sub_start = (t // hrs_per_sub) * hrs_per_sub
    raw_next = t + 1
    if raw_next < sub_start + hrs_per_sub:
        return raw_next
    if circular:
        return sub_start + (raw_next - sub_start) % hrs_per_sub
    return None

def get_hist(t, K, hrs_per_sub, circular):
    sub_start = (t // hrs_per_sub) * hrs_per_sub
    idxs = []
    for k in range(K):
        raw_prev = t - k
        if raw_prev >= sub_start:
            idxs.append(raw_prev)
        elif circular:
            idxs.append(sub_start + (raw_prev - sub_start) % hrs_per_sub)
    return idxs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    m = gp.Model()
    m.setParam('TimeLimit', args.time_limit)
    m.setParam('Threads', 1)
    m.setParam('OutputFlag', 1)

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    gens = data['generators']
    lines = data['transmission_lines']
    demand = data['demand']
    zones = list(demand.keys())
    
    generators_ids = [g['id'] for g in gens]
    lines_ids = [l['id'] for l in lines]
    storage_generators = [g['id'] for g in gens if g.get('is_storage', False)]
    hydro_generators = [g['id'] for g in gens if g.get('is_hydro', False)]
    uc_generators = [g['id'] for g in gens if g.get('is_UC', False)]
    
    zone_gens = {z: [] for z in zones}
    zone_storage = {z: [] for z in zones}
    for g in gens:
        z_str = str(g['zone'])
        zone_gens[z_str].append(g['id'])
        if g.get('is_storage', False):
            zone_storage[z_str].append(g['id'])

    lines_from = {z: [] for z in zones}
    lines_to = {z: [] for z in zones}
    for l in lines:
        f_str = str(l['from_zone'])
        t_str = str(l['to_zone'])
        lines_from[f_str].append(l['id'])
        lines_to[t_str].append(l['id'])

    num_subperiods = data['subperiods']['num_subperiods']
    hrs_per_sub = data['subperiods']['hours_per_subperiod']
    subperiod_weights = data['subperiods']['subperiod_weights']
    circular = data['subperiods']['circular_indexing']
    total_timesteps = data['problem_dimensions']['total_timesteps']

    availability_profiles_data = data.get('availability_profiles', {})
    avail = {}
    for g in gens:
        g_id = g['id']
        prof = availability_profiles_data.get(str(g_id))
        if prof is not None:
            if prof['type'] == 'constant':
                avail[g_id] = [float(prof['value'])] * total_timesteps
            else:
                avail[g_id] = [float(v) for v in prof['values']]
        else:
            avail[g_id] = [1.0] * total_timesteps

    y_P_new = m.addVars(generators_ids, vtype=GRB.INTEGER, lb=0, name="y_P_new")
    y_P_ret = m.addVars(generators_ids, vtype=GRB.INTEGER, lb=0, name="y_P_ret")
    y_E_new = m.addVars(storage_generators, vtype=GRB.INTEGER, lb=0, name="y_E_new")
    y_E_ret = m.addVars(storage_generators, vtype=GRB.INTEGER, lb=0, name="y_E_ret")
    y_F_new = m.addVars(lines_ids, vtype=GRB.INTEGER, lb=0, name="y_F_new")

    capacity_P = {}
    for g in gens:
        g_id = g['id']
        size = float(g['capacity_size_MW'])
        existing = float(g['existing_capacity_MW'])
        m.addConstr(y_P_new[g_id] * size <= float(g['max_capacity_MW']))
        m.addConstr(y_P_ret[g_id] * size <= existing)
        if not g.get('can_retire', True):
            y_P_ret[g_id].ub = 0
            
        capacity_P[g_id] = existing + size * (y_P_new[g_id] - y_P_ret[g_id])

    capacity_E = {}
    for g in gens:
        if not g.get('is_storage', False): continue
        g_id = g['id']
        e_size = float(g['storage_capacity_size_MWh'])
        e_exist = float(g['existing_storage_capacity_MWh'])
        m.addConstr(y_E_new[g_id] * e_size <= float(g['max_storage_capacity_MWh']))
        m.addConstr(y_E_ret[g_id] * e_size <= e_exist)
        if not g.get('can_retire', True):
            y_E_ret[g_id].ub = 0
            
        capacity_E[g_id] = e_exist + e_size * (y_E_new[g_id] - y_E_ret[g_id])
        m.addConstr(capacity_E[g_id] >= capacity_P[g_id] * float(g['min_duration_MWh_per_MW']))
        m.addConstr(capacity_E[g_id] <= capacity_P[g_id] * float(g['max_duration_MWh_per_MW']))

    capacity_F = {}
    for l in lines:
        l_id = l['id']
        m.addConstr(y_F_new[l_id] <= float(l['max_new_capacity_MW']))
        capacity_F[l_id] = float(l['existing_capacity_MW']) + y_F_new[l_id]

    Gen = m.addVars(generators_ids, total_timesteps, lb=0, name="Gen")
    Withdraw = m.addVars(storage_generators, total_timesteps, lb=0, name="Withdraw")
    Flow = m.addVars(lines_ids, total_timesteps, lb=-GRB.INFINITY, name="Flow")

    nse_segs = data['nse_segments']
    nse_ids = [s['segment'] for s in nse_segs]
    NSE = m.addVars(nse_ids, zones, total_timesteps, lb=0, name="NSE")

    Committed = m.addVars(uc_generators, total_timesteps, vtype=GRB.INTEGER, lb=0, name="Committed")
    Started = m.addVars(uc_generators, total_timesteps, vtype=GRB.INTEGER, lb=0, name="Started")
    Shut = m.addVars(uc_generators, total_timesteps, vtype=GRB.INTEGER, lb=0, name="Shut")

    SoC = m.addVars(storage_generators, total_timesteps, lb=0, name="SoC")

    Reserve = m.addVars(hydro_generators, total_timesteps, lb=0, name="Reserve")
    Spill = m.addVars(hydro_generators, total_timesteps, lb=0, name="Spill")

    for t in range(total_timesteps):
        for z in zones:
            gen_z = gp.quicksum(Gen[g_id, t] for g_id in zone_gens[z])
            with_z = gp.quicksum(Withdraw[g_id, t] for g_id in zone_storage[z])
            out_f = gp.quicksum(Flow[l_id, t] for l_id in lines_from[z])
            in_f = gp.quicksum(Flow[l_id, t] for l_id in lines_to[z])
            nse_z = gp.quicksum(NSE[k, z, t] for k in nse_ids)
            m.addConstr(gen_z - with_z - out_f + in_f + nse_z == demand[z][t])

    for l_id in lines_ids:
        for t in range(total_timesteps):
            m.addConstr(Flow[l_id, t] <= capacity_F[l_id])
            m.addConstr(Flow[l_id, t] >= -capacity_F[l_id])

    for s in nse_segs:
        k = s['segment']
        mf = float(s['max_frac'])
        for z in zones:
            for t in range(total_timesteps):
                m.addConstr(NSE[k, z, t] <= mf * demand[z][t])

    for g in gens:
        g_id = g['id']
        is_uc = g.get('is_UC', False)
        is_stor = g.get('is_storage', False)
        is_hydro = g.get('is_hydro', False)
        size = float(g['capacity_size_MW'])
        min_out = float(g.get('min_output_frac', 0.0))
        
        for t in range(total_timesteps):
            av_t = avail[g_id][t]
            
            if not is_uc:
                m.addConstr(Gen[g_id, t] <= av_t * capacity_P[g_id])
            else:
                m.addConstr(Gen[g_id, t] <= Committed[g_id, t] * av_t * size)
                m.addConstr(Gen[g_id, t] >= Committed[g_id, t] * min_out * size)
                
            if is_stor:
                m.addConstr(Withdraw[g_id, t] <= av_t * capacity_P[g_id])
                m.addConstr(Gen[g_id, t] + Withdraw[g_id, t] <= capacity_P[g_id])
                
            if (not is_uc) and (not is_stor) and (not is_hydro):
                if min_out > 0:
                    m.addConstr(Gen[g_id, t] >= min_out * capacity_P[g_id])
                    
            if is_hydro:
                if min_out > 0:
                    m.addConstr(Gen[g_id, t] + Spill[g_id, t] >= min_out * capacity_P[g_id])

    for g in gens:
        if not g.get('is_storage', False): continue
        g_id = g['id']
        chg_eff = float(g['charge_efficiency'])
        dis_eff = float(g['discharge_efficiency'])
        mu = float(g['self_discharge_rate'])
        
        for t in range(total_timesteps):
            m.addConstr(Withdraw[g_id, t] * chg_eff <= capacity_E[g_id] - SoC[g_id, t])
            m.addConstr(Gen[g_id, t] / dis_eff <= SoC[g_id, t])
            m.addConstr(SoC[g_id, t] <= capacity_E[g_id])
            nxt = get_next(t, hrs_per_sub, circular)
            if nxt is not None:
                m.addConstr(SoC[g_id, nxt] == SoC[g_id, t] * (1.0 - mu) + Withdraw[g_id, t] * chg_eff - Gen[g_id, t] / dis_eff)

    for g in gens:
        if not g.get('is_hydro', False): continue
        g_id = g['id']
        dur = float(g['duration_MWh_per_MW'])
        for t in range(total_timesteps):
            m.addConstr(Reserve[g_id, t] <= dur * capacity_P[g_id])
            nxt = get_next(t, hrs_per_sub, circular)
            if nxt is not None:
                inflow = avail[g_id][t] * capacity_P[g_id]
                m.addConstr(Reserve[g_id, nxt] == Reserve[g_id, t] + inflow - Gen[g_id, t] - Spill[g_id, t])

    for g in gens:
        if g.get('is_UC', False) or g.get('is_storage', False): continue
        g_id = g['id']
        rup = g.get('ramp_up_frac_per_hr')
        rdn = g.get('ramp_dn_frac_per_hr')
        if rup is None and rdn is None: continue
        rup = float(rup) if rup is not None else 1.0
        rdn = float(rdn) if rdn is not None else 1.0
        for t in range(total_timesteps):
            prv = get_prev(t, hrs_per_sub, circular)
            if prv is not None:
                m.addConstr(Gen[g_id, t] - Gen[g_id, prv] <= rup * capacity_P[g_id])
                m.addConstr(Gen[g_id, prv] - Gen[g_id, t] <= rdn * capacity_P[g_id])

    for g in gens:
        if not g.get('is_UC', False): continue
        g_id = g['id']
        size = float(g['capacity_size_MW'])
        rup = float(g.get('ramp_up_frac_per_hr', 1.0))
        rdn = float(g.get('ramp_dn_frac_per_hr', 1.0))
        min_out = float(g.get('min_output_frac', 0.0))
        up_time = min(int(g['min_up_time_hr']), hrs_per_sub)
        dn_time = min(int(g['min_down_time_hr']), hrs_per_sub)
        
        for t in range(total_timesteps):
            m.addConstr(Committed[g_id, t] * size <= capacity_P[g_id])
            prv = get_prev(t, hrs_per_sub, circular)
            if prv is not None:
                m.addConstr(Committed[g_id, t] - Committed[g_id, prv] == Started[g_id, t] - Shut[g_id, t])
                av_t = avail[g_id][t]
                av_prv = avail[g_id][prv]
                coef_start = min(av_t, max(min_out, rup))
                m.addConstr(Gen[g_id, t] - Gen[g_id, prv] <= size * rup * (Committed[g_id, t] - Started[g_id, t]) + size * coef_start * Started[g_id, t] - size * min_out * Shut[g_id, t])
                coef_shut = min(av_prv, max(min_out, rdn))
                m.addConstr(Gen[g_id, prv] - Gen[g_id, t] <= size * rdn * (Committed[g_id, t] - Started[g_id, t]) + size * coef_shut * Shut[g_id, t] - size * min_out * Started[g_id, t])

            hist_up = get_hist(t, up_time, hrs_per_sub, circular)
            if hist_up:
                m.addConstr(gp.quicksum(Started[g_id, ht] for ht in hist_up) <= Committed[g_id, t])

            hist_dn = get_hist(t, dn_time, hrs_per_sub, circular)
            if hist_dn:
                m.addConstr((Committed[g_id, t] + gp.quicksum(Shut[g_id, ht] for ht in hist_dn)) * size <= capacity_P[g_id])

    rps_slacks = {}
    co2_slacks = {}
    policy = data.get('policy', {})
    if policy.get('RPS_enabled', False):
        for s in range(num_subperiods):
            rps_slacks[s] = m.addVar(lb=0, name=f"rps_slack_{s}")
        total_rps_expr = gp.LinExpr()
        total_dem_expr = gp.LinExpr()
        for s in range(num_subperiods):
            w = subperiod_weights[s] / 168.0
            s_start = s * hrs_per_sub
            for h in range(hrs_per_sub):
                t = s_start + h
                for g in gens:
                    if g.get('is_RPS_qualifying', False):
                        total_rps_expr += w * Gen[g['id'], t]
                for z in zones:
                    total_dem_expr += w * demand[z][t]
        m.addConstr(total_rps_expr + gp.quicksum(rps_slacks.values()) >= float(policy['RPS_share']) * total_dem_expr)

    if policy.get('CO2_cap_enabled', False):
        for s in range(num_subperiods):
            co2_slacks[s] = m.addVar(lb=0, name=f"co2_slack_{s}")
        total_co2_expr = gp.LinExpr()
        total_dem_expr = gp.LinExpr()
        for s in range(num_subperiods):
            w = subperiod_weights[s] / 168.0
            s_start = s * hrs_per_sub
            for h in range(hrs_per_sub):
                t = s_start + h
                for g in gens:
                    co2_fac = float(g.get('co2_tons_per_MWh', 0.0))
                    if co2_fac > 0:
                        total_co2_expr += w * co2_fac * Gen[g['id'], t]
                        if g.get('is_storage', False):
                            total_co2_expr += w * co2_fac * Withdraw[g['id'], t]
                for z in zones:
                    total_dem_expr += w * demand[z][t]
        m.addConstr(total_co2_expr - gp.quicksum(co2_slacks.values()) <= float(policy['CO2_cap_tons_per_MWh']) * total_dem_expr)

    obj = gp.LinExpr()
    
    # Fixed costs
    for g in gens:
        g_id = g['id']
        size = float(g['capacity_size_MW'])
        
        inv_c = float(g.get('inv_cost_per_MW_yr', 0.0))
        if inv_c != 0:
            obj += size * inv_c * y_P_new[g_id]
            
        if g.get('is_hydro', False):
            dur = float(g['duration_MWh_per_MW'])
            e_inv = float(g.get('inv_cost_energy_per_MWh_yr', 0.0))
            if e_inv != 0:
                obj += size * dur * e_inv * y_P_new[g_id]
                
        fom_c = float(g.get('fom_cost_per_MW_yr', 0.0))
        if fom_c != 0:
            obj += fom_c * capacity_P[g_id]
            
        if g.get('is_hydro', False):
            dur = float(g['duration_MWh_per_MW'])
            e_fom = float(g.get('fom_cost_energy_per_MWh_yr', 0.0))
            if e_fom != 0:
                obj += dur * e_fom * capacity_P[g_id]

        if g.get('is_storage', False):
            e_size = float(g['storage_capacity_size_MWh'])
            e_inv = float(g.get('inv_cost_energy_per_MWh_yr', 0.0))
            e_fom = float(g.get('fom_cost_energy_per_MWh_yr', 0.0))
            if e_inv != 0:
                obj += e_size * e_inv * y_E_new[g_id]
            if e_fom != 0:
                obj += e_fom * capacity_E[g_id]

    for l in lines:
        l_id = l['id']
        inv_c = float(l['inv_cost_per_MW_yr'])
        if inv_c != 0:
            obj += inv_c * y_F_new[l_id]

    # Variable operating costs
    var_cost = {g['id']: float(g.get('var_cost_per_MWh', 0.0)) for g in gens}
    start_cost = {g['id']: float(g.get('start_cost_per_unit', 0.0)) for g in gens}
    nse_cost = {s['segment']: float(s['cost_per_MWh']) for s in nse_segs}

    for s in range(num_subperiods):
        w = subperiod_weights[s] / 168.0
        s_start = s * hrs_per_sub
        for h in range(hrs_per_sub):
            t = s_start + h
            for g_id in generators_ids:
                if var_cost[g_id] != 0:
                    obj += w * var_cost[g_id] * Gen[g_id, t]
            for g_id in storage_generators:
                if var_cost[g_id] != 0:
                    obj += w * var_cost[g_id] * Withdraw[g_id, t]
            for g_id in uc_generators:
                if start_cost[g_id] != 0:
                    obj += w * start_cost[g_id] * Started[g_id, t]
            for z in zones:
                for k in nse_ids:
                    if nse_cost[k] != 0:
                        obj += w * nse_cost[k] * NSE[k, z, t]

    if policy.get('RPS_enabled', False):
        pen_rps = float(policy['RPS_noncompliance_cost_per_MWh'])
        if pen_rps != 0:
            for val in rps_slacks.values():
                obj += pen_rps * val

    if policy.get('CO2_cap_enabled', False):
        pen_co2 = float(policy['CO2_noncompliance_cost_per_ton'])
        if pen_co2 != 0:
            for val in co2_slacks.values():
                obj += pen_co2 * val

    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize(cb)

    if m.SolCount > 0:
        inv_decisions = []
        for g in gens:
            g_id = g['id']
            size = float(g['capacity_size_MW'])
            existing = float(g['existing_capacity_MW'])
            ypn = float(round(y_P_new[g_id].X))
            ypr = float(round(y_P_ret[g_id].X))
            dec = {
                "generator_id": g_id,
                "resource_type": g['resource_type'],
                "zone": g['zone'],
                "y_P_new": ypn,
                "y_P_ret": ypr,
                "y_P_total_MW": float(existing + size * (ypn - ypr))
            }
            if g.get('is_storage', False):
                esize = float(g['storage_capacity_size_MWh'])
                eexist = float(g['existing_storage_capacity_MWh'])
                yen = float(round(y_E_new[g_id].X))
                yer = float(round(y_E_ret[g_id].X))
                dec["y_E_new"] = yen
                dec["y_E_ret"] = yer
                dec["y_E_total_MWh"] = float(eexist + esize * (yen - yer))
            inv_decisions.append(dec)

        tx_decisions = []
        for l in lines:
            l_id = l['id']
            yfn = float(round(y_F_new[l_id].X))
            tx_decisions.append({
                "line_id": l_id,
                "y_F_new_MW": yfn,
                "total_capacity_MW": float(l['existing_capacity_MW']) + yfn
            })

        output = {
            "objective_value": m.ObjVal,
            "investment_decisions": inv_decisions,
            "transmission_decisions": tx_decisions
        }

        with open(args.solution_path, 'w') as f:
            json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()