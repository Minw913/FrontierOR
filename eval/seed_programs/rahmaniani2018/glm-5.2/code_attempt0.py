import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = solve(instance, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def solve(instance, time_limit, logger):
    import gurobipy as gp
    from gurobipy import GRB

    dims = instance['problem_dimensions']
    num_zones = dims['num_zones']
    T = dims['total_timesteps']
    num_lines = dims['num_transmission_lines']
    num_segments = dims['num_demand_segments']

    sub = instance['subperiods']
    num_sub = sub['num_subperiods']
    hps = sub['hours_per_subperiod']
    sp_w = sub['subperiod_weights']

    policy = instance['policy']
    rps_en = policy['RPS_enabled']
    rps_share = policy['RPS_share']
    co2_en = policy['CO2_cap_enabled']
    co2_cap_rate = policy.get('CO2_cap_tons_per_MWh')
    rps_pen = policy['RPS_noncompliance_cost_per_MWh']
    co2_pen = policy['CO2_noncompliance_cost_per_ton']

    generators = instance['generators']
    lines = instance['transmission_lines']
    demand = instance['demand']
    nse_segs = instance['nse_segments']
    av_prof = instance['availability_profiles']

    # Hour weights
    hw = []
    for w in range(num_sub):
        for h in range(hps):
            hw.append(sp_w[w] / hps)

    # Availability arrays
    avail = {}
    for g in generators:
        gid = g['id']
        p = av_prof[str(gid)]
        if p['type'] == 'constant':
            avail[gid] = [p['value']] * T
        else:
            avail[gid] = p['values']

    def prev_t(t):
        h = t % hps
        return t + hps - 1 if h == 0 else t - 1

    def sub_of(t):
        return t // hps

    # Total weighted demand
    total_wd = 0.0
    for z in range(num_zones):
        zd = demand[str(z)]
        for t in range(T):
            total_wd += hw[t] * zd[t]

    if co2_en:
        co2_cap = (co2_cap_rate if co2_cap_rate is not None else 0.05) * total_wd
    else:
        co2_cap = float('inf')

    # ID lists
    all_gids = [g['id'] for g in generators]
    stor_gids = [g['id'] for g in generators if g['is_storage']]
    hydro_gids = [g['id'] for g in generators if g['is_hydro']]
    uc_gids = [g['id'] for g in generators if g['is_UC']]
    reg_gids = [g['id'] for g in generators
                if not g['is_UC'] and not g['is_storage'] and not g['is_hydro']]
    line_ids = [l['id'] for l in lines]
    gen_map = {g['id']: g for g in generators}

    # Model
    m = gp.Model('expansion')
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = 1

    # --- Variables ---

    # Investment
    yPnew = {}
    yPret = {}
    yP = {}
    yEnew = {}
    yEret = {}
    yE = {}
    yFnew = {}

    for g in generators:
        gid = g['id']
        cs = g['capacity_size_MW']
        ex = g['existing_capacity_MW']
        mc = g['max_capacity_MW']

        max_new = int((mc - ex) / cs) if cs > 0 else 0
        max_ret = int(ex / cs) if (cs > 0 and g['can_retire']) else 0

        yPnew[gid] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max_new, name=f"yPnew_{gid}")
        yPret[gid] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max_ret, name=f"yPret_{gid}")
        yP[gid] = m.addVar(lb=0, ub=mc, name=f"yP_{gid}")

        if g['is_storage']:
            es = g['storage_capacity_size_MWh']
            eex = g['existing_storage_capacity_MWh']
            emc = g['max_storage_capacity_MWh']
            max_ne = int((emc - eex) / es) if es > 0 else 0
            max_re = int(eex / es) if es > 0 else 0

            yEnew[gid] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max_ne, name=f"yEnew_{gid}")
            yEret[gid] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=max_re, name=f"yEret_{gid}")
            yE[gid] = m.addVar(lb=0, ub=emc, name=f"yE_{gid}")

    for l in lines:
        lid = l['id']
        yFnew[lid] = m.addVar(vtype=GRB.INTEGER, lb=0,
                              ub=l['max_new_capacity_MW'],
                              name=f"yFnew_{lid}")

    # Operational (bulk with addVars)
    xinj = m.addVars(all_gids, range(T), lb=0, name="xinj") if all_gids else {}

    xwdw = m.addVars(stor_gids, range(T), lb=0, name="xwdw") if stor_gids else {}
    xsoc = m.addVars(stor_gids, range(T), lb=0, name="xsoc") if stor_gids else {}

    xhydro = m.addVars(hydro_gids, range(T), lb=0, name="xhydro") if hydro_gids else {}
    xspill = m.addVars(hydro_gids, range(T), lb=0, name="xspill") if hydro_gids else {}

    xflow = m.addVars(line_ids, range(T), lb=-GRB.INFINITY, name="xflow") if line_ids else {}

    # NSE (individual due to per-zone, per-t upper bounds)
    xnse = {}
    for s in range(num_segments):
        seg = nse_segs[s]
        ub_frac = seg['max_frac']
        for z in range(num_zones):
            zd = demand[str(z)]
            for t in range(T):
                xnse[s, z, t] = m.addVar(lb=0, ub=ub_frac * zd[t],
                                         name=f"xnse_{s}_{z}_{t}")

    # UC
    xcommit = m.addVars(uc_gids, range(T), lb=0, name="xcommit") if uc_gids else {}
    xstart = m.addVars(uc_gids, range(T), lb=0, name="xstart") if uc_gids else {}
    xshut = m.addVars(uc_gids, range(T), lb=0, name="xshut") if uc_gids else {}

    # Policy slack
    xRPS = m.addVars(range(num_sub), lb=0, name="xRPS") if rps_en else {}
    xCO2 = m.addVars(range(num_sub), lb=0, name="xCO2") if co2_en else {}

    m.update()

    # --- Constraints ---

    # Investment: yP = existing + cs * (yPnew - yPret)
    for g in generators:
        gid = g['id']
        cs = g['capacity_size_MW']
        ex = g['existing_capacity_MW']
        m.addConstr(yP[gid] == ex + cs * (yPnew[gid] - yPret[gid]))

        if g['is_storage']:
            es = g['storage_capacity_size_MWh']
            eex = g['existing_storage_capacity_MWh']
            m.addConstr(yE[gid] == eex + es * (yEnew[gid] - yEret[gid]))
            min_d = g['min_duration_MWh_per_MW']
            max_d = g['max_duration_MWh_per_MW']
            m.addConstr(yE[gid] >= min_d * yP[gid])
            m.addConstr(yE[gid] <= max_d * yP[gid])

    # Regular generators
    for gid in reg_gids:
        g = gen_map[gid]
        min_out = g['min_output_frac']
        ramp_up = g.get('ramp_up_frac_per_hr', 1.0)
        ramp_dn = g.get('ramp_dn_frac_per_hr', 1.0)

        for t in range(T):
            pt = prev_t(t)
            av = avail[gid][t]
            m.addConstr(xinj[gid, t] <= av * yP[gid])
            if min_out > 0:
                m.addConstr(xinj[gid, t] >= min_out * yP[gid])
            if ramp_up < 1.0:
                m.addConstr(xinj[gid, t] - xinj[gid, pt] <= ramp_up * yP[gid])
            if ramp_dn < 1.0:
                m.addConstr(xinj[gid, pt] - xinj[gid, t] <= ramp_dn * yP[gid])

    # Storage
    for gid in stor_gids:
        g = gen_map[gid]
        ch_eff = g['charge_efficiency']
        dis_eff = g['discharge_efficiency']
        sd = g['self_discharge_rate']

        for t in range(T):
            pt = prev_t(t)
            av = avail[gid][t]

            m.addConstr(xwdw[gid, t] <= av * yP[gid])
            m.addConstr(xinj[gid, t] + xwdw[gid, t] <= yP[gid])
            m.addConstr(xsoc[gid, t] <= yE[gid])
            m.addConstr(ch_eff * xwdw[gid, t] <= yE[gid] - xsoc[gid, pt])
            if dis_eff > 0:
                m.addConstr(xinj[gid, t] <= dis_eff * xsoc[gid, pt])
                m.addConstr(
                    xsoc[gid, t] == (1 - sd) * xsoc[gid, pt]
                    + ch_eff * xwdw[gid, t] - xinj[gid, t] / dis_eff
                )
            else:
                m.addConstr(xinj[gid, t] == 0)
                m.addConstr(
                    xsoc[gid, t] == (1 - sd) * xsoc[gid, pt]
                    + ch_eff * xwdw[gid, t]
                )

    # Hydro
    for gid in hydro_gids:
        g = gen_map[gid]
        duration = g['duration_MWh_per_MW']
        min_out = g['min_output_frac']
        ramp_up = g.get('ramp_up_frac_per_hr', 1.0)
        ramp_dn = g.get('ramp_dn_frac_per_hr', 1.0)

        for t in range(T):
            pt = prev_t(t)
            av = avail[gid][t]
            inflow_expr = av * yP[gid]

            # Generation limited by power capacity (not by inflow)
            m.addConstr(xinj[gid, t] <= yP[gid])
            m.addConstr(xinj[gid, t] + xspill[gid, t] >= min_out * yP[gid])
            m.addConstr(xhydro[gid, t] <= duration * yP[gid])
            m.addConstr(
                xhydro[gid, t] == xhydro[gid, pt] + inflow_expr
                - xinj[gid, t] - xspill[gid, t]
            )
            if ramp_up < 1.0:
                m.addConstr(xinj[gid, t] - xinj[gid, pt] <= ramp_up * yP[gid])
            if ramp_dn < 1.0:
                m.addConstr(xinj[gid, pt] - xinj[gid, t] <= ramp_dn * yP[gid])

    # UC
    for gid in uc_gids:
        g = gen_map[gid]
        cs = g['capacity_size_MW']
        min_out = g['min_output_frac']
        ramp_up = g['ramp_up_frac_per_hr']
        ramp_dn = g['ramp_dn_frac_per_hr']
        min_up = g['min_up_time_hr']
        min_down = g['min_down_time_hr']

        for t in range(T):
            pt = prev_t(t)
            av = avail[gid][t]
            ru_cap = min(av, max(min_out, ramp_up))
            rd_cap = min(av, max(min_out, ramp_dn))

            # Capacity bounds
            m.addConstr(xcommit[gid, t] * cs <= yP[gid])
            m.addConstr(xstart[gid, t] * cs <= yP[gid])
            m.addConstr(xshut[gid, t] * cs <= yP[gid])

            # Generation bounds
            m.addConstr(xinj[gid, t] >= xcommit[gid, t] * min_out * cs)
            m.addConstr(xinj[gid, t] <= xcommit[gid, t] * av * cs)

            # Commitment balance
            m.addConstr(xcommit[gid, t] == xcommit[gid, pt]
                        + xstart[gid, t] - xshut[gid, t])

            # Min up time
            if min_up > 0:
                s_t = sub_of(t)
                h_t = t % hps
                win = [s_t * hps + (h_t - k) % hps for k in range(min_up)]
                m.addConstr(xcommit[gid, t]
                            >= gp.quicksum(xstart[gid, i] for i in win))

            # Min down time
            if min_down > 0:
                s_t = sub_of(t)
                h_t = t % hps
                win = [s_t * hps + (h_t - k) % hps for k in range(min_down)]
                m.addConstr(
                    xcommit[gid, t]
                    + gp.quicksum(xshut[gid, i] for i in win)
                    <= yP[gid] / cs
                )

            # Ramp up
            m.addConstr(
                xinj[gid, t] - xinj[gid, pt]
                <= cs * ramp_up * (xcommit[gid, t] - xstart[gid, t])
                + cs * ru_cap * xstart[gid, t]
                - cs * min_out * xshut[gid, t]
            )

            # Ramp down
            m.addConstr(
                xinj[gid, pt] - xinj[gid, t]
                <= cs * ramp_dn * (xcommit[gid, t] - xstart[gid, t])
                + cs * rd_cap * xshut[gid, t]
                - cs * min_out * xstart[gid, t]
            )

    # Power balance
    for z in range(num_zones):
        z_gens = [g['id'] for g in generators if g['zone'] == z]
        z_stor = [g['id'] for g in generators if g['zone'] == z and g['is_storage']]
        l_from = [l['id'] for l in lines if l['from_zone'] == z]
        l_to = [l['id'] for l in lines if l['to_zone'] == z]
        zd = demand[str(z)]

        for t in range(T):
            lhs = gp.LinExpr()
            for gid in z_gens:
                lhs += xinj[gid, t]
            for gid in z_stor:
                lhs -= xwdw[gid, t]
            for lid in l_from:
                lhs -= xflow[lid, t]
            for lid in l_to:
                lhs += xflow[lid, t]
            for s in range(num_segments):
                lhs += xnse[s, z, t]
            m.addConstr(lhs == zd[t])

    # Transmission capacity
    for l in lines:
        lid = l['id']
        tc = l['existing_capacity_MW'] + yFnew[lid]
        for t in range(T):
            m.addConstr(xflow[lid, t] <= tc)
            m.addConstr(xflow[lid, t] >= -tc)

    # RPS
    if rps_en:
        rps_gen = gp.LinExpr()
        for g in generators:
            if g['is_RPS_qualifying']:
                gid = g['id']
                for t in range(T):
                    rps_gen += hw[t] * xinj[gid, t]
        rps_slack = gp.quicksum(xRPS[w] for w in range(num_sub))
        m.addConstr(rps_gen + rps_slack >= rps_share * total_wd)

    # CO2
    if co2_en:
        co2_total = gp.LinExpr()
        for g in generators:
            gid = g['id']
            co2_rate = g['co2_tons_per_MWh']
            for t in range(T):
                co2_total += hw[t] * co2_rate * xinj[gid, t]
            if g['is_storage']:
                for t in range(T):
                    co2_total += hw[t] * co2_rate * xwdw[gid, t]
        co2_slack = gp.quicksum(xCO2[w] for w in range(num_sub))
        m.addConstr(co2_total - co2_slack <= co2_cap)

    # --- Objective ---
    obj = gp.LinExpr()

    # Fixed costs
    for g in generators:
        gid = g['id']
        cs = g['capacity_size_MW']

        obj += g['inv_cost_per_MW_yr'] * cs * yPnew[gid]
        obj += g['fom_cost_per_MW_yr'] * yP[gid]

        if g['is_hydro']:
            dur = g['duration_MWh_per_MW']
            obj += g.get('inv_cost_energy_per_MWh_yr', 0) * dur * cs * yPnew[gid]
            obj += g.get('fom_cost_energy_per_MWh_yr', 0) * dur * yP[gid]

        if g['is_storage']:
            es = g['storage_capacity_size_MWh']
            obj += g.get('inv_cost_energy_per_MWh_yr', 0) * es * yEnew[gid]
            obj += g.get('fom_cost_energy_per_MWh_yr', 0) * yE[gid]

    for l in lines:
        obj += l['inv_cost_per_MW_yr'] * yFnew[l['id']]

    # Variable costs
    for g in generators:
        gid = g['id']
        vc = g['var_cost_per_MWh']
        obj += gp.quicksum(hw[t] * vc * xinj[gid, t] for t in range(T))
        if g['is_storage']:
            obj += gp.quicksum(hw[t] * vc * xwdw[gid, t] for t in range(T))

    # NSE costs
    for s in range(num_segments):
        cost = nse_segs[s]['cost_per_MWh']
        for z in range(num_zones):
            obj += gp.quicksum(hw[t] * cost * xnse[s, z, t] for t in range(T))

    # Startup costs
    for gid in uc_gids:
        g = gen_map[gid]
        sc = g['start_cost_per_unit']
        obj += gp.quicksum(hw[t] * sc * xstart[gid, t] for t in range(T))

    # Policy penalties
    if rps_en:
        obj += rps_pen * gp.quicksum(xRPS[w] for w in range(num_sub))
    if co2_en:
        obj += co2_pen * gp.quicksum(xCO2[w] for w in range(num_sub))

    m.setObjective(obj, GRB.MINIMIZE)

    # Callback for logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    if logger:
        m.optimize(callback)
    else:
        m.optimize()

    # Fallback to LP relaxation if no MIP solution found
    if m.SolCount == 0:
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = max(10, time_limit // 2)
        for v in m.getVars():
            if v.VType in [GRB.INTEGER, GRB.BINARY]:
                v.VType = GRB.CONTINUOUS
        m.optimize()
        if m.SolCount == 0:
            return {
                "objective_value": 0.0,
                "model_variables": {},
                "investment_decisions": [],
                "transmission_decisions": []
            }

    obj_val = m.ObjVal

    # --- Extract solution ---
    tol = 1e-6
    model_vars = {}

    for g in generators:
        gid = g['id']
        v = yPnew[gid].X
        if abs(v) > tol:
            model_vars[f"yPnew_{gid}"] = int(round(v))
        v = yPret[gid].X
        if abs(v) > tol:
            model_vars[f"yPret_{gid}"] = int(round(v))
        v = yP[gid].X
        if abs(v) > tol:
            model_vars[f"yP_{gid}"] = v
        if g['is_storage']:
            v = yEnew[gid].X
            if abs(v) > tol:
                model_vars[f"yEnew_{gid}"] = int(round(v))
            v = yEret[gid].X
            if abs(v) > tol:
                model_vars[f"yEret_{gid}"] = int(round(v))
            v = yE[gid].X
            if abs(v) > tol:
                model_vars[f"yE_{gid}"] = v

    for lid in line_ids:
        v = yFnew[lid].X
        if abs(v) > tol:
            model_vars[f"yFnew_{lid}"] = int(round(v))

    for gid in all_gids:
        for t in range(T):
            v = xinj[gid, t].X
            if abs(v) > tol:
                model_vars[f"xinj_{gid}_{t}"] = v

    for gid in stor_gids:
        for t in range(T):
            v = xwdw[gid, t].X
            if abs(v) > tol:
                model_vars[f"xwdw_{gid}_{t}"] = v
            v = xsoc[gid, t].X
            if abs(v) > tol:
                model_vars[f"xsoc_{gid}_{t}"] = v

    for gid in hydro_gids:
        for t in range(T):
            v = xhydro[gid, t].X
            if abs(v) > tol:
                model_vars[f"xhydro_{gid}_{t}"] = v

    for lid in line_ids:
        for t in range(T):
            v = xflow[lid, t].X
            if abs(v) > tol:
                model_vars[f"xflow_{lid}_{t}"] = v

    for s in range(num_segments):
        for z in range(num_zones):
            for t in range(T):
                v = xnse[s, z, t].X
                if abs(v) > tol:
                    model_vars[f"xnse_{s}_{z}_{t}"] = v

    for gid in uc_gids:
        for t in range(T):
            v = xcommit[gid, t].X
            if abs(v) > tol:
                model_vars[f"xcommit_{gid}_{t}"] = v
            v = xstart[gid, t].X
            if abs(v) > tol:
                model_vars[f"xstart_{gid}_{t}"] = v
            v = xshut[gid, t].X
            if abs(v) > tol:
                model_vars[f"xshut_{gid}_{t}"] = v

    if rps_en:
        for w in range(num_sub):
            v = xRPS[w].X
            if abs(v) > tol:
                model_vars[f"xRPS_{w}"] = v
    if co2_en:
        for w in range(num_sub):
            v = xCO2[w].X
            if abs(v) > tol:
                model_vars[f"xCO2_{w}"] = v

    # Investment decisions
    inv_decisions = []
    for g in generators:
        gid = g['id']
        entry = {
            "generator_id": gid,
            "resource_type": g['resource_type'],
            "zone": g['zone'],
            "y_P_new": int(round(yPnew[gid].X)),
            "y_P_ret": int(round(yPret[gid].X)),
            "y_P_total_MW": yP[gid].X
        }
        if g['is_storage']:
            entry["y_E_new"] = int(round(yEnew[gid].X))
            entry["y_E_ret"] = int(round(yEret[gid].X))
            entry["y_E_total_MWh"] = yE[gid].X
        inv_decisions.append(entry)

    trans_decisions = []
    for l in lines:
        lid = l['id']
        trans_decisions.append({
            "line_id": lid,
            "y_F_new_MW": int(round(yFnew[lid].X)),
            "total_capacity_MW": l['existing_capacity_MW'] + yFnew[lid].X
        })

    solution = {
        "objective_value": obj_val,
        "model_variables": model_vars,
        "investment_decisions": inv_decisions,
        "transmission_decisions": trans_decisions
    }

    if logger:
        logger.log_solution(obj_val, solution)

    return solution


if __name__ == '__main__':
    main()