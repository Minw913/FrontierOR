import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Extract dimensions
    num_zones = data['problem_dimensions']['num_zones']
    num_weeks = data['problem_dimensions']['num_weeks']
    hours_per_week = data['problem_dimensions']['hours_per_week']
    T = data['problem_dimensions']['total_timesteps']
    G = data['problem_dimensions']['num_generators']
    L = data['problem_dimensions']['num_transmission_lines']
    num_nse = data['problem_dimensions']['num_demand_segments']

    policy = data['policy']
    RPS_enabled = policy['RPS_enabled']
    CO2_enabled = policy.get('CO2_cap_enabled', False)
    RPS_share = policy.get('RPS_share', 0.0)
    RPS_penalty = policy.get('RPS_noncompliance_cost_per_MWh', 1000.0)
    CO2_penalty = policy.get('CO2_noncompliance_cost_per_ton', 100.0)
    CO2_cap_per_MWh = policy.get('CO2_cap_tons_per_MWh', 0.05)

    subperiods = data['subperiods']
    num_subperiods = subperiods['num_subperiods']
    H = subperiods['hours_per_subperiod']
    sub_w = subperiods['subperiod_weights']
    circular = subperiods.get('circular_indexing', True)

    generators = data['generators']
    trans_lines = data['transmission_lines']
    demand = data['demand']
    nse_segments = data['nse_segments']
    avail = data['availability_profiles']

    # Precompute total weighted demand
    total_demand = 0.0
    for z_str, dlist in demand.items():
        for t in range(T):
            w_t = sub_w[t // H] / float(H)
            total_demand += w_t * dlist[t]

    CO2_cap = CO2_cap_per_MWh * total_demand if CO2_enabled else 0.0

    # Create model
    m = gp.Model("PowerExpansion")
    m.Params.TimeLimit = float(args.time_limit)
    m.Params.Threads = 1
    m.Params.MIPGap = 0.01
    m.Params.LogToConsole = 0
    m.Params.Method = 2
    m.Params.Heuristics = 0.2
    m.Params.LazyConstraints = 0

    # Decision variables - Investment
    yP_new = [None] * G
    yP_ret = [None] * G
    P_total = [None] * G
    yE_new = [None] * G
    yE_ret = [None] * G
    E_total = [None] * G

    for g in range(G):
        gen = generators[g]
        cap_size = gen['capacity_size_MW']
        exist_P = gen['existing_capacity_MW']
        max_P = gen['max_capacity_MW']
        is_storage_g = gen.get('is_storage', False)
        can_ret = gen.get('can_retire', True)

        max_new_units = int((max_P - exist_P) / cap_size + 1e-3) if cap_size > 1e-6 else 0
        max_ret_units = int(exist_P / cap_size + 1e-3) if cap_size > 1e-6 else 0

        yP_new[g] = m.addVar(lb=0, ub=max_new_units, vtype=GRB.INTEGER, name=f"yP_new_{g}")
        yP_ret[g] = m.addVar(lb=0, ub=max_ret_units if can_ret else 0.0, vtype=GRB.INTEGER, name=f"yP_ret_{g}")
        P_total[g] = m.addVar(lb=0, ub=max_P, name=f"Ptot_{g}")
        m.addConstr(P_total[g] == exist_P + cap_size * (yP_new[g] - yP_ret[g]), name=f"Pdef_{g}")

        if is_storage_g:
            e_size = gen.get('storage_capacity_size_MWh', 1.0)
            exist_E = gen.get('existing_storage_capacity_MWh', 0.0)
            max_E = gen.get('max_storage_capacity_MWh', 1e9)
            max_new_e = int((max_E - exist_E) / e_size + 1e-3) if e_size > 1e-6 else 0
            max_ret_e = int(exist_E / e_size + 1e-3) if e_size > 1e-6 else 0
            yE_new[g] = m.addVar(lb=0, ub=max_new_e, vtype=GRB.INTEGER, name=f"yE_new_{g}")
            yE_ret[g] = m.addVar(lb=0, ub=max_ret_e if can_ret else 0.0, vtype=GRB.INTEGER, name=f"yE_ret_{g}")
            E_total[g] = m.addVar(lb=0, name=f"Etot_{g}")
            m.addConstr(E_total[g] == exist_E + e_size * (yE_new[g] - yE_ret[g]), name=f"Edef_{g}")
        else:
            E_total[g] = None
            yE_new[g] = None
            yE_ret[g] = None

    # Transmission
    yF_new = [None] * L
    F_total = [None] * L
    for l_idx in range(L):
        line = trans_lines[l_idx]
        exist_F = line['existing_capacity_MW']
        max_new = line['max_new_capacity_MW']
        yF_new[l_idx] = m.addVar(lb=0, ub=max_new, vtype=GRB.CONTINUOUS, name=f"yF_new_{l_idx}")
        F_total[l_idx] = m.addVar(lb=0, name=f"Ftot_{l_idx}")
        m.addConstr(F_total[l_idx] == exist_F + yF_new[l_idx], name=f"Fdef_{l_idx}")

    # Operational variables
    gen_var = [[None] * T for _ in range(G)]
    withdraw = [[None] * T for _ in range(G)]
    flow = [[None] * T for _ in range(L)]
    nse = [[[None] * T for _ in range(num_zones)] for _ in range(num_nse)]
    soc = [[None] * T for _ in range(G)]
    reservoir = [[None] * T for _ in range(G)]
    spill = [[None] * T for _ in range(G)]
    commit = [[None] * T for _ in range(G)]
    startup = [[None] * T for _ in range(G)]
    shutdown = [[None] * T for _ in range(G)]

    for g in range(G):
        gen = generators[g]
        is_uc = gen.get('is_UC', False)
        is_stor = gen.get('is_storage', False)
        is_h = gen.get('is_hydro', False)
        for t in range(T):
            gen_var[g][t] = m.addVar(lb=0, name=f"g_{g}_{t}")
            if is_stor:
                withdraw[g][t] = m.addVar(lb=0, name=f"w_{g}_{t}")
                soc[g][t] = m.addVar(lb=0, name=f"soc_{g}_{t}")
            if is_h:
                reservoir[g][t] = m.addVar(lb=0, name=f"res_{g}_{t}")
                spill[g][t] = m.addVar(lb=0, name=f"sp_{g}_{t}")
            if is_uc:
                commit[g][t] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"on_{g}_{t}")
                startup[g][t] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"su_{g}_{t}")
                shutdown[g][t] = m.addVar(lb=0, vtype=GRB.INTEGER, name=f"sd_{g}_{t}")

    for l in range(L):
        for t in range(T):
            flow[l][t] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"f_{l}_{t}")

    for s in range(num_nse):
        for z in range(num_zones):
            for t in range(T):
                nse[s][z][t] = m.addVar(lb=0, name=f"nse{s}_{z}_{t}")

    rps_slack = m.addVar(lb=0, name="rps_slack") if RPS_enabled else None
    co2_slack = m.addVar(lb=0, name="co2_slack") if CO2_enabled else None

    # Objective components
    fixed_cost = gp.LinExpr()
    var_cost = gp.LinExpr()
    nse_cost = gp.LinExpr()
    startup_cost_expr = gp.LinExpr()
    penalty_cost = gp.LinExpr()

    # Fixed costs
    for g in range(G):
        gen = generators[g]
        cap_size = gen['capacity_size_MW']
        fixed_cost.add(gen['inv_cost_per_MW_yr'] * cap_size * yP_new[g])
        fixed_cost.add(gen['fom_cost_per_MW_yr'] * P_total[g])
        if gen.get('is_storage', False):
            e_size = gen.get('storage_capacity_size_MWh', 1.0)
            fixed_cost.add(gen.get('inv_cost_energy_per_MWh_yr', 0.0) * e_size * (yE_new[g] or 0))
            fixed_cost.add(gen.get('fom_cost_energy_per_MWh_yr', 0.0) * (E_total[g] or 0))
        if gen.get('is_hydro', False):
            dur = gen.get('duration_MWh_per_MW', 1.0)
            fixed_cost.add(gen.get('inv_cost_per_MW_yr', 0.0) * dur * cap_size * yP_new[g])
            fixed_cost.add(gen.get('fom_cost_per_MW_yr', 0.0) * dur * P_total[g])

    for l_idx in range(L):
        fixed_cost.add(trans_lines[l_idx]['inv_cost_per_MW_yr'] * yF_new[l_idx])

    # Variable costs, totals, and hourly weight
    hourly_weight = [sub_w[t // H] / float(H) for t in range(T)]
    total_gen_rps = gp.LinExpr()
    total_co2 = gp.LinExpr()

    for g in range(G):
        gen = generators[g]
        is_uc = gen.get('is_UC', False)
        is_stor = gen.get('is_storage', False)
        is_h = gen.get('is_hydro', False)
        cap_size = gen['capacity_size_MW']
        min_out = gen.get('min_output_frac', 0.0)
        avail_profile = avail[str(gen['id'])]
        is_ts = avail_profile['type'] == 'timeseries'
        a_vals = avail_profile.get('values', [1.0] * T)
        a_const = avail_profile.get('value', 1.0)

        for t in range(T):
            wt = hourly_weight[t]
            avail_t = a_vals[t] if is_ts else a_const
            Pvar = P_total[g]

            var_cost.add(gen['var_cost_per_MWh'] * wt * gen_var[g][t])
            if is_stor:
                var_cost.add(gen['var_cost_per_MWh'] * wt * withdraw[g][t])

            if gen.get('is_RPS_qualifying', False):
                total_gen_rps.add(wt * gen_var[g][t])
            total_co2.add(gen['co2_tons_per_MWh'] * wt * gen_var[g][t])
            if is_stor:
                total_co2.add(gen['co2_tons_per_MWh'] * wt * withdraw[g][t])

            # Capacity & min output
            if is_uc:
                m.addConstr(gen_var[g][t] <= avail_t * cap_size * commit[g][t], name=f"genmaxuc_{g}_{t}")
                m.addConstr(gen_var[g][t] >= min_out * cap_size * commit[g][t], name=f"genminuc_{g}_{t}")
                m.addConstr(commit[g][t] <= (Pvar / cap_size) + 1e-6, name=f"commitmax_{g}_{t}")
                m.addConstr(startup[g][t] <= (Pvar / cap_size) + 1e-6, name=f"startupmax_{g}_{t}")
                m.addConstr(shutdown[g][t] <= (Pvar / cap_size) + 1e-6, name=f"shutdownmax_{g}_{t}")
            else:
                m.addConstr(gen_var[g][t] <= avail_t * Pvar, name=f"genmax_{g}_{t}")
                if not (is_stor or is_h):
                    m.addConstr(gen_var[g][t] >= min_out * Pvar, name=f"genmin_{g}_{t}")

            if is_stor:
                E_cap = E_total[g]
                ch_eff = gen.get('charge_efficiency', 0.95)
                dis_eff = gen.get('discharge_efficiency', 0.95)
                self_d = gen.get('self_discharge_rate', 0.0)
                min_dur = gen.get('min_duration_MWh_per_MW', 0.0)
                max_dur = gen.get('max_duration_MWh_per_MW', 100.0)
                m.addConstr(withdraw[g][t] <= avail_t * Pvar, name=f"wmax_{g}_{t}")
                m.addConstr(gen_var[g][t] + withdraw[g][t] <= Pvar, name=f"powlim_{g}_{t}")
                m.addConstr(ch_eff * withdraw[g][t] <= E_cap - soc[g][t], name=f"chargelim_{g}_{t}")
                m.addConstr(gen_var[g][t] <= dis_eff * soc[g][t], name=f"dischargelim_{g}_{t}")
                m.addConstr(soc[g][t] <= E_cap, name=f"socmax_{g}_{t}")
                m.addConstr(E_cap >= min_dur * Pvar - 1e-4, name=f"mindur_{g}")
                m.addConstr(E_cap <= max_dur * Pvar + 1e-4, name=f"maxdur_{g}")

            if is_h:
                dur = gen.get('duration_MWh_per_MW', 1.0)
                m.addConstr(reservoir[g][t] <= dur * Pvar, name=f"resmax_{g}_{t}")
                m.addConstr(gen_var[g][t] + spill[g][t] >= min_out * Pvar, name=f"hydromin_{g}_{t}")

    # Storage and hydro dynamics + UC logic
    for g in range(G):
        gen = generators[g]
        is_stor = gen.get('is_storage', False)
        is_h = gen.get('is_hydro', False)
        is_uc = gen.get('is_UC', False)
        cap_size = gen['capacity_size_MW']
        min_out = gen.get('min_output_frac', 0.0)
        ramp_up = gen.get('ramp_up_frac_per_hr', 1.0)
        ramp_dn = gen.get('ramp_dn_frac_per_hr', 1.0)
        min_up = gen.get('min_up_time_hr', 1)
        min_dn = gen.get('min_down_time_hr', 1)

        for t in range(T):
            t_prev = t - 1
            if (t % H == 0) and circular:
                t_prev = t + H - 1

            if is_stor:
                ch_eff = gen.get('charge_efficiency', 0.95)
                dis_eff = gen.get('discharge_efficiency', 0.95)
                self_d = gen.get('self_discharge_rate', 0.0)
                m.addConstr(soc[g][t] == (1.0 - self_d) * soc[g][t_prev] +
                            ch_eff * withdraw[g][t_prev] - gen_var[g][t_prev] / dis_eff,
                            name=f"socbal_{g}_{t}")
            if is_h:
                inflow_profile = avail.get(str(gen['id']), {}).get('values', [0.0] * T)
                inflow = inflow_profile[t]
                m.addConstr(reservoir[g][t] == reservoir[g][t_prev] + inflow * P_total[g] -
                            gen_var[g][t_prev] - spill[g][t_prev], name=f"resbal_{g}_{t}")

            if is_uc:
                m.addConstr(commit[g][t] == commit[g][t_prev] + startup[g][t] - shutdown[g][t],
                            name=f"commitbal_{g}_{t}")
                startup_cost_expr.add(gen.get('start_cost_per_unit', 0.0) * hourly_weight[t] * startup[g][t])

            # Ramping (applied to non-storage where possible)
            if not is_stor and ('ramp_up_frac_per_hr' in gen):
                Pvar = P_total[g]
                m.addConstr(gen_var[g][t] - gen_var[g][t_prev] <= ramp_up * Pvar, name=f"rampup_{g}_{t}")
                m.addConstr(gen_var[g][t_prev] - gen_var[g][t] <= ramp_dn * Pvar, name=f"rampdn_{g}_{t}")

    # Power balance
    zone_resources = [[] for _ in range(num_zones)]
    for g_idx, gen in enumerate(generators):
        z = int(gen['zone'])
        if 0 <= z < num_zones:
            zone_resources[z].append(g_idx)

    zone_from = [[] for _ in range(num_zones)]
    zone_to = [[] for _ in range(num_zones)]
    for l_idx, line in enumerate(trans_lines):
        fz = int(line.get('from_zone', 0))
        tz = int(line.get('to_zone', 0))
        if 0 <= fz < num_zones:
            zone_from[fz].append(l_idx)
        if 0 <= tz < num_zones:
            zone_to[tz].append(l_idx)

    for t in range(T):
        wt = hourly_weight[t]
        for z in range(num_zones):
            balance = gp.LinExpr(demand[str(z)][t])
            for g_idx in zone_resources[z]:
                balance.add(-gen_var[g_idx][t])
                if generators[g_idx].get('is_storage', False):
                    balance.add(+withdraw[g_idx][t])
                if generators[g_idx].get('is_hydro', False):
                    balance.add(+spill[g_idx][t])
            for l_idx in zone_from[z]:
                balance.add(+flow[l_idx][t])
            for l_idx in zone_to[z]:
                balance.add(-flow[l_idx][t])
            for s_idx in range(num_nse):
                balance.add(-nse[s_idx][z][t])
            m.addConstr(balance == 0, name=f"bal_z{z}_t{t}")

            # NSE limits
            d_t = demand[str(z)][t]
            for s in range(num_nse):
                seg = nse_segments[s]
                m.addConstr(nse[s][z][t] <= seg['max_frac'] * d_t, name=f"nsemax_{s}_{z}_{t}")
                nse_cost.add(wt * seg['cost_per_MWh'] * nse[s][z][t])

    # Transmission limits
    for l in range(L):
        for t in range(T):
            m.addConstr(flow[l][t] <= F_total[l], name=f"flowmax_{l}_{t}")
            m.addConstr(flow[l][t] >= -F_total[l], name=f"flowmin_{l}_{t}")

    # Policy constraints
    if RPS_enabled and rps_slack is not None:
        m.addConstr(total_gen_rps + rps_slack >= RPS_share * total_demand, name="rps")
        penalty_cost.add(RPS_penalty * rps_slack)
    if CO2_enabled and co2_slack is not None:
        m.addConstr(total_co2 <= CO2_cap + co2_slack, name="co2cap")
        penalty_cost.add(CO2_penalty * co2_slack)

    m.setObjective(fixed_cost + var_cost + nse_cost + startup_cost_expr + penalty_cost, GRB.MINIMIZE)

    # Callback for incumbent logging
    best_obj = [float('inf')]
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.OBJBST)
            if obj_val < best_obj[0] - 1e-3:
                best_obj[0] = obj_val
                if logger:
                    logger.log(float(obj_val))

    m.optimize(callback)

    # Extract solution
    if m.SolCount > 0:
        final_obj = m.ObjVal
        invest_list = []
        for g in range(G):
            gen = generators[g]
            entry = {
                "generator_id": int(gen['id']),
                "resource_type": gen['resource_type'],
                "zone": int(gen['zone']),
                "y_P_new": float(yP_new[g].X),
                "y_P_ret": float(yP_ret[g].X),
                "y_P_total_MW": float(P_total[g].X)
            }
            if gen.get('is_storage', False) and E_total[g] is not None:
                entry["y_E_new"] = float(yE_new[g].X if yE_new[g] is not None else 0.0)
                entry["y_E_ret"] = float(yE_ret[g].X if yE_ret[g] is not None else 0.0)
                entry["y_E_total_MWh"] = float(E_total[g].X)
            invest_list.append(entry)

        trans_list = []
        for l_idx in range(L):
            line = trans_lines[l_idx]
            trans_list.append({
                "line_id": int(line['id']),
                "y_F_new_MW": float(yF_new[l_idx].X),
                "total_capacity_MW": float(F_total[l_idx].X)
            })

        solution = {
            "objective_value": float(final_obj),
            "investment_decisions": invest_list,
            "transmission_decisions": trans_list
        }
    else:
        # Fallback feasible solution (no new builds)
        invest_list = []
        for gen in generators:
            entry = {
                "generator_id": int(gen['id']),
                "resource_type": gen['resource_type'],
                "zone": int(gen['zone']),
                "y_P_new": 0.0,
                "y_P_ret": 0.0,
                "y_P_total_MW": float(gen['existing_capacity_MW'])
            }
            if gen.get('is_storage', False):
                entry["y_E_new"] = 0.0
                entry["y_E_ret"] = 0.0
                entry["y_E_total_MWh"] = float(gen.get('existing_storage_capacity_MWh', 0.0))
            invest_list.append(entry)
        trans_list = []
        for line in trans_lines:
            trans_list.append({
                "line_id": int(line['id']),
                "y_F_new_MW": 0.0,
                "total_capacity_MW": float(line['existing_capacity_MW'])
            })
        solution = {
            "objective_value": 1e12,
            "investment_decisions": invest_list,
            "transmission_decisions": trans_list
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()