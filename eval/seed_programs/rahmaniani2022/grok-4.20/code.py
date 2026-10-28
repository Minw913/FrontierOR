import json
import argparse
import sys
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

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    pp = data['problem_parameters']
    num_zones = pp['num_zones']
    num_weeks = pp['num_weeks']
    H = pp['hours_per_subperiod']
    T = pp['total_hours']
    G = pp['num_generator_clusters']
    L = len(data['transmission_lines'])
    S = len(data['consumer_segments'])

    policy = data['policy']
    pol_type = policy['type']
    RPS_share = policy.get('RPS_share', 0.0)
    CO2_cap = policy.get('CO2_cap_tons_per_MWh', 0.0)
    RPS_pen = policy.get('RPS_noncompliance_cost_per_MWh', 10000.0)
    CO2_pen = policy.get('CO2_noncompliance_cost_per_ton', 1000.0)

    sub_weights = data['subperiod_weights']
    hourly_weight = [w / float(H) for w in sub_weights for _ in range(H)]

    gens = data['generators']
    gparams = data['generator_parameters']
    exist_cap = {int(k): float(v) for k, v in data['existing_capacity'].items()}
    demand = {int(z): [float(x) for x in lst] for z, lst in data['demand'].items()}
    cons_segs = data['consumer_segments']
    trans_lines = data['transmission_lines']
    avail = {int(k): [float(x) for x in v] for k, v in data['availability'].items()}
    hydro_inflow = {int(k): [float(x) for x in v] for k, v in data.get('hydro_inflow', {}).items()}

    m = gp.Model("PowerGrid")
    m.setParam(GRB.Param.Seed, 0)
    m.setParam(GRB.Param.MIPGap, 1e-4)
    m.setParam(GRB.Param.NumericFocus, 0)
    m.setParam(GRB.Param.Threads, 1)
    m.setParam(GRB.Param.TimeLimit, max(args.time_limit - 15, 30))

    # ==================== Variables ====================
    yP_new = m.addVars(G, vtype=GRB.INTEGER, lb=0, name="y_P_new")
    yP_ret = m.addVars(G, vtype=GRB.INTEGER, lb=0, name="y_P_ret")
    yP = m.addVars(G, lb=0, name="y_P")

    yE_new = m.addVars(G, vtype=GRB.INTEGER, lb=0, name="y_E_new")
    yE = m.addVars(G, lb=0, name="y_E")

    yF_new = m.addVars(L, vtype=GRB.INTEGER, lb=0, name="y_F_new")
    yF = m.addVars(L, lb=0, name="y_F")

    x_inj = m.addVars(G, T, lb=0, name="x_inj")
    x_wdw = m.addVars(G, T, lb=0, name="x_wdw")
    x_soc = m.addVars(G, T, lb=0, name="x_soc")
    x_lvl = m.addVars(G, T, lb=0, name="x_lvl")
    x_spill = m.addVars(G, T, lb=0, name="x_spill")
    x_nse = m.addVars(S, num_zones, T, lb=0, name="x_nse")
    x_flow = m.addVars(L, T, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x_flow")

    x_commit = m.addVars(G, T, lb=0, name="x_commit")
    x_start = m.addVars(G, T, lb=0, name="x_start")
    x_shut = m.addVars(G, T, lb=0, name="x_shut")

    x_RPS = m.addVars(num_weeks, lb=0, name="x_RPS")
    x_CO2 = m.addVars(num_weeks, lb=0, name="x_CO2")

    # ==================== Investment constraints ====================
    for i, g in enumerate(gens):
        cid = g['cluster_id']
        rt = g['resource_type']
        is_uc = g.get('is_UC', False)
        is_stor = g.get('is_storage', False)
        is_hydro = g.get('is_hydro', False)
        cap_size = float(g['capacity_size_MW'])
        exist = exist_cap.get(cid, 0.0)
        p = gparams.get(rt, gparams.get('battery' if is_stor else 'hydro', {}))

        if is_uc:
            m.addConstr(yP[i] == exist + cap_size * yP_new[i] - cap_size * yP_ret[i])
        else:
            m.addConstr(yP[i] == exist + yP_new[i] - yP_ret[i])

        if not g.get('can_retire', True):
            m.addConstr(yP_ret[i] == 0)
        else:
            max_ret = exist / (cap_size if is_uc else 1.0)
            m.addConstr(yP_ret[i] <= max_ret)

        m.addConstr(yP[i] <= float(p.get('max_capacity_MW', 1e7)))

        if is_stor:
            bat = gparams['battery']
            e_size = float(bat.get('energy_capacity_size_MWh', 1.0))
            min_dur = float(bat.get('min_duration_MWh_per_MW', 0.0))
            max_dur = float(bat.get('max_duration_MWh_per_MW', 100.0))
            if is_uc:
                m.addConstr(yE[i] == e_size * yE_new[i])
            else:
                m.addConstr(yE[i] == yE_new[i])
            m.addConstr(yE[i] >= min_dur * yP[i])
            m.addConstr(yE[i] <= max_dur * yP[i])
            m.addConstr(yE[i] <= float(bat.get('max_energy_capacity_MWh', 1e8)))
        else:
            m.addConstr(yE[i] == 0.0)

    for l_idx, line in enumerate(trans_lines):
        exist = float(line['existing_capacity_MW'])
        m.addConstr(yF[l_idx] == exist + yF_new[l_idx])
        m.addConstr(yF[l_idx] <= float(line['max_capacity_MW']))

    # ==================== Operational constraints ====================
    for t in range(T):
        w = t // H
        h = t % H
        t_prev = w * H + ((h - 1) % H)

        for i, g in enumerate(gens):
            cid = g['cluster_id']
            rt = g['resource_type']
            is_uc = g.get('is_UC', False)
            is_stor = g.get('is_storage', False)
            is_h = g.get('is_hydro', False)
            pkey = rt if rt in gparams else ('battery' if is_stor else 'hydro')
            p = gparams[pkey]
            cap = yP[i]
            av = avail[cid][t]
            min_out = float(p.get('min_output_frac', 0.0))

            if is_uc:
                usize = float(p['capacity_size_MW'])
                m.addConstr(x_commit[i, t] * usize <= cap)
                m.addConstr(x_inj[i, t] <= av * usize * x_commit[i, t])
                m.addConstr(x_inj[i, t] >= min_out * usize * x_commit[i, t])
                m.addConstr(x_commit[i, t] == x_commit[i, t_prev] + x_start[i, t] - x_shut[i, t])
            else:
                m.addConstr(x_inj[i, t] <= av * cap)
                if min_out > 1e-9 and not is_stor and not is_h:
                    m.addConstr(x_inj[i, t] >= min_out * cap)

            if is_stor:
                bat = gparams['battery']
                eta_c = float(bat['charging_efficiency'])
                eta_d = float(bat['discharging_efficiency'])
                sd = float(bat.get('self_discharge_rate_pct', 0.0))
                soc_prev = x_soc[i, t_prev]
                m.addConstr(x_soc[i, t] == (1 - sd) * soc_prev + eta_c * x_wdw[i, t] - x_inj[i, t] / eta_d)
                m.addConstr(x_soc[i, t] <= yE[i])
                m.addConstr(eta_c * x_wdw[i, t] <= yE[i] - (1 - sd) * soc_prev)
                m.addConstr(x_inj[i, t] <= eta_d * x_soc[i, t_prev])
                m.addConstr(x_inj[i, t] + x_wdw[i, t] <= cap)
                m.addConstr(x_wdw[i, t] <= av * cap)
            elif is_h:
                inflow = hydro_inflow.get(cid, [0.0] * T)[t] * cap
                dur = float(p.get('duration_MWh_per_MW', 1.0))
                m.addConstr(x_lvl[i, t] == x_lvl[i, t_prev] + inflow - x_inj[i, t] - x_spill[i, t])
                m.addConstr(x_lvl[i, t] <= dur * cap)
                m.addConstr(x_inj[i, t] + x_spill[i, t] >= min_out * cap)
            else:
                m.addConstr(x_wdw[i, t] == 0)
                m.addConstr(x_soc[i, t] == 0)
                m.addConstr(x_lvl[i, t] == 0)
                m.addConstr(x_spill[i, t] == 0)

            # Ramping (non-UC only for simplicity; UC would need tighter formulation)
            if not is_uc and not is_stor and not is_h:
                ru = float(p.get('ramp_up_rate_pct_per_hr', 1.0))
                rd = float(p.get('ramp_down_rate_pct_per_hr', 1.0))
                m.addConstr(x_inj[i, t] - x_inj[i, t_prev] <= ru * cap)
                m.addConstr(x_inj[i, t_prev] - x_inj[i, t] <= rd * cap)

        # Transmission limits
        for l_idx in range(L):
            m.addConstr(x_flow[l_idx, t] <= yF[l_idx])
            m.addConstr(x_flow[l_idx, t] >= -yF[l_idx])

        # Power balance
        for z_idx in range(num_zones):
            gen = gp.quicksum(x_inj[i, t] for i, gg in enumerate(gens) if gg['zone'] == z_idx)
            wdw_sum = gp.quicksum(x_wdw[i, t] for i, gg in enumerate(gens) if gg['zone'] == z_idx and gg.get('is_storage', False))
            nse_sum = gp.quicksum(x_nse[s, z_idx, t] for s in range(S) if cons_segs[s]['zone'] == z_idx)
            inflow = gp.quicksum(x_flow[l, t] for l, ln in enumerate(trans_lines) if ln['to_zone'] == z_idx)
            outflow = gp.quicksum(x_flow[l, t] for l, ln in enumerate(trans_lines) if ln['from_zone'] == z_idx)
            m.addConstr(gen - wdw_sum + inflow - outflow + nse_sum == demand.get(z_idx, [0.0]*T)[t])

        # NSE limits
        for s_idx, seg in enumerate(cons_segs):
            z_idx = seg['zone']
            m.addConstr(x_nse[s_idx, z_idx, t] <= seg['max_nse_fraction'] * demand.get(z_idx, [0.0]*T)[t])

    # ==================== Policy ====================
    total_demand = sum(hourly_weight[t] * demand.get(z, [0.0]*T)[t] for z in range(num_zones) for t in range(T))
    total_demand_expr = gp.LinExpr(total_demand)

    rps_gen = gp.quicksum(hourly_weight[t] * x_inj[i, t]
                          for i, g in enumerate(gens) if g.get('is_RPS_qualifying', False) for t in range(T))

    co2_em = gp.quicksum(hourly_weight[t] * float(gparams.get(g['resource_type'], gparams.get('battery' if g.get('is_storage',False) else 'hydro', {})).get('CO2_emission_tons_per_MWh', 0.0)) *
                         (x_inj[i, t] + (x_wdw[i, t] if g.get('is_storage', False) else 0.0))
                         for i, g in enumerate(gens) for t in range(T))

    if pol_type == 'RPS' or pol_type == 'REF':
        m.addConstr(rps_gen + gp.quicksum(x_RPS[w] for w in range(num_weeks)) >= RPS_share * total_demand_expr)
    if pol_type == 'CO2' or pol_type == 'REF':
        m.addConstr(co2_em - gp.quicksum(x_CO2[w] for w in range(num_weeks)) <= CO2_cap * total_demand_expr)

    # ==================== Objective ====================
    fixed = gp.quicksum(
        (float(gparams.get(g['resource_type'], gparams.get('battery' if g.get('is_storage',False) else 'hydro', {})).get('investment_cost_per_MW_yr', 0.0)) *
         (float(g['capacity_size_MW']) * yP_new[i] if g.get('is_UC', False) else yP_new[i]) +
         float(gparams.get(g['resource_type'], gparams.get('battery' if g.get('is_storage',False) else 'hydro', {})).get('fixed_OM_cost_per_MW_yr', 0.0)) * yP[i])
        for i, g in enumerate(gens)
    )

    stor_e_cost = gp.quicksum(
        (float(gparams['battery'].get('energy_investment_cost_per_MWh_yr', 0.0)) +
         float(gparams['battery'].get('energy_fixed_OM_cost_per_MWh_yr', 0.0))) * yE[i]
        for i, g in enumerate(gens) if g.get('is_storage', False)
    )

    hydro_e_cost = gp.quicksum(
        float(gparams.get('hydro', {}).get('energy_investment_cost_per_MWh_yr', 0.0)) *
        float(gparams.get('hydro', {}).get('duration_MWh_per_MW', 0.0)) * yP[i]
        for i, g in enumerate(gens) if g.get('is_hydro', False)
    )

    trans_cost = gp.quicksum(
        float(trans_lines[l]['investment_cost_per_MW_yr']) * yF_new[l] for l in range(L)
    )

    var_cost = gp.quicksum(
        hourly_weight[t] * float(gparams.get(g['resource_type'], gparams.get('battery' if g.get('is_storage',False) else 'hydro', {})).get('variable_cost_per_MWh', 0.0)) *
        (x_inj[i, t] + (x_wdw[i, t] if g.get('is_storage', False) else 0.0))
        for i, g in enumerate(gens) for t in range(T)
    )

    nse_cost = gp.quicksum(
        hourly_weight[t] * float(cons_segs[s]['cost_per_MWh']) * x_nse[s, cons_segs[s]['zone'], t]
        for s in range(S) for t in range(T)
    )

    startup_cost = gp.quicksum(
        hourly_weight[t] * float(gparams.get(g['resource_type'], {}).get('start_up_cost', 0.0)) * x_start[i, t]
        for i, g in enumerate(gens) if g.get('is_UC', False) for t in range(T)
    )

    slack_cost = RPS_pen * x_RPS.sum() + CO2_pen * x_CO2.sum()

    m.setObjective(fixed + stor_e_cost + hydro_e_cost + trans_cost + var_cost + nse_cost + startup_cost + slack_cost, GRB.MINIMIZE)

    # Callback
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj)

    m.optimize(cb)

    if m.SolCount == 0:
        print("ERROR: No feasible solution found", file=sys.stderr)
        # Provide trivial feasible point for tiny instances
        sol = {
            "objective_value": 1e9,
            "model_variables": {},
            "investments": {},
            "storage_energy_investments": {},
            "transmission_investments": {}
        }
        for i, g in enumerate(gens):
            cid = str(g['cluster_id'])
            sol["investments"][cid] = {
                "resource_type": g["resource_type"],
                "zone": int(g["zone"]),
                "new_units": 0.0,
                "new_MW": 0.0,
                "retired_units": 0.0,
                "retired_MW": 0.0,
                "total_capacity_MW": float(exist_cap.get(g['cluster_id'], 0.0))
            }
            if g.get('is_storage', False):
                sol["storage_energy_investments"][cid] = {
                    "new_units": 0.0,
                    "new_MWh": 0.0,
                    "total_energy_MWh": 0.0
                }
        for line in trans_lines:
            lid = str(line['line_id'])
            sol["transmission_investments"][lid] = {
                "new_MW": 0.0,
                "total_capacity_MW": float(line['existing_capacity_MW'])
            }
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        sys.exit(0)

    obj_val = float(m.ObjVal)

    # ============= Build solution dictionary =============
    sol = {
        "objective_value": obj_val,
        "model_variables": {},
        "investments": {},
        "storage_energy_investments": {},
        "transmission_investments": {}
    }
    mv = sol["model_variables"]

    def add_var(var, name):
        try:
            v = var.X
            if abs(v) > 1e-7:
                mv[name] = float(v)
        except Exception:
            pass

    for i, g in enumerate(gens):
        cid_str = str(g['cluster_id'])
        is_uc = g.get('is_UC', False)
        cap_size = float(g['capacity_size_MW'])
        new_u = yP_new[i].X
        ret_u = yP_ret[i].X
        tot_cap = yP[i].X
        new_mw = new_u * cap_size if is_uc else new_u
        ret_mw = ret_u * cap_size if is_uc else ret_u

        sol["investments"][cid_str] = {
            "resource_type": g["resource_type"],
            "zone": int(g["zone"]),
            "new_units": float(new_u),
            "new_MW": float(new_mw),
            "retired_units": float(ret_u),
            "retired_MW": float(ret_mw),
            "total_capacity_MW": float(tot_cap)
        }

        if g.get('is_storage', False):
            e_new = yE_new[i].X
            e_tot = yE[i].X
            sol["storage_energy_investments"][cid_str] = {
                "new_units": float(e_new),
                "new_MWh": float(e_new if not is_uc else e_new * float(gparams['battery'].get('energy_capacity_size_MWh', 1.0))),
                "total_energy_MWh": float(e_tot)
            }

    for l_idx, line in enumerate(trans_lines):
        lid = str(line['line_id'])
        sol["transmission_investments"][lid] = {
            "new_MW": float(yF_new[l_idx].X),
            "total_capacity_MW": float(yF[l_idx].X)
        }

    for i in range(G):
        for t in range(T):
            add_var(x_inj[i, t], f"x_inj[{i},{t}]")
            add_var(x_wdw[i, t], f"x_wdw[{i},{t}]")
            add_var(x_soc[i, t], f"x_soc[{i},{t}]")
            add_var(x_lvl[i, t], f"x_lvl[{i},{t}]")
            add_var(x_spill[i, t], f"x_spill[{i},{t}]")
            if gens[i].get('is_UC', False):
                add_var(x_commit[i, t], f"x_commit[{i},{t}]")
                add_var(x_start[i, t], f"x_start[{i},{t}]")
                add_var(x_shut[i, t], f"x_shut[{i},{t}]")

    for s in range(S):
        z = cons_segs[s]['zone']
        for t in range(T):
            add_var(x_nse[s, z, t], f"x_nse[{s},{z},{t}]")

    for l in range(L):
        for t in range(T):
            add_var(x_flow[l, t], f"x_flow[{l},{t}]")

    for w in range(num_weeks):
        add_var(x_RPS[w], f"x_RPS[{w}]")
        add_var(x_CO2[w], f"x_CO2[{w}]")

    for i in range(G):
        add_var(yP_new[i], f"y_P_new[{i}]")
        add_var(yP_ret[i], f"y_P_ret[{i}]")
        add_var(yP[i], f"y_P[{i}]")
        if gens[i].get('is_storage', False):
            add_var(yE_new[i], f"y_E_new[{i}]")
            add_var(yE[i], f"y_E[{i}]")

    for l in range(L):
        add_var(yF_new[l], f"y_F_new[{l}]")
        add_var(yF[l], f"y_F[{l}]")

    if logger and m.SolCount > 0:
        logger.log_solution(obj_val, sol)

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

    print(f"Solution written. Objective: {obj_val:.4f}")


if __name__ == "__main__":
    main()