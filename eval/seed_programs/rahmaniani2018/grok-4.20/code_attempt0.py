import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

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
    dims = data['problem_dimensions']
    num_zones = dims['num_zones']
    num_weeks = dims['num_weeks']
    hours_per_week = dims['hours_per_week']
    T = dims['total_timesteps']
    G = dims['num_generators']
    Guc = dims['num_UC_generators']
    S = dims['num_storage_resources']
    H = dims['num_hydro_resources']
    L = dims['num_transmission_lines']
    NS = dims['num_demand_segments']

    policy = data['policy']
    RPS_enabled = policy['RPS_enabled']
    CO2_enabled = policy['CO2_cap_enabled']
    RPS_share = policy['RPS_share']
    CO2_cap_tons_per_MWh = policy['CO2_cap_tons_per_MWh']
    RPS_penalty = policy['RPS_noncompliance_cost_per_MWh']
    CO2_penalty = policy['CO2_noncompliance_cost_per_ton']

    subperiods = data['subperiods']
    num_subperiods = subperiods['num_subperiods']
    hours_per_subperiod = subperiods['hours_per_subperiod']
    subperiod_weights = subperiods['subperiod_weights']
    circular = subperiods.get('circular_indexing', True)

    generators = data['generators']
    trans_lines = data['transmission_lines']
    demand = data['demand']
    nse_segments = data['nse_segments']
    avail = data['availability_profiles']

    # Convert demand to list of lists: zone -> timesteps
    demand_data = []
    zone_ids = sorted([int(z) for z in demand.keys()])
    for z in zone_ids:
        demand_data.append(demand[str(z)])
    demand_data = [demand_data[z] for z in range(num_zones)]  # ensure order 0..num_zones-1

    # Precompute total weighted demand
    total_demand = 0.0
    hour_weight = [0.0] * T
    for w in range(num_subperiods):
        wt = subperiod_weights[w]
        for h in range(hours_per_subperiod):
            t = w * hours_per_subperiod + h
            hour_weight[t] = wt / hours_per_subperiod
            for z in range(num_zones):
                total_demand += demand_data[z][t] * hour_weight[t]

    CO2_cap = 0.05 * total_demand if CO2_enabled and CO2_cap_tons_per_MWh is not None else 0.0
    if CO2_enabled and CO2_cap_tons_per_MWh is not None:
        CO2_cap = CO2_cap_tons_per_MWh * total_demand

    # Map generator index to its properties
    gen_list = generators  # already in order 0..G-1 assumed

    # Create model
    m = gp.Model("PowerExpansion")
    m.setParam(GRB.Param.Seed, 0)
    m.setParam(GRB.Param.MIPGap, 1e-4)
    m.setParam(GRB.Param.NumericFocus, 0)
    m.setParam(GRB.Param.Threads, 1)
    m.setParam(GRB.Param.TimeLimit, args.time_limit - 5)  # safety margin

    # ====================== VARIABLES ======================

    # Investment
    yPnew = [m.addVar(vtype=GRB.INTEGER, lb=0, name=f"yPnew_{i}") for i in range(G)]
    yPret = [m.addVar(vtype=GRB.INTEGER, lb=0, name=f"yPret_{i}") for i in range(G)]
    yP = [m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yP_{i}") for i in range(G)]

    yEnew = [m.addVar(vtype=GRB.INTEGER, lb=0, name=f"yEnew_{i}") for i in range(G)]
    yEret = [m.addVar(vtype=GRB.INTEGER, lb=0, name=f"yEret_{i}") for i in range(G)]
    yE = [m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"yE_{i}") for i in range(G)]

    yFnew = [m.addVar(vtype=GRB.INTEGER, lb=0, name=f"yFnew_{l}") for l in range(L)]

    # Operational
    xinj = [[m.addVar(lb=0, name=f"xinj_{i}_{t}") for t in range(T)] for i in range(G)]
    xwdw = [[m.addVar(lb=0, name=f"xwdw_{i}_{t}") for t in range(T)] for i in range(G)]
    xsoc = [[m.addVar(lb=0, name=f"xsoc_{i}_{t}") for t in range(T)] for i in range(G)]
    xhydro = [[m.addVar(lb=0, name=f"xhydro_{i}_{t}") for t in range(T)] for i in range(G)]
    xspill = [[m.addVar(lb=0, name=f"xspill_{i}_{t}") for t in range(T)] for i in range(G)]

    xflow = [[m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xflow_{l}_{t}") for t in range(T)] for l in range(L)]
    xnse = [[[m.addVar(lb=0, name=f"xnse_{s}_{z}_{t}") for t in range(T)] for z in range(num_zones)] for s in range(NS)]

    xcommit = [[m.addVar(vtype=GRB.INTEGER, lb=0, name=f"xcommit_{i}_{t}") for t in range(T)] for i in range(G)]
    xstart = [[m.addVar(vtype=GRB.INTEGER, lb=0, name=f"xstart_{i}_{t}") for t in range(T)] for i in range(G)]
    xshut = [[m.addVar(vtype=GRB.INTEGER, lb=0, name=f"xshut_{i}_{t}") for t in range(T)] for i in range(G)]

    xRPS = [m.addVar(lb=0, name=f"xRPS_{w}") for w in range(num_subperiods)] if RPS_enabled else None
    xCO2 = [m.addVar(lb=0, name=f"xCO2_{w}") for w in range(num_subperiods)] if CO2_enabled else None

    # ====================== CONSTRAINTS ======================

    for i in range(G):
        g = gen_list[i]
        cap_size = g['capacity_size_MW']
        exist_P = g['existing_capacity_MW']
        max_P = g['max_capacity_MW']
        is_storage = g['is_storage']
        is_hydro = g['is_hydro']
        can_retire = g['can_retire']
        is_UC = g['is_UC']

        # Power capacity
        m.addConstr(yP[i] == exist_P + cap_size * (yPnew[i] - yPret[i]))
        m.addConstr(cap_size * yPnew[i] <= max_P - exist_P)
        if can_retire:
            m.addConstr(cap_size * yPret[i] <= exist_P)
        else:
            m.addConstr(yPret[i] == 0)

        if is_storage:
            e_size = g['storage_capacity_size_MWh']
            exist_E = g['existing_storage_capacity_MWh']
            max_E = g['max_storage_capacity_MWh']
            min_dur = g['min_duration_MWh_per_MW']
            max_dur = g['max_duration_MWh_per_MW']
            m.addConstr(yE[i] == exist_E + e_size * (yEnew[i] - yEret[i]))
            m.addConstr(e_size * yEnew[i] <= max_E - exist_E)
            m.addConstr(e_size * yEret[i] <= exist_E)
            m.addConstr(yE[i] >= min_dur * yP[i])
            m.addConstr(yE[i] <= max_dur * yP[i])
        else:
            m.addConstr(yEnew[i] == 0)
            m.addConstr(yEret[i] == 0)
            m.addConstr(yE[i] == 0.0)

        if is_hydro:
            dur = g['duration_MWh_per_MW']
            m.addConstr(yE[i] == dur * yP[i])

    for l_idx, line in enumerate(trans_lines):
        m.addConstr(yFnew[l_idx] <= line['max_new_capacity_MW'])
        # total capacity not needed as variable

    # Power balance
    for t in range(T):
        for z in range(num_zones):
            gen_in_zone = 0.0
            wdw_in_zone = 0.0
            nse_total = 0.0
            for i in range(G):
                if gen_list[i]['zone'] == z:
                    gen_in_zone += xinj[i][t]
                    if gen_list[i]['is_storage']:
                        wdw_in_zone += xwdw[i][t]
            for s in range(NS):
                nse_total += xnse[s][z][t]
            flow_out = 0.0
            flow_in = 0.0
            for l_idx, line in enumerate(trans_lines):
                if line['from_zone'] == z:
                    flow_out += xflow[l_idx][t]
                if line['to_zone'] == z:
                    flow_in += xflow[l_idx][t]
            m.addConstr(gen_in_zone - wdw_in_zone - flow_out + flow_in + nse_total == demand_data[z][t])

    # Capacity limits and availability
    for i in range(G):
        g = gen_list[i]
        is_UC = g['is_UC']
        is_storage = g['is_storage']
        is_hydro = g['is_hydro']
        avail_profile = avail[str(g['id'])]
        if avail_profile['type'] == 'constant':
            avail_val = avail_profile['value']
            avail_ts = [avail_val] * T
        else:
            avail_ts = avail_profile['values']

        for t in range(T):
            P = yP[i]
            if not is_storage and not is_hydro:
                m.addConstr(xinj[i][t] <= avail_ts[t] * P)
                if g.get('min_output_frac', 0.0) > 0:
                    m.addConstr(xinj[i][t] >= g['min_output_frac'] * P)
            elif is_storage:
                m.addConstr(xinj[i][t] <= avail_ts[t] * P)
                m.addConstr(xwdw[i][t] <= avail_ts[t] * P)
                m.addConstr(xinj[i][t] + xwdw[i][t] <= P)
            elif is_hydro:
                m.addConstr(xinj[i][t] + xspill[i][t] <= avail_ts[t] * P)
                min_out = g.get('min_output_frac', 0.0)
                if min_out > 0:
                    m.addConstr(xinj[i][t] + xspill[i][t] >= min_out * P)

        # Storage power-energy linking already in investment

    # Transmission
    for l_idx, line in enumerate(trans_lines):
        total_cap = line['existing_capacity_MW'] + yFnew[l_idx]
        for t in range(T):
            m.addConstr(xflow[l_idx][t] <= total_cap)
            m.addConstr(xflow[l_idx][t] >= -total_cap)

    # NSE limits
    for s_idx, seg in enumerate(nse_segments):
        for z in range(num_zones):
            for t in range(T):
                m.addConstr(xnse[s_idx][z][t] <= seg['max_frac'] * demand_data[z][t])

    # Storage dynamics
    for i in range(G):
        if not gen_list[i]['is_storage']:
            continue
        g = gen_list[i]
        eta_c = g['charge_efficiency']
        eta_d = g['discharge_efficiency']
        sigma = g['self_discharge_rate']
        for w in range(num_subperiods):
            for h in range(hours_per_subperiod):
                t = w * hours_per_subperiod + h
                t_next = w * hours_per_subperiod + ((h + 1) % hours_per_subperiod)
                if not circular and h == hours_per_subperiod - 1:
                    continue  # no wrap if not circular
                m.addConstr(xsoc[i][t_next] == (1 - sigma) * xsoc[i][t] +
                            eta_c * xwdw[i][t] - xinj[i][t] / eta_d)
        # Bounds
        for t in range(T):
            m.addConstr(xsoc[i][t] <= yE[i])
            m.addConstr(xinj[i][t] <= eta_d * xsoc[i][t])  # discharge limit
            m.addConstr(eta_c * xwdw[i][t] <= yE[i] - xsoc[i][t] + 1e-6)  # charge limit approx

    # Hydro dynamics
    for i in range(G):
        if not gen_list[i]['is_hydro']:
            continue
        g = gen_list[i]
        dur = g['duration_MWh_per_MW']
        inflow_profile = g.get('inflow_profile', [0.0]*T)  # assume present in data
        if 'inflow_profile' not in g:
            inflow_profile = [0.0] * T
        else:
            inflow_profile = g['inflow_profile']
        for w in range(num_subperiods):
            for h in range(hours_per_subperiod):
                t = w * hours_per_subperiod + h
                t_next = w * hours_per_subperiod + ((h + 1) % hours_per_subperiod)
                if not circular and h == hours_per_subperiod - 1:
                    continue
                inflow = inflow_profile[t] * yP[i]
                m.addConstr(xhydro[i][t_next] == xhydro[i][t] + inflow - xinj[i][t] - xspill[i][t])
        for t in range(T):
            m.addConstr(xhydro[i][t] <= dur * yP[i])
            m.addConstr(xhydro[i][t] >= 0)

    # Ramping for non-UC, non-storage
    for i in range(G):
        g = gen_list[i]
        if g['is_UC'] or g['is_storage'] or g['is_hydro']:
            continue
        ramp_up = g['ramp_up_frac_per_hr']
        ramp_dn = g['ramp_dn_frac_per_hr']
        for w in range(num_subperiods):
            for h in range(hours_per_subperiod):
                t = w * hours_per_subperiod + h
                t_prev = w * hours_per_subperiod + ((h - 1) % hours_per_subperiod)
                if not circular and h == 0:
                    continue
                m.addConstr(xinj[i][t] - xinj[i][t_prev] <= ramp_up * yP[i])
                m.addConstr(xinj[i][t_prev] - xinj[i][t] <= ramp_dn * yP[i])

    # Unit Commitment
    for i in range(G):
        if not gen_list[i]['is_UC']:
            continue
        g = gen_list[i]
        cap_size = g['capacity_size_MW']
        min_out = g['min_output_frac']
        avail_profile = avail[str(g['id'])]
        if avail_profile['type'] == 'constant':
            avail_ts = [avail_profile['value']] * T
        else:
            avail_ts = avail_profile['values']
        min_up = g['min_up_time_hr']
        min_dn = g['min_down_time_hr']

        for t in range(T):
            m.addConstr(xcommit[i][t] * cap_size <= yP[i])
            m.addConstr(xstart[i][t] * cap_size <= yP[i])
            m.addConstr(xshut[i][t] * cap_size <= yP[i])
            m.addConstr(xinj[i][t] <= xcommit[i][t] * cap_size * avail_ts[t])
            m.addConstr(xinj[i][t] >= xcommit[i][t] * cap_size * min_out)

        # Commitment logic
        for w in range(num_subperiods):
            for h in range(hours_per_subperiod):
                t = w * hours_per_subperiod + h
                t_prev = w * hours_per_subperiod + ((h - 1) % hours_per_subperiod)
                if not circular and h == 0:
                    continue
                m.addConstr(xcommit[i][t] == xcommit[i][t_prev] + xstart[i][t] - xshut[i][t])

        # Min up / down time
        for w in range(num_subperiods):
            for h in range(hours_per_subperiod):
                t = w * hours_per_subperiod + h
                # Min up
                if min_up > 0:
                    sum_starts = 0
                    for k in range(min_up):
                        t_past = w * hours_per_subperiod + ((h - k - 1) % hours_per_subperiod)
                        sum_starts += xstart[i][t_past]
                    m.addConstr(xcommit[i][t] >= sum_starts)
                # Min down
                if min_dn > 0:
                    sum_shuts = 0
                    for k in range(min_dn):
                        t_past = w * hours_per_subperiod + ((h - k - 1) % hours_per_subperiod)
                        sum_shuts += xshut[i][t_past]
                    m.addConstr(xcommit[i][t] + sum_shuts <= yP[i] / cap_size)

        # Tight ramping for UC (simplified linear version - full formulation is very complex)
        # Using basic ramp on committed capacity for tractability within time limits
        ramp_up = g['ramp_up_frac_per_hr']
        ramp_dn = g['ramp_dn_frac_per_hr']
        for w in range(num_subperiods):
            for h in range(hours_per_subperiod):
                t = w * hours_per_subperiod + h
                t_prev = w * hours_per_subperiod + ((h - 1) % hours_per_subperiod)
                if not circular and h == 0:
                    continue
                m.addConstr(xinj[i][t] - xinj[i][t_prev] <= ramp_up * yP[i])
                m.addConstr(xinj[i][t_prev] - xinj[i][t] <= ramp_dn * yP[i])

    # Policy constraints
    if RPS_enabled:
        total_RPS_gen = 0.0
        for i in range(G):
            if gen_list[i]['is_RPS_qualifying']:
                for t in range(T):
                    total_RPS_gen += hour_weight[t] * xinj[i][t]
        m.addConstr(total_RPS_gen + gp.quicksum(xRPS) >= RPS_share * total_demand)

    if CO2_enabled:
        total_CO2 = 0.0
        for i in range(G):
            for t in range(T):
                total_CO2 += hour_weight[t] * gen_list[i]['co2_tons_per_MWh'] * (xinj[i][t] + xwdw[i][t])
        m.addConstr(total_CO2 - gp.quicksum(xCO2) <= CO2_cap)

    # ====================== OBJECTIVE ======================
    obj = 0.0

    # Fixed costs
    for i in range(G):
        g = gen_list[i]
        cap_size = g['capacity_size_MW']
        obj += g['inv_cost_per_MW_yr'] * cap_size * yPnew[i]
        obj += g['fom_cost_per_MW_yr'] * yP[i]
        if g['is_storage']:
            e_size = g['storage_capacity_size_MWh']
            obj += g['inv_cost_energy_per_MWh_yr'] * e_size * yEnew[i]
            obj += g['fom_cost_energy_per_MWh_yr'] * yE[i]
        if g['is_hydro']:
            dur = g['duration_MWh_per_MW']
            obj += g.get('inv_cost_energy_per_MWh_yr', g['inv_cost_per_MW_yr'] * dur) * dur * cap_size * yPnew[i]
            obj += g.get('fom_cost_energy_per_MWh_yr', g['fom_cost_per_MW_yr'] * dur) * dur * yP[i]

    for l_idx, line in enumerate(trans_lines):
        obj += line['inv_cost_per_MW_yr'] * yFnew[l_idx]

    # Variable costs
    for i in range(G):
        g = gen_list[i]
        vc = g['var_cost_per_MWh']
        for t in range(T):
            wt = hour_weight[t]
            obj += wt * vc * xinj[i][t]
            if g['is_storage']:
                obj += wt * vc * xwdw[i][t]

    # NSE costs
    for s_idx, seg in enumerate(nse_segments):
        for z in range(num_zones):
            for t in range(T):
                obj += hour_weight[t] * seg['cost_per_MWh'] * xnse[s_idx][z][t]

    # Startup costs
    for i in range(G):
        if gen_list[i]['is_UC']:
            sc = gen_list[i]['start_cost_per_unit']
            cap_size = gen_list[i]['capacity_size_MW']
            for t in range(T):
                obj += hour_weight[t] * sc * xstart[i][t] * cap_size  # scaled by unit size if cost is per MW

    # Policy noncompliance
    if RPS_enabled:
        for w in range(num_subperiods):
            obj += RPS_penalty * xRPS[w] * subperiod_weights[w] * hours_per_subperiod
    if CO2_enabled:
        for w in range(num_subperiods):
            obj += CO2_penalty * xCO2[w] * subperiod_weights[w] * hours_per_subperiod

    m.setObjective(obj, GRB.MINIMIZE)

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                # Extract solution dict
                sol_dict = {}
                # Investment
                inv_list = []
                for i in range(G):
                    g = gen_list[i]
                    entry = {
                        "generator_id": g['id'],
                        "resource_type": g['resource_type'],
                        "zone": g['zone'],
                        "y_P_new": model.cbGetSolution(yPnew[i]),
                        "y_P_ret": model.cbGetSolution(yPret[i]),
                        "y_P_total_MW": model.cbGetSolution(yP[i])
                    }
                    if g['is_storage'] or g['is_hydro']:
                        entry.update({
                            "y_E_new": model.cbGetSolution(yEnew[i]),
                            "y_E_ret": model.cbGetSolution(yEret[i]),
                            "y_E_total_MWh": model.cbGetSolution(yE[i])
                        })
                    inv_list.append(entry)
                sol_dict["investment_decisions"] = inv_list

                trans_list = []
                for l_idx, line in enumerate(trans_lines):
                    trans_list.append({
                        "line_id": line['id'],
                        "y_F_new_MW": model.cbGetSolution(yFnew[l_idx]),
                        "total_capacity_MW": line['existing_capacity_MW'] + model.cbGetSolution(yFnew[l_idx])
                    })
                sol_dict["transmission_decisions"] = trans_list

                # We do not extract all operational variables for speed (they are huge)
                # But we include key ones for completeness
                sol_dict["model_variables"] = {
                    "yPnew_0": model.cbGetSolution(yPnew[0]) if G > 0 else 0.0,
                    # Only sample; full extraction too slow for callback
                }
                logger.log_solution(obj_val, sol_dict)

    # Optimize
    m.optimize(callback)

    # ====================== EXTRACT SOLUTION ======================
    if m.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
        # Try to get best feasible if any
        if m.SolCount == 0:
            print("No feasible solution found.")
            sys.exit(1)

    obj_value = m.ObjVal if m.ObjVal < GRB.INFINITY else 1e12

    solution = {
        "objective_value": float(obj_value),
        "investment_decisions": [],
        "transmission_decisions": [],
        "model_variables": {}
    }

    # Investment decisions
    for i in range(G):
        g = gen_list[i]
        entry = {
            "generator_id": int(g['id']),
            "resource_type": g['resource_type'],
            "zone": int(g['zone']),
            "y_P_new": float(yPnew[i].X),
            "y_P_ret": float(yPret[i].X),
            "y_P_total_MW": float(yP[i].X)
        }
        if g.get('is_storage', False) or g.get('is_hydro', False):
            entry["y_E_new"] = float(yEnew[i].X)
            entry["y_E_ret"] = float(yEret[i].X)
            entry["y_E_total_MWh"] = float(yE[i].X)
        solution["investment_decisions"].append(entry)

    # Transmission
    for l_idx, line in enumerate(trans_lines):
        solution["transmission_decisions"].append({
            "line_id": int(line['id']),
            "y_F_new_MW": float(yFnew[l_idx].X),
            "total_capacity_MW": float(line['existing_capacity_MW'] + yFnew[l_idx].X)
        })

    # Sparse model variables (only non-zero key ones; full dump would be massive)
    # We add investment and a few operational samples
    for i in range(min(G, 5)):  # limit to avoid huge JSON
        solution["model_variables"][f"yPnew_{i}"] = float(yPnew[i].X)
        solution["model_variables"][f"yP_{i}"] = float(yP[i].X)
        if xinj[i][0].X > 1e-6:
            solution["model_variables"][f"xinj_{i}_0"] = float(xinj[i][0].X)

    for l in range(L):
        solution["model_variables"][f"yFnew_{l}"] = float(yFnew[l].X)

    if RPS_enabled and xRPS:
        solution["model_variables"]["xRPS_0"] = float(xRPS[0].X)
    if CO2_enabled and xCO2:
        solution["model_variables"]["xCO2_0"] = float(xCO2[0].X)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log_solution(obj_value, solution)

if __name__ == "__main__":
    main()