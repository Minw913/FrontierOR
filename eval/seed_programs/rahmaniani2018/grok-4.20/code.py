import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
import sys
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
    dims = data['problem_dimensions']
    num_zones = dims['num_zones']
    T = dims['total_timesteps']
    G = dims['num_generators']
    L = dims['num_transmission_lines']
    NS = dims['num_demand_segments']
    hours_per_subperiod = data['subperiods']['hours_per_subperiod']
    num_subperiods = data['subperiods']['num_subperiods']
    subperiod_weights = data['subperiods']['subperiod_weights']
    circular = data['subperiods'].get('circular_indexing', True)

    policy = data['policy']
    RPS_enabled = policy.get('RPS_enabled', False)
    CO2_enabled = policy.get('CO2_cap_enabled', False)
    RPS_share = policy.get('RPS_share', 0.0)
    CO2_cap_tons_per_MWh = policy.get('CO2_cap_tons_per_MWh')
    RPS_penalty = policy.get('RPS_noncompliance_cost_per_MWh', 0.0)
    CO2_penalty = policy.get('CO2_noncompliance_cost_per_ton', 0.0)

    generators = data['generators']
    trans_lines = data['transmission_lines']
    demand = data['demand']
    nse_segments = data['nse_segments']
    availability = data['availability_profiles']

    # Demand as list of lists: zones x timesteps. Zone keys are strings.
    zone_keys = [str(z) for z in range(num_zones)]
    demand_data = []
    for k in zone_keys:
        if k in demand:
            demand_data.append(demand[k])
        elif str(k) in demand:
            demand_data.append(demand[str(k)])
        else:
            # Fallback: try integer key if any
            found = False
            for dk in demand.keys():
                if int(dk) == int(k):
                    demand_data.append(demand[dk])
                    found = True
                    break
            if not found:
                demand_data.append([0.0] * T)

    # Hour weights and total demand
    hour_weight = [0.0] * T
    total_demand = 0.0
    for w in range(num_subperiods):
        wt = subperiod_weights[w]
        for h in range(hours_per_subperiod):
            t = w * hours_per_subperiod + h
            hour_weight[t] = wt / hours_per_subperiod
            for z in range(num_zones):
                total_demand += demand_data[z][t] * hour_weight[t]

    CO2_cap = 0.0
    if CO2_enabled and CO2_cap_tons_per_MWh is not None:
        CO2_cap = CO2_cap_tons_per_MWh * total_demand

    # Model
    m = gp.Model("PowerSystemExpansion")
    m.setParam(GRB.Param.Seed, 0)
    m.setParam(GRB.Param.MIPGap, 1e-4)
    m.setParam(GRB.Param.NumericFocus, 0)
    m.setParam(GRB.Param.Threads, 1)
    m.setParam(GRB.Param.TimeLimit, max(10, args.time_limit - 10))

    # ====================== VARIABLES ======================
    yPnew = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name=f"yPnew_{i}") for i in range(G)]
    yPret = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name=f"yPret_{i}") for i in range(G)]
    yP = [m.addVar(lb=0, name=f"yP_{i}") for i in range(G)]

    yEnew = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name=f"yEnew_{i}") for i in range(G)]
    yEret = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name=f"yEret_{i}") for i in range(G)]
    yE = [m.addVar(lb=0, name=f"yE_{i}") for i in range(G)]

    yFnew = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=GRB.INFINITY, name=f"yFnew_{l}") for l in range(L)]

    xinj = [[m.addVar(lb=0, name=f"xinj_{i}_{t}") for t in range(T)] for i in range(G)]
    xwdw = [[m.addVar(lb=0, name=f"xwdw_{i}_{t}") for t in range(T)] for i in range(G)]
    xsoc = [[m.addVar(lb=0, name=f"xsoc_{i}_{t}") for t in range(T)] for i in range(G)]
    xhydro = [[m.addVar(lb=0, name=f"xhydro_{i}_{t}") for t in range(T)] for i in range(G)]
    xspill = [[m.addVar(lb=0, name=f"xspill_{i}_{t}") for t in range(T)] for i in range(G)]

    xflow = [[m.addVar(lb=-GRB.INFINITY, name=f"xflow_{l}_{t}") for t in range(T)] for l in range(L)]
    xnse = [[[m.addVar(lb=0, name=f"xnse_{s}_{z}_{t}") for t in range(T)]
             for z in range(num_zones)] for s in range(NS)]

    xcommit = [[m.addVar(vtype=GRB.INTEGER, lb=0, name=f"xcommit_{i}_{t}") for t in range(T)] for i in range(G)]
    xstart = [[m.addVar(vtype=GRB.INTEGER, lb=0, name=f"xstart_{i}_{t}") for t in range(T)] for i in range(G)]
    xshut = [[m.addVar(vtype=GRB.INTEGER, lb=0, name=f"xshut_{i}_{t}") for t in range(T)] for i in range(G)]

    xRPS = [m.addVar(lb=0, name=f"xRPS_{w}") for w in range(num_subperiods)] if RPS_enabled else None
    xCO2 = [m.addVar(lb=0, name=f"xCO2_{w}") for w in range(num_subperiods)] if CO2_enabled else None

    # ====================== INVESTMENT CONSTRAINTS ======================
    for i, g in enumerate(generators):
        cap_size = g['capacity_size_MW']
        exist_P = g['existing_capacity_MW']
        max_P = g['max_capacity_MW']
        is_storage = g.get('is_storage', False)
        is_hydro = g.get('is_hydro', False)
        can_retire = g.get('can_retire', True)

        m.addConstr(yP[i] == exist_P + cap_size * (yPnew[i] - yPret[i]), name=f"Pbalance_{i}")
        m.addConstr(cap_size * yPnew[i] <= max_P - exist_P + 1e-6, name=f"Pnewlim_{i}")
        if can_retire:
            m.addConstr(cap_size * yPret[i] <= exist_P + 1e-6, name=f"Pretlim_{i}")
        else:
            m.addConstr(yPret[i] == 0, name=f"Pno_retire_{i}")

        if is_storage or is_hydro:
            e_size = g.get('storage_capacity_size_MWh', 1.0)
            exist_E = g.get('existing_storage_capacity_MWh', 0.0)
            max_E = g.get('max_storage_capacity_MWh', 1e6)
            m.addConstr(yE[i] == exist_E + e_size * (yEnew[i] - yEret[i]), name=f"Ebalance_{i}")
            m.addConstr(e_size * yEnew[i] <= max_E - exist_E + 1e-6, name=f"Enewlim_{i}")
            m.addConstr(e_size * yEret[i] <= exist_E + 1e-6, name=f"Eretlim_{i}")

            if is_storage and not is_hydro:
                min_dur = g.get('min_duration_MWh_per_MW', 0.0)
                max_dur = g.get('max_duration_MWh_per_MW', 100.0)
                m.addConstr(yE[i] >= min_dur * yP[i], name=f"mindur_{i}")
                m.addConstr(yE[i] <= max_dur * yP[i], name=f"maxdur_{i}")
        else:
            m.addConstr(yEnew[i] == 0, name=f"Enew0_{i}")
            m.addConstr(yEret[i] == 0, name=f"Eret0_{i}")
            m.addConstr(yE[i] == 0, name=f"Ezero_{i}")

        if is_hydro:
            dur = g.get('duration_MWh_per_MW', 1.0)
            m.addConstr(yE[i] == dur * yP[i], name=f"hydro_dur_{i}")

    for l_idx, line in enumerate(trans_lines):
        m.addConstr(yFnew[l_idx] <= line['max_new_capacity_MW'], name=f"Fnewlim_{l_idx}")

    # ====================== POWER BALANCE ======================
    for t in range(T):
        for z in range(num_zones):
            inj = gp.quicksum(xinj[i][t] for i in range(G) if generators[i]['zone'] == z)
            wdw = gp.quicksum(xwdw[i][t] for i in range(G) if generators[i].get('is_storage', False) and generators[i]['zone'] == z)
            nse = gp.quicksum(xnse[s][z][t] for s in range(NS))
            outflow = gp.quicksum(xflow[l][t] for l in range(L) if trans_lines[l]['from_zone'] == z)
            inflow = gp.quicksum(xflow[l][t] for l in range(L) if trans_lines[l]['to_zone'] == z)
            m.addConstr(inj - wdw - outflow + inflow + nse == demand_data[z][t], name=f"balance_z{z}_t{t}")

    # ====================== RESOURCE CONSTRAINTS ======================
    for i, g in enumerate(generators):
        is_UC = g.get('is_UC', False)
        is_storage = g.get('is_storage', False)
        is_hydro = g.get('is_hydro', False)
        avail_info = availability.get(str(g['id']), availability.get(g['id'], {}))
        if avail_info.get('type') == 'constant':
            avail_ts = [avail_info.get('value', 1.0)] * T
        else:
            avail_ts = avail_info.get('values', [1.0] * T)

        P = yP[i]
        for t in range(T):
            if not is_storage and not is_hydro:
                m.addConstr(xinj[i][t] <= avail_ts[t] * P, name=f"avail_{i}_{t}")
                m.addConstr(xinj[i][t] >= g.get('min_output_frac', 0.0) * P, name=f"minout_{i}_{t}")
            elif is_storage:
                m.addConstr(xinj[i][t] <= avail_ts[t] * P)
                m.addConstr(xwdw[i][t] <= avail_ts[t] * P)
                m.addConstr(xinj[i][t] + xwdw[i][t] <= P)
            else:  # hydro
                m.addConstr(xinj[i][t] + xspill[i][t] <= avail_ts[t] * P)
                m.addConstr(xinj[i][t] + xspill[i][t] >= g.get('min_output_frac', 0.0) * P)

        # Transmission limits
        for l_idx, line in enumerate(trans_lines):
            total_cap = line['existing_capacity_MW'] + yFnew[l_idx]
            for t_idx in range(T):
                m.addConstr(xflow[l_idx][t_idx] <= total_cap, name=f"flow_up_{l_idx}_{t_idx}")
                m.addConstr(xflow[l_idx][t_idx] >= -total_cap, name=f"flow_dn_{l_idx}_{t_idx}")

    # NSE limits
    for s_idx, seg in enumerate(nse_segments):
        for z in range(num_zones):
            for t in range(T):
                m.addConstr(xnse[s_idx][z][t] <= seg['max_frac'] * demand_data[z][t], name=f"nse_lim_{s_idx}_{z}_{t}")

    # ====================== STORAGE & HYDRO DYNAMICS ======================
    for i, g in enumerate(generators):
        if g.get('is_storage', False):
            eta_c = g['charge_efficiency']
            eta_d = g['discharge_efficiency']
            sigma = g['self_discharge_rate']
            for w in range(num_subperiods):
                for h_idx in range(hours_per_subperiod):
                    t = w * hours_per_subperiod + h_idx
                    t_next = w * hours_per_subperiod + ((h_idx + 1) % hours_per_subperiod)
                    if not circular and h_idx == hours_per_subperiod - 1:
                        continue
                    m.addConstr(
                        xsoc[i][t_next] == (1 - sigma) * xsoc[i][t] + eta_c * xwdw[i][t] - xinj[i][t] / eta_d,
                        name=f"soc_{i}_{t}"
                    )
            for t in range(T):
                m.addConstr(xsoc[i][t] <= yE[i])
                m.addConstr(xinj[i][t] <= eta_d * xsoc[i][t] + 1e-6)
                m.addConstr(eta_c * xwdw[i][t] <= yE[i] - xsoc[i][t] + 1e-6)

        if g.get('is_hydro', False):
            dur = g.get('duration_MWh_per_MW', 1.0)
            inflow_key = str(g['id'])
            inflow = g.get('inflow_profile', [0.0] * T)
            for w in range(num_subperiods):
                for h_idx in range(hours_per_subperiod):
                    t = w * hours_per_subperiod + h_idx
                    t_next = w * hours_per_subperiod + ((h_idx + 1) % hours_per_subperiod)
                    if not circular and h_idx == hours_per_subperiod - 1:
                        continue
                    inflow_val = inflow[t] if isinstance(inflow, list) and len(inflow) > t else 0.0
                    m.addConstr(
                        xhydro[i][t_next] == xhydro[i][t] + inflow_val * yP[i] - xinj[i][t] - xspill[i][t],
                        name=f"hydro_{i}_{t}"
                    )
            for t in range(T):
                m.addConstr(xhydro[i][t] <= dur * yP[i])
                m.addConstr(xhydro[i][t] >= 0)

    # ====================== RAMPING (simplified for speed) ======================
    for i, g in enumerate(generators):
        if g.get('is_UC', False) or g.get('is_storage', False):
            continue
        ru = g.get('ramp_up_frac_per_hr', 1.0)
        rd = g.get('ramp_dn_frac_per_hr', 1.0)
        for w in range(num_subperiods):
            for h in range(hours_per_subperiod):
                t = w * hours_per_subperiod + h
                t_prev = w * hours_per_subperiod + ((h - 1) % hours_per_subperiod)
                if not circular and h == 0:
                    continue
                m.addConstr(xinj[i][t] - xinj[i][t_prev] <= ru * yP[i])
                m.addConstr(xinj[i][t_prev] - xinj[i][t] <= rd * yP[i])

    # ====================== UNIT COMMITMENT (basic) ======================
    for i, g in enumerate(generators):
        if not g.get('is_UC', False):
            continue
        cap_size = g['capacity_size_MW']
        min_out = g.get('min_output_frac', 0.0)
        avail_info = availability.get(str(g['id']), {})
        if avail_info.get('type') == 'constant':
            avail_ts = [avail_info.get('value', 1.0)] * T
        else:
            avail_ts = avail_info.get('values', [1.0] * T)

        for t in range(T):
            m.addConstr(xcommit[i][t] * cap_size <= yP[i])
            m.addConstr(xinj[i][t] <= xcommit[i][t] * cap_size * avail_ts[t])
            m.addConstr(xinj[i][t] >= xcommit[i][t] * cap_size * min_out)

        for w in range(num_subperiods):
            for h in range(hours_per_subperiod):
                t = w * hours_per_subperiod + h
                t_prev = w * hours_per_subperiod + ((h - 1) % hours_per_subperiod)
                if not circular and h == 0:
                    continue
                m.addConstr(xcommit[i][t] == xcommit[i][t_prev] + xstart[i][t] - xshut[i][t])

    # ====================== POLICY CONSTRAINTS ======================
    if RPS_enabled and xRPS is not None:
        rps_gen = gp.quicksum(hour_weight[t] * xinj[i][t] for i in range(G)
                              if generators[i].get('is_RPS_qualifying', False) for t in range(T))
        m.addConstr(rps_gen + gp.quicksum(xRPS) >= RPS_share * total_demand)

    if CO2_enabled and xCO2 is not None:
        co2_em = gp.quicksum(hour_weight[t] * generators[i].get('co2_tons_per_MWh', 0.0) * (xinj[i][t] + xwdw[i][t])
                             for i in range(G) for t in range(T))
        m.addConstr(co2_em - gp.quicksum(xCO2) <= CO2_cap)

    # ====================== OBJECTIVE ======================
    obj = 0.0

    # Investment + Fixed O&M
    for i, g in enumerate(generators):
        cap_size = g['capacity_size_MW']
        obj += g.get('inv_cost_per_MW_yr', 0.0) * cap_size * yPnew[i]
        obj += g.get('fom_cost_per_MW_yr', 0.0) * yP[i]

        if g.get('is_storage', False) or g.get('is_hydro', False):
            e_size = g.get('storage_capacity_size_MWh', 1.0)
            obj += g.get('inv_cost_energy_per_MWh_yr', 0.0) * e_size * yEnew[i]
            obj += g.get('fom_cost_energy_per_MWh_yr', 0.0) * yE[i]

    for l, line in enumerate(trans_lines):
        obj += line.get('inv_cost_per_MW_yr', 0.0) * yFnew[l]

    # Variable costs
    for i, g in enumerate(generators):
        vc = g.get('var_cost_per_MWh', 0.0)
        for t in range(T):
            obj += hour_weight[t] * vc * (xinj[i][t] + xwdw[i][t])

    # NSE
    for s, seg in enumerate(nse_segments):
        for z in range(num_zones):
            for t in range(T):
                obj += hour_weight[t] * seg.get('cost_per_MWh', 1000.0) * xnse[s][z][t]

    # Startup
    for i, g in enumerate(generators):
        if g.get('is_UC', False):
            sc = g.get('start_cost_per_unit', 0.0)
            for t in range(T):
                obj += hour_weight[t] * sc * xstart[i][t]

    # Policy penalties
    if RPS_enabled and xRPS is not None:
        for w_idx in range(num_subperiods):
            obj += RPS_penalty * xRPS[w_idx] * subperiod_weights[w_idx] * hours_per_subperiod
    if CO2_enabled and xCO2 is not None:
        for w_idx in range(num_subperiods):
            obj += CO2_penalty * xCO2[w_idx] * subperiod_weights[w_idx] * hours_per_subperiod

    m.setObjective(obj, GRB.MINIMIZE)

    # Callback for logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger and obj_val < GRB.INFINITY:
                try:
                    sol_dict = {
                        "investment_decisions": [],
                        "transmission_decisions": [],
                        "model_variables": {}
                    }
                    for i_g, g in enumerate(generators):
                        entry = {
                            "generator_id": int(g['id']),
                            "resource_type": g['resource_type'],
                            "zone": int(g['zone']),
                            "y_P_new": float(model.cbGetSolution(yPnew[i_g])),
                            "y_P_ret": float(model.cbGetSolution(yPret[i_g])),
                            "y_P_total_MW": float(model.cbGetSolution(yP[i_g]))
                        }
                        if g.get('is_storage', False) or g.get('is_hydro', False):
                            entry.update({
                                "y_E_new": float(model.cbGetSolution(yEnew[i_g])),
                                "y_E_ret": float(model.cbGetSolution(yEret[i_g])),
                                "y_E_total_MWh": float(model.cbGetSolution(yE[i_g]))
                            })
                        sol_dict["investment_decisions"].append(entry)

                    for l_idx, line in enumerate(trans_lines):
                        sol_dict["transmission_decisions"].append({
                            "line_id": int(line['id']),
                            "y_F_new_MW": float(model.cbGetSolution(yFnew[l_idx])),
                            "total_capacity_MW": float(line['existing_capacity_MW'] + model.cbGetSolution(yFnew[l_idx]))
                        })

                    logger.log_solution(obj_val, sol_dict)
                except Exception:
                    if logger:
                        logger.log(obj_val)

    # Optimize
    m.optimize(callback)

    # ====================== WRITE SOLUTION ======================
    final_obj = m.ObjVal if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT) and m.SolCount > 0 else 1e10

    solution = {
        "objective_value": float(final_obj),
        "investment_decisions": [],
        "transmission_decisions": [],
        "model_variables": {}
    }

    if m.SolCount > 0:
        for i, g in enumerate(generators):
            entry = {
                "generator_id": int(g.get('id', i)),
                "resource_type": g.get('resource_type', 'unknown'),
                "zone": int(g.get('zone', 0)),
                "y_P_new": float(yPnew[i].X),
                "y_P_ret": float(yPret[i].X),
                "y_P_total_MW": float(yP[i].X)
            }
            if g.get('is_storage', False) or g.get('is_hydro', False):
                entry.update({
                    "y_E_new": float(yEnew[i].X),
                    "y_E_ret": float(yEret[i].X),
                    "y_E_total_MWh": float(yE[i].X)
                })
            solution["investment_decisions"].append(entry)

        for l_idx, line in enumerate(trans_lines):
            solution["transmission_decisions"].append({
                "line_id": int(line.get('id', l_idx)),
                "y_F_new_MW": float(yFnew[l_idx].X),
                "total_capacity_MW": float(line.get('existing_capacity_MW', 0.0) + yFnew[l_idx].X)
            })

        # Sparse model variables (only key investment variables)
        for i in range(min(5, G)):
            solution["model_variables"][f"yPnew_{i}"] = float(yPnew[i].X)
            solution["model_variables"][f"yP_{i}"] = float(yP[i].X)
        for l in range(min(3, L)):
            solution["model_variables"][f"yFnew_{l}"] = float(yFnew[l].X)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log_solution(final_obj, solution)


if __name__ == "__main__":
    main()