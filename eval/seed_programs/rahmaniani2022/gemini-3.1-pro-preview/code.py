import json
import argparse
import sys
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Power System Capacity Expansion")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    logger = None
    if SolutionLogger and args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    m = gp.Model("CapacityExpansion")
    m.setParam("TimeLimit", args.time_limit)
    m.setParam("Seed", 0)
    m.setParam("MIPGap", 1e-4)
    m.setParam("NumericFocus", 0)
    m.setParam("Threads", 1)

    def gcb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj)
            except:
                pass

    P = data["problem_parameters"]
    N_zones = P["num_zones"]
    N_gen = P["num_generator_clusters"]
    T = P["total_hours"]
    W = P["num_weeks"]
    H = P["hours_per_subperiod"]

    y_P_new = m.addVars(N_gen, vtype=GRB.INTEGER, name="y_P_new")
    y_P_ret = m.addVars(N_gen, vtype=GRB.INTEGER, name="y_P_ret")
    y_P = m.addVars(N_gen, name="y_P")

    y_E_new = m.addVars(N_gen, vtype=GRB.INTEGER, name="y_E_new")
    y_E = m.addVars(N_gen, name="y_E")

    N_lines = len(data.get("transmission_lines", []))
    y_F_new = m.addVars(N_lines, vtype=GRB.INTEGER, name="y_F_new")
    y_F = m.addVars(N_lines, name="y_F")

    x_inj = m.addVars(N_gen, T, name="x_inj")
    x_wdw = m.addVars(N_gen, T, name="x_wdw")
    x_soc = m.addVars(N_gen, T, name="x_soc")
    x_spill = m.addVars(N_gen, T, name="x_spill")
    x_lvl = m.addVars(N_gen, T, name="x_lvl")

    x_commit = m.addVars(N_gen, T, name="x_commit")
    x_start = m.addVars(N_gen, T, name="x_start")
    x_shut = m.addVars(N_gen, T, name="x_shut")

    x_flow = m.addVars(N_lines, T, lb=-GRB.INFINITY, name="x_flow")
    
    x_nse = {}
    for seg in data.get("consumer_segments", []):
        s = seg["segment_id"]
        z = seg["zone"]
        max_frac = seg["max_nse_fraction"]
        for t in range(T):
            dem = data["demand"][str(z)][t]
            x_nse[s, z, t] = m.addVar(ub=max_frac * dem, name=f"x_nse[{s},{z},{t}]")

    x_RPS = m.addVars(W, name="x_RPS")
    x_CO2 = m.addVars(W, name="x_CO2")

    pol = data.get("policy", {})
    pol_type = pol.get("type", "REF")
    if pol_type != "RPS":
        for w in range(W): x_RPS[w].ub = 0
    if pol_type != "CO2":
        for w in range(W): x_CO2[w].ub = 0

    gens = data["generators"]
    params = data["generator_parameters"]
    
    for i in range(N_gen):
        gen = gens[i]
        cid = str(gen["cluster_id"])
        ext_cap = data["existing_capacity"].get(cid, 0.0)
        inc = gen["capacity_size_MW"] if gen["is_UC"] else 1.0
        
        m.addConstr(y_P[i] == ext_cap + inc * (y_P_new[i] - y_P_ret[i]))
        m.addConstr(y_P_ret[i] * inc <= ext_cap)
        if not gen.get("can_retire", True):
            m.addConstr(y_P_ret[i] == 0)
            
        if gen.get("is_storage", False):
            p = params["battery"]
            min_dur = p.get("min_duration_MWh_per_MW", 0.0)
            max_dur = p.get("max_duration_MWh_per_MW", 4.0)
            ext_energy = m.addVar(name=f"ext_energy_{i}")
            m.addConstr(ext_energy >= min_dur * ext_cap)
            m.addConstr(ext_energy <= max_dur * ext_cap)
            
            e_inc = p.get("energy_capacity_size_MWh", 1.0) if gen["is_UC"] else 1.0
            m.addConstr(y_E[i] == ext_energy + e_inc * y_E_new[i])
            m.addConstr(y_E[i] >= min_dur * y_P[i])
            m.addConstr(y_E[i] <= max_dur * y_P[i])
        else:
            m.addConstr(y_E_new[i] == 0)
            m.addConstr(y_E[i] == 0)
            for t in range(T):
                x_wdw[i, t].ub = 0
                x_soc[i, t].ub = 0

        if not gen.get("is_hydro", False):
            for t in range(T):
                x_lvl[i, t].ub = 0
                x_spill[i, t].ub = 0
                
        if not gen.get("is_UC", False):
            for t in range(T):
                x_commit[i, t].ub = 0
                x_start[i, t].ub = 0
                x_shut[i, t].ub = 0

    cap_limits = {}
    for i in range(N_gen):
        z = gens[i]["zone"]
        rtype = gens[i]["resource_type"]
        cap_limits.setdefault((z, rtype), []).append(i)

    for (z, rtype), idxs in cap_limits.items():
        if rtype in params:
            max_cap = params[rtype].get("max_capacity_MW", None)
            if max_cap is not None:
                expr = gp.quicksum(y_P_new[i] * (gens[i]["capacity_size_MW"] if gens[i]["is_UC"] else 1.0) for i in idxs)
                m.addConstr(expr <= max_cap)
            if rtype == "battery":
                max_e_cap = params["battery"].get("max_energy_capacity_MWh", None)
                if max_e_cap is not None:
                    expr_e = gp.quicksum(y_E_new[i] * (params["battery"]["energy_capacity_size_MWh"] if gens[i]["is_UC"] else 1.0) for i in idxs)
                    m.addConstr(expr_e <= max_e_cap)

    for l in range(N_lines):
        line = data["transmission_lines"][l]
        ext_cap = line.get("existing_capacity_MW", 0.0)
        m.addConstr(y_F[l] == ext_cap + y_F_new[l] * 1.0)
        m.addConstr(y_F[l] <= line["max_capacity_MW"])
        for t in range(T):
            m.addConstr(x_flow[l, t] <= y_F[l])
            m.addConstr(x_flow[l, t] >= -y_F[l])

    avail_cache = {i: data["availability"][str(gens[i]["cluster_id"])] for i in range(N_gen)}
    hydro_inflow = data.get("hydro_inflow", {})
    hydro_cache = {i: hydro_inflow.get(str(gens[i]["cluster_id"]), []) for i in range(N_gen)}

    for w in range(W):
        for h in range(H):
            t = w * H + h
            prev_h = (h - 1) % H
            prev_t = w * H + prev_h
            
            for i in range(N_gen):
                gen = gens[i]
                p = params[gen["resource_type"]]
                avail = avail_cache[i][t]
                prev_avail = avail_cache[i][prev_t]
                unit_size = p.get("capacity_size_MW", 1.0)
                min_out = p.get("min_output_frac", 0.0)
                
                if gen.get("is_UC", False):
                    m.addConstr(x_commit[i, t] * unit_size <= y_P[i])
                    m.addConstr(x_inj[i, t] >= x_commit[i, t] * unit_size * min_out)
                    m.addConstr(x_inj[i, t] <= x_commit[i, t] * unit_size * avail)
                    m.addConstr(x_commit[i, t] - x_commit[i, prev_t] == x_start[i, t] - x_shut[i, t])
                    
                    ramp_up = p.get("ramp_up_rate_pct_per_hr", 1.0)
                    ramp_dn = p.get("ramp_down_rate_pct_per_hr", 1.0)
                    N_curr = x_commit[i, t]
                    N_start = x_start[i, t]
                    N_shut = x_shut[i, t]
                    
                    m.addConstr(x_inj[i, t] - x_inj[i, prev_t] <= 
                                ramp_up * unit_size * (N_curr - N_start) + 
                                unit_size * min(avail, max(min_out, ramp_up)) * N_start - 
                                unit_size * min_out * N_shut)
                                
                    m.addConstr(x_inj[i, prev_t] - x_inj[i, t] <= 
                                ramp_dn * unit_size * (N_curr - N_start) + 
                                unit_size * min(prev_avail, max(min_out, ramp_dn)) * N_shut - 
                                unit_size * min_out * N_start)
                else:
                    m.addConstr(x_inj[i, t] <= y_P[i] * avail)
                    if not gen.get("is_storage", False) and not gen.get("is_hydro", False) and min_out > 0:
                        m.addConstr(x_inj[i, t] >= y_P[i] * min_out)
                    
                    if "ramp_up_rate_pct_per_hr" in p:
                        m.addConstr(x_inj[i, t] - x_inj[i, prev_t] <= p["ramp_up_rate_pct_per_hr"] * y_P[i])
                    if "ramp_down_rate_pct_per_hr" in p:
                        m.addConstr(x_inj[i, prev_t] - x_inj[i, t] <= p["ramp_down_rate_pct_per_hr"] * y_P[i])

                if gen.get("is_storage", False):
                    char_eff = p.get("charging_efficiency", 1.0)
                    dis_eff = p.get("discharging_efficiency", 1.0)
                    self_dis = p.get("self_discharge_rate_pct", 0.0)
                    m.addConstr(x_soc[i, t] == (1 - self_dis) * x_soc[i, prev_t] + x_wdw[i, t] * char_eff - x_inj[i, t] / dis_eff)
                    m.addConstr(x_soc[i, t] <= y_E[i])
                    m.addConstr(x_wdw[i, t] * char_eff <= y_E[i] - x_soc[i, prev_t])
                    m.addConstr(x_inj[i, t] / dis_eff <= x_soc[i, prev_t])
                    m.addConstr(x_inj[i, t] + x_wdw[i, t] <= y_P[i])
                    m.addConstr(x_wdw[i, t] <= y_P[i] * avail)

                if gen.get("is_hydro", False):
                    inflow = hydro_cache[i][t]
                    dur = p.get("duration_MWh_per_MW", 0.0)
                    m.addConstr(x_lvl[i, t] == x_lvl[i, prev_t] + inflow * y_P[i] - x_inj[i, t] - x_spill[i, t])
                    m.addConstr(x_lvl[i, t] <= dur * y_P[i])
                    m.addConstr(x_inj[i, t] + x_spill[i, t] >= min_out * y_P[i])

            for i in range(N_gen):
                gen = gens[i]
                if not gen.get("is_UC", False): continue
                p = params[gen["resource_type"]]
                UT = min(H, p.get("min_up_time_hr", 0))
                if UT > 0:
                    expr = gp.quicksum(x_start[i, w * H + ((h - k) % H)] for k in range(UT))
                    m.addConstr(x_commit[i, t] >= expr)
                
                DT = min(H, p.get("min_down_time_hr", 0))
                if DT > 0:
                    expr = gp.quicksum(x_shut[i, w * H + ((h - k) % H)] for k in range(DT))
                    m.addConstr(x_commit[i, t] + expr <= y_P[i] / p.get("capacity_size_MW", 1.0))

    gens_in_zone = {z: [] for z in range(N_zones)}
    for i, gen in enumerate(gens):
        gens_in_zone[gen["zone"]].append(i)
        
    lines_to_zone = {z: [] for z in range(N_zones)}
    lines_from_zone = {z: [] for z in range(N_zones)}
    for l, line in enumerate(data.get("transmission_lines", [])):
        lines_to_zone[line["to_zone"]].append(l)
        lines_from_zone[line["from_zone"]].append(l)

    segs_in_zone = {z: [] for z in range(N_zones)}
    for seg in data.get("consumer_segments", []):
        segs_in_zone[seg["zone"]].append(seg["segment_id"])

    for z in range(N_zones):
        for t in range(T):
            inj_expr = gp.quicksum(x_inj[i, t] - x_wdw[i, t] for i in gens_in_zone[z])
            flow_expr = gp.quicksum(x_flow[l, t] for l in lines_to_zone[z]) - gp.quicksum(x_flow[l, t] for l in lines_from_zone[z])
            nse_expr = gp.quicksum(x_nse[s, z, t] for s in segs_in_zone[z])
            m.addConstr(inj_expr + flow_expr + nse_expr == data["demand"][str(z)][t])

    D_tot = 0
    for w in range(W):
        W_t = data["subperiod_weights"][w] / 168.0
        for h in range(H):
            t = w * H + h
            for z in range(N_zones):
                D_tot += W_t * data["demand"][str(z)][t]

    if pol_type == "RPS":
        rps_rhs = pol.get("RPS_share", 0.0) * D_tot
        rps_expr = gp.quicksum(x_RPS[w] for w in range(W))
        for w in range(W):
            W_t = data["subperiod_weights"][w] / 168.0
            for h in range(H):
                t = w * H + h
                for i in range(N_gen):
                    if gens[i].get("is_RPS_qualifying", False):
                        rps_expr += W_t * x_inj[i, t]
        m.addConstr(rps_expr >= rps_rhs)
        
    elif pol_type == "CO2":
        co2_rhs = pol.get("CO2_cap_tons_per_MWh", 0.0) * D_tot
        co2_expr = -gp.quicksum(x_CO2[w] for w in range(W))
        for w in range(W):
            W_t = data["subperiod_weights"][w] / 168.0
            for h in range(H):
                t = w * H + h
                for i in range(N_gen):
                    p = params[gens[i]["resource_type"]]
                    co2_val = p.get("CO2_emission_tons_per_MWh", 0.0)
                    if co2_val > 0:
                        co2_expr += W_t * co2_val * x_inj[i, t]
                        if gens[i].get("is_storage", False):
                            co2_expr += W_t * co2_val * x_wdw[i, t]
        m.addConstr(co2_expr <= co2_rhs)

    obj_terms = []
    for i in range(N_gen):
        gen = gens[i]
        p = params[gen["resource_type"]]
        inc = gen["capacity_size_MW"] if gen["is_UC"] else 1.0
        obj_terms.append(y_P_new[i] * inc * p.get("investment_cost_per_MW_yr", 0.0))
        obj_terms.append(y_P[i] * p.get("fixed_OM_cost_per_MW_yr", 0.0))
        
        if gen.get("is_storage", False):
            e_inc = p.get("energy_capacity_size_MWh", 1.0) if gen["is_UC"] else 1.0
            obj_terms.append(y_E_new[i] * e_inc * p.get("energy_investment_cost_per_MWh_yr", 0.0))
            obj_terms.append(y_E[i] * p.get("energy_fixed_OM_cost_per_MWh_yr", 0.0))
            
        if gen.get("is_hydro", False):
            dur = p.get("duration_MWh_per_MW", 0.0)
            obj_terms.append(y_P_new[i] * inc * dur * p.get("energy_investment_cost_per_MWh_yr", 0.0))
            obj_terms.append(y_P[i] * dur * p.get("energy_fixed_OM_cost_per_MWh_yr", 0.0))

    for l in range(N_lines):
        line = data["transmission_lines"][l]
        obj_terms.append(y_F_new[l] * 1.0 * line.get("investment_cost_per_MW_yr", 0.0))

    for w in range(W):
        W_t = data["subperiod_weights"][w] / 168.0
        for h in range(H):
            t = w * H + h
            for i in range(N_gen):
                gen = gens[i]
                p = params[gen["resource_type"]]
                var_cost = p.get("variable_cost_per_MWh", 0.0)
                if var_cost > 0:
                    obj_terms.append(W_t * var_cost * x_inj[i, t])
                    if gen.get("is_storage", False):
                        obj_terms.append(W_t * var_cost * x_wdw[i, t])
                if gen.get("is_UC", False):
                    sc = p.get("start_up_cost", 0.0)
                    if sc > 0:
                        obj_terms.append(W_t * sc * x_start[i, t])
                        
            for seg in data.get("consumer_segments", []):
                s = seg["segment_id"]
                z = seg["zone"]
                obj_terms.append(W_t * seg["cost_per_MWh"] * x_nse[s, z, t])

    if pol_type == "RPS":
        rps_pen = pol.get("RPS_noncompliance_cost_per_MWh", 0.0)
        if rps_pen > 0:
            obj_terms.append(rps_pen * gp.quicksum(x_RPS[w] for w in range(W)))
    elif pol_type == "CO2":
        co2_pen = pol.get("CO2_noncompliance_cost_per_ton", 0.0)
        if co2_pen > 0:
            obj_terms.append(co2_pen * gp.quicksum(x_CO2[w] for w in range(W)))

    m.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
    m.optimize(gcb)

    if m.SolCount > 0:
        model_vars_dict = {}
        for v in m.getVars():
            if abs(v.X) > 1e-6:
                model_vars_dict[v.VarName] = v.X

        investments = {}
        storage_energy_investments = {}
        for i in range(N_gen):
            gen = gens[i]
            cid = str(gen["cluster_id"])
            inc = gen["capacity_size_MW"] if gen["is_UC"] else 1.0
            investments[cid] = {
                "resource_type": gen["resource_type"],
                "zone": gen["zone"],
                "new_units": y_P_new[i].X,
                "new_MW": y_P_new[i].X * inc,
                "retired_units": y_P_ret[i].X,
                "retired_MW": y_P_ret[i].X * inc,
                "total_capacity_MW": y_P[i].X
            }
            if gen.get("is_storage", False):
                e_inc = params["battery"].get("energy_capacity_size_MWh", 1.0) if gen["is_UC"] else 1.0
                storage_energy_investments[cid] = {
                    "new_units": y_E_new[i].X,
                    "new_MWh": y_E_new[i].X * e_inc,
                    "total_energy_MWh": y_E[i].X
                }

        transmission_investments = {}
        for l in range(N_lines):
            line = data["transmission_lines"][l]
            lid = str(line["line_id"])
            transmission_investments[lid] = {
                "new_MW": y_F_new[l].X * 1.0,
                "total_capacity_MW": y_F[l].X
            }

        output = {
            "objective_value": m.ObjVal,
            "model_variables": model_vars_dict,
            "investments": investments,
            "storage_energy_investments": storage_energy_investments,
            "transmission_investments": transmission_investments
        }

        with open(args.solution_path, "w") as f:
            json.dump(output, f, indent=2)

        if logger:
            logger.log_solution(m.ObjVal, output)

if __name__ == "__main__":
    main()