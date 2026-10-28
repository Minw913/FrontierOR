import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB

ALLOWED_PREFIXES = (
    "yPnew_", "yPret_", "yP_",
    "yEnew_", "yEret_", "yE_",
    "yFnew_",
    "xinj_", "xwdw_", "xsoc_", "xhydro_",
    "xflow_",
    "xnse_",
    "xcommit_", "xstart_", "xshut_",
    "xRPS_", "xCO2_"
)

def is_allowed(name):
    for p in ALLOWED_PREFIXES:
        if name.startswith(p):
            return True
    return False

def get_prev(t, w, H, circular):
    if t > w * H:
        return t - 1
    else:
        if circular:
            return w * H + H - 1
        else:
            return None

def get_window(t, w, H, duration, circular):
    indices = []
    for i in range(duration):
        k = t - i
        if k >= w * H:
            indices.append(k)
        else:
            if circular:
                indices.append(w * H + (k % H))
    return indices

def build_solution_dict(var_dict, obj_val, data):
    inv_decs = []
    for g in data.get('generators', []):
        i = g['id']
        d = {
            "generator_id": i,
            "resource_type": g.get('resource_type', ''),
            "zone": g['zone'],
            "y_P_new": var_dict.get(f"yPnew_{i}", 0.0),
            "y_P_ret": var_dict.get(f"yPret_{i}", 0.0),
            "y_P_total_MW": var_dict.get(f"yP_{i}", 0.0)
        }
        if g.get('is_storage'):
            d["y_E_new"] = var_dict.get(f"yEnew_{i}", 0.0)
            d["y_E_ret"] = var_dict.get(f"yEret_{i}", 0.0)
            d["y_E_total_MWh"] = var_dict.get(f"yE_{i}", 0.0)
        inv_decs.append(d)
        
    tx_decs = []
    for l in data.get('transmission_lines', []):
        idx = l['id']
        new_cap = var_dict.get(f"yFnew_{idx}", 0.0)
        tx_decs.append({
            "line_id": idx,
            "y_F_new_MW": new_cap,
            "total_capacity_MW": l.get('existing_capacity_MW', 0.0) + new_cap
        })
            
    filtered_vars = {k: v for k, v in var_dict.items() if is_allowed(k)}
        
    return {
        "objective_value": obj_val,
        "model_variables": filtered_vars,
        "investment_decisions": inv_decs,
        "transmission_decisions": tx_decs
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, default=3600)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    model = gp.Model("PowerSystemPlanning")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)

    dims = data['problem_dimensions']
    num_zones = dims['num_zones']
    total_timesteps = dims['total_timesteps']
    
    subperiods = data['subperiods']
    num_subperiods = subperiods['num_subperiods']
    H = subperiods['hours_per_subperiod']
    subperiod_weights = subperiods['subperiod_weights']
    circular = subperiods['circular_indexing']
    
    generators = data.get('generators', [])
    lines = data.get('transmission_lines', [])
    nse_segments = data.get('nse_segments', [])
    demand = data.get('demand', {})
    avail_prof = data.get('availability_profiles', {})
    policy = data.get('policy', {})

    def get_avail(i, t):
        prof = avail_prof.get(str(i))
        if not prof:
            return 1.0
        if prof.get('type') == 'constant':
            return prof.get('value', 1.0)
        else:
            return prof.get('values', [1.0] * total_timesteps)[t]

    # Dictionaries to store variables
    yPnew_vars, yPret_vars, yP_vars = {}, {}, {}
    yEnew_vars, yEret_vars, yE_vars = {}, {}, {}
    yFnew_vars = {}
    
    xinj_vars, xwdw_vars, xsoc_vars, xhydro_vars, xspill_vars = {}, {}, {}, {}, {}
    xcommit_vars, xstart_vars, xshut_vars = {}, {}, {}
    xflow_vars, xnse_vars = {}, {}
    xRPS_vars, xCO2_vars = {}, {}
    
    # Auxiliary for initial states when circular=False
    socinit_vars, hydroinit_vars, commitinit_vars = {}, {}, {}

    # 1. Variables creation
    for g in generators:
        i = g['id']
        yPnew_vars[i] = model.addVar(vtype=GRB.INTEGER, name=f"yPnew_{i}")
        yPret_vars[i] = model.addVar(vtype=GRB.INTEGER, name=f"yPret_{i}")
        yP_vars[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"yP_{i}")
        
        if g.get('is_storage'):
            yEnew_vars[i] = model.addVar(vtype=GRB.INTEGER, name=f"yEnew_{i}")
            yEret_vars[i] = model.addVar(vtype=GRB.INTEGER, name=f"yEret_{i}")
            yE_vars[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"yE_{i}")
            for w in range(num_subperiods):
                socinit_vars[i, w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"socinit_{i}_{w}")
                
        if g.get('is_hydro'):
            for w in range(num_subperiods):
                hydroinit_vars[i, w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"hydroinit_{i}_{w}")
                
        if g.get('is_UC') and not circular:
            for w in range(num_subperiods):
                commitinit_vars[i, w] = model.addVar(vtype=GRB.INTEGER, name=f"commitinit_{i}_{w}")
                
        for t in range(total_timesteps):
            xinj_vars[i, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"xinj_{i}_{t}")
            if g.get('is_storage'):
                xwdw_vars[i, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"xwdw_{i}_{t}")
                xsoc_vars[i, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"xsoc_{i}_{t}")
            if g.get('is_hydro'):
                xhydro_vars[i, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"xhydro_{i}_{t}")
                xspill_vars[i, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"spill_{i}_{t}")
            if g.get('is_UC'):
                xcommit_vars[i, t] = model.addVar(vtype=GRB.INTEGER, name=f"xcommit_{i}_{t}")
                xstart_vars[i, t] = model.addVar(vtype=GRB.INTEGER, name=f"xstart_{i}_{t}")
                xshut_vars[i, t] = model.addVar(vtype=GRB.INTEGER, name=f"xshut_{i}_{t}")

    for l in lines:
        idx = l['id']
        yFnew_vars[idx] = model.addVar(vtype=GRB.INTEGER, name=f"yFnew_{idx}")
        for t in range(total_timesteps):
            xflow_vars[idx, t] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xflow_{idx}_{t}")

    for z in range(num_zones):
        for seg in nse_segments:
            s_idx = seg['segment']
            for t in range(total_timesteps):
                xnse_vars[s_idx, z, t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"xnse_{s_idx}_{z}_{t}")

    if policy.get('RPS_enabled') and policy.get('RPS_share') is not None:
        for w in range(num_subperiods):
            xRPS_vars[w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"xRPS_{w}")
            
    if policy.get('CO2_cap_enabled') and policy.get('CO2_cap_tons_per_MWh') is not None:
        for w in range(num_subperiods):
            xCO2_vars[w] = model.addVar(vtype=GRB.CONTINUOUS, name=f"xCO2_{w}")

    # 2. Constraints
    for g in generators:
        i = g['id']
        cap_size = g.get('capacity_size_MW', 1.0)
        yP = yP_vars[i]
        
        model.addConstr(yP == g['existing_capacity_MW'] + cap_size * (yPnew_vars[i] - yPret_vars[i]))
        if 'max_capacity_MW' in g:
            model.addConstr(yP <= g['max_capacity_MW'])
        if 'max_new_capacity_MW' in g:
            model.addConstr(cap_size * yPnew_vars[i] <= g['max_new_capacity_MW'])
            
        model.addConstr(cap_size * yPret_vars[i] <= g['existing_capacity_MW'])
        if not g.get('can_retire', True):
            model.addConstr(yPret_vars[i] == 0)
            
        if g.get('is_storage'):
            e_cap_size = g.get('storage_capacity_size_MWh', 1.0)
            yE = yE_vars[i]
            model.addConstr(yE == g['existing_storage_capacity_MWh'] + e_cap_size * (yEnew_vars[i] - yEret_vars[i]))
            if 'max_storage_capacity_MWh' in g:
                model.addConstr(yE <= g['max_storage_capacity_MWh'])
            model.addConstr(e_cap_size * yEret_vars[i] <= g['existing_storage_capacity_MWh'])
            model.addConstr(yE >= g.get('min_duration_MWh_per_MW', 0) * yP)
            model.addConstr(yE <= g.get('max_duration_MWh_per_MW', 1e9) * yP)

        for w in range(num_subperiods):
            for t in range(w * H, w * H + H):
                avail = get_avail(i, t)
                prev_t = get_prev(t, w, H, circular)
                xinj = xinj_vars[i, t]
                
                # UC Limits
                if g.get('is_UC'):
                    xcommit = xcommit_vars[i, t]
                    xstart = xstart_vars[i, t]
                    xshut = xshut_vars[i, t]
                    
                    model.addConstr(xcommit * cap_size <= yP)
                    model.addConstr(xstart * cap_size <= yP)
                    model.addConstr(xshut * cap_size <= yP)
                    
                    model.addConstr(xinj >= xcommit * g.get('min_output_frac', 0) * cap_size)
                    model.addConstr(xinj <= xcommit * avail * cap_size)
                    
                    if prev_t is not None:
                        commit_prev = xcommit_vars[i, prev_t]
                    else:
                        commit_prev = commitinit_vars[i, w]
                        model.addConstr(commit_prev * cap_size <= yP)
                        
                    model.addConstr(xcommit == commit_prev + xstart - xshut)
                    
                    up_window = get_window(t, w, H, g.get('min_up_time_hr', 0), circular)
                    if up_window:
                        model.addConstr(xcommit >= gp.quicksum(xstart_vars[i, k] for k in up_window))
                        
                    down_window = get_window(t, w, H, g.get('min_down_time_hr', 0), circular)
                    if down_window:
                        model.addConstr(xcommit + gp.quicksum(xshut_vars[i, k] for k in down_window) <= (g['existing_capacity_MW'] / cap_size) + yPnew_vars[i] - yPret_vars[i])

                    if prev_t is not None:
                        R_U = g.get('ramp_up_frac_per_hr', 1.0)
                        R_D = g.get('ramp_dn_frac_per_hr', 1.0)
                        M = g.get('min_output_frac', 0.0)
                        xinj_prev = xinj_vars[i, prev_t]
                        
                        start_factor_up = min(avail, max(M, R_U))
                        shut_factor_up = M
                        model.addConstr(xinj - xinj_prev <= cap_size * R_U * (xcommit - xstart) + cap_size * start_factor_up * xstart - cap_size * shut_factor_up * xshut)
                        
                        start_factor_dn = M
                        shut_factor_dn = min(avail, max(M, R_D))
                        model.addConstr(xinj_prev - xinj <= cap_size * R_D * (xcommit - xstart) + cap_size * shut_factor_dn * xshut - cap_size * start_factor_dn * xstart)

                # Non-UC Dispatch bounds
                else:
                    model.addConstr(xinj <= avail * yP)
                    if not g.get('is_hydro'):
                        model.addConstr(xinj >= g.get('min_output_frac', 0) * yP)
                        
                    if prev_t is not None and not g.get('is_storage'):
                        if 'ramp_up_frac_per_hr' in g:
                            model.addConstr(xinj - xinj_vars[i, prev_t] <= g['ramp_up_frac_per_hr'] * yP)
                        if 'ramp_dn_frac_per_hr' in g:
                            model.addConstr(xinj_vars[i, prev_t] - xinj <= g['ramp_dn_frac_per_hr'] * yP)

                # Storage Operations
                if g.get('is_storage'):
                    xwdw = xwdw_vars[i, t]
                    xsoc = xsoc_vars[i, t]
                    c_eff = g.get('charge_efficiency', 1.0)
                    d_eff = g.get('discharge_efficiency', 1.0)
                    
                    model.addConstr(xwdw <= avail * yP)
                    model.addConstr(xinj + xwdw <= yP)
                    model.addConstr(xsoc <= yE_vars[i])
                    
                    if prev_t is not None:
                        soc_prev = xsoc_vars[i, prev_t]
                    else:
                        soc_prev = socinit_vars[i, w]
                        model.addConstr(soc_prev <= yE_vars[i])
                        
                    model.addConstr(xsoc == (1 - g.get('self_discharge_rate', 0.0)) * soc_prev + c_eff * xwdw - (xinj / d_eff if d_eff > 0 else 0))
                    model.addConstr(c_eff * xwdw <= yE_vars[i] - soc_prev)
                    if d_eff > 0:
                        model.addConstr(xinj / d_eff <= soc_prev)

                # Hydro Operations
                if g.get('is_hydro'):
                    xhydro = xhydro_vars[i, t]
                    xspill = xspill_vars[i, t]
                    dur = g.get('duration_MWh_per_MW', 0)
                    
                    model.addConstr(xhydro <= dur * yP)
                    model.addConstr(xinj + xspill >= g.get('min_output_frac', 0) * yP)
                    
                    if prev_t is not None:
                        hydro_prev = xhydro_vars[i, prev_t]
                    else:
                        hydro_prev = hydroinit_vars[i, w]
                        model.addConstr(hydro_prev <= dur * yP)
                        
                    model.addConstr(xhydro == hydro_prev + avail * yP - xinj - xspill)

    # Transmission Limits
    for l in lines:
        idx = l['id']
        model.addConstr(yFnew_vars[idx] <= l.get('max_new_capacity_MW', 1e9))
        for t in range(total_timesteps):
            total_cap = l['existing_capacity_MW'] + yFnew_vars[idx]
            model.addConstr(xflow_vars[idx, t] <= total_cap)
            model.addConstr(xflow_vars[idx, t] >= -total_cap)

    # NSE Limits
    for z in range(num_zones):
        for seg in nse_segments:
            s_idx = seg['segment']
            for t in range(total_timesteps):
                model.addConstr(xnse_vars[s_idx, z, t] <= seg['max_frac'] * demand[str(z)][t])

    # Power Balance
    gen_in_zone = {z: [] for z in range(num_zones)}
    stor_in_zone = {z: [] for z in range(num_zones)}
    for g in generators:
        gen_in_zone[g['zone']].append(g['id'])
        if g.get('is_storage'):
            stor_in_zone[g['zone']].append(g['id'])

    for z in range(num_zones):
        for t in range(total_timesteps):
            inj = gp.quicksum(xinj_vars[i, t] for i in gen_in_zone[z])
            wdw = gp.quicksum(xwdw_vars[i, t] for i in stor_in_zone[z])
            flow_out = gp.quicksum(xflow_vars[l['id'], t] for l in lines if l['from_zone'] == z)
            flow_in = gp.quicksum(xflow_vars[l['id'], t] for l in lines if l['to_zone'] == z)
            nse = gp.quicksum(xnse_vars[seg['segment'], z, t] for seg in nse_segments)
            model.addConstr(inj - wdw - flow_out + flow_in + nse == demand[str(z)][t])

    # Policies
    total_demand_val = gp.quicksum((subperiod_weights[w] / 168.0) * demand[str(z)][t] 
                                   for w in range(num_subperiods) for t in range(w * H, w * H + H) for z in range(num_zones))
    
    if policy.get('RPS_enabled') and policy.get('RPS_share') is not None:
        total_rps = gp.quicksum((subperiod_weights[w] / 168.0) * xinj_vars[g['id'], t] 
                                for w in range(num_subperiods) for t in range(w * H, w * H + H) for g in generators if g.get('is_RPS_qualifying'))
        model.addConstr(total_rps + gp.quicksum(xRPS_vars.values()) >= policy['RPS_share'] * total_demand_val)

    if policy.get('CO2_cap_enabled') and policy.get('CO2_cap_tons_per_MWh') is not None:
        total_emis = 0
        for w in range(num_subperiods):
            h_weight = subperiod_weights[w] / 168.0
            for t in range(w * H, w * H + H):
                for g in generators:
                    ef = g.get('co2_tons_per_MWh', 0)
                    if ef > 0:
                        term = xinj_vars[g['id'], t]
                        if g.get('is_storage'):
                            term += xwdw_vars[g['id'], t]
                        total_emis += h_weight * ef * term
        model.addConstr(total_emis - gp.quicksum(xCO2_vars.values()) <= policy['CO2_cap_tons_per_MWh'] * total_demand_val)

    # Objective
    obj = gp.LinExpr(0)
    for g in generators:
        i = g['id']
        cap_size = g.get('capacity_size_MW', 1.0)
        
        inv_cost = g.get('inv_cost_per_MW_yr', 0) * cap_size * yPnew_vars[i]
        fom_cost = g.get('fom_cost_per_MW_yr', 0) * yP_vars[i]
        if g.get('is_hydro'):
            inv_cost += g.get('inv_cost_energy_per_MWh_yr', 0) * g.get('duration_MWh_per_MW', 0) * cap_size * yPnew_vars[i]
            fom_cost += g.get('fom_cost_energy_per_MWh_yr', 0) * g.get('duration_MWh_per_MW', 0) * yP_vars[i]
        obj += inv_cost + fom_cost
        
        if g.get('is_storage'):
            e_cap_size = g.get('storage_capacity_size_MWh', 1.0)
            obj += g.get('inv_cost_energy_per_MWh_yr', 0) * e_cap_size * yEnew_vars[i]
            obj += g.get('fom_cost_energy_per_MWh_yr', 0) * yE_vars[i]
            
    for l in lines:
        obj += l.get('inv_cost_per_MW_yr', 0) * yFnew_vars[l['id']]

    for w in range(num_subperiods):
        h_weight = subperiod_weights[w] / 168.0
        for t in range(w * H, w * H + H):
            for g in generators:
                i = g['id']
                vc = g.get('var_cost_per_MWh', 0)
                if vc > 0:
                    obj += h_weight * vc * xinj_vars[i, t]
                    if g.get('is_storage'):
                        obj += h_weight * vc * xwdw_vars[i, t]
                if g.get('is_UC') and g.get('start_cost_per_unit', 0) > 0:
                    obj += h_weight * g['start_cost_per_unit'] * xstart_vars[i, t]
            for z in range(num_zones):
                for seg in nse_segments:
                    sc = seg.get('cost_per_MWh', 0)
                    if sc > 0:
                        obj += h_weight * sc * xnse_vars[seg['segment'], z, t]

    if policy.get('RPS_enabled') and policy.get('RPS_noncompliance_cost_per_MWh', 0) > 0:
        obj += policy['RPS_noncompliance_cost_per_MWh'] * gp.quicksum(xRPS_vars.values())
    if policy.get('CO2_cap_enabled') and policy.get('CO2_noncompliance_cost_per_ton', 0) > 0:
        obj += policy['CO2_noncompliance_cost_per_ton'] * gp.quicksum(xCO2_vars.values())

    model.setObjective(obj, GRB.MINIMIZE)
    
    # Store all variables for rapid extraction in callback
    model._vars = model.getVars()
    
    best_obj = float('inf')
    def cb(model_cb, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj - 1e-5:
                best_obj = obj_val
                if logger:
                    try:
                        sol_vals = model_cb.cbGetSolution(model_cb._vars)
                        var_dict = {}
                        for v, val in zip(model_cb._vars, sol_vals):
                            if abs(val) > 1e-6:
                                var_dict[v.VarName] = val
                        sol_dict = build_solution_dict(var_dict, obj_val, data)
                        logger.log_solution(obj_val, sol_dict)
                    except Exception:
                        logger.log(obj_val)
                        
    model.optimize(cb)

    if model.SolCount > 0:
        var_dict = {}
        for v in model.getVars():
            if abs(v.X) > 1e-6:
                var_dict[v.VarName] = v.X
        final_sol = build_solution_dict(var_dict, model.ObjVal, data)
        with open(args.solution_path, 'w') as f:
            json.dump(final_sol, f, indent=2)
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()