import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def build_solution(w_vals, y_vals, data):
    vessels_out = []
    berths_info = {}
    qc_usage = {tau: 0 for tau in range(1, data['H'] + data['E'] + 1)}
    
    for v in range(data['num_vessels']):
        found = False
        for b in range(data['num_berths']):
            for p_idx, p in enumerate(data['vessels'][v]['qc_profiles']):
                ht = p['handling_time']
                t_min = max(1, data['vessels'][v]['feasible_time_window'][0])
                t_max = min(data['H'], data['vessels'][v]['feasible_time_window'][1] - ht + 1)
                for t in range(t_min, t_max + 1):
                    val = w_vals.get((v, b, p_idx, t), 0.0)
                    if val > 0.5:
                        alpha = t
                        beta = t + ht - 1
                        subblocks = [s for s in range(data['num_subblocks']) if y_vals.get((v, s), 0.0) > 0.5]
                        subblocks.sort()
                        
                        vessels_out.append({
                            "vessel_id": data['vessels'][v]['vessel_id'],
                            "alpha": alpha,
                            "beta": beta,
                            "berth": b,
                            "profile": p_idx,
                            "subblocks": subblocks,
                            "handling_time": ht,
                            "qc_per_timestep": p['qc_per_timestep']
                        })
                        
                        for i in range(ht):
                            qc_usage[t + i] += p['qc_per_timestep'][i]
                            
                        if b not in berths_info:
                            berths_info[b] = {'rho': data['H'] + data['E'], 'varsigma': 0}
                        berths_info[b]['rho'] = min(berths_info[b]['rho'], alpha)
                        berths_info[b]['varsigma'] = max(berths_info[b]['varsigma'], beta)
                        
                        found = True
                        break
                if found: break
            if found: break

    berths_out = []
    for b in range(data['num_berths']):
        if b in berths_info:
            berths_out.append({
                "berth": b,
                "rho": berths_info[b]['rho'],
                "varsigma": berths_info[b]['varsigma']
            })
        else:
            berths_out.append({
                "berth": b,
                "rho": 0,
                "varsigma": 0
            })
            
    qc_usage_out = {str(k): v for k, v in qc_usage.items() if v > 0}
    
    vessels_out.sort(key=lambda x: x['vessel_id'])
    berths_out.sort(key=lambda x: x['berth'])
    
    return {
        "vessels": vessels_out,
        "berths": berths_out,
        "qc_usage": qc_usage_out
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    num_vessels = data['num_vessels']
    num_berths = data['num_berths']
    num_QCs = data['num_QCs']
    num_subblocks = data['num_subblocks']
    H = data['H']
    E = data['E']
    penalty_cost_coefficient_co = data['penalty_cost_coefficient_co']
    QC_capacity_per_timestep = data['QC_capacity_per_timestep']
    loading_route_lengths_D_L = data['loading_route_lengths_D_L']
    average_unloading_route_lengths_D_U = data['average_unloading_route_lengths_D_U']
    vessels = data['vessels']

    model = gp.Model("ContainerTerminal")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit

    # w keys
    w_keys = []
    w_keys_by_v = {v: [] for v in range(num_vessels)}
    w_keys_by_v_b = {(v, b): [] for v in range(num_vessels) for b in range(num_berths)}
    
    for v in range(num_vessels):
        for b in range(num_berths):
            for p_idx, p in enumerate(vessels[v]['qc_profiles']):
                ht = p['handling_time']
                t_min = max(1, vessels[v]['feasible_time_window'][0])
                t_max = min(H, vessels[v]['feasible_time_window'][1] - ht + 1)
                for t in range(t_min, t_max + 1):
                    key = (v, b, p_idx, t)
                    w_keys.append(key)
                    w_keys_by_v[v].append(key)
                    w_keys_by_v_b[(v, b)].append(key)

    w = model.addVars(w_keys, vtype=GRB.BINARY, name="w")
    
    # y keys
    y_keys = []
    y_keys_by_v = {v: [] for v in range(num_vessels)}
    y_keys_by_s = {s: [] for s in range(num_subblocks)}
    
    for v in range(num_vessels):
        for s in range(num_subblocks):
            key = (v, s)
            y_keys.append(key)
            y_keys_by_v[v].append(key)
            y_keys_by_s[s].append(key)
            
    y = model.addVars(y_keys, vtype=GRB.BINARY, name="y")
    
    # z keys
    z_keys = []
    for v in range(num_vessels):
        for s in range(num_subblocks):
            for b in range(num_berths):
                z_keys.append((v, s, b))
    z = model.addVars(z_keys, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z")
    
    # x, alpha, beta
    x = model.addVars(num_vessels, num_berths, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    alpha = model.addVars(num_vessels, vtype=GRB.CONTINUOUS, name="alpha")
    beta = model.addVars(num_vessels, vtype=GRB.CONTINUOUS, name="beta")
    
    for v in range(num_vessels):
        for b in range(num_berths):
            model.addConstr(x[v, b] == gp.quicksum(w[k] for k in w_keys_by_v_b[(v, b)]))
            
    for v in range(num_vessels):
        alpha_expr = gp.LinExpr(0)
        beta_expr = gp.LinExpr(0)
        for k in w_keys_by_v[v]:
            v_id, b, p_idx, t = k
            ht = vessels[v_id]['qc_profiles'][p_idx]['handling_time']
            alpha_expr += w[k] * t
            beta_expr += w[k] * (t + ht - 1)
        model.addConstr(alpha[v] == alpha_expr)
        model.addConstr(beta[v] == beta_expr)

    # 1. sum w == 1
    for v in range(num_vessels):
        model.addConstr(gp.quicksum(w[k] for k in w_keys_by_v[v]) == 1)

    # 2. sum y == reserved_subblocks
    for v in range(num_vessels):
        model.addConstr(gp.quicksum(y[k] for k in y_keys_by_v[v]) == vessels[v]['reserved_subblocks'])

    # 3. sum y <= 1
    for s in range(num_subblocks):
        model.addConstr(gp.quicksum(y[k] for k in y_keys_by_s[s]) <= 1)

    # 4. z >= y + x - 1
    for v in range(num_vessels):
        for b in range(num_berths):
            for s in range(num_subblocks):
                model.addConstr(z[v, s, b] >= y[v, s] + x[v, b] - 1)

    # QC capacity
    qc_usage_expr = {tau: [] for tau in range(1, H+E+1)}
    for v in range(num_vessels):
        for b in range(num_berths):
            for p_idx, p in enumerate(vessels[v]['qc_profiles']):
                ht = p['handling_time']
                t_min = max(1, vessels[v]['feasible_time_window'][0])
                t_max = min(H, vessels[v]['feasible_time_window'][1] - ht + 1)
                for t in range(t_min, t_max + 1):
                    w_var = w[v, b, p_idx, t]
                    for i in range(ht):
                        tau = t + i
                        qc_usage_expr[tau].append((w_var, p['qc_per_timestep'][i]))

    for tau in range(E+1, H+1):
        expr = gp.LinExpr(0)
        for w_var, coeff in qc_usage_expr[tau]:
            expr += w_var * coeff
        model.addConstr(expr <= QC_capacity_per_timestep[tau-1])

    for tau in range(1, E+1):
        expr1 = gp.LinExpr(0)
        for w_var, coeff in qc_usage_expr[tau]:
            expr1 += w_var * coeff
        expr2 = gp.LinExpr(0)
        if tau + H in qc_usage_expr:
            for w_var, coeff in qc_usage_expr[tau + H]:
                expr2 += w_var * coeff
        model.addConstr(expr1 + expr2 <= QC_capacity_per_timestep[tau-1])

    # Berth overlap
    berth_usage_expr = {(b, tau): [] for b in range(num_berths) for tau in range(1, H+E+1)}
    for v in range(num_vessels):
        for b in range(num_berths):
            for p_idx, p in enumerate(vessels[v]['qc_profiles']):
                ht = p['handling_time']
                t_min = max(1, vessels[v]['feasible_time_window'][0])
                t_max = min(H, vessels[v]['feasible_time_window'][1] - ht + 1)
                for t in range(t_min, t_max + 1):
                    w_var = w[v, b, p_idx, t]
                    for i in range(ht):
                        tau = t + i
                        berth_usage_expr[(b, tau)].append(w_var)

    for b in range(num_berths):
        for tau in range(1, H+E+1):
            if len(berth_usage_expr[(b, tau)]) > 0:
                model.addConstr(gp.quicksum(berth_usage_expr[(b, tau)]) <= 1)

    # start_b and end_b
    start_b = model.addVars(num_berths, vtype=GRB.CONTINUOUS, lb=0, ub=H, name="start_b")
    end_b = model.addVars(num_berths, vtype=GRB.CONTINUOUS, lb=0, ub=H+E, name="end_b")
    M = H + E
    for b in range(num_berths):
        model.addConstr(end_b[b] - start_b[b] <= H - 1)
        for v in range(num_vessels):
            model.addConstr(end_b[b] >= beta[v] - M * (1 - x[v, b]))
            model.addConstr(start_b[b] <= alpha[v] + M * (1 - x[v, b]))

    # Penalty cost
    s_def = model.addVars(num_vessels, vtype=GRB.CONTINUOUS, lb=0, name="s_def")
    e_def = model.addVars(num_vessels, vtype=GRB.CONTINUOUS, lb=0, name="e_def")
    for v in range(num_vessels):
        exp_start = vessels[v]['expected_time_window'][0]
        exp_end = vessels[v]['expected_time_window'][1]
        model.addConstr(s_def[v] >= exp_start - alpha[v])
        model.addConstr(e_def[v] >= beta[v] - exp_end)

    # Objective
    obj = gp.LinExpr(0)
    for v in range(num_vessels):
        obj += vessels[v]['penalty_cost_coefficient'] * (s_def[v] + e_def[v])
        for b in range(num_berths):
            D_U = average_unloading_route_lengths_D_U[b]
            obj += penalty_cost_coefficient_co * D_U * vessels[v]['unloading_containers'] * x[v, b]
            
        rs = vessels[v]['reserved_subblocks'] if vessels[v]['reserved_subblocks'] > 0 else 1
        for s in range(num_subblocks):
            for b in range(num_berths):
                D_L = loading_route_lengths_D_L[s][b]
                cost_per_unit = penalty_cost_coefficient_co * (vessels[v]['loading_containers'] / rs) * D_L
                obj += cost_per_unit * z[v, s, b]

    model.setObjective(obj, GRB.MINIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                w_vals_list = model.cbGetSolution(list(w.values()))
                w_vals = {w_keys[i]: w_vals_list[i] for i in range(len(w_keys))}
                y_vals_list = model.cbGetSolution(list(y.values()))
                y_vals = {y_keys[i]: y_vals_list[i] for i in range(len(y_keys))}
                sol_dict = build_solution(w_vals, y_vals, data)
                sol_dict["objective_value"] = obj_val
                logger.log_solution(obj_val, sol_dict)

    model.optimize(callback)

    if model.SolCount > 0:
        w_vals_list = model.getAttr('X', w)
        w_vals = {w_keys[i]: w_vals_list[i] for i in range(len(w_keys))}
        y_vals_list = model.getAttr('X', y)
        y_vals = {y_keys[i]: y_vals_list[i] for i in range(len(y_keys))}
        final_sol = build_solution(w_vals, y_vals, data)
        final_sol["objective_value"] = model.ObjVal
        with open(args.solution_path, 'w') as f:
            json.dump(final_sol, f, indent=2)

if __name__ == "__main__":
    main()