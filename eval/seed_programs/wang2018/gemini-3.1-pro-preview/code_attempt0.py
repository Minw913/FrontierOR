import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to a JSONL file for logging intermediate solutions.")
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    num_vessels = data['num_vessels']
    num_berths = data['num_berths']
    num_subblocks = data['num_subblocks']
    H = data['H']
    E = data['E']
    vessels = data['vessels']
    QC_capacity_per_timestep = data['QC_capacity_per_timestep']
    penalty_cost_coefficient_co = data['penalty_cost_coefficient_co']
    loading_route_lengths_D_L = data['loading_route_lengths_D_L']
    average_unloading_route_lengths_D_U = data['average_unloading_route_lengths_D_U']

    # Precompute valid start time windows for each vessel and profile
    valid_t = []
    for i in range(num_vessels):
        vt_i = []
        FW_start = vessels[i]['feasible_time_window'][0]
        FW_end = vessels[i]['feasible_time_window'][1]
        for p in range(len(vessels[i]['qc_profiles'])):
            h_ip = vessels[i]['qc_profiles'][p]['handling_time']
            # End time must not exceed the end of the feasible time window
            vt_i.append(list(range(FW_start, FW_end - h_ip + 2)))
        valid_t.append(vt_i)

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model("ContainerTerminal", env=env)
    
    # Apply standard constraints to strictly match evaluation requirements
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)

    # ------------------
    # Decision Variables
    # ------------------
    
    # Berth assignment 
    y = model.addVars(num_vessels, num_berths, vtype=GRB.BINARY, name="y")
    for i in range(num_vessels):
        coef = penalty_cost_coefficient_co * vessels[i]['unloading_containers']
        for b in range(num_berths):
            y[i, b].Obj = coef * average_unloading_route_lengths_D_U[b]

    # Yard subblock assignment 
    z = model.addVars(num_vessels, num_berths, num_subblocks, vtype=GRB.BINARY, name="z")
    for i in range(num_vessels):
        R_i = vessels[i]['reserved_subblocks']
        if R_i > 0:
            coef = penalty_cost_coefficient_co * (vessels[i]['loading_containers'] / R_i)
            for b in range(num_berths):
                for k in range(num_subblocks):
                    z[i, b, k].Obj = coef * loading_route_lengths_D_L[k][b]

    # Time and Profile Selection
    x_keys = []
    for i in range(num_vessels):
        for p in range(len(vessels[i]['qc_profiles'])):
            for t in valid_t[i][p]:
                x_keys.append((i, p, t))
    x = model.addVars(x_keys, vtype=GRB.BINARY, name="x")

    # Time deviations for objective penalties
    delta_start = model.addVars(num_vessels, vtype=GRB.CONTINUOUS, name="delta_start")
    delta_end = model.addVars(num_vessels, vtype=GRB.CONTINUOUS, name="delta_end")
    for i in range(num_vessels):
        delta_start[i].Obj = vessels[i]['penalty_cost_coefficient']
        delta_end[i].Obj = vessels[i]['penalty_cost_coefficient']

    # Continuous variables computing actual start and end times
    s = model.addVars(num_vessels, vtype=GRB.CONTINUOUS, name="s")
    e = model.addVars(num_vessels, vtype=GRB.CONTINUOUS, name="e")

    # Sequence and non-overlap tracking
    sb_keys = [(i, j) for i in range(num_vessels) for j in range(i + 1, num_vessels)]
    same_berth = model.addVars(sb_keys, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="same_berth")
    delta = model.addVars(sb_keys, vtype=GRB.BINARY, name="delta")

    # ------------------
    # Constraints 
    # ------------------

    # Vessel strictly assigned to exactly one berth
    model.addConstrs((y.sum(i, '*') == 1 for i in range(num_vessels)), name="y_sum")
    
    # Vessel strictly uses exactly one QC profile and starts at exactly one valid time
    model.addConstrs((x.sum(i, '*', '*') == 1 for i in range(num_vessels)), name="x_sum")

    # Time relationships mappings
    model.addConstrs((s[i] == gp.quicksum(t * x[i, p, t] for p in range(len(vessels[i]['qc_profiles'])) for t in valid_t[i][p]) for i in range(num_vessels)), name="s_def")
    model.addConstrs((e[i] == gp.quicksum((t + vessels[i]['qc_profiles'][p]['handling_time'] - 1) * x[i, p, t] for p in range(len(vessels[i]['qc_profiles'])) for t in valid_t[i][p]) for i in range(num_vessels)), name="e_def")

    model.addConstrs((delta_start[i] >= vessels[i]['expected_time_window'][0] - s[i] for i in range(num_vessels)), name="d_start")
    model.addConstrs((delta_end[i] >= e[i] - vessels[i]['expected_time_window'][1] for i in range(num_vessels)), name="d_end")

    # Subblock allocations bounds
    model.addConstrs((z.sum(i, b, '*') == vessels[i]['reserved_subblocks'] * y[i, b] for i in range(num_vessels) for b in range(num_berths)), name="z_req")
    model.addConstrs((z.sum('*', '*', k) <= 1 for k in range(num_subblocks)), name="z_cap")

    # Overlapping and Wrap-Around Limits
    model.addConstrs((same_berth[i, j] >= y[i, b] + y[j, b] - 1 for i, j in sb_keys for b in range(num_berths)), name="sb_def")

    M1 = H + E + 1
    model.addConstrs((e[i] + 1 <= s[j] + M1 * (1 - delta[i, j]) + M1 * (1 - same_berth[i, j]) for i, j in sb_keys), name="non_overlap_ij")
    model.addConstrs((e[j] + 1 <= s[i] + M1 * delta[i, j] + M1 * (1 - same_berth[i, j]) for i, j in sb_keys), name="non_overlap_ji")
    
    # Gap boundary checks to make sure cyclical scheduling wraps around flawlessly
    model.addConstrs((e[j] - s[i] <= H - 1 + M1 * (1 - same_berth[min(i, j), max(i, j)]) for i in range(num_vessels) for j in range(num_vessels) if i != j), name="wrap_around")

    # QC Utilizations
    U_expr = {tau: [] for tau in range(1, H + E + 1)}
    for i in range(num_vessels):
        for p in range(len(vessels[i]['qc_profiles'])):
            h_ip = vessels[i]['qc_profiles'][p]['handling_time']
            for t in valid_t[i][p]:
                for offset in range(h_ip):
                    tau = t + offset
                    qc_amt = vessels[i]['qc_profiles'][p]['qc_per_timestep'][offset]
                    if qc_amt > 0:
                        U_expr[tau].append(qc_amt * x[i, p, t])

    U = {tau: gp.quicksum(U_expr[tau]) for tau in range(1, H + E + 1)}
    
    model.addConstrs((U[t] + U.get(t + H, 0) <= QC_capacity_per_timestep[t - 1] for t in range(1, E + 1)), name="qc_cap_early")
    model.addConstrs((U[t] <= QC_capacity_per_timestep[t - 1] for t in range(E + 1, H + 1)), name="qc_cap_late")
    model.addConstrs((U[t] <= QC_capacity_per_timestep[t - 1] for t in range(H + 1, H + E + 1)), name="qc_cap_ext")

    def construct_solution(val_func, obj_val):
        sol_vessels = []
        berth_assigned = {b: [] for b in range(num_berths)}
        qc_usage = {}
        
        for i in range(num_vessels):
            assigned_b = None
            for b in range(num_berths):
                if val_func(y[i, b]) > 0.5:
                    assigned_b = b
                    break
                    
            assigned_p, assigned_t = None, None
            for p in range(len(vessels[i]['qc_profiles'])):
                for t in valid_t[i][p]:
                    if val_func(x[i, p, t]) > 0.5:
                        assigned_p = p
                        assigned_t = t
                        break
                if assigned_p is not None:
                    break
                    
            assigned_subblocks = []
            if assigned_b is not None:
                for k in range(num_subblocks):
                    if val_func(z[i, assigned_b, k]) > 0.5:
                        assigned_subblocks.append(k)
                        
            # Fallbacks in case unexpected extraction behaviors triggered
            if assigned_b is None: assigned_b = 0
            if assigned_p is None: assigned_p = 0
            if assigned_t is None: 
                for p in range(len(vessels[i]['qc_profiles'])):
                    if valid_t[i][p]:
                        assigned_p = p
                        assigned_t = valid_t[i][p][0]
                        break
                        
            handling_time = vessels[i]['qc_profiles'][assigned_p]['handling_time']
            qc_pts = vessels[i]['qc_profiles'][assigned_p]['qc_per_timestep']
            
            v_dict = {
                "vessel_id": vessels[i]['vessel_id'],
                "alpha": int(assigned_t),
                "beta": int(assigned_t + handling_time - 1),
                "berth": int(assigned_b),
                "profile": int(assigned_p),
                "subblocks": assigned_subblocks,
                "handling_time": int(handling_time),
                "qc_per_timestep": qc_pts
            }
            sol_vessels.append(v_dict)
            berth_assigned[assigned_b].append(v_dict)
            
            for offset in range(handling_time):
                step = int(assigned_t + offset)
                qc_usage[step] = qc_usage.get(step, 0) + qc_pts[offset]
                
        sol_berths = []
        for b in range(num_berths):
            if len(berth_assigned[b]) > 0:
                rho = min(v['alpha'] for v in berth_assigned[b])
                varsigma = max(v['beta'] for v in berth_assigned[b])
            else:
                rho = 0
                varsigma = 0
            sol_berths.append({
                "berth": b,
                "rho": int(rho),
                "varsigma": int(varsigma)
            })
            
        sol_qc = {str(k): int(v) for k, v in qc_usage.items() if v > 0}
        
        return {
            "objective_value": float(obj_val),
            "vessels": sol_vessels,
            "berths": sol_berths,
            "qc_usage": sol_qc
        }

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                try:
                    sol_dict = construct_solution(lambda var: model.cbGetSolution(var), obj)
                    logger.log_solution(obj, sol_dict)
                except Exception:
                    pass 

    model.optimize(cb)

    if model.SolCount > 0:
        sol_dict = construct_solution(lambda var: var.X, model.ObjVal)
        with open(args.solution_path, 'w') as f:
            json.dump(sol_dict, f, indent=2)

if __name__ == '__main__':
    main()