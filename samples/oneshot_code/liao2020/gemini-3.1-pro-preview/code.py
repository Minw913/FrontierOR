import argparse
import json
import time
import random
import gurobipy as gp

def compute_cost(T, scenario, c_w, c_g, c_o, L):
    P = len(T)
    S = [0.0] * P
    W = [0.0] * P
    C = [0.0] * P
    
    u = scenario[0]['u']
    vA = scenario[0]['vA']
    vI = scenario[0]['vI']
    q = scenario[0]['q']
    v = vA if q >= 0.5 else vI
    
    S[0] = max(T[0], T[0] + u) # T[0] is strictly 0
    W[0] = S[0] - T[0]
    C[0] = S[0] + v
    cost = c_w * W[0]
    
    for i in range(1, P):
        ui = scenario[i]['u']
        vAi = scenario[i]['vA']
        vIi = scenario[i]['vI']
        qi = scenario[i]['q']
        vi = vAi if qi >= 0.5 else vIi
        
        S[i] = max(T[i] + ui, C[i-1])
        W[i] = S[i] - T[i]
        C[i] = S[i] + vi
        
        I_prev = S[i] - C[i-1]
        cost += c_w * W[i] + c_g * I_prev
        
    O = max(0.0, C[P-1] - L)
    cost += c_o * O
    return cost

def eval_wce_cg(T, assignment, patients, c_w, c_g, c_o, L):
    P = len(assignment)
    
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    model = gp.Model("cg", env=env)
    
    # Bounded dual variables to gracefully handle initially implicitly infeasible primals
    M_val = 1e5
    y_u = model.addVars(P, lb=-M_val, ub=M_val)
    y_q = model.addVars(P, lb=-M_val, ub=M_val)
    y_vA = model.addVars(P, lb=-M_val, ub=M_val)
    y_vI = model.addVars(P, lb=-M_val, ub=M_val)
    y_0 = model.addVar(lb=-M_val, ub=M_val)
    
    obj_expr = y_0
    for i, p_idx in enumerate(assignment):
        pt = patients[p_idx]
        obj_expr += y_u[i] * pt['mean_arrival_time_deviation']
        obj_expr += y_q[i] * pt['mean_prep_adequacy']
        obj_expr += y_vA[i] * pt['mean_duration_adequate_prep']
        obj_expr += y_vI[i] * pt['mean_duration_inadequate_prep']
    model.setObjective(obj_expr, gp.GRB.MINIMIZE)

    def add_scenario(s_vars):
        cost = compute_cost(T, s_vars, c_w, c_g, c_o, L)
        expr = y_0
        for i in range(P):
            expr += y_u[i] * s_vars[i]['u']
            expr += y_q[i] * s_vars[i]['q']
            expr += y_vA[i] * s_vars[i]['vA']
            expr += y_vI[i] * s_vars[i]['vI']
        model.addConstr(expr >= cost)

    # Initial spanning scenarios
    s_lb, s_ub, s_mix = [], [], []
    for p_idx in assignment:
        pt = patients[p_idx]
        s_lb.append({'u': pt['lower_bound_arrival_time_deviation'], 'q': 0, 'vA': pt['lower_bound_duration_adequate_prep'], 'vI': pt['lower_bound_duration_inadequate_prep']})
        s_ub.append({'u': pt['upper_bound_arrival_time_deviation'], 'q': 1, 'vA': pt['upper_bound_duration_adequate_prep'], 'vI': pt['upper_bound_duration_inadequate_prep']})
        s_mix.append({'u': pt['upper_bound_arrival_time_deviation'], 'q': 0, 'vA': pt['lower_bound_duration_adequate_prep'], 'vI': pt['upper_bound_duration_inadequate_prep']})
    add_scenario(s_lb)
    add_scenario(s_ub)
    add_scenario(s_mix)
    
    rng = random.Random(73)
    try:
        for it in range(40):
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                break
                
            cur_y0 = y_0.X
            cur_yu = [y_u[i].X for i in range(P)]
            cur_yq = [y_q[i].X for i in range(P)]
            cur_yvA = [y_vA[i].X for i in range(P)]
            cur_yvI = [y_vI[i].X for i in range(P)]
            
            best_val = -1e9
            best_scenario = None
            
            for restart in range(12):
                curr_s = []
                for i, p_idx in enumerate(assignment):
                    pt = patients[p_idx]
                    curr_s.append({
                        'u': rng.choice([pt['lower_bound_arrival_time_deviation'], pt['upper_bound_arrival_time_deviation']]),
                        'q': rng.choice([0, 1]),
                        'vA': rng.choice([pt['lower_bound_duration_adequate_prep'], pt['upper_bound_duration_adequate_prep']]),
                        'vI': rng.choice([pt['lower_bound_duration_inadequate_prep'], pt['upper_bound_duration_inadequate_prep']])
                    })
                
                def eval_sep(s):
                    c = compute_cost(T, s, c_w, c_g, c_o, L)
                    sub = cur_y0
                    for i in range(P):
                        sub += cur_yu[i]*s[i]['u'] + cur_yq[i]*s[i]['q'] + cur_yvA[i]*s[i]['vA'] + cur_yvI[i]*s[i]['vI']
                    return c - sub
                    
                curr_val = eval_sep(curr_s)
                changed = True
                while changed:
                    changed = False
                    for i in range(P):
                        pt = patients[assignment[i]]
                        for key, bounds in [
                            ('u', (pt['lower_bound_arrival_time_deviation'], pt['upper_bound_arrival_time_deviation'])),
                            ('q', (0, 1)),
                            ('vA', (pt['lower_bound_duration_adequate_prep'], pt['upper_bound_duration_adequate_prep'])),
                            ('vI', (pt['lower_bound_duration_inadequate_prep'], pt['upper_bound_duration_inadequate_prep']))
                        ]:
                            old_v = curr_s[i][key]
                            new_v = bounds[1] if old_v == bounds[0] else bounds[0]
                            curr_s[i][key] = new_v
                            nval = eval_sep(curr_s)
                            if nval > curr_val + 1e-6:
                                curr_val = nval
                                changed = True
                            else:
                                curr_s[i][key] = old_v
                                
                if curr_val > best_val:
                    best_val = curr_val
                    best_scenario = [{k: v for k,v in curr_s[i].items()} for i in range(P)]
                    
            if best_val > 1e-4:
                add_scenario(best_scenario)
            else:
                break
        return float(model.ObjVal)
    except gp.GurobiError:
        return 1e9

def optimize_T_heuristic(assignment, scenarios, patients, c_w, c_g, c_o, L):
    P = len(assignment)
    if P == 0: return [], 0.0
    
    def eval_saa(T):
        total = sum(compute_cost(T, s, c_w, c_g, c_o, L) for s in scenarios)
        return total / len(scenarios)

    T = [0.0] * P
    for i in range(1, P):
        pt = patients[assignment[i-1]]
        mean_v = pt['mean_prep_adequacy'] * pt['mean_duration_adequate_prep'] + (1 - pt['mean_prep_adequacy']) * pt['mean_duration_inadequate_prep']
        T[i] = min(L, T[i-1] + mean_v)

    best_T = list(T)
    best_obj = eval_saa(best_T)

    step_sizes = [15.0, 5.0, 1.0]
    for step in step_sizes:
        while True:
            improved = False
            for i in range(1, P):
                for dir in [-1, 1]:
                    new_T = list(best_T)
                    valid = True
                    for j in range(i, P):
                        new_T[j] += dir * step
                        if not (0 <= new_T[j] <= L):
                            valid = False
                    if valid and new_T[i] >= new_T[i-1]:
                        obj = eval_saa(new_T)
                        if obj < best_obj - 1e-4:
                            best_obj = obj
                            best_T = new_T
                            improved = True
            if not improved:
                for i in range(1, P):
                    for dir in [-1, 1]:
                        new_T = list(best_T)
                        new_T[i] += dir * step
                        if 0 <= new_T[i] <= L and new_T[i-1] <= new_T[i] and (i == P-1 or new_T[i] <= new_T[i+1]):
                            obj = eval_saa(new_T)
                            if obj < best_obj - 1e-4:
                                best_obj = obj
                                best_T = new_T
                                improved = True
            if not improved:
                break
    return best_T, best_obj

def refine_T_exact(T, assignment, best_obj_exact, patients, c_w, c_g, c_o, L_prob, start_time, time_limit):
    P = len(assignment)
    if P <= 1: return T, best_obj_exact
    
    step_sizes = [5.0, 1.0]
    best_T = list(T)
    best_obj = best_obj_exact
    for step in step_sizes:
        while True:
            improved = False
            for i in range(1, P):
                if time.time() - start_time > time_limit - 2:
                    return best_T, best_obj
                for dir in [-1, 1]:
                    new_T = list(best_T)
                    valid = True
                    for j in range(i, P):
                        new_T[j] += dir * step
                        if not (0 <= new_T[j] <= L_prob):
                            valid = False
                    if valid and new_T[i] >= new_T[i-1]:
                        obj = eval_wce_cg(new_T, assignment, patients, c_w, c_g, c_o, L_prob)
                        if obj < best_obj - 1e-4:
                            best_obj = obj
                            best_T = new_T
                            improved = True
            if not improved:
                break
    return best_T, best_obj

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    args = parser.parse_args()

    start_time = time.time()
    
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
        
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    P_num = data["num_patients_P"]
    L_prob = data["provider_service_hours_L_minutes"]
    cost_struct_key = list(data["cost_structures"].keys())[0]
    costs = data["cost_structures"][cost_struct_key]
    c_w, c_g, c_o = costs["c_w"], costs["c_g"], costs["c_o"]
    
    patients = {p_data["patient_index"]: p_data for p_data in data["patients"]}
    
    def sort_key(k):
        pt = patients[k]
        q = pt['mean_prep_adequacy']
        muA = pt['mean_duration_adequate_prep']
        muI = pt['mean_duration_inadequate_prep']
        return q * muA + (1 - q) * muI
        
    best_assignment = sorted(patients.keys(), key=sort_key)
    
    rng = random.Random(42)
    crn = [[{'u': rng.random(), 'q': rng.random(), 'vA': rng.random(), 'vI': rng.random()} for _ in range(max(P_num, 40))] for _ in range(150)]
    
    def get_saa_scenarios(assign):
        scen = []
        for s_idx in range(len(crn)):
            s = []
            for i, p_idx in enumerate(assign):
                pt = patients[p_idx]
                def sample_pt(L, U, M, r):
                    if U - L < 1e-5: return M
                    prob_U = (M - L) / (U - L)
                    return U if r < prob_U else L
                s.append({
                    'u': sample_pt(pt['lower_bound_arrival_time_deviation'], pt['upper_bound_arrival_time_deviation'], pt['mean_arrival_time_deviation'], crn[s_idx][i]['u']),
                    'q': 1 if crn[s_idx][i]['q'] < pt['mean_prep_adequacy'] else 0,
                    'vA': sample_pt(pt['lower_bound_duration_adequate_prep'], pt['upper_bound_duration_adequate_prep'], pt['mean_duration_adequate_prep'], crn[s_idx][i]['vA']),
                    'vI': sample_pt(pt['lower_bound_duration_inadequate_prep'], pt['upper_bound_duration_inadequate_prep'], pt['mean_duration_inadequate_prep'], crn[s_idx][i]['vI']),
                })
            scen.append(s)
        return scen

    base_scen = get_saa_scenarios(best_assignment)
    best_T, best_saa_obj = optimize_T_heuristic(best_assignment, base_scen, patients, c_w, c_g, c_o, L_prob)
    best_obj = eval_wce_cg(best_T, best_assignment, patients, c_w, c_g, c_o, L_prob)
    
    if logger: logger.log(best_obj)

    improved = True
    while improved:
        if time.time() - start_time > args.time_limit - 10:
            break
        improved = False
        for i in range(P_num):
            for j in range(i+1, P_num):
                if time.time() - start_time > args.time_limit - 10: break
                
                test_assign = list(best_assignment)
                test_assign[i], test_assign[j] = test_assign[j], test_assign[i]
                
                test_scen = get_saa_scenarios(test_assign)
                test_T, test_saa_obj = optimize_T_heuristic(test_assign, test_scen, patients, c_w, c_g, c_o, L_prob)
                
                if test_saa_obj < best_saa_obj + 0.1 * abs(best_obj): 
                    test_obj = eval_wce_cg(test_T, test_assign, patients, c_w, c_g, c_o, L_prob)
                    if test_obj < best_obj - 1e-4:
                        ref_T, ref_obj = refine_T_exact(test_T, test_assign, test_obj, patients, c_w, c_g, c_o, L_prob, start_time, args.time_limit)
                        best_obj = ref_obj
                        best_assignment = test_assign
                        best_T = ref_T
                        best_saa_obj = test_saa_obj
                        improved = True
                        if logger: logger.log(best_obj)
                        break
            if improved: break

    assignment_out = {str(pt): idx+1 for idx, pt in enumerate(best_assignment)}
    schedule_out = {str(idx+1): float(best_T[idx]) for idx in range(P_num)}
    patient_start_times = {str(pt): float(best_T[idx]) for idx, pt in enumerate(best_assignment)}

    solution = {
        "objective_value": float(best_obj),
        "assignment": assignment_out,
        "schedule": schedule_out,
        "patient_start_times": patient_start_times
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    solve()