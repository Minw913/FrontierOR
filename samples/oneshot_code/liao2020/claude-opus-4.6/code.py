import argparse
import json
import math
import time
import itertools

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    P = data['num_patients_P']
    L = data['provider_service_hours_L_minutes']
    patients = data['patients']
    cost_structures = data['cost_structures']
    
    cost_name = list(cost_structures.keys())[0]
    cs = cost_structures[cost_name]
    c_w = cs['c_w']
    c_g = cs['c_g']
    c_o = cs['c_o']
    
    mu_A = []
    mu_I = []
    d_AL = []
    d_AU = []
    d_IL = []
    d_IU = []
    mu_q = []
    mu_u = []
    u_L_arr = []
    u_U_arr = []
    
    for p in patients:
        mu_A.append(p['mean_duration_adequate_prep'])
        mu_I.append(p['mean_duration_inadequate_prep'])
        d_AL.append(p['lower_bound_duration_adequate_prep'])
        d_AU.append(p['upper_bound_duration_adequate_prep'])
        d_IL.append(p['lower_bound_duration_inadequate_prep'])
        d_IU.append(p['upper_bound_duration_inadequate_prep'])
        mu_q.append(p['mean_prep_adequacy'])
        mu_u.append(p['mean_arrival_time_deviation'])
        u_L_arr.append(p['lower_bound_arrival_time_deviation'])
        u_U_arr.append(p['upper_bound_arrival_time_deviation'])
    
    mean_duration = [mu_q[p] * mu_A[p] + (1 - mu_q[p]) * mu_I[p] for p in range(P)]
    
    def get_patient_atoms(p_idx):
        """Return list of (duration, arrival_dev, prep_type) atoms for patient p."""
        p = p_idx
        atoms = []
        d_a_vals = set()
        d_a_vals.add(d_AL[p])
        d_a_vals.add(d_AU[p])
        d_i_vals = set()
        d_i_vals.add(d_IL[p])
        d_i_vals.add(d_IU[p])
        u_vals = set()
        u_vals.add(u_L_arr[p])
        u_vals.add(u_U_arr[p])
        
        for d_val in sorted(d_a_vals):
            for u_val in sorted(u_vals):
                atoms.append((d_val, u_val, 'A'))
        
        for d_val in sorted(d_i_vals):
            for u_val in sorted(u_vals):
                atoms.append((d_val, u_val, 'I'))
        
        return atoms
    
    def evaluate_cost_scenario(perm, schedule, durations, arrivals):
        P_loc = len(perm)
        C_prev = 0.0
        total_wait = 0.0
        total_idle = 0.0
        
        for i in range(P_loc):
            arrival_time = schedule[i] + arrivals[i]
            B_i = max(arrival_time, C_prev)
            
            wait_i = max(0.0, B_i - schedule[i])
            if i > 0:
                idle_i = max(0.0, B_i - C_prev)
                total_idle += idle_i
            
            total_wait += wait_i
            C_prev = B_i + durations[i]
        
        overtime = max(0.0, C_prev - L)
        cost = c_w * total_wait + c_g * total_idle + c_o * overtime
        return cost
    
    def solve_worst_case_lp(perm, schedule):
        """Solve for worst-case E[cost] using LP over joint distributions."""
        import gurobipy as gp
        from gurobipy import GRB
        
        P_loc = len(perm)
        
        pos_atoms = []
        for i in range(P_loc):
            atoms = get_patient_atoms(perm[i])
            pos_atoms.append(atoms)
        
        n_atoms = [len(a) for a in pos_atoms]
        
        total_joint = 1
        for n in n_atoms:
            total_joint *= n
        
        if total_joint > 1000000:
            return evaluate_worst_case_product(perm, schedule)
        
        joint_indices = list(itertools.product(*[range(n) for n in n_atoms]))
        
        costs = []
        for ji in joint_indices:
            durations = [pos_atoms[i][ji[i]][0] for i in range(P_loc)]
            arrivals_vals = [pos_atoms[i][ji[i]][1] for i in range(P_loc)]
            cost = evaluate_cost_scenario(perm, schedule, durations, arrivals_vals)
            costs.append(cost)
        
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('Method', 0)
        
        n_joint = len(joint_indices)
        prob_vars = model.addVars(n_joint, lb=0.0, ub=1.0, name='p')
        
        model.addConstr(gp.quicksum(prob_vars[j] for j in range(n_joint)) == 1.0)
        
        for i in range(P_loc):
            p_idx = perm[i]
            atoms = pos_atoms[i]
            
            adequate_indices = [j for j, ji in enumerate(joint_indices) if atoms[ji[i]][2] == 'A']
            inadequate_indices = [j for j, ji in enumerate(joint_indices) if atoms[ji[i]][2] == 'I']
            
            if mu_q[p_idx] > 0 and mu_q[p_idx] < 1:
                model.addConstr(
                    gp.quicksum(prob_vars[j] for j in adequate_indices) == mu_q[p_idx]
                )
            elif mu_q[p_idx] == 1.0:
                for j in inadequate_indices:
                    model.addConstr(prob_vars[j] == 0)
            elif mu_q[p_idx] == 0.0:
                for j in adequate_indices:
                    model.addConstr(prob_vars[j] == 0)
            
            if mu_q[p_idx] > 0 and len(adequate_indices) > 0:
                model.addConstr(
                    gp.quicksum(prob_vars[j] * atoms[joint_indices[j][i]][0]
                               for j in adequate_indices) == mu_q[p_idx] * mu_A[p_idx]
                )
            
            if (1 - mu_q[p_idx]) > 0 and len(inadequate_indices) > 0:
                model.addConstr(
                    gp.quicksum(prob_vars[j] * atoms[joint_indices[j][i]][0]
                               for j in inadequate_indices) == (1 - mu_q[p_idx]) * mu_I[p_idx]
                )
            
            model.addConstr(
                gp.quicksum(prob_vars[j] * atoms[joint_indices[j][i]][1]
                           for j in range(n_joint)) == mu_u[p_idx]
            )
        
        model.setObjective(
            gp.quicksum(prob_vars[j] * costs[j] for j in range(n_joint)),
            GRB.MAXIMIZE
        )
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            return model.objVal
        elif model.SolCount > 0:
            return model.objVal
        else:
            return float('inf')
    
    def evaluate_worst_case_product(perm, schedule):
        """Evaluate using product of marginal worst-case distributions (approximation)."""
        P_loc = len(perm)
        
        pos_scenarios = []
        for i in range(P_loc):
            p_idx = perm[i]
            scenarios = get_patient_product_scenarios(p_idx)
            pos_scenarios.append(scenarios)
        
        n_total = 1
        for sc in pos_scenarios:
            n_total *= len(sc)
        
        if n_total <= 200000:
            total_cost = 0.0
            for combo in itertools.product(*pos_scenarios):
                prob = 1.0
                durations = []
                arrivals_vals = []
                for d_val, u_val, p_val in combo:
                    prob *= p_val
                    durations.append(d_val)
                    arrivals_vals.append(u_val)
                cost = evaluate_cost_scenario(perm, schedule, durations, arrivals_vals)
                total_cost += prob * cost
            return total_cost
        else:
            # Sample-based approximation without numpy
            import random
            random.seed(42)
            n_samples = 10000
            total_cost = 0.0
            
            pos_cdfs = []
            for sc_list in pos_scenarios:
                cum = []
                s = 0.0
                for d_val, u_val, p_val in sc_list:
                    s += p_val
                    cum.append(s)
                pos_cdfs.append((sc_list, cum))
            
            for _ in range(n_samples):
                durations = []
                arrivals_vals = []
                for sc_list, cum in pos_cdfs:
                    r = random.random()
                    idx = 0
                    while idx < len(cum) - 1 and r > cum[idx]:
                        idx += 1
                    durations.append(sc_list[idx][0])
                    arrivals_vals.append(sc_list[idx][1])
                cost = evaluate_cost_scenario(perm, schedule, durations, arrivals_vals)
                total_cost += cost
            
            return total_cost / n_samples
    
    def get_patient_product_scenarios(p_idx):
        """For patient p, return list of (duration, arrival_dev, probability) under product of worst-case marginals."""
        p = p_idx
        scenarios = []
        
        if d_AU[p] > d_AL[p]:
            p_A_lo = (d_AU[p] - mu_A[p]) / (d_AU[p] - d_AL[p])
            p_A_hi = 1 - p_A_lo
        else:
            p_A_lo = 1.0
            p_A_hi = 0.0
        
        if d_IU[p] > d_IL[p]:
            p_I_lo = (d_IU[p] - mu_I[p]) / (d_IU[p] - d_IL[p])
            p_I_hi = 1 - p_I_lo
        else:
            p_I_lo = 1.0
            p_I_hi = 0.0
        
        if u_U_arr[p] > u_L_arr[p]:
            p_u_lo = (u_U_arr[p] - mu_u[p]) / (u_U_arr[p] - u_L_arr[p])
            p_u_hi = 1 - p_u_lo
        else:
            p_u_lo = 1.0
            p_u_hi = 0.0
        
        for q, pq in [(1, mu_q[p]), (0, 1 - mu_q[p])]:
            if pq <= 1e-15:
                continue
            if q == 1:
                dur_scenarios = []
                if p_A_lo > 1e-15:
                    dur_scenarios.append((d_AL[p], p_A_lo))
                if p_A_hi > 1e-15:
                    dur_scenarios.append((d_AU[p], p_A_hi))
                if not dur_scenarios:
                    dur_scenarios = [(mu_A[p], 1.0)]
            else:
                dur_scenarios = []
                if p_I_lo > 1e-15:
                    dur_scenarios.append((d_IL[p], p_I_lo))
                if p_I_hi > 1e-15:
                    dur_scenarios.append((d_IU[p], p_I_hi))
                if not dur_scenarios:
                    dur_scenarios = [(mu_I[p], 1.0)]
            
            arr_scenarios = []
            if p_u_lo > 1e-15:
                arr_scenarios.append((u_L_arr[p], p_u_lo))
            if p_u_hi > 1e-15:
                arr_scenarios.append((u_U_arr[p], p_u_hi))
            if not arr_scenarios:
                arr_scenarios = [(mu_u[p], 1.0)]
            
            for d_val, pd in dur_scenarios:
                for u_val, pu in arr_scenarios:
                    scenarios.append((d_val, u_val, pq * pd * pu))
        
        return scenarios
    
    def optimize_schedule_for_perm(perm):
        """Optimize start times for a given permutation using grid search + refinement."""
        P_loc = len(perm)
        
        if P_loc == 1:
            schedule = [0.0]
            cost = solve_worst_case_lp(perm, schedule)
            return schedule, cost
        
        # Check if LP approach is feasible
        n_atoms = [len(get_patient_atoms(perm[i])) for i in range(P_loc)]
        total_joint = 1
        for n in n_atoms:
            total_joint *= n
        use_lp = total_joint <= 500000
        
        eval_func = solve_worst_case_lp if use_lp else evaluate_worst_case_product
        
        best_cost = float('inf')
        best_schedule = None
        
        # Try multiple buffer fractions
        for buffer_frac in [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            s = [0.0] * P_loc
            for i in range(1, P_loc):
                p_idx = perm[i-1]
                interval = mean_duration[p_idx] * (1 + buffer_frac)
                s[i] = s[i-1] + max(interval, 0)
            
            # Clip to [0, L] and ensure non-decreasing
            for i in range(P_loc):
                s[i] = min(max(s[i], 0), L)
            for i in range(1, P_loc):
                s[i] = max(s[i], s[i-1])
            
            cost = eval_func(perm, s)
            if cost < best_cost:
                best_cost = cost
                best_schedule = s[:]
        
        # Local refinement: coordinate descent on intervals
        # For each interval s[i]-s[i-1], try a few values
        improved = True
        iteration = 0
        while improved and iteration < 5:
            improved = False
            iteration += 1
            for i in range(1, P_loc):
                if time.time() - start_time > time_limit * 0.8:
                    break
                
                current_interval = best_schedule[i] - best_schedule[i-1]
                # Try different intervals
                best_interval = current_interval
                best_local_cost = best_cost
                
                # Search range
                lo = 0.0
                hi = max(current_interval * 2, mean_duration[perm[i-1]] * 2)
                hi = min(hi, L - best_schedule[i-1])
                
                for trial_interval in [lo + (hi - lo) * k / 10 for k in range(11)]:
                    s = best_schedule[:]
                    s[i] = s[i-1] + trial_interval
                    # Adjust subsequent times if needed
                    for j in range(i+1, P_loc):
                        if s[j] < s[j-1]:
                            s[j] = s[j-1]
                    for j in range(P_loc):
                        s[j] = min(max(s[j], 0), L)
                    
                    cost = eval_func(perm, s)
                    if cost < best_local_cost - 0.001:
                        best_local_cost = cost
                        best_interval = trial_interval
                
                if best_local_cost < best_cost - 0.001:
                    best_schedule[i] = best_schedule[i-1] + best_interval
                    for j in range(i+1, P_loc):
                        if best_schedule[j] < best_schedule[j-1]:
                            best_schedule[j] = best_schedule[j-1]
                    for j in range(P_loc):
                        best_schedule[j] = min(max(best_schedule[j], 0), L)
                    best_cost = best_local_cost
                    improved = True
        
        # Finer refinement
        improved = True
        iteration = 0
        while improved and iteration < 3:
            improved = False
            iteration += 1
            for i in range(1, P_loc):
                if time.time() - start_time > time_limit * 0.85:
                    break
                
                current_interval = best_schedule[i] - best_schedule[i-1]
                step = max(current_interval * 0.1, 0.5)
                best_local_cost = best_cost
                
                for delta in [-3*step, -2*step, -step, step, 2*step, 3*step]:
                    trial_interval = max(current_interval + delta, 0.0)
                    s = best_schedule[:]
                    s[i] = s[i-1] + trial_interval
                    for j in range(i+1, P_loc):
                        if s[j] < s[j-1]:
                            s[j] = s[j-1]
                    for j in range(P_loc):
                        s[j] = min(max(s[j], 0), L)
                    
                    cost = eval_func(perm, s)
                    if cost < best_local_cost - 0.001:
                        best_local_cost = cost
                        trial_best = s[:]
                
                if best_local_cost < best_cost - 0.001:
                    best_schedule = trial_best[:]
                    best_cost = best_local_cost
                    improved = True
        
        return best_schedule, best_cost
    
    # Main optimization loop
    best_overall_cost = float('inf')
    best_overall_perm = None
    best_overall_schedule = None
    
    n_perms = math.factorial(P)
    
    # Heuristic orderings
    heuristic_perms = set()
    
    spt_order = tuple(sorted(range(P), key=lambda p: mean_duration[p]))
    heuristic_perms.add(spt_order)
    
    lpt_order = tuple(sorted(range(P), key=lambda p: -mean_duration[p]))
    heuristic_perms.add(lpt_order)
    
    def duration_range(p):
        r_A = d_AU[p] - d_AL[p]
        r_I = d_IU[p] - d_IL[p]
        return mu_q[p] * r_A + (1 - mu_q[p]) * r_I
    
    var_asc = tuple(sorted(range(P), key=lambda p: duration_range(p)))
    heuristic_perms.add(var_asc)
    
    var_desc = tuple(sorted(range(P), key=lambda p: -duration_range(p)))
    heuristic_perms.add(var_desc)
    
    heuristic_perms.add(tuple(range(P)))
    
    # Sort by mean duration with variance as tiebreaker
    mixed1 = tuple(sorted(range(P), key=lambda p: (mean_duration[p], duration_range(p))))
    heuristic_perms.add(mixed1)
    mixed2 = tuple(sorted(range(P), key=lambda p: (mean_duration[p], -duration_range(p))))
    heuristic_perms.add(mixed2)
    
    heuristic_perms = list(heuristic_perms)
    
    # Evaluate heuristic orderings
    for perm_tuple in heuristic_perms:
        if time.time() - start_time > time_limit * 0.7:
            break
        
        perm = list(perm_tuple)
        schedule, cost = optimize_schedule_for_perm(perm)
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_perm = perm[:]
            best_overall_schedule = schedule[:]
            if logger:
                logger.log(best_overall_cost)
    
    # Decide whether to enumerate all permutations
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    time_per_perm = elapsed / max(len(heuristic_perms), 1) if heuristic_perms else 1.0
    
    if n_perms <= 5040 and n_perms * time_per_perm < remaining * 0.8:
        # Enumerate all permutations
        evaluated = set(tuple(list(hp)) for hp in heuristic_perms)
        
        for perm_tuple in itertools.permutations(range(P)):
            if time.time() - start_time > time_limit * 0.9:
                break
            
            if perm_tuple in evaluated:
                continue
            evaluated.add(perm_tuple)
            
            perm = list(perm_tuple)
            schedule, cost = optimize_schedule_for_perm(perm)
            
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall_perm = perm[:]
                best_overall_schedule = schedule[:]
                if logger:
                    logger.log(best_overall_cost)
    else:
        # Local search: swap and insertion neighborhoods
        if best_overall_perm is not None:
            current_perm = best_overall_perm[:]
            current_cost = best_overall_cost
            
            improved = True
            while improved and time.time() - start_time < time_limit * 0.85:
                improved = False
                
                # Swap neighborhood
                for i in range(P):
                    for j in range(i+1, P):
                        if time.time() - start_time > time_limit * 0.85:
                            break
                        
                        new_perm = current_perm[:]
                        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                        
                        schedule, cost = optimize_schedule_for_perm(new_perm)
                        
                        if cost < current_cost - 0.01:
                            current_perm = new_perm
                            current_cost = cost
                            improved = True
                            
                            if cost < best_overall_cost:
                                best_overall_cost = cost
                                best_overall_perm = new_perm[:]
                                best_overall_schedule = schedule[:]
                                if logger:
                                    logger.log(best_overall_cost)
                    if time.time() - start_time > time_limit * 0.85:
                        break
                
                # Insertion neighborhood
                if time.time() - start_time < time_limit * 0.85:
                    for i in range(P):
                        for j in range(P):
                            if i == j:
                                continue
                            if time.time() - start_time > time_limit * 0.85:
                                break
                            
                            new_perm = current_perm[:]
                            patient = new_perm.pop(i)
                            new_perm.insert(j, patient)
                            
                            schedule, cost = optimize_schedule_for_perm(new_perm)
                            
                            if cost < current_cost - 0.01:
                                current_perm = new_perm
                                current_cost = cost
                                improved = True
                                
                                if cost < best_overall_cost:
                                    best_overall_cost = cost
                                    best_overall_perm = new_perm[:]
                                    best_overall_schedule = schedule[:]
                                    if logger:
                                        logger.log(best_overall_cost)
                        if time.time() - start_time > time_limit * 0.85:
                            break
    
    # Build output
    if best_overall_perm is None:
        best_overall_perm = list(range(P))
        best_overall_schedule = [0.0] * P
        best_overall_cost = 0.0
    
    assignment = {}
    patient_start_times = {}
    schedule_out = {}
    
    for pos_idx in range(P):
        patient_idx = best_overall_perm[pos_idx]
        patient_1indexed = patient_idx + 1
        position_1indexed = pos_idx + 1
        
        assignment[str(patient_1indexed)] = position_1indexed
        schedule_out[str(position_1indexed)] = best_overall_schedule[pos_idx]
        patient_start_times[str(patient_1indexed)] = best_overall_schedule[pos_idx]
    
    solution = {
        "objective_value": best_overall_cost,
        "assignment": assignment,
        "schedule": schedule_out,
        "patient_start_times": patient_start_times
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)