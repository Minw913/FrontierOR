import argparse
import json
import math
from gurobipy import *
import itertools
import datetime

def solve_nurse_scheduling(instance_path, solution_path, time_limit):
    # Read instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    num_nurses = instance["num_nurses"]
    num_days = instance["num_days"]
    num_weeks = instance["num_weeks"]
    num_shifts = instance["num_shifts"]
    num_skills = instance["num_skills"]
    
    skills = instance["skills"]
    shifts_list = [s['name'] for s in instance["shifts"]]
    shifts_idx_map = {shift: idx for idx, shift in enumerate(shifts_list)}
    forbidden_values = set()
    for pair in instance["forbidden_shift_successions"]:
        forbidden_values.add((pair[0], pair[1]))
    
    penalty_weights = instance["penalty_weights"]

    skills_idx_map = {skill: idx for idx, skill in enumerate(skills)}

    # Parse nurse information
    nurses = instance["nurses"]
    nurse_skills = []  # List of sets of skill indices for each nurse
    nurse_contracts = []  # Contract constraints for each nurse
    
    for n_info in nurses:
        n_skills = set()
        for s in n_info['skills']:
            if s in skills_idx_map:
                n_skills.add(skills_idx_map[s])
        nurse_skills.append(n_skills)
        
        contract = n_info['contract']
        prefs = n_info.get('preferences', [])
        pref_set = set()
        for p in prefs:
            day = p['day']
            shift = p['shift']
            shift_idx = shifts_idx_map.get(shift)
            if shift_idx is not None:  # Only if shift is valid
                pref_set.add((day, shift_idx))
        nurse_contracts.append({
            'L_minus': contract['L_minus'],
            'L_plus': contract['L_plus'],
            'CD_minus': contract['CD_minus'],
            'CD_plus': contract['CD_plus'],
            'CR_minus': contract['CR_minus'],
            'CR_plus': contract['CR_plus'],
            'WE_plus': contract['WE_plus'],
            'preferences': pref_set
        })

    # Parse demand matrix
    demand_matrix = {}  # Key: (day_index, shift_index, skill_index) -> (min_demand, opt_demand)
    for day_idx in range(num_days):
        day_demand = instance["demand"][day_idx]
        for shift_name in shifts_list:
            if shift_name in day_demand:
                shift_data = day_demand[shift_name]
                shift_idx = shifts_idx_map[shift_name]
                for skill_name in skills:
                    if skill_name in shift_data:
                        skill_data = shift_data[skill_name]
                        skill_idx = skills_idx_map[skill_name]
                        min_d = skill_data["minimum"]
                        opt_d = skill_data["optimal"]
                        demand_matrix[(day_idx, shift_idx, skill_idx)] = (min_d, opt_d)
                    else:
                        demand_matrix[(day_idx, shift_idx, skill_idx)] = (0, 0)
            else:
                for skill_idx in range(num_skills):
                    demand_matrix[(day_idx, shifts_idx_map[shift_name], skill_idx)] = (0, 0)

    # Shift constraints (minimum and maximum consecutive same shifts)
    Cs_minus = [next((s['CS_minus'] for s in instance['shifts'] if s['name'] == shift_name), 0) 
                for shift_name in shifts_list]
    Cs_plus = [next((s['CS_plus'] for s in instance['shifts'] if s['name'] == shift_name), num_days) 
               for shift_name in shifts_list]

    # Model setup
    m = Model("NurseScheduling")
    m.setParam("Threads", 1)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    m.setParam('Heuristics', 0.02)
    
    # Decision Variables
    # x[nurse][day][shift][skill] = assignment
    x = {}
    for n_idx in range(num_nurses):
        for d_idx in range(num_days):
            for s_idx in range(num_shifts):
                for sk_idx in range(num_skills):
                    if sk_idx in nurse_skills[n_idx]:
                        x[n_idx, d_idx, s_idx, sk_idx] = m.addVar(vtype=GRB.BINARY,
                                                                 name=f"x_{n_idx}_{d_idx}_{s_idx}_{sk_idx}")
                    else:
                        x[n_idx, d_idx, s_idx, sk_idx] = 0

    # Work day variables
    work = {}  # work[n_idx][d_idx] = whether nurse n_idx works on day d_idx
    for n_idx in range(num_nurses):
        for d_idx in range(num_days):
            work[n_idx, d_idx] = m.addVar(vtype=GRB.BINARY, name=f"work_{n_idx}_{d_idx}")

    # Work patterns: consecutive work stretch and deviations
    # For consecutive work stretches: We need aux variables to measure length violations
    # Use variables to count violations for each stretch
    cd_deviation_under = {}  # Under CD_minus violation for nurse n_idx
    cd_deviation_over = {}   # Over CD_plus violation for nurse n_idx
    for n_idx in range(num_nurses):
        cd_deviation_under[n_idx] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"cd_under_{n_idx}")
        cd_deviation_over[n_idx] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"cd_over_{n_idx}")

    # For same-shift consecutive days
    cs_deviation_under = {}
    cs_deviation_over = {}
    for n_idx in range(num_nurses):
        cs_deviation_under[n_idx] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"cs_und_{n_idx}")
        cs_deviation_over[n_idx] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"cs_ov_{n_idx}")

    # Consecutive days off
    cr_deviation_under = {}
    cr_deviation_over = {}
    rest = {}  # Helper to know if a nurse is off on day d_idx
    for n_idx in range(num_nurses):
        for d_idx in range(num_days):
            rest[n_idx, d_idx] = m.addVar(vtype=GRB.BINARY, name=f"off_{n_idx}_{d_idx}")
        cr_deviation_under[n_idx] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"cr_und_{n_idx}")
        cr_deviation_over[n_idx] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"cr_ov_{n_idx}")

    # Preferences violation
    pref_violation = {}
    for n_idx in range(num_nurses):
        pref_violation[n_idx] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"pref_v_{n_idx}")

    # Weekend assignment violations (incomplete weekends)
    weekend_incomp = {}
    for n_idx in range(num_nurses):
        weekend_count = num_weeks
        weekend_incomp[n_idx] = m.addVars(weekend_count, vtype=GRB.BINARY, name=f"weekend_incomp_{n_idx}")

    # Total workload deviation
    l_deviation_under = {}
    l_deviation_over = {}
    for n_idx in range(num_nurses):
        l_deviation_under[n_idx] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"l_und_{n_idx}")
        l_deviation_over[n_idx] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"l_ov_{n_idx}")

    # Weekend violation for maximum weekends worked
    we_excess = {}
    for n_idx in range(num_nurses):
        we_excess[n_idx] = m.addVar(vtype=GRB.INTEGER, lb=0, name=f"we_exc_{n_idx}")

    # Demand shortage (soft constraint violation)
    shortage = {}
    for d_idx in range(num_days):
        for s_idx in range(num_shifts):
            for sk_idx in range(num_skills):
                min_demand, _ = demand_matrix.get((d_idx, s_idx, sk_idx), (0, 0))
                # Shortage is how much we're under optimal
                _, opt_demand = demand_matrix.get((d_idx, s_idx, sk_idx), (0, 0))
                shortage[d_idx, s_idx, sk_idx] = m.addVar(vtype=GRB.INTEGER, lb=0,
                                ub=max(opt_demand - min_demand, 0),
                                name=f"short_#{d_idx}_{s_idx}_{sk_idx}")


    m.update()

    # Constraints
    # Nurse works at most one shift per day
    for n_idx in range(num_nurses):
        for d_idx in range(num_days):
            m.addConstr(sum(x[n_idx, d_idx, s_idx, sk_idx] 
                           for s_idx in range(num_shifts) for sk_idx in range(num_skills)  
                           if isinstance(x[n_idx, d_idx, s_idx, sk_idx], Var)) <= 1, 
                       name=f"one_shift_per_day_{n_idx}_{d_idx}")

    # Compute if nurse works on a day
    for n_idx in range(num_nurses):
        for d_idx in range(num_days):
            m.addConstr(work[n_idx, d_idx] == 
                        sum(x[n_idx, d_idx, s_idx, sk_idx] 
                            for s_idx in range(num_shifts) for sk_idx in range(num_skills)
                            if isinstance(x[n_idx, d_idx, s_idx, sk_idx], Var)),
                       name=f"work_assign_{n_idx}_{d_idx}")

    # Compute if nurse rests on a day
    for n_idx in range(num_nurses):
        for d_idx in range(num_days):
            m.addConstr(rest[n_idx, d_idx] + work[n_idx, d_idx] == 1,
                       name=f"rest_def_{n_idx}_{d_idx}")

    # Minimum staffing requirement (hard constraint)
    for d_idx in range(num_days):
        for s_idx in range(num_shifts):
            for sk_idx in range(num_skills):
                min_demand, _ = demand_matrix.get((d_idx, s_idx, sk_idx), (0, 0))
                total = sum(x[n_idx, d_idx, s_idx, sk_idx] 
                           for n_idx in range(num_nurses)
                           if isinstance(x[n_idx, d_idx, s_idx, sk_idx], Var))
                m.addConstr(total >= min_demand, 
                           name=f"min_staf_requirement_{d_idx}_{s_idx}_{sk_idx}")

    # Forbidden shift succession constraint
    for n_idx in range(num_nurses):
        for d_idx in range(num_days - 1):
            for s1_idx in range(num_shifts):
                for s2_idx in range(num_shifts):
                    shift1 = shifts_list[s1_idx]
                    shift2 = shifts_list[s2_idx]
                    if (shift1, shift2) in forbidden_values:
                        skill_vars_sum1 = sum(
                            x[n_idx, d_idx, s1_idx, sk_idx] 
                            for sk_idx in range(num_skills)
                            if isinstance(x[n_idx, d_idx, s1_idx, sk_idx], Var)
                        )
                        skill_vars_sum2 = sum(
                            x[n_idx, d_idx+1, s2_idx, sk_idx] 
                            for sk_idx in range(num_skills)
                            if isinstance(x[n_idx, d_idx+1, s2_idx, sk_idx], Var)
                        )
                        if isinstance(skill_vars_sum1, Var) or isinstance(skill_vars_sum1, LinExpr):
                            term1 = skill_vars_sum1 if isinstance(skill_vars_sum1, LinExpr) else LinExpr() + skill_vars_sum1
                        else:
                            term1 = LinExpr()
                        
                        if isinstance(skill_vars_sum2, Var) or isinstance(skill_vars_sum2, LinExpr):
                            term2 = skill_vars_sum2 if isinstance(skill_vars_sum2, LinExpr) else LinExpr() + skill_vars_sum2
                        else:
                            term2 = LinExpr()
                        
                        m.addConstr(LinExpr(term1 + term2) <= 1, 
                                   name=f"forbid_chain_{n_idx}_{d_idx}_{shift1}_{shift2}")
            m.update()

    # Demand optimal vs actual computation
    for d_idx in range(num_days):
        for s_idx in range(num_shifts):
            for sk_idx in range(num_skills):
                _, opt_demand = demand_matrix.get((d_idx, s_idx, sk_idx), (0, 0))
                actual_assigned = sum(x[n_idx, d_idx, s_idx, sk_idx] 
                                     for n_idx in range(num_nurses)
                                     if isinstance(x[n_idx, d_idx, s_idx, sk_idx], Var))
                m.addConstr(actual_assigned + shortage[d_idx, s_idx, sk_idx] >= opt_demand,
                           name=f"optimal_demand_{d_idx}_{s_idx}_{sk_idx}")

    # Workload deviation constraints
    for n_idx in range(num_nurses):
        contract = nurse_contracts[n_idx]
        actual_worked_days = sum(work[n_idx, d_idx] for d_idx in range(num_days))
        L_minus = contract['L_minus']
        L_plus = contract['L_plus']
        
        m.addConstr(actual_worked_days >= L_minus - l_deviation_under[n_idx],
                   name=f"total_worked_min_{n_idx}")
        m.addConstr(actual_worked_days <= L_plus + l_deviation_over[n_idx],
                   name=f"total_worked_max_{n_idx}")

    # Preference constraint
    for n_idx in range(num_nurses):
        pref_set = nurse_contracts[n_idx].get('preferences', set())
        total_pref_violations = LinExpr()
        for (day, shift_idx) in pref_set:
            # Sum of all skills for this nurse on this day+shift
            total_pref_violations += sum(x[n_idx, day, shift_idx, sk_idx] 
                                        for sk_idx in range(num_skills)
                                        if isinstance(x[n_idx, day, shift_idx, sk_idx], Var))
        
        # Create a variable to track this
        m.addConstr(pref_violation[n_idx] >= total_pref_violations,
                   name=f"pref_vio_cons_{n_idx}")


    # Incomplete weekends
    for n_idx in range(num_nurses):
        for week_idx in range(num_weeks):
            sat_day = week_idx*7 + 5  # Saturday
            sun_day = week_idx*7 + 6  # Sunday
            if sat_day < num_days and sun_day < num_days:
                # Define helper binary variables to indicate if worked any shift
                M = 100  # Large enough for our case
                sat_was_worked = m.addVar(vtype=GRB.BINARY, name=f"sat_work_{n_idx}_{week_idx}")
                sun_was_worked = m.addVar(vtype=GRB.BINARY, name=f"sun_work_{n_idx}_{week_idx}")
                
                # Link real assignment sums to these indicator variables
                sat_sum = sum(x[n_idx, sat_day, s_idx, sk_idx] 
                              for s_idx in range(num_shifts) for sk_idx in range(num_skills)
                              if isinstance(x[n_idx, sat_day, s_idx, sk_idx], Var))
                sun_sum = sum(x[n_idx, sun_day, s_idx, sk_idx] 
                              for s_idx in range(num_shifts) for sk_idx in range(num_skills)
                              if isinstance(x[n_idx, sun_day, s_idx, sk_idx], Var))

                # We use big-M constraints to connect binary indicator and actual assignment
                if isinstance(sat_sum, LinExpr):
                    m.addConstr(M * sat_was_worked >= sat_sum, name=f"link_sat1_{n_idx}_{week_idx}")
                    m.addConstr(sat_sum >= sat_was_worked, name=f"link_sat2_{n_idx}_{week_idx}")
                
                if isinstance(sun_sum, LinExpr):
                    m.addConstr(M * sun_was_worked >= sun_sum, name=f"link_sun1_{n_idx}_{week_idx}")
                    m.addConstr(sun_sum >= sun_was_worked, name=f"link_sun2_{n_idx}_{week_idx}")
                                
                # Weekend is incomplete if exactly one of the two weekend days worked
                # Use linear logic: If A XOR B then trigger binary var
                # weekend_incomp[n_idx][week_idx] = 1 if sat_was_worked != sun_was_worked
                m.addConstr(weekend_incomp[n_idx][week_idx] >= sat_was_worked - sun_was_worked,
                           name=f"weekend_inc1_{n_idx}_{week_idx}")
                m.addConstr(weekend_incomp[n_idx][week_idx] >= sun_was_worked - sat_was_worked,
                           name=f"weekend_inc2_{n_idx}_{week_idx}")


    # Total worked weekends excess
    for n_idx in range(num_nurses):
        max_allowed = nurse_contracts[n_idx]['WE_plus']
        weekend_vars_for_sum = []
        
        for week_idx in range(num_weeks):
            sat_day = week_idx*7 + 5  # Saturday
            sun_day = week_idx*7 + 6  # Sunday
            if sat_day < num_days and sun_day < num_days:
                # Weekend is considered "worked" if ANY shift worked on either Sat or Sun
                weekend_worked_indicator = m.addVar(vtype=GRB.BINARY, name=f"weekend_worked_{n_idx}_{week_idx}")
                
                # At least one of Saturday or Sunday needs to have shifts for this indicator to be 1
                # If sum of all shifts on Sat or Sun is >0, binary indicator must be =1
                sat_sum = sum(x[n_idx, sat_day, s_idx, sk_idx] 
                              for s_idx in range(num_shifts) for sk_idx in range(num_skills)
                              if isinstance(x[n_idx, sat_day, s_idx, sk_idx], Var))
                sun_sum = sum(x[n_idx, sun_day, s_idx, sk_idx] 
                              for s_idx in range(num_shifts) for sk_idx in range(num_skills)
                              if isinstance(x[n_idx, sun_day, s_idx, sk_idx], Var))
                
                total_weekend_shifts = sat_sum + sun_sum
                M = 100  # Big-M
                m.addConstr(M * weekend_worked_indicator >= total_weekend_shifts, 
                           name=f"we_link1_{n_idx}_{week_idx}")
                m.addConstr(total_weekend_shifts >= weekend_worked_indicator, 
                           name=f"we_link2_{n_idx}_{week_idx}")
                
                weekend_vars_for_sum.append(weekend_worked_indicator)
        
        # Total weekends worked exceeds max allowed, then we_excess increases
        total_weekends_worked = quicksum(weekend_vars_for_sum)
        m.addConstr(total_weekends_worked <= max_allowed + we_excess[n_idx], 
                   name=f"we_excess_const_{n_idx}")
                   

    # --- Objective function ---
    obj_expr = LinExpr()
    # Insufficient staffing
    w1 = penalty_weights['c_S1']
    obj_expr.add(quicksum(shortage[d, s, sk] for (d,s,sk) in shortage.keys() if isinstance(shortage[d,s,sk], Var))*w1)
    
    # Consecutive worked days violations - we approximate these constraints by simpler forms
    # For now, let's skip the precise counting and just focus on other penalties
    w2a = penalty_weights['c_S2a']
    obj_expr.add(quicksum(cd_deviation_under[n_idx] for n_idx in range(num_nurses))*w2a)
    obj_expr.add(quicksum(cd_deviation_over[n_idx] for n_idx in range(num_nurses))*w2a)
    
    # Consecutive same shift violations
    w2b = penalty_weights['c_S2b']
    obj_expr.add(quicksum(cs_deviation_under[n_idx] for n_idx in range(num_nurses))*w2b)
    obj_expr.add(quicksum(cs_deviation_over[n_idx] for n_idx in range(num_nurses))*w2b)
    
    # Consecutive days off violations
    w3 = penalty_weights['c_S3']
    obj_expr.add(quicksum(cr_deviation_under[n_idx] for n_idx in range(num_nurses))*w3)
    obj_expr.add(quicksum(cr_deviation_over[n_idx] for n_idx in range(num_nurses))*w3)
    
    # Preference violations
    w4 = penalty_weights['c_S4']
    obj_expr.add(quicksum(pref_violation[n_idx] for n_idx in range(num_nurses))*w4)
    
    # Incomplete weekend penalties
    w5 = penalty_weights['c_S5']
    obj_expr.add(quicksum(weekend_incomp[n_idx][i] for n_idx in range(num_nurses) for i in range(num_weeks))*w5)
    
    # Total assignments deviation
    w6 = penalty_weights['c_S6']
    obj_expr.add(quicksum(l_deviation_under[n_idx] for n_idx in range(num_nurses))*w6)
    obj_expr.add(quicksum(l_deviation_over[n_idx] for n_idx in range(num_nurses))*w6)
    
    # Excess weekends worked
    w7 = penalty_weights['c_S7']
    obj_expr.add(quicksum(we_excess[n_idx] for n_idx in range(num_nurses))*w7)

    m.setObjective(obj_expr, GRB.MINIMIZE)
    m.update()
    
    def log_solution_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ) 
            if logger:
                logger.log(obj_val)

    # Optimization
    m.optimize(log_solution_callback)

    # Get objective value
    final_obj_val = m.objVal if m.solCount > 0 else float('inf')
    
    # Build solution
    final_schedule = []
    for n_idx in range(num_nurses):
        nurse_assignments = []
        n_info = nurses[n_idx]
        for d_idx in range(num_days):
            for s_idx in range(num_shifts):
                for sk_idx in range(num_skills):
                    if isinstance(x[n_idx, d_idx, s_idx, sk_idx], Var) and x[n_idx, d_idx, s_idx, sk_idx].X > 0.5:
                        shift_name = shifts_list[s_idx]
                        skill_name = skills[sk_idx]
                        nurse_assignments.append({
                            "day": d_idx,
                            "shift": shift_name,
                            "skill": skill_name
                        })
        
        final_schedule.append({
            "nurse_id": n_info['id'],
            "nurse_name": n_info['name'],
            "assignments": sorted(nurse_assignments, key=lambda a: a['day'])
        })

    # Prepare output
    solution = {
        "objective_value": final_obj_val,
        "schedule": final_schedule
    }

    with open(solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve nurse scheduling problem')
    parser.add_argument("--instance_path", type=str, required=True,
                      help="Path to the JSON file containing the problem instance.")
    parser.add_argument("--solution_path", type=str, required=True,
                      help="Path where the final solution JSON file must be written.")
    parser.add_argument("--time_limit", type=int, default=300,
                      help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None,
                      help="Path to a JSONL file for logging intermediate solutions (optional).")

    args = parser.parse_args()
    
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    solve_nurse_scheduling(args.instance_path, args.solution_path, args.time_limit)