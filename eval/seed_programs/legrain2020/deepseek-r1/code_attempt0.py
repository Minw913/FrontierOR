import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path):
    with open(instance_path) as f:
        instance = json.load(f)
    
    num_nurses = instance['num_nurses']
    num_weeks = instance['num_weeks']
    num_days = instance['num_days']
    num_shifts = instance['num_shifts']
    num_skills = instance['num_skills']
    skills = instance['skills']
    shifts = instance['shifts']
    penalty_weights = instance['penalty_weights']
    nurses = instance['nurses']
    demand_data = instance['demand']
    forbidden_shift_successions = instance['forbidden_shift_successions']
    
    shift_name_to_index = {s['name']: idx for idx, s in enumerate(shifts)}
    shift_index_to_name = [s['name'] for s in shifts]
    
    skill_name_to_index = {skill: idx for idx, skill in enumerate(skills)}
    skill_index_to_name = skills.copy()
    
    min_demand = np.zeros((num_days, num_shifts, num_skills), dtype=int)
    opt_demand = np.zeros((num_days, num_shifts, num_skills), dtype=int)
    
    for day in range(num_days):
        day_demand = demand_data[day]
        for shift_name, skill_dict in day_demand.items():
            if shift_name not in shift_name_to_index:
                continue
            s_idx = shift_name_to_index[shift_name]
            for skill_name, values in skill_dict.items():
                if skill_name not in skill_name_to_index:
                    continue
                k_idx = skill_name_to_index[skill_name]
                min_demand[day, s_idx, k_idx] = values['minimum']
                opt_demand[day, s_idx, k_idx] = values['optimal']
    
    nurse_skills = []
    for nurse in nurses:
        skills_n = []
        for skill_name in nurse['skills']:
            if skill_name in skill_name_to_index:
                skills_n.append(skill_name_to_index[skill_name])
        nurse_skills.append(set(skills_n))
    
    forbidden_pairs = []
    for pair in forbidden_shift_successions:
        s1 = pair[0]
        s2 = pair[1]
        if s1 in shift_name_to_index and s2 in shift_name_to_index:
            s1_idx = shift_name_to_index[s1]
            s2_idx = shift_name_to_index[s2]
            forbidden_pairs.append((s1_idx, s2_idx))
    
    model = gp.Model('nurse_scheduling')
    
    x = {}
    work = {}
    z = {}
    
    for n in range(num_nurses):
        for d in range(num_days):
            work_key = (n, d)
            work[work_key] = model.addVar(vtype=GRB.BINARY, name=f'work_{n}_{d}')
            for s in range(num_shifts):
                z_key = (n, d, s)
                z[z_key] = model.addVar(vtype=GRB.BINARY, name=f'z_{n}_{d}_{s}')
                for k in nurse_skills[n]:
                    x_key = (n, d, s, k)
                    x[x_key] = model.addVar(vtype=GRB.BINARY, name=f'x_{n}_{d}_{s}_{k}')
    
    shortfall = {}
    for d in range(num_days):
        for s in range(num_shifts):
            for k in range(num_skills):
                shortfall_key = (d, s, k)
                shortfall[shortfall_key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'shortfall_{d}_{s}_{k}')
    
    L_work = {}
    end_work = {}
    under_work = {}
    over_work = {}
    penalty_work_stretch = {}
    
    for n in range(num_nurses):
        for d in range(num_days):
            key = (n, d)
            L_work[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'L_work_{n}_{d}')
            end_work[key] = model.addVar(vtype=GRB.BINARY, name=f'end_work_{n}_{d}')
            under_work[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'under_work_{n}_{d}')
            over_work[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'over_work_{n}_{d}')
            penalty_work_stretch[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'penalty_work_{n}_{d}')
    
    L_rest = {}
    end_rest = {}
    under_rest = {}
    over_rest = {}
    penalty_rest_stretch = {}
    
    for n in range(num_nurses):
        for d in range(num_days):
            key = (n, d)
            L_rest[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'L_rest_{n}_{d}')
            end_rest[key] = model.addVar(vtype=GRB.BINARY, name=f'end_rest_{n}_{d}')
            under_rest[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'under_rest_{n}_{d}')
            over_rest[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'over_rest_{n}_{d}')
            penalty_rest_stretch[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'penalty_rest_{n}_{d}')
    
    L_shift = {}
    end_shift = {}
    under_shift = {}
    over_shift = {}
    penalty_shift_stretch = {}
    
    for n in range(num_nurses):
        for d in range(num_days):
            for s in range(num_shifts):
                key = (n, d, s)
                L_shift[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'L_shift_{n}_{d}_{s}')
                end_shift[key] = model.addVar(vtype=GRB.BINARY, name=f'end_shift_{n}_{d}_{s}')
                under_shift[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'under_shift_{n}_{d}_{s}')
                over_shift[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'over_shift_{n}_{d}_{s}')
                penalty_shift_stretch[key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'penalty_shift_{n}_{d}_{s}')
    
    work_weekend = {}
    both_worked = {}
    
    for n in range(num_nurses):
        for w in range(num_weeks):
            key = (n, w)
            work_weekend[key] = model.addVar(vtype=GRB.BINARY, name=f'work_weekend_{n}_{w}')
            both_worked[key] = model.addVar(vtype=GRB.BINARY, name=f'both_worked_{n}_{w}')
    
    total_work = {}
    under_total = {}
    over_total = {}
    penalty_total = {}
    over_weekends = {}
    penalty_weekends = {}
    
    for n in range(num_nurses):
        total_work[n] = model.addVar(vtype=GRB.CONTINUOUS, name=f'total_work_{n}')
        under_total[n] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'under_total_{n}')
        over_total[n] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'over_total_{n}')
        penalty_total[n] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'penalty_total_{n}')
        over_weekends[n] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'over_weekends_{n}')
        penalty_weekends[n] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'penalty_weekends_{n}')
    
    model.update()
    
    for d in range(num_days):
        for s in range(num_shifts):
            for k in range(num_skills):
                if min_demand[d, s, k] > 0:
                    nurses_with_skill = [n for n in range(num_nurses) if k in nurse_skills[n]]
                    model.addConstr(
                        gp.quicksum(x.get((n, d, s, k), 0) for n in nurses_with_skill) >= min_demand[d, s, k],
                        name=f'demand_min_{d}_{s}_{k}'
                    )
                model.addConstr(
                    shortfall[(d, s, k)] >= opt_demand[d, s, k] - gp.quicksum(x.get((n, d, s, k), 0) for n in range(num_nurses) if k in nurse_skills[n]),
                    name=f'shortfall_{d}_{s}_{k}'
                )
    
    for n in range(num_nurses):
        for d in range(num_days):
            work_key = (n, d)
            expr = gp.LinExpr()
            for s in range(num_shifts):
                expr_s = gp.LinExpr()
                for k in nurse_skills[n]:
                    x_key = (n, d, s, k)
                    if x_key in x:
                        expr += x[x_key]
                        expr_s += x[x_key]
                model.addConstr(z[(n, d, s)] == expr_s, name=f'z_{n}_{d}_{s}')
            model.addConstr(work[work_key] == expr, name=f'work_{n}_{d}')
    
    for n in range(num_nurses):
        for d in range(num_days - 1):
            for (s1, s2) in forbidden_pairs:
                model.addConstr(z[(n, d, s1)] + z[(n, d+1, s2)] <= 1, name=f'forbidden_{n}_{d}_{s1}_{s2}')
    
    for n in range(num_nurses):
        contract = nurses[n]['contract']
        CD_minus = contract['CD_minus']
        CD_plus = contract['CD_plus']
        for d in range(num_days):
            work_key = (n, d)
            if d == 0:
                model.addGenConstrIndicator(work[work_key], True, L_work[(n, d)] == 1, name=f'L_work_ind1_{n}_{d}')
                model.addGenConstrIndicator(work[work_key], False, L_work[(n, d)] == 0, name=f'L_work_ind0_{n}_{d}')
            else:
                prev_key = (n, d-1)
                model.addGenConstrIndicator(work[work_key], True, L_work[(n, d)] == L_work[prev_key] + 1, name=f'L_work_ind1_{n}_{d}')
                model.addGenConstrIndicator(work[work_key], False, L_work[(n, d)] == 0, name=f'L_work_ind0_{n}_{d}')
            
            if d == num_days - 1:
                model.addConstr(end_work[(n, d)] == work[work_key], name=f'end_work_{n}_{d}')
            else:
                next_key = (n, d+1)
                model.addConstr(end_work[(n, d)] <= work[work_key], name=f'end_work_ub1_{n}_{d}')
                model.addConstr(end_work[(n, d)] <= 1 - work[next_key], name=f'end_work_ub2_{n}_{d}')
                model.addConstr(end_work[(n, d)] >= work[work_key] - work[next_key], name=f'end_work_lb_{n}_{d}')
            
            model.addConstr(under_work[(n, d)] >= CD_minus - L_work[(n, d)], name=f'under_work_{n}_{d}')
            model.addConstr(over_work[(n, d)] >= L_work[(n, d)] - CD_plus, name=f'over_work_{n}_{d}')
            
            M = penalty_weights['c_S2a'] * (CD_minus + CD_plus)
            model.addConstr(penalty_work_stretch[(n, d)] <= penalty_weights['c_S2a'] * (under_work[(n, d)] + over_work[(n, d)]), 
                           name=f'penalty_work_ub1_{n}_{d}')
            model.addConstr(penalty_work_stretch[(n, d)] <= M * end_work[(n, d)], 
                           name=f'penalty_work_ub2_{n}_{d}')
            model.addConstr(penalty_work_stretch[(n, d)] >= penalty_weights['c_S2a'] * (under_work[(n, d)] + over_work[(n, d)]) - M * (1 - end_work[(n, d)]), 
                           name=f'penalty_work_lb_{n}_{d}')
    
    for n in range(num_nurses):
        contract = nurses[n]['contract']
        CR_minus = contract['CR_minus']
        CR_plus = contract['CR_plus']
        for d in range(num_days):
            work_key = (n, d)
            if d == 0:
                model.addGenConstrIndicator(work[work_key], False, L_rest[(n, d)] == 1, name=f'L_rest_ind1_{n}_{d}')
                model.addGenConstrIndicator(work[work_key], True, L_rest[(n, d)] == 0, name=f'L_rest_ind0_{n}_{d}')
            else:
                prev_key = (n, d-1)
                model.addGenConstrIndicator(work[work_key], False, L_rest[(n, d)] == L_rest[prev_key] + 1, name=f'L_rest_ind1_{n}_{d}')
                model.addGenConstrIndicator(work[work_key], True, L_rest[(n, d)] == 0, name=f'L_rest_ind0_{n}_{d}')
            
            if d == num_days - 1:
                model.addConstr(end_rest[(n, d)] == 1 - work[work_key], name=f'end_rest_{n}_{d}')
            else:
                next_key = (n, d+1)
                model.addConstr(end_rest[(n, d)] <= 1 - work[work_key], name=f'end_rest_ub1_{n}_{d}')
                model.addConstr(end_rest[(n, d)] <= work[next_key], name=f'end_rest_ub2_{n}_{d}')
                model.addConstr(end_rest[(n, d)] >= (1 - work[work_key]) + work[next_key] - 1, name=f'end_rest_lb_{n}_{d}')
            
            model.addConstr(under_rest[(n, d)] >= CR_minus - L_rest[(n, d)], name=f'under_rest_{n}_{d}')
            model.addConstr(over_rest[(n, d)] >= L_rest[(n, d)] - CR_plus, name=f'over_rest_{n}_{d}')
            
            M = penalty_weights['c_S3'] * (CR_minus + CR_plus)
            model.addConstr(penalty_rest_stretch[(n, d)] <= penalty_weights['c_S3'] * (under_rest[(n, d)] + over_rest[(n, d)]), 
                           name=f'penalty_rest_ub1_{n}_{d}')
            model.addConstr(penalty_rest_stretch[(n, d)] <= M * end_rest[(n, d)], 
                           name=f'penalty_rest_ub2_{n}_{d}')
            model.addConstr(penalty_rest_stretch[(n, d)] >= penalty_weights['c_S3'] * (under_rest[(n, d)] + over_rest[(n, d)]) - M * (1 - end_rest[(n, d)]), 
                           name=f'penalty_rest_lb_{n}_{d}')
    
    for n in range(num_nurses):
        for d in range(num_days):
            for s in range(num_shifts):
                shift_info = shifts[s]
                CS_minus = shift_info['CS_minus']
                CS_plus = shift_info['CS_plus']
                z_key = (n, d, s)
                L_shift_key = (n, d, s)
                if d == 0:
                    model.addGenConstrIndicator(z[z_key], True, L_shift[L_shift_key] == 1, name=f'L_shift_ind1_{n}_{d}_{s}')
                    model.addGenConstrIndicator(z[z_key], False, L_shift[L_shift_key] == 0, name=f'L_shift_ind0_{n}_{d}_{s}')
                else:
                    prev_key = (n, d-1, s)
                    model.addGenConstrIndicator(z[z_key], True, L_shift[L_shift_key] == L_shift[prev_key] + 1, name=f'L_shift_ind1_{n}_{d}_{s}')
                    model.addGenConstrIndicator(z[z_key], False, L_shift[L_shift_key] == 0, name=f'L_shift_ind0_{n}_{d}_{s}')
                
                if d == num_days - 1:
                    model.addConstr(end_shift[(n, d, s)] == z[z_key], name=f'end_shift_{n}_{d}_{s}')
                else:
                    next_key = (n, d+1, s)
                    model.addConstr(end_shift[(n, d, s)] <= z[z_key], name=f'end_shift_ub1_{n}_{d}_{s}')
                    model.addConstr(end_shift[(n, d, s)] <= 1 - z[next_key], name=f'end_shift_ub2_{n}_{d}_{s}')
                    model.addConstr(end_shift[(n, d, s)] >= z[z_key] - z[next_key], name=f'end_shift_lb_{n}_{d}_{s}')
                
                model.addConstr(under_shift[(n, d, s)] >= CS_minus - L_shift[(n, d, s)], name=f'under_shift_{n}_{d}_{s}')
                model.addConstr(over_shift[(n, d, s)] >= L_shift[(n, d, s)] - CS_plus, name=f'over_shift_{n}_{d}_{s}')
                
                M = penalty_weights['c_S2b'] * (CS_minus + CS_plus)
                model.addConstr(penalty_shift_stretch[(n, d, s)] <= penalty_weights['c_S2b'] * (under_shift[(n, d, s)] + over_shift[(n, d, s)]), 
                               name=f'penalty_shift_ub1_{n}_{d}_{s}')
                model.addConstr(penalty_shift_stretch[(n, d, s)] <= M * end_shift[(n, d, s)], 
                               name=f'penalty_shift_ub2_{n}_{d}_{s}')
                model.addConstr(penalty_shift_stretch[(n, d, s)] >= penalty_weights['c_S2b'] * (under_shift[(n, d, s)] + over_shift[(n, d, s)]) - M * (1 - end_shift[(n, d, s)]), 
                               name=f'penalty_shift_lb_{n}_{d}_{s}')
    
    for n in range(num_nurses):
        for w in range(num_weeks):
            sat = 7 * w + 5
            sun = 7 * w + 6
            key = (n, w)
            if sat < num_days and sun < num_days:
                model.addConstr(work_weekend[key] >= work[(n, sat)], name=f'work_weekend_ub1_{n}_{w}')
                model.addConstr(work_weekend[key] >= work[(n, sun)], name=f'work_weekend_ub2_{n}_{w}')
                model.addConstr(work_weekend[key] <= work[(n, sat)] + work[(n, sun)], name=f'work_weekend_ub3_{n}_{w}')
                model.addConstr(both_worked[key] <= work[(n, sat)], name=f'both_worked_ub1_{n}_{w}')
                model.addConstr(both_worked[key] <= work[(n, sun)], name=f'both_worked_ub2_{n}_{w}')
                model.addConstr(both_worked[key] >= work[(n, sat)] + work[(n, sun)] - 1, name=f'both_worked_lb_{n}_{w}')
            elif sat < num_days:
                model.addConstr(work_weekend[key] == work[(n, sat)], name=f'work_weekend_{n}_{w}')
                model.addConstr(both_worked[key] == 0, name=f'both_worked_{n}_{w}')
            elif sun < num_days:
                model.addConstr(work_weekend[key] == work[(n, sun)], name=f'work_weekend_{n}_{w}')
                model.addConstr(both_worked[key] == 0, name=f'both_worked_{n}_{w}')
            else:
                model.addConstr(work_weekend[key] == 0, name=f'work_weekend_{n}_{w}')
                model.addConstr(both_worked[key] == 0, name=f'both_worked_{n}_{w}')
    
    for n in range(num_nurses):
        contract = nurses[n]['contract']
        L_minus = contract['L_minus']
        L_plus = contract['L_plus']
        model.addConstr(total_work[n] == gp.quicksum(work[(n, d)] for d in range(num_days)), name=f'total_work_{n}')
        model.addConstr(under_total[n] >= max(0, L_minus - total_work[n]), name=f'under_total_{n}')
        model.addConstr(over_total[n] >= max(0, total_work[n] - L_plus), name=f'over_total_{n}')
        model.addConstr(penalty_total[n] == penalty_weights['c_S6'] * (under_total[n] + over_total[n]), name=f'penalty_total_{n}')
    
    for n in range(num_nurses):
        contract = nurses[n]['contract']
        WE_plus = contract['WE_plus']
        model.addConstr(over_weekends[n] == gp.quicksum(work_weekend[(n, w)] for w in range(num_weeks)) - WE_plus, name=f'over_weekends_{n}')
        model.addConstr(over_weekends[n] >= 0, name=f'over_weekends_lb_{n}')
        model.addConstr(penalty_weekends[n] == penalty_weights['c_S7'] * over_weekends[n], name=f'penalty_weekends_{n}')
    
    obj = gp.quicksum(shortfall.values()) * penalty_weights['c_S1']
    
    obj += gp.quicksum(penalty_work_stretch.values())
    obj += gp.quicksum(penalty_shift_stretch.values())
    obj += gp.quicksum(penalty_rest_stretch.values())
    
    for n_idx, nurse in enumerate(nurses):
        for pref in nurse['preferences']:
            day = pref['day']
            shift_name = pref['shift']
            if day < num_days and shift_name in shift_name_to_index:
                s_idx = shift_name_to_index[shift_name]
                obj += penalty_weights['c_S4'] * z[(n_idx, day, s_idx)]
    
    for n in range(num_nurses):
        for w in range(num_weeks):
            sat = 7 * w + 5
            sun = 7 * w + 6
            if sat < num_days and sun < num_days:
                obj += penalty_weights['c_S5'] * (work[(n, sat)] + work[(n, sun)] - 2 * both_worked[(n, w)])
            elif sat < num_days:
                obj += penalty_weights['c_S5'] * work[(n, sat)]
            elif sun < num_days:
                obj += penalty_weights['c_S5'] * work[(n, sun)]
    
    obj += gp.quicksum(penalty_total.values())
    obj += gp.quicksum(penalty_weekends.values())
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    try:
        if logger:
            model.optimize(callback)
        else:
            model.optimize()
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        schedule = []
        for n_idx, nurse in enumerate(nurses):
            assignments = []
            for d in range(num_days):
                assigned = False
                for s in range(num_shifts):
                    for k in nurse_skills[n_idx]:
                        x_key = (n_idx, d, s, k)
                        if x_key in x and x[x_key].X > 0.5:
                            shift_name = shift_index_to_name[s]
                            skill_name = skill_index_to_name[k]
                            assignments.append({
                                'day': d,
                                'shift': shift_name,
                                'skill': skill_name
                            })
                            assigned = True
                            break
                    if assigned:
                        break
            schedule.append({
                'nurse_id': nurse['id'],
                'nurse_name': nurse['name'],
                'assignments': assignments
            })
        
        solution = {
            'objective_value': model.ObjVal,
            'schedule': schedule
        }
    else:
        schedule = []
        for nurse in nurses:
            schedule.append({
                'nurse_id': nurse['id'],
                'nurse_name': nurse['name'],
                'assignments': []
            })
        solution = {
            'objective_value': None,
            'schedule': schedule
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
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)