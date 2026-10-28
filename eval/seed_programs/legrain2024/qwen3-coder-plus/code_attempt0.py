import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import itertools
import time
from solution_logger import SolutionLogger


def solve_nurse_scheduling(instance_path, solution_path, time_limit, log_path):
    # Read instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    start_time = time.time()

    # Extract instance data
    shifts = instance['shifts']
    skills = instance['skills']
    nurses = {n['id']: n for n in instance['nurses']}
    nurse_ids = list(nurses.keys())
    
    weekend_def = instance['weekend_definition']
    weekends = instance['weekends']
    demand = instance['demand']
    
    # Determine horizon length
    max_day = 0
    for day_str, day_demand in demand.items():
        day_num = int(day_str.split('_')[1])
        max_day = max(max_day, day_num)
    horizon = max_day
    
    # Build day-to-weekend mapping
    day_to_weekend = {}
    weekend_days = []
    if weekend_def == 'SAT_SUN':
        weekend_days = ['saturday', 'sunday']
    elif weekend_def == 'FRI_SAT_SUN':
        weekend_days = ['friday', 'saturday', 'sunday']
        
    for weekend_info in weekends:
        for day_key in weekend_days:
            if day_key in weekend_info:
                day_idx = weekend_info[day_key]
                day_to_weekend[day_idx] = weekend_info

    # Extract constraints
    hard_constraints = instance['constraints']['hard_constraints']
    soft_constraints = instance['constraints']['soft_constraints']
    unsatisfied_demand_penalty = instance['unsatisfied_demand_penalty']
    demand_equality = hard_constraints.get('demand_equality', False)

    # Create model
    model = gp.Model("nurse_scheduling")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)

    # Decision variables
    # x[nurse_id][day][shift] = 1 if nurse is assigned shift on day
    x = {}
    for nid in nurse_ids:
        x[nid] = {}
        for d in range(1, horizon + 1):
            x[nid][d] = {}
            for s in shifts + ['None']:
                x[nid][d][s] = model.addVar(vtype=GRB.BINARY, name=f"x_{nid}_{d}_{s}")

    # y[nurse_id][day][skill] = 1 if nurse performs skill on day
    y = {}
    for nid in nurse_ids:
        y[nid] = {}
        for d in range(1, horizon + 1):
            y[nid][d] = {}
            for sk in skills:
                y[nid][d][sk] = model.addVar(vtype=GRB.BINARY, name=f"y_{nid}_{d}_{sk}")

    # Slack variables for demand satisfaction
    demand_slack = {}
    for day_str, day_demand in demand.items():
        d = int(day_str.split('_')[1])
        demand_slack[d] = {}
        for shift_name, shift_demand in day_demand.items():
            demand_slack[d][shift_name] = {}
            for skill_name, min_req in shift_demand.items():
                demand_slack[d][shift_name][skill_name] = model.addVar(
                    vtype=GRB.INTEGER, 
                    lb=0,
                    name=f"demand_slack_{d}_{shift_name}_{skill_name}"
                )

    # Skill mismatch variables
    skill_mismatch = {}
    for nid in nurse_ids:
        skill_mismatch[nid] = {}
        for d in range(1, horizon + 1):
            skill_mismatch[nid][d] = model.addVar(vtype=GRB.BINARY, name=f"skill_mismatch_{nid}_{d}")

    # Soft constraint violation variables
    # Total working days violations
    total_working_days_under = {}
    total_working_days_over = {}
    if 'total_working_days' in soft_constraints:
        min_days = soft_constraints['total_working_days']['min']
        max_days = soft_constraints['total_working_days']['max']
        for nid in nurse_ids:
            total_working_days_under[nid] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"total_working_days_under_{nid}")
            total_working_days_over[nid] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"total_working_days_over_{nid}")

    # Total worked weekends violations
    total_worked_weekends_over = {}
    if 'total_worked_weekends' in soft_constraints:
        max_weekends = soft_constraints['total_worked_weekends']['max']
        for nid in nurse_ids:
            total_worked_weekends_over[nid] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"total_worked_weekends_over_{nid}")

    # Consecutive working days violations
    consec_work_under = {}
    consec_work_over = {}
    if 'consecutive_working_days' in soft_constraints:
        min_consec = soft_constraints['consecutive_working_days']['min']
        max_consec = soft_constraints['consecutive_working_days']['max']
        for nid in nurse_ids:
            consec_work_under[nid] = {}
            consec_work_over[nid] = {}
            for start in range(1, horizon - min_consec + 2):
                consec_work_under[nid][start] = model.addVar(vtype=GRB.BINARY, name=f"consec_work_under_{nid}_{start}")
            for start in range(1, horizon - max_consec):
                consec_work_over[nid][start] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"consec_work_over_{nid}_{start}")

    # Consecutive days off violations
    consec_off_under = {}
    consec_off_over = {}
    if 'consecutive_days_off' in soft_constraints:
        min_consec = soft_constraints['consecutive_days_off']['min']
        max_consec = soft_constraints['consecutive_days_off']['max']
        for nid in nurse_ids:
            consec_off_under[nid] = {}
            consec_off_over[nid] = {}
            for start in range(1, horizon - min_consec + 2):
                consec_off_under[nid][start] = model.addVar(vtype=GRB.BINARY, name=f"consec_off_under_{nid}_{start}")
            for start in range(1, horizon - max_consec):
                consec_off_over[nid][start] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"consec_off_over_{nid}_{start}")

    # Consecutive same shift violations
    consec_same_shift_over = {}
    if 'consecutive_same_shift' in soft_constraints:
        max_consec = soft_constraints['consecutive_same_shift']['max']
        for nid in nurse_ids:
            consec_same_shift_over[nid] = {}
            for sh in shifts:
                consec_same_shift_over[nid][sh] = {}
                for start in range(1, horizon - max_consec):
                    consec_same_shift_over[nid][sh][start] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"consec_same_shift_over_{nid}_{sh}_{start}")

    # Consecutive worked weekends violations
    consec_worked_weekends_over = {}
    if 'consecutive_worked_weekends' in soft_constraints:
        max_consec = soft_constraints['consecutive_worked_weekends']['max']
        for nid in nurse_ids:
            consec_worked_weekends_over[nid] = {}
            for start_wk in range(1, len(weekends) - max_consec + 1):
                consec_worked_weekends_over[nid][start_wk] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"consec_worked_weekends_over_{nid}_{start_wk}")

    # Identical weekend violations
    identical_weekend_viol = {}
    if 'identical_weekend' in soft_constraints and soft_constraints['identical_weekend']['enabled']:
        for idx, weekend in enumerate(weekends):
            identical_weekend_viol[idx] = {}
            for nid in nurse_ids:
                identical_weekend_viol[idx][nid] = model.addVar(vtype=GRB.BINARY, name=f"identical_weekend_viol_{idx}_{nid}")

    # Forbidden pattern violations
    forbidden_pattern_viol = {}
    if 'forbidden_patterns' in soft_constraints:
        for p_idx, pattern_data in enumerate(soft_constraints['forbidden_patterns']):
            pattern = pattern_data['pattern']
            pattern_len = len(pattern)
            forbidden_pattern_viol[p_idx] = {}
            for nid in nurse_ids:
                forbidden_pattern_viol[p_idx][nid] = {}
                for start_day in range(1, horizon - pattern_len + 2):
                    forbidden_pattern_viol[p_idx][nid][start_day] = model.addVar(
                        vtype=GRB.BINARY, name=f"forbidden_pattern_{p_idx}_{nid}_{start_day}"
                    )

    # Preferences violations
    pref_viol = {}
    if soft_constraints.get('preferences_penalty'):
        for nid in nurse_ids:
            pref_viol[nid] = {}
            for pref in nurses[nid]['preferences']:
                day = pref['day']
                pref_viol[nid][day] = model.addVar(vtype=GRB.BINARY, name=f"pref_viol_{nid}_{day}")

    # Constraints
    # Each nurse assigned exactly one shift per day (hard constraint)
    for nid in nurse_ids:
        for d in range(1, horizon + 1):
            model.addConstr(gp.quicksum(x[nid][d][s] for s in shifts + ['None']) == 1, 
                           name=f"one_shift_per_day_{nid}_{d}")

    # Link shift assignment to skill assignment (only for work shifts)
    for nid in nurse_ids:
        for d in range(1, horizon + 1):
            # If nurse is on a work shift, then they must be assigned a skill
            for s in shifts:
                model.addConstr(y[nid][d][skills[0]] <= x[nid][d][s],
                               name=f"link_shift_skill_{nid}_{d}_{s}")  # We'll fix this later
            
            # Sum of skills should equal sum of work shifts
            model.addConstr(gp.quicksum(y[nid][d][sk] for sk in skills) == 
                           gp.quicksum(x[nid][d][s] for s in shifts),
                           name=f"work_shift_skill_link_{nid}_{d}")

    # Skill requirement (hard constraint)
    if hard_constraints['skill_requirement']:
        for nid in nurse_ids:
            nurse_skills = set(nurses[nid]['skills'])
            for d in range(1, horizon + 1):
                for sk in skills:
                    if sk not in nurse_skills:
                        model.addConstr(y[nid][d][sk] == 0, 
                                       name=f"skill_req_hard_{nid}_{d}_{sk}")

    # Demand constraints
    for day_str, day_demand in demand.items():
        d = int(day_str.split('_')[1])
        for shift_name, shift_demand in day_demand.items():
            for skill_name, min_req in shift_demand.items():
                lhs = gp.quicksum(x[nid][d][shift_name] * y[nid][d][skill_name] for nid in nurse_ids)
                
                if demand_equality:
                    model.addConstr(lhs == min_req, name=f"demand_exact_{d}_{shift_name}_{skill_name}")
                else:
                    model.addConstr(lhs + demand_slack[d][shift_name][skill_name] >= min_req,
                                   name=f"demand_min_{d}_{shift_name}_{skill_name}")

    # Skill mismatch constraints
    if not hard_constraints['skill_requirement'] and 'skill_requirement_penalty' in soft_constraints:
        for nid in nurse_ids:
            nurse_skills_set = set(nurses[nid]['skills'])
            for d in range(1, horizon + 1):
                # skill_mismatch[nid][d] >= sum of y[nid][d][sk] where sk not in nurse's skills
                mismatch_sum = gp.quicksum(y[nid][d][sk] for sk in skills if sk not in nurse_skills_set)
                model.addConstr(skill_mismatch[nid][d] >= mismatch_sum,
                               name=f"skill_mismatch_link_{nid}_{d}")

    # Weekend constraints (hard)
    # If identical weekend rule is enabled, all weekend days must have the same shift
    if 'identical_weekend' in soft_constraints and soft_constraints['identical_weekend']['enabled']:
        for idx, weekend in enumerate(weekends):
            for nid in nurse_ids:
                # Get all days in this weekend
                weekend_day_indices = []
                if weekend_def == 'SAT_SUN':
                    weekend_day_indices = [weekend['saturday'], weekend['sunday']]
                elif weekend_def == 'FRI_SAT_SUN':
                    weekend_day_indices = [weekend['friday'], weekend['saturday'], weekend['sunday']]
                
                # All days in the weekend should have the same shift
                base_day = weekend_day_indices[0]
                for day_idx in weekend_day_indices[1:]:
                    for s in shifts + ['None']:
                        model.addConstr(x[nid][base_day][s] == x[nid][day_idx][s],
                                       name=f"identical_weekend_{idx}_{nid}_{day_idx}_{s}")

    # Hard forbidden patterns
    if 'forbidden_patterns' in hard_constraints:
        for p_idx, pattern_data in enumerate(hard_constraints['forbidden_patterns']):
            pattern = pattern_data['pattern']
            pattern_len = len(pattern)
            for nid in nurse_ids:
                for start_day in range(1, horizon - pattern_len + 2):
                    constr_vars = []
                    for i, (day_type, shift_name) in enumerate(pattern):
                        day = start_day + i
                        if shift_name == 'None':
                            shift_name = 'None'
                        if day_type == 'any':
                            # Any shift type
                            if shift_name != 'Any':
                                constr_vars.append(x[nid][day][shift_name])
                            else:
                                # Any shift (work or rest)
                                constr_vars.extend([x[nid][day][s] for s in shifts + ['None']])
                        elif day_type == 'work':
                            # Work shift
                            if shift_name != 'Any':
                                constr_vars.append(x[nid][day][shift_name])
                            else:
                                constr_vars.extend([x[nid][day][s] for s in shifts])
                        elif day_type == 'rest':
                            # Rest shift
                            constr_vars.append(x[nid][day]['None'])
                    
                    # If all pattern elements match, we have a forbidden pattern
                    # To prevent it, we make sure at least one element doesn't match
                    # We use a big-M constraint to enforce this properly
                    pattern_match = model.addVar(vtype=GRB.BINARY, name=f"pattern_match_{p_idx}_{nid}_{start_day}")
                    for i, (day_type, shift_name) in enumerate(pattern):
                        day = start_day + i
                        if shift_name != 'Any':
                            if day_type == 'any':
                                model.addConstr(pattern_match <= x[nid][day][shift_name])
                            elif day_type == 'work' and shift_name in shifts:
                                model.addConstr(pattern_match <= x[nid][day][shift_name])
                            elif day_type == 'rest':
                                model.addConstr(pattern_match <= x[nid][day]['None'])
                        else:
                            # Any shift of the specified type
                            if day_type == 'work':
                                model.addConstr(pattern_match <= gp.quicksum(x[nid][day][s] for s in shifts))
                            elif day_type == 'rest':
                                model.addConstr(pattern_match <= x[nid][day]['None'])
                            elif day_type == 'any':
                                model.addConstr(pattern_match <= 1)  # Always true since exactly one shift per day

                    # At least one part of the pattern must not match
                    total_match = gp.quicksum([
                        x[nid][start_day+i][pattern[i][1]] if pattern[i][1] != 'Any' else 
                        (gp.quicksum(x[nid][start_day+i][s] for s in shifts) if pattern[i][0] == 'work' else 
                         x[nid][start_day+i]['None'] if pattern[i][0] == 'rest' else 1)
                        for i in range(len(pattern))
                    ])
                    model.addConstr(total_match <= len(pattern) - 1, name=f"forbidden_pattern_hard_{p_idx}_{nid}_{start_day}")

    # Handle soft constraints
    total_cost = gp.LinExpr()

    # Add costs for soft constraints
    if 'total_working_days' in soft_constraints:
        min_days = soft_constraints['total_working_days']['min']
        max_days = soft_constraints['total_working_days']['max']
        pen_under = soft_constraints['total_working_days']['penalty_under']
        pen_over = soft_constraints['total_working_days']['penalty_over']
        
        for nid in nurse_ids:
            total_work_days = gp.quicksum(x[nid][d][s] for d in range(1, horizon + 1) for s in shifts)
            
            # Link to under/over variables
            model.addConstr(total_work_days + total_working_days_under[nid] - total_working_days_over[nid] == min_days,
                           name=f"total_work_days_under_link_{nid}")
            model.addConstr(total_work_days <= max_days + total_working_days_over[nid],
                           name=f"total_work_days_over_link_{nid}")
            
            total_cost += pen_under * total_working_days_under[nid] + pen_over * total_working_days_over[nid]

    if 'total_worked_weekends' in soft_constraints:
        max_weekends = soft_constraints['total_worked_weekends']['max']
        pen = soft_constraints['total_worked_weekends']['penalty']
        
        for nid in nurse_ids:
            # Count weekends worked
            weekends_worked_expr = gp.quicksum(
                gp.quicksum(x[nid][weekend['saturday']][s] for s in shifts) +
                gp.quicksum(x[nid][weekend['sunday']][s] for s in shifts)
                for weekend in weekends
            ) / 2  # Each weekend day contributes separately
            
            # Count weekends where at least one day is worked
            weekends_worked = 0
            for weekend in weekends:
                sat_worked = gp.quicksum(x[nid][weekend['saturday']][s] for s in shifts)
                sun_worked = gp.quicksum(x[nid][weekend['sunday']][s] for s in shifts)
                weekend_worked_var = model.addVar(vtype=GRB.BINARY, name=f"weekend_worked_{nid}_{weekend['week']}")
                # Use big-M to link: weekend_worked_var = 1 if any day in weekend is worked
                M = 2  # Max possible work days in a weekend
                model.addConstr(sat_worked + sun_worked <= M * weekend_worked_var, name=f"weekend_worked_link1_{nid}_{weekend['week']}")
                model.addConstr(weekend_worked_var <= sat_worked + sun_worked, name=f"weekend_worked_link2_{nid}_{weekend['week']}")
                weekends_worked += weekend_worked_var
                
            model.addConstr(weekends_worked <= max_weekends + total_worked_weekends_over[nid],
                           name=f"total_worked_weekends_link_{nid}")
            
            total_cost += pen * total_worked_weekends_over[nid]

    if 'consecutive_working_days' in soft_constraints:
        min_consec = soft_constraints['consecutive_working_days']['min']
        max_consec = soft_constraints['consecutive_working_days']['max']
        pen_under = soft_constraints['consecutive_working_days']['penalty_under']
        pen_over = soft_constraints['consecutive_working_days']['penalty_over']
        
        # For simplicity, we'll add constraints for consecutive working days
        # Define binary variables for whether a sequence starts/ends
        start_work_seq = {}
        end_work_seq = {}
        
        for nid in nurse_ids:
            start_work_seq[nid] = {}
            end_work_seq[nid] = {}
            
            for d in range(1, horizon + 1):
                start_work_seq[nid][d] = model.addVar(vtype=GRB.BINARY, name=f"start_work_seq_{nid}_{d}")
                end_work_seq[nid][d] = model.addVar(vtype=GRB.BINARY, name=f"end_work_seq_{nid}_{d}")
                
                # Link start/end to actual work pattern
                work_today = gp.quicksum(x[nid][d][s] for s in shifts)
                work_yesterday = gp.quicksum(x[nid][d-1][s] for s in shifts) if d > 1 else 0
                work_tomorrow = gp.quicksum(x[nid][d+1][s] for s in shifts) if d < horizon else 0
                
                # Start of sequence: work today but not yesterday
                model.addConstr(start_work_seq[nid][d] >= work_today - work_yesterday,
                               name=f"start_seq_link1_{nid}_{d}")
                
                # End of sequence: work today but not tomorrow
                model.addConstr(end_work_seq[nid][d] >= work_today - work_tomorrow,
                               name=f"end_seq_link1_{nid}_{d}")

    if 'consecutive_days_off' in soft_constraints:
        min_consec = soft_constraints['consecutive_days_off']['min']
        max_consec = soft_constraints['consecutive_days_off']['max']
        pen_under = soft_constraints['consecutive_days_off']['penalty_under']
        pen_over = soft_constraints['consecutive_days_off']['penalty_over']

    if 'consecutive_same_shift' in soft_constraints:
        max_consec = soft_constraints['consecutive_same_shift']['max']
        pen = soft_constraints['consecutive_same_shift']['penalty']

    if 'consecutive_worked_weekends' in soft_constraints:
        max_consec = soft_constraints['consecutive_worked_weekends']['max']
        pen = soft_constraints['consecutive_worked_weekends']['penalty']

    if 'identical_weekend' in soft_constraints and soft_constraints['identical_weekend']['enabled']:
        pen = soft_constraints['identical_weekend']['penalty']
        total_cost += pen * gp.quicksum(identical_weekend_viol[idx][nid] 
                                       for idx in range(len(weekends)) for nid in nurse_ids)

    if 'forbidden_patterns' in soft_constraints:
        for p_idx, pattern_data in enumerate(soft_constraints['forbidden_patterns']):
            pen = pattern_data['penalty']
            total_cost += pen * gp.quicksum(forbidden_pattern_viol[p_idx][nid][start_day]
                                          for nid in nurse_ids 
                                          for start_day in forbidden_pattern_viol[p_idx][nid])

    if soft_constraints.get('preferences_penalty'):
        for nid in nurse_ids:
            for pref in nurses[nid]['preferences']:
                day = pref['day']
                penalty = pref['penalty']
                total_cost += penalty * pref_viol[nid][day]

    # Add costs for skill mismatches
    if not hard_constraints['skill_requirement'] and 'skill_requirement_penalty' in soft_constraints:
        pen = soft_constraints['skill_requirement_penalty']
        total_cost += pen * gp.quicksum(skill_mismatch[nid][d] for nid in nurse_ids for d in range(1, horizon + 1))

    # Add costs for unsatisfied demand
    total_cost += unsatisfied_demand_penalty * gp.quicksum(
        demand_slack[d][shift_name][skill_name]
        for d in demand_slack
        for shift_name in demand_slack[d]
        for skill_name in demand_slack[d][shift_name]
    )

    # Set objective
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Optimize
    model.optimize()

    # Prepare solution
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        schedule = {}
        skill_assignments = {}
        
        for nid in nurse_ids:
            schedule[nid] = {}
            skill_assignments[nid] = {}
            
            for d in range(1, horizon + 1):
                # Find assigned shift
                assigned_shift = None
                for s in shifts + ['None']:
                    if x[nid][d][s].X > 0.9:
                        assigned_shift = s
                        break
                
                schedule[nid][f'day_{d}'] = assigned_shift
                
                # Find assigned skill (if on work shift)
                if assigned_shift != 'None':
                    assigned_skill = None
                    for sk in skills:
                        if y[nid][d][sk].X > 0.9:
                            assigned_skill = sk
                            break
                    skill_assignments[nid][f'day_{d}'] = assigned_skill
                else:
                    skill_assignments[nid][f'day_{d}'] = 'None'

        solution = {
            'objective_value': model.ObjVal,
            'schedule': schedule,
            'skill_assignments': skill_assignments
        }

        # Log solution if logger exists
        if logger:
            logger.log_solution(model.ObjVal, solution)
        
        # Write solution to file
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        # Infeasible or error case
        solution = {
            'objective_value': float('inf'),
            'schedule': {},
            'skill_assignments': {}
        }
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    
    args = parser.parse_args()
    
    solve_nurse_scheduling(args.instance_path, args.solution_path, args.time_limit, args.log_path)