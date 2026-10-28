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
    if 'identical_weekend' in soft_constraints and soft_constraints['identical_weekend'].get('enabled', False):
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
                # This constraint ensures that if x[nid][d][s] = 1 for any work shift s, 
                # then exactly one skill is assigned for that day
                model.addConstr(gp.quicksum(y[nid][d][sk] for sk in skills) >= x[nid][d][s],
                               name=f"link_shift_skill_{nid}_{d}_{s}")
            
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
                # Sum of nurses assigned to this shift and skill on this day
                # For each nurse, day, and shift combination, we need to count if the nurse
                # is assigned to both the shift and the skill on that day
                # We linearize this: z[nid,d,shift,skill] = x[nid][d][shift_name] * y[nid][d][skill_name]
                expr = gp.LinExpr()
                for nid in nurse_ids:
                    # Create auxiliary variable for the product x[nid][d][shift_name] * y[nid][d][skill_name]
                    z_var_name = f"z_{nid}_{d}_{shift_name}_{skill_name}"
                    z = model.addVar(vtype=GRB.BINARY, name=z_var_name)
                    
                    # Linearization constraints for z = x * y
                    model.addConstr(z <= x[nid][d][shift_name])
                    model.addConstr(z <= y[nid][d][skill_name])
                    model.addConstr(z >= x[nid][d][shift_name] + y[nid][d][skill_name] - 1)
                    
                    expr += z
                
                if demand_equality:
                    model.addConstr(expr == min_req, name=f"demand_exact_{d}_{shift_name}_{skill_name}")
                else:
                    model.addConstr(expr + demand_slack[d][shift_name][skill_name] >= min_req,
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
    if 'identical_weekend' in soft_constraints and soft_constraints['identical_weekend'].get('enabled', False):
        for idx, weekend in enumerate(weekends):
            for nid in nurse_ids:
                # Get all days in this weekend
                weekend_day_indices = []
                if weekend_def == 'SAT_SUN':
                    weekend_day_indices = [weekend['saturday'], weekend['sunday']]
                elif weekend_def == 'FRI_SAT_SUN':
                    if 'friday' in weekend:
                        weekend_day_indices = [weekend['friday'], weekend['saturday'], weekend['sunday']]
                    else:
                        # Fallback if 'friday' is not present
                        weekend_day_indices = [weekend['saturday'], weekend['sunday']]
                
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
                    # Check if the pattern occurs starting at start_day
                    # We need to add constraints to prevent this pattern
                    # Create binary variables to detect if pattern matches
                    pattern_match = model.addVar(vtype=GRB.BINARY, name=f"pattern_match_{p_idx}_{nid}_{start_day}")
                    
                    # For each position in the pattern
                    for i, (day_type, shift_name) in enumerate(pattern):
                        day = start_day + i
                        
                        if shift_name != 'Any':
                            # Specific shift
                            if day_type == 'any':
                                # Direct match of shift
                                model.addConstr(pattern_match <= x[nid][day][shift_name], 
                                               name=f"pattern_match_direct_{p_idx}_{nid}_{start_day}_{i}")
                            elif day_type == 'work':
                                # Shift must be the specific work shift
                                model.addConstr(pattern_match <= x[nid][day][shift_name], 
                                               name=f"pattern_match_work_{p_idx}_{nid}_{start_day}_{i}")
                            elif day_type == 'rest':
                                # Shift must be rest
                                if shift_name == 'None':
                                    model.addConstr(pattern_match <= x[nid][day]['None'], 
                                                   name=f"pattern_match_rest_{p_idx}_{nid}_{start_day}_{i}")
                                else:
                                    # Impossible condition - rest day but non-None shift
                                    model.addConstr(pattern_match == 0, 
                                                   name=f"pattern_impossible_{p_idx}_{nid}_{start_day}_{i}")
                        else:
                            # Any shift - just check day type
                            if day_type == 'work':
                                # Any work shift is fine
                                model.addConstr(pattern_match <= gp.quicksum(x[nid][day][s] for s in shifts), 
                                               name=f"pattern_match_any_work_{p_idx}_{nid}_{start_day}_{i}")
                            elif day_type == 'rest':
                                # Must be rest
                                model.addConstr(pattern_match <= x[nid][day]['None'], 
                                               name=f"pattern_match_any_rest_{p_idx}_{nid}_{start_day}_{i}")
                            elif day_type == 'any':
                                # Any shift is fine
                                pass  # No additional constraint needed since pattern_match is binary
    
                    # If all positions matched, then pattern_match = 1
                    # But since we're forbidding, we add constraint that pattern_match = 0
                    model.addConstr(pattern_match == 0, name=f"forbid_pattern_{p_idx}_{nid}_{start_day}")

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
            # Count weekends where nurse works at least one day
            weekend_worked_vars = []
            for w_idx, weekend in enumerate(weekends):
                # Determine weekend days
                weekend_day_indices = []
                if weekend_def == 'SAT_SUN':
                    weekend_day_indices = [weekend['saturday'], weekend['sunday']]
                elif weekend_def == 'FRI_SAT_SUN':
                    if 'friday' in weekend:
                        weekend_day_indices = [weekend['friday'], weekend['saturday'], weekend['sunday']]
                    else:
                        weekend_day_indices = [weekend['saturday'], weekend['sunday']]
                
                # At least one day worked in weekend
                weekend_worked = model.addVar(vtype=GRB.BINARY, name=f"weekend_worked_{nid}_{w_idx}")
                # Link: weekend_worked = 1 if any day in weekend is worked
                # Use big-M formulation
                M = len(weekend_day_indices)  # Max possible work days in a weekend
                model.addConstr(gp.quicksum(
                    gp.quicksum(x[nid][day_idx][s] for s in shifts) 
                    for day_idx in weekend_day_indices
                ) <= M * weekend_worked, name=f"weekend_worked_link1_{nid}_{w_idx}")
                
                # If weekend_worked = 1, then at least one day was worked
                model.addConstr(gp.quicksum(
                    gp.quicksum(x[nid][day_idx][s] for s in shifts) 
                    for day_idx in weekend_day_indices
                ) >= weekend_worked, name=f"weekend_worked_link2_{nid}_{w_idx}")
                
                weekend_worked_vars.append(weekend_worked)
            
            total_weekends_worked = gp.quicksum(weekend_worked_vars)
            model.addConstr(total_weekends_worked <= max_weekends + total_worked_weekends_over[nid],
                           name=f"total_worked_weekends_link_{nid}")
            
            total_cost += pen * total_worked_weekends_over[nid]

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

    # Add costs for preferences
    if soft_constraints.get('preferences_penalty', True):  # Default to True if not specified
        for nid in nurse_ids:
            for pref in nurses[nid]['preferences']:
                day = pref['day']
                penalty = pref['penalty']
                
                # Handle different preference types
                if pref['type'] == 'day_off':
                    # Want a day off but got a work shift
                    model.addConstr(pref_viol[nid][day] >= gp.quicksum(x[nid][day][s] for s in shifts),
                                   name=f"pref_viol_day_off_{nid}_{day}")
                    
                elif pref['type'] == 'day_on':
                    # Want to work but got a day off
                    model.addConstr(pref_viol[nid][day] >= x[nid][day]['None'],
                                   name=f"pref_viol_day_on_{nid}_{day}")
                                   
                elif pref['type'] == 'shift_on':
                    # Want a specific shift
                    wanted_shift = pref['shift']
                    model.addConstr(pref_viol[nid][day] >= 1 - x[nid][day][wanted_shift],
                                   name=f"pref_viol_shift_on_{nid}_{day}_{wanted_shift}")
                                   
                elif pref['type'] == 'shift_off':
                    # Don't want a specific shift
                    unwanted_shift = pref['shift']
                    model.addConstr(pref_viol[nid][day] >= x[nid][day][unwanted_shift],
                                   name=f"pref_viol_shift_off_{nid}_{day}_{unwanted_shift}")
                
                total_cost += penalty * pref_viol[nid][day]

    # Add costs for identical weekend violations if needed
    if 'identical_weekend' in soft_constraints and soft_constraints['identical_weekend'].get('enabled', False):
        pen = soft_constraints['identical_weekend']['penalty']
        # We handle this through the soft constraint formulation
        for idx, weekend in enumerate(weekends):
            weekend_day_indices = []
            if weekend_def == 'SAT_SUN':
                weekend_day_indices = [weekend['saturday'], weekend['sunday']]
            elif weekend_def == 'FRI_SAT_SUN':
                if 'friday' in weekend:
                    weekend_day_indices = [weekend['friday'], weekend['saturday'], weekend['sunday']]
                else:
                    weekend_day_indices = [weekend['saturday'], weekend['sunday']]
            
            for nid in nurse_ids:
                # Create constraints to measure weekend difference
                base_day = weekend_day_indices[0]
                for day_idx in weekend_day_indices[1:]:
                    # Check if shifts are different
                    # Use XOR logic: diff = |x1 - x2| for binary vars, which we linearize
                    for s in shifts + ['None']:
                        # If shift differs, we might have a violation
                        # Create binary var indicating if shifts differ
                        diff_var = model.addVar(vtype=GRB.BINARY, name=f"weekend_diff_{idx}_{nid}_{day_idx}_{s}")
                        # If x[nid][base_day][s] != x[nid][day_idx][s], then diff_var = 1
                        # This is: |x1 - x2|, linearized as:
                        # diff_var >= x1 - x2
                        # diff_var >= x2 - x1
                        model.addConstr(diff_var >= x[nid][base_day][s] - x[nid][day_idx][s])
                        model.addConstr(diff_var >= x[nid][day_idx][s] - x[nid][base_day][s])
                        
                        # Link to identical weekend violation
                        model.addConstr(identical_weekend_viol[idx][nid] >= diff_var)

        total_cost += pen * gp.quicksum(identical_weekend_viol[idx][nid] 
                                      for idx in identical_weekend_viol 
                                      for nid in identical_weekend_viol[idx])

    # Add costs for forbidden pattern violations
    if 'forbidden_patterns' in soft_constraints:
        for p_idx, pattern_data in enumerate(soft_constraints['forbidden_patterns']):
            pen = pattern_data['penalty']
            total_cost += pen * gp.quicksum(forbidden_pattern_viol[p_idx][nid][start_day]
                                          for nid in nurse_ids 
                                          for start_day in forbidden_pattern_viol[p_idx][nid])

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