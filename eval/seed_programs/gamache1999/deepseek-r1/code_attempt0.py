import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import os
import sys
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path):
    # Read instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    planning_horizon_days = instance['planning_horizon_days']
    crew_members = instance['crew_members']
    pairings = instance['pairings']
    work_rules = instance['work_rules']
    
    # Initialize solution logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Precompute blocked days for each crew member
    blocked_days = {}
    for cm in crew_members:
        blocked = set()
        for act in cm['pre_assigned_activities']:
            start = act['start_day']
            end = act['end_day']
            for d in range(start, end + 1):
                blocked.add(d)
        blocked_days[cm['id']] = blocked
    
    # Extract IDs and set up mappings
    crew_ids = [cm['id'] for cm in crew_members]
    pairing_ids = [p['id'] for p in pairings]
    
    # Create model
    m = gp.Model('CrewRostering')
    m.setParam('OutputFlag', 0)
    m.setParam('Threads', 1)
    m.setParam('TimeLimit', time_limit)
    
    # Create variables: x[c_id, p_id] and s[p_id]
    x = m.addVars(crew_ids, pairing_ids, vtype=GRB.BINARY, name="x")
    s = m.addVars(pairing_ids, vtype=GRB.INTEGER, lb=0, name="s")
    
    # Set forbidden assignments (qualifications and blocked days)
    for cm in crew_members:
        c_id = cm['id']
        quals = set(cm['qualifications'])
        blocked = blocked_days[c_id]
        for p in pairings:
            p_id = p['id']
            # Check qualifications
            if p['aircraft_type'] not in quals:
                x[c_id, p_id].ub = 0
            else:
                # Check blocked days
                start_day = p['start_day']
                duration_days = p['duration_days']
                overlap = False
                for d in range(start_day, start_day + duration_days):
                    if d in blocked:
                        overlap = True
                        break
                if overlap:
                    x[c_id, p_id].ub = 0
    
    # Coverage constraints
    for p in pairings:
        p_id = p['id']
        coverage_req = p['coverage_requirement']
        lhs = gp.quicksum(x[c_id, p_id] for c_id in crew_ids) + s[p_id]
        m.addConstr(lhs == coverage_req, name=f"coverage_{p_id}")
    
    # Day conflict constraints (one pairing per crew per day)
    pairings_by_day = [[] for _ in range(planning_horizon_days)]
    for p in pairings:
        p_id = p['id']
        start = p['start_day']
        end = start + p['duration_days']
        for d in range(start, end):
            if d < planning_horizon_days:
                pairings_by_day[d].append(p_id)
    
    for d in range(planning_horizon_days):
        for c_id in crew_ids:
            p_ids = pairings_by_day[d]
            if p_ids:
                vars = [x[c_id, p_id] for p_id in p_ids]
                m.addConstr(gp.quicksum(vars) <= 1, name=f"conflict_c{c_id}_d{d}")
    
    # Cross-pairing constraints
    cross_pairings = [p['id'] for p in pairings if p['is_cross_pairing']]
    for c_id in crew_ids:
        vars = [x[c_id, p_id] for p_id in cross_pairings]
        if vars:
            m.addConstr(gp.quicksum(vars) <= 1, name=f"cross_{c_id}")
    
    # Monthly flight hour constraints
    monthly_flight_limit = work_rules['monthly_flight_limit_hours']
    for cm in crew_members:
        c_id = cm['id']
        expr = gp.quicksum(x[c_id, p['id']] * p['flight_hours'] for p in pairings)
        m.addConstr(expr <= monthly_flight_limit, name=f"monthly_{c_id}")
    
    # Rolling flight hour constraints
    rolling_flight_limit = work_rules['rolling_flight_limit_hours']
    for cm in crew_members:
        c_id = cm['id']
        prev_hours = cm['flight_hours_previous_15_days']
        expr = prev_hours
        for p in pairings:
            if p['start_day'] <= 14:
                expr += x[c_id, p['id']] * p['flight_hours']
        m.addConstr(expr <= rolling_flight_limit, name=f"rolling_{c_id}")
    
    # Consecutive early departure constraints
    early_pairings_by_day = [[] for _ in range(planning_horizon_days)]
    for p in pairings:
        if p['departure_hour'] < 8.0:
            d0 = p['start_day']
            if d0 < planning_horizon_days:
                early_pairings_by_day[d0].append(p['id'])
    
    for c_id in crew_ids:
        for d in range(planning_horizon_days - 2):
            vars_d = [x[c_id, p_id] for p_id in early_pairings_by_day[d]]
            vars_d1 = [x[c_id, p_id] for p_id in early_pairings_by_day[d+1]]
            vars_d2 = [x[c_id, p_id] for p_id in early_pairings_by_day[d+2]]
            all_vars = vars_d + vars_d1 + vars_d2
            if all_vars:
                m.addConstr(gp.quicksum(all_vars) <= 2, name=f"early_c{c_id}_d{d}")
    
    # Set objective
    total_uncovered_duration = gp.quicksum(s[p['id']] * p['duration_minutes'] for p in pairings)
    m.setObjective(total_uncovered_duration, GRB.MINIMIZE)
    
    # Optimize with callback for logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    
    try:
        if logger:
            m.optimize(callback)
        else:
            m.optimize()
    except gp.GurobiError as e:
        print(f"Optimization error: {e}", file=sys.stderr)
    
    # Prepare solution
    solution = {}
    if m.SolCount > 0:
        solution['objective_value'] = m.objVal
        rosters = {}
        for cm in crew_members:
            c_id = cm['id']
            assigned_pairings = []
            for p in pairings:
                if x[c_id, p['id']].X > 0.5:
                    assigned_pairings.append(p['id'])
            
            total_minutes = planning_horizon_days * 24 * 60
            time_on_pairings = sum(p['duration_minutes'] for p in pairings if x[c_id, p['id']].X > 0.5)
            num_blocked_days = len(blocked_days[c_id])
            time_on_pre_assigned = num_blocked_days * 24 * 60
            unproductive_time = total_minutes - time_on_pairings - time_on_pre_assigned
            
            rosters[str(c_id)] = {
                "roster_index": 0,
                "pairings": assigned_pairings,
                "unproductive_time": int(unproductive_time)
            }
        
        uncovered_pairings = []
        for p in pairings:
            p_id = p['id']
            count = s[p_id].X
            count_int = int(round(count))
            if count_int > 0:
                uncovered_pairings.append({
                    "pairing_id": p_id,
                    "uncovered_count": count_int,
                    "duration_minutes": p['duration_minutes']
                })
        
        solution['rosters'] = rosters
        solution['uncovered_pairings'] = uncovered_pairings
    else:
        # Fallback: no assignments (all pairings uncovered)
        total_uncovered_duration = sum(p['duration_minutes'] * p['coverage_requirement'] for p in pairings)
        solution['objective_value'] = total_uncovered_duration
        rosters = {}
        for cm in crew_members:
            c_id = cm['id']
            num_blocked_days = len(blocked_days[c_id])
            unproductive_time = planning_horizon_days * 24 * 60 - num_blocked_days * 24 * 60
            rosters[str(c_id)] = {
                "roster_index": 0,
                "pairings": [],
                "unproductive_time": int(unproductive_time)
            }
        solution['rosters'] = rosters
        solution['uncovered_pairings'] = [{
            "pairing_id": p['id'],
            "uncovered_count": p['coverage_requirement'],
            "duration_minutes": p['duration_minutes']
        } for p in pairings]
    
    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve Crew Rostering Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to input JSON instance')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output JSON solution')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file for intermediate solutions')
    args = parser.parse_args()
    
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)