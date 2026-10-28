import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    # Load solution logger if provided
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Load problem instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    planning_horizon = data['planning_horizon_days']
    rules = data['work_rules']
    pairings = data['pairings']
    crew = data['crew_members']

    # Classify work vs rest types to define which ones interrupt rests
    work_types = {'training', 'reserve_block', 'transition'}
    
    # Initialize optimization model
    m = gp.Model()
    m.setParam('TimeLimit', args.time_limit)
    m.setParam('Threads', 1)
    m.setParam('OutputFlag', 0)

    # Decision variables for pairing coverage
    y = {}
    req = {}
    dur = {}
    for p in pairings:
        pid = p['id']
        req[pid] = p['coverage_requirement']
        dur[pid] = p['duration_minutes']
        y[pid] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=req[pid], name=f"y_{pid}")

    x = {}
    V_c = {}
    Z = {}
    W = {}

    global_slacks_W = []
    global_slacks_M = []
    global_slacks_S = []

    # Process each crew member
    for c in crew:
        cid = c['id']
        quals = set(c['qualifications'])
        pre_assign = c.get('pre_assigned_activities', [])
        
        rest_days_c = set()
        work_days_c = set()
        reserve_ends = set()
        pre_assign_intervals = []
        
        for a in pre_assign:
            sd = a['start_day']
            ed = a['end_day']
            atype = a['type']
            pre_assign_intervals.append((sd, ed))
            
            for d in range(sd, ed + 1):
                if atype.lower() in work_types:
                    work_days_c.add(d)
                else:
                    rest_days_c.add(d)
                    
            if atype.lower() == 'reserve_block':
                reserve_ends.add(ed)
                
        # Filter valid pairings
        valid_p = []
        for p in pairings:
            if p['aircraft_type'] not in quals:
                continue
                
            p_start = p['start_day']
            p_end = p_start + p['duration_days'] - 1
            
            # Pre-assigned overlaps
            overlap = any((max(p_start, asd) <= min(p_end, aed)) for (asd, aed) in pre_assign_intervals)
            if overlap: continue
                
            # Reserve blocks must be followed by noon/later pairing or rest (i.e. start >= 12.0)
            if (p_start - 1) in reserve_ends and p['departure_hour'] < 12.0:
                continue
                    
            valid_p.append(p)
            x[cid, p['id']] = m.addVar(vtype=GRB.BINARY, name=f"x_{cid}_{p['id']}")
            
        V_c[cid] = valid_p

        # Duty logic / Schedule continuous times
        for d in range(planning_horizon):
            Z[cid, d] = m.addVar(vtype=GRB.BINARY, name=f"Z_{cid}_{d}")
            if d in work_days_c:
                m.addConstr(Z[cid, d] == 1)
            for p in V_c[cid]:
                if p['start_day'] <= d <= p['start_day'] + p['duration_days'] - 1:
                    m.addConstr(Z[cid, d] >= x[cid, p['id']])

        # Overlap constraints internally among valid pairings
        for i in range(len(valid_p)):
            for j in range(i + 1, len(valid_p)):
                p1, p2 = valid_p[i], valid_p[j]
                t1_start = p1['start_day'] + p1['departure_hour'] / 24.0
                t1_end   = p1['start_day'] + p1['duration_days'] - 1 + p1['return_hour'] / 24.0
                t2_start = p2['start_day'] + p2['departure_hour'] / 24.0
                t2_end   = p2['start_day'] + p2['duration_days'] - 1 + p2['return_hour'] / 24.0
                if max(t1_start, t2_start) < min(t1_end, t2_end):
                    m.addConstr(x[cid, p1['id']] + x[cid, p2['id']] <= 1)

        # Monthly limit
        m.addConstr(gp.quicksum(x[cid, p['id']] * p['flight_hours'] for p in V_c[cid]) <= rules['monthly_flight_limit_hours'])
        
        # Rolling flight limit
        rolling_sum = c.get('flight_hours_previous_15_days', 0.0)
        rolling_sum += gp.quicksum(x[cid, p['id']] * p['flight_hours'] for p in V_c[cid] if p['start_day'] <= 14)
        m.addConstr(rolling_sum <= rules.get('rolling_flight_limit_hours', 85.0))
        
        # Cross pairing bound
        if 'max_cross_pairings_per_month' in rules:
            m.addConstr(gp.quicksum(x[cid, p['id']] for p in V_c[cid] if p['is_cross_pairing']) <= rules['max_cross_pairings_per_month'])

        # Consecutive early departures < 8:00 AM limitation
        early_p_by_day = {d: [] for d in range(planning_horizon)}
        for p in V_c[cid]:
            if p['departure_hour'] < 8.0:
                early_p_by_day[p['start_day']].append(p)
                
        for d in range(planning_horizon - 2):
            if any(early_p_by_day[d]) or any(early_p_by_day[d+1]) or any(early_p_by_day[d+2]):
                expr = gp.quicksum(x[cid, p['id']] for k in (d, d+1, d+2) for p in early_p_by_day[k])
                m.addConstr(expr <= rules.get('max_consecutive_early_departures_before_8am', 2))

        # Relaxation for rest logic formulation
        for d in range(planning_horizon - 1):
            W[cid, d] = m.addVar(vtype=GRB.BINARY, name=f"W_{cid}_{d}")
            m.addConstr(W[cid, d] <= 1 - Z[cid, d])
            m.addConstr(W[cid, d] <= 1 - Z[cid, d+1])
            
        num_weeks = (planning_horizon + 6) // 7
        for w in range(num_weeks):
            start_d, max_start = 7 * w, 7 * w + 5
            possible_starts = [d for d in range(start_d, max_start + 1) if d < planning_horizon - 1]
            if possible_starts:
                s_var = m.addVar(vtype=GRB.CONTINUOUS, lb=0)
                m.addConstr(gp.quicksum(W[cid, d] for d in possible_starts) + s_var >= 1)
                global_slacks_W.append(s_var)

        M_vars = []
        for d in range(planning_horizon - 3):
            M_d = m.addVar(vtype=GRB.BINARY)
            m.addConstr(M_d <= 1 - Z[cid, d])
            m.addConstr(M_d <= 1 - Z[cid, d+1])
            m.addConstr(M_d <= 1 - Z[cid, d+2])
            m.addConstr(M_d <= 1 - Z[cid, d+3])
            M_vars.append(M_d)
            
        if M_vars:
            sM = m.addVar(vtype=GRB.CONTINUOUS, lb=0)
            m.addConstr(gp.quicksum(M_vars) + sM >= 1)
            global_slacks_M.append(sM)
            
        for d in range(max(1, planning_horizon - 7)):
            valid_ks = [k for k in range(d, d + 9) if k < planning_horizon - 1]
            if valid_ks:
                s_spr = m.addVar(vtype=GRB.CONTINUOUS, lb=0)
                m.addConstr(gp.quicksum(W[cid, k] for k in valid_ks) + s_spr >= 1)
                global_slacks_S.append(s_spr)

    # Calculate overall assignment coverage
    for pid in dur:
        m.addConstr(y[pid] == gp.quicksum(x[c['id'], pid] for c in crew if (c['id'], pid) in x))

    obj_expr = gp.quicksum((req[pid] - y[pid]) * dur[pid] for pid in dur)
    penalty_expr = (gp.quicksum(global_slacks_W) * 1e6 +
                    gp.quicksum(global_slacks_M) * 5e4 +
                    gp.quicksum(global_slacks_S) * 1e5)

    m.setObjective(obj_expr + penalty_expr, GRB.MINIMIZE)
    m._y = y
    m._logger = logger
    m._req = req
    m._dur = dur

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            y_vals = model.cbGetSolution(model._y)
            true_obj = sum((model._req[pid] - round(y_vals[pid])) * model._dur[pid] for pid in model._dur)
            if model._logger:
                model._logger.log(true_obj)

    m.optimize(cb)

    if m.SolCount > 0:
        true_obj_val = sum((req[pid] - round(y[pid].X)) * dur[pid] for pid in dur)
    else:
        true_obj_val = sum((req[pid]) * dur[pid] for pid in dur)

    output = {
        "objective_value": true_obj_val,
        "rosters": {},
        "uncovered_pairings": []
    }

    if m.SolCount > 0:
        for pid in dur:
            covered = round(y[pid].X)
            uncovered = req[pid] - covered
            if uncovered > 0:
                output["uncovered_pairings"].append({
                    "pairing_id": pid,
                    "uncovered_count": int(uncovered),
                    "duration_minutes": dur[pid]
                })

        for c in crew:
            cid = c['id']
            assigned_p = []
            if cid in V_c:
                for p in V_c[cid]:
                    if x[cid, p['id']].X > 0.5:
                        assigned_p.append(p['id'])
                        
            # Determine total Unproductive duration explicitly handling calendar spans limit
            T = planning_horizon * 24 * 60
            assigned_dur = sum(dur[pid] for pid in assigned_p)
            
            pre_assigned_dur = 0
            for a in c.get('pre_assigned_activities', []):
                d_days = a.get('duration_days', a['end_day'] - a['start_day'] + 1)
                pre_assigned_dur += d_days * 24 * 60
                
            unprod = max(0, T - assigned_dur - pre_assigned_dur)
            
            output["rosters"][str(cid)] = {
                "roster_index": 0,
                "pairings": assigned_p,
                "unproductive_time": int(unprod)
            }
    else:
        for c in crew:
            output["rosters"][str(c['id'])] = {
                "roster_index": 0,
                "pairings": [],
                "unproductive_time": int(planning_horizon * 24 * 60)
            }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    main()