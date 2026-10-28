import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_days = data['num_days']
    num_nurses = data['num_nurses']
    num_shift_types = data['num_shift_types']
    num_weekends = data['num_weekends']

    shift_types = data['shift_types']
    length = {s['id']: s['length_minutes'] for s in shift_types}

    forbidden = data.get('forbidden_shift_rotations', [])

    nurses = data['nurses']
    
    # Map max shifts per type
    max_shifts = {n['id']: n.get('max_shifts_per_type', {}) for n in nurses}
    min_total_minutes = {n['id']: n['min_total_minutes'] for n in nurses}
    max_total_minutes = {n['id']: n['max_total_minutes'] for n in nurses}
    min_consecutive_shifts = {n['id']: n['min_consecutive_shifts'] for n in nurses}
    max_consecutive_shifts = {n['id']: n['max_consecutive_shifts'] for n in nurses}
    min_consecutive_days_off = {n['id']: n['min_consecutive_days_off'] for n in nurses}
    max_weekends = {n['id']: n['max_weekends'] for n in nurses}
    day_off_requests = {n['id']: set(n.get('day_off_requests', [])) for n in nurses}

    shift_on_requests = data.get('shift_on_requests', [])
    shift_off_requests = data.get('shift_off_requests', [])
    coverage_requirements = data.get('coverage_requirements', [])

    model = gp.Model("nurse_rostering")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    # Variables
    x = {}
    for n in nurses:
        for d in range(num_days):
            for s in shift_types:
                x[n['id'], d, s['id']] = model.addVar(vtype=GRB.BINARY, name=f'x_{n["id"]}_{d}_{s["id"]}')

    w = {}
    o = {}
    for n in nurses:
        for d in range(num_days):
            w[n['id'], d] = model.addVar(vtype=GRB.BINARY, name=f'w_{n["id"]}_{d}')
            o[n['id'], d] = model.addVar(vtype=GRB.BINARY, name=f'o_{n["id"]}_{d}')

    s_work = {}
    s_off = {}
    for n in nurses:
        for d in range(num_days):
            s_work[n['id'], d] = model.addVar(vtype=GRB.BINARY, name=f's_work_{n["id"]}_{d}')
            s_off[n['id'], d] = model.addVar(vtype=GRB.BINARY, name=f's_off_{n["id"]}_{d}')

    under = {}
    over = {}
    for c in coverage_requirements:
        d = c['day']
        s = c['shift_type']
        under[d, s] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f'under_{d}_{s}')
        over[d, s] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f'over_{d}_{s}')

    weekend_worked = {}
    for n in nurses:
        for w_idx in range(num_weekends):
            weekend_worked[n['id'], w_idx] = model.addVar(vtype=GRB.BINARY, name=f'we_{n["id"]}_{w_idx}')

    model.update()

    # Constraints
    for n in nurses:
        for d in range(num_days):
            model.addConstr(gp.quicksum(x[n['id'], d, s['id']] for s in shift_types) == w[n['id'], d])
            model.addConstr(w[n['id'], d] + o[n['id'], d] == 1)

    for n in nurses:
        for d in day_off_requests[n['id']]:
            model.addConstr(w[n['id'], d] == 0)

    for f in forbidden:
        s1 = f['shift_type']
        s2 = f['cannot_be_followed_by']
        for n in nurses:
            for d in range(num_days - 1):
                model.addConstr(x[n['id'], d, s1] + x[n['id'], d+1, s2] <= 1)

    for n in nurses:
        for s in shift_types:
            max_s = max_shifts[n['id']].get(str(s['id']), 0)
            if max_s is not None:
                model.addConstr(gp.quicksum(x[n['id'], d, s['id']] for d in range(num_days)) <= max_s)

    for n in nurses:
        total_mins = gp.quicksum(length[s['id']] * x[n['id'], d, s['id']] for d in range(num_days) for s in shift_types)
        model.addConstr(total_mins >= min_total_minutes[n['id']])
        model.addConstr(total_mins <= max_total_minutes[n['id']])

    for n in nurses:
        mC = min_consecutive_shifts[n['id']]
        MC = max_consecutive_shifts[n['id']]
        mO = min_consecutive_days_off[n['id']]
        
        for d in range(num_days - MC):
            model.addConstr(gp.quicksum(w[n['id'], d+k] for k in range(MC+1)) <= MC)
            
        for d in range(num_days):
            if d == 0:
                model.addConstr(s_work[n['id'], d] >= w[n['id'], d] - 1)
            else:
                model.addConstr(s_work[n['id'], d] >= w[n['id'], d] - w[n['id'], d-1])
            if d + mC <= num_days:
                model.addConstr(gp.quicksum(w[n['id'], d+k] for k in range(mC)) >= mC * s_work[n['id'], d])
                
        for d in range(num_days):
            if d == 0:
                model.addConstr(s_off[n['id'], d] >= o[n['id'], d] - 1)
            else:
                model.addConstr(s_off[n['id'], d] >= o[n['id'], d] - o[n['id'], d-1])
            if d + mO <= num_days:
                model.addConstr(gp.quicksum(o[n['id'], d+k] for k in range(mO)) >= mO * s_off[n['id'], d])

    for n in nurses:
        for w_idx in range(num_weekends):
            sat = 7 * w_idx + 5
            sun = 7 * w_idx + 6
            if sat < num_days:
                model.addConstr(weekend_worked[n['id'], w_idx] >= w[n['id'], sat])
            if sun < num_days:
                model.addConstr(weekend_worked[n['id'], w_idx] >= w[n['id'], sun])
        model.addConstr(gp.quicksum(weekend_worked[n['id'], w_idx] for w_idx in range(num_weekends)) <= max_weekends[n['id']])

    for c in coverage_requirements:
        d = c['day']
        s = c['shift_type']
        assigned = gp.quicksum(x[n['id'], d, s] for n in nurses)
        model.addConstr(assigned + under[d, s] - over[d, s] == c['preferred'])

    obj = gp.LinExpr()
    for c in coverage_requirements:
        d = c['day']
        s = c['shift_type']
        obj += c['under_weight'] * under[d, s] + c['over_weight'] * over[d, s]
    
    for r in shift_on_requests:
        obj += r['penalty'] * (1 - x[r['nurse_id'], r['day'], r['shift_type']])
        
    for r in shift_off_requests:
        obj += r['penalty'] * x[r['nurse_id'], r['day'], r['shift_type']]

    model.setObjective(obj, GRB.MINIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                schedule = {}
                for n in nurses:
                    nid = n['id']
                    schedule[str(nid)] = {}
                    for d in range(num_days):
                        assigned_shift = None
                        for s in shift_types:
                            if round(model.cbGetSolution(x[nid, d, s['id']])) == 1:
                                assigned_shift = s['id']
                                break
                        schedule[str(nid)][str(d)] = assigned_shift
                
                coverage_under = {}
                coverage_over = {}
                for c in coverage_requirements:
                    d = c['day']
                    s = c['shift_type']
                    u_val = round(model.cbGetSolution(under[d, s]))
                    o_val = round(model.cbGetSolution(over[d, s]))
                    if u_val > 0:
                        coverage_under[f'{d}_{s}'] = u_val
                    if o_val > 0:
                        coverage_over[f'{d}_{s}'] = o_val
                
                sol_dict = {
                    "schedule": schedule,
                    "coverage_under": coverage_under,
                    "coverage_over": coverage_over
                }
                logger.log_solution(obj_val, sol_dict)

    model.optimize(callback)

    if model.SolCount == 0:
        output = {
            "objective_value": 0,
            "solution": {
                "schedule": {},
                "coverage_under": {},
                "coverage_over": {}
            }
        }
    else:
        obj_val = model.ObjVal
        schedule = {}
        for n in nurses:
            nid = n['id']
            schedule[str(nid)] = {}
            for d in range(num_days):
                assigned_shift = None
                for s in shift_types:
                    if round(x[nid, d, s['id']].X) == 1:
                        assigned_shift = s['id']
                        break
                schedule[str(nid)][str(d)] = assigned_shift
        
        coverage_under = {}
        coverage_over = {}
        for c in coverage_requirements:
            d = c['day']
            s = c['shift_type']
            u_val = round(under[d, s].X)
            o_val = round(over[d, s].X)
            if u_val > 0:
                coverage_under[f'{d}_{s}'] = u_val
            if o_val > 0:
                coverage_over[f'{d}_{s}'] = o_val
        
        output = {
            "objective_value": float(obj_val),
            "solution": {
                "schedule": schedule,
                "coverage_under": coverage_under,
                "coverage_over": coverage_over
            }
        }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == '__main__':
    main()