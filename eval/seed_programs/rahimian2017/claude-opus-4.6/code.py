import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_days = data['num_days']
    num_nurses = data['num_nurses']
    num_shift_types = data['num_shift_types']
    num_weekends = data['num_weekends']
    
    shift_types = data['shift_types']
    shift_length = {}
    for st in shift_types:
        shift_length[st['id']] = st['length_minutes']
    
    shift_ids = [st['id'] for st in shift_types]
    
    forbidden = set()
    for fr in data['forbidden_shift_rotations']:
        forbidden.add((fr['shift_type'], fr['cannot_be_followed_by']))
    
    nurses = data['nurses']
    
    # Build shift-on and shift-off request lookups
    shift_on_req = {}
    for req in data['shift_on_requests']:
        key = (req['nurse_id'], req['day'], req['shift_type'])
        shift_on_req[key] = req['penalty']
    
    shift_off_req = {}
    for req in data['shift_off_requests']:
        key = (req['nurse_id'], req['day'], req['shift_type'])
        shift_off_req[key] = req['penalty']
    
    coverage_req = {}
    for cr in data['coverage_requirements']:
        coverage_req[(cr['day'], cr['shift_type'])] = {
            'preferred': cr['preferred'],
            'under_weight': cr['under_weight'],
            'over_weight': cr['over_weight']
        }
    
    # Model
    model = gp.Model("nurse_scheduling")
    model.setParam('TimeLimit', max(1, time_limit - int(time.time() - start_time) - 2))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Decision variables: x[n, d, s] = 1 if nurse n works shift s on day d
    x = {}
    for n_data in nurses:
        n = n_data['id']
        day_offs = set(n_data['day_off_requests'])
        for d in range(num_days):
            if d in day_offs:
                continue
            for s in shift_ids:
                x[n, d, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{n}_{d}_{s}")
    
    # Working variable: w[n, d] = 1 if nurse n works any shift on day d
    w = {}
    for n_data in nurses:
        n = n_data['id']
        for d in range(num_days):
            w[n, d] = model.addVar(vtype=GRB.BINARY, name=f"w_{n}_{d}")
    
    # Coverage under/over variables
    under = {}
    over = {}
    for d in range(num_days):
        for s in shift_ids:
            under[d, s] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"under_{d}_{s}")
            over[d, s] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"over_{d}_{s}")
    
    model.update()
    
    # Constraints
    
    # 1. At most one shift per nurse per day
    for n_data in nurses:
        n = n_data['id']
        day_offs = set(n_data['day_off_requests'])
        for d in range(num_days):
            shifts_on_day = [x[n, d, s] for s in shift_ids if (n, d, s) in x]
            if shifts_on_day:
                model.addConstr(gp.quicksum(shifts_on_day) == w[n, d])
            else:
                model.addConstr(w[n, d] == 0)
    
    # 2. Day off requests (hard constraint)
    for n_data in nurses:
        n = n_data['id']
        day_offs = set(n_data['day_off_requests'])
        for d in day_offs:
            model.addConstr(w[n, d] == 0)
    
    # 3. Forbidden shift rotations
    for n_data in nurses:
        n = n_data['id']
        for d in range(num_days - 1):
            for s1, s2 in forbidden:
                if (n, d, s1) in x and (n, d + 1, s2) in x:
                    model.addConstr(x[n, d, s1] + x[n, d + 1, s2] <= 1)
    
    # 4. Max shifts per type
    for n_data in nurses:
        n = n_data['id']
        max_per_type = n_data['max_shifts_per_type']
        for s in shift_ids:
            s_key = str(s)
            if s_key in max_per_type:
                max_s = max_per_type[s_key]
                shifts_of_type = [x[n, d, s] for d in range(num_days) if (n, d, s) in x]
                if shifts_of_type:
                    model.addConstr(gp.quicksum(shifts_of_type) <= max_s)
    
    # 5. Min/max total minutes
    for n_data in nurses:
        n = n_data['id']
        total_minutes = gp.quicksum(
            x[n, d, s] * shift_length[s]
            for d in range(num_days)
            for s in shift_ids
            if (n, d, s) in x
        )
        model.addConstr(total_minutes >= n_data['min_total_minutes'])
        model.addConstr(total_minutes <= n_data['max_total_minutes'])
    
    # 6. Max consecutive shifts
    for n_data in nurses:
        n = n_data['id']
        max_con = n_data['max_consecutive_shifts']
        window = max_con + 1
        # Before horizon and after horizon are days off
        for start in range(num_days - window + 1):
            model.addConstr(
                gp.quicksum(w[n, d] for d in range(start, start + window)) <= max_con
            )
        # Boundary: first days (day before horizon is off)
        # Already handled since windows starting at 0 include the boundary effect
        # Actually the constraint as stated: in any window of length max_con+1, sum <= max_con
        # The day before horizon is off, so for windows that would include it:
        # Window ending before start is implicitly handled
        # We need to also consider windows that straddle the boundaries
        # Since day -1 is off and day num_days is off, we don't need extra constraints
        # because the existing windows cover all internal stretches
    
    # 7. Min consecutive shifts
    # If nurse is off on day d and off on day d+k where k < min_con+1 (k = gap+1),
    # then nurse cannot work all days d+1..d+k-1
    # Formally: for gap of length L < min_con between two off days,
    # the nurse cannot work all L days in between.
    # 
    # Equivalent formulation: if w[n,d]=0 and w[n,d+k]=0 for 1 <= k-1 < min_con,
    # then sum(w[n,d+1]..w[n,d+k-1]) <= k-2
    # But we also need boundary conditions where before/after horizon the nurse is OFF.
    
    for n_data in nurses:
        n = n_data['id']
        min_con = n_data['min_consecutive_shifts']
        if min_con <= 1:
            continue
        
        # Internal: for each pair of days that are 2 to min_con apart
        for length in range(1, min_con):
            # Block of 'length' consecutive days, bounded by off days
            for start in range(-1, num_days):
                end = start + length + 1
                if end > num_days:
                    break
                
                # start is off, end is off, days start+1..end-1 must not all be working
                # unless the block length >= min_con
                
                work_days = list(range(max(0, start + 1), min(num_days, end)))
                if len(work_days) == 0:
                    continue
                if len(work_days) >= min_con:
                    continue
                
                L = len(work_days)
                
                if start == -1 and end == num_days:
                    # Both boundaries: nurse assumed OFF before and after
                    # All work_days is entire horizon with length < min_con
                    # sum(w) <= L - 1 (can't work all days)
                    # Actually: nurse can't have a block of length < min_con
                    # But they could work some subset. The constraint is no isolated block < min_con.
                    # This case: if L < min_con, nurse cannot work all L days contiguously starting from day 0
                    # Actually this would prevent any work at all if horizon < min_con, which seems extreme
                    # Let me re-think the constraint formulation
                    pass
                elif start == -1:
                    # Before horizon is off, end is within horizon
                    # If w[n, end] = 0, then nurse can't work all of days 0..end-1
                    if L > 0:
                        model.addConstr(
                            gp.quicksum(w[n, dd] for dd in work_days) <= L - 1 + w[n, end]
                        )
                elif end == num_days:
                    # After horizon is off, start is within horizon
                    # If w[n, start] = 0, then nurse can't work all of days start+1..num_days-1
                    if L > 0:
                        model.addConstr(
                            gp.quicksum(w[n, dd] for dd in work_days) <= L - 1 + w[n, start]
                        )
                else:
                    # Both start and end within horizon
                    # If w[n,start]=0 and w[n,end]=0 then sum(w[n,start+1..end-1]) <= L-1
                    # Equivalent: sum(work) <= L - 1 + w[n,start] + w[n,end]
                    # But we need: sum(work) - L + 1 <= 0 when both boundaries are 0
                    # sum(work) <= L - 1 + w[n,start] + w[n,end]
                    model.addConstr(
                        gp.quicksum(w[n, dd] for dd in work_days) <= L - 1 + w[n, start] + w[n, end]
                    )
    
    # 8. Min consecutive days off
    # Analogous to min consecutive shifts but for off days
    # At boundaries, nurse assumed OFF for infinite stretch (so boundary helps satisfy min)
    # An isolated block of off days shorter than min_con_off is forbidden
    # If nurse works on day d and works on day d+k, and k-1 < min_con_off,
    # then nurse cannot be off all days d+1..d+k-1
    # Equivalent: (1-w) for those days, sum <= k-2 + (1-w[n,d]) + (1-w[n,end])
    # i.e., sum of off days in between <= L-1 + (1-w[start]) + (1-w[end])
    # where off[d] = 1 - w[d]
    
    for n_data in nurses:
        n = n_data['id']
        min_con_off = n_data['min_consecutive_days_off']
        if min_con_off <= 1:
            continue
        
        for length in range(1, min_con_off):
            for start in range(-1, num_days):
                end = start + length + 1
                if end > num_days:
                    break
                
                off_days = list(range(max(0, start + 1), min(num_days, end)))
                if len(off_days) == 0:
                    continue
                L = len(off_days)
                if L >= min_con_off:
                    continue
                
                # off[d] = 1 - w[n,d]
                # sum of off days in the block
                off_sum = gp.quicksum(1 - w[n, dd] for dd in off_days)
                
                if start == -1 and end == num_days:
                    # Both boundaries assumed off (infinite stretch), so boundary off days
                    # extend the block. This means the block is actually connected to 
                    # infinite off stretches on both sides, so it's fine. No constraint needed.
                    pass
                elif start == -1:
                    # Before horizon is off (infinite), so block extends left.
                    # If w[n, end] = 1 (nurse works on end), the off block is
                    # infinite_left + off_days, which is >= min_con_off. So fine.
                    # Actually no: the block is the days in off_days that are off.
                    # If end is working, then off block = infinite off + off_days.
                    # That's already >= min_con_off. So no constraint needed for left boundary.
                    pass
                elif end == num_days:
                    # After horizon is off (infinite), similar reasoning
                    pass
                else:
                    # Both start and end within horizon
                    # If w[n,start]=1 and w[n,end]=1, then sum_off <= L-1
                    # off_sum <= L - 1 + (1 - w[n,start]) + (1 - w[n,end])
                    model.addConstr(
                        off_sum <= L - 1 + (1 - w[n, start]) + (1 - w[n, end])
                    )
    
    # 9. Max weekends
    # Weekend d is Saturday=5+7*k, Sunday=6+7*k for k=0..num_weekends-1
    wk = {}
    for n_data in nurses:
        n = n_data['id']
        for k in range(num_weekends):
            sat = 5 + 7 * k
            sun = 6 + 7 * k
            wk[n, k] = model.addVar(vtype=GRB.BINARY, name=f"wk_{n}_{k}")
            if sat < num_days:
                model.addConstr(w[n, sat] <= wk[n, k])
            if sun < num_days:
                model.addConstr(w[n, sun] <= wk[n, k])
        
        model.addConstr(
            gp.quicksum(wk[n, k] for k in range(num_weekends)) <= n_data['max_weekends']
        )
    
    # 10. Coverage constraints
    for d in range(num_days):
        for s in shift_ids:
            assigned = gp.quicksum(
                x[n_data['id'], d, s]
                for n_data in nurses
                if (n_data['id'], d, s) in x
            )
            pref = coverage_req.get((d, s), {}).get('preferred', 0)
            model.addConstr(assigned + under[d, s] - over[d, s] == pref)
    
    # Objective
    obj = gp.LinExpr()
    
    # Shift-on requests: penalty if NOT assigned
    for (n, d, s), penalty in shift_on_req.items():
        if (n, d, s) in x:
            obj += penalty * (1 - x[n, d, s])
        else:
            obj += penalty  # can't assign, penalty always incurred
    
    # Shift-off requests: penalty if assigned
    for (n, d, s), penalty in shift_off_req.items():
        if (n, d, s) in x:
            obj += penalty * x[n, d, s]
    
    # Coverage penalties
    for d in range(num_days):
        for s in shift_ids:
            cr = coverage_req.get((d, s), None)
            if cr:
                obj += cr['under_weight'] * under[d, s]
                obj += cr['over_weight'] * over[d, s]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Store best solution for callback
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)
    
    remaining_time = max(1, time_limit - int(time.time() - start_time) - 2)
    model.setParam('TimeLimit', remaining_time)
    
    model.optimize(callback)
    
    # Extract solution
    schedule = {}
    for n_data in nurses:
        n = n_data['id']
        nurse_schedule = {}
        for d in range(num_days):
            assigned_shift = None
            for s in shift_ids:
                if (n, d, s) in x:
                    val = x[n, d, s].X
                    if val > 0.5:
                        assigned_shift = s
                        break
            if assigned_shift is not None:
                nurse_schedule[str(d)] = assigned_shift
            else:
                nurse_schedule[str(d)] = None
        schedule[str(n)] = nurse_schedule
    
    coverage_under = {}
    coverage_over = {}
    for d in range(num_days):
        for s in shift_ids:
            u_val = under[d, s].X
            o_val = over[d, s].X
            if u_val > 0.5:
                coverage_under[f"{d}_{s}"] = round(u_val)
            if o_val > 0.5:
                coverage_over[f"{d}_{s}"] = round(o_val)
    
    obj_val = model.ObjVal
    
    solution = {
        "objective_value": obj_val,
        "solution": {
            "schedule": schedule,
            "coverage_under": coverage_under,
            "coverage_over": coverage_over
        }
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()