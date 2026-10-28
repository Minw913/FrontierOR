import json
import argparse
import sys
from gurobipy import *

def main():
    parser = argparse.ArgumentParser(description='Nurse Rostering Problem Solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Initialize logger
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Parse instance data
    num_days = instance['num_days']
    num_weeks = instance['num_weeks']
    num_nurses = instance['num_nurses']
    num_shift_types = instance['num_shift_types']
    num_weekends = instance['num_weekends']

    shift_types = instance['shift_types']
    shift_ids = [st['id'] for st in shift_types]
    shift_lengths = {st['id']: st['length_minutes'] for st in shift_types}

    forbidden_rotations = instance.get('forbidden_shift_rotations', [])
    nurses_data = instance['nurses']
    shift_on_requests = instance.get('shift_on_requests', [])
    shift_off_requests = instance.get('shift_off_requests', [])
    coverage_requirements = instance.get('coverage_requirements', [])

    # Build forbidden set
    forbidden_set = set()
    for fr in forbidden_rotations:
        forbidden_set.add((fr['shift_type'], fr['cannot_be_followed_by']))

    # Nurse data
    nurse_ids = [n['id'] for n in nurses_data]
    nurse_data = {n['id']: n for n in nurses_data}

    days = list(range(num_days))

    # Weekends: Saturday = 7*w + 5, Sunday = 7*w + 6
    weekends = []
    for w in range(num_weekends):
        sat = w * 7 + 5
        sun = w * 7 + 6
        if sat < num_days and sun < num_days:
            weekends.append((sat, sun))

    # Create model
    model = Model("nurse_rostering")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables: x[n, d, s] = 1 if nurse n assigned shift s on day d
    x = {}
    for n_id in nurse_ids:
        for d in days:
            for s_id in shift_ids:
                x[n_id, d, s_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{n_id}_{d}_{s_id}")

    # Coverage variables
    under = {}
    over = {}
    for cr in coverage_requirements:
        d = cr['day']
        s = cr['shift_type']
        if (d, s) not in under:
            under[d, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"under_{d}_{s}")
            over[d, s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"over_{d}_{s}")

    model.update()

    # Constraint 1: At most one shift per nurse per day
    for n_id in nurse_ids:
        for d in days:
            model.addConstr(quicksum(x[n_id, d, s_id] for s_id in shift_ids) <= 1)

    # Constraint 2: Forbidden rotations
    for n_id in nurse_ids:
        for d in range(num_days - 1):
            for (s1, s2) in forbidden_set:
                model.addConstr(x[n_id, d, s1] + x[n_id, d + 1, s2] <= 1)

    # Constraint 3: Max shifts per type
    for n_id in nurse_ids:
        nd = nurse_data[n_id]
        max_shifts = nd.get('max_shifts_per_type', {})
        for s_id in shift_ids:
            max_s = max_shifts.get(str(s_id), num_days)
            if max_s < num_days:
                model.addConstr(quicksum(x[n_id, d, s_id] for d in days) <= max_s)

    # Constraint 4: Min/max total minutes
    for n_id in nurse_ids:
        nd = nurse_data[n_id]
        total_time = quicksum(shift_lengths[s_id] * x[n_id, d, s_id]
                              for d in days for s_id in shift_ids)
        model.addConstr(total_time >= nd['min_total_minutes'])
        model.addConstr(total_time <= nd['max_total_minutes'])

    # Constraint 5: Max consecutive shifts
    for n_id in nurse_ids:
        nd = nurse_data[n_id]
        max_consec = nd['max_consecutive_shifts']
        window = max_consec + 1
        if window <= num_days:
            for d in range(num_days - window + 1):
                model.addConstr(
                    quicksum(quicksum(x[n_id, d + k, s_id] for s_id in shift_ids)
                             for k in range(window)) <= max_consec)

    # Constraint 6: Min consecutive shifts (no isolated short working blocks)
    for n_id in nurse_ids:
        nd = nurse_data[n_id]
        min_consec = nd['min_consecutive_shifts']
        if min_consec > 1:
            for i in range(num_days):
                for gap in range(1, min_consec):
                    j = i + gap + 1
                    if j >= num_days:
                        break
                    lhs = quicksum(quicksum(x[n_id, k, s_id] for s_id in shift_ids)
                                   for k in range(i + 1, j))
                    wi = quicksum(x[n_id, i, s_id] for s_id in shift_ids)
                    wj = quicksum(x[n_id, j, s_id] for s_id in shift_ids)
                    model.addConstr(lhs <= gap - 1 + wi + wj)

    # Constraint 7: Min consecutive days off (no isolated short off blocks)
    for n_id in nurse_ids:
        nd = nurse_data[n_id]
        min_off = nd['min_consecutive_days_off']
        if min_off > 1:
            for i in range(num_days):
                for gap in range(1, min_off):
                    j = i + gap + 1
                    if j >= num_days:
                        break
                    lhs = quicksum(quicksum(x[n_id, k, s_id] for s_id in shift_ids)
                                   for k in range(i + 1, j))
                    wi = quicksum(x[n_id, i, s_id] for s_id in shift_ids)
                    wj = quicksum(x[n_id, j, s_id] for s_id in shift_ids)
                    model.addConstr(lhs >= wi + wj - 1)

    # Constraint 8: Max weekends
    for n_id in nurse_ids:
        nd = nurse_data[n_id]
        max_we = nd['max_weekends']
        if max_we < len(weekends) and len(weekends) > 0:
            we_vars = []
            for idx, (sat, sun) in enumerate(weekends):
                we_var = model.addVar(vtype=GRB.BINARY, name=f"we_{n_id}_{idx}")
                wsat = quicksum(x[n_id, sat, s_id] for s_id in shift_ids)
                wsun = quicksum(x[n_id, sun, s_id] for s_id in shift_ids)
                model.addConstr(we_var >= wsat)
                model.addConstr(we_var >= wsun)
                we_vars.append(we_var)
            model.addConstr(quicksum(we_vars) <= max_we)

    # Constraint 9: Day-off requests (hard)
    for n_id in nurse_ids:
        nd = nurse_data[n_id]
        for d in nd.get('day_off_requests', []):
            if d < num_days:
                for s_id in shift_ids:
                    model.addConstr(x[n_id, d, s_id] == 0)

    # Coverage constraints
    for cr in coverage_requirements:
        d = cr['day']
        s = cr['shift_type']
        preferred = cr['preferred']
        if d < num_days and s in shift_ids and (d, s) in under:
            actual = quicksum(x[n_id, d, s] for n_id in nurse_ids)
            model.addConstr(actual + under[d, s] - over[d, s] == preferred)

    # Objective
    obj = LinExpr()

    # Shift-on request penalties: penalty if NOT assigned
    for req in shift_on_requests:
        n_id = req['nurse_id']
        d = req['day']
        s_id = req['shift_type']
        penalty = req['penalty']
        if n_id in nurse_data and d < num_days and s_id in shift_ids:
            obj += penalty * (1 - x[n_id, d, s_id])

    # Shift-off request penalties: penalty if assigned
    for req in shift_off_requests:
        n_id = req['nurse_id']
        d = req['day']
        s_id = req['shift_type']
        penalty = req['penalty']
        if n_id in nurse_data and d < num_days and s_id in shift_ids:
            obj += penalty * x[n_id, d, s_id]

    # Coverage penalties
    for cr in coverage_requirements:
        d = cr['day']
        s = cr['shift_type']
        if (d, s) in under:
            obj += cr['under_weight'] * under[d, s]
            obj += cr['over_weight'] * over[d, s]

    model.setObjective(obj, GRB.MINIMIZE)
    model.update()

    # Solution extraction helper
    def build_solution_dict(obj_val, use_cb=False, mdl=None):
        schedule = {}
        for n_id in nurse_ids:
            schedule[str(n_id)] = {}
            for d in days:
                assigned = None
                for s_id in shift_ids:
                    if use_cb:
                        val = mdl.cbGetSolution(x[n_id, d, s_id])
                    else:
                        val = x[n_id, d, s_id].X
                    if val > 0.5:
                        assigned = s_id
                        break
                schedule[str(n_id)][str(d)] = assigned

        coverage_under_out = {}
        coverage_over_out = {}
        for cr in coverage_requirements:
            d = cr['day']
            s = cr['shift_type']
            if (d, s) in under:
                if use_cb:
                    u_val = mdl.cbGetSolution(under[d, s])
                    o_val = mdl.cbGetSolution(over[d, s])
                else:
                    u_val = under[d, s].X
                    o_val = over[d, s].X
                if u_val > 0.001:
                    coverage_under_out[f"{d}_{s}"] = int(round(u_val))
                if o_val > 0.001:
                    coverage_over_out[f"{d}_{s}"] = int(round(o_val))

        return {
            "objective_value": obj_val,
            "solution": {
                "schedule": schedule,
                "coverage_under": coverage_under_out,
                "coverage_over": coverage_over_out
            }
        }

    # Callback for logging incumbents
    best_obj = [float('inf')]

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    try:
                        sol = build_solution_dict(obj_val, use_cb=True, mdl=mdl)
                        logger.log_solution(obj_val, sol)
                    except Exception:
                        logger.log(obj_val)

    # Solve with callback
    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        solution = build_solution_dict(objective_value, use_cb=False)
    else:
        # No solution found - try to produce a trivial solution
        objective_value = float('inf')
        schedule = {}
        for n_id in nurse_ids:
            schedule[str(n_id)] = {}
            for d in days:
                schedule[str(n_id)][str(d)] = None
        solution = {
            "objective_value": objective_value,
            "solution": {
                "schedule": schedule,
                "coverage_under": {},
                "coverage_over": {}
            }
        }

    # Log final solution
    if logger:
        logger.log_solution(objective_value, solution)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()