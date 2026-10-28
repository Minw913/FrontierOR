import json
import argparse
import time
import sys
from typing import Dict, List, Any
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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    time_limit = args.time_limit

    # Extract data
    num_nurses = instance['num_nurses']
    num_days = instance['num_days']
    num_shifts = instance['num_shifts']
    num_skills = instance['num_skills']
    shifts = instance['shifts']
    skills = instance['skills']
    forbidden = instance['forbidden_shift_successions']
    penalty = instance['penalty_weights']
    nurses = instance['nurses']
    demand = instance['demand']

    # Maps
    shift_idx = {s['name']: i for i, s in enumerate(shifts)}
    skill_idx = {s: i for i, s in enumerate(skills)}
    nurse_list = list(range(num_nurses))

    # Precompute forbidden
    forbidden_set = set()
    for pair in forbidden:
        if len(pair) == 2:
            s1 = shift_idx.get(pair[0])
            s2 = shift_idx.get(pair[1])
            if s1 is not None and s2 is not None:
                forbidden_set.add((s1, s2))

    # Precompute weekend days
    num_weeks = instance['num_weeks']
    weekends = []
    for w in range(num_weeks):
        sat = w * 7 + 5
        sun = w * 7 + 6
        weekends.append((sat, sun))

    # Create model
    model = gp.Model("NurseScheduling")
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.LogToConsole, 0)

    # Variables
    # x[n,d,s,k] = 1 if nurse n works day d on shift s with skill k
    x = {}
    for n in nurse_list:
        for d in range(num_days):
            for s in range(num_shifts):
                for k_idx, skill_name in enumerate(skills):
                    if skill_name in nurses[n]['skills']:
                        x[n, d, s, k_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_{n}_{d}_{s}_{k_idx}")

    # y[n,d] = 1 if nurse n works on day d (any shift)
    y = {}
    for n in nurse_list:
        for d in range(num_days):
            y[n, d] = model.addVar(vtype=GRB.BINARY, name=f"y_{n}_{d}")

    # z[n,d,s] = 1 if nurse n works shift s on day d
    z = {}
    for n in nurse_list:
        for d in range(num_days):
            for s in range(num_shifts):
                z[n, d, s] = model.addVar(vtype=GRB.BINARY, name=f"z_{n}_{d}_{s}")

    # Total worked days deviation
    total_work = {}
    total_dev = {}
    for n_idx in nurse_list:
        total_work[n_idx] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=num_days, name=f"total_{n_idx}")
        total_dev[n_idx] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"total_dev_{n_idx}")

    # Weekend worked
    weekend_work = {}
    excess_weekend = {}
    for n_idx in nurse_list:
        for w_idx in range(num_weeks):
            weekend_work[n_idx, w_idx] = model.addVar(vtype=GRB.BINARY, name=f"we_{n_idx}_{w_idx}")
        excess_weekend[n_idx] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"excess_we_{n_idx}")

    # Incomplete weekend
    incomplete = {}
    for n_idx in nurse_list:
        for w_idx in range(num_weeks):
            incomplete[n_idx, w_idx] = model.addVar(vtype=GRB.BINARY, name=f"inc_{n_idx}_{w_idx}")

    # Work start indicators
    work_start = {}
    for n in nurse_list:
        for d in range(num_days):
            work_start[n, d] = model.addVar(vtype=GRB.BINARY, name=f"ws_{n}_{d}")

    # Off start indicators
    off_start = {}
    for n in nurse_list:
        for d in range(num_days):
            off_start[n, d] = model.addVar(vtype=GRB.BINARY, name=f"os_{n}_{d}")

    # Shift start indicators
    shift_start = {}
    for n in nurse_list:
        for d in range(num_days):
            for s in range(num_shifts):
                shift_start[n, d, s] = model.addVar(vtype=GRB.BINARY, name=f"ss_{n}_{d}_{s}")

    # Staffing shortfall
    shortfall = {}
    for d_idx in range(num_days):
        day_demand = demand[d_idx]
        for s_name, s_data in day_demand.items():
            if s_name not in shift_idx:
                continue
            s = shift_idx[s_name]
            for sk_name, req in s_data.items():
                if sk_name not in skill_idx:
                    continue
                k = skill_idx[sk_name]
                shortfall[d_idx, s, k] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"short_{d_idx}_{s}_{k}")

    # Objective
    obj_terms = []

    # 1. Staffing shortfall
    c_S1 = penalty.get('c_S1', 1)
    for d_idx in range(num_days):
        day_demand = demand[d_idx]
        for s_name, s_data in day_demand.items():
            if s_name not in shift_idx:
                continue
            s = shift_idx[s_name]
            for sk_name, req in s_data.items():
                if sk_name not in skill_idx:
                    continue
                k = skill_idx[sk_name]
                opt = req.get('optimal', 0)
                min_d = req.get('minimum', 0)
                coverage = gp.quicksum(x[n, d_idx, s, k] for n in nurse_list if (n, d_idx, s, k) in x)
                short_var = shortfall.get((d_idx, s, k))
                if short_var is not None:
                    model.addConstr(coverage + short_var >= opt, name=f"opt_cover_{d_idx}_{s}_{k}")
                    model.addConstr(coverage >= min_d, name=f"min_cover_{d_idx}_{s}_{k}")
                    obj_terms.append(c_S1 * short_var)

    # 2. Total assignment deviation
    c_S6 = penalty.get('c_S6', 1)
    for n_idx in nurse_list:
        L_minus = nurses[n_idx]['contract']['L_minus']
        L_plus = nurses[n_idx]['contract']['L_plus']
        model.addConstr(total_work[n_idx] == gp.quicksum(y[n_idx, d] for d in range(num_days)))
        model.addConstr(total_dev[n_idx] >= L_minus - total_work[n_idx])
        model.addConstr(total_dev[n_idx] >= total_work[n_idx] - L_plus)
        obj_terms.append(c_S6 * total_dev[n_idx])

    # 3. Excess weekends
    c_S7 = penalty.get('c_S7', 1)
    for n_idx in nurse_list:
        WE_plus = nurses[n_idx]['contract']['WE_plus']
        we_sum = gp.quicksum(weekend_work[n_idx, w] for w in range(num_weeks))
        model.addConstr(excess_weekend[n_idx] >= we_sum - WE_plus)
        obj_terms.append(c_S7 * excess_weekend[n_idx])

    # 4. Incomplete weekends
    c_S5 = penalty.get('c_S5', 1)
    for n_idx in nurse_list:
        for w_idx, (sat, sun) in enumerate(weekends):
            model.addConstr(incomplete[n_idx, w_idx] >= y[n_idx, sat] + y[n_idx, sun] - 1)
            model.addConstr(incomplete[n_idx, w_idx] >= 1 - y[n_idx, sat] - y[n_idx, sun])
            model.addConstr(weekend_work[n_idx, w_idx] >= y[n_idx, sat])
            model.addConstr(weekend_work[n_idx, w_idx] >= y[n_idx, sun])
            obj_terms.append(c_S5 * incomplete[n_idx, w_idx])

    # 5. Preference violations
    c_S4 = penalty.get('c_S4', 1)
    for n_idx, nurse in enumerate(nurses):
        for pref in nurse.get('preferences', []):
            d = pref['day']
            s_name = pref['shift']
            if s_name in shift_idx and 0 <= d < num_days:
                s = shift_idx[s_name]
                obj_terms.append(c_S4 * z[n_idx, d, s])

    # 6. Consecutive worked-days violations (linear approximation)
    c_S2a = penalty.get('c_S2a', 1)
    for n_idx in nurse_list:
        CD_minus = nurses[n_idx]['contract']['CD_minus']
        CD_plus = nurses[n_idx]['contract']['CD_plus']
        # Link work starts
        model.addConstr(work_start[n_idx, 0] >= y[n_idx, 0])
        model.addConstr(work_start[n_idx, 0] <= y[n_idx, 0])
        for d in range(1, num_days):
            model.addConstr(work_start[n_idx, d] >= y[n_idx, d] - y[n_idx, d - 1])
            model.addConstr(work_start[n_idx, d] <= y[n_idx, d])
            model.addConstr(work_start[n_idx, d] <= 1 - y[n_idx, d - 1])
        # Very approximate penalty: penalize every work start (encourages longer stretches)
        # and add a small term for potential bound violations. This is a relaxation.
        obj_terms.append(0.5 * c_S2a * gp.quicksum(work_start[n_idx, d] for d in range(num_days)))

    # 7. Consecutive days-off violations (linear approximation)
    c_S3 = penalty.get('c_S3', 1)
    for n_idx in nurse_list:
        # Link off starts
        model.addConstr(off_start[n_idx, 0] >= 1 - y[n_idx, 0])
        model.addConstr(off_start[n_idx, 0] <= 1 - y[n_idx, 0])
        for d in range(1, num_days):
            model.addConstr(off_start[n_idx, d] >= (1 - y[n_idx, d]) - (1 - y[n_idx, d - 1]))
            model.addConstr(off_start[n_idx, d] <= 1 - y[n_idx, d])
            model.addConstr(off_start[n_idx, d] <= y[n_idx, d - 1])
        obj_terms.append(0.5 * c_S3 * gp.quicksum(off_start[n_idx, d] for d in range(num_days)))

    # 8. Consecutive same-shift violations (linear approximation)
    c_S2b = penalty.get('c_S2b', 1)
    for n_idx in nurse_list:
        for s in range(num_shifts):
            for d in range(1, num_days):
                model.addConstr(shift_start[n_idx, d, s] >= z[n_idx, d, s] - z[n_idx, d - 1, s])
                model.addConstr(shift_start[n_idx, d, s] <= z[n_idx, d, s])
                model.addConstr(shift_start[n_idx, d, s] <= 1 - z[n_idx, d - 1, s])
            # Add term for each possible start to penalize too frequent changes
            obj_terms.append(0.3 * c_S2b * gp.quicksum(shift_start[n_idx, d, s] for d in range(num_days)))

    # Linking constraints: y, z, x
    for n in nurse_list:
        for d in range(num_days):
            model.addConstr(y[n, d] == gp.quicksum(z[n, d, s] for s in range(num_shifts)))
            model.addConstr(y[n, d] <= 1)
            for s in range(num_shifts):
                sum_x = gp.quicksum(x[n, d, s, k] for k_idx, k in enumerate(range(num_skills)) 
                                  if (n, d, s, k) in x)
                model.addConstr(z[n, d, s] == sum_x)

    # Forbidden shift successions
    for n in nurse_list:
        for d in range(num_days - 1):
            for s1, s2 in forbidden_set:
                model.addConstr(z[n, d, s1] + z[n, d + 1, s2] <= 1)

    # Objective
    model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)

    # Callback
    best_obj = float('inf')
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val < best_obj - 1e-4:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL]:
        objective_value = model.objVal if model.objVal < GRB.INFINITY else 0.0
        schedule = []

        for n_idx in range(num_nurses):
            nurse = nurses[n_idx]
            nurse_id = nurse['id']
            nurse_name = nurse['name']
            assignments = []
            for d in range(num_days):
                for s in range(num_shifts):
                    for k_idx in range(num_skills):
                        var = x.get((n_idx, d, s, k_idx))
                        if var is not None and var.X > 0.4:
                            assignments.append({
                                "day": int(d),
                                "shift": shifts[s]['name'],
                                "skill": skills[k_idx]
                            })
            if assignments:
                schedule.append({
                    "nurse_id": int(nurse_id),
                    "nurse_name": nurse_name,
                    "assignments": assignments
                })

        solution = {
            "objective_value": float(objective_value),
            "schedule": schedule
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No feasible solution
        solution = {
            "objective_value": 0.0,
            "schedule": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()