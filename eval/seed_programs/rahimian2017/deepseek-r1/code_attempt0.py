import argparse
import json
import time
import gurobipy as gp
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    num_days = data['num_days']
    num_weeks = data['num_weeks']
    num_nurses = data['num_nurses']
    num_shift_types = data['num_shift_types']
    num_weekends = data['num_weekends']
    shift_types = data['shift_types']
    forbidden_shift_rotations = data['forbidden_shift_rotations']
    nurses = data['nurses']
    shift_on_requests = data.get('shift_on_requests', [])
    shift_off_requests = data.get('shift_off_requests', [])
    coverage_requirements = data['coverage_requirements']

    shift_type_ids = [s['id'] for s in shift_types]
    shift_lengths = {s['id']: s['length_minutes'] for s in shift_types}
    
    forbidden_dict = {}
    for rule in forbidden_shift_rotations:
        s1 = rule['shift_type']
        s2 = rule['cannot_be_followed_by']
        if s1 not in forbidden_dict:
            forbidden_dict[s1] = set()
        forbidden_dict[s1].add(s2)
    
    weekends = []
    for w in range(num_weekends):
        sat = 5 + 7 * w
        sun = 6 + 7 * w
        weekends.append((sat, sun))
    
    coverage_dict = {}
    for req in coverage_requirements:
        key = (req['day'], req['shift_type'])
        coverage_dict[key] = req

    nurse_id_to_index = {nurse['id']: idx for idx, nurse in enumerate(nurses)}
    nurse_index_to_id = {idx: nurse['id'] for idx, nurse in enumerate(nurses)}
    
    time_used = time.time() - start_time
    if time_used >= args.time_limit:
        build_trivial_solution_and_save(data, nurses, coverage_requirements, shift_on_requests, shift_off_requests, args.solution_path)
        return

    try:
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)
        time_left = max(1, args.time_limit - time_used)
        model.setParam('TimeLimit', time_left)

        z = {}
        for n_idx in range(num_nurses):
            for d in range(num_days):
                for s in shift_type_ids:
                    z[(n_idx, d, s)] = model.addVar(vtype=gp.GRB.BINARY, name=f"z_{n_idx}_{d}_{s}")
        
        y = {}
        for n_idx in range(num_nurses):
            for w in range(num_weekends):
                y[(n_idx, w)] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{n_idx}_{w}")
        
        u = {}
        o = {}
        for d in range(num_days):
            for s in shift_type_ids:
                u[(d, s)] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"u_{d}_{s}")
                o[(d, s)] = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"o_{d}_{s}")
        
        model.update()
        
        for n_idx in range(num_nurses):
            for d in range(num_days):
                model.addConstr(
                    gp.quicksum(z[(n_idx, d, s)] for s in shift_type_ids) <= 1,
                    name=f"one_shift_{n_idx}_{d}"
                )
        
        for n_idx in range(num_nurses):
            for d in range(num_days - 1):
                for s1, forbidden_set in forbidden_dict.items():
                    for s2 in forbidden_set:
                        model.addConstr(
                            z[(n_idx, d, s1)] + z[(n_idx, d+1, s2)] <= 1,
                            name=f"forbidden_{n_idx}_{d}_{s1}_{s2}"
                        )
        
        for n_idx in range(num_nurses):
            nurse = nurses[n_idx]
            max_shifts_per_type = nurse['max_shifts_per_type']
            for s in shift_type_ids:
                s_str = str(s)
                if s_str in max_shifts_per_type:
                    max_shifts = max_shifts_per_type[s_str]
                    model.addConstr(
                        gp.quicksum(z[(n_idx, d, s)] for d in range(num_days)) <= max_shifts,
                        name=f"max_shifts_{n_idx}_{s}"
                    )
        
        for n_idx in range(num_nurses):
            nurse = nurses[n_idx]
            total_minutes = gp.quicksum(
                shift_lengths[s] * z[(n_idx, d, s)] 
                for d in range(num_days) 
                for s in shift_type_ids
            )
            model.addConstr(total_minutes >= nurse['min_total_minutes'], name=f"min_minutes_{n_idx}")
            model.addConstr(total_minutes <= nurse['max_total_minutes'], name=f"max_minutes_{n_idx}")
        
        for n_idx in range(num_nurses):
            nurse = nurses[n_idx]
            max_consec = nurse['max_consecutive_shifts']
            if max_consec > 0 and num_days >= max_consec + 1:
                for d in range(0, num_days - max_consec):
                    total_in_window = gp.quicksum(
                        gp.quicksum(z[(n_idx, d+i, s)] for s in shift_type_ids)
                        for i in range(max_consec + 1)
                    )
                    model.addConstr(total_in_window <= max_consec, name=f"max_consec_{n_idx}_{d}")
        
        for n_idx in range(num_nurses):
            nurse = nurses[n_idx]
            min_consec = nurse['min_consecutive_shifts']
            if min_consec >= 2 and num_days >= min_consec:
                for d in range(0, num_days - min_consec + 1):
                    sum_middle = gp.quicksum(
                        gp.quicksum(z[(n_idx, d+i, s)] for s in shift_type_ids)
                        for i in range(1, min_consec)
                    )
                    sum_bound1 = gp.quicksum(z[(n_idx, d, s)] for s in shift_type_ids)
                    sum_bound2 = gp.quicksum(z[(n_idx, d+min_consec, s)] for s in shift_type_ids)
                    model.addConstr(
                        sum_middle <= (min_consec - 2) + sum_bound1 + sum_bound2,
                        name=f"min_consec_{n_idx}_{d}"
                    )
        
        for n_idx in range(num_nurses):
            nurse = nurses[n_idx]
            min_off = nurse['min_consecutive_days_off']
            if min_off >= 2 and num_days >= min_off:
                for d in range(0, num_days - min_off + 1):
                    sum_middle = gp.quicksum(
                        gp.quicksum(z[(n_idx, d+i, s)] for s in shift_type_ids)
                        for i in range(1, min_off)
                    )
                    sum_bound1 = gp.quicksum(z[(n_idx, d, s)] for s in shift_type_ids)
                    sum_bound2 = gp.quicksum(z[(n_idx, d+min_off, s)] for s in shift_type_ids)
                    model.addConstr(
                        sum_middle >= sum_bound1 + sum_bound2 - 1,
                        name=f"min_off_{n_idx}_{d}"
                    )
        
        for n_idx in range(num_nurses):
            for w in range(num_weekends):
                sat, sun = weekends[w]
                if sat < num_days:
                    model.addConstr(
                        y[(n_idx, w)] >= gp.quicksum(z[(n_idx, sat, s)] for s in shift_type_ids),
                        name=f"weekend_{n_idx}_{w}_sat"
                    )
                if sun < num_days:
                    model.addConstr(
                        y[(n_idx, w)] >= gp.quicksum(z[(n_idx, sun, s)] for s in shift_type_ids),
                        name=f"weekend_{n_idx}_{w}_sun"
                    )
        
        for n_idx in range(num_nurses):
            nurse = nurses[n_idx]
            model.addConstr(
                gp.quicksum(y[(n_idx, w)] for w in range(num_weekends)) <= nurse['max_weekends'],
                name=f"max_weekends_{n_idx}"
            )
        
        for n_idx in range(num_nurses):
            nurse = nurses[n_idx]
            for day_off in nurse['day_off_requests']:
                if day_off < num_days:
                    model.addConstr(
                        gp.quicksum(z[(n_idx, day_off, s)] for s in shift_type_ids) == 0,
                        name=f"day_off_{n_idx}_{day_off}"
                    )
        
        total_penalty = 0
        
        for req in shift_on_requests:
            n_id = req['nurse_id']
            d = req['day']
            s = req['shift_type']
            penalty_val = req['penalty']
            n_idx = nurse_id_to_index[n_id]
            if d < num_days and s in shift_type_ids:
                total_penalty += penalty_val * (1 - z[(n_idx, d, s)])
        
        for req in shift_off_requests:
            n_id = req['nurse_id']
            d = req['day']
            s = req['shift_type']
            penalty_val = req['penalty']
            n_idx = nurse_id_to_index[n_id]
            if d < num_days and s in shift_type_ids:
                total_penalty += penalty_val * z[(n_idx, d, s)]
        
        for d in range(num_days):
            for s in shift_type_ids:
                if (d, s) in coverage_dict:
                    req = coverage_dict[(d, s)]
                    preferred = req['preferred']
                    actual = gp.quicksum(z[(n_idx, d, s)] for n_idx in range(num_nurses))
                    total_penalty += req['under_weight'] * u[(d, s)] + req['over_weight'] * o[(d, s)]
                    model.addConstr(u[(d, s)] >= preferred - actual, name=f"under_{d}_{s}")
                    model.addConstr(o[(d, s)] >= actual - preferred, name=f"over_{d}_{s}")
        
        model.setObjective(total_penalty, gp.GRB.MINIMIZE)
        
        def callback(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                obj_val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj_val)
        
        model.optimize(callback if logger else None)
        
        if model.SolCount > 0:
            schedule = {}
            for n_idx in range(num_nurses):
                nurse_id = nurse_index_to_id[n_idx]
                nurse_schedule = {}
                for d in range(num_days):
                    assigned_shift = None
                    for s in shift_type_ids:
                        if abs(z[(n_idx, d, s)].X - 1.0) < 1e-5:
                            assigned_shift = s
                            break
                    nurse_schedule[str(d)] = assigned_shift
                schedule[str(nurse_id)] = nurse_schedule
            
            coverage_under = {}
            coverage_over = {}
            for d in range(num_days):
                for s in shift_type_ids:
                    key = f"{d}_{s}"
                    if (d, s) in u:
                        under_val = u[(d, s)].X
                        over_val = o[(d, s)].X
                        coverage_under[key] = under_val
                        coverage_over[key] = over_val
            
            objective_value = model.ObjVal
            solution = {
                "schedule": schedule,
                "coverage_under": coverage_under,
                "coverage_over": coverage_over
            }
            output = {
                "objective_value": objective_value,
                "solution": solution
            }
        else:
            output = build_trivial_solution(data, nurses, coverage_requirements, shift_on_requests, shift_off_requests)
        
    except (gp.GurobiError, Exception) as e:
        print(f"Error during optimization: {str(e)}")
        output = build_trivial_solution(data, nurses, coverage_requirements, shift_on_requests, shift_off_requests)
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f)

def build_trivial_solution(data, nurses, coverage_requirements, shift_on_requests, shift_off_requests):
    num_days = data['num_days']
    schedule = {}
    for nurse in nurses:
        nurse_id = nurse['id']
        nurse_schedule = {}
        for d in range(num_days):
            nurse_schedule[str(d)] = None
        schedule[str(nurse_id)] = nurse_schedule
    
    coverage_under = {}
    coverage_over = {}
    for req in coverage_requirements:
        d = req['day']
        s = req['shift_type']
        key = f"{d}_{s}"
        preferred = req['preferred']
        coverage_under[key] = preferred
        coverage_over[key] = 0
    
    total_penalty = 0
    for req in shift_on_requests:
        total_penalty += req['penalty']
    for req in coverage_requirements:
        total_penalty += req['under_weight'] * req['preferred']
    
    solution = {
        "schedule": schedule,
        "coverage_under": coverage_under,
        "coverage_over": coverage_over
    }
    return {
        "objective_value": total_penalty,
        "solution": solution
    }

if __name__ == "__main__":
    main()