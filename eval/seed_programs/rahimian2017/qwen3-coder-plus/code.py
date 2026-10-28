import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_nurse_scheduling(instance_data, time_limit):
    # Extract data
    num_days = instance_data["num_days"]
    num_nurses = instance_data["num_nurses"]
    num_shift_types = instance_data["num_shift_types"]
    shift_types = instance_data["shift_types"]
    forbidden_shift_rotations = instance_data["forbidden_shift_rotations"]
    nurses = instance_data["nurses"]
    shift_on_requests = instance_data["shift_on_requests"]
    shift_off_requests = instance_data["shift_off_requests"]
    coverage_requirements = instance_data["coverage_requirements"]

    # Create mapping from shift ID to index
    shift_id_to_idx = {st["id"]: idx for idx, st in enumerate(shift_types)}
    # Also map idx to ID for reverse lookup
    shift_idx_to_id = {idx: st["id"] for idx, st in enumerate(shift_types)}

    # Build forbidden successor map
    forbidden_successor_map = {}
    for shift_id in shift_id_to_idx.keys():
        forbidden_successor_map[shift_id] = set()
        
    for forbidden in forbidden_shift_rotations:
        shift_id_from = forbidden["shift_type"]
        shift_id_to = forbidden["cannot_be_followed_by"]
        if shift_id_from in shift_id_to_idx:
            forbidden_successor_map[shift_id_from].add(shift_id_to)

    # Create model
    model = gp.Model("Nurse Scheduling")
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)  # Suppress output unless we want logging

    # Decision variables: x[nurse][day][shift_type]
    x = {}
    for n in range(num_nurses):
        x[n] = {}
        for d in range(num_days):
            x[n][d] = {}
            for s in range(num_shift_types):
                x[n][d][s] = model.addVar(vtype=GRB.BINARY, name=f"x_{n}_{d}_{s}")

    # Variable for whether a nurse works on a day
    y = {}
    for n in range(num_nurses):
        y[n] = {}
        for d in range(num_days):
            y[n][d] = model.addVar(vtype=GRB.BINARY, name=f"y_{n}_{d}")

    # Variable linking x and y: y[n][d] = sum of x[n][d][s] over all shifts s
    for n in range(num_nurses):
        for d in range(num_days):
            model.addConstr(y[n][d] == gp.quicksum(x[n][d][s] for s in range(num_shift_types)))

    # Variables for shift on and shift off penalties
    shift_on_penalty_vars = {}
    for req in shift_on_requests:
        if req["nurse_id"] < num_nurses and req["day"] < num_days:
            n = req["nurse_id"]
            day = req["day"]
            shift_type = req["shift_type"]
            if shift_type in shift_id_to_idx:
                shift_type_idx = shift_id_to_idx[shift_type]
                var_name = f"shift_on_pen_{n}_{day}_{shift_type}"
                var = model.addVar(vtype=GRB.BINARY, name=var_name)
                shift_on_penalty_vars[(n, day, shift_type)] = var
                # If the nurse is NOT assigned the requested shift, penalty is incurred
                # So var = 1 if x[n][day][shift_type_idx] = 0
                # This means the penalty is applied if the nurse does not get the requested shift
                model.addConstr(var >= 1 - x[n][day][shift_type_idx])

    shift_off_penalty_vars = {}
    for req in shift_off_requests:
        if req["nurse_id"] < num_nurses and req["day"] < num_days:
            n = req["nurse_id"]
            day = req["day"]
            shift_type = req["shift_type"]
            if shift_type in shift_id_to_idx:
                shift_type_idx = shift_id_to_idx[shift_type]
                var_name = f"shift_off_pen_{n}_{day}_{shift_type}"
                var = model.addVar(vtype=GRB.BINARY, name=var_name)
                shift_off_penalty_vars[(n, day, shift_type)] = var
                # If the nurse IS assigned the requested shift, penalty is incurred
                # So var = x[n][day][shift_type_idx]
                model.addConstr(var >= x[n][day][shift_type_idx])
                model.addConstr(var <= x[n][day][shift_type_idx])

    # Variables to track actual staffing levels
    actual_staffing = {}
    coverage_under = {}
    coverage_over = {}
    
    for req in coverage_requirements:
        if req["day"] < num_days and req["shift_type"] in shift_id_to_idx:
            d = req["day"]
            s = req["shift_type"]
            staffing_var = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"actual_staffing_{d}_{s}")
            actual_staffing[(d, s)] = staffing_var
            
            coverage_under[(d, s)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"under_{d}_{s}")
            coverage_over[(d, s)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"over_{d}_{s}")

    # Constraints: Each nurse works at most one shift per day - redundant since we linked x to y
    
    # Constraint: Each nurse works at most one shift per day explicitly
    for nurse in range(num_nurses):
        for day in range(num_days):
            model.addConstr(gp.quicksum(x[nurse][day][s] for s in range(num_shift_types)) <= 1)

    # Constraint: Forbidden shift transitions
    for n in range(num_nurses):
        for d in range(num_days - 1):  # From day 0 to day N-2
            for s1 in range(num_shift_types):
                shift_id_from = shift_idx_to_id[s1]  # Get actual shift_id
                for s2 in range(num_shift_types):
                    shift_id_to = shift_idx_to_id[s2]  # Get actual shift_id
                    if shift_id_to in forbidden_successor_map.get(shift_id_from, set()):
                        model.addConstr(x[n][d][s1] + x[n][d+1][s2] <= 1)

    # Constraint: Max shifts of each type per nurse
    for n in range(num_nurses):
        nurse_info = nurses[n]
        for s_idx in range(num_shift_types):
            shift_id = shift_idx_to_id[s_idx]
            max_shifts = nurse_info["max_shifts_per_type"].get(shift_id, 0)
            model.addConstr(gp.quicksum(x[n][d][s_idx] for d in range(num_days)) <= max_shifts)

    # Constraint: Min and max total working time per nurse
    for n in range(num_nurses):
        nurse_info = nurses[n]
        min_minutes = nurse_info["min_total_minutes"]
        max_minutes = nurse_info["max_total_minutes"]
        work_expr = gp.quicksum(
            x[n][d][s] * shift_types[s]["length_minutes"]
            for d in range(num_days)
            for s in range(num_shift_types)
        )
        if min_minutes > 0:
            model.addConstr(work_expr >= min_minutes)
        if max_minutes > 0 and max_minutes < float('inf'):
            model.addConstr(work_expr <= max_minutes)

    # Constraint: Max consecutive shifts
    # In any window of size max_consecutive_shifts + 1, we have at most max_consecutive_shifts days with work
    for n in range(num_nurses):
        max_cons = nurses[n]["max_consecutive_shifts"]
        if max_cons < num_days:
            for start_day in range(num_days - max_cons):
                window_days = list(range(start_day, start_day + max_cons + 1))
                model.addConstr(gp.quicksum(y[n][d] for d in window_days) <= max_cons)

    # Min consecutive shifts - modeled differently for clarity
    # Ensure sequences of work days are at least min_consecutive_shifts long
    # To model min consecutive shifts:
    # If someone works day D and doesn't work day D-1 but works day D+1, they must work consecutive days
    # We'll use auxiliary variables for start and stop of work sequences
    
    start_work = {}
    end_work = {}
    for n in range(num_nurses):
        for d in range(num_days):
            start_work[n, d] = model.addVar(vtype=GRB.BINARY, name=f"start_{n}_{d}")
            end_work[n, d] = model.addVar(vtype=GRB.BINARY, name=f"end_{n}_{d}")

    # Defining start_work variables: work on day d but not on d-1 (or it's day 0)
    for n in range(num_nurses):
        for d in range(num_days):
            if d == 0:
                # On first day, start if y[n][0]=1
                model.addConstr(start_work[n, d] >= y[n][d])
                model.addConstr(start_work[n, d] <= y[n][d])
            else:
                # start_work[n,d] = y[n][d] AND not y[n][d-1]
                # This is equivalent to: 
                # start_work[n,d] <= y[n][d]
                # start_work[n,d] <= 1 - y[n][d-1]
                # start_work[n,d] >= y[n][d] - y[n][d-1]
                model.addConstr(start_work[n, d] <= y[n][d])
                model.addConstr(start_work[n, d] <= 1 - y[n][d-1])
                model.addConstr(start_work[n, d] >= y[n][d] - y[n][d-1])
    
    # Defining end_work variables: don't work on d but did on d-1
    for n in range(num_nurses):
        for d in range(num_days):
            if d == 0:
                # We're modeling end of shift patterns - so we don't care about boundary on first day
                # Actually, for consistency, let's say we don't start/end unless there was a transition
                pass  # For day 0, no ending if starting at beginning
            else:    
                # end_work[n,d] = 1-y[n][d] AND y[n][d-1]
                # This is equivalent to: 
                # end_work[n,d] <= 1 - y[n][d]
                # end_work[n,d] <= y[n][d-1]
                # end_work[n,d] >= y[n][d-1] - y[n][d]
                model.addConstr(end_work[n, d] <= 1 - y[n][d])
                model.addConstr(end_work[n, d] <= y[n][d-1])
                model.addConst = model.addConstr(end_work[n, d] >= y[n][d-1] - y[n][d])

    # Apply min consecutive shift constraint
    for n in range(num_nurses):
        min_cons = nurses[n]["min_consecutive_shifts"]
        if min_cons > 1:  
            for d_start in range(num_days):
                # If there's a shift started at day d_start, 
                # ensure that for at least min_cons-1 days starting at d_start, y[n][d] remains 1
                consecutive_end_index = min(day + min_cons - 1, num_days - 1)
                if d_start <= num_days - min_cons:  # Only applies if sequence can fit
                    # If start_work[n][d_start] = 1, require y[n][d] = 1 for the next (min_cons-1) days
                    for idx in range(min_cons):  # From start to start + min_cons - 1
                        day_idx = d_start + idx
                        if day_idx < num_days:
                            model.addConstr(y[n][day_idx] >= start_work[n, d_start])

    # Similarly, for min consecutive off days:
    for n in range(num_nurses):
        min_off = nurses[n]["min_consecutive_days_off"]
        if min_off > 1:
            for d_start in range(num_days - min_off + 1):
                # Need to use end_work at start of off sequence and ensure we're truly off for min_off
                off_start = model.addVar(vtype=GRB.BINARY, name=f"off_seq_start_{n}_{d_start}")
                
                # off_start = 1 iff work ended at day d_start and next day is still off 
                if d_start == 0:
                    # Special handling if d_start is 0 - assume we were working before
                    model.addConstr(off_start <= 1 - y[n][0])
                elif d_start > 0:
                    # A start of off seq occurs when work ends day d_start-1
                    model.addConstr(off_start >= end_work[n, d_start])
                    model.addConstr(off_start <= end_work[n, d_start])
                    
                if d_start <= num_days - min_off:
                    for idx in range(min_off):
                        day_idx = d_start + idx
                        if day_idx < num_days:
                            model.addConstr(1 - y[n][day_idx] >= off_start)
    
    # Constraint: Max weekends worked per nurse
    for n in range(num_nurses):
        max_weekends = nurses[n]["max_weekends"]
        weekend_vars = []

        num_weeks = instance_data["num_weeks"]
        for week_num in range(num_weeks):
            saturday = week_num * 7 + 5
            sunday = week_num * 7 + 6

            if saturday >= num_days or sunday >= num_days:
                continue

            # Create a variable to indicate if the weekend was worked by nurse n
            # Worked if nurse works Saturday or Sunday (any shift)
            weekend_worked = model.addVar(vtype=GRB.BINARY, name=f"n{n}_week_{week_num}")

            # Define weekend_worked as (worked_saturday OR worked_sunday)
            shift_sum_sat = gp.quicksum(x[n][saturday][s] for s in range(num_shift_types))
            shift_sum_sun = gp.quicksum(x[n][sunday][s] for s in range(num_shift_types))
            
            # We model OR condition:
            # weekend_worked >= shift_sum_sat  (if shift_sum_sat > 0, weekend_worked = 1)
            # weekend_worked >= shift_sum_sun  (if shift_sum_sun > 0, weekend_worked = 1) 
            # weekend_worked <= shift_sum_sat + shift_sum_sun (weekend_worked = 0 if both are 0)
            
            M = 999  # large number
            model.addConstr(weekend_worked >= shift_sum_sat / M) # This is not right for general case
            model.addConstr(weekend_worked >= shift_sum_sun / M)
            model.addConstr(weekend_worked <= shift_sum_sat + shift_sum_sun)
            
            # Correct linearization:
            # weekend_worked >= x if x>0, otherwise doesn't matter (OR: Z >= A + B -1)
            total_shifts = shift_sum_sat + shift_sum_sun
            # This is: Z >= A, Z >= B, Z <= A+B, or more precisely for Z = A ∨ B:
            # Z >= (A +B)/(total_possible) clamped correctly
            # Instead:
            # weekend_worked >= any shift on sat/sun
            # weekend_worked <= total shifts 
            model.addConstr(weekend_worked >= y[n][saturday])
            model.addConstr(weekend_worked >= y[n][sunday])
            model.addConstr(weekend_worked <= y[n][saturday] + y[n][sunday])
            
            weekend_vars.append(weekend_worked)

        model.addConstr(gp.quicksum(weekend_vars) <= max_weekends)

    # Constraint: Requested days off
    for n in range(num_nurses):
        for day in nurses[n]["day_off_requests"]:
            if day < num_days:
                for s in range(num_shift_types):
                    model.addConstr(x[n][day][s] == 0)

    # Track actual staffing
    for req in coverage_requirements:
        if req["day"] < num_days and req["shift_type"] in shift_id_to_idx:
            d = req["day"]
            s_type = req["shift_type"]  
            s_idx = shift_id_to_idx[s_type] # Convert to integer index
            # Count all nurses assigned to the shift matching s_type on day d
            model.addConstr(
                actual_staffing[(d, s_type)] ==
                gp.quicksum(x[n][d][s_idx] for n in range(num_nurses))
            )

    # Define coverage deviation constraints
    for req in coverage_requirements:
        if req["day"] < num_days and req["shift_type"] in shift_id_to_idx:
            d = req["day"]
            s_type = req["shift_type"]
            preferred = req["preferred"]
            model.addConstr(
                actual_staffing[(d, s_type)] - coverage_over[(d, s_type)] + coverage_under[(d, s_type)] == preferred
            )

    # Calculate objective - sum of all penalties
    obj_expr = gp.LinExpr()
    
    # Add shift-on penalties
    for (n, day, shift_type), var in shift_on_penalty_vars.items():
        req_dict = next(r for r in shift_on_requests if r["nurse_id"]==n and r["day"]==day and r["shift_type"]==shift_type)
        obj_expr += var * req_dict["penalty"]
    
    # Add shift-off penalties
    for (n, day, shift_type), var in shift_off_penalty_vars.items():
        req_dict = next(r for r in shift_off_requests if r["nurse_id"]==n and r["day"]==day and r["shift_type"]==shift_type)
        obj_expr += var * req_dict["penalty"]

    # Add under and over coverage penalties
    for req in coverage_requirements:
        if req["day"] < num_days and req["shift_type"] in shift_id_to_idx:
            d = req["day"]
            s_type = req["shift_type"]
            obj_expr += coverage_under[(d, s_type)] * req["under_weight"]
            obj_expr += coverage_over[(d, s_type)] * req["over_weight"]

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Solve the model
    
    def my_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    if args.log_path:
        model.optimize(my_callback)
    else:
        model.optimize()

    # Prepare solution output
    schedule = {str(n): {} for n in range(num_nurses)}
    for n in range(num_nurses):
        for d in range(num_days):
            schedule[str(n)][str(d)] = None
            for s in range(num_shift_types):
                if x[n][d][s].x > 0.5:
                    schedule[str(n)][str(d)] = shift_idx_to_id[s]  # Use original shift id
                    break

    # Prepare coverage output
    coverage_under_sol = {}
    coverage_over_sol = {}

    for req in coverage_requirements:
        if req["day"] < num_days and req["shift_type"] in shift_id_to_idx:
            d = req["day"]
            s_type = req["shift_type"]
            
            under_val = coverage_under[(d, s_type)].x
            if abs(under_val) > 1e-6:
                coverage_under_sol[f"{d}_{s_type}"] = round(under_val)

            over_val = coverage_over[(d, s_type)].x
            if abs(over_val) > 1e-6 and over_val > 0.001:
                coverage_over_sol[f"{d}_{s_type}"] = round(over_val)

    objective = model.objVal if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0 else float('inf')

    sol_output = {
        "objective_value": objective,
        "solution": {
            "schedule": schedule,
            "coverage_under": coverage_under_sol,
            "coverage_over": coverage_over_sol
        }
    }

    return sol_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None, required=False)
    global args
    global logger
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, "r") as f:
        instance_data = json.load(f)

    solution = solve_nurse_scheduling(instance_data, args.time_limit)

    # Write solution
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()