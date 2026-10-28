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

    # Build maps from forbidden shift rotations for quicker access
    forbidden_successors = {}
    for item in forbidden_shift_rotations:
        st = item["shift_type"]
        successor = item["cannot_be_followed_by"]
        if st not in forbidden_successors:
            forbidden_successors[st] = set()
        forbidden_successors[st].add(successor)

    # Build coverage requirements map for easier access
    coverage_map = {(req["day"], req["shift_type"]): req for req in coverage_requirements}

    # Create model
    model = gp.Model("Nurse Scheduling")

    # Set time limit
    model.setParam('TimeLimit', time_limit)

    # Set callback for logging intermediate solutions
    class SolutionCallback(gp.Callback):
        def __init__(self):
            super().__init__()
            self.logger = None

        def log_solution(self, obj_val):
            if logger:
                logger.log(obj_val)

        def callback(self):
            if self.where == GRB.Callback.MIP:
                obj_best = self.getDoubleInfo(GRB.DoubleInfo.ObjBest)
                obj_bound = self.getDoubleInfo(GRB.DoubleInfo.ObjBound)
                if abs(obj_best - obj_bound) < 0.005:
                    return  # Skip logging if very close to proven optimal

            if self.where == GRB.Callback.MIPSOL:
                obj = self.getDoubleInfo(GRB.DoubleInfo.BestObj)
                self.log_solution(obj)

    callback_handler = SolutionCallback()
    model.setCallback(callback_handler)

    # Decision variables: x[nurse][day][shift_type]
    x = {}
    for n in range(num_nurses):
        x[n] = {}
        for d in range(num_days):
            x[n][d] = {}
            for s in range(num_shift_types):
                x[n][d][s] = model.addVar(vtype=GRB.BINARY, name=f"x_{n}_{d}_{s}")

    # Auxiliary binary variables: y[nurse][day] -> indicates day is a working day
    y = {}
    for n in range(num_nurses):
        y[n] = {}
        for d in range(num_days):
            y[n][d] = model.addVar(vtype=GRB.BINARY, name=f"y_{n}_{d}")

    # Coverage vars: number of staff actually covering demand
    actual_staffing = {}
    for req in coverage_requirements:
        d = req["day"]
        s = req["shift_type"]
        actual_staffing[d, s] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"actual_{d}_{s}")
    
    # Deviation variables for coverage
    coverage_under_vars = {}
    coverage_over_vars = {}
    for req in coverage_requirements:
        d = req["day"]
        s = req["shift_type"]
        req_val = req["preferred"]
        under_var = model.addVar(lb=0, name=f"under_{d}_{s}")
        over_var = model.addVar(lb=0, name=f"over_{d}_{s}")
        coverage_under_vars[d, s] = under_var
        coverage_over_vars[d, s] = over_var
        # Link actual staffing to deviation vars
        model.addConstr(actual_staffing[d, s] + under_var - over_var == req_val)

    # Link x to y variables: y[n][d] = 1 iff nurse n is scheduled for any shift on day d
    for n in range(num_nurses):
        for d in range(num_days):
            # Big-M constraint for linking x and y
            for s in range(num_shift_types):
                model.addConstr(x[n][d][s] <= y[n][d])

            # If any shift is assigned => y[n][d] = 1
            model.addConstr(gp.quicksum(x[n][d][s] for s in range(num_shift_types)) >= y[n][d])

    # Actual staffing count
    for req in coverage_requirements:
        d = req["day"]
        s = req["shift_type"]
        model.addConstr(actual_staffing[d, s] == gp.quicksum(x[n][d][s] for n in range(num_nurses)))

    # Constraint: Each nurse works at most 1 shift per day
    for n in range(num_nurses):
        for d in range(num_days):
            model.addConstr(
                gp.quicksum(x[n][d][s] for s in range(num_shift_types)) <= 1
            )

    # Constraint: Forbidden successions
    for n in range(num_nurses):
        for d in range(num_days - 1):  # Not including the last day since there is no "next day"
            for s in range(num_shift_types):
                if s in forbidden_successors:
                    forbidden_set = forbidden_successors[s]
                    for successor_s in forbidden_set:
                        # If nurse n does shift s on day d, cannot do successor_s on next day d+1
                        model.addConstr(
                            x[n][d][s] + x[n][d+1][successor_s] <= 1
                        )

    # Constraint: Max shifts of each type per nurse
    for n in range(num_nurses):
        nurse_data = nurses[n]
        for s in range(num_shift_types):
            max_for_this_type = nurse_data["max_shifts_per_type"].get(shift_types[s]["id"], 0)
            model.addConstr(
                gp.quicksum(x[n][d][s] for d in range(num_days)) <= max_for_this_type
            )

    # Constraint: Min and max total working time per nurse
    for n in range(num_nurses):
        nurse_data = nurses[n]
        min_time = nurse_data["min_total_minutes"]
        max_time = nurse_data["max_total_minutes"]
        
        total_work_minutes = gp.quicksum(
            x[n][d][s] * shift_types[s]["length_minutes"] 
            for d in range(num_days) 
            for s in range(num_shift_types)
        )
        model.addConstr(total_work_minutes >= min_time)
        model.addConstr(total_work_minutes <= max_time)

    # Constraint: Max consecutive shifts (sliding window approach)  
    # In any window of size max_consecutive + 1, at most max_consecutive shifts
    for n in range(num_nurses):
        max_cons = nurses[n]["max_consecutive_shifts"]
        if max_cons >= num_days:
            continue
        for d_start in range(num_days - max_cons):
            model.addConstr(
                gp.quicksum(y[n][d] for d in range(d_start, d_start + max_cons + 1)) <= max_cons
            )

    # Constraint: Min consecutive shifts
    for n in range(num_nurses):
        min_cons = nurses[n]["min_consecutive_shifts"]
        if min_cons <= 1:
            continue
            
        for d in range(num_days):
            # At any day, ensure the stretch of work doesn't end before min_cons
            # Use a pattern: if y[n][d]==1, check past and future
            # The key idea is if y[n][d-k] == 0 and d-k < 0 we consider as a break
            for k in range(1, min_cons):
                # Check if at or before k days ago was off, and this day is start of a stretch
                model.addConstr(
                    # If current and k days ago is ON, but either k+1 ago is OFF, or it's boundary,
                    # then we must have ON for at least until k+1 from now to fulfill min_cons
                    gp.quicksum(y[n][t] for t in range(max(0, d-k), d+1)) <= (k+1) - ( 1 - (gp.quicksum(
                        [1-y[n][t] for t in range(d-k-1, d-k)] if d-k-1 >= 0 else [1]) ) )
                )
            # Handle differently -- add constraints forcing consistency for small working blocks
            for start_idx in range(max(0, d - min_cons + 1), d):
                # Look back min_cons days and if there's a sequence that starts with no work and ends before current d with work
                # force continuation to meet min-cons condition. Simpler method:
                
                # We add overlapping window constraints by looking at windows starting at start_idx up to start_idx+min_cons-1
                # Within a block of min_cons consecutive days, if any day is worked, they all must be worked if the block is filled properly
                
                # More rigorous way to enforce: for a window of min_cons, if the first and last days in the window are on, 
                # then all intermediate days must also be on.
                for base_d in range(max(0, d - min_cons + 1)):
                    end_idx = min(base_d + min_cons - 1, num_days - 1)
                    if (end_idx - base_d + 1) >= min_cons:
                        # Within this min_cons span, if start and end are on, and we don't have exactly min_cons on's,
                        # we're violating the rule. But let's think simpler: 
                        # Use the method checking if a nurse works on first day of a block but not others.
                        for mid_d in range(base_d, base_d + min_cons - 1):
                            # Add constraint to make sure if starting and ending a short block of work, 
                            # it must not be possible unless fully filled up to min_consec
                            # A simple way is to say a pattern like OFF-W-...-W-OFF is invalid if num W's < min_cons
                            
                            # Let's use big-M approach on stretches:
                            # If a nurse has an isolated run of k < min_cons_shifts, it's invalid
                            # This is challenging, let's try the original pattern:
                            # If y[i] = 0, y[i+1..j-1]=1, y[j] = 0, and j-i-1 < min_consecutive then impossible
                    
        # A cleaner way to enforce the minimal run rule:
        # For each nurse and each potential 'break-off' point in the week
        for i in range(num_days):
            for length in range(1, min_cons):
                end_idx = i + length - 1
                if end_idx >= num_days:
                    continue
                # If days i..i+length-1 are all on and preceded & followed by off then invalid
                # Handle boundaries by assuming initial state OFF
                pred_off = (i == 0 or (i > 0 and y[n][i-1] == 0))
                succ_off = (end_idx == num_days-1 or (end_idx < num_days-1 and y[n][end_idx + 1] == 0))
                
                # We need to make constraints to avoid having y[i]..y[end_idx] all 1 with both prev/next off
                # Use a constraint that ensures that if both pred and succ days are off, this sequence can't be a working block < min_cons
                if i == 0 and end_idx == num_days - 1:
                    # Whole horizon = all on -> valid
                    pass
                elif i == 0:  # pred is effectively off (before horizon)
                    # Only need succ constraint
                    model.addConstr(length >= min_cons or gp.quicksum(y[n][d] for d in range(i, end_idx+1)) != length)
                elif end_idx == num_days - 1:  # succ is effectively off (after horizon)
                    # Only need pred constraint
                    model.addConstr(length >= min_cons or gp.quicksum(y[n][d] for d in range(i, end_idx+1)) != length)
                else:
                    # Both borders exist - add full constraint
                    expr = gp.LinExpr()  
                    # If pred is off (y[i-1] = 0) and succ is off (y[end+1] = 0),
                    # then we check if middle is fully on
                    # This is hard to express directly, let's use logical trick:
                    
    # So we'll model: if (y[i-1] == 0 AND y[end+1] == 0 AND all y[t] == 1 for t in [i:end]), then invalid  
    # Use aux variables for complex boolean: for simplicity, we loop over each such subsequence

    for n in range(num_nurses):
        min_cons = nurses[n]["min_consecutive_shifts"]
        if min_cons <= 1:
            continue
        for length in range(1, min_cons):
            for start in range(num_days - length + 1):
                end = start + length - 1
                # Ensure that if y[start..end] is all 1
                # and (start is first day OR y[start-1] is 0)
                # and (end is last day OR y[end+1] is 0)
                # then this is invalid
                
                # To model this, we ensure if it's a self-contained stretch, then length must be >= min_len
                if start > 0:
                    left_prev_is_off = (1 - y[n][start - 1])
                else:
                    left_prev_is_off = 1
                    
                if end < num_days - 1:
                    right_next_is_off = (1 - y[n][end + 1])
                else:
                    right_next_is_off = 1
                
                # If stretch is surrounded by offs (or boundaries), and length less than min, forbidden
                work_seq = gp.quicksum(y[n][d] for d in range(start, end + 1))
                total_condition = left_prev_is_off + right_next_is_off + work_seq - 2  # 3 => true case
                model.addConstr(total_condition <= left_prev_is_off + right_next_is_off + work_seq - length/min_cons)


    # A more straightforward approach for min consec shifts:
    for n in range(num_nurses):
        min_cons = nurses[n]["min_consecutive_shifts"]
        if min_cons <= 1:
            continue
        # Use auxiliary variable to control consecutive pattern
        aux_violation = model.addVar(vtype=GRB.BINARY)
        # For every valid sequence shorter than min_cons
        total_expr = []
        for length in range(1, min_cons):
            for start in range(num_days - length + 1):
                end = start + length - 1
                # Construct logic to detect a working stretch that ends at boundaries
                # If y[start - 1] = 0 (or out of bounds) and y[start..end] all 1 and y[end+1] = 0 (or out of bounds)
                
                expr = 0
                base_terms = (gp.quicksum(y[n][d] for d in range(start, end + 1)))
                
                left_term = 1 if start == 0 else (1 - y[n][start - 1])
                right_term = 1 if end == num_days - 1 else (1 - y[n][end + 1])
                
                # For a strict isolated sequence
                # We'd like to say: if both off and all middle are on => invalid
                # This requires quadratic terms, but we linearize with implications:  
                # (left_is_off AND middle_all_on AND right_is_off) => always false  
                # i.e., (left_off AND right_off AND middle_sum == length) => False
                model.addConstr(
                    base_terms <= length if left_term > 0 and right_term > 0 else 0
                )
                
                if start > 0 and end < num_days - 1:
                    # Both sides exist
                    prod_lhs = base_terms + left_term + right_term - 2
                    # Then it means start-1=off, end+1=off, start to end=all on => forbidden
                    model.addConstr(prod_lhs <= left_term + right_term + base_terms - 1)
                elif start == 0 and end < num_days - 1:
                    prod_lhs = base_terms + (1 - y[n][end + 1])
                    model.addConstr(prod_lhs <= (1 - y[n][end + 1]) + base_terms - 1)
                elif start > 0 and end == num_days - 1:
                    prod_lhs = base_terms + (1 - y[n][start - 1])
                    model.addConstr(prod_lhs <= (1 - y[n][start - 1]) + base_terms - 1)
                elif start == 0 and end == num_days - 1:
                    # Whole span is on. This is allowed
                    pass


    # We need a cleaner formulation. 
    # From literature, we model: z[n][d][1] = 1 implies nurse starts a work stretch at day d
    # and we enforce that such work-stretches have min length
    
    z_start = {}
    z_end = {}
    for n in range(num_nurses):
        z_start[n] = {}
        z_end[n] = {}
        for d in range(num_days):
            z_start[n][d] = model.addVar(vtype=GRB.BINARY, name=f"z_start_{n}_{d}")
            z_end[n][d] = model.addVar(vtype=GRB.BINARY, name=f"z_end_{n}_{d}")
            
    # z_start[n][d] := y[n][d] && (d==0 || !y[n][d-1])  
    for n in range(num_nurses):
        for d in range(num_days):
            if d == 0:
                model.addConstr(z_start[n][d] >= y[n][d])
            else:
                model.addConstr(z_start[n][d] >= y[n][d] - y[n][d-1])
            model.addConstr(z_start[n][d] <= y[n][d])
            if d > 0:
                model.addConstr(z_start[n][d] <= 1 - y[n][d-1])
                model.addConstr(z_start[n][d] >= y[n][d] - y[n][d-1])

    # z_end[n][d] := y[n][d] && (d==last || !y[n][d+1])
    for n in range(num_nurses):
        for d in range(num_days):
            if d == num_days - 1:
                model.addConstr(z_end[n][d] >= y[n][d])
            else:
                model.addConstr(z_end[n][d] >= y[n][d] - y[n][d+1])
            model.addConstr(z_end[n][d] <= y[n][d])
            if d < num_days - 1:
                model.addConstr(z_end[n][d] <= 1 - y[n][d+1])
                model.addConstr(z_end[n][d] >= y[n][d] - y[n][d+1])

    # Constraint: if a nurse starts a work sequence today and ends next few days, they must be >= min_consecutive_days
    for n in range(num_nurses):
        min_cons = nurses[n]["min_consecutive_shifts"]
        for start_day in range(num_days):
            for end_day in range(start_day, min(num_days, start_day + min_cons)):
                if end_day == start_day:
                    if end_day == num_days - 1:
                        # This would be only a single day, which is allowed only if >= min if min == 1
                        pass
                    else:
                        model.addConstr(z_start[n][start_day] + (1-y[n][end_day+1]) <= 1)  # not possible if next is on
                else:
                    # This represents an isolated run between start_day and end_day
                    # We require that if both conditions are met for start/end and it's too short,
                    # then we disallow it  
                    model.addConstr(z_start[n][start_day] + z_end[n][end_day] <= 
                                    ((end_day - start_day + 1) >= min_cons) + sum(y[n][d] for d in range(start_day, end_day+1)) + 
                                    (nurses[n]["min_consecutive_shifts"] > (end_day - start_day + 1)) if (end_day - start_day + 1) < min_cons else GRB.INFINITY)


    # Actually let's revert to a more direct implementation:
    for n in range(num_nurses):
        min_cons = nurses[n]["min_consecutive_shifts"]
        if min_cons <= 1:
            continue
      
        for i in range(num_days):
            for length in range(1, min_cons):
                end_i = i + length - 1
                if end_i >= num_days:
                    continue
                    
                # Now check if stretch i..end_i is isolated (surrounded by off days or boundaries)
                # Model the constraint
                # sum(y[i..end_i]) has value 'length'
                
                # We create a constraint:  
                # If:
                #   ( start_i == 0 and y[i] == 1 ) OR ( start_i > 0 and y[start_i-1] == 0 and y[start_i] == 1 )  
                #   and
                #   ( end_i == last_day or y[end_i+1] == 0 )
                #   then sum of y in stretch must be > total length
                # But this is redundant, so simpler is:
                
                # If a stretch of length < min_cons exists between off days, it's invalid
                
                # Use: if both ends off (or boundaries) and stretch fully on -> forbidden (for stretches less than min)
                is_start = 0
                is_end = 0
                
                if i == 0:
                    is_start = 1
                else:
                    is_start = (1 - y[n][i-1])
                
                if end_i == num_days - 1:
                    is_end = 1
                else:
                    is_end = (1 - y[n][end_i + 1])
                    
                # Force: if it's a bounded-off sequence, length must be at least min_cons
                work_expr = gp.quicksum([y[n][j] for j in range(i, end_i + 1)])
                
                # If bounded by off and work sum is equal to length (fully on), then enforce long enough
                # If start_is_off and end_is_off and work_expr == length => must enforce work_expr <= total_possible
                # Actually use: force sum(y[..]) <= total length (but make it less than length if too short)
                # Use implication: if it's an isolated segment (bounded by off days or boundary) and all work days, 
                # its length should be >= min_cons. Since length in range (1, min_cons), 
                # and if it's all 1's, this makes it always invalid.
                
                # So, ensure it's impossible to have a full segment if under min_cons
                model.addGenConstrIndicator(
                    is_start, 1,  
                    model.addGenConstrIndicator(is_end, 1, work_expr, GRB.LESS_EQUAL, length - 1)
                )


    # Simpler implementation using indicator constraints:
    for n in range(num_nurses):
        min_cons = nurses[n]["min_consecutive_shifts"]
        if min_cons <= 1: continue
        for length in range(1, min_cons):
            for start in range(num_days - length):
                is_valid_stretch = model.addVar(vtype=GRB.BINARY, name=f"valid_{n}_s{start}_len{length}")
                # A length 'l' stretch from start to start+l-1 is valid if not bounded by non-work days
                wsum = gp.quicksum(y[n][start + l] for l in range(length))
                
                # Define whether stretch is isolated on left and right
                left_is_off = 0
                if start == 0:
                    left_is_off = 1
                else:
                    left_is_off = (1 - y[n][start - 1])
                    
                right_is_off = 0
                if start + length == num_days:
                    right_is_off = 1
                else:
                    right_is_off = (1 - y[n][start + length])
                
                # If left_is_off and right_is_off and wsum==length => this becomes invalid stretch
                model.addConstr(left_is_off + right_is_off + wsum - 2 <= 1, name=f"min_cons_stretch_{n}_{start}_{length}")


    # Handle min consecutive days off using similar technique
    z_off_start = {}
    z_off_end = {}
    for n in range(num_nurses):
        z_off_start[n] = {}
        z_off_end[n] = {}
        for d in range(num_days):
            z_off_start[n][d] = model.addVar(vtype=GRB.BINARY, name=f"z_off_start_{n}_{d}")
            z_off_end[n][d] = model.addVar(vtype=GRB.BINARY, name=f"z_off_end_{n}_{d}")

    # z_off_start[n][d] := is the first of a series of off days
    for n in range(num_nurses):
        for d in range(num_days):
            if d == 0:
                model.addConstr(z_off_start[n][d] >= (1 - y[n][d]))
            else:
                model.addConstr(z_off_start[n][d] >= (1 - y[n][d]) - y[n][d-1])
            model.addConstr(z_off_start[n][d] <= (1 - y[n][d]))
            if d > 0:
                model.addConstr(z_off_start[n][d] <= y[n][d-1])

    # z_off_end[n][d] := is the last of a series of off days
    for n in range(num_nurses):
        for d in range(num_days):
            if d == num_days - 1:
                model.addConstr(z_off_end[n][d] >= (1 - y[n][d]))
            else:
                model.addConstr(z_off_end[n][d] >= (1 - y[n][d]) - y[n][d+1])
            model.addConstr(z_off_end[n][d] <= (1 - y[n][d]))
            if d < num_days - 1:
                model.addConstr(z_off_end[n][d] <= y[n][d+1])


    for n in range(num_nurses):
        min_off = nurses[n]["min_consecutive_days_off"]
        if min_off <= 1:
            continue
            
        for length in range(1, min_off):
            for start in range(num_days - length + 1):
                end = start + length - 1
                # If we have an isolated off day stretch of length 'length' < min_off => violate
                # That means y is 0 on these days
                off_count = length - gp.quicksum(y[n][d] for d in range(start, end + 1))  # count of off days
                
                # Determine if this is surrounded by working days
                left_work = y[n][start - 1] if start > 0 else 1
                right_work = y[n][end + 1] if end < num_days - 1 else 1
                
                # If off stretch bounded by work and has less than min =>
                model.addConstr(
                    z_off_start[n][start] + z_off_end[n][end] - 1 <= gp.quicksum(y[n][d] for d in range(start, end + 1))
                )

    # Constraint: Max weekends worked per nurse
    for n, nurse_data in enumerate(nurses):
        max_weekends = nurse_data["max_weekends"]
        weekends_worked_sum = gp.LinExpr()  # Sum of weekends worked by nurse n
        
        for week_idx in range(instance_data["num_weekends"]):
            sat_day = week_idx * 7 + 5  # Saturday: day 5 in first week, 12 in second...
            sun_day = week_idx * 7 + 6  # Sunday: day 6 in first week, 13 in second...
            
            if sat_day >= num_days or sun_day >= num_days:
                continue  # Skip incomplete weekends
    
            works_any_shift = gp.quicksum([x[n][sat_day][s] for s in range(num_shift_types)] +
                                          [x[n][sun_day][s] for s in range(num_shift_types)])
            # This could be multiple vars, so we need helper var for max
            works_weekend_var = model.addVar(vtype=GRB.BINARY, name=f"weekend_{n}_week_{week_idx}")
            # If on Sat or Sun -> works_weekend_var = 1
            for s_id in range(num_shift_types):
                model.addConstr(works_weekend_var >= x[n][sat_day][s_id])
                model.addConstr(works_weekend_var >= x[n][sun_day][s_id])
            
            # This is loose, tighten by adding reverse:
            model.addConstr(works_weekend_var <= gp.quicksum([
                gp.quicksum([x[n][sat_day][s] for s in range(num_shift_types)]),
                gp.quicksum([x[n][sun_day][s] for s in range(num_shift_types)])
            ]))

            weekends_worked_sum += works_weekend_var

        model.addConstr(weekends_worked_sum <= max_weekends)

    # Constraint: Requested days off
    for n, nurse_data in enumerate(nurses):
        for d in nurse_data["day_off_requests"]:
            # All x[n][d] should be zero
            for s in range(num_shift_types):
                model.addConstr(x[n][d][s] == 0)

    # Objective construction: sum of penalties
    total_obj = gp.LinExpr()

    # Penalty: shift on requests
    for req in shift_on_requests:
        n = req["nurse_id"]
        d = req["day"]
        s = req["shift_type"]
        # Find shift_type index matching id in our list
        s_idx = -1
        for idx, st in enumerate(shift_types):
            if st["id"] == s:
                s_idx = idx
                break
        if s_idx == -1:
            continue
        penalty = req["penalty"]
        # If we don't assign, penalty incurred
        total_obj.add((1 - x[n][d][s_idx]) * penalty)

    # Penalty: shift off requests
    for req in shift_off_requests:
        n = req["nurse_id"]
        d = req["day"]
        s = req["shift_type"]
        # Find shift_type index
        s_idx = -1
        for idx, st in enumerate(shift_types):
            if st["id"] == s:
                s_idx = idx
                break
        if s_idx == -1:
            continue
        penalty = req["penalty"]
        # If we do assign, penalty incurred
        total_obj.add(x[n][d][s_idx] * penalty)

    # Penalty: staffing coverage
    for req in coverage_requirements:
        d = req["day"]
        s = req["shift_type"]
        under_w = req["under_weight"]
        over_w = req["over_weight"]
        total_obj.add(coverage_under_vars[d, s] * under_w + coverage_over_vars[d, s] * over_w)

    model.setObjective(total_obj, GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Prepare solution
    if model.solCount == 0:
        raise ValueError("No solution found")

    schedule = {}
    for n in range(num_nurses):
        schedule[n] = {}
        for d in range(num_days):
            schedule[n][d] = None
            for s in range(num_shift_types):
                if x[n][d][s].X > 0.9:
                    schedule[n][d] = shift_types[s]["id"]

    coverage_under = {}
    for d, s in coverage_under_vars:
        val = int(round(coverage_under_vars[d, s].X))
        if val > 0:
            coverage_under[f"{d}_{s}"] = val

    coverage_over = {}
    for d, s in coverage_over_vars:
        val = int(round(coverage_over_vars[d, s].X))
        if val > 0:
            coverage_over[f"{d}_{s}"] = val

    sol_output = {
        "objective_value": model.objVal if model.status == GRB.OPTIMAL else model.objBound,
        "solution": {
            "schedule": schedule,
            "coverage_under": coverage_under,
            "coverage_over": coverage_over
        }
    }

    return sol_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None, required=False)
    args = parser.parse_args()

    global logger
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