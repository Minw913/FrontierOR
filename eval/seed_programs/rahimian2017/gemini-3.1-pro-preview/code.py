import argparse
import json
import gurobipy as gb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    solve(instance, args.solution_path, args.time_limit, logger)


def solve(instance, solution_path, time_limit, logger):
    num_days = int(instance['num_days'])
    num_weeks = int(instance['num_weeks'])
    nurses = instance['nurses']
    shift_types = instance['shift_types']

    model = gb.Model("NurseScheduling")
    model.setParam('Threads', 1)
    if time_limit > 0:
        model.setParam('TimeLimit', time_limit)
        
    model.setParam('MIPFocus', 1)

    # Pre-select valid (Nurse, Day, ShiftType) assignments to minimize variable creation
    valid_x = []
    for n in nurses:
        n_id = n['id']
        req_off = set(n.get('day_off_requests', []))
        max_shifts_dict = n.get('max_shifts_per_type', {})
        for d in range(num_days):
            if d in req_off:
                continue
            for s in shift_types:
                s_id = s['id']
                max_s = int(max_shifts_dict.get(str(s_id), num_days))
                if max_s > 0:
                    valid_x.append((n_id, d, s_id))

    x = {}
    for (n_id, d, s_id) in valid_x:
        x[n_id, d, s_id] = model.addVar(vtype=gb.GRB.BINARY, name=f"x_{n_id}_{d}_{s_id}")

    # Build continuous W (Working Day) expressions representing if a nurse works any shift on a particular day
    # W[n_id, d] will always be an instance of gb.LinExpr to cleanly handle constraint building below
    W = {}
    for n in nurses:
        n_id = n['id']
        req_off = set(n.get('day_off_requests', []))
        for d in range(num_days):
            if d in req_off:
                W[n_id, d] = gb.LinExpr()
            else:
                vars_for_day = [x[n_id, d, s['id']] for s in shift_types if (n_id, d, s['id']) in x]
                if vars_for_day:
                    w_expr = gb.LinExpr([1.0]*len(vars_for_day), vars_for_day)
                    W[n_id, d] = w_expr
                    model.addConstr(w_expr <= 1, name=f"AtMostOneShift_{n_id}_{d}")
                else:
                    W[n_id, d] = gb.LinExpr()

    # Constraint: Forbidden Shift Rotations
    forbidden_map = {}
    for fr in instance.get('forbidden_shift_rotations', []):
        s1 = fr['shift_type']
        s2 = fr['cannot_be_followed_by']
        forbidden_map.setdefault(s1, set()).add(s2)

    for n in nurses:
        n_id = n['id']
        for d in range(num_days - 1):
            for s1_id, s2_set in forbidden_map.items():
                if (n_id, d, s1_id) in x:
                    for s2_id in s2_set:
                        if (n_id, d + 1, s2_id) in x:
                            model.addConstr(
                                x[n_id, d, s1_id] + x[n_id, d + 1, s2_id] <= 1,
                                name=f"ForbidRot_{n_id}_{d}_{s1_id}_{s2_id}"
                            )

    # Constraint: Maximum shifts per type
    for n in nurses:
        n_id = n['id']
        max_shifts_dict = n.get('max_shifts_per_type', {})
        for s in shift_types:
            s_id = s['id']
            max_s = int(max_shifts_dict.get(str(s_id), num_days))
            vars_s = [x[n_id, d, s_id] for d in range(num_days) if (n_id, d, s_id) in x]
            if vars_s:
                model.addConstr(gb.LinExpr([1.0]*len(vars_s), vars_s) <= max_s, name=f"MaxShiftType_{n_id}_{s_id}")

    # Constraint: Total Working Time
    for n in nurses:
        n_id = n['id']
        min_t = int(n['min_total_minutes'])
        max_t = int(n['max_total_minutes'])
        
        coeffs = []
        vars_t = []
        for d in range(num_days):
            for s in shift_types:
                s_id = s['id']
                if (n_id, d, s_id) in x:
                    coeffs.append(s['length_minutes'])
                    vars_t.append(x[n_id, d, s_id])
        if vars_t:
            expr_time = gb.LinExpr(coeffs, vars_t)
            model.addConstr(expr_time >= min_t, name=f"MinTotalTime_{n_id}")
            model.addConstr(expr_time <= max_t, name=f"MaxTotalTime_{n_id}")
        else:
            if min_t > 0:
                # Should not be possible to satisfy if bounded and no assignments enabled
                dummy = model.addVar(ub=0.0, name=f"DummyInfeas_{n_id}")
                model.addConstr(dummy >= min_t, name=f"InfeasTime_{n_id}")

    # Constraint: Maximum Consecutive Shifts
    for n in nurses:
        n_id = n['id']
        K = int(n['max_consecutive_shifts'])
        for i in range(num_days - K):
            expr = gb.LinExpr()
            for m in range(K + 1):
                expr += W[n_id, i + m]
            model.addConstr(expr <= K, name=f"MaxConsecutiveShifts_{n_id}_{i}")

    # Constraint: Minimum Consecutive Shifts
    for n in nurses:
        n_id = n['id']
        M = int(n['min_consecutive_shifts'])
        for k in range(1, min(M, num_days)):
            for i in range(1, num_days - k):
                expr = gb.LinExpr()
                expr += W[n_id, i - 1]
                expr += W[n_id, i + k]
                
                rhs = gb.LinExpr()
                for m in range(i, i + k):
                    rhs += W[n_id, m]
                rhs += 1 - k
                model.addConstr(expr >= rhs, name=f"MinConsecutiveShifts_{n_id}_{i}_{k}")

    # Constraint: Minimum Consecutive Days Off
    for n in nurses:
        n_id = n['id']
        L = int(n['min_consecutive_days_off'])
        for k in range(1, min(L, num_days)):
            for i in range(1, num_days - k):
                expr = gb.LinExpr()
                for m in range(i, i + k):
                    expr += W[n_id, m]
                
                rhs = gb.LinExpr()
                rhs += W[n_id, i - 1]
                rhs += W[n_id, i + k]
                rhs += -1
                model.addConstr(expr >= rhs, name=f"MinConsecutiveDaysOff_{n_id}_{i}_{k}")

    # Constraint: Maximum Weekends Worked
    for n in nurses:
        n_id = n['id']
        max_w = int(n['max_weekends'])
        we_vars = []
        for w in range(num_weeks):
            d_sat = w * 7 + 5
            d_sun = w * 7 + 6
            we_var = model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, ub=1, name=f"WE_{n_id}_{w}")
            
            model.addConstr(we_var >= W[n_id, d_sat])
            model.addConstr(we_var >= W[n_id, d_sun])
            
            rhs = gb.LinExpr()
            rhs += W[n_id, d_sat]
            rhs += W[n_id, d_sun]
            model.addConstr(we_var <= rhs)
            
            we_vars.append(we_var)
        
        if we_vars:
            model.addConstr(gb.quicksum(we_vars) <= max_w, name=f"MaxWeekends_{n_id}")

    # Constraints & Objective Component: Staffing Coverage Target Level Setup
    obj = gb.LinExpr()
    U_vars, O_vars = {}, {}

    for req in instance.get('coverage_requirements', []):
        d = req['day']
        s_id = req['shift_type']
        pref = req['preferred']
        uw = float(req['under_weight'])
        ow = float(req['over_weight'])
        
        U = model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, name=f"U_{d}_{s_id}")
        O = model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, name=f"O_{d}_{s_id}")
        U_vars[(d, s_id)] = U
        O_vars[(d, s_id)] = O
        
        assigned_vars = [x[n['id'], d, s_id] for n in nurses if (n['id'], d, s_id) in x]
        if assigned_vars:
            assigned = gb.LinExpr([1.0]*len(assigned_vars), assigned_vars)
        else:
            assigned = gb.LinExpr()
            
        model.addConstr(assigned + U - O == pref, name=f"CoverageReq_{d}_{s_id}")
        
        if uw > 0:
            obj += uw * U
        if ow > 0:
            obj += ow * O

    # Objective Component: Shift-On Requests Formulation
    for req in instance.get('shift_on_requests', []):
        n_id, d, s_id = req['nurse_id'], req['day'], req['shift_type']
        p = float(req['penalty'])
        if p > 0:
            if (n_id, d, s_id) in x:
                obj += p - p * x[n_id, d, s_id]
            else:
                obj += p

    # Objective Component: Shift-Off Requests Formulation
    for req in instance.get('shift_off_requests', []):
        n_id, d, s_id = req['nurse_id'], req['day'], req['shift_type']
        p = float(req['penalty'])
        if p > 0:
            if (n_id, d, s_id) in x:
                obj += p * x[n_id, d, s_id]

    model.setObjective(obj, gb.GRB.MINIMIZE)

    # Custom Logging Callback Hook Registration
    def logging_callback(model, where):
        if where == gb.GRB.Callback.MIPSOL:
            if logger is not None:
                obj_val = model.cbGet(gb.GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)

    # Initiate Optimization Solving Phase
    model.optimize(logging_callback)

    if model.SolCount > 0:
        schedule_out = {}
        for n in nurses:
            n_id_str = str(n['id'])
            schedule_out[n_id_str] = {}
            for d in range(num_days):
                assigned_shift = None
                for s in shift_types:
                    s_id = s['id']
                    if (n['id'], d, s_id) in x and x[n['id'], d, s_id].X > 0.5:
                        assigned_shift = s_id
                        break
                schedule_out[n_id_str][str(d)] = assigned_shift

        cov_under_out = {}
        cov_over_out = {}
        
        for (d, s_id) in U_vars.keys():
            v_u = int(round(U_vars[(d, s_id)].X))
            v_o = int(round(O_vars[(d, s_id)].X))
            
            # Avoid any potential minor degeneracy combinations natively handled by integer constraints mapping
            if v_u > 0 and v_o > 0:
                m = min(v_u, v_o)
                v_u -= m
                v_o -= m

            if v_u > 0:
                cov_under_out[f"{d}_{s_id}"] = v_u
            if v_o > 0:
                cov_over_out[f"{d}_{s_id}"] = v_o

        solution_record = {
            "objective_value": float(model.ObjVal),
            "solution": {
                "schedule": schedule_out,
                "coverage_under": cov_under_out,
                "coverage_over": cov_over_out
            }
        }
        
        with open(solution_path, 'w') as f:
            json.dump(solution_record, f, indent=2)

if __name__ == "__main__":
    main()