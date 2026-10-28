import argparse
import json
import sys
import gurobipy as gb

def main():
    parser = argparse.ArgumentParser(description="Nurse Scheduling Optimization")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the JSON problem instance.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to the output JSON solution.")
    parser.add_argument('--time_limit', type=int, required=True, help="Time limit for the solver in seconds.")
    parser.add_argument('--log_path', type=str, default=None, help="Path to the log file (optional).")
    
    args = parser.parse_args()
    
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_days = data['num_days']
    num_weeks = data['num_weeks']
    nurses = data['nurses']
    shifts = data['shifts']
    skills = data['skills']
    penalties = data['penalty_weights']
    
    model = gb.Model("NurseScheduling")
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 1)

    # Variables
    x = {}
    for i in nurses:
        n_id = i['id']
        for d in range(num_days):
            for s in shifts:
                s_name = s['name']
                for k in i['skills']:
                    x[n_id, d, s_name, k] = model.addVar(vtype=gb.GRB.BINARY, name=f"x_{n_id}_{d}_{s_name}_{k}")

    y_var = {}
    w_var = {}

    for i in nurses:
        n_id = i['id']
        for d in range(num_days):
            # Total works by this nurse on this day
            y_var[n_id, d] = model.addVar(vtype=gb.GRB.BINARY, name=f"y_{n_id}_{d}")
            model.addConstr(y_var[n_id, d] == gb.quicksum(x[n_id, d, s['name'], k] for s in shifts for k in i['skills']))
            model.addConstr(y_var[n_id, d] <= 1) # At most one shift per day
            
            # Shift presence helper
            for s in shifts:
                s_name = s['name']
                w_var[n_id, d, s_name] = model.addVar(vtype=gb.GRB.BINARY, name=f"w_{n_id}_{d}_{s_name}")
                model.addConstr(w_var[n_id, d, s_name] == gb.quicksum(x[n_id, d, s_name, k] for k in i['skills']))

    # Helper getters handling boundary conditions seamlessly
    def get_y(n_id, d):
        if d < 0 or d >= num_days:
            return 0
        return y_var[n_id, d]

    def get_u(n_id, d):
        if d < 0 or d >= num_days:
            return 0
        return 1 - y_var[n_id, d]

    def get_w(n_id, d, s_name):
        if d < 0 or d >= num_days:
            return 0
        return w_var[n_id, d, s_name]

    obj_terms = []

    # 1. Staffing Demands (c_S1)
    c_S1 = penalties.get('c_S1', 0)
    for d in range(num_days):
        day_demand = data['demand'][d] if d < len(data['demand']) else {}
        for s in shifts:
            s_name = s['name']
            shift_demand = day_demand.get(s_name, {})
            for k in skills:
                req = shift_demand.get(k, {})
                min_req = req.get('minimum', 0)
                opt_req = req.get('optimal', 0)
                
                worked = gb.quicksum(x[i['id'], d, s_name, k] for i in nurses if k in i['skills'])
                
                if min_req > 0:
                    model.addConstr(worked >= min_req, name=f"min_req_{d}_{s_name}_{k}")
                
                if opt_req > min_req:
                    slack = model.addVar(lb=0, name=f"slack_{d}_{s_name}_{k}")
                    model.addConstr(worked + slack >= opt_req, name=f"opt_req_{d}_{s_name}_{k}")
                    obj_terms.append(c_S1 * slack)

    # 2. Forbidden Shift Successions (Hard constraint)
    for s1, s2 in data.get('forbidden_shift_successions', []):
        for i in nurses:
            n_id = i['id']
            for d in range(num_days - 1):
                model.addConstr(w_var[n_id, d, s1] + w_var[n_id, d+1, s2] <= 1, name=f"fss_{n_id}_{d}_{s1}_{s2}")

    # Process Nurse Contracts
    c_S2a = penalties.get('c_S2a', 0)
    c_S2b = penalties.get('c_S2b', 0)
    c_S3 = penalties.get('c_S3', 0)
    c_S4 = penalties.get('c_S4', 0)
    c_S5 = penalties.get('c_S5', 0)
    c_S6 = penalties.get('c_S6', 0)
    c_S7 = penalties.get('c_S7', 0)

    for nurse in nurses:
        n_id = nurse['id']
        contract = nurse['contract']
        
        # Min/Max Consecutive Worked Days (c_S2a)
        CD_min = contract.get('CD_minus', 1)
        CD_max = contract.get('CD_plus', num_days)
        
        if CD_min > 1:
            for l in range(1, CD_min):
                for d in range(num_days - l + 1):
                    z_w = model.addVar(lb=0)
                    rhs = gb.LinExpr()
                    rhs += get_y(n_id, d)
                    rhs -= get_y(n_id, d-1)
                    rhs += get_y(n_id, d+l-1)
                    rhs -= get_y(n_id, d+l)
                    for t in range(d, d+l):
                        rhs -= 1
                        rhs += get_y(n_id, t)
                    rhs -= 1
                    model.addConstr(z_w >= rhs)
                    obj_terms.append(c_S2a * (CD_min - l) * z_w)

        for d in range(num_days - CD_max):
            v_w = model.addVar(lb=0)
            rhs = gb.LinExpr()
            for t in range(d, d + CD_max + 1):
                rhs += get_y(n_id, t)
            rhs -= CD_max
            model.addConstr(v_w >= rhs)
            obj_terms.append(c_S2a * v_w)

        # Min/Max Consecutive Days Off (c_S3)
        CR_min = contract.get('CR_minus', 1)
        CR_max = contract.get('CR_plus', num_days)
        
        if CR_min > 1:
            for l in range(1, CR_min):
                for d in range(num_days - l + 1):
                    z_u = model.addVar(lb=0)
                    rhs = gb.LinExpr()
                    rhs += get_u(n_id, d)
                    rhs -= get_u(n_id, d-1)
                    rhs += get_u(n_id, d+l-1)
                    rhs -= get_u(n_id, d+l)
                    for t in range(d, d+l):
                        rhs -= 1
                        rhs += get_u(n_id, t)
                    rhs -= 1
                    model.addConstr(z_u >= rhs)
                    obj_terms.append(c_S3 * (CR_min - l) * z_u)

        for d in range(num_days - CR_max):
            v_u = model.addVar(lb=0)
            rhs = gb.LinExpr()
            for t in range(d, d + CR_max + 1):
                rhs += get_u(n_id, t)
            rhs -= CR_max
            model.addConstr(v_u >= rhs)
            obj_terms.append(c_S3 * v_u)

        # Total Assignments (c_S6)
        L_min = contract.get('L_minus', 0)
        L_max = contract.get('L_plus', num_days)
        total_y = gb.quicksum(y_var[n_id, d] for d in range(num_days))
        dev_up = model.addVar(lb=0)
        dev_dn = model.addVar(lb=0)
        model.addConstr(dev_up >= total_y - L_max)
        model.addConstr(dev_dn >= L_min - total_y)
        obj_terms.append(c_S6 * (dev_up + dev_dn))

        # Weekend Policies (c_S5, c_S7)
        WE_max = contract.get('WE_plus', num_weeks)
        we_worked_vars = []
        for w in range(num_weeks):
            sat = 7 * w + 5
            sun = 7 * w + 6
            we_worked = model.addVar(vtype=gb.GRB.BINARY)
            model.addConstr(we_worked >= get_y(n_id, sat))
            model.addConstr(we_worked >= get_y(n_id, sun))
            we_worked_vars.append(we_worked)
            
            inc_we = model.addVar(lb=0)
            model.addConstr(inc_we >= get_y(n_id, sat) - get_y(n_id, sun))
            model.addConstr(inc_we >= get_y(n_id, sun) - get_y(n_id, sat))
            obj_terms.append(c_S5 * inc_we)
            
        excess_we = model.addVar(lb=0)
        model.addConstr(excess_we >= gb.quicksum(we_worked_vars) - WE_max)
        obj_terms.append(c_S7 * excess_we)

        # Preferences (c_S4)
        for pref in nurse.get('preferences', []):
            pref_d = pref['day']
            pref_s = pref['shift']
            if pref_s in [s['name'] for s in shifts] and 0 <= pref_d < num_days:
                obj_terms.append(c_S4 * w_var[n_id, pref_d, pref_s])

        # Min/Max Consecutive Same-Shift (c_S2b)
        for s in shifts:
            s_name = s['name']
            CS_min = s.get('CS_minus', 1)
            CS_max = s.get('CS_plus', num_days)
            
            if CS_min > 1:
                for l in range(1, CS_min):
                    for d in range(num_days - l + 1):
                        z_s = model.addVar(lb=0)
                        rhs = gb.LinExpr()
                        rhs += get_w(n_id, d, s_name)
                        rhs -= get_w(n_id, d-1, s_name)
                        rhs += get_w(n_id, d+l-1, s_name)
                        rhs -= get_w(n_id, d+l, s_name)
                        for t in range(d, d+l):
                            rhs -= 1
                            rhs += get_w(n_id, t, s_name)
                        rhs -= 1
                        model.addConstr(z_s >= rhs)
                        obj_terms.append(c_S2b * (CS_min - l) * z_s)

            for d in range(num_days - CS_max):
                v_s = model.addVar(lb=0)
                rhs = gb.LinExpr()
                for t in range(d, d + CS_max + 1):
                    rhs += get_w(n_id, t, s_name)
                rhs -= CS_max
                model.addConstr(v_s >= rhs)
                obj_terms.append(c_S2b * v_s)

    model.setObjective(gb.quicksum(obj_terms), gb.GRB.MINIMIZE)

    # Logger Callback Configuration
    def callback(cb_model, where):
        if where == gb.GRB.Callback.MIPSOL:
            obj = cb_model.cbGet(gb.GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Execute Optimization
    model.optimize(callback)

    # Prepare standard output
    out_schedule = []
    
    if model.SolCount > 0:
        obj_val = model.ObjVal
        for nurse in nurses:
            n_id = nurse['id']
            n_name = nurse['name']
            n_assigns = []
            
            for d in range(num_days):
                for s in shifts:
                    s_name = s['name']
                    for k in nurse['skills']:
                        if x[n_id, d, s_name, k].X > 0.5:
                            n_assigns.append({
                                "day": d,
                                "shift": s_name,
                                "skill": k
                            })
            
            out_schedule.append({
                "nurse_id": n_id,
                "nurse_name": n_name,
                "assignments": n_assigns
            })
    else:
        obj_val = 0.0
        for nurse in nurses:
            out_schedule.append({
                "nurse_id": nurse['id'],
                "nurse_name": nurse['name'],
                "assignments": []
            })
            
    output = {
        "objective_value": float(obj_val),
        "schedule": out_schedule
    }

    # Finalize solution to target schema
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()