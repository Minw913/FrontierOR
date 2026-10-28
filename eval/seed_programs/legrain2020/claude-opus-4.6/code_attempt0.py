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
    
    num_nurses = data['num_nurses']
    num_weeks = data['num_weeks']
    num_days = data['num_days']
    num_shifts = data['num_shifts']
    num_skills = data['num_skills']
    skills = data['skills']
    shifts_data = data['shifts']
    shift_names = [s['name'] for s in shifts_data]
    forbidden = data['forbidden_shift_successions']
    weights = data['penalty_weights']
    nurses_data = data['nurses']
    demand_data = data['demand']
    
    # Index mappings
    shift_idx = {s['name']: i for i, s in enumerate(shifts_data)}
    skill_idx = {sk: i for i, sk in enumerate(skills)}
    
    # Shift properties
    CS_minus = {s['name']: s['CS_minus'] for s in shifts_data}
    CS_plus = {s['name']: s['CS_plus'] for s in shifts_data}
    
    # Forbidden succession pairs as index pairs
    forbidden_pairs = set()
    for pair in forbidden:
        if len(pair) == 2:
            forbidden_pairs.add((pair[0], pair[1]))
    
    # Parse demand: demand_data is a list of length num_days
    # demand_dict[day][shift][skill] = (min, opt)
    demand_dict = {}
    for d in range(num_days):
        demand_dict[d] = {}
        day_demand = demand_data[d]
        for sn in shift_names:
            demand_dict[d][sn] = {}
            if sn in day_demand:
                for sk in skills:
                    if sk in day_demand[sn]:
                        demand_dict[d][sn][sk] = (day_demand[sn][sk]['minimum'], day_demand[sn][sk]['optimal'])
                    else:
                        demand_dict[d][sn][sk] = (0, 0)
            else:
                for sk in skills:
                    demand_dict[d][sn][sk] = (0, 0)
    
    # Nurse data
    nurse_skills = {}
    nurse_contract = {}
    nurse_prefs = {}
    for n_data in nurses_data:
        nid = n_data['id']
        nurse_skills[nid] = set(n_data['skills'])
        nurse_contract[nid] = n_data['contract']
        nurse_prefs[nid] = set()
        for p in n_data['preferences']:
            nurse_prefs[nid].add((p['day'], p['shift']))
    
    nurse_ids = [n['id'] for n in nurses_data]
    nurse_names = {n['id']: n['name'] for n in nurses_data}
    
    # Build MIP model
    model = gp.Model("nurse_scheduling")
    model.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time) - 5))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
    
    # Decision variables
    # x[n, d, s, k] = 1 if nurse n works day d, shift s, skill k
    x = {}
    for n in nurse_ids:
        for d in range(num_days):
            for s in shift_names:
                for k in skills:
                    if k in nurse_skills[n]:
                        x[n, d, s, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{n}_{d}_{s}_{k}")
    
    # w[n, d] = 1 if nurse n works on day d
    w = {}
    for n in nurse_ids:
        for d in range(num_days):
            w[n, d] = model.addVar(vtype=GRB.BINARY, name=f"w_{n}_{d}")
    
    # y[n, d, s] = 1 if nurse n works shift s on day d
    y = {}
    for n in nurse_ids:
        for d in range(num_days):
            for s in shift_names:
                y[n, d, s] = model.addVar(vtype=GRB.BINARY, name=f"y_{n}_{d}_{s}")
    
    model.update()
    
    # Link x, y, w
    # w[n,d] = sum_s y[n,d,s]
    for n in nurse_ids:
        for d in range(num_days):
            model.addConstr(w[n, d] == gp.quicksum(y[n, d, s] for s in shift_names))
    
    # y[n,d,s] = sum_k x[n,d,s,k] for k in nurse_skills[n]
    for n in nurse_ids:
        for d in range(num_days):
            for s in shift_names:
                valid_skills = [k for k in skills if k in nurse_skills[n]]
                if valid_skills:
                    model.addConstr(y[n, d, s] == gp.quicksum(x[n, d, s, k] for k in valid_skills))
                else:
                    model.addConstr(y[n, d, s] == 0)
    
    # At most one shift per day per nurse
    for n in nurse_ids:
        for d in range(num_days):
            model.addConstr(gp.quicksum(y[n, d, s] for s in shift_names) <= 1)
    
    # Hard constraint: minimum demand
    for d in range(num_days):
        for s in shift_names:
            for k in skills:
                min_dem, opt_dem = demand_dict[d][s][k]
                if min_dem > 0:
                    eligible = [n for n in nurse_ids if k in nurse_skills[n]]
                    model.addConstr(
                        gp.quicksum(x[n, d, s, k] for n in eligible if (n, d, s, k) in x) >= min_dem
                    )
    
    # Forbidden shift successions (hard)
    for n in nurse_ids:
        for d in range(num_days - 1):
            for s1, s2 in forbidden_pairs:
                if s1 in shift_names and s2 in shift_names:
                    model.addConstr(y[n, d, s1] + y[n, d + 1, s2] <= 1)
    
    # === Soft constraint penalties ===
    obj_terms = []
    
    # S1: Insufficient staffing (optimal coverage shortfall)
    under_cover = {}
    for d in range(num_days):
        for s in shift_names:
            for k in skills:
                min_dem, opt_dem = demand_dict[d][s][k]
                if opt_dem > 0:
                    under_cover[d, s, k] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"uc_{d}_{s}_{k}")
                    eligible = [n for n in nurse_ids if k in nurse_skills[n]]
                    model.addConstr(
                        gp.quicksum(x[n, d, s, k] for n in eligible if (n, d, s, k) in x) + under_cover[d, s, k] >= opt_dem
                    )
                    obj_terms.append(weights['c_S1'] * under_cover[d, s, k])
    
    # S4: Preference violations
    for n in nurse_ids:
        for (pd, ps) in nurse_prefs[n]:
            if pd < num_days and ps in shift_names:
                obj_terms.append(weights['c_S4'] * y[n, pd, ps])
    
    # S5: Incomplete weekends
    # Weekend is Saturday (day 5+7*w) and Sunday (day 6+7*w)
    we_vars = {}
    for n in nurse_ids:
        for wk in range(num_weeks):
            sat = 5 + 7 * wk
            sun = 6 + 7 * wk
            if sat < num_days and sun < num_days:
                we_pen = model.addVar(vtype=GRB.BINARY, name=f"we_{n}_{wk}")
                we_vars[n, wk] = we_pen
                # Incomplete weekend: works exactly one of sat/sun
                # we_pen >= w[n,sat] - w[n,sun]
                # we_pen >= w[n,sun] - w[n,sat]
                model.addConstr(we_pen >= w[n, sat] - w[n, sun])
                model.addConstr(we_pen >= w[n, sun] - w[n, sat])
                obj_terms.append(weights['c_S5'] * we_pen)
    
    # S6: Total assignment deviation
    for n in nurse_ids:
        contract = nurse_contract[n]
        L_minus = contract['L_minus']
        L_plus = contract['L_plus']
        total_work = gp.quicksum(w[n, d] for d in range(num_days))
        
        over_total = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"over_total_{n}")
        under_total = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"under_total_{n}")
        model.addConstr(total_work - over_total <= L_plus)
        model.addConstr(total_work + under_total >= L_minus)
        obj_terms.append(weights['c_S6'] * over_total)
        obj_terms.append(weights['c_S6'] * under_total)
    
    # S7: Worked weekends excess
    for n in nurse_ids:
        contract = nurse_contract[n]
        WE_plus = contract['WE_plus']
        
        # we_work[n, wk] = 1 if nurse works at least one day in weekend wk
        we_work = {}
        for wk in range(num_weeks):
            sat = 5 + 7 * wk
            sun = 6 + 7 * wk
            if sat < num_days and sun < num_days:
                we_work[n, wk] = model.addVar(vtype=GRB.BINARY, name=f"wew_{n}_{wk}")
                model.addConstr(we_work[n, wk] >= w[n, sat])
                model.addConstr(we_work[n, wk] >= w[n, sun])
                model.addConstr(we_work[n, wk] <= w[n, sat] + w[n, sun])
        
        total_we = gp.quicksum(we_work[n, wk] for wk in range(num_weeks) if (n, wk) in we_work)
        over_we = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"over_we_{n}")
        model.addConstr(total_we - over_we <= WE_plus)
        obj_terms.append(weights['c_S7'] * over_we)
    
    # S2a: Consecutive worked days violations
    # We use a stretch-based approach with auxiliary variables
    # For each nurse, for each possible start of a work stretch, track violations
    # Simpler approach: penalize each violation day individually
    
    # Consecutive worked days: for each nurse, identify work stretches
    # We'll use the classic approach with binary indicators for stretch boundaries
    
    # For minimum consecutive worked days (CD_minus):
    # If a work stretch is shorter than CD_minus, penalize each missing day
    # A work stretch starts at day d if w[n,d]=1 and (d=0 or w[n,d-1]=0)
    # A work stretch ends at day d if w[n,d]=1 and (d=num_days-1 or w[n,d+1]=0)
    
    # For practical MIP formulation, we use: for stretches shorter than CD_minus,
    # penalize. For stretches longer than CD_plus, penalize.
    
    # Approach: For max consecutive worked days CD_plus:
    # For any window of CD_plus+1 consecutive days, at least one must be off
    for n in nurse_ids:
        contract = nurse_contract[n]
        CD_plus = contract['CD_plus']
        CD_minus = contract['CD_minus']
        
        # Max consecutive: any CD_plus+1 consecutive days must have a day off
        for d in range(num_days - CD_plus):
            window = range(d, d + CD_plus + 1)
            # sum w[n,d'] <= CD_plus for d' in window
            # Violation: excess = max(0, sum - CD_plus)
            # Actually for hard-ish approach: we penalize each day over
            pass  # We'll use a different formulation below
    
    # More efficient: use penalty variables for consecutive violations
    # For max consecutive worked days:
    # s2a_over[n, d] = 1 if there's a stretch of CD_plus+1 worked days ending at d
    # Penalize each such occurrence
    
    # Actually, let's use a cleaner formulation:
    # For consecutive worked days max violation:
    # For each window of (CD_plus + 1) consecutive days, if all are worked, that's a violation
    # Each extra day beyond CD_plus in a stretch contributes 1 penalty
    
    # Use: for each d, if days d-CD_plus through d are all worked, penalty of 1
    for n in nurse_ids:
        contract = nurse_contract[n]
        CD_plus = contract['CD_plus']
        CD_minus = contract['CD_minus']
        
        # Over consecutive worked days
        for d in range(CD_plus, num_days):
            window = range(d - CD_plus, d + 1)
            v = model.addVar(vtype=GRB.BINARY, name=f"s2a_over_{n}_{d}")
            # v >= sum(w[n,d'] for d' in window) - CD_plus
            model.addConstr(v >= gp.quicksum(w[n, dd] for dd in window) - CD_plus)
            obj_terms.append(weights['c_S2a'] * v)
        
        # Under consecutive worked days (stretch shorter than CD_minus)
        # A stretch of length L < CD_minus incurs (CD_minus - L) penalty
        # Detect: nurse works on day d but not on d-1 (or d=0), and doesn't work on d+j for j < CD_minus
        # For each possible short stretch of length L (1 to CD_minus-1):
        # start[n,d] = w[n,d] * (1 - w[n,d-1]) for d>0, or w[n,0] for d=0
        if CD_minus > 1:
            for d in range(num_days):
                for L in range(1, CD_minus):
                    if d + L - 1 < num_days:
                        # Stretch of exactly L days starting at d: 
                        # w[n,d]=1, ..., w[n,d+L-1]=1, w[n,d+L]=0 (or d+L=num_days)
                        # w[n,d-1]=0 (or d=0)
                        # Penalty: CD_minus - L
                        v = model.addVar(vtype=GRB.BINARY, name=f"s2a_under_{n}_{d}_{L}")
                        
                        terms = []
                        # All days d..d+L-1 must be worked
                        for dd in range(d, d + L):
                            terms.append(w[n, dd])
                        # Day before must be off (if exists)
                        if d > 0:
                            terms.append(1 - w[n, d - 1])
                        # Day after must be off (if exists)
                        if d + L < num_days:
                            terms.append(1 - w[n, d + L])
                        
                        num_terms = len(terms)
                        model.addConstr(v >= gp.quicksum(terms) - (num_terms - 1))
                        obj_terms.append(weights['c_S2a'] * (CD_minus - L) * v)
    
    # S2b: Consecutive same-shift violations
    for n in nurse_ids:
        for s in shift_names:
            cs_plus = CS_plus[s]
            cs_minus = CS_minus[s]
            
            # Max consecutive same shift
            for d in range(cs_plus, num_days):
                window = range(d - cs_plus, d + 1)
                v = model.addVar(vtype=GRB.BINARY, name=f"s2b_over_{n}_{s}_{d}")
                model.addConstr(v >= gp.quicksum(y[n, dd, s] for dd in window) - cs_plus)
                obj_terms.append(weights['c_S2b'] * v)
            
            # Min consecutive same shift
            if cs_minus > 1:
                for d in range(num_days):
                    for L in range(1, cs_minus):
                        if d + L - 1 < num_days:
                            v = model.addVar(vtype=GRB.BINARY, name=f"s2b_under_{n}_{s}_{d}_{L}")
                            terms = []
                            for dd in range(d, d + L):
                                terms.append(y[n, dd, s])
                            if d > 0:
                                terms.append(1 - y[n, d - 1, s])
                            if d + L < num_days:
                                terms.append(1 - y[n, d + L, s])
                            
                            num_terms = len(terms)
                            model.addConstr(v >= gp.quicksum(terms) - (num_terms - 1))
                            obj_terms.append(weights['c_S2b'] * (cs_minus - L) * v)
    
    # S3: Consecutive days off violations
    for n in nurse_ids:
        contract = nurse_contract[n]
        CR_plus = contract['CR_plus']
        CR_minus = contract['CR_minus']
        
        # r[n,d] = 1 - w[n,d] (rest)
        # Max consecutive days off
        for d in range(CR_plus, num_days):
            window = range(d - CR_plus, d + 1)
            v = model.addVar(vtype=GRB.BINARY, name=f"s3_over_{n}_{d}")
            # All days in window are off: sum (1-w) >= CR_plus+1, i.e., sum w <= 0
            # v >= 1 - sum w[n,dd] for dd in window  -- but this doesn't quite work
            # v >= (CR_plus+1) - sum w[n,dd] for dd in window - CR_plus
            # = 1 - sum w[n,dd]
            model.addConstr(v >= 1 - gp.quicksum(w[n, dd] for dd in window))
            obj_terms.append(weights['c_S3'] * v)
        
        # Min consecutive days off
        if CR_minus > 1:
            for d in range(num_days):
                for L in range(1, CR_minus):
                    if d + L - 1 < num_days:
                        v = model.addVar(vtype=GRB.BINARY, name=f"s3_under_{n}_{d}_{L}")
                        terms = []
                        # All days d..d+L-1 must be off: (1-w[n,dd])
                        for dd in range(d, d + L):
                            terms.append(1 - w[n, dd])
                        # Day before must be worked (if exists)
                        if d > 0:
                            terms.append(w[n, d - 1])
                        # Day after must be worked (if exists)
                        if d + L < num_days:
                            terms.append(w[n, d + L])
                        
                        num_terms = len(terms)
                        model.addConstr(v >= gp.quicksum(terms) - (num_terms - 1))
                        obj_terms.append(weights['c_S3'] * (CR_minus - L) * v)
    
    # Set objective
    model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
    
    # Callback for logging
    best_obj = [float('inf')]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)
    
    model.optimize(callback)
    
    # Extract solution
    schedule = []
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        for n in nurse_ids:
            assignments = []
            for d in range(num_days):
                for s in shift_names:
                    if y[n, d, s].X > 0.5:
                        for k in skills:
                            if k in nurse_skills[n] and (n, d, s, k) in x and x[n, d, s, k].X > 0.5:
                                assignments.append({
                                    "day": d,
                                    "shift": s,
                                    "skill": k
                                })
                                break
            schedule.append({
                "nurse_id": n,
                "nurse_name": nurse_names[n],
                "assignments": assignments
            })
        
        solution = {
            "objective_value": obj_val,
            "schedule": schedule
        }
    else:
        # No feasible solution found, return empty
        for n in nurse_ids:
            schedule.append({
                "nurse_id": n,
                "nurse_name": nurse_names[n],
                "assignments": []
            })
        solution = {
            "objective_value": float('inf'),
            "schedule": schedule
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