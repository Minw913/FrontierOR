import json
import argparse
import time
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    start_time = time.time()
    
    num_patients = instance['num_patients']
    num_hospitals = instance['num_hospitals']
    num_days = instance['num_days']
    num_ors = instance['num_ors_per_hospital']
    num_scenarios = instance['num_scenarios']
    
    # Estimate model size to decide approach
    estimated_z_vars = num_scenarios * num_hospitals * num_days * num_patients * num_ors
    
    solution = None
    
    if estimated_z_vars <= 300000:
        # Try direct MIP
        solution = solve_direct_mip(instance, args.time_limit, logger, start_time)
    
    if solution is None:
        # Try Benders decomposition
        solution = solve_benders(instance, args.time_limit, logger, start_time)
    
    if solution is None:
        # Fallback to greedy
        solution = greedy_solution(instance)
        if logger:
            logger.log_solution(solution['objective_value'], solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


def parse_instance(instance):
    """Parse instance into convenient data structures."""
    num_patients = instance['num_patients']
    num_hospitals = instance['num_hospitals']
    num_days = instance['num_days']
    num_ors = instance['num_ors_per_hospital']
    num_scenarios = instance['num_scenarios']
    
    patients = instance['patients']
    hospitals = instance['hospitals']
    scenarios = instance['scenarios']
    
    # Hospital-day data
    B = {}  # B[h,d] = time limit
    F = {}  # F[h,d] = OR opening cost
    G = {}  # G[h,d] = suite opening cost
    hospital_ids = []
    for h_data in hospitals:
        h = h_data['hospital_id']
        hospital_ids.append(h)
        for day_data in h_data['days']:
            d = day_data['day_id']
            B[h, d] = day_data['B_hd']
            F[h, d] = day_data['F_hd']
            G[h, d] = day_data['G_hd']
    
    # Patient data
    patient_ids = []
    is_mandatory = {}
    c_sched = {}  # c_sched[p, d]
    c_unsched = {}
    c_cancel = {}
    for p_data in patients:
        p = p_data['patient_id']
        patient_ids.append(p)
        is_mandatory[p] = p_data['is_mandatory']
        c_unsched[p] = p_data['c_unsched']
        c_cancel[p] = p_data['c_cancel']
        for day_label_str, benefit in p_data['c_sched_per_day'].items():
            d = int(day_label_str) - 1  # Convert 1-based label to 0-based day
            c_sched[p, d] = benefit
    
    # Scenario durations
    dur = {}  # dur[s, p]
    scenario_ids = []
    for s_data in scenarios:
        s = s_data['scenario_id']
        scenario_ids.append(s)
        durations = s_data['surgery_durations_minutes']
        for i, p in enumerate(patient_ids):
            dur[s, p] = durations[i]
    
    return {
        'num_patients': num_patients,
        'num_hospitals': num_hospitals,
        'num_days': num_days,
        'num_ors': num_ors,
        'num_scenarios': num_scenarios,
        'hospital_ids': hospital_ids,
        'patient_ids': patient_ids,
        'scenario_ids': scenario_ids,
        'B': B, 'F': F, 'G': G,
        'is_mandatory': is_mandatory,
        'c_sched': c_sched,
        'c_unsched': c_unsched,
        'c_cancel': c_cancel,
        'dur': dur
    }


def solve_direct_mip(instance, time_limit, logger, start_time):
    """Solve using direct MIP formulation with Gurobi."""
    try:
        from gurobipy import Model, GRB, quicksum
    except ImportError:
        return None
    
    data = parse_instance(instance)
    P = data['patient_ids']
    H = data['hospital_ids']
    D = range(data['num_days'])
    O = range(data['num_ors'])
    S = data['scenario_ids']
    
    B = data['B']
    F = data['F']
    G = data['G']
    is_mandatory = data['is_mandatory']
    c_sched = data['c_sched']
    c_unsched = data['c_unsched']
    c_cancel = data['c_cancel']
    dur = data['dur']
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining <= 5:
        return None
    
    model = Model("surgery_scheduling")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, remaining - 2)
    model.Params.OutputFlag = 0
    
    # First-stage variables
    u = {}
    for h in H:
        for d in D:
            u[h, d] = model.addVar(vtype=GRB.BINARY, name=f"u_{h}_{d}")
    
    y = {}
    for h in H:
        for d in D:
            for o in O:
                y[h, d, o] = model.addVar(vtype=GRB.BINARY, name=f"y_{h}_{d}_{o}")
    
    x = {}
    for h in H:
        for d in D:
            for p in P:
                for o in O:
                    x[h, d, p, o] = model.addVar(vtype=GRB.BINARY, name=f"x_{h}_{d}_{p}_{o}")
    
    w = {}
    for p in P:
        if not is_mandatory[p]:
            w[p] = model.addVar(vtype=GRB.BINARY, name=f"w_{p}")
    
    # Second-stage variables
    q = {}
    for s in S:
        for p in P:
            q[s, p] = model.addVar(vtype=GRB.BINARY, name=f"q_{s}_{p}")
    
    # Linearization variables: z[s,h,d,p,o] = x[h,d,p,o] * q[s,p]
    z = {}
    for s in S:
        for h in H:
            for d in D:
                for p in P:
                    for o in O:
                        z[s, h, d, p, o] = model.addVar(vtype=GRB.BINARY, name=f"z_{s}_{h}_{d}_{p}_{o}")
    
    model.update()
    
    # Constraints
    # 1. Suite-OR linking
    for h in H:
        for d in D:
            for o in O:
                model.addConstr(y[h, d, o] <= u[h, d])
    
    # 2. Patient-OR linking
    for h in H:
        for d in D:
            for p in P:
                for o in O:
                    model.addConstr(x[h, d, p, o] <= y[h, d, o])
    
    # 3. Assignment constraints
    for p in P:
        if is_mandatory[p]:
            model.addConstr(quicksum(x[h, d, p, o] for h in H for d in D for o in O) == 1)
        else:
            model.addConstr(quicksum(x[h, d, p, o] for h in H for d in D for o in O) + w[p] == 1)
    
    # 4. Linearization: z = x * q
    for s in S:
        for h in H:
            for d in D:
                for p in P:
                    for o in O:
                        model.addConstr(z[s, h, d, p, o] <= x[h, d, p, o])
                        model.addConstr(z[s, h, d, p, o] <= q[s, p])
                        model.addConstr(z[s, h, d, p, o] >= x[h, d, p, o] + q[s, p] - 1)
    
    # 5. Time constraints
    for s in S:
        for h in H:
            for d in D:
                for o in O:
                    model.addConstr(
                        quicksum(dur[s, p] * (x[h, d, p, o] - z[s, h, d, p, o]) for p in P) <= B[h, d]
                    )
    
    # 6. q only for scheduled patients
    for s in S:
        for p in P:
            if not is_mandatory[p]:
                model.addConstr(q[s, p] <= 1 - w[p])
    
    # Objective
    obj = 0
    for h in H:
        for d in D:
            obj += G[h, d] * u[h, d]
    for h in H:
        for d in D:
            for o in O:
                obj += F[h, d] * y[h, d, o]
    for p in P:
        for h in H:
            for d in D:
                for o in O:
                    obj += c_sched.get((p, d), 0) * x[h, d, p, o]
    for p in P:
        if not is_mandatory[p]:
            obj += c_unsched[p] * w[p]
    num_scen = len(S)
    for s in S:
        for p in P:
            obj += (1.0 / num_scen) * c_cancel[p] * q[s, p]
    
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    
    # Set greedy solution as warm start
    greedy_sol = greedy_solution(instance)
    if greedy_sol is not None:
        try:
            for key, val in greedy_sol['u'].items():
                parts = key.split(',')
                h, d = int(parts[0]), int(parts[1])
                if (h, d) in u:
                    u[h, d].Start = val
            for key, val in greedy_sol['y'].items():
                parts = key.split(',')
                h, d, o = int(parts[0]), int(parts[1]), int(parts[2])
                if (h, d, o) in y:
                    y[h, d, o].Start = val
            for key, val in greedy_sol['x'].items():
                parts = key.split(',')
                h, d, p, o = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                if (h, d, p, o) in x:
                    x[h, d, p, o].Start = val
            for key, val in greedy_sol['w'].items():
                p = int(key)
                if p in w:
                    w[p].Start = val
        except:
            pass
    
    model.optimize()
    
    if model.SolCount > 0:
        return extract_solution(model, u, y, x, w, q, data, instance)
    
    return None


def extract_solution(model, u, y, x, w, q, data, instance):
    """Extract solution from Gurobi model."""
    H = data['hospital_ids']
    D = range(data['num_days'])
    O = range(data['num_ors'])
    P = data['patient_ids']
    S = data['scenario_ids']
    is_mandatory = data['is_mandatory']
    
    sol_u = {}
    for h in H:
        for d in D:
            val = int(round(u[h, d].X))
            sol_u[f"{h},{d}"] = val
    
    sol_y = {}
    for h in H:
        for d in D:
            for o in O:
                val = int(round(y[h, d, o].X))
                sol_y[f"{h},{d},{o}"] = val
    
    sol_x = {}
    for h in H:
        for d in D:
            for p in P:
                for o in O:
                    val = int(round(x[h, d, p, o].X))
                    if val == 1:
                        sol_x[f"{h},{d},{p},{o}"] = 1
    
    sol_w = {}
    for p in P:
        if not is_mandatory[p]:
            sol_w[str(p)] = int(round(w[p].X))
    
    # Compute objective
    obj = model.ObjVal
    
    solution = {
        "objective_value": obj,
        "u": sol_u,
        "y": sol_y,
        "x": sol_x,
        "w": sol_w
    }
    return solution


def solve_benders(instance, time_limit, logger, start_time):
    """Solve using Benders decomposition."""
    try:
        from gurobipy import Model, GRB, quicksum
    except ImportError:
        return None
    
    data = parse_instance(instance)
    P = data['patient_ids']
    H = data['hospital_ids']
    D = range(data['num_days'])
    O = range(data['num_ors'])
    S = data['scenario_ids']
    
    B = data['B']
    F = data['F']
    G = data['G']
    is_mandatory = data['is_mandatory']
    c_sched = data['c_sched']
    c_unsched = data['c_unsched']
    c_cancel = data['c_cancel']
    dur = data['dur']
    num_scen = len(S)
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining <= 5:
        return None
    
    # Master problem
    master = Model("master")
    master.Params.Seed = 0
    master.Params.MIPGap = 1e-4
    master.Params.NumericFocus = 0
    master.Params.Threads = 1
    master.Params.OutputFlag = 0
    
    u = {}
    for h in H:
        for d in D:
            u[h, d] = master.addVar(vtype=GRB.BINARY, name=f"u_{h}_{d}")
    
    y = {}
    for h in H:
        for d in D:
            for o in O:
                y[h, d, o] = master.addVar(vtype=GRB.BINARY, name=f"y_{h}_{d}_{o}")
    
    x = {}
    for h in H:
        for d in D:
            for p in P:
                for o in O:
                    x[h, d, p, o] = master.addVar(vtype=GRB.BINARY, name=f"x_{h}_{d}_{p}_{o}")
    
    w = {}
    for p in P:
        if not is_mandatory[p]:
            w[p] = master.addVar(vtype=GRB.BINARY, name=f"w_{p}")
    
    theta = master.addVar(lb=0, name="theta")
    
    master.update()
    
    # Constraints
    for h in H:
        for d in D:
            for o in O:
                master.addConstr(y[h, d, o] <= u[h, d])
    
    for h in H:
        for d in D:
            for p in P:
                for o in O:
                    master.addConstr(x[h, d, p, o] <= y[h, d, o])
    
    for p in P:
        if is_mandatory[p]:
            master.addConstr(quicksum(x[h, d, p, o] for h in H for d in D for o in O) == 1)
        else:
            master.addConstr(quicksum(x[h, d, p, o] for h in H for d in D for o in O) + w[p] == 1)
    
    # Objective
    obj = theta
    for h in H:
        for d in D:
            obj += G[h, d] * u[h, d]
    for h in H:
        for d in D:
            for o in O:
                obj += F[h, d] * y[h, d, o]
    for p in P:
        for h in H:
            for d in D:
                for o in O:
                    obj += c_sched.get((p, d), 0) * x[h, d, p, o]
    for p in P:
        if not is_mandatory[p]:
            obj += c_unsched[p] * w[p]
    
    master.setObjective(obj, GRB.MINIMIZE)
    master.update()
    
    # Set warm start from greedy
    greedy_sol = greedy_solution(instance)
    if greedy_sol is not None:
        try:
            for key, val in greedy_sol['u'].items():
                parts = key.split(',')
                h, d = int(parts[0]), int(parts[1])
                if (h, d) in u:
                    u[h, d].Start = val
            for key, val in greedy_sol['y'].items():
                parts = key.split(',')
                h, d, o = int(parts[0]), int(parts[1]), int(parts[2])
                if (h, d, o) in y:
                    y[h, d, o].Start = val
            for key, val in greedy_sol['x'].items():
                parts = key.split(',')
                h, d, p, o = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                if (h, d, p, o) in x:
                    x[h, d, p, o].Start = val
            for key, val in greedy_sol['w'].items():
                p = int(key)
                if p in w:
                    w[p].Start = val
        except:
            pass
    
    best_solution = None
    best_obj = float('inf')
    
    max_iterations = 50
    for iteration in range(max_iterations):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining <= 5:
            break
        
        master.Params.TimeLimit = max(1, min(remaining - 3, 60))
        master.optimize()
        
        if master.SolCount == 0:
            break
        
        # Extract first-stage solution
        x_val = {}
        for h in H:
            for d in D:
                for p in P:
                    for o in O:
                        x_val[h, d, p, o] = round(x[h, d, p, o].X)
        
        # Solve subproblems for each scenario
        total_second_stage_cost = 0
        cuts_to_add = []
        
        for s in S:
            scenario_cost = 0
            for h in H:
                for d in D:
                    for o in O:
                        # Get assigned patients
                        assigned = [(p, dur[s, p]) for p in P if x_val[h, d, p, o] == 1]
                        total_time = sum(d for _, d in assigned)
                        
                        if total_time > B[h, d]:
                            # Need cancellation - solve knapsack
                            excess = total_time - B[h, d]
                            cancel_cost = solve_knapsack_cost(assigned, excess, c_cancel)
                            scenario_cost += cancel_cost
                            
                            # Generate Benders cut
                            # LP dual: lambda = marginal cost per unit of excess
                            lam = compute_lambda(assigned, excess, c_cancel)
                            if lam > 0:
                                # Cut: theta >= (1/S) * lam * (sum_p dur[s,p]*x[h,d,p,o] - B[h,d]) + const
                                rhs_const = (1.0/num_scen) * lam * (-B[h, d])
                                lhs_terms = [(1.0/num_scen) * lam * dur[s, p] for p in P]
                                cuts_to_add.append((rhs_const, lhs_terms, h, d, o, s))
            
            total_second_stage_cost += scenario_cost
        
        expected_cancel_cost = total_second_stage_cost / num_scen
        
        # Compute full objective for this solution
        first_stage_cost = 0
        for h in H:
            for d in D:
                first_stage_cost += G[h, d] * round(u[h, d].X)
        for h in H:
            for d in D:
                for o in O:
                    first_stage_cost += F[h, d] * round(y[h, d, o].X)
        for p in P:
            for h in H:
                for d in D:
                    for o in O:
                        if x_val[h, d, p, o] == 1:
                            first_stage_cost += c_sched.get((p, d), 0)
        for p in P:
            if not is_mandatory[p] and p in w:
                first_stage_cost += c_unsched[p] * round(w[p].X)
        
        full_obj = first_stage_cost + expected_cancel_cost
        
        if full_obj < best_obj:
            best_obj = full_obj
            best_solution = build_solution_from_values(
                u, y, x_val, w, data, instance, full_obj
            )
            if logger:
                logger.log_solution(full_obj, best_solution)
        
        # Check if theta is a good approximation
        theta_val = theta.X
        if expected_cancel_cost <= theta_val + 1e-6:
            # Converged
            break
        
        # Add Benders cuts
        if len(cuts_to_add) == 0:
            # No cuts to add but theta is too low - add a simple cut
            master.addConstr(theta >= expected_cancel_cost)
        else:
            # Add aggregated cut
            # theta >= sum of (1/S) * lam * (sum_p dur * x - B) for violated ORs
            # Aggregate into a single cut for efficiency
            agg_rhs = 0
            agg_coeffs = {}
            for rhs_const, lhs_terms, h, d, o, s in cuts_to_add:
                agg_rhs += rhs_const
                for i, p in enumerate(P):
                    coeff = lhs_terms[i]
                    if coeff != 0:
                        agg_coeffs[h, d, p, o] = agg_coeffs.get((h, d, p, o), 0) + coeff
            
            if agg_coeffs:
                cut_expr = quicksum(agg_coeffs[h, d, p, o] * x[h, d, p, o] 
                                   for (h, d, p, o) in agg_coeffs)
                master.addConstr(theta >= agg_rhs + cut_expr)
        
        master.update()
    
    return best_solution


def solve_knapsack_cost(assigned, excess, c_cancel):
    """Solve minimum cost knapsack to cover excess time."""
    # Sort by cost per minute (ascending) - cancel cheapest per minute first
    items = sorted(assigned, key=lambda item: c_cancel[item[0]] / max(item[1], 0.001))
    
    total_cancelled = 0
    total_cost = 0
    for p, d in items:
        if total_cancelled >= excess:
            break
        total_cancelled += d
        total_cost += c_cancel[p]
    
    return total_cost


def compute_lambda(assigned, excess, c_cancel):
    """Compute LP dual variable (marginal cost per unit time)."""
    if not assigned or excess <= 0:
        return 0
    
    # Sort by cost per minute
    items = sorted(assigned, key=lambda item: c_cancel[item[0]] / max(item[1], 0.001))
    
    total_cancelled = 0
    for p, d in items:
        total_cancelled += d
        if total_cancelled >= excess:
            return c_cancel[p] / max(d, 0.001)
    
    # All items cancelled, use last item's ratio
    if items:
        return c_cancel[items[-1][0]] / max(items[-1][1], 0.001)
    return 0


def build_solution_from_values(u, y, x_val, w, data, instance, obj_val):
    """Build solution dictionary from variable values."""
    H = data['hospital_ids']
    D = range(data['num_days'])
    O = range(data['num_ors'])
    P = data['patient_ids']
    is_mandatory = data['is_mandatory']
    
    sol_u = {}
    for h in H:
        for d in D:
            sol_u[f"{h},{d}"] = int(round(u[h, d].X))
    
    sol_y = {}
    for h in H:
        for d in D:
            for o in O:
                sol_y[f"{h},{d},{o}"] = int(round(y[h, d, o].X))
    
    sol_x = {}
    for h in H:
        for d in D:
            for p in P:
                for o in O:
                    if x_val[h, d, p, o] == 1:
                        sol_x[f"{h},{d},{p},{o}"] = 1
    
    sol_w = {}
    for p in P:
        if not is_mandatory[p]:
            sol_w[str(p)] = int(round(w[p].X))
    
    return {
        "objective_value": obj_val,
        "u": sol_u,
        "y": sol_y,
        "x": sol_x,
        "w": sol_w
    }


def greedy_solution(instance):
    """Construct a greedy feasible solution."""
    data = parse_instance(instance)
    P = data['patient_ids']
    H = data['hospital_ids']
    D = list(range(data['num_days']))
    O = list(range(data['num_ors']))
    S = data['scenario_ids']
    
    B = data['B']
    F = data['F']
    G = data['G']
    is_mandatory = data['is_mandatory']
    c_sched = data['c_sched']
    c_unsched = data['c_unsched']
    c_cancel = data['c_cancel']
    dur = data['dur']
    num_scen = len(S)
    
    # Compute expected durations
    expected_dur = {}
    for p in P:
        expected_dur[p] = sum(dur[s, p] for s in S) / num_scen
    
    # Sort patients: mandatory first, then by health score (omega_p) descending
    patients_sorted = sorted(P, key=lambda p: (
        -int(is_mandatory[p]),
        -instance['patients'][P.index(p)]['omega_p'] if P.index(p) < len(instance['patients']) else 0
    ))
    
    # Track capacity usage per (h, d, o)
    capacity_used = {(h, d, o): 0 for h in H for d in D for o in O}
    
    # Track assignments
    assignment = {}  # p -> (h, d, o)
    postponed = set()
    
    # Track which suites and ORs are opened
    suite_open = set()
    or_open = set()
    
    for p in patients_sorted:
        best_assignment = None
        best_cost = float('inf')
        
        for h in H:
            for d in D:
                for o in O:
                    # Check capacity with expected duration
                    if capacity_used[h, d, o] + expected_dur[p] <= B[h, d]:
                        # Compute cost of this assignment
                        cost = -c_sched.get((p, d), 0)  # Negative benefit = cost reduction
                        # Add marginal opening cost if needed
                        if (h, d) not in suite_open:
                            cost += G[h, d]
                        if (h, d, o) not in or_open:
                            cost += F[h, d]
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_assignment = (h, d, o)
        
        if best_assignment is not None:
            h, d, o = best_assignment
            assignment[p] = best_assignment
            capacity_used[h, d, o] += expected_dur[p]
            suite_open.add((h, d))
            or_open.add((h, d, o))
        elif not is_mandatory[p]:
            postponed.add(p)
        else:
            # Mandatory patient - must assign, try harder (allow overcapacity)
            best_assignment = None
            best_cost = float('inf')
            for h in H:
                for d in D:
                    for o in O:
                        cost = -c_sched.get((p, d), 0)
                        if (h, d) not in suite_open:
                            cost += G[h, d]
                        if (h, d, o) not in or_open:
                            cost += F[h, d]
                        if cost < best_cost:
                            best_cost = cost
                            best_assignment = (h, d, o)
            if best_assignment:
                h, d, o = best_assignment
                assignment[p] = best_assignment
                capacity_used[h, d, o] += expected_dur[p]
                suite_open.add((h, d))
                or_open.add((h, d, o))
    
    # Compute expected cancellation cost
    expected_cancel_cost = 0
    for s in S:
        for h in H:
            for d in D:
                for o in O:
                    assigned_patients = [p for p in P if p in assignment and assignment[p] == (h, d, o)]
                    total_time = sum(dur[s, p] for p in assigned_patients)
                    if total_time > B[h, d]:
                        excess = total_time - B[h, d]
                        # Greedy cancellation: cancel cheapest per minute first
                        items = sorted(assigned_patients, 
                                      key=lambda p: c_cancel[p] / max(dur[s, p], 0.001))
                        cancelled_time = 0
                        for p in items:
                            if cancelled_time >= excess:
                                break
                            cancelled_time += dur[s, p]
                            expected_cancel_cost += c_cancel[p]
    
    expected_cancel_cost /= num_scen
    
    # Compute objective
    obj = 0
    for h, d in suite_open:
        obj += G[h, d]
    for h, d, o in or_open:
        obj += F[h, d]
    for p, (h, d, o) in assignment.items():
        obj += c_sched.get((p, d), 0)
    for p in postponed:
        obj += c_unsched[p]
    obj += expected_cancel_cost
    
    # Build solution
    sol_u = {}
    for h in H:
        for d in D:
            sol_u[f"{h},{d}"] = 1 if (h, d) in suite_open else 0
    
    sol_y = {}
    for h in H:
        for d in D:
            for o in O:
                sol_y[f"{h},{d},{o}"] = 1 if (h, d, o) in or_open else 0
    
    sol_x = {}
    for p, (h, d, o) in assignment.items():
        sol_x[f"{h},{d},{p},{o}"] = 1
    
    sol_w = {}
    for p in P:
        if not is_mandatory[p]:
            sol_w[str(p)] = 1 if p in postponed else 0
    
    return {
        "objective_value": obj,
        "u": sol_u,
        "y": sol_y,
        "x": sol_x,
        "w": sol_w
    }


if __name__ == '__main__':
    main()