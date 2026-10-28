import json
import argparse
import time
import random
import sys

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    pass

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def get_heuristic_schedule(jobs_data, num_machines, iterations=1000):
    best_makespan = float('inf')
    best_x = {}
    best_z = {}
    best_s = {}

    for _ in range(iterations):
        job_ops = [0] * len(jobs_data)
        job_avail = [1] * len(jobs_data)
        machine_avail = [1] * num_machines
        
        x_dict = {}
        z_dict = {}
        s_dict = {}
        
        remaining_ops = sum(len(j['operations']) for j in jobs_data)
        while remaining_ops > 0:
            ready_jobs = [i for i, j in enumerate(jobs_data) if job_ops[i] < len(j['operations'])]
            i = random.choice(ready_jobs)
            j = job_ops[i]
            op = jobs_data[i]['operations'][j]
            eligible = op['eligible_machines']
            
            if random.random() < 0.8:
                best_m = -1
                best_c = float('inf')
                best_start = -1
                for m in eligible:
                    p = op['processing_times'][str(m)]
                    start = max(job_avail[i], machine_avail[m])
                    c = start + p
                    if c < best_c:
                        best_c = c
                        best_m = m
                        best_start = start
            else:
                best_m = random.choice(eligible)
                p = op['processing_times'][str(best_m)]
                best_start = max(job_avail[i], machine_avail[best_m])
                best_c = best_start + p

            m = best_m
            p = op['processing_times'][str(m)]
            start_t = best_start
            
            x_dict[(i, j, m)] = 1
            s_dict[(i, j, m, start_t)] = 1
            for t in range(start_t, start_t + p):
                z_dict[(i, j, m, t)] = 1
                
            job_avail[i] = start_t + p
            machine_avail[m] = start_t + p
            job_ops[i] += 1
            remaining_ops -= 1
            
        makespan = max(machine_avail)
        if makespan < best_makespan:
            best_makespan = makespan
            best_x = x_dict
            best_z = z_dict
            best_s = s_dict
            
    return best_makespan, best_x, best_z, best_s

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    
    args = parser.parse_args()
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    jobs = instance.get('jobs', [])
    num_machines = instance.get('num_machines', 0)
    is_preempt = instance.get('preemption_allowed', True)
    
    total_ops = sum(len(j['operations']) for j in jobs)
    if total_ops == 0:
        with open(args.solution_path, 'w') as f:
            json.dump({"objective_value": 0.0, "assignment": {}, "schedule": {}}, f)
        return

    # 1. Run heuristic to find an upper bound (UB) and a warm-start schedule
    best_makespan, best_x, best_z, best_s = get_heuristic_schedule(jobs, num_machines, iterations=1000)
    UB = best_makespan
    
    # 2. Precompute earliest start (ES) and latest completion (LC) bounds over all jobs
    ES = []
    LC = []
    for i, job in enumerate(jobs):
        job_es = []
        curr_es = 0
        for op in job['operations']:
            job_es.append(curr_es)
            curr_es += min(op['processing_times'][str(m)] for m in op['eligible_machines'])
        ES.append(job_es)
        
        job_lc = []
        num_ops = len(job['operations'])
        for j in range(num_ops):
            rem = 0
            for k in range(j + 1, num_ops):
                rem += min(job['operations'][k]['processing_times'][str(m)] for m in job['operations'][k]['eligible_machines'])
            job_lc.append(UB - rem)
        LC.append(job_lc)

    # 3. Model construction
    model = gp.Model("fjsp")
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)
    
    x_vars = {}
    z_vars_dict = {}
    z_exprs_dict = {}
    s_vars = {}
    
    for i, job in enumerate(jobs):
        for j, op in enumerate(job['operations']):
            for m in op['eligible_machines']:
                x_vars[i, j, m] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{m}")
                
            if is_preempt:
                for m in op['eligible_machines']:
                    for t in range(ES[i][j] + 1, LC[i][j]):
                        z_vars_dict[i, j, m, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}_{m}_{t}")
            else:
                for m in op['eligible_machines']:
                    p = op['processing_times'][str(m)]
                    for t in range(ES[i][j] + 1, LC[i][j] - p + 1):
                        s_vars[i, j, m, t] = model.addVar(vtype=GRB.BINARY, name=f"s_{i}_{j}_{m}_{t}")

    # Assignment logic formulation
    for i, job in enumerate(jobs):
        for j, op in enumerate(job['operations']):
            model.addConstr(gp.quicksum(x_vars[i, j, m] for m in op['eligible_machines']) == 1)
            
            if is_preempt:
                for m in op['eligible_machines']:
                    valid_z = [z_vars_dict[i, j, m, t] for t in range(ES[i][j] + 1, LC[i][j]) if (i, j, m, t) in z_vars_dict]
                    p = op['processing_times'][str(m)]
                    model.addConstr(gp.quicksum(valid_z) == p * x_vars[i, j, m])
                    for z_v in valid_z:
                        model.addConstr(z_v <= x_vars[i, j, m])
            else:
                for m in op['eligible_machines']:
                    p = op['processing_times'][str(m)]
                    valid_s = [s_vars[i, j, m, t] for t in range(ES[i][j] + 1, LC[i][j] - p + 1) if (i, j, m, t) in s_vars]
                    model.addConstr(gp.quicksum(valid_s) == x_vars[i, j, m])
                    
                    for t in range(ES[i][j] + 1, LC[i][j]):
                        tau_start = max(ES[i][j] + 1, t - p + 1)
                        tau_end = min(t, LC[i][j] - p)
                        v_s = [s_vars[i, j, m, tau] for tau in range(tau_start, tau_end + 1) if (i, j, m, tau) in s_vars]
                        if v_s:
                            z_exprs_dict[i, j, m, t] = gp.quicksum(v_s)

    # Machine capacity formulation
    target_z_dict = z_vars_dict if is_preempt else z_exprs_dict
    for m in range(num_machines):
        for t in range(1, UB):
            z_mt = []
            for i, job in enumerate(jobs):
                for j, op in enumerate(job['operations']):
                    if m in op['eligible_machines'] and (i, j, m, t) in target_z_dict:
                        z_mt.append(target_z_dict[i, j, m, t])
            if z_mt:
                model.addConstr(gp.quicksum(z_mt) <= 1)

    # Precedence logic formulation
    if is_preempt:
        for i, job in enumerate(jobs):
            for j in range(len(job['operations']) - 1):
                op_j1 = job['operations'][j+1]
                op_j = job['operations'][j]
                for t in range(ES[i][j+1] + 1, LC[i][j+1]):
                    z_j1_t = []
                    for m in op_j1['eligible_machines']:
                        if (i, j+1, m, t) in z_vars_dict:
                            z_j1_t.append(z_vars_dict[i, j+1, m, t])
                    if not z_j1_t: continue
                    
                    z_j_tau = []
                    for m in op_j['eligible_machines']:
                        for tau in range(t, LC[i][j]):
                            if (i, j, m, tau) in z_vars_dict:
                                z_j_tau.append(z_vars_dict[i, j, m, tau])
                    if z_j_tau:
                        M_val = max(op_j['processing_times'][str(m)] for m in op_j['eligible_machines'])
                        model.addConstr(gp.quicksum(z_j_tau) <= M_val * (1 - gp.quicksum(z_j1_t)))
    else:
        for i, job in enumerate(jobs):
            for j in range(len(job['operations']) - 1):
                op_j = job['operations'][j]
                op_j1 = job['operations'][j+1]
                end_expr = gp.LinExpr()
                for m in op_j['eligible_machines']:
                    p = op_j['processing_times'][str(m)]
                    for t in range(ES[i][j] + 1, LC[i][j] - p + 1):
                        if (i, j, m, t) in s_vars:
                            end_expr += (t + p) * s_vars[i, j, m, t]
                            
                start_j1_expr = gp.LinExpr()
                for m in op_j1['eligible_machines']:
                    p1 = op_j1['processing_times'][str(m)]
                    for t in range(ES[i][j+1] + 1, LC[i][j+1] - p1 + 1):
                        if (i, j+1, m, t) in s_vars:
                            start_j1_expr += t * s_vars[i, j+1, m, t]
                model.addConstr(end_expr <= start_j1_expr)

    # Objective formulation
    cmax = model.addVar(vtype=GRB.INTEGER, name="cmax")
    for i, job in enumerate(jobs):
        if not job['operations']: continue
        if is_preempt:
            for j, op in enumerate(job['operations']):
                for m in op['eligible_machines']:
                    for t in range(ES[i][j] + 1, LC[i][j]):
                        if (i, j, m, t) in z_vars_dict:
                            model.addConstr(cmax >= (t + 1) * z_vars_dict[i, j, m, t])
        else:
            j = len(job['operations']) - 1
            op = job['operations'][j]
            end_expr = gp.LinExpr()
            for m in op['eligible_machines']:
                p = op['processing_times'][str(m)]
                for t in range(ES[i][j] + 1, LC[i][j] - p + 1):
                    if (i, j, m, t) in s_vars:
                        end_expr += (t + p) * s_vars[i, j, m, t]
            model.addConstr(cmax >= end_expr)

    model.setObjective(cmax, GRB.MINIMIZE)

    # 4. Inject Warm Start
    for var_key, var in x_vars.items():
        var.Start = 1 if var_key in best_x else 0
        
    if is_preempt:
        for var_key, var in z_vars_dict.items():
            var.Start = 1 if var_key in best_z else 0
    else:
        for var_key, var in s_vars.items():
            var.Start = 1 if var_key in best_s else 0

    # 5. Optimize
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None
    if logger:
        logger.log(best_makespan) # Log the heuristic value just in case
        
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    build_time = time.time() - start_time
    remaining_limit = max(0.1, float(args.time_limit) - build_time)
    model.setParam('TimeLimit', remaining_limit)
    
    try:
        model.optimize(cb)
    except gp.GurobiError:
        pass

    # 6. Solution Output formatting
    assignment = {}
    schedule = {}
    
    if hasattr(model, 'SolCount') and model.SolCount > 0:
        for i, job in enumerate(jobs):
            for j, op in enumerate(job['operations']):
                key = f"({i},{j})"
                assigned_m = -1
                for m in op['eligible_machines']:
                    if x_vars[i, j, m].X > 0.5:
                        assigned_m = m
                        break
                assignment[key] = assigned_m
                
                sched = []
                if is_preempt:
                    for t in range(ES[i][j] + 1, LC[i][j]):
                        if (i, j, assigned_m, t) in z_vars_dict: 
                            if z_vars_dict[i, j, assigned_m, t].X > 0.5:
                                sched.append(t)
                else:
                    p = op['processing_times'][str(assigned_m)]
                    start_val = None
                    for t in range(ES[i][j] + 1, LC[i][j] - p + 1):
                        if (i, j, assigned_m, t) in s_vars:
                            if s_vars[i, j, assigned_m, t].X > 0.5:
                                start_val = t
                                break
                    if start_val is not None:
                        sched = list(range(start_val, start_val + p))
                schedule[key] = sched
        obj_val = model.ObjVal
    else:
        for i, job in enumerate(jobs):
            for j, op in enumerate(job['operations']):
                key = f"({i},{j})"
                assigned_m = -1
                for m in op['eligible_machines']:
                    if (i, j, m) in best_x:
                        assigned_m = m
                        break
                assignment[key] = assigned_m
                
                sched = []
                if is_preempt:
                    for t in range(1, int(UB + 2)):
                        if (i, j, assigned_m, t) in best_z:
                            sched.append(t)
                else:
                    p = op['processing_times'][str(assigned_m)]
                    start_val = None
                    for t in range(1, int(UB + 2)):
                        if (i, j, assigned_m, t) in best_s:
                            start_val = t
                            break
                    if start_val is not None:
                        sched = list(range(start_val, start_val + p))
                schedule[key] = sched
        obj_val = best_makespan

    output = {
        "objective_value": float(obj_val),
        "assignment": assignment,
        "schedule": schedule
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f)

if __name__ == "__main__":
    solve()