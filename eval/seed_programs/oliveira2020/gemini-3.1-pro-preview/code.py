import argparse
import json
import math
import sys
import time
import random

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def run_heuristics(jobs, m, H, end_time):
    best_schedule = None
    best_obj = float('inf')
    
    edd_sorted = sorted(jobs, key=lambda j: (j['d'], j['p']))
    wspt_sorted = sorted(jobs, key=lambda j: (-j['w']/max(1, j['p']), j['d']))
    
    strategies = [('WSPT', 0), ('EDD', 0)]
    for K in [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        strategies.append(('ATC', K))
        
    for strat, K in strategies:
        if time.time() > end_time: 
            break
            
        avail = [0] * m
        sched = []
        obj = 0
        
        if strat == 'EDD':
            for best_j in edd_sorted:
                t = min(avail)
                idx = avail.index(t)
                
                start_time = t
                comp_time = t + best_j['p']
                avail[idx] = comp_time
                if comp_time > best_j['d']:
                    obj += best_j['w'] * (comp_time - best_j['d'])
                sched.append({
                    'job': best_j['id'],
                    'start_time': start_time,
                    'completion_time': comp_time
                })
        elif strat == 'WSPT':
            for best_j in wspt_sorted:
                t = min(avail)
                idx = avail.index(t)
                        
                start_time = t
                comp_time = t + best_j['p']
                avail[idx] = comp_time
                if comp_time > best_j['d']:
                    obj += best_j['w'] * (comp_time - best_j['d'])
                sched.append({
                    'job': best_j['id'],
                    'start_time': start_time,
                    'completion_time': comp_time
                })
        elif strat == 'ATC':
            U = jobs[:]
            total_p = sum(j['p'] for j in U)
            
            abort = False
            while U:
                if time.time() > end_time:
                    abort = True
                    break
                    
                t = min(avail)
                idx = avail.index(t)
                
                p_avg = max(1e-5, total_p / len(U))
                
                best_j = None
                best_score = -1.0
                best_idx = -1
                
                for i, j in enumerate(U):
                    diff = j['d'] - t - j['p']
                    if diff <= 0:
                        score = j['w'] / max(1, j['p'])
                    else:
                        exponent = -diff / (K * p_avg)
                        if exponent < -700:
                            score = 0.0
                        else:
                            score = (j['w'] / max(1, j['p'])) * math.exp(exponent)
                        
                    if score > best_score:
                        best_score = score
                        best_j = j
                        best_idx = i
                        
                start_time = t
                comp_time = t + best_j['p']
                avail[idx] = comp_time
                if comp_time > best_j['d']:
                    obj += best_j['w'] * (comp_time - best_j['d'])
                
                sched.append({
                    'job': best_j['id'],
                    'start_time': start_time,
                    'completion_time': comp_time
                })
                
                U[best_idx], U[-1] = U[-1], U[best_idx]
                U.pop()
                total_p -= best_j['p']
                
            if abort:
                continue
                
        if obj < best_obj:
            best_obj = obj
            best_schedule = sched
            
    return best_obj, best_schedule

def get_full_schedule(sequence, jobs_dict, m):
    avail = [0] * m
    sched = []
    for j_id in sequence:
        t = min(avail)
        idx = avail.index(t)
        j = jobs_dict[j_id]
        comp_time = t + j['p']
        avail[idx] = comp_time
        sched.append({
            'job': j_id,
            'start_time': t,
            'completion_time': comp_time
        })
    return sched

def local_search(jobs_dict, initial_sched, m, end_time, logger):
    sorted_sched = sorted(initial_sched, key=lambda x: x['start_time'])
    seq = [s['job'] for s in sorted_sched]
    
    def evaluate(sequence):
        avail = [0] * m
        obj = 0
        for j_id in sequence:
            t = min(avail)
            idx = avail.index(t)
            
            j = jobs_dict[j_id]
            comp_time = t + j['p']
            avail[idx] = comp_time
            if comp_time > j['d']:
                obj += j['w'] * (comp_time - j['d'])
        return obj

    best_seq = seq[:]
    best_obj = evaluate(best_seq)
    
    curr_seq = best_seq[:]
    curr_obj = best_obj
    
    n = len(seq)
    if n < 2: 
        return best_obj, initial_sched
    
    T = best_obj * 0.01 if best_obj > 0 else 10.0
    alpha = 0.99
    
    iters = 0
    while time.time() < end_time:
        if random.random() < 0.5:
            i, j = random.sample(range(n), 2)
            curr_seq[i], curr_seq[j] = curr_seq[j], curr_seq[i]
            is_swap = True
        else:
            i, j = random.sample(range(n), 2)
            job_i = curr_seq.pop(i)
            curr_seq.insert(j, job_i)
            is_swap = False
            
        new_obj = evaluate(curr_seq)
        
        if new_obj < best_obj:
            best_obj = new_obj
            best_seq = curr_seq[:]
            curr_obj = new_obj
            if logger:
                full_sched = get_full_schedule(best_seq, jobs_dict, m)
                true_obj, formatted_sched = format_schedule(full_sched, jobs_dict, m)
                logger.log_solution(true_obj, formatted_sched)
        else:
            diff = new_obj - curr_obj
            if diff < 0:
                curr_obj = new_obj
            else:
                if T > 1e-5:
                    exponent = -diff / T
                    if exponent < -700:
                        prob = 0.0
                    else:
                        prob = math.exp(exponent)
                        
                    if random.random() < prob:
                        curr_obj = new_obj
                    else:
                        if is_swap:
                            curr_seq[i], curr_seq[j] = curr_seq[j], curr_seq[i]
                        else:
                            job_i = curr_seq.pop(j)
                            curr_seq.insert(i, job_i)
                else:
                    if is_swap:
                        curr_seq[i], curr_seq[j] = curr_seq[j], curr_seq[i]
                    else:
                        job_i = curr_seq.pop(j)
                        curr_seq.insert(i, job_i)
                        
        iters += 1
        if iters % 100 == 0:
            T *= alpha
            
    final_sched = get_full_schedule(best_seq, jobs_dict, m)
    return best_obj, final_sched

def format_schedule(sched, jobs_dict, m):
    sorted_sched = sorted(sched, key=lambda x: x['start_time'])
    machine_avail = [0] * m
    machine_last = [0] * m
    
    full_schedule = []
    obj = 0.0
    for s in sorted_sched:
        j_id = s['job']
        p = jobs_dict[j_id]['p']
        st = int(round(s['start_time']))
        
        assigned_m = -1
        for k in range(m):
            if machine_avail[k] <= st:
                assigned_m = k
                break
        
        if assigned_m == -1:
            t_min = min(machine_avail)
            assigned_m = machine_avail.index(t_min)
            st = max(st, machine_avail[assigned_m])
            
        ct = st + p
        prev = machine_last[assigned_m]
        w = jobs_dict[j_id]['w']
        d = jobs_dict[j_id]['d']
        wt = float(w * max(0, ct - d))
        
        full_schedule.append({
            "machine_id": int(assigned_m),
            "prev_job": int(prev),
            "job": int(j_id),
            "start_time": int(st),
            "completion_time": int(ct),
            "weighted_tardiness": wt
        })
        
        machine_avail[assigned_m] = ct
        machine_last[assigned_m] = j_id
        obj += wt
        
    return obj, full_schedule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_cpu = time.time()
    
    logger = None
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="minimize")
        
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    H = instance["time_horizon"]
    n = instance["parameters"]["n"]
    m = instance["parameters"]["m"]
    
    jobs_list = []
    jobs_dict = {}
    for j_data in instance["jobs"]:
        j = {
            'id': j_data['id'],
            'p': j_data['processing_time'],
            'w': j_data['weight'],
            'd': j_data['due_date']
        }
        jobs_list.append(j)
        jobs_dict[j['id']] = j
        
    heur_end_time = start_cpu + args.time_limit * 0.1
    best_heuristic_obj, best_heuristic_sched_raw = run_heuristics(jobs_list, m, H, heur_end_time)
    heuristic_obj, heuristic_solution = format_schedule(best_heuristic_sched_raw, jobs_dict, m)
    
    if logger:
        logger.log_solution(heuristic_obj, heuristic_solution)
        
    max_time_indexed_vars = 100_000
    max_sequence_vars = 10_000

    formulation_type = None
    if n * H <= max_time_indexed_vars:
        formulation_type = "time_indexed"
    elif n * n <= max_sequence_vars:
        formulation_type = "sequence"
        
    if formulation_type is None:
        ls_end_time = start_cpu + args.time_limit - 0.5
    else:
        ls_end_time = start_cpu + args.time_limit * 0.2
        
    ls_obj, ls_sched_raw = local_search(jobs_dict, best_heuristic_sched_raw, m, ls_end_time, logger)
    
    if ls_obj < heuristic_obj:
        best_initial_sched_raw = ls_sched_raw
        final_obj, final_sched = format_schedule(ls_sched_raw, jobs_dict, m)
        if logger:
            logger.log_solution(final_obj, final_sched)
    else:
        best_initial_sched_raw = best_heuristic_sched_raw
        final_obj, final_sched = heuristic_obj, heuristic_solution

    # Try solving with Gurobi if available and instance is not too large
    if GUROBI_AVAILABLE and formulation_type is not None:
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            model = gp.Model("WeightedTardiness", env=env)
            
            rem_time = args.time_limit - (time.time() - start_cpu) - 1.0
            if rem_time <= 0:
                rem_time = 1.0
            model.setParam('TimeLimit', rem_time)
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            
            x = {}
            C = {}
            job_ids = list(jobs_dict.keys())
            
            if formulation_type == "time_indexed":
                x_indices = [(j, t) for j in job_ids for t in range(jobs_dict[j]['p'], H + 1)]
                x = model.addVars(x_indices, vtype=GRB.BINARY, name="x")
                
                obj_expr = gp.quicksum(jobs_dict[j]['w'] * max(0, t - jobs_dict[j]['d']) * x[j, t] for j, t in x_indices)
                model.setObjective(obj_expr, GRB.MINIMIZE)
                
                model.addConstrs((gp.quicksum(x[j, t] for t in range(jobs_dict[j]['p'], H + 1)) == 1 for j in job_ids), name="once")
                
                for tau in range(H):
                    active_vars = []
                    for j in job_ids:
                        p = jobs_dict[j]['p']
                        for t in range(max(p, tau + 1), min(H + 1, tau + p + 1)):
                            if (j, t) in x:
                                active_vars.append(x[j, t])
                    if active_vars:
                        model.addConstr(gp.quicksum(active_vars) <= m, name=f"cap_{tau}")
                
                sorted_sched = sorted(best_initial_sched_raw, key=lambda i: i['start_time'])
                machine_avail = [0] * m
                for s in sorted_sched:
                    j = s['job']
                    st = int(round(s['start_time']))
                    
                    assigned_m = -1
                    for k in range(m):
                        if machine_avail[k] <= st:
                            assigned_m = k
                            break
                    if assigned_m == -1:
                        t_min = min(machine_avail)
                        assigned_m = machine_avail.index(t_min)
                        st = max(st, machine_avail[assigned_m])
                        
                    ct = st + jobs_dict[j]['p']
                    if (j, ct) in x:
                        x[j, ct].Start = 1.0
                    machine_avail[assigned_m] = ct
                        
            else:
                C = model.addVars(job_ids, vtype=GRB.CONTINUOUS, lb=0.0, name="C")
                T = model.addVars(job_ids, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
                z_indices = [(i, j) for i in [0] + job_ids for j in job_ids if i != j]
                z = model.addVars(z_indices, vtype=GRB.BINARY, name="z")
                
                model.addConstrs((T[j] >= C[j] - jobs_dict[j]['d'] for j in job_ids), name="tardiness")
                model.addConstrs((gp.quicksum(z[i, j] for i in [0] + job_ids if i != j) == 1 for j in job_ids), name="pred")
                model.addConstrs((gp.quicksum(z[i, j] for j in job_ids if i != j) <= 1 for i in job_ids), name="succ")
                model.addConstr(gp.quicksum(z[0, j] for j in job_ids) <= m, name="max_m")
                
                for i in job_ids:
                    for j in job_ids:
                        if i != j:
                            model.addConstr(C[j] >= C[i] + jobs_dict[j]['p'] - H * (1 - z[i, j]))
                            
                model.addConstrs((C[j] >= jobs_dict[j]['p'] for j in job_ids), name="min_c")
                model.addConstrs((C[j] <= H for j in job_ids), name="max_c")
                
                model.setObjective(gp.quicksum(jobs_dict[j]['w'] * T[j] for j in job_ids), GRB.MINIMIZE)
                
                sorted_sched = sorted(best_initial_sched_raw, key=lambda i: i['start_time'])
                machine_avail = [0] * m
                machine_last = [0] * m
                for s in sorted_sched:
                    j = s['job']
                    st = int(round(s['start_time']))
                    
                    assigned_m = -1
                    for k in range(m):
                        if machine_avail[k] <= st:
                            assigned_m = k
                            break
                    if assigned_m == -1: 
                        t_min = min(machine_avail)
                        assigned_m = machine_avail.index(t_min)
                        st = max(st, machine_avail[assigned_m])
                        
                    ct = st + jobs_dict[j]['p']
                    C[j].Start = ct
                    
                    prev = machine_last[assigned_m]
                    if (prev, j) in z:
                        z[prev, j].Start = 1.0
                    
                    machine_avail[assigned_m] = ct
                    machine_last[assigned_m] = j

            model._x = x if formulation_type == "time_indexed" else None
            model._C = C if formulation_type == "sequence" else None
            
            def cb(mdl, where):
                if where == GRB.Callback.MIPSOL:
                    obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if logger:
                        try:
                            sched_raw = []
                            if formulation_type == "time_indexed":
                                x_vars = list(mdl._x.values())
                                x_keys = list(mdl._x.keys())
                                sol = mdl.cbGetSolution(x_vars)
                                for k, v in zip(x_keys, sol):
                                    if v > 0.5:
                                        j, t = k
                                        st = t - jobs_dict[j]['p']
                                        sched_raw.append({'job': j, 'start_time': st})
                            else:
                                c_vars = list(mdl._C.values())
                                c_keys = list(mdl._C.keys())
                                sol = mdl.cbGetSolution(c_vars)
                                for j, v in zip(c_keys, sol):
                                    st = v - jobs_dict[j]['p']
                                    sched_raw.append({'job': j, 'start_time': st})
                                    
                            true_obj, full_sched = format_schedule(sched_raw, jobs_dict, m)
                            logger.log_solution(true_obj, full_sched)
                        except Exception as e:
                            logger.log(obj)
                            
            model.optimize(cb)
            
            if model.SolCount > 0:
                sched_raw = []
                if formulation_type == "time_indexed":
                    for (j, t), var in x.items():
                        if var.X > 0.5:
                            st = t - jobs_dict[j]['p']
                            sched_raw.append({'job': j, 'start_time': st})
                else:
                    for j, var in C.items():
                        ct = var.X
                        st = ct - jobs_dict[j]['p']
                        sched_raw.append({'job': j, 'start_time': st})
                final_obj, final_sched = format_schedule(sched_raw, jobs_dict, m)

        except Exception as e:
            # Catch all Gurobi initialization / model construction / solving errors (e.g. license limits)
            print(f"Gurobi Error: {e}", file=sys.stderr)
            # We already have a valid final_obj and final_sched from heuristics/local search to fall back on.
            pass

    out = {
        "objective_value": final_obj,
        "schedule": final_sched
    }
    with open(args.solution_path, 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()