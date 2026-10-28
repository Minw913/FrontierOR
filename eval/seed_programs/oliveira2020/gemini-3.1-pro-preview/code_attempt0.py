import argparse
import json
import math
import sys
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def run_heuristics(jobs, m, H):
    best_schedule = None
    best_obj = float('inf')
    
    strategies = [('EDD', 0), ('WSPT', 0)]
    for K in [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        strategies.append(('ATC', K))
        
    for strat, K in strategies:
        U = jobs[:]
        avail = [0] * m
        sched = []
        obj = 0
        
        while U:
            t = min(avail)
            machine = avail.index(t)
            
            if strat == 'EDD':
                best_j = min(U, key=lambda j: (j['d'], j['p']))
            elif strat == 'WSPT':
                best_j = max(U, key=lambda j: (j['w']/j['p'], -j['d']))
            elif strat == 'ATC':
                p_avg = sum(j['p'] for j in U) / len(U)
                best_j = None
                best_score = -1
                for j in U:
                    score = (j['w'] / j['p']) * math.exp(-max(0, j['d'] - t - j['p']) / (K * p_avg))
                    if score > best_score:
                        best_score = score
                        best_j = j
                        
            start_time = t
            comp_time = t + best_j['p']
            avail[machine] = comp_time
            obj += best_j['w'] * max(0, comp_time - best_j['d'])
            
            sched.append({
                'job': best_j['id'],
                'start_time': start_time,
                'completion_time': comp_time
            })
            U.remove(best_j)
            
        if obj < best_obj:
            best_obj = obj
            best_schedule = sched
            
    return best_obj, best_schedule

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
            assigned_m = machine_avail.index(min(machine_avail))
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
        
    best_heuristic_obj, best_heuristic_sched_raw = run_heuristics(jobs_list, m, H)
    heuristic_obj, heuristic_solution = format_schedule(best_heuristic_sched_raw, jobs_dict, m)
    
    if logger:
        logger.log_solution(heuristic_obj, heuristic_solution)
        
    N_nz = 5 * n * H
    formulation_type = "time_indexed" if N_nz <= 3_000_000 else "sequence"
    
    env = gp.Env(empty=True)
    env.start()
    
    model = gp.Model("WeightedTardiness", env=env)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    x = {}
    C = {}
    
    if formulation_type == "time_indexed":
        x_indices = [(j, t) for j in range(1, n + 1) for t in range(jobs_dict[j]['p'], H + 1)]
        x = model.addVars(x_indices, vtype=GRB.BINARY, name="x")
        
        u_indices = [(j, tau) for j in range(1, n + 1) for tau in range(H)]
        u = model.addVars(u_indices, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="u")
        
        obj_expr = gp.quicksum(jobs_dict[j]['w'] * max(0, t - jobs_dict[j]['d']) * x[j, t] for j, t in x_indices)
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        model.addConstrs((gp.quicksum(x[j, t] for t in range(jobs_dict[j]['p'], H + 1)) == 1 for j in range(1, n + 1)), name="once")
        
        def get_x(j, t):
            return x.get((j, t), 0)
            
        model.addConstrs(
            (u[j, tau] - (u[j, tau - 1] if tau > 0 else 0) == get_x(j, tau + jobs_dict[j]['p']) - get_x(j, tau)
             for j in range(1, n + 1) for tau in range(H)), name="flow"
        )
        
        model.addConstrs((gp.quicksum(u[j, tau] for j in range(1, n + 1)) <= m for tau in range(H)), name="cap")
        
        for s in best_heuristic_sched_raw:
            j = s['job']
            t = s['start_time'] + jobs_dict[j]['p']
            if (j, t) in x:
                x[j, t].Start = 1.0
                
    else:
        C = model.addVars(range(1, n + 1), vtype=GRB.CONTINUOUS, lb=0.0, name="C")
        T = model.addVars(range(1, n + 1), vtype=GRB.CONTINUOUS, lb=0.0, name="T")
        z_indices = [(i, j) for i in range(n + 1) for j in range(1, n + 1) if i != j]
        z = model.addVars(z_indices, vtype=GRB.BINARY, name="z")
        
        model.addConstrs((T[j] >= C[j] - jobs_dict[j]['d'] for j in range(1, n + 1)), name="tardiness")
        model.addConstrs((gp.quicksum(z[i, j] for i in range(n + 1) if i != j) == 1 for j in range(1, n + 1)), name="pred")
        model.addConstrs((gp.quicksum(z[i, j] for j in range(1, n + 1) if i != j) <= 1 for i in range(1, n + 1)), name="succ")
        model.addConstr(gp.quicksum(z[0, j] for j in range(1, n + 1)) <= m, name="max_m")
        
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i != j:
                    model.addConstr(C[j] >= C[i] + jobs_dict[j]['p'] - H * (1 - z[i, j]))
                    
        model.addConstrs((C[j] >= jobs_dict[j]['p'] for j in range(1, n + 1)), name="min_c")
        model.addConstrs((C[j] <= H for j in range(1, n + 1)), name="max_c")
        
        model.setObjective(gp.quicksum(jobs_dict[j]['w'] * T[j] for j in range(1, n + 1)), GRB.MINIMIZE)
        
        sorted_sched = sorted(best_heuristic_sched_raw, key=lambda i: i['start_time'])
        machine_avail = [0] * m
        machine_last = [0] * m
        for s in sorted_sched:
            j = s['job']
            st = s['start_time']
            ct = st + jobs_dict[j]['p']
            C[j].Start = ct
            
            assigned_m = -1
            for k in range(m):
                if machine_avail[k] <= st:
                    assigned_m = k
                    break
            if assigned_m == -1: 
                assigned_m = machine_avail.index(min(machine_avail))
            
            prev = machine_last[assigned_m]
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
                        for (j, t), var in mdl._x.items():
                            if mdl.cbGetSolution(var) > 0.5:
                                st = t - jobs_dict[j]['p']
                                sched_raw.append({'job': j, 'start_time': st})
                    else:
                        for j, var in mdl._C.items():
                            ct = mdl.cbGetSolution(var)
                            st = ct - jobs_dict[j]['p']
                            sched_raw.append({'job': j, 'start_time': st})
                    true_obj, full_sched = format_schedule(sched_raw, jobs_dict, m)
                    logger.log_solution(true_obj, full_sched)
                except Exception:
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
    else:
        final_obj, final_sched = heuristic_obj, heuristic_solution
        
    out = {
        "objective_value": final_obj,
        "schedule": final_sched
    }
    with open(args.solution_path, 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()