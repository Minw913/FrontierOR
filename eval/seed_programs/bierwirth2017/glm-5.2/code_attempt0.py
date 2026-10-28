import json
import argparse
import sys
import time
from gurobipy import Model, GRB, quicksum

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    return parser.parse_args()

def load_instance(path):
    with open(path) as f:
        return json.load(f)

def build_heuristic_schedule(instance):
    jobs = sorted(instance['jobs'], key=lambda x: x['job_id'])
    num_machines = instance['num_machines']
    
    # Try several orderings of jobs to find the best heuristic schedule
    orderings = []
    orderings.append(sorted(range(len(jobs)), key=lambda j: jobs[j]['release_date']))
    orderings.append(sorted(range(len(jobs)), key=lambda j: jobs[j]['due_date']))
    orderings.append(sorted(range(len(jobs)), key=lambda j: -jobs[j]['weight']))
    
    total_pt = [sum(op['processing_time'] for op in jobs[j]['operations']) for j in range(len(jobs))]
    orderings.append(sorted(range(len(jobs)), key=lambda j: -jobs[j]['weight'] / total_pt[j] if total_pt[j] > 0 else 0))
    orderings.append(sorted(range(len(jobs)), key=lambda j: jobs[j]['due_date'] / jobs[j]['weight'] if jobs[j]['weight'] > 0 else float('inf')))
    
    best_solution = None
    best_obj = float('inf')
    
    for order in orderings:
        machine_free = [0] * num_machines
        job_start_times = {}
        for j in order:
            job = jobs[j]
            ops = job['operations']
            start_times = []
            t = job['release_date']
            for op in ops:
                m = op['machine']
                p = op['processing_time']
                start = max(t, machine_free[m])
                start_times.append(start)
                machine_free[m] = start + p
                t = start + p
            job_start_times[j] = start_times
        
        schedule = []
        obj = 0.0
        for j in range(len(jobs)):
            job = jobs[j]
            ops = job['operations']
            start_times = job_start_times[j]
            operations = []
            for idx, op in enumerate(ops):
                operations.append({
                    "machine": op['machine'],
                    "start_time": start_times[idx]
                })
            completion_time = start_times[-1] + ops[-1]['processing_time']
            tardiness = max(0.0, completion_time - job['due_date'])
            schedule.append({
                "job_id": job['job_id'],
                "completion_time": float(completion_time),
                "tardiness": float(tardiness),
                "operations": operations
            })
            obj += job['weight'] * tardiness
        if obj < best_obj:
            best_obj = obj
            best_solution = {"objective_value": float(best_obj), "schedule": schedule}
    return best_solution, best_obj

def main():
    args = parse_args()
    start_time = time.time()
    
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    instance = load_instance(args.instance_path)
    jobs = sorted(instance['jobs'], key=lambda x: x['job_id'])
    num_jobs = len(jobs)
    num_machines = instance['num_machines']
    
    # Build operation list
    op_list = []
    job_first_op = {}
    job_last_op = {}
    job_ops = {}  # job_id -> list of (op_id, machine, proc_time)
    op_id_counter = 0
    for j, job in enumerate(jobs):
        job_ops[j] = []
        prev_id = None
        for i, op in enumerate(job['operations']):
            op_id = op_id_counter
            op_id_counter += 1
            op_dict = {
                'op_id': op_id,
                'job_id': j,
                'machine': op['machine'],
                'proc_time': op['processing_time'],
                'prev_op_id': prev_id,
                'next_op_id': None
            }
            op_list.append(op_dict)
            job_ops[j].append((op_id, op['machine'], op['processing_time']))
            if i == 0:
                job_first_op[j] = op_id
            if prev_id is not None:
                op_list[prev_id]['next_op_id'] = op_id
            prev_id = op_id
        job_last_op[j] = prev_id
    
    # Heuristic solution
    heur_solution, heur_obj = build_heuristic_schedule(instance)
    if logger:
        logger.log_solution(heur_obj, heur_solution)
    
    # MIP model
    model = Model('jobshop_wt')
    model.setParam('OutputFlag', 0)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    max_release = max(job['release_date'] for job in jobs)
    total_proc = sum(op['proc_time'] for op in op_list)
    H = max_release + total_proc + 1  # big M
    
    # Variables
    S = {}
    for op in op_list:
        S[op['op_id']] = model.addVar(lb=0, ub=H, name=f"S_{op['op_id']}")
    T = {}
    for j in range(num_jobs):
        T[j] = model.addVar(lb=0, name=f"T_{j}")
    
    # Machine disjunction variables
    machine_ops = {m: [] for m in range(num_machines)}
    for op in op_list:
        machine_ops[op['machine']].append(op['op_id'])
    
    y = {}
    for m in range(num_machines):
        ops_m = machine_ops[m]
        for i in range(len(ops_m)):
            for k in range(i+1, len(ops_m)):
                op_i = ops_m[i]
                op_k = ops_m[k]
                y[(op_i, op_k)] = model.addVar(vtype=GRB.BINARY, name=f"y_{op_i}_{op_k}")
    
    model.update()
    
    # Constraints
    for j in range(num_jobs):
        model.addConstr(S[job_first_op[j]] >= jobs[j]['release_date'])
    for op in op_list:
        if op['next_op_id'] is not None:
            model.addConstr(S[op['next_op_id']] >= S[op['op_id']] + op['proc_time'])
    
    for m in range(num_machines):
        ops_m = machine_ops[m]
        for i in range(len(ops_m)):
            for k in range(i+1, len(ops_m)):
                op_i = ops_m[i]
                op_k = ops_m[k]
                p_i = op_list[op_i]['proc_time']
                p_k = op_list[op_k]['proc_time']
                model.addConstr(S[op_k] >= S[op_i] + p_i - H * (1 - y[(op_i, op_k)]))
                model.addConstr(S[op_i] >= S[op_k] + p_k - H * y[(op_i, op_k)])
    
    for j in range(num_jobs):
        last_op = job_last_op[j]
        p_last = op_list[last_op]['proc_time']
        model.addConstr(T[j] >= S[last_op] + p_last - jobs[j]['due_date'])
    
    model.setObjective(quicksum(jobs[j]['weight'] * T[j] for j in range(num_jobs)), GRB.MINIMIZE)
    
    # MIP start from heuristic
    if heur_solution is not None:
        for j in range(num_jobs):
            job_sched = heur_solution['schedule'][j]
            T[j].Start = job_sched['tardiness']
            for idx, (op_id, m, p) in enumerate(job_ops[j]):
                S[op_id].Start = job_sched['operations'][idx]['start_time']
        for m in range(num_machines):
            ops_m = machine_ops[m]
            for i in range(len(ops_m)):
                for k in range(i+1, len(ops_m)):
                    op_i = ops_m[i]
                    op_k = ops_m[k]
                    s_i = S[op_i].Start
                    s_k = S[op_k].Start
                    if s_i + op_list[op_i]['proc_time'] <= s_k:
                        y[(op_i, op_k)].Start = 1.0
                    else:
                        y[(op_i, op_k)].Start = 0.0
        model.update()
    
    # Prepare structures for callback and final extraction
    S_vars_list = [S[op['op_id']] for op in op_list]
    T_vars_list = [T[j] for j in range(num_jobs)]
    weights = [jobs[j]['weight'] for j in range(num_jobs)]
    due_dates = [jobs[j]['due_date'] for j in range(num_jobs)]
    
    job_ops_list = []
    for j in range(num_jobs):
        ops = []
        for op_id, m, p in job_ops[j]:
            ops.append((op_id, m, p))
        job_ops_list.append(ops)
    
    # Callback for logging incumbents
    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                S_val = model.cbGetSolution(S_vars_list)
                T_val = model.cbGetSolution(T_vars_list)
                schedule = []
                obj_val = 0.0
                for j in range(num_jobs):
                    ops = job_ops_list[j]
                    operations = []
                    completion_time = 0.0
                    for idx, (op_id, m, p) in enumerate(ops):
                        start = S_val[op_id]
                        operations.append({
                            "machine": m,
                            "start_time": float(start)
                        })
                        completion_time = start + p
                    tardiness = max(0.0, completion_time - due_dates[j])
                    schedule.append({
                        "job_id": j,
                        "completion_time": float(completion_time),
                        "tardiness": float(tardiness),
                        "operations": operations
                    })
                    obj_val += weights[j] * tardiness
                logger.log_solution(obj_val, {"objective_value": float(obj_val), "schedule": schedule})
        
        model.setCallback(callback)
    
    # Set remaining time limit
    elapsed = time.time() - start_time
    remaining = max(1.0, args.time_limit - elapsed)
    model.setParam('TimeLimit', remaining)
    
    try:
        model.optimize()
    except Exception as e:
        print(f"Gurobi optimization failed: {e}", file=sys.stderr)
    
    # Determine final solution
    final_solution = None
    if model.SolCount > 0:
        S_val = model.getAttr('X', S_vars_list)
        schedule = []
        obj_val = 0.0
        for j in range(num_jobs):
            ops = job_ops_list[j]
            operations = []
            completion_time = 0.0
            for idx, (op_id, m, p) in enumerate(ops):
                start = S_val[op_id]
                operations.append({
                    "machine": m,
                    "start_time": float(start)
                })
                completion_time = start + p
            tardiness = max(0.0, completion_time - due_dates[j])
            schedule.append({
                "job_id": j,
                "completion_time": float(completion_time),
                "tardiness": float(tardiness),
                "operations": operations
            })
            obj_val += weights[j] * tardiness
        final_solution = {"objective_value": float(obj_val), "schedule": schedule}
    else:
        final_solution = heur_solution
    
    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)

if __name__ == '__main__':
    main()