import sys
import time
import json
import random
import math
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def get_makespan(seq, p_times, no_idle_flags, num_machines):
    n = len(seq)
    C_prev = [0] * n
    C_curr = [0] * n
    
    for k in range(num_machines):
        is_no_idle = no_idle_flags[k]
        if not is_no_idle:
            c = C_prev[0] + p_times[seq[0]][k]
            C_curr[0] = c
            for i in range(1, n):
                prev_c = C_prev[i]
                p = p_times[seq[i]][k]
                c = (prev_c + p) if prev_c > c else (c + p)
                C_curr[i] = c
        else:
            max_S = 0
            curr_sum = 0
            for i in range(n):
                S_req = C_prev[i] - curr_sum
                if S_req > max_S:
                    max_S = S_req
                curr_sum += p_times[seq[i]][k]
            
            c = max_S + p_times[seq[0]][k]
            C_curr[0] = c
            for i in range(1, n):
                c += p_times[seq[i]][k]
                C_curr[i] = c
                
        C_prev, C_curr = C_curr, C_prev
        
    return C_prev[-1]

def compute_full_completion_times(seq, p_times, no_idle_flags, num_machines):
    n = len(seq)
    C = [[0] * num_machines for _ in range(n)]
    
    for k in range(num_machines):
        if not no_idle_flags[k]:
            C[0][k] = (C[0][k-1] if k > 0 else 0) + p_times[seq[0]][k]
            for i in range(1, n):
                prev_c = C[i][k-1] if k > 0 else 0
                C[i][k] = max(prev_c, C[i-1][k]) + p_times[seq[i]][k]
        else:
            max_S = 0
            curr_sum = 0
            for i in range(n):
                prev_c = C[i][k-1] if k > 0 else 0
                S_req = prev_c - curr_sum
                if S_req > max_S:
                    max_S = S_req
                curr_sum += p_times[seq[i]][k]
            
            C[0][k] = max_S + p_times[seq[0]][k]
            for i in range(1, n):
                C[i][k] = C[i-1][k] + p_times[seq[i]][k]
                
    comp_times_dict = {}
    for k in range(num_machines):
        comp_times_dict[f"machine_{k}"] = [C[i][k] for i in range(n)]
        
    return comp_times_dict

def neh_initial(jobs, p_times, no_idle_flags, num_machines):
    sum_p = [sum(p_times[j]) for j in jobs]
    sorted_jobs = sorted(jobs, key=lambda j: sum_p[j], reverse=True)
    
    seq = [sorted_jobs[0]]
    for i in range(1, len(sorted_jobs)):
        job = sorted_jobs[i]
        best_pos = -1
        best_mk = float('inf')
        for pos in range(len(seq) + 1):
            seq.insert(pos, job)
            mk = get_makespan(seq, p_times, no_idle_flags, num_machines)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
            seq.pop(pos)
        seq.insert(best_pos, job)
        
    return seq, get_makespan(seq, p_times, no_idle_flags, num_machines)

def destruct_construct(seq, d, p_times, no_idle_flags, num_machines):
    removed_jobs = random.sample(seq, d)
    removed_set = set(removed_jobs)
    new_seq = [j for j in seq if j not in removed_set]
    
    for job in removed_jobs:
        best_pos = -1
        best_mk = float('inf')
        for pos in range(len(new_seq) + 1):
            new_seq.insert(pos, job)
            mk = get_makespan(new_seq, p_times, no_idle_flags, num_machines)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
            new_seq.pop(pos)
        new_seq.insert(best_pos, job)
        
    return new_seq, best_mk

def local_search(seq, p_times, no_idle_flags, num_machines, best_known):
    n = len(seq)
    current_mk = get_makespan(seq, p_times, no_idle_flags, num_machines)
    improved = True
    while improved:
        improved = False
        job_indices = list(range(n))
        random.shuffle(job_indices)
        for i in job_indices:
            job = seq.pop(i)
            for pos in range(n):
                if pos == i:
                    continue
                seq.insert(pos, job)
                mk = get_makespan(seq, p_times, no_idle_flags, num_machines)
                if mk < current_mk:
                    current_mk = mk
                    improved = True
                    break
                seq.pop(pos)
                
            if improved:
                break
            else:
                seq.insert(i, job)
                
    return seq, current_mk

def iterated_greedy(p_times, no_idle_flags, num_machines, num_jobs, max_time_for_ig, global_start_time, log_solution, initial_seq, initial_mk):
    current_seq = initial_seq.copy()
    current_mk = initial_mk
    best_seq_saved = initial_seq.copy()
    best_mk = initial_mk
    
    sum_p = sum(p_times[j][k] for j in range(num_jobs) for k in range(num_machines))
    T_0 = 0.4 * sum_p / (num_jobs * num_machines)
    d = min(4, num_jobs)
    
    while time.time() - global_start_time < max_time_for_ig:
        new_seq, new_mk = destruct_construct(current_seq, d, p_times, no_idle_flags, num_machines)
        new_seq, new_mk = local_search(new_seq, p_times, no_idle_flags, num_machines, best_mk)
        
        if new_mk < best_mk:
            best_mk = new_mk
            best_seq_saved = new_seq.copy()
            log_solution(best_mk)
            
        if new_mk <= current_mk:
            current_seq = new_seq
            current_mk = new_mk
        else:
            delta = new_mk - current_mk
            prob = math.exp(-delta / T_0) if T_0 > 0 else 0
            if random.random() <= prob:
                current_seq = new_seq
                current_mk = new_mk
                
    return best_seq_saved, best_mk

def run_gurobi(p_times, no_idle_flags, num_machines, num_jobs, best_seq, start_time, time_limit, log_solution):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    
    rem_time = time_limit - (time.time() - start_time) - 1.0
    if rem_time <= 0:
        return best_seq, get_makespan(best_seq, p_times, no_idle_flags, num_machines)
        
    env.setParam("TimeLimit", rem_time)
    env.setParam("MIPFocus", 1)
    env.start()
    
    model = gp.Model("MixedNoIdlePFSP", env=env)
    
    x = model.addVars(num_jobs, num_jobs, vtype=GRB.BINARY, name="x")
    
    for i in range(num_jobs):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_jobs)) == 1, name=f"pos_{i}")
    for j in range(num_jobs):
        model.addConstr(gp.quicksum(x[i, j] for i in range(num_jobs)) == 1, name=f"job_{j}")
        
    P = [[None] * num_machines for _ in range(num_jobs)]
    for i in range(num_jobs):
        for k in range(num_machines):
            P[i][k] = gp.quicksum(x[i, j] * p_times[j][k] for j in range(num_jobs) if p_times[j][k] > 0)
            
    C = model.addVars(num_jobs, num_machines, vtype=GRB.CONTINUOUS, lb=0, name="C")
    
    for k in range(num_machines):
        is_no_idle = no_idle_flags[k]
        
        # Position 0
        if k == 0:
            model.addConstr(C[0, 0] >= P[0][0])
        else:
            model.addConstr(C[0, k] >= C[0, k-1] + P[0][k])
            
        # Positions 1 to N-1
        for i in range(1, num_jobs):
            if not is_no_idle:
                model.addConstr(C[i, k] >= C[i-1, k] + P[i][k])
                if k > 0:
                    model.addConstr(C[i, k] >= C[i, k-1] + P[i][k])
            else:
                model.addConstr(C[i, k] == C[i-1, k] + P[i][k])
                if k > 0:
                    model.addConstr(C[i, k] >= C[i, k-1] + P[i][k])
                    
    model.setObjective(C[num_jobs-1, num_machines-1], GRB.MINIMIZE)
    
    if best_seq is not None:
        for i in range(num_jobs):
            job = best_seq[i]
            x[i, job].Start = 1.0
            
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            log_solution(round(obj))

    model.optimize(cb)
    
    try:
        if model.SolCount > 0:
            final_seq = [-1] * num_jobs
            for i in range(num_jobs):
                for j in range(num_jobs):
                    if x[i, j].X > 0.5:
                        final_seq[i] = j
                        break
            for v in final_seq:
                if v == -1:
                    return best_seq, get_makespan(best_seq, p_times, no_idle_flags, num_machines)
            return final_seq, get_makespan(final_seq, p_times, no_idle_flags, num_machines)
    except Exception:
        pass
        
    return best_seq, get_makespan(best_seq, p_times, no_idle_flags, num_machines)

def solve(instance_data, time_limit, log_path):
    global_start_time = time.time()
    
    num_jobs = instance_data["n_jobs"]
    num_machines = instance_data["n_machines"]
    no_idle_flags = instance_data["no_idle_flags"]
    p_times = instance_data["processing_times"]
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    def log_solution(obj):
        if logger:
            logger.log(float(obj))
            
    best_seq, best_mk = neh_initial(list(range(num_jobs)), p_times, no_idle_flags, num_machines)
    log_solution(best_mk)
    
    if num_jobs <= 150:
        ig_time = time_limit * 0.4
    else:
        ig_time = time_limit * 0.8
        
    ig_time = min(ig_time, time_limit - 1.5)
    
    if ig_time > 0 and (time.time() - global_start_time) < ig_time:
        best_seq, best_mk = iterated_greedy(
            p_times, no_idle_flags, num_machines, num_jobs, 
            ig_time, global_start_time, log_solution, best_seq, best_mk
        )
        
    rem_time = time_limit - (time.time() - global_start_time) - 0.5
    if rem_time > 1.0 and num_jobs <= 150:
        best_seq, best_mk = run_gurobi(
            p_times, no_idle_flags, num_machines, num_jobs, 
            best_seq, global_start_time, time_limit, log_solution
        )
        
    C = compute_full_completion_times(best_seq, p_times, no_idle_flags, num_machines)
    
    return best_seq, best_mk, C

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        instance_data = json.load(f)
        
    best_seq, best_mk, comp_times = solve(instance_data, args.time_limit, args.log_path)
    
    solution = {
        "objective_value": float(best_mk),
        "sequence": best_seq,
        "completion_times": comp_times
    }
    
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)