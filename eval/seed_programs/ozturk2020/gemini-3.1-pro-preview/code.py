import sys
import json
import argparse
import time
import random
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

class MIPCallbackData:
    def __init__(self, logger, best_obj):
        self.logger = logger
        self.best_obj = best_obj

def mip_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        try:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            cb_data = model._cb_data
            if obj < cb_data.best_obj - 1e-4:
                cb_data.best_obj = obj
                if cb_data.logger:
                    cb_data.logger.log(obj)
        except Exception:
            pass

def solve_greedy(jobs, M, Cap, logger):
    best_obj = float('inf')
    best_sol = None
    random.seed(42)
    
    N = len(jobs)
    num_iters = 1000
    if N > 100: num_iters = 500
    if N > 300: num_iters = 100
    if N > 500: num_iters = 50
    
    for iteration in range(num_iters):
        m_av = [0] * M
        U = list(jobs)
        batches = []
        
        while U:
            m_idx = m_av.index(min(m_av))
            t_m = m_av[m_idx]
            
            future_r = sorted(list(set(j['release_date'] for j in U if j['release_date'] >= t_m)))[:5]
            if t_m not in future_r:
                future_r.insert(0, t_m)
                
            candidate_batches = []
            for t in future_r:
                avail_at_t = [j for j in U if j['release_date'] <= t]
                if not avail_at_t: continue
                
                seeds = []
                if iteration == 0:
                    avail_at_t.sort(key=lambda x: (x['processing_time']))
                    seeds = avail_at_t[:1]
                elif iteration == 1:
                    avail_at_t.sort(key=lambda x: (-x['processing_time']))
                    seeds = avail_at_t[:1]
                else:
                    seeds = avail_at_t if len(avail_at_t) <= 5 else random.sample(avail_at_t, 5)
                
                for seed in seeds:
                    curr_batch = [seed]
                    curr_cap = seed['size']
                    others = [x for x in avail_at_t if x != seed]
                    
                    if iteration % 3 == 0:
                        others.sort(key=lambda x: abs(x['processing_time'] - seed['processing_time']))
                    elif iteration % 3 == 1:
                        others.sort(key=lambda x: -x['size'])
                    else:
                        random.shuffle(others)
                        
                    for o in others:
                        if curr_cap + o['size'] <= Cap:
                            curr_batch.append(o)
                            curr_cap += o['size']
                            
                    p_b = max(x['processing_time'] for x in curr_batch)
                    cb_obj = (t + p_b) * len(curr_batch)
                    score = cb_obj / (len(curr_batch)**2)
                    candidate_batches.append((score, t, p_b, curr_batch))
            
            if not candidate_batches:
                m_av[m_idx] += 1
                continue
                
            if iteration < 5:
                candidate_batches.sort(key=lambda x: x[0])
                best_cand = candidate_batches[0]
            else:
                candidate_batches.sort(key=lambda x: x[0])
                pool_size = max(1, len(candidate_batches)//3)
                best_cand = random.choice(candidate_batches[:pool_size])
                
            ch_t = best_cand[1]
            ch_pb = best_cand[2]
            ch_b = best_cand[3]
            
            m_av[m_idx] = ch_t + ch_pb
            batches.append((m_idx, ch_t, ch_pb, [x['job_id'] for x in ch_b]))
            for x in ch_b:
                U.remove(x)
                
        obj = sum((b[1]+b[2]) * len(b[3]) for b in batches)
        if obj < best_obj:
            best_obj = obj
            best_sol = batches
            if logger:
                logger.log(float(obj))
                
    return best_obj, best_sol

def generate_final_output(sol_tuples, M, job_id_to_job):
    m_batches = defaultdict(list)
    for b in sol_tuples:
        m_batches[b[0]].append(b)
        
    sol_batches = []
    sol_job_assignments = {}
    obj_val = 0
    
    for m in range(M):
        if m not in m_batches: continue
        batches = sorted(m_batches[m], key=lambda x: x[1])
        curr_t = 0
        for b_idx, b in enumerate(batches):
            assigned_jobs = b[3]
            r_max = max(job_id_to_job[j_id]['release_date'] for j_id in assigned_jobs)
            p_max = max(job_id_to_job[j_id]['processing_time'] for j_id in assigned_jobs)
            
            st = max(int(curr_t), int(r_max))
            pt = int(p_max)
            ct = st + pt
            
            batch_id = b_idx + 1
            sol_batches.append({
                "batch_id": batch_id,
                "machine": m,
                "start_time": st,
                "processing_time": pt,
                "jobs": assigned_jobs
            })
            for j_id in assigned_jobs:
                sol_job_assignments[str(j_id)] = {
                    "batch": batch_id,
                    "machine": m,
                    "completion_time": ct
                }
            obj_val += ct * len(assigned_jobs)
            curr_t = ct
            
    return {
        "objective_value": float(obj_val),
        "batches": sol_batches,
        "job_assignments": sol_job_assignments
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
        
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    jobs = data['jobs']
    N = data['num_jobs']
    M = data['num_machines']
    Cap = data['batch_capacity']
    
    job_id_to_job = {j['job_id']: j for j in jobs}
    job_id_to_idx = {j['job_id']: idx for idx, j in enumerate(jobs)}
    
    if N == 0:
        output_dict = generate_final_output([], M, job_id_to_job)
        with open(args.solution_path, 'w') as f:
            json.dump(output_dict, f, indent=2)
        sys.exit(0)
    
    best_greedy_obj, best_greedy_sol = solve_greedy(jobs, M, Cap, logger)
    
    # Check remaining time
    elapsed = time.time() - start_time
    rem_time = args.time_limit - elapsed - 2.0
    
    sol_tuples = best_greedy_sol
    
    if rem_time > 0:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model(env=env)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', rem_time)
        
        # Max batches on any machine from greedy solution
        max_b_greedy = 0
        m_counts = defaultdict(int)
        for b in best_greedy_sol:
            m_counts[b[0]] += 1
        if m_counts:
            max_b_greedy = max(m_counts.values())
        
        K = min(N, max(max_b_greedy + 2, int(N / M * 3) + 2))
        
        # Upper bound on completion times
        r = [j['release_date'] for j in jobs]
        p = [j['processing_time'] for j in jobs]
        s = [j['size'] for j in jobs]
        job_id = [j['job_id'] for j in jobs]
        H = max(r) + sum(p) if N > 0 else 0
        
        x = model.addVars(M, K, N, vtype=GRB.BINARY, name="x")
        u = model.addVars(M, K, vtype=GRB.BINARY, name="u")
        S = model.addVars(M, K, lb=0, ub=H, vtype=GRB.CONTINUOUS, name="S")
        P = model.addVars(M, K, lb=0, ub=H, vtype=GRB.CONTINUOUS, name="P")
        C = model.addVars(M, K, lb=0, ub=H, vtype=GRB.CONTINUOUS, name="C")
        Z = model.addVars(N, lb=0, ub=H, vtype=GRB.CONTINUOUS, name="Z")
        
        # 1. Assignment
        for j in range(N):
            model.addConstr(gp.quicksum(x[m, k, j] for m in range(M) for k in range(K)) == 1)
            
        # 2. Capacity & usage linkage
        for m in range(M):
            for k in range(K):
                model.addConstr(gp.quicksum(s[j] * x[m, k, j] for j in range(N)) <= Cap * u[m, k])
                model.addConstr(u[m, k] <= gp.quicksum(x[m, k, j] for j in range(N)))
                for j in range(N):
                    model.addConstr(x[m, k, j] <= u[m, k])
                    
        # 3. Push empty batches
        for m in range(M):
            for k in range(1, K):
                model.addConstr(u[m, k] <= u[m, k-1])
                
        # 4. Machine Symmetry Breaking
        for m in range(M - 1):
            model.addConstr(gp.quicksum(u[m, k] for k in range(K)) >= gp.quicksum(u[m+1, k] for k in range(K)))
            
        # 5. Processing time bounds
        for m in range(M):
            for k in range(K):
                for j in range(N):
                    model.addConstr(P[m, k] >= p[j] * x[m, k, j])
                    
        # 6. Start time bounds
        for m in range(M):
            for k in range(K):
                for j in range(N):
                    model.addConstr(S[m, k] >= r[j] * x[m, k, j])
                    
        # 7. Sequencing and Completion time
        for m in range(M):
            for k in range(K):
                model.addConstr(C[m, k] == S[m, k] + P[m, k])
                if k > 0:
                    model.addConstr(S[m, k] >= C[m, k-1])
                    
        # 8. Objective linking
        for m in range(M):
            for k in range(K):
                for j in range(N):
                    model.addConstr(Z[j] >= C[m, k] - H * (1 - x[m, k, j]))
                    
        model.setObjective(gp.quicksum(Z[j] for j in range(N)), GRB.MINIMIZE)
        
        # Inject Greedy Start
        greedy_by_m = defaultdict(list)
        for b in best_greedy_sol:
            greedy_by_m[b[0]].append(b)
            
        for m in range(M):
            batches = sorted(greedy_by_m[m], key=lambda i: i[1])
            for k in range(K):
                assigned_set = set(batches[k][3]) if k < len(batches) else set()
                for j in range(N):
                    x[m, k, j].Start = 1.0 if job_id[j] in assigned_set else 0.0
                    
        cb_data = MIPCallbackData(logger, best_greedy_obj)
        model._cb_data = cb_data
        
        model.optimize(mip_callback)
        
        if model.SolCount > 0:
            sol_tuples = []
            for m in range(M):
                for k in range(K):
                    assigned = [job_id[j] for j in range(N) if x[m, k, j].X > 0.5]
                    if assigned:
                        sol_tuples.append((m, S[m, k].X, 0, assigned))
                        
    output_dict = generate_final_output(sol_tuples, M, job_id_to_job)
    
    with open(args.solution_path, 'w') as f:
        json.dump(output_dict, f, indent=2)

if __name__ == "__main__":
    main()