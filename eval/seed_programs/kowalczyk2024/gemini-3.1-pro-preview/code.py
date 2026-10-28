import sys
import os
import json
import time
import math
import random
import argparse
import gurobipy as gp

# Utility module as specified
from solution_logger import SolutionLogger

def get_machine_cost(seq, p, d, w):
    cost = 0
    t = 0
    for j in seq:
        t += p[j]
        if t > d[j]:
            cost += w[j] * (t - d[j])
    return cost

def evaluate_schedule(seqs, p, d, w):
    return sum(get_machine_cost(seq, p, d, w) for seq in seqs)

def construct_baseline(rule, n, m, p, d, w):
    seqs = [[] for _ in range(m)]
    machine_times = [0] * m
    unassigned = set(range(n))
    
    if rule == 'WSPT':
        sorted_jobs = sorted(range(n), key=lambda j: w[j]/p[j], reverse=True)
    elif rule == 'EDD':
        sorted_jobs = sorted(range(n), key=lambda j: d[j])
    else:  # MDD
        sorted_jobs = list(range(n))

    while unassigned:
        m_idx = min(range(m), key=lambda x: machine_times[x])
        t = machine_times[m_idx]
        
        if rule == 'MDD':
            best_j = min(unassigned, key=lambda j: max(t + p[j], d[j]))
        else:
            for j in sorted_jobs:
                if j in unassigned:
                    best_j = j
                    break
                    
        seqs[m_idx].append(best_j)
        machine_times[m_idx] += p[best_j]
        unassigned.remove(best_j)
        
    return seqs

def grasp_construct(alpha, n, m, p, d, w):
    seqs = [[] for _ in range(m)]
    machine_times = [0] * m
    unassigned = set(range(n))
    
    while unassigned:
        m_idx = min(range(m), key=lambda x: machine_times[x])
        t = machine_times[m_idx]
        
        avg_p = sum(p[j] for j in unassigned) / len(unassigned)
        K = random.uniform(0.5, 3.0)
        
        indices = []
        for j in unassigned:
            slack = max(0, d[j] - p[j] - t)
            val = (w[j] / p[j]) * math.exp(-slack / (K * avg_p)) if avg_p > 0 else (w[j] / p[j])
            indices.append((val, j))
            
        max_idx = max(indices, key=lambda x: x[0])[0]
        min_idx = min(indices, key=lambda x: x[0])[0]
        thresh = max_idx - alpha * (max_idx - min_idx)
        
        rcl = [x[1] for x in indices if x[0] >= thresh]
        chosen = random.choice(rcl)
        
        seqs[m_idx].append(chosen)
        machine_times[m_idx] += p[chosen]
        unassigned.remove(chosen)
        
    return seqs

def local_search(solution, m, p, d, w):
    current_cost = evaluate_schedule(solution, p, d, w)
    improved = True
    
    while improved:
        improved = False
        # Phase 1: Shift jobs
        for m1 in range(m):
            seq1 = solution[m1]
            for i in range(len(seq1)):
                j_id = seq1[i]
                new_seq1 = seq1[:i] + seq1[i+1:]
                cost1_base = get_machine_cost(seq1, p, d, w)
                cost1_new = get_machine_cost(new_seq1, p, d, w)
                delta_m1_removal = cost1_new - cost1_base
                
                for m2 in range(m):
                    target_len = len(new_seq1) + 1 if m1 == m2 else len(solution[m2]) + 1
                    seq2 = solution[m2]
                    cost2_base = get_machine_cost(seq2, p, d, w) if m1 != m2 else cost1_base
                    
                    for k in range(target_len):
                        if m1 == m2 and k == i: continue
                        if m1 == m2:
                            test_seq = new_seq1[:k] + [j_id] + new_seq1[k:]
                            delta = get_machine_cost(test_seq, p, d, w) - cost1_base
                            if delta < -1e-6:
                                solution[m1] = test_seq
                                current_cost += delta
                                improved = True
                                break
                        else:
                            test_seq2 = seq2[:k] + [j_id] + seq2[k:]
                            delta = delta_m1_removal + get_machine_cost(test_seq2, p, d, w) - cost2_base
                            if delta < -1e-6:
                                solution[m1] = new_seq1
                                solution[m2] = test_seq2
                                current_cost += delta
                                improved = True
                                break
                    if improved: break
                if improved: break
        
        if improved: continue
        
        # Phase 2: Swap jobs
        for m1 in range(m):
            seq1 = solution[m1]
            for i in range(len(seq1)):
                for m2 in range(m1, m):
                    start_j = i + 1 if m1 == m2 else 0
                    seq2 = solution[m2]
                    for j in range(start_j, len(seq2)):
                        j1 = seq1[i]
                        j2 = seq2[j]
                        if m1 == m2:
                            test_seq = seq1[:]
                            test_seq[i], test_seq[j] = test_seq[j], test_seq[i]
                            delta = get_machine_cost(test_seq, p, d, w) - get_machine_cost(seq1, p, d, w)
                            if delta < -1e-6:
                                solution[m1] = test_seq
                                current_cost += delta
                                improved = True
                                break
                        else:
                            test_seq1 = seq1[:]
                            test_seq2 = seq2[:]
                            test_seq1[i] = j2
                            test_seq2[j] = j1
                            delta = get_machine_cost(test_seq1, p, d, w) + get_machine_cost(test_seq2, p, d, w) \
                                    - (get_machine_cost(seq1, p, d, w) + get_machine_cost(seq2, p, d, w))
                            if delta < -1e-6:
                                solution[m1] = test_seq1
                                solution[m2] = test_seq2
                                current_cost += delta
                                improved = True
                                break
                    if improved: break
                if improved: break

    return solution, current_cost

def build_output_schedule(seqs, original_ids, p):
    job_starts = {}
    for seq in seqs:
        t = 0
        for j in seq:
            job_starts[str(original_ids[j])] = t
            t += p[j]
    return job_starts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    program_start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load problem data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data["n"]
    m = data["m"]
    planning_horizon = data.get("T", sys.maxsize)
    
    original_ids, p, d, w = [], [], [], []
    for job in data["jobs"]:
        original_ids.append(job["id"])
        p.append(job["p"])
        d.append(job["d"])
        w.append(job["w"])
        
    sum_p = sum(p)

    best_solution = None
    best_cost = float('inf')

    # Base heuristic constructions
    for base in ['WSPT', 'EDD', 'MDD']:
        sol = construct_baseline(base, n, m, p, d, w)
        sol, cost = local_search(sol, m, p, d, w)
        if cost < best_cost:
            best_cost = cost
            best_solution = [seq[:] for seq in sol]
            if logger:
                logger.log(best_cost)
                
    if best_cost == 0:
        res = {
            "objective_value": float(best_cost),
            "schedule": build_output_schedule(best_solution, original_ids, p)
        }
        with open(args.solution_path, 'w') as f:
            json.dump(res, f, indent=2)
        sys.exit(0)

    # GRASP Algorithm phase
    heuristic_duration = min(float(args.time_limit) * 0.15, 20.0)
    while True:
        elapsed = time.time() - program_start_time
        if elapsed > heuristic_duration or elapsed > args.time_limit - 1.0:
            break
            
        alpha = random.uniform(0.1, 0.5)
        sol = grasp_construct(alpha, n, m, p, d, w)
        sol, cost = local_search(sol, m, p, d, w)
        
        if cost < best_cost:
            best_cost = cost
            best_solution = [seq[:] for seq in sol]
            if logger:
                logger.log(best_cost)

    remaining_time = args.time_limit - (time.time() - program_start_time)
    
    # Mathematical Modeling Configuration
    if remaining_time > 5:
        # Check TI dimension
        time_domains = {}
        for j in range(n):
            limit = min(planning_horizon, sum_p, d[j] + int(math.floor(best_cost / w[j])))
            time_domains[j] = list(range(p[j], limit + 1))
            
        total_vars = sum(len(time_domains[j]) for j in range(n))
        use_ti = total_vars <= 500000

        try:
            env = gp.Env(empty=True)
            env.setParam("LogToConsole", 0)
            env.start()
            model = gp.Model("WeightedTardiness", env=env)
            
            model.setParam('TimeLimit', remaining_time)
            model.setParam('MIPGap', 1e-4)
            model.setParam('Threads', 1)
            
            model._best_obj = best_cost
            model._global_best_cost = best_cost
            model._global_best_seqs = [seq[:] for seq in best_solution]
            
            if use_ti:
                x_ti = {}
                x_ti_list = []
                for j in range(n):
                    for t in time_domains[j]:
                        var = model.addVar(vtype=gp.GRB.BINARY)
                        x_ti[j, t] = var
                        x_ti_list.append(var)
                
                model._x_vars_list = x_ti_list
                model._time_domains = time_domains
                
                # Assign MIP Start
                for seq in best_solution:
                    timer = 0
                    for j in seq:
                        timer += p[j]
                        if timer in time_domains[j]:
                            x_ti[j, timer].Start = 1
                
                # Constraints
                for j in range(n):
                    model.addConstr(gp.quicksum(x_ti[j, t] for t in time_domains[j]) == 1)
                    
                max_T = max((time_domains[j][-1] for j in range(n) if time_domains[j]), default=0)
                
                exprs = {t: [] for t in range(1, max_T + 1)}
                for j in range(n):
                    for t in time_domains[j]:
                        s_min = max(1, t - p[j] + 1)
                        s_max = t
                        for s in range(s_min, s_max + 1):
                            if s <= max_T:
                                exprs[s].append(x_ti[j, t])
                                
                for t in range(1, max_T + 1):
                    if len(exprs[t]) > 0:
                        model.addConstr(gp.quicksum(exprs[t]) <= m)
                        
                obj_expr = gp.LinExpr()
                for j in range(n):
                    for t in time_domains[j]:
                        tardiness = max(0, t - d[j])
                        if tardiness > 0:
                            obj_expr.add(x_ti[j, t], w[j] * tardiness)
                            
                model.setObjective(obj_expr, gp.GRB.MINIMIZE)

                def callback_ti(m_cb, where):
                    if where == gp.GRB.Callback.MIPSOL:
                        obj = m_cb.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                        if obj < m_cb._best_obj - 1e-5:
                            m_cb._best_obj = obj
                            try:
                                sol = m_cb.cbGetSolution(m_cb._x_vars_list)
                                job_intervals = []
                                count = 0
                                for j_idx in range(n):
                                    for t_val in m_cb._time_domains[j_idx]:
                                        if sol[count] > 0.5:
                                            job_intervals.append((j_idx, t_val - p[j_idx], t_val))
                                            break
                                        count += 1
                                        
                                if len(job_intervals) == n:
                                    job_intervals.sort(key=lambda x: x[1])
                                    seqs_cb = [[] for _ in range(m)]
                                    ends = [0] * m
                                    for j_idx, start_v, _ in job_intervals:
                                        valid = [idx for idx in range(m) if ends[idx] <= start_v + 1e-4]
                                        idx = max(valid, key=lambda idx: ends[idx]) if valid else min(range(m), key=lambda idx: ends[idx])
                                        seqs_cb[idx].append(j_idx)
                                        ends[idx] += p[j_idx]
                                        
                                    cost_cb = evaluate_schedule(seqs_cb, p, d, w)
                                    if cost_cb < m_cb._global_best_cost:
                                        m_cb._global_best_cost = cost_cb
                                        m_cb._global_best_seqs = [s[:] for s in seqs_cb]
                                        if logger:
                                            logger.log(cost_cb)
                            except Exception:
                                pass

                model.optimize(callback_ti)
                
            else:
                x = {}
                x_vars_list = []
                for u in range(n + 1):
                    for v in range(n):
                        if u != v:
                            var = model.addVar(vtype=gp.GRB.BINARY)
                            x[u, v] = var
                            x_vars_list.append(var)
                            
                model._x_vars_list = x_vars_list
                
                C_var = {}
                T_var = {}
                for v in range(n):
                    C_var[v] = model.addVar(lb=p[v], ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
                    T_var[v] = model.addVar(lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
                    
                for seq in best_solution:
                    u = n
                    for v in seq:
                        x[u, v].Start = 1
                        u = v
                        
                model.addConstr(gp.quicksum(x[n, v] for v in range(n)) <= m)
                limit_lookup = {u: min(planning_horizon, sum_p, d[u] + int(math.floor(best_cost / w[u]))) for u in range(n)}
                
                for v in range(n):
                    model.addConstr(gp.quicksum(x[u, v] for u in range(n+1) if u != v) == 1)
                    model.addConstr(gp.quicksum(x[v, tmp_w] for tmp_w in range(n) if tmp_w != v) <= 1)
                    model.addConstr(T_var[v] >= C_var[v] - d[v])
                    
                    for u in range(n):
                        if u != v:
                            model.addConstr(C_var[v] >= C_var[u] + p[v] - limit_lookup[u] * (1 - x[u, v]))
                            
                model.setObjective(gp.quicksum(w[v] * T_var[v] for v in range(n)), gp.GRB.MINIMIZE)
                
                def callback_seq(m_cb, where):
                    if where == gp.GRB.Callback.MIPSOL:
                        obj = m_cb.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                        if obj < m_cb._best_obj - 1e-5:
                            m_cb._best_obj = obj
                            try:
                                sol = m_cb.cbGetSolution(m_cb._x_vars_list)
                                count = 0
                                adj_mult = {u_idx: [] for u_idx in range(n + 1)}
                                for u_idx in range(n + 1):
                                    for v_idx in range(n):
                                        if u_idx != v_idx:
                                            if sol[count] > 0.5:
                                                adj_mult[u_idx].append(v_idx)
                                            count += 1
                                            
                                starts = adj_mult[n]
                                seqs_cb = [[] for _ in range(m)]
                                for i, start_node in enumerate(starts):
                                    if i >= m: break
                                    curr = start_node
                                    while True:
                                        seqs_cb[i].append(curr)
                                        if len(adj_mult[curr]) > 0:
                                            curr = adj_mult[curr][0]
                                        else:
                                            break
                                            
                                cost_cb = evaluate_schedule(seqs_cb, p, d, w)
                                if cost_cb < m_cb._global_best_cost:
                                    m_cb._global_best_cost = cost_cb
                                    m_cb._global_best_seqs = [s[:] for s in seqs_cb]
                                    if logger:
                                        logger.log(cost_cb)
                            except Exception:
                                pass
                                
                model.optimize(callback_seq)
                
            if model._global_best_cost < best_cost:
                best_cost = model._global_best_cost
                best_solution = model._global_best_seqs
                
        except Exception as e:
            pass
            
    res = {
        "objective_value": float(best_cost),
        "schedule": build_output_schedule(best_solution, original_ids, p)
    }
    with open(args.solution_path, 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    main()