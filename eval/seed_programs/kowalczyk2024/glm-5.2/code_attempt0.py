import argparse
import json
import math
import sys
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def compute_objective(S, p, d, w, ids):
    obj = 0
    for i in ids:
        C = S[i] + p[i]
        tard = max(0, C - d[i])
        obj += w[i] * tard
    return obj


def build_solution_dict(S, ids):
    return {str(i): int(round(S[i])) for i in ids}


def atc_heuristic(n, m, p, d, w, ids, K=1.0):
    """ATC heuristic for identical parallel machines."""
    avg_p = sum(p[i] for i in ids) / n if n > 0 else 1.0
    scheduled = set()
    machine_free = [0] * m
    S = {}
    while len(scheduled) < n:
        t = min(machine_free)
        k = machine_free.index(t)
        best_jid = None
        best_val = -1.0
        for jid in ids:
            if jid in scheduled:
                continue
            slack = d[jid] - p[jid] - t
            if slack <= 0:
                val = (w[jid] / p[jid])
            else:
                val = (w[jid] / p[jid]) * math.exp(-slack / (K * avg_p))
            if val > best_val:
                best_val = val
                best_jid = jid
        if best_jid is None:
            break
        S[best_jid] = t
        machine_free[k] = t + p[best_jid]
        scheduled.add(best_jid)
    obj = compute_objective(S, p, d, w, ids)
    return obj, S


def get_machine_assignment(S, p, ids, m):
    """Reconstruct machine assignment from start times."""
    sorted_jobs = sorted(ids, key=lambda i: S[i])
    machine_end = [0] * m
    job_machine = {}
    machine_jobs = [[] for _ in range(m)]
    for jid in sorted_jobs:
        placed = False
        for k in range(m):
            if machine_end[k] <= S[jid]:
                job_machine[jid] = k
                machine_end[k] = S[jid] + p[jid]
                machine_jobs[k].append(jid)
                placed = True
                break
        if not placed:
            job_machine[jid] = 0
            machine_jobs[0].append(jid)
    return job_machine, machine_jobs


def local_search(S, p, d, w, ids, m, time_limit, start_time):
    """Insertion-based local search across machines."""
    job_machine, machine_jobs = get_machine_assignment(S, p, ids, m)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for jid in ids:
            if time.time() - start_time > time_limit:
                break
            orig_machine = job_machine[jid]
            orig_seq = [x for x in machine_jobs[orig_machine] if x != jid]
            for new_k in range(m):
                if new_k == orig_machine:
                    continue
                if time.time() - start_time > time_limit:
                    break
                new_seq = list(machine_jobs[new_k])
                for pos in range(len(new_seq) + 1):
                    trial_new = new_seq[:pos] + [jid] + new_seq[pos:]
                    # Build new S for affected machines only
                    new_S = dict(S)  # copy
                    t = 0
                    for jj in orig_seq:
                        new_S[jj] = t
                        t += p[jj]
                    t = 0
                    for jj in trial_new:
                        new_S[jj] = t
                        t += p[jj]
                    new_obj = compute_objective(new_S, p, d, w, ids)
                    old_obj = compute_objective(S, p, d, w, ids)
                    if new_obj < old_obj - 1e-9:
                        S = new_S
                        machine_jobs[orig_machine] = orig_seq
                        machine_jobs[new_k] = trial_new
                        job_machine[jid] = new_k
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()
    
    with open(args.instance_path) as f:
        instance = json.load(f)
    
    n = instance['n']
    m = instance['m']
    T = instance.get('T')
    jobs = instance['jobs']
    ids = [job['id'] for job in jobs]
    p = {job['id']: job['p'] for job in jobs}
    d = {job['id']: job['d'] for job in jobs}
    w = {job['id']: job['w'] for job in jobs}
    
    if n == 0:
        solution = {"objective_value": 0.0, "schedule": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(0.0, {})
        return
    
    # ---- Heuristic phase ----
    best_heur_obj = float('inf')
    best_heur_S = None
    K_values = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    for K in K_values:
        obj, S = atc_heuristic(n, m, p, d, w, ids, K=K)
        if obj < best_heur_obj:
            best_heur_obj = obj
            best_heur_S = S
    
    # Local search improvement
    ls_time_limit = min(15, max(2, args.time_limit // 4))
    best_heur_S = local_search(best_heur_S, p, d, w, ids, m,
                               ls_time_limit, start_time)
    best_heur_obj = compute_objective(best_heur_S, p, d, w, ids)
    
    if logger:
        logger.log_solution(best_heur_obj, build_solution_dict(best_heur_S, ids))
    
    best_obj = best_heur_obj
    best_S = {i: int(round(best_heur_S[i])) for i in ids}
    
    # ---- MIP phase for small/medium instances ----
    mip_threshold = 50
    if n <= mip_threshold:
        try:
            model = gp.Model()
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            remaining = args.time_limit - (time.time() - start_time) - 1
            model.Params.TimeLimit = max(remaining, 5)
            model.Params.OutputFlag = 0
            
            x = {}
            for i in range(n):
                for j in range(n):
                    if i != j:
                        x[ids[i], ids[j]] = model.addVar(vtype=GRB.BINARY,
                                                         name=f'x_{i}_{j}')
            
            C = model.addVars(ids, vtype=GRB.CONTINUOUS, name='C')
            Tard = model.addVars(ids, vtype=GRB.CONTINUOUS, lb=0, name='Tard')
            
            model.setObjective(gp.quicksum(w[i] * Tard[i] for i in ids),
                               GRB.MINIMIZE)
            
            for i in ids:
                model.addConstr(
                    C[i] == p[i] + gp.quicksum(p[j] * x[j, i]
                                               for j in ids if j != i))
            
            for i in ids:
                model.addConstr(Tard[i] >= C[i] - d[i])
            
            for i in range(n):
                for j in range(i + 1, n):
                    model.addConstr(x[ids[i], ids[j]] + x[ids[j], ids[i]] <= 1)
            
            # Transitivity constraints
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        model.addConstr(
                            x[ids[i], ids[j]] + x[ids[j], ids[k]]
                            - x[ids[i], ids[k]] <= 1)
            
            # At most m chains (machines)
            model.addConstr(
                gp.quicksum(x[ids[i], ids[j]]
                            for i in range(n) for j in range(n) if i != j)
                >= n - m)
            
            if T is not None:
                for i in ids:
                    model.addConstr(C[i] <= T)
            
            # MIP start from heuristic
            job_machine, _ = get_machine_assignment(best_S, p, ids, m)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if (job_machine.get(ids[i]) == job_machine.get(ids[j])
                                and best_S[ids[i]] < best_S[ids[j]]):
                            x[ids[i], ids[j]].Start = 1
                        else:
                            x[ids[i], ids[j]].Start = 0
            
            state = {'best_obj': best_obj, 'best_S': best_S}
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < state['best_obj'] - 1e-6:
                        xvals = model.cbGetSolution(x)
                        S_sol = {}
                        for i in ids:
                            S_sol[i] = int(round(
                                sum(p[j] * round(xvals[j, i])
                                    for j in ids if j != i)))
                        state['best_obj'] = obj
                        state['best_S'] = S_sol
                        if logger:
                            logger.log_solution(
                                obj, build_solution_dict(S_sol, ids))
            
            model.optimize(callback)
            
            best_obj = state['best_obj']
            best_S = state['best_S']
            
        except Exception as e:
            print(f"MIP failed: {e}", file=sys.stderr)
    
    # ---- Write final solution ----
    solution = {
        "objective_value": float(best_obj),
        "schedule": build_solution_dict(best_S, ids)
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()