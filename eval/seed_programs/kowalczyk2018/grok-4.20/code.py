import json
import argparse
import time
import sys
from itertools import combinations
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    m = instance['m']
    jobs = instance['jobs']
    
    # jobs are already sorted by nonincreasing w/p
    p = [job['p'] for job in jobs]
    w = [job['w'] for job in jobs]
    orig_indices = [job['j'] for job in jobs]
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    best_obj = float('inf')
    best_assignment = None
    best_orders = None
    
    # Heuristic: list scheduling in WSPT order, assign to machine with smallest current load
    machine_loads = [0] * m
    machine_jobs = [[] for _ in range(m)]
    
    for i in range(n):
        # assign to machine with smallest current completion time
        min_load_idx = machine_loads.index(min(machine_loads))
        machine_jobs[min_load_idx].append(i)
        machine_loads[min_load_idx] += p[i]
    
    # For each machine, order by original WSPT (already sorted globally, so preserve relative order)
    for k in range(m):
        machine_jobs[k].sort()  # smaller index has higher w/p
    
    current_obj = 0
    machine_costs = []
    machine_schedules = []
    
    for k in range(m):
        sched = machine_jobs[k]
        if not sched:
            machine_costs.append(0)
            machine_schedules.append([])
            continue
        completion = 0
        sched_cost = 0
        schedule_list = []
        for idx in sched:
            completion += p[idx]
            sched_cost += w[idx] * completion
            schedule_list.append({
                "j_orig": orig_indices[idx],
                "p": p[idx],
                "w": w[idx],
                "C_j": completion
            })
        current_obj += sched_cost
        machine_costs.append(sched_cost)
        machine_schedules.append(schedule_list)
    
    best_obj = current_obj
    best_assignment = machine_jobs
    best_orders = [list(range(len(s))) for s in machine_jobs]  # already ordered
    if logger:
        logger.log(best_obj)
    
    # Try to improve with local search if time allows
    elapsed = time.time() - start_time
    if elapsed < args.time_limit and n <= 25:  # small n, try MIP
        try:
            model = gp.Model("P||sum_wCj")
            model.Params.TimeLimit = max(1, args.time_limit - elapsed - 2)
            model.Params.LogToConsole = 0
            model.Params.Threads = 1
            
            # Binary variables: job to machine
            x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
            
            # Completion time variables
            C = model.addVars(n, vtype=GRB.CONTINUOUS, name="C", lb=0)
            
            # Objective
            model.setObjective(gp.quicksum(w[i] * C[i] for i in range(n)), GRB.MINIMIZE)
            
            # Each job assigned to exactly one machine
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, k] for k in range(m)) == 1)
            
            # Linearize completion times with big-M
            M = sum(p) + 1
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    for k in range(m):
                        # if both on same machine and i before j
                        # we use position-based or pairwise ordering
                        pass
            
            # Better formulation: use pairwise ordering variables
            # y_ijk = 1 if job i precedes job j on machine k
            if n <= 18:  # only for very small n
                y = model.addVars(n, n, m, vtype=GRB.BINARY, name="y")
                
                for i in range(n):
                    for k in range(m):
                        model.addConstr(C[i] >= p[i] * x[i, k])
                
                for i in range(n):
                    for j in range(i+1, n):
                        for k in range(m):
                            # if both on k, then one must precede the other
                            model.addConstr(y[i,j,k] + y[j,i,k] >= x[i,k] + x[j,k] - 1)
                            model.addConstr(y[i,j,k] <= x[i,k])
                            model.addConstr(y[i,j,k] <= x[j,k])
                            model.addConstr(y[j,i,k] <= x[i,k])
                            model.addConstr(y[j,i,k] <= x[j,k])
                            
                            model.addConstr(C[i] + p[j] - M*(1 - y[i,j,k]) <= C[j])
                            model.addConstr(C[j] + p[i] - M*(1 - y[j,i,k]) <= C[i])
                
                for i in range(n):
                    for k in range(m):
                        model.addConstr(C[i] <= M * x[i,k] + (1-x[i,k]) * M * 100)
            
            else:
                # Use position based formulation
                max_pos = n
                z = model.addVars(n, m, max_pos, vtype=GRB.BINARY, name="z")  # z_ikp = 1 if job i in position p on machine k
                
                for i in range(n):
                    for k in range(m):
                        model.addConstr(gp.quicksum(z[i,k,p] for p in range(max_pos)) == x[i,k])
                
                for k in range(m):
                    for pos in range(max_pos):
                        model.addConstr(gp.quicksum(z[i,k,pos] for i in range(n)) <= 1)
                
                # Completion time constraints
                for i in range(n):
                    model.addConstr(C[i] == gp.quicksum(
                        p[i] * z[i,k,pos] * (pos+1) + 
                        gp.quicksum(p[j] * z[j,k,r] for j in range(n) for r in range(pos)) * z[i,k,pos]
                        for k in range(m) for pos in range(max_pos)
                    ))
                    # This is nonlinear, need to linearize further - too complicated
                
                # Skip MIP for larger instances, stick with heuristic for speed/quality trade-off
                model = None
            
            if model is not None and n <= 12:  # only solve MIP for very small instances
                model.optimize()
                
                if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
                    new_obj = model.objVal
                    if new_obj < best_obj - 1e-4:
                        best_obj = new_obj
                        # extract solution...
                        best_assignment = [[] for _ in range(m)]
                        for i in range(n):
                            for k in range(m):
                                if x[i,k].X > 0.5:
                                    best_assignment[k].append(i)
                                    break
                        for k in range(m):
                            best_assignment[k].sort()  # WSPT order
                        if logger:
                            logger.log(best_obj)
        except:
            pass  # fall back to heuristic solution
    
    # Final solution construction using best found
    final_machines = []
    final_objective = 0.0
    machine_loads = [0] * m
    machine_costs = []
    
    for k in range(m):
        sched = best_assignment[k] if best_assignment is not None else machine_jobs[k]
        if not sched:
            final_machines.append({
                "schedule_cost": 0,
                "jobs": []
            })
            continue
        
        # ensure WSPT order within machine
        sched_sorted = sorted(sched)  # since globally sorted by w/p, this preserves order
        
        completion = 0
        sched_cost = 0
        job_list = []
        
        for idx in sched_sorted:
            completion += p[idx]
            weighted = w[idx] * completion
            sched_cost += weighted
            job_list.append({
                "j_orig": orig_indices[idx],
                "p": p[idx],
                "w": w[idx],
                "C_j": completion
            })
        
        final_objective += sched_cost
        final_machines.append({
            "schedule_cost": sched_cost,
            "jobs": job_list
        })
    
    solution = {
        "objective_value": float(final_objective),
        "machines": final_machines
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    # Final log of best
    if logger and best_obj < float('inf'):
        logger.log(best_obj)

if __name__ == "__main__":
    main()