import json
import argparse
import time
import sys
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB
import itertools

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    m = instance['m']
    T = instance['T']
    jobs = instance['jobs']
    
    # Sort jobs by id to ensure consistent ordering
    jobs = sorted(jobs, key=lambda x: x['id'])
    job_ids = [j['id'] for j in jobs]
    p = [j['p'] for j in jobs]
    d = [j['d'] for j in jobs]
    w = [j['w'] for j in jobs]
    
    total_p = sum(p)
    max_t = min(T, total_p)
    
    # Create model
    model = gp.Model("ParallelMachineWeightedTardiness")
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.LogToConsole, 0)
    
    # Variables
    # Binary: x[i,k] = 1 if job i starts at time k
    x = {}
    for i in range(n):
        for k in range(max_t + 1):
            if k + p[i] <= max_t:
                x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")
    
    # tardiness variables
    tard = {}
    for i in range(n):
        tard[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tard_{i}")
    
    # Objective
    obj = gp.quicksum(w[i] * tard[i] for i in range(n))
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    # Each job assigned exactly once
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(max_t + 1) if (i, k) in x) == 1)
    
    # No overlapping on machines: at most m jobs running at any time
    for t in range(max_t):
        active = []
        for i in range(n):
            for k in range(max(0, t - p[i] + 1), t + 1):
                if (i, k) in x:
                    active.append(x[i, k])
        if active:
            model.addConstr(gp.quicksum(active) <= m)
    
    # Tardiness definition
    for i in range(n):
        completion_expr = gp.quicksum((k + p[i]) * x[i, k] for k in range(max_t + 1) if (i, k) in x)
        model.addConstr(tard[i] >= completion_expr - d[i])
    
    # Callback for logging incumbents
    incumbent_obj = float('inf')
    def callback(model, where):
        nonlocal incumbent_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val < incumbent_obj - 1e-6:
                incumbent_obj = obj_val
                if logger:
                    logger.log(float(obj_val))
    
    # Optimize
    model.optimize(callback)
    
    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL]:
        start_times = {}
        for i in range(n):
            for k in range(max_t + 1):
                if (i, k) in x and x[i, k].X > 0.5:
                    start_times[str(job_ids[i])] = float(k)
                    break
        objective_value = float(model.ObjVal)
    else:
        # Fallback: use list scheduling by WSPT if no solution found
        # This should rarely happen
        sorted_jobs = sorted(range(n), key=lambda i: w[i] / p[i], reverse=True)
        machine_loads = [0] * m
        start_times_dict = {}
        for idx in sorted_jobs:
            mach = min(range(m), key=lambda mm: machine_loads[mm])
            start_times_dict[str(job_ids[idx])] = float(machine_loads[mach])
            machine_loads[mach] += p[idx]
        
        # Compute objective for this schedule
        total_tard = 0
        for idx in range(n):
            jid = job_ids[idx]
            s = start_times_dict[str(jid)]
            c = s + p[idx]
            total_tard += w[idx] * max(0, c - d[idx])
        objective_value = float(total_tard)
        start_times = start_times_dict
    
    # Write solution
    solution = {
        "objective_value": objective_value,
        "schedule": start_times
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()