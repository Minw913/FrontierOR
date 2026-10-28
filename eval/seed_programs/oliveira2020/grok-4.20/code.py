import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

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

    jobs = instance['jobs']
    n = instance['parameters']['n']
    m = instance['parameters']['m']
    time_horizon = instance['time_horizon']
    p = [j['processing_time'] for j in jobs]
    w = [j['weight'] for j in jobs]
    d = [j['due_date'] for j in jobs]
    job_ids = [j['id'] for j in jobs]

    # Model
    model = gp.Model("ParallelMachineWeightedTardiness")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('LogToConsole', 0)

    # Variables
    # Assignment to machines
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

    # Start time
    s = model.addVars(n, vtype=GRB.INTEGER, lb=0, ub=time_horizon, name="s")

    # Tardiness
    T = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="T")

    # Sequencing variables: y_ijk = 1 if job i precedes job j on machine k
    y = model.addVars(n, n, m, vtype=GRB.BINARY, name="y")

    # Objective
    obj = gp.quicksum(w[i] * T[i] for i in range(n))
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # Each job assigned to exactly one machine
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) == 1, name=f"assign_{i}")

    # Tardiness definition
    for i in range(n):
        model.addConstr(T[i] >= s[i] + p[i] - d[i], name=f"tardiness_{i}")
        model.addConstr(T[i] >= 0, name=f"tardiness_nonneg_{i}")

    # No overlap on same machine
    bigM = time_horizon + 1
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(m):
                model.addConstr(
                    s[i] + p[i] <= s[j] + bigM * (3 - x[i,k] - x[j,k] - y[i,j,k]),
                    name=f"seq1_{i}_{j}_{k}"
                )
                model.addConstr(
                    s[j] + p[j] <= s[i] + bigM * (2 - x[i,k] - x[j,k] + y[i,j,k]),
                    name=f"seq2_{i}_{j}_{k}"
                )

    # y only active when both on same machine
    for i in range(n):
        for j in range(i+1, n):
            for k in range(m):
                model.addConstr(y[i,j,k] + y[j,i,k] <= x[i,k] + x[j,k] - 1 + (1 - x[i,k]) + (1 - x[j,k]), 
                               name=f"y_active_{i}_{j}_{k}")
                # Antisymmetry when both assigned
                model.addConstr(y[i,j,k] + y[j,i,k] <= 1, name=f"anti_{i}_{j}_{k}")

    # At most one ordering
    for i in range(n):
        for j in range(i+1, n):
            for k in range(m):
                model.addConstr(y[i,j,k] + y[j,i,k] <= x[i,k] + x[j,k] - 1, name=f"order_{i}_{j}_{k}")

    # Symmetry breaking: order jobs by index on same machine when possible
    for i in range(n):
        for j in range(i+1, n):
            for k in range(m):
                if p[i] >= p[j]:  # prefer smaller index first if possible
                    model.addConstr(y[i,j,k] >= x[i,k] + x[j,k] - 1, name=f"sym_{i}_{j}_{k}")

    # Callback for incumbent logging
    best_obj = float('inf')
    solution_dicts = []

    def callback(model, where):
        nonlocal best_obj, solution_dicts
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj_val < best_obj - 1e-6:
                best_obj = obj_val
                # Extract solution
                sol_x = model.cbGetSolution(x)
                sol_s = model.cbGetSolution(s)
                
                schedule = []
                machine_jobs = [[] for _ in range(m)]
                
                for i in range(n):
                    for k in range(m):
                        if sol_x[i,k] > 0.5:
                            machine_jobs[k].append((sol_s[i], i))
                            break
                
                # Build prev_job links
                for k in range(m):
                    machine_jobs[k].sort()  # sort by start time
                    prev = 0
                    for start_t, i_idx in machine_jobs[k]:
                        job_id = job_ids[i_idx]
                        completion = int(start_t + p[i_idx])
                        tard = max(0, completion - d[i_idx])
                        wtard = w[i_idx] * tard
                        
                        schedule.append({
                            "machine_id": k,
                            "prev_job": prev,
                            "job": job_id,
                            "start_time": int(start_t),
                            "completion_time": completion,
                            "weighted_tardiness": float(wtard)
                        })
                        prev = job_id
                
                # Sort schedule by job id for consistent output
                schedule.sort(key=lambda x: x["job"])
                
                sol_dict = {
                    "objective_value": float(obj_val),
                    "schedule": schedule
                }
                
                solution_dicts.append(sol_dict)
                if logger:
                    logger.log_solution(float(obj_val), sol_dict)

    model.optimize(callback)

    # Final solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
        if model.SolCount > 0:
            obj_val = model.objVal
            x_sol = model.getAttr('X', x)
            s_sol = model.getAttr('X', s)
            
            schedule = []
            machine_jobs = [[] for _ in range(m)]
            
            for i in range(n):
                for k in range(m):
                    if x_sol[i,k] > 0.5:
                        machine_jobs[k].append((s_sol[i], i))
                        break
            
            for k in range(m):
                machine_jobs[k].sort()
                prev = 0
                for start_t, i_idx in machine_jobs[k]:
                    job_id = job_ids[i_idx]
                    completion = int(start_t + p[i_idx])
                    tard = max(0, completion - d[i_idx])
                    wtard = float(w[i_idx] * tard)
                    
                    schedule.append({
                        "machine_id": int(k),
                        "prev_job": int(prev),
                        "job": int(job_id),
                        "start_time": int(start_t),
                        "completion_time": int(completion),
                        "weighted_tardiness": wtard
                    })
                    prev = job_id
            
            schedule.sort(key=lambda x: x["job"])
            
            final_solution = {
                "objective_value": float(obj_val),
                "schedule": schedule
            }
            
            # Log final if not already logged via callback
            if logger and (not solution_dicts or abs(solution_dicts[-1]["objective_value"] - obj_val) > 1e-4):
                logger.log_solution(float(obj_val), final_solution)
        else:
            # No feasible solution found - return dummy with high objective
            final_solution = {
                "objective_value": 1e10,
                "schedule": []
            }
    else:
        final_solution = {
            "objective_value": 1e10,
            "schedule": []
        }

    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)

if __name__ == "__main__":
    main()