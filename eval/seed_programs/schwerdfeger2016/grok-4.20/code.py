import json
import argparse
import time
import sys
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    m = instance['num_machines']
    n = instance['num_jobs']
    p = instance['processing_times']
    total_p = sum(p)
    avg = total_p / m

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    best_obj = float('inf')
    best_assignment = None
    best_ct = None
    best_norm = None

    # Greedy LPT for initial solution
    machines = [0.0] * m
    assignment = [[] for _ in range(m)]
    for j in range(n):
        min_idx = machines.index(min(machines))
        machines[min_idx] += p[j]
        assignment[min_idx].append(j)
    
    current_ct = machines[:]
    sqsum = sum(x * x for x in current_ct)
    current_obj = math.sqrt(sqsum) / avg
    if current_obj < best_obj:
        best_obj = current_obj
        best_assignment = [sorted(assignment[i]) for i in range(m)]
        best_ct = [float(x) for x in current_ct]
        best_norm = float(current_obj)
        if logger:
            logger.log(best_norm)
    
    # If very small instance, solve exactly with MIP
    if n <= 20 and m <= 5 and (time.time() - start_time < args.time_limit * 0.8):
        try:
            model = gp.Model("squared_dev")
            model.setParam(GRB.Param.TimeLimit, max(1, args.time_limit - (time.time() - start_time)))
            model.setParam(GRB.Param.Threads, 1)
            model.setParam(GRB.Param.MIPGap, 1e-4)
            model.setParam(GRB.Param.LogToConsole, 0)
            
            x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
            
            ct = model.addVars(m, lb=0.0, name="ct")
            for i in range(m):
                model.addConstr(ct[i] == gp.quicksum(p[j] * x[j, i] for j in range(n)))
            
            sq = model.addVars(m, lb=0.0, name="sq")
            for i in range(m):
                model.addConstr(sq[i] >= ct[i] * ct[i])
            
            model.setObjective(gp.quicksum(sq[i] for i in range(m)), GRB.MINIMIZE)
            
            for j in range(n):
                model.addConstr(gp.quicksum(x[j, i] for i in range(m)) == 1)
            
            # Warm start
            for j in range(n):
                for i in range(m):
                    x[j, i].start = 0
            for i in range(m):
                for j_idx in assignment[i]:
                    x[j_idx, i].start = 1
            
            model.optimize()
            
            if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and model.SolCount > 0:
                new_ct = [ct[i].X for i in range(m)]
                new_sqsum = sum(x * x for x in new_ct)
                new_norm = math.sqrt(new_sqsum) / avg
                if new_norm < best_obj - 1e-8:
                    best_obj = new_norm
                    new_assign = [[] for _ in range(m)]
                    for j in range(n):
                        for i in range(m):
                            if x[j, i].X > 0.5:
                                new_assign[i].append(j)
                                break
                    best_assignment = [sorted(lst) for lst in new_assign]
                    best_ct = [float(round(c, 6)) for c in new_ct]
                    best_norm = float(new_norm)
                    if logger:
                        logger.log(best_norm)
        except:
            pass

    # Local search improvement - swap and move
    improved = True
    current_assignment = [lst[:] for lst in best_assignment]
    current_ct_list = [sum(p[j] for j in lst) for lst in current_assignment]
    current_sq = sum(float(x)**2 for x in current_ct_list)
    
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    iter_limit = 10000 if n > 30 else 50000
    it = 0
    
    while improved and it < iter_limit and (time.time() - start_time < args.time_limit - 0.5):
        it += 1
        improved = False
        # Try moving a job from one machine to another
        for i1 in range(m):
            for i2 in range(m):
                if i1 == i2 or not current_assignment[i1]:
                    continue
                for jj_idx, j in enumerate(current_assignment[i1]):
                    delta = p[j]
                    new_c1 = current_ct_list[i1] - delta
                    new_c2 = current_ct_list[i2] + delta
                    new_sq_total = current_sq - current_ct_list[i1]**2 - current_ct_list[i2]**2 + new_c1**2 + new_c2**2
                    if new_sq_total < current_sq - 1e-6:
                        # accept move
                        current_assignment[i1].pop(jj_idx)
                        current_assignment[i2].append(j)
                        current_ct_list[i1] = new_c1
                        current_ct_list[i2] = new_c2
                        current_sq = new_sq_total
                        improved = True
                        new_norm_val = math.sqrt(current_sq) / avg
                        if new_norm_val < best_obj - 1e-9:
                            best_obj = new_norm_val
                            best_assignment = [sorted(lst[:]) for lst in current_assignment]
                            best_ct = [float(c) for c in current_ct_list]
                            best_norm = float(new_norm_val)
                            if logger:
                                logger.log(best_norm)
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
            
        # Try swapping two jobs between two machines
        for i1 in range(m):
            for i2 in range(i1+1, m):
                for j1 in current_assignment[i1]:
                    for j2 in current_assignment[i2]:
                        delta = p[j1] - p[j2]
                        new_c1 = current_ct_list[i1] - p[j1] + p[j2]
                        new_c2 = current_ct_list[i2] - p[j2] + p[j1]
                        new_sq_total = current_sq - current_ct_list[i1]**2 - current_ct_list[i2]**2 + new_c1**2 + new_c2**2
                        if new_sq_total < current_sq - 1e-6:
                            # perform swap
                            idx1 = current_assignment[i1].index(j1)
                            idx2 = current_assignment[i2].index(j2)
                            current_assignment[i1][idx1] = j2
                            current_assignment[i2][idx2] = j1
                            current_ct_list[i1] = new_c1
                            current_ct_list[i2] = new_c2
                            current_sq = new_sq_total
                            improved = True
                            new_norm_val = math.sqrt(current_sq) / avg
                            if new_norm_val < best_obj - 1e-9:
                                best_obj = new_norm_val
                                best_assignment = [sorted(lst[:]) for lst in current_assignment]
                                best_ct = [float(c) for c in current_ct_list]
                                best_norm = float(new_norm_val)
                                if logger:
                                    logger.log(best_norm)
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    # Final output
    if best_norm is None:
        # fallback
        best_norm = float(current_obj)
        best_assignment = [sorted(assignment[i]) for i in range(m)]
        best_ct = [float(machines[i]) for i in range(m)]
    
    output = {
        "objective_value": best_norm,
        "assignment": {},
        "machine_completion_times": {}
    }
    
    for i in range(m):
        mach_key = f"machine_{i}"
        output["assignment"][mach_key] = best_assignment[i]
        output["machine_completion_times"][mach_key] = best_ct[i]
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()