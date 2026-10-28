import argparse
import json
import math
import time
import heapq

def compute_obj(machines, mu):
    sum_sq = sum((m - mu)**2 for m in machines)
    return math.sqrt(max(0.0, sum_sq)) / mu

def canonicalize_assignment(assignment, num_jobs):
    def get_min_idx(machine_jobs):
        return min(machine_jobs) if machine_jobs else num_jobs
    return sorted(assignment, key=get_min_idx)

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    
    args = parser.parse_args()
    
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Proceed to load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    num_machines = instance['num_machines']
    num_jobs = instance['num_jobs']
    processing_times = instance['processing_times']
    
    mu = sum(processing_times) / num_machines
    
    # Heuristic 1: Longest Processing Time (LPT) assignment using min-heap for speed
    machines_heap = [(0, i) for i in range(num_machines)]
    assignment = [[] for _ in range(num_machines)]
    machines = [0] * num_machines
    
    for j, p in enumerate(processing_times):
        load, m_idx = heapq.heappop(machines_heap)
        assignment[m_idx].append(j)
        machines[m_idx] += p
        heapq.heappush(machines_heap, (machines[m_idx], m_idx))
        
    initial_obj = compute_obj(machines, mu)
    if logger:
        logger.log(initial_obj)
        
    # Determine the maximum time allowed for the Local Search phase
    # Yield early if we can definitely fit a Gurobi optimization attempt.
    if num_jobs * num_machines <= 2000000:
        max_ls_time = min(float(args.time_limit) - 2.0, 10.0)
    else:
        max_ls_time = float(args.time_limit) - 2.0
    
    # Heuristic 2: Fast Local Search (Steepest Descent load balancing)
    improved = True
    while improved and (time.time() - start_time) < max_ls_time:
        improved = False
        sorted_m = sorted(range(len(machines)), key=lambda i: machines[i])
        
        # 1. Try single element move from heavily loaded to lightly loaded
        for heavily_loaded in reversed(sorted_m):
            if machines[heavily_loaded] <= mu:
                break
            for j in assignment[heavily_loaded]:
                pj = processing_times[j]
                for lightly_loaded in sorted_m:
                    if heavily_loaded == lightly_loaded or machines[lightly_loaded] >= machines[heavily_loaded]:
                        continue
                    
                    load_h = machines[heavily_loaded]
                    load_l = machines[lightly_loaded]
                    
                    diff_before = load_h**2 + load_l**2
                    diff_after = (load_h - pj)**2 + (load_l + pj)**2
                    
                    if diff_after < diff_before:
                        assignment[heavily_loaded].remove(j)
                        assignment[lightly_loaded].append(j)
                        machines[heavily_loaded] -= pj
                        machines[lightly_loaded] += pj
                        improved = True
                        break
                if improved: break
            if improved: break
            
        # 2. If single move failed, try swapping elements between machine pairs
        if not improved:
            for m1 in reversed(sorted_m):
                if (time.time() - start_time) > max_ls_time:
                    break
                for m2 in sorted_m:
                    if m1 == m2 or machines[m1] <= machines[m2]:
                        continue
                    for j1 in assignment[m1]:
                        pj1 = processing_times[j1]
                        for j2 in assignment[m2]:
                            pj2 = processing_times[j2]
                            if pj1 <= pj2:
                                continue # Only swap a larger job to a less loaded machine
                                
                            load1 = machines[m1]
                            load2 = machines[m2]
                            
                            diff_before = load1**2 + load2**2
                            diff_after = (load1 - pj1 + pj2)**2 + (load2 + pj1 - pj2)**2
                            if diff_after < diff_before:
                                assignment[m1].remove(j1)
                                assignment[m2].remove(j2)
                                assignment[m1].append(j2)
                                assignment[m2].append(j1)
                                machines[m1] += pj2 - pj1
                                machines[m2] += pj1 - pj2
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break
        
        if improved and logger:
            logger.log(compute_obj(machines, mu))

    # To satisfy future symmetry-breaking logic without breaking valid assignments
    assignment = canonicalize_assignment(assignment, num_jobs)
    machines = [sum(processing_times[j] for j in m) for m in assignment]

    # Gurobi optimization layer
    # Only if problem instance variables scale allows it, and we have minimum buffer time left
    if num_jobs * num_machines <= 2000000 and float(args.time_limit) - (time.time() - start_time) > 4.0:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            time_left = float(args.time_limit) - (time.time() - start_time) - 1.0
            
            if time_left > 0.1:
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.start()
                
                model = gp.Model(env=env)
                model.setParam("TimeLimit", max(0.1, time_left))
                model.setParam("Threads", 1)  # Strictly enforcing a single thread environment
                
                # Setup MIQP matching assignment objective behavior
                x = model.addVars(num_machines, num_jobs, vtype=GRB.BINARY, name="x")
                C = model.addVars(num_machines, vtype=GRB.CONTINUOUS, name="C")
                
                for j in range(num_jobs):
                    model.addConstr(x.sum('*', j) == 1)
                    
                for i in range(num_machines):
                    model.addConstr(C[i] == gp.quicksum(processing_times[j] * x[i, j] for j in range(num_jobs)))
                
                # The minimization proxy is mathematically equal to the final normalized standard
                model.setObjective(gp.quicksum(C[i] * C[i] for i in range(num_machines)), GRB.MINIMIZE)

                # Perfect identity symmetry breaking bounds
                for j in range(min(num_machines, num_jobs)):
                    for i in range(j + 1, num_machines):
                        model.addConstr(x[i, j] == 0)

                # Inject the superiorly robust LS assignments as initial states
                for i in range(num_machines):
                    for j in assignment[i]:
                        x[i, j].Start = 1
                
                # Enable accurate objective logging per intermediate convergence
                model._C = [C[i] for i in range(num_machines)]
                model._mu = mu
                model._logger = logger
                
                def cb(model, where):
                    if where == GRB.Callback.MIPSOL:
                        C_vals = model.cbGetSolution(model._C)
                        sum_sq_dev = sum((c - model._mu)**2 for c in C_vals)
                        true_obj = math.sqrt(max(0.0, sum_sq_dev)) / model._mu
                        if model._logger:
                            model._logger.log(true_obj)
                            
                model.optimize(cb)
                
                if model.SolCount > 0:
                    new_assignment = [[] for _ in range(num_machines)]
                    for i in range(num_machines):
                        for j in range(num_jobs):
                            if x[i, j].X > 0.5:
                                new_assignment[i].append(j)
                    assignment = new_assignment
                    machines = [sum(processing_times[j] for j in m) for m in assignment]
        except ImportError:
            pass
            
    final_obj = compute_obj(machines, mu)
    if logger:
        logger.log(final_obj)
        
    output = {
        "objective_value": final_obj,
        "assignment": {str(i): assignment[i] for i in range(num_machines)},
        "machine_completion_times": {str(i): machines[i] for i in range(num_machines)}
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()