import json
import argparse
import time
import sys


def read_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


def greedy_wspt(n, m, p, w, jobs_list):
    """Assign jobs in WSPT order to least loaded machine."""
    machine_jobs = [[] for _ in range(m)]
    machine_loads = [0] * m
    
    for job in jobs_list:
        j = job['j']
        best_k = 0
        min_load = machine_loads[0]
        for k in range(1, m):
            if machine_loads[k] < min_load:
                min_load = machine_loads[k]
                best_k = k
        machine_jobs[best_k].append(j)
        machine_loads[best_k] += p[j]
    
    return machine_jobs


def evaluate_machine(job_list, p, w):
    cost = 0
    current_time = 0
    for j in job_list:
        current_time += p[j]
        cost += w[j] * current_time
    return cost


def evaluate_solution(machine_jobs, p, w):
    total = 0
    for mj in machine_jobs:
        total += evaluate_machine(mj, p, w)
    return total


def find_insert_pos(to_list, j, p, w):
    """Find position to insert job j maintaining WSPT order (non-increasing w/p)."""
    for idx in range(len(to_list)):
        other = to_list[idx]
        # Compare w[other]/p[other] vs w[j]/p[j]
        # other should come before j if w[other]*p[j] >= w[j]*p[other]
        if p[other] == 0 and p[j] == 0:
            continue
        elif p[other] == 0:
            continue  # other has infinite ratio, stays before
        elif p[j] == 0:
            return idx  # j has infinite ratio, goes before other
        elif w[other] * p[j] < w[j] * p[other]:
            return idx
    return len(to_list)


def local_search(machine_jobs, p, w, n, m, time_budget, start_time):
    """Local search with move neighborhood."""
    best_cost = evaluate_solution(machine_jobs, p, w)
    
    max_passes = 200
    for _ in range(max_passes):
        if time.time() - start_time > time_budget:
            break
        
        improved_this_pass = False
        
        for j in range(1, n + 1):
            if time.time() - start_time > time_budget:
                break
            
            # Find current machine and position
            from_k = -1
            pos_from = -1
            for k in range(m):
                try:
                    pos_from = machine_jobs[k].index(j)
                    from_k = k
                    break
                except ValueError:
                    continue
            
            if from_k == -1:
                continue
            
            from_list = machine_jobs[from_k]
            
            # Cost of removing j from from_k
            c_j_from = 0
            for i in range(pos_from + 1):
                c_j_from += p[from_list[i]]
            suffix_w_from = 0
            for i in range(pos_from + 1, len(from_list)):
                suffix_w_from += w[from_list[i]]
            remove_gain = w[j] * c_j_from + p[j] * suffix_w_from
            
            best_delta = 0
            best_to_k = -1
            best_insert_pos = -1
            
            for to_k in range(m):
                if to_k == from_k:
                    continue
                
                to_list = machine_jobs[to_k]
                insert_pos = find_insert_pos(to_list, j, p, w)
                
                prefix_p = 0
                for i in range(insert_pos):
                    prefix_p += p[to_list[i]]
                c_j_to = prefix_p + p[j]
                
                suffix_w_to = 0
                for i in range(insert_pos, len(to_list)):
                    suffix_w_to += w[to_list[i]]
                
                insert_cost = w[j] * c_j_to + p[j] * suffix_w_to
                delta = insert_cost - remove_gain
                
                if delta < best_delta:
                    best_delta = delta
                    best_to_k = to_k
                    best_insert_pos = insert_pos
            
            if best_to_k != -1:
                machine_jobs[from_k].pop(pos_from)
                machine_jobs[best_to_k].insert(best_insert_pos, j)
                best_cost += best_delta
                improved_this_pass = True
        
        if not improved_this_pass:
            break
    
    return machine_jobs, best_cost


def local_search_fast(machine_jobs, p, w, n, m, time_budget, start_time):
    """Faster local search using precomputed data structures."""
    best_cost = evaluate_solution(machine_jobs, p, w)
    
    # Build job -> (machine, position) mapping
    job_machine = [0] * (n + 1)
    for k in range(m):
        for j in machine_jobs[k]:
            job_machine[j] = k
    
    max_passes = 500
    for _ in range(max_passes):
        if time.time() - start_time > time_budget:
            break
        
        improved_this_pass = False
        
        for j in range(1, n + 1):
            if time.time() - start_time > time_budget:
                break
            
            from_k = job_machine[j]
            from_list = machine_jobs[from_k]
            
            try:
                pos_from = from_list.index(j)
            except ValueError:
                continue
            
            # Removal gain
            c_j_from = 0
            for i in range(pos_from + 1):
                c_j_from += p[from_list[i]]
            suffix_w_from = 0
            for i in range(pos_from + 1, len(from_list)):
                suffix_w_from += w[from_list[i]]
            remove_gain = w[j] * c_j_from + p[j] * suffix_w_from
            
            best_delta = 0
            best_to_k = -1
            best_insert_pos = -1
            
            for to_k in range(m):
                if to_k == from_k:
                    continue
                
                to_list = machine_jobs[to_k]
                insert_pos = find_insert_pos(to_list, j, p, w)
                
                prefix_p = 0
                for i in range(insert_pos):
                    prefix_p += p[to_list[i]]
                c_j_to = prefix_p + p[j]
                
                suffix_w_to = 0
                for i in range(insert_pos, len(to_list)):
                    suffix_w_to += w[to_list[i]]
                
                insert_cost = w[j] * c_j_to + p[j] * suffix_w_to
                delta = insert_cost - remove_gain
                
                if delta < best_delta:
                    best_delta = delta
                    best_to_k = to_k
                    best_insert_pos = insert_pos
            
            if best_to_k != -1:
                machine_jobs[from_k].pop(pos_from)
                machine_jobs[best_to_k].insert(best_insert_pos, j)
                job_machine[j] = best_to_k
                best_cost += best_delta
                improved_this_pass = True
        
        if not improved_this_pass:
            break
    
    return machine_jobs, best_cost


def solve_mip(p, w, n, m, time_limit, warm_start):
    """Solve the scheduling problem using Gurobi MIP."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    
    try:
        model = gp.Model("weighted_scheduling")
        model.setParam('OutputFlag', 0)
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(time_limit, 1))
        
        M_big = sum(p[j] for j in range(1, n + 1))
        if M_big == 0:
            M_big = 1
        
        # Decision variables
        x = {}
        for j in range(1, n + 1):
            for k in range(m):
                x[j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{k}")
        
        # Completion time variables (partial sums)
        C = {}
        for j in range(1, n + 1):
            for k in range(m):
                C[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=M_big, name=f"C_{j}_{k}")
        
        # Linearized product variables
        f = {}
        for j in range(1, n + 1):
            for k in range(m):
                f[j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=M_big, name=f"f_{j}_{k}")
        
        model.update()
        
        # Objective: minimize sum of w_j * C_j
        obj = gp.quicksum(w[j] * f[j, k] for j in range(1, n + 1) for k in range(m))
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Assignment constraints
        for j in range(1, n + 1):
            model.addConstr(gp.quicksum(x[j, k] for k in range(m)) == 1, name=f"assign_{j}")
        
        # Completion time recurrence: C[j,k] = C[j-1,k] + p[j]*x[j,k]
        for k in range(m):
            for j in range(1, n + 1):
                if j == 1:
                    model.addConstr(C[j, k] == p[j] * x[j, k], name=f"Crec_{j}_{k}")
                else:
                    model.addConstr(C[j, k] == C[j - 1, k] + p[j] * x[j, k], name=f"Crec_{j}_{k}")
        
        # Linearization: f[j,k] = x[j,k] * C[j,k]
        for j in range(1, n + 1):
            for k in range(m):
                model.addConstr(f[j, k] <= M_big * x[j, k], name=f"lin1_{j}_{k}")
                model.addConstr(f[j, k] <= C[j, k], name=f"lin2_{j}_{k}")
                model.addConstr(f[j, k] >= C[j, k] - M_big * (1 - x[j, k]), name=f"lin3_{j}_{k}")
        
        # Warm start from heuristic solution
        if warm_start is not None:
            warm_set = set()
            for k in range(m):
                for j in warm_start[k]:
                    warm_set.add((j, k))
            
            for j in range(1, n + 1):
                for k in range(m):
                    if (j, k) in warm_set:
                        x[j, k].Start = 1.0
                    else:
                        x[j, k].Start = 0.0
        
        model.optimize()
        
        if model.SolCount > 0:
            machine_jobs = [[] for _ in range(m)]
            for j in range(1, n + 1):
                for k in range(m):
                    if x[j, k].X > 0.5:
                        machine_jobs[k].append(j)
                        break
            
            # Ensure WSPT order on each machine
            def wspt_key(j):
                if p[j] == 0:
                    return float('inf')
                return w[j] / p[j]
            
            for k in range(m):
                machine_jobs[k].sort(key=lambda j: -wspt_key(j))
            
            return machine_jobs
        
        return None
    
    except Exception:
        return None


def build_solution(machine_jobs, p, w, n, m):
    """Build solution dictionary matching the output schema."""
    machines = []
    total_cost = 0
    
    for k in range(m):
        job_list = machine_jobs[k]
        machine_cost = 0
        current_time = 0
        jobs_output = []
        
        for j in job_list:
            current_time += p[j]
            machine_cost += w[j] * current_time
            jobs_output.append({
                "j_orig": j,
                "p": p[j],
                "w": w[j],
                "C_j": current_time
            })
        
        total_cost += machine_cost
        machines.append({
            "schedule_cost": int(machine_cost),
            "jobs": jobs_output
        })
    
    return {
        "objective_value": float(total_cost),
        "machines": machines
    }


def main():
    parser = argparse.ArgumentParser(description='Minimize total weighted completion time on parallel machines')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    
    # Read instance
    instance = read_instance(args.instance_path)
    n = instance['n']
    m = instance['m']
    jobs_list = instance['jobs']
    
    # Store job data (1-indexed)
    p = [0] * (n + 1)
    w = [0] * (n + 1)
    for job in jobs_list:
        p[job['j']] = job['p']
        w[job['j']] = job['w']
    
    # Special case: m >= n, each job on its own machine
    if m >= n:
        machine_jobs = [[jobs_list[i]['j']] for i in range(n)]
        # Pad with empty machines
        while len(machine_jobs) < m:
            machine_jobs.append([])
        best_cost = evaluate_solution(machine_jobs, p, w)
        sol = build_solution(machine_jobs, p, w, n, m)
        if logger:
            logger.log_solution(best_cost, sol)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return
    
    # Special case: all processing times are zero
    if all(p[j] == 0 for j in range(1, n + 1)):
        machine_jobs = [[] for _ in range(m)]
        for job in jobs_list:
            machine_jobs[0].append(job['j'])
        best_cost = 0
        sol = build_solution(machine_jobs, p, w, n, m)
        if logger:
            logger.log_solution(best_cost, sol)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return
    
    # Phase 1: Greedy WSPT assignment
    machine_jobs = greedy_wspt(n, m, p, w, jobs_list)
    best_cost = evaluate_solution(machine_jobs, p, w)
    best_assignment = [list(mj) for mj in machine_jobs]
    
    sol = build_solution(best_assignment, p, w, n, m)
    if logger:
        logger.log_solution(best_cost, sol)
    
    # Phase 2: Local search
    ls_time_budget = args.time_limit * 0.55
    if time.time() - start_time < ls_time_budget:
        if n <= 200:
            machine_jobs, cost = local_search(machine_jobs, p, w, n, m, ls_time_budget, start_time)
        else:
            machine_jobs, cost = local_search_fast(machine_jobs, p, w, n, m, ls_time_budget, start_time)
        
        if cost < best_cost:
            best_cost = cost
            best_assignment = [list(mj) for mj in machine_jobs]
            sol = build_solution(best_assignment, p, w, n, m)
            if logger:
                logger.log_solution(best_cost, sol)
    
    # Phase 3: MIP with Gurobi
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 3 and n * m <= 25000:
        mip_time = remaining - 1.0
        mip_result = solve_mip(p, w, n, m, mip_time, best_assignment)
        if mip_result is not None:
            mip_cost = evaluate_solution(mip_result, p, w)
            if mip_cost < best_cost:
                best_cost = mip_cost
                best_assignment = mip_result
                sol = build_solution(best_assignment, p, w, n, m)
                if logger:
                    logger.log_solution(best_cost, sol)
                
                # Post-MIP local search if time permits
                remaining2 = args.time_limit - (time.time() - start_time)
                if remaining2 > 2:
                    machine_jobs2, cost2 = local_search_fast(
                        [list(mj) for mj in best_assignment], p, w, n, m,
                        time.time() - start_time + remaining2 - 0.5, start_time
                    )
                    if cost2 < best_cost:
                        best_cost = cost2
                        best_assignment = machine_jobs2
                        sol = build_solution(best_assignment, p, w, n, m)
                        if logger:
                            logger.log_solution(best_cost, sol)
    
    # Write final solution
    sol = build_solution(best_assignment, p, w, n, m)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()