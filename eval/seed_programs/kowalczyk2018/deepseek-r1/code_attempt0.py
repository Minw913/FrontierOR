import argparse
import json
import time
import gurobipy as gp
from solution_logger import SolutionLogger

def greedy_heuristic(instance):
    n = instance['n']
    m = instance['m']
    jobs = instance['jobs']
    
    machines_time = [0] * m
    machines_schedules = [[] for _ in range(m)]
    
    for job in jobs:
        best_cost = float('inf')
        best_machine = None
        for k in range(m):
            completion_time = machines_time[k] + job['p']
            cost = job['w'] * completion_time
            if cost < best_cost:
                best_cost = cost
                best_machine = k
        machines_time[best_machine] += job['p']
        job_data = {
            'j_orig': job['j'],
            'p': job['p'],
            'w': job['w'],
            'C_j': machines_time[best_machine]
        }
        machines_schedules[best_machine].append(job_data)
    
    total_obj = 0
    machines_output = []
    for k in range(m):
        machine_jobs = machines_schedules[k]
        machine_cost = sum(job['w'] * job['C_j'] for job in machine_jobs)
        total_obj += machine_cost
        machines_output.append({
            'schedule_cost': machine_cost,
            'jobs': machine_jobs
        })
    
    return {
        'objective_value': float(total_obj),
        'machines': machines_output
    }

def run_mip(instance, time_limit, initial_solution=None, logger=None):
    n = instance['n']
    m = instance['m']
    jobs = instance['jobs']
    T = sum(job['p'] for job in jobs)
    
    try:
        model = gp.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)
        
        x = {}
        s = {}
        for j in range(n):
            for k in range(m):
                x[(j, k)] = model.addVar(vtype=gp.GRB.BINARY, name=f'x_{j}_{k}')
            s[j] = model.addVar(lb=0, ub=T, vtype=gp.GRB.CONTINUOUS, name=f's_{j}')
        
        obj_expr = gp.quicksum(jobs[j]['w'] * (s[j] + jobs[j]['p']) for j in range(n))
        model.setObjective(obj_expr, gp.GRB.MINIMIZE)
        
        for j in range(n):
            model.addConstr(gp.quicksum(x[(j, k)] for k in range(m)) == 1, f'assign_{j}')
        
        for k in range(m):
            for i in range(n):
                for j in range(i+1, n):
                    model.addConstr(
                        s[j] >= s[i] + jobs[i]['p'] - T * (2 - x[(i, k)] - x[(j, k)]),
                        f'seq_{i}_{j}_{k}'
                    )
        
        if initial_solution is not None:
            job_index_map = {job['j']: idx for idx, job in enumerate(jobs)}
            assignment = [-1] * n
            for machine in initial_solution['machines']:
                for job_data in machine['jobs']:
                    j_orig = job_data['j_orig']
                    if j_orig in job_index_map:
                        idx = job_index_map[j_orig]
                        assignment[idx] = int(machine['jobs'].index(job_data) >= 0)
            for j in range(n):
                if assignment[j] != -1:
                    for k in range(m):
                        if k == assignment[j]:
                            x[(j, k)].start = 1
                        else:
                            x[(j, k)].start = 0
                    s[j].start = job_data['C_j'] - jobs[j]['p']
        
        if logger is not None:
            def callback(model, where):
                if where == gp.GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj_val)
            model.optimize(callback)
        else:
            model.optimize()
        
        if model.SolCount > 0:
            total_obj = model.objVal
            machines_schedules = [[] for _ in range(m)]
            for j in range(n):
                for k in range(m):
                    if x[(j, k)].x > 0.5:
                        comp_time = s[j].x + jobs[j]['p']
                        job_data = {
                            'j_orig': jobs[j]['j'],
                            'p': jobs[j]['p'],
                            'w': jobs[j]['w'],
                            'C_j': comp_time
                        }
                        machines_schedules[k].append(job_data)
                        break
            for k in range(m):
                machines_schedules[k].sort(key=lambda job: job['C_j'])
            machines_output = []
            for k in range(m):
                machine_jobs = machines_schedules[k]
                machine_cost = sum(job['w'] * job['C_j'] for job in machine_jobs)
                machines_output.append({
                    'schedule_cost': machine_cost,
                    'jobs': machine_jobs
                })
            return {
                'objective_value': total_obj,
                'machines': machines_output
            }
        else:
            return None
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return None

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
    
    start_time = time.time()
    greedy_sol = greedy_heuristic(instance)
    greedy_obj = greedy_sol['objective_value']
    
    if logger:
        logger.log(greedy_obj)
    
    n = instance['n']
    m = instance['m']
    final_sol = greedy_sol
    
    if n <= 20 and m <= 5:
        time_so_far = time.time() - start_time
        time_remaining = max(1, args.time_limit - time_so_far)
        mip_sol = run_mip(instance, time_remaining, initial_solution=greedy_sol, logger=logger)
        if mip_sol is not None and mip_sol['objective_value'] < greedy_obj:
            final_sol = mip_sol
    
    with open(args.solution_path, 'w') as f:
        json.dump(final_sol, f, indent=2)

if __name__ == "__main__":
    main()