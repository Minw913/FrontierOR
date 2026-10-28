import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def greedy_completion_time(instance):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    
    min_processing = [min(processing_times[j]) for j in range(num_jobs)]
    ratios_global = []
    for j in range(num_jobs):
        if min_processing[j] == 0:
            ratios_global.append(float('inf'))
        else:
            ratios_global.append(weights[j] / min_processing[j])
    
    job_order = sorted(range(num_jobs), key=lambda j: ratios_global[j], reverse=True)
    
    machines = [{'jobs': [], 'total_w': 0} for _ in range(num_machines)]
    
    for j in job_order:
        best_increase = float('inf')
        best_machine = None
        best_pos = None
        best_r_ij = None
        
        for i in range(num_machines):
            p_ij = processing_times[j][i]
            if p_ij == 0:
                r_ij = float('inf')
            else:
                r_ij = weights[j] / p_ij
            
            total_processing_A = 0
            total_weight_A = 0
            pos = 0
            for job_tuple in machines[i]['jobs']:
                if job_tuple[0] >= r_ij:
                    total_processing_A += job_tuple[1]
                    total_weight_A += job_tuple[2]
                    pos += 1
                else:
                    break
            total_weight_B = machines[i]['total_w'] - total_weight_A
            
            increase = weights[j] * (total_processing_A + p_ij) + p_ij * total_weight_B
            if increase < best_increase:
                best_increase = increase
                best_machine = i
                best_pos = pos
                best_r_ij = r_ij
        
        machines[best_machine]['jobs'].insert(best_pos, (best_r_ij, processing_times[j][best_machine], weights[j], j))
        machines[best_machine]['total_w'] += weights[j]
    
    schedule = {}
    completion_times = {}
    total_weighted_completion = 0.0
    
    for i in range(num_machines):
        machine_schedule = []
        time = 0.0
        for job_tuple in machines[i]['jobs']:
            job_index = job_tuple[3]
            p_ij = job_tuple[1]
            time += p_ij
            completion_times[str(job_index)] = time
            total_weighted_completion += weights[job_index] * time
            machine_schedule.append(job_index)
        schedule[str(i)] = machine_schedule
    
    solution = {
        "objective_value": total_weighted_completion,
        "schedule": schedule,
        "completion_times": completion_times
    }
    return solution

def greedy_tardy_jobs(instance):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    due_dates = instance['jobs']['due_dates']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    
    jobs_sorted = sorted(range(num_jobs), key=lambda j: due_dates[j])
    
    machines = [{'total': 0.0, 'min_d': float('inf')} for _ in range(num_machines)]
    assignment = [-1] * num_jobs
    
    for j in jobs_sorted:
        candidate = None
        best_total = float('inf')
        for i in range(num_machines):
            p_ij = processing_times[j][i]
            new_total = machines[i]['total'] + p_ij
            new_min_d = min(machines[i]['min_d'], due_dates[j])
            if new_total <= new_min_d:
                if new_total < best_total:
                    best_total = new_total
                    candidate = i
        if candidate is not None:
            assignment[j] = candidate
            machines[candidate]['total'] = best_total
            machines[candidate]['min_d'] = min(machines[candidate]['min_d'], due_dates[j])
    
    tardy_jobs = [j for j in range(num_jobs) if assignment[j] == -1]
    on_time_jobs = [j for j in range(num_jobs) if assignment[j] != -1]
    total_tardy_weight = sum(weights[j] for j in tardy_jobs)
    
    machine_jobs = [[] for _ in range(num_machines)]
    for j in range(num_jobs):
        if assignment[j] != -1:
            machine_jobs[assignment[j]].append(j)
    
    schedule = {}
    completion_times = {}
    for i in range(num_machines):
        jobs_on_machine = machine_jobs[i]
        jobs_sorted = sorted(jobs_on_machine, key=lambda j: due_dates[j])
        time = 0.0
        for job in jobs_sorted:
            time += processing_times[job][i]
            completion_times[str(job)] = time
        schedule[str(i)] = jobs_sorted
    
    solution = {
        "objective_value": total_tardy_weight,
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": tardy_jobs,
        "on_time_jobs": on_time_jobs
    }
    return solution

def mip_tardy_jobs(instance, time_remaining, callback_func=None):
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    due_dates = instance['jobs']['due_dates']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, time_remaining))
    
    x = model.addVars(num_machines, num_jobs, vtype=GRB.BINARY, name="x")
    
    obj_expr = gp.quicksum(weights[j] * x[i, j] for i in range(num_machines) for j in range(num_jobs))
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    for j in range(num_jobs):
        model.addConstr(gp.quicksum(x[i, j] for i in range(num_machines)) <= 1, name=f"job_{j}_assign")
    
    S = {}
    for j in range(num_jobs):
        S[j] = [l for l in range(num_jobs) if due_dates[l] <= due_dates[j]]
    
    for i in range(num_machines):
        for j in range(num_jobs):
            expr = gp.quicksum(processing_times[l][i] * x[i, l] for l in S[j])
            model.addGenConstrIndicator(x[i, j], 1, expr <= due_dates[j], name=f"ind_{i}_{j}")
    
    if callback_func:
        model.optimize(callback_func)
    else:
        model.optimize()
    
    assignment = [-1] * num_jobs
    if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
        for i in range(num_machines):
            for j in range(num_jobs):
                if x[i, j].X > 0.5:
                    assignment[j] = i
    
    tardy_jobs = [j for j in range(num_jobs) if assignment[j] == -1]
    on_time_jobs = [j for j in range(num_jobs) if assignment[j] != -1]
    total_tardy_weight = sum(weights[j] for j in tardy_jobs)
    
    machine_jobs = [[] for _ in range(num_machines)]
    for j in range(num_jobs):
        if assignment[j] != -1:
            machine_jobs[assignment[j]].append(j)
    
    schedule = {}
    completion_times = {}
    for i in range(num_machines):
        jobs_on_machine = machine_jobs[i]
        jobs_sorted = sorted(jobs_on_machine, key=lambda j: due_dates[j])
        time = 0.0
        for job in jobs_sorted:
            time += processing_times[job][i]
            completion_times[str(job)] = time
        schedule[str(i)] = jobs_sorted
    
    solution = {
        "objective_value": total_tardy_weight,
        "schedule": schedule,
        "completion_times": completion_times,
        "tardy_jobs": tardy_jobs,
        "on_time_jobs": on_time_jobs
    }
    return solution

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
    
    if instance['objective'] == 'minimize_total_weighted_completion_time':
        sol = greedy_completion_time(instance)
        if logger:
            logger.log(sol['objective_value'])
    else:
        num_jobs = instance['num_jobs']
        num_machines = instance['num_machines']
        total_weight = sum(instance['jobs']['weights'])
        
        def mip_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                tardy_val = total_weight - obj_val
                if logger:
                    logger.log(tardy_val)
        
        if num_jobs <= 100 and num_machines <= 20 and num_jobs * num_machines <= 200:
            try:
                time_remaining = args.time_limit - (time.time() - start_time)
                sol = mip_tardy_jobs(instance, time_remaining, mip_callback)
                if logger:
                    logger.log(sol['objective_value'])
            except Exception as e:
                sol = greedy_tardy_jobs(instance)
                if logger:
                    logger.log(sol['objective_value'])
        else:
            sol = greedy_tardy_jobs(instance)
            if logger:
                logger.log(sol['objective_value'])
    
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)

if __name__ == '__main__':
    main()