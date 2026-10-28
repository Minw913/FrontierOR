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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    problem_type = instance['problem_type']
    machine_type = instance['machine_type']
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']
    due_dates = instance.get('jobs', {}).get('due_dates')
    speeds = instance.get('machines', {}).get('speeds')
    base_processing_times = instance.get('jobs', {}).get('base_processing_times')
    objective_name = instance['objective']
    congestion_q = instance.get('congestion_parameter_q')

    is_tardy = (objective_name == 'minimize_weighted_number_of_tardy_jobs')
    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    if num_jobs == 0:
        objective_value = 0.0
        schedule = {f"machine_{m}": [] for m in range(num_machines)}
        completion_times = {}
        tardy_jobs = []
        on_time_jobs = []
        if is_tardy:
            tardy_jobs = []
            on_time_jobs = []
    else:
        try:
            m = gp.Model("Scheduling")
            m.setParam('TimeLimit', args.time_limit - 5)
            m.setParam('Threads', 1)
            m.setParam('MIPGap', 1e-4)
            m.setParam('LogToConsole', 0)

            if not is_tardy:
                # Assignment variables
                x = m.addVars(num_jobs, num_machines, vtype=GRB.BINARY, name="x")
                # Position-based completion modeling with big-M ordering
                C = m.addVars(num_jobs, name="C", lb=0)
                # Binary ordering variables y_ijk = 1 if job i before job j on machine k
                y = m.addVars(num_jobs, num_jobs, num_machines, vtype=GRB.BINARY, name="y")

                M = sum(max(processing_times[i][k] for k in range(num_machines)) for i in range(num_jobs)) + 1

                # Each job assigned to exactly one machine
                for i in range(num_jobs):
                    m.addConstr(gp.quicksum(x[i, k] for k in range(num_machines)) == 1)

                # Completion time bounds
                for i in range(num_jobs):
                    for k in range(num_machines):
                        p_ik = processing_times[i][k]
                        m.addConstr(C[i] >= p_ik * x[i, k])

                # Ordering and disjunctive constraints
                for i in range(num_jobs):
                    for j in range(num_jobs):
                        if i == j:
                            continue
                        for k in range(num_machines):
                            p_jk = processing_times[j][k]
                            m.addConstr(C[i] + p_jk - M * (3 - y[i, j, k] - x[i, k] - x[j, k]) <= C[j])
                            m.addConstr(C[j] + p_jk * (1 - y[i, j, k]) - M * (3 - (1 - y[i, j, k]) - x[i, k] - x[j, k]) <= C[i] + M * (2 - x[i, k] - x[j, k]))

                # Objective: sum w_i * C_i
                obj = gp.quicksum(weights[i] * C[i] for i in range(num_jobs))
                m.setObjective(obj, GRB.MINIMIZE)

                # Callback for incumbent logging
                def callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                        if logger:
                            logger.log(float(obj_val))

                m.optimize(callback)

                if m.status == GRB.INFEASIBLE:
                    raise Exception("Model infeasible")

                objective_value = m.objVal
                sol_x = m.getAttr('X', x)
                sol_C = m.getAttr('X', C)

                # Build schedule
                machine_jobs = [[] for _ in range(num_machines)]
                for i in range(num_jobs):
                    for k in range(num_machines):
                        if sol_x[i, k] > 0.5:
                            machine_jobs[k].append(i)
                            break

                # Sort each machine's jobs by completion time
                schedule = {}
                completion_times = {}
                for k in range(num_machines):
                    jobs_on_k = machine_jobs[k]
                    jobs_on_k.sort(key=lambda i: sol_C[i])
                    schedule[f"machine_{k}"] = jobs_on_k
                    for idx, j in enumerate(jobs_on_k):
                        completion_times[str(j)] = float(sol_C[j])

                tardy_jobs = []
                on_time_jobs = []

            else:
                # Weighted number of tardy jobs
                # Use assignment + tardiness indicator
                x = m.addVars(num_jobs, num_machines, vtype=GRB.BINARY, name="x")
                tardy = m.addVars(num_jobs, vtype=GRB.BINARY, name="tardy")
                C = m.addVars(num_jobs, name="C", lb=0)

                M = sum(max(processing_times[i][k] for k in range(num_machines)) for i in range(num_jobs)) * 2 + max(due_dates) + 1

                for i in range(num_jobs):
                    m.addConstr(gp.quicksum(x[i, k] for k in range(num_machines)) == 1 - tardy[i])

                for i in range(num_jobs):
                    for k in range(num_machines):
                        p_ik = processing_times[i][k]
                        m.addConstr(C[i] <= due_dates[i] + M * tardy[i])
                        m.addConstr(C[i] >= p_ik - M * (1 - x[i, k]))

                # No two jobs on same machine without ordering
                y = m.addVars(num_jobs, num_jobs, num_machines, vtype=GRB.BINARY, name="y")
                for i in range(num_jobs):
                    for j in range(i+1, num_jobs):
                        for k in range(num_machines):
                            pij = processing_times[i][k]
                            pjj = processing_times[j][k]
                            m.addConstr(C[i] + pjj - M * (3 - y[i,j,k] - x[i,k] - x[j,k]) <= C[j])
                            m.addConstr(C[j] + pij - M * (3 - y[j,i,k] - x[i,k] - x[j,k]) <= C[i])

                obj = gp.quicksum(weights[i] * tardy[i] for i in range(num_jobs))
                m.setObjective(obj, GRB.MINIMIZE)

                def callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                        if logger:
                            logger.log(float(obj_val))

                m.optimize(callback)

                if m.status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                    # Fallback: make all tardy
                    objective_value = float(sum(weights))
                    schedule = {f"machine_{m}": [] for m in range(num_machines)}
                    completion_times = {}
                    tardy_jobs = list(range(num_jobs))
                    on_time_jobs = []
                else:
                    objective_value = m.objVal
                    sol_x = m.getAttr('X', x)
                    sol_tardy = m.getAttr('X', tardy)
                    sol_C = m.getAttr('X', C)

                    machine_jobs = [[] for _ in range(num_machines)]
                    tardy_jobs = []
                    on_time_jobs = []
                    completion_times = {}

                    for i in range(num_jobs):
                        if sol_tardy[i] > 0.5:
                            tardy_jobs.append(i)
                        else:
                            on_time_jobs.append(i)
                            for k_idx in range(num_machines):
                                if sol_x[i, k_idx] > 0.5:
                                    machine_jobs[k_idx].append(i)
                                    completion_times[str(i)] = float(sol_C[i])
                                    break

                    schedule = {}
                    for k in range(num_machines):
                        jobs_k = machine_jobs[k]
                        jobs_k.sort(key=lambda j: sol_C[j])
                        schedule[f"machine_{k}"] = jobs_k

        except Exception as e:
            # Emergency fallback
            if is_tardy:
                objective_value = float(sum(weights))
                schedule = {f"machine_{m}": [] for m in range(num_machines)}
                completion_times = {}
                tardy_jobs = list(range(num_jobs))
                on_time_jobs = []
            else:
                # Simple list scheduling fallback for completion time
                loads = [0.0] * num_machines
                job_assign = [-1] * num_jobs
                job_complete = [0.0] * num_jobs
                machine_lists = [[] for _ in range(num_machines)]

                # Sort jobs by weight / avg_p descending
                avg_p = [sum(processing_times[i]) / num_machines for i in range(num_jobs)]
                job_order = sorted(range(num_jobs), key=lambda i: weights[i] / max(avg_p[i], 1e-6), reverse=True)

                for j in job_order:
                    best_m = 0
                    best_load = loads[0]
                    for k in range(1, num_machines):
                        if loads[k] < best_load:
                            best_load = loads[k]
                            best_m = k
                    p = processing_times[j][best_m]
                    loads[best_m] += p
                    job_assign[j] = best_m
                    job_complete[j] = loads[best_m]
                    machine_lists[best_m].append(j)

                objective_value = sum(weights[j] * job_complete[j] for j in range(num_jobs))
                schedule = {}
                completion_times = {}
                for k in range(num_machines):
                    # sort by completion
                    ordered = sorted(machine_lists[k], key=lambda jid: job_complete[jid])
                    schedule[f"machine_{k}"] = ordered
                    for jid in ordered:
                        completion_times[str(jid)] = float(job_complete[jid])
                tardy_jobs = []
                on_time_jobs = []

    # Ensure output matches schema exactly
    result = {
        "objective_value": float(objective_value),
        "schedule": schedule,
        "completion_times": completion_times
    }

    if is_tardy:
        result["tardy_jobs"] = [int(j) for j in tardy_jobs]
        result["on_time_jobs"] = [int(j) for j in on_time_jobs]
    else:
        result["tardy_jobs"] = []
        result["on_time_jobs"] = []

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    if logger:
        logger.log(float(objective_value))

if __name__ == "__main__":
    main()