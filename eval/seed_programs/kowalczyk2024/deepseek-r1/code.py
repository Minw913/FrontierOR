import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Minimize Total Weighted Tardiness')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    jobs = instance['jobs']
    n = instance['n']
    m = instance['m']
    T = instance['T']

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    # Create model
    model = gp.Model('TWT')
    model.setParam("Threads", 1)
    model.Params.OutputFlag = 0

    # Variables
    a = {}  # Assignment variables: a[job_id, k, p]
    for job in jobs:
        job_id = job['id']
        for k in range(m):
            for p in range(1, n + 1):
                a[(job_id, k, p)] = model.addVar(vtype=GRB.BINARY, name=f'a_{job_id}_{k}_{p}')

    x = {}  # Completion times: x[(k, p)]
    for k in range(m):
        for p in range(n + 1):
            x[(k, p)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'x_{k}_{p}')

    s = {job['id']: model.addVar(vtype=GRB.CONTINUOUS, name=f's_{job["id"]}') for job in jobs}
    T_vars = {job['id']: model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'T_{job["id"]}') for job in jobs}

    # Objective
    model.setObjective(gp.quicksum(T_vars[job['id']] * job['w'] for job in jobs), GRB.MINIMIZE)

    # Constraints
    for job in jobs:
        job_id = job['id']
        model.addConstr(gp.quicksum(a[job_id, k, p] for k in range(m) for p in range(1, n + 1)) == 1)

    for k in range(m):
        for p in range(1, n + 1):
            model.addConstr(gp.quicksum(a[job['id'], k, p] for job in jobs) <= 1)

    for k in range(m):
        model.addConstr(x[k, 0] == 0)
        for p in range(1, n + 1):
            sum_p = gp.quicksum(a[job['id'], k, p] * job['p'] for job in jobs)
            model.addConstr(x[k, p] == x[k, p-1] + sum_p)

    for job in jobs:
        job_id = job['id']
        p = job['p']
        d = job['d']
        model.addConstr(T_vars[job_id] >= s[job_id] + p - d)
        model.addConstr(T_vars[job_id] >= 0)
        for k in range(m):
            for p_pos in range(1, n + 1):
                model.addGenConstrIndicator(a[job_id, k, p_pos], True, s[job_id] == x[k, p_pos - 1])

    # Callback for logging
    model._logger = logger

    def callback(model, where):
        if where == GRB.Callback.MIPSOL and model._logger:
            model._logger.log(model.cbGet(GRB.Callback.MIPSOL_OBJ))

    model.Params.TimeLimit = args.time_limit
    model.optimize(callback)

    # Extract solution
    schedule = {}
    if model.SolCount > 0:
        for job in jobs:
            job_id = job['id']
            for k in range(m):
                found = False
                for p in range(1, n + 1):
                    if a[job_id, k, p].X > 0.5:
                        schedule[job_id] = x[k, p-1].X
                        found = True
                        break
                if found:
                    break
        obj_value = model.ObjVal
    else:
        # Fallback: greedy assignment (not optimal but valid)
        schedule = {job['id']: 0.0 for job in jobs}
        machine_times = [0.0] * m
        for job in sorted(jobs, key=lambda j: (-j['w'], j['d'])):
            min_machine = machine_times.index(min(machine_times))
            schedule[job['id']] = machine_times[min_machine]
            machine_times[min_machine] += job['p']
        obj_value = sum(max(0, schedule[j['id']] + j['p'] - j['d']) * j['w'] for j in jobs)

    with open(args.solution_path, 'w') as f:
        json.dump({'objective_value': obj_value, 'schedule': schedule}, f)

if __name__ == '__main__':
    main()