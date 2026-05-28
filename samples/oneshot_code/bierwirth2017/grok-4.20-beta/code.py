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

    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    jobs = instance['jobs']

    # Precompute data
    release = [j['release_date'] for j in jobs]
    due = [j['due_date'] for j in jobs]
    weight = [j['weight'] for j in jobs]
    ops = [j['operations'] for j in jobs]  # list of dicts: machine, processing_time

    # Big-M
    total_horizon = sum(sum(op['processing_time'] for op in job_ops) for job_ops in ops) + max(release)
    M = total_horizon * 2

    model = gp.Model("JobShopTWT")

    # Decision variables
    start = {}
    for j in range(num_jobs):
        for o in range(num_machines):
            start[j, o] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"start_{j}_{o}")

    tard = {}
    for j in range(num_jobs):
        tard[j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tard_{j}")

    # Binary variables for disjunctive constraints
    y = {}
    for j1 in range(num_jobs):
        for j2 in range(j1 + 1, num_jobs):
            for m in range(num_machines):
                y[j1, j2, m] = model.addVar(vtype=GRB.BINARY, name=f"y_{j1}_{j2}_{m}")

    model.setObjective(gp.quicksum(weight[j] * tard[j] for j in range(num_jobs)), GRB.MINIMIZE)

    # Constraints
    for j in range(num_jobs):
        # Release date for first operation
        model.addConstr(start[j, 0] >= release[j], f"release_{j}")

        # Technological precedence
        for o in range(1, num_machines):
            model.addConstr(start[j, o] >= start[j, o-1] + ops[j][o-1]['processing_time'],
                            f"prec_{j}_{o}")

        # Tardiness definition
        completion = start[j, num_machines-1] + ops[j][num_machines-1]['processing_time']
        model.addConstr(tard[j] >= completion - due[j], f"tard_def_{j}")

    # Machine disjunctive constraints
    for m_idx in range(num_machines):
        # Find which operation index corresponds to each machine for each job
        op_idx_for_machine = {}
        for j in range(num_jobs):
            for o_idx, op in enumerate(ops[j]):
                if op['machine'] == m_idx:
                    op_idx_for_machine[j] = o_idx
                    break

        for j1 in range(num_jobs):
            for j2 in range(j1 + 1, num_jobs):
                o1 = op_idx_for_machine[j1]
                o2 = op_idx_for_machine[j2]
                p1 = ops[j1][o1]['processing_time']
                p2 = ops[j2][o2]['processing_time']

                yvar = y[j1, j2, m_idx]
                model.addConstr(start[j1, o1] + p1 <= start[j2, o2] + M * yvar,
                                f"disj_{j1}_{j2}_{m_idx}_1")
                model.addConstr(start[j2, o2] + p2 <= start[j1, o1] + M * (1 - yvar),
                                f"disj_{j1}_{j2}_{m_idx}_2")

    # Set time limit
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    model.Params.MIPGap = 0.01  # Allow some gap for speed

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def incumbent_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    model.optimize(incumbent_callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        start_sol = {}
        for j in range(num_jobs):
            for o in range(num_machines):
                start_sol[j, o] = start[j, o].X

        schedule = []
        objective = 0.0
        for j_idx in range(num_jobs):
            job_ops = []
            for o_idx in range(num_machines):
                mach = ops[j_idx][o_idx]['machine']
                st = start_sol[j_idx, o_idx]
                job_ops.append({
                    "machine": int(mach),
                    "start_time": float(st)
                })
            completion = float(start_sol[j_idx, num_machines-1] + ops[j_idx][num_machines-1]['processing_time'])
            tardiness = max(0.0, completion - due[j_idx])
            objective += weight[j_idx] * tardiness
            schedule.append({
                "job_id": int(j_idx),
                "completion_time": completion,
                "tardiness": float(tardiness),
                "operations": job_ops
            })

        result = {
            "objective_value": float(objective),
            "schedule": schedule
        }
    else:
        # Infeasible or no solution - return dummy with high objective
        result = {
            "objective_value": 1e6,
            "schedule": []
        }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()