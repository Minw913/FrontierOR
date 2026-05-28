import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance_data, time_limit, logger):
    num_jobs = instance_data['num_jobs']
    num_machines = instance_data['num_machines']
    jobs = instance_data['jobs']

    model = Model('job_shop_scheduling')
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi logging to console

    # Decision variables
    start_vars = {}
    for job in jobs:
        for i, op in enumerate(job['operations']):
            start_vars[(job['job_id'], i)] = model.addVar(vtype=GRB.CONTINUOUS, name=f"start_{job['job_id']}_{i}")

    # Objective: minimize total weighted tardiness
    tardiness_vars = {}
    for job in jobs:
        tardiness_vars[job['job_id']] = model.addVar(vtype=GRB.CONTINUOUS, name=f"tardiness_{job['job_id']}")
        completion_time = start_vars[(job['job_id'], len(job['operations'])-1)] + job['operations'][-1]['processing_time']
        model.addConstr(tardiness_vars[job['job_id']] >= completion_time - job['due_date'], name=f"tardiness_def_{job['job_id']}")
        model.addConstr(tardiness_vars[job['job_id']] >= 0, name=f"non_neg_tardiness_{job['job_id']}")

    model.setObjective(quicksum(job['weight'] * tardiness_vars[job['job_id']] for job in jobs), GRB.MINIMIZE)

    # Constraints
    for job in jobs:
        # Release date constraint
        model.addConstr(start_vars[(job['job_id'], 0)] >= job['release_date'], name=f"release_date_{job['job_id']}")

        # Operation precedence constraints
        for i in range(len(job['operations'])-1):
            model.addConstr(start_vars[(job['job_id'], i+1)] >= start_vars[(job['job_id'], i)] + job['operations'][i]['processing_time'], 
                            name=f"precedence_{job['job_id']}_{i}")

    # No overlap constraints between operations on the same machine
    op_machine_map = {}
    for job in jobs:
        for i, op in enumerate(job['operations']):
            machine = op['machine']
            if machine not in op_machine_map:
                op_machine_map[machine] = []
            op_machine_map[machine].append((job['job_id'], i))

    for machine in op_machine_map.values():
        for i in range(len(machine)):
            for j in range(i+1, len(machine)):
                job_id1, op_idx1 = machine[i]
                job_id2, op_idx2 = machine[j]
                if job_id1 != job_id2:
                    y = model.addVar(vtype=GRB.BINARY, name=f"y_{job_id1}_{op_idx1}_{job_id2}_{op_idx2}")
                    model.addConstr(start_vars[(job_id1, op_idx1)] + jobs[job_id1]['operations'][op_idx1]['processing_time'] <= start_vars[(job_id2, op_idx2)] + 1e6 * y, 
                                    name=f"no_overlap1_{job_id1}_{op_idx1}_{job_id2}_{op_idx2}")
                    model.addConstr(start_vars[(job_id2, op_idx2)] + jobs[job_id2]['operations'][op_idx2]['processing_time'] <= start_vars[(job_id1, op_idx1)] + 1e6 * (1-y), 
                                    name=f"no_overlap2_{job_id1}_{op_idx1}_{job_id2}_{op_idx2}")

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        if logger:
            logger.log(objective_value)

        solution = {
            "objective_value": objective_value,
            "schedule": []
        }

        for job in jobs:
            job_id = job['job_id']
            job_schedule = {
                "job_id": job_id,
                "completion_time": start_vars[(job_id, len(job['operations'])-1)].X + job['operations'][-1]['processing_time'],
                "tardiness": tardiness_vars[job_id].X,
                "operations": []
            }
            for i, op in enumerate(job['operations']):
                job_schedule['operations'].append({
                    "machine": op['machine'],
                    "start_time": start_vars[(job_id, i)].X
                })
            solution['schedule'].append(job_schedule)

        return solution
    else:
        raise Exception(f"Model failed to solve: status {model.status}")

def main():
    parser = argparse.ArgumentParser(description='Job Shop Scheduling Solver')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional)')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance_data = read_instance(args.instance_path)
    solution = solve(instance_data, args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()