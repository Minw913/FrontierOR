import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance_data(instance_path):
    with open(instance_path, 'r') as file:
        instance_data = json.load(file)
    return instance_data

def solve_instance(instance_data, time_limit, logger):
    num_jobs = instance_data['num_jobs']
    num_machines = instance_data['num_machines']
    jobs = instance_data['jobs']

    max_makespan = sum(max(int(operation['processing_times'][str(machine)]) for machine in operation['eligible_machines']) for job in jobs for operation in job['operations'])

    model = Model("ManufacturingFacilityScheduling")
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  

    assignment = model.addVars((((job['job_id'], operation_id), machine) for job in jobs for operation_id, operation in enumerate(job['operations']) for machine in operation['eligible_machines']), vtype=GRB.BINARY, name='assignment')
    processing = model.addVars((((job['job_id'], operation_id), machine, t) for job in jobs for operation_id, operation in enumerate(job['operations']) for machine in operation['eligible_machines'] for t in range(max_makespan)), vtype=GRB.BINARY, name='processing')
    makespan = model.addVar(vtype=GRB.INTEGER, name='makespan')

    for job in jobs:
        job_id = job['job_id']
        operations = job['operations']
        for operation_id, operation in enumerate(operations):
            eligible_machines = operation['eligible_machines']

            model.addConstr(quicksum((assignment[(job_id, operation_id), machine] for machine in eligible_machines)) == 1, f"assign_one_machine_{job_id}_{operation_id}")

            for machine in eligible_machines:
                processing_time = int(operation['processing_times'][str(machine)])
                model.addConstr(quicksum((processing[(job_id, operation_id), machine, t] for t in range(max_makespan))) == assignment[(job_id, operation_id), machine] * processing_time, f"processing_time_{job_id}_{operation_id}_{machine}")

            for t in range(max_makespan):
                for machine in eligible_machines:
                    model.addConstr(processing[(job_id, operation_id), machine, t] <= assignment[(job_id, operation_id), machine], f"process_on_assigned_machine_{job_id}_{operation_id}_{machine}_{t}")

    for machine in range(num_machines):
        for t in range(max_makespan):
            model.addConstr(quicksum((processing[(job['job_id'], operation_id), machine, t] for job in jobs for operation_id, operation in enumerate(job['operations']) if machine in operation['eligible_machines'])) <= 1, f"machine_capacity_{machine}_{t}")

    for job in jobs:
        job_id = job['job_id']
        operations = job['operations']
        for i in range(len(operations) - 1):
            current_operation_id = i
            next_operation_id = i + 1
            for machine1 in operations[current_operation_id]['eligible_machines']:
                for machine2 in operations[next_operation_id]['eligible_machines']:
                    end_time_current = quicksum((t * processing[(job_id, current_operation_id), machine1, t] for t in range(max_makespan))) + quicksum((processing[(job_id, current_operation_id), machine1, t] for t in range(max_makespan)))
                    model.addConstr(end_time_current <= quicksum((t * processing[(job_id, next_operation_id), machine2, t] for t in range(max_makespan))), f"precedence_{job_id}_{current_operation_id}_{next_operation_id}_{machine1}_{machine2}")

    for job in jobs:
        job_id = job['job_id']
        operations = job['operations']
        for operation_id, operation in enumerate(operations):
            eligible_machines = operation['eligible_machines']
            model.addConstr(makespan >= quicksum((t * processing[(job_id, operation_id), machine, t] for machine in eligible_machines for t in range(max_makespan))) + quicksum((processing[(job_id, operation_id), machine, t] for machine in eligible_machines for t in range(max_makespan))), f"makespan_definition_{job_id}_{operation_id}")

    model.setObjective(makespan, GRB.MINIMIZE)

    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution_assignment = {}
        solution_schedule = {}
        for job in jobs:
            job_id = job['job_id']
            operations = job['operations']
            for operation_id, operation in enumerate(operations):
                key = (job_id, operation_id)
                for machine in operation['eligible_machines']:
                    if assignment[key, machine].x > 0.5:
                        solution_assignment[str(key)] = machine
                        schedule = [t+1 for t in range(max_makespan) if processing[key, machine, t].x > 0.5]
                        solution_schedule[str(key)] = schedule
                        break

        objective_value = makespan.x
        if logger:
            logger.log(objective_value)

        return {
            "objective_value": objective_value,
            "assignment": solution_assignment,
            "schedule": solution_schedule
        }
    else:
        raise Exception("No solution found")

def main():
    parser = argparse.ArgumentParser(description='Manufacturing Facility Scheduling')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the JSON file containing the problem instance')
    parser.add_argument('--solution_path', type=str, required=True, help='Path where the final solution JSON file must be written')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, help='Path to a JSONL file for logging intermediate solutions')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance_data = read_instance_data(args.instance_path)
    solution = solve_instance(instance_data, args.time_limit, logger)

    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == "__main__":
    main()