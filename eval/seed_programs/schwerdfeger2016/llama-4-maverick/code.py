import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance['num_machines'], instance['num_jobs'], instance['processing_times']

def solve_instance(num_machines, num_jobs, processing_times, time_limit):
    model = gp.Model('ParallelMachinesScheduling')
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0
    
    # Decision variables: x[j, m] = 1 if job j is assigned to machine m, 0 otherwise
    x = model.addVars(num_jobs, num_machines, vtype=GRB.BINARY, name='x')
    
    # Objective is to minimize the sum of squared completion times
    completion_times = gp.quicksum(processing_times[j] * x[j, m] for j in range(num_jobs) for m in range(num_machines))
    avg_completion_time = sum(processing_times) / num_machines
    model.setObjective(gp.quicksum((gp.quicksum(processing_times[j] * x[j, m] for j in range(num_jobs)) - avg_completion_time) ** 2 for m in range(num_machines)), GRB.MINIMIZE)
    
    # Constraints: Each job is assigned to exactly one machine
    for j in range(num_jobs):
        model.addConstr(gp.quicksum(x[j, m] for m in range(num_machines)) == 1, name=f'JobAssignment_{j}')
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        assignment = {}
        machine_completion_times = {}
        avg_completion_time = sum(processing_times) / num_machines
        for m in range(num_machines):
            machine_jobs = [j for j in range(num_jobs) if x[j, m].X > 0.5]
            assignment[f'machine_{m}'] = machine_jobs
            machine_completion_times[f'machine_{m}'] = sum(processing_times[j] for j in machine_jobs)
        
        sum_squared_deviations = sum((machine_completion_times[f'machine_{m}'] - avg_completion_time) ** 2 for m in range(num_machines))
        objective_value = (sum_squared_deviations ** 0.5) / avg_completion_time
        
        return objective_value, assignment, machine_completion_times
    else:
        raise Exception(f"Failed to find a feasible solution. Model status: {model.status}")

def main():
    parser = argparse.ArgumentParser(description='Solve parallel machines scheduling problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions (optional).')
    args = parser.parse_args()
    
    num_machines, num_jobs, processing_times = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Initial solution using a simple heuristic (assign jobs to machines in a round-robin manner)
    initial_assignment = {}
    initial_machine_completion_times = {f'machine_{m}': 0 for m in range(num_machines)}
    for j in range(num_jobs):
        machine = j % num_machines
        if f'machine_{machine}' not in initial_assignment:
            initial_assignment[f'machine_{machine}'] = []
        initial_assignment[f'machine_{machine}'].append(j)
        initial_machine_completion_times[f'machine_{machine}'] += processing_times[j]
    avg_completion_time = sum(processing_times) / num_machines
    initial_sum_squared_deviations = sum((initial_machine_completion_times[f'machine_{m}'] - avg_completion_time) ** 2 for m in range(num_machines))
    initial_objective_value = (initial_sum_squared_deviations ** 0.5) / avg_completion_time
    if logger:
        logger.log(initial_objective_value)
    
    # Solve the instance using Gurobi
    try:
        objective_value, assignment, machine_completion_times = solve_instance(num_machines, num_jobs, processing_times, args.time_limit)
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    
    solution = {
        'objective_value': objective_value,
        'assignment': assignment,
        'machine_completion_times': machine_completion_times
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()