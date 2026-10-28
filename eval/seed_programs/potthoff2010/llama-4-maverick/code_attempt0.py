import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
import time

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def write_solution(solution_path, solution):
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model, solution_dict = build_model(instance)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    model.optimize(lambda model, where: callback(model, where, logger))

    if model.SolCount > 0:
        solution_dict = retrieve_solution(instance, model, solution_dict)
        write_solution(args.solution_path, solution_dict)
    else:
        print("No solution found.")

def build_model(instance):
    model = Model("Crew Rescheduling")
    duties = instance['active_duties'] + instance['reserve_duties']
    tasks = instance['tasks']

    task_cover = model.addVars(tasks, vtype=GRB.BINARY, name='task_cover')
    duty_completion = model.addVars(duties, range(len(duties)), vtype=GRB.BINARY, name='duty_completion')

    model.setObjective(
        quicksum(instance['parameters']['cost_cancel_A_to_B'] * (1 - task_cover[task['task_id']]) for task in tasks if task['task_type'] == 'A_to_B') +
        quicksum(instance['parameters']['cost_cancel_A_to_A'] * (1 - task_cover[task['task_id']]) for task in tasks if task['task_type'] == 'A_to_A') +
        quicksum(duty_completion[duty['duty_id'], i] * compute_completion_cost(instance, duty, i) for i, duty in enumerate(duties)),
        GRB.MINIMIZE
    )

    for task in tasks:
        task_id = task['task_id']
        model.addConstr(quicksum(duty_completion[duty['duty_id'], i] for i, duty in enumerate(duties) if can_cover_task(duty, task)) >= task_cover[task_id], name=f'cover_{task_id}')

    for duty in duties:
        model.addConstr(quicksum(duty_completion[duty['duty_id'], i] for i in range(len(duties))) == 1, name=f'duty_completion_{duty["duty_id"]}')

    return model, {duty['duty_id']: None for duty in duties}

def can_cover_task(duty, task):
    # Simplified check, actual implementation depends on detailed duty and task data
    return task['task_id'] in duty['tasks']

def compute_completion_cost(instance, duty, completion_index):
    # Simplified cost computation, actual implementation depends on detailed instance and duty data
    parameters = instance['parameters']
    cost = parameters['cost_duty_changed'] if completion_index > 0 else 0
    # Additional costs for reassigned tasks, new transfers, and taxi use need to be computed based on the specific completion
    return cost

def callback(model, where, logger):
    if where == GRB.Callback.MIPSOL:
        objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if logger:
            logger.log(objective_value)

def retrieve_solution(instance, model, solution_dict):
    duties = instance['active_duties'] + instance['reserve_duties']
    tasks = instance['tasks']

    task_cover = {task['task_id']: model.getVarByName(f'task_cover[{task["task_id"]}]').X for task in tasks}
    duty_completion = {(duty['duty_id'], i): model.getVarByName(f'duty_completion[{duty["duty_id"]},{i}]').X for i, duty in enumerate(duties)}

    canceled_tasks = [task['task_id'] for task in tasks if task_cover[task['task_id']] < 0.5]

    solution_dict['released_problem_variant'] = instance['released_problem_variant']
    solution_dict['objective_value'] = model.ObjVal
    solution_dict['canceled_tasks'] = canceled_tasks
    solution_dict['duty_assignments'] = {}

    for duty in duties:
        for i in range(len(duties)):
            if duty_completion[(duty['duty_id'], i)] > 0.5:
                solution_dict['duty_assignments'][duty['duty_id']] = {
                    'completion_index': i,
                    'cost': compute_completion_cost(instance, duty, i),
                    # Other fields like 'covered_tasks', 'uses_taxi', 'release_at_relief_point', 'return_station', 'return_time_minutes' need to be computed based on the selected completion
                }

    return solution_dict

if __name__ == "__main__":
    main()