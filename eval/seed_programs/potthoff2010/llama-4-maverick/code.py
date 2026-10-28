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
    if model is None:
        print("Model building failed.")
        return
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
    try:
        model = Model("Crew Rescheduling")
        duties = instance['active_duties'] + instance['reserve_duties']
        tasks = instance['tasks']

        task_cover = model.addVars([task['task_id'] for task in tasks], vtype=GRB.BINARY, name='task_cover')
        duty_completion = model.addVars([duty['duty_id'] for duty in duties], vtype=GRB.INTEGER, lb=0, ub=len(duties)-1, name='duty_completion')

        model.setObjective(
            quicksum(instance['parameters']['cost_cancel_A_to_B'] * (1 - task_cover[task['task_id']]) for task in tasks if task['task_type'] == 'A_to_B') +
            quicksum(instance['parameters']['cost_cancel_A_to_A'] * (1 - task_cover[task['task_id']]) for task in tasks if task['task_type'] == 'A_to_A'),
            GRB.MINIMIZE
        )

        for task in tasks:
            task_id = task['task_id']
            model.addConstr(quicksum(duty_completion.sum('*') for duty in duties if task_id in duty.get('tasks', [])) >= task_cover[task_id], name=f'cover_{task_id}')

        for duty in duties:
            model.addConstr(duty_completion[duty['duty_id']] >= 0, name=f'duty_completion_lb_{duty["duty_id"]}')
            model.addConstr(duty_completion[duty['duty_id']] <= len(duties)-1, name=f'duty_completion_ub_{duty["duty_id"]}')

        return model, {'released_problem_variant': instance['released_problem_variant'], 'duty_assignments': {duty['duty_id']: None for duty in duties}}
    except Exception as e:
        print(f"Error building model: {e}")
        return None, None

def callback(model, where, logger):
    if where == GRB.Callback.MIPSOL:
        objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if logger:
            logger.log(objective_value)

def retrieve_solution(instance, model, solution_dict):
    duties = instance['active_duties'] + instance['reserve_duties']
    tasks = instance['tasks']

    task_cover = {task['task_id']: model.getVarByName(f'task_cover[{task["task_id"]}]').X for task in tasks}
    duty_completion = {duty['duty_id']: round(model.getVarByName(f'duty_completion[{duty["duty_id"]}]').X) for duty in duties}

    canceled_tasks = [task['task_id'] for task in tasks if task_cover[task['task_id']] < 0.5]

    solution_dict['objective_value'] = model.ObjVal
    solution_dict['canceled_tasks'] = canceled_tasks
    solution_dict['duty_assignments'] = {}

    for duty in duties:
        completion_index = duty_completion[duty['duty_id']]
        solution_dict['duty_assignments'][duty['duty_id']] = {
            'completion_index': completion_index,
            'cost': 0,
            'covered_tasks': [],
            'uses_taxi': False,
            'release_at_relief_point': False,
            'return_station': duty.get('crew_base'),
            'return_time_minutes': duty.get('end_time_minutes', 0)
        }

    return solution_dict

if __name__ == "__main__":
    main()