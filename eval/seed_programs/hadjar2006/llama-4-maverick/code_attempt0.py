import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def build_model(instance):
    model = Model("VehicleRouting")
    model.modelSense = GRB.MINIMIZE

    num_tasks = instance['num_tasks']
    num_depots = instance['num_depots']
    tasks = instance['tasks']
    depots = instance['depots']
    compatible_pairs = instance['compatible_pairs']
    depot_to_task_costs = instance['depot_to_task_costs']
    task_to_depot_costs = instance['task_to_depot_costs']

    # Create variables
    flow = {}
    for pair in compatible_pairs:
        task_i = pair['task_i']
        task_j = pair['task_j']
        flow[(task_i, task_j)] = model.addVar(vtype=GRB.BINARY, name=f"flow_{task_i}_{task_j}")

    for cost in depot_to_task_costs:
        depot_id = cost['depot_id']
        task_id = cost['task_id']
        flow[(depot_id, task_id)] = model.addVar(vtype=GRB.BINARY, name=f"flow_{depot_id}_{task_id}")

    for cost in task_to_depot_costs:
        task_id = cost['task_id']
        depot_id = cost['depot_id']
        flow[(task_id, depot_id)] = model.addVar(vtype=GRB.BINARY, name=f"flow_{task_id}_{depot_id}")

    # Objective function
    objective = quicksum(pair['cost'] * flow[(pair['task_i'], pair['task_j'])] for pair in compatible_pairs)
    objective += quicksum(cost['cost'] * flow[(cost['depot_id'], cost['task_id'])] for cost in depot_to_task_costs)
    objective += quicksum(cost['cost'] * flow[(cost['task_id'], cost['depot_id'])] for cost in task_to_depot_costs)
    model.setObjective(objective)

    # Constraints
    for task in tasks:
        task_id = task['id']
        inflows = quicksum(flow.get((i, task_id), 0) for i in range(-1, num_tasks) if (i, task_id) in flow)
        outflows = quicksum(flow.get((task_id, j), 0) for j in range(-1, num_depots) if (task_id, j) in flow)
        model.addConstr(inflows == outflows, f"balance_{task_id}")

    for depot in depots:
        depot_id = depot['id']
        inflows = quicksum(flow.get((i, depot_id), 0) for i in range(-1, num_tasks) if (i, depot_id) in flow)
        outflows = quicksum(flow.get((depot_id, j), 0) for j in range(num_tasks) if (depot_id, j) in flow)
        model.addConstr(inflows == outflows, f"balance_depot_{depot_id}")
        model.addConstr(outflows <= depot['capacity'], f"capacity_{depot_id}")

    for task in tasks:
        task_id = task['id']
        inflows = quicksum(flow.get((i, task_id), 0) for i in range(-1, num_tasks) if (i, task_id) in flow)
        model.addConstr(inflows == 1, f"cover_{task_id}")

    return model, flow

def extract_solution(instance, model, flow):
    routes = []
    tasks = instance['tasks']
    depots = instance['depots']

    for depot in depots:
        depot_id = depot['id']
        for task in tasks:
            task_id = task['id']
            if (depot_id, task_id) in flow and flow[(depot_id, task_id)].x > 0.5:
                route = {'depot': depot_id, 'tasks': [task_id]}
                current_task_id = task_id
                while True:
                    next_task_id = None
                    for pair in instance['compatible_pairs']:
                        if pair['task_i'] == current_task_id and flow[(pair['task_i'], pair['task_j'])].x > 0.5:
                            next_task_id = pair['task_j']
                            route['tasks'].append(next_task_id)
                            current_task_id = next_task_id
                            break
                    if next_task_id is None:
                        break
                for cost in instance['task_to_depot_costs']:
                    if cost['task_id'] == current_task_id and cost['depot_id'] == depot_id and flow[(current_task_id, depot_id)].x > 0.5:
                        route['cost'] = sum(instance['depot_to_task_costs'][i]['cost'] for i, cost in enumerate(instance['depot_to_task_costs']) if cost['depot_id'] == depot_id and cost['task_id'] == route['tasks'][0])
                        route['cost'] += sum(instance['compatible_pairs'][i]['cost'] for i, pair in enumerate(instance['compatible_pairs']) if pair['task_i'] == route['tasks'][i] and pair['task_j'] == route['tasks'][i+1] for i in range(len(route['tasks'])-1))
                        route['cost'] += sum(instance['task_to_depot_costs'][i]['cost'] for i, cost in enumerate(instance['task_to_depot_costs']) if cost['task_id'] == route['tasks'][-1] and cost['depot_id'] == depot_id)
                        routes.append(route)
                        break

    objective_value = model.ObjVal

    solution = {'objective_value': objective_value, 'routes': routes}
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    model, flow = build_model(instance)

    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    model.Params.TimeLimit = args.time_limit
    model.optimize(lambda model, where: callback(model, where, logger))

    solution = extract_solution(instance, model, flow)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

def callback(model, where, logger):
    if where == GRB.Callback.MIPSOL:
        objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if logger:
            logger.log(objective_value)

if __name__ == "__main__":
    main()