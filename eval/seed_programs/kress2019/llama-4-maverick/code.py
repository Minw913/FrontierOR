import argparse
import json
from gurobipy import Model, GRB, quicksum

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance, time_limit, log_path=None):
    S = instance['S']
    n = instance['n']
    m = instance['m']
    p = instance['p']
    sigma_w = instance['sigma_w']
    sigma_l = instance['sigma_l']
    seaside_containers = instance['seaside_containers']
    landside_containers = instance['landside_containers']

    model = Model('crane_scheduling')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit

    T = 2 * n * (S + 1 + 2 * p)
    crane_w_positions = model.addVars(range(T), name='crane_w_positions', vtype=GRB.INTEGER)
    crane_l_positions = model.addVars(range(T), name='crane_l_positions', vtype=GRB.INTEGER)
    makespan = model.addVar(name='makespan', vtype=GRB.INTEGER)

    model.setObjective(makespan, GRB.MINIMIZE)

    model.addConstr(crane_w_positions[0] == sigma_w)
    model.addConstr(crane_l_positions[0] == sigma_l)

    for t in range(T - 1):
        model.addConstr(crane_w_positions[t + 1] >= crane_w_positions[t] - 1)
        model.addConstr(crane_w_positions[t + 1] <= crane_w_positions[t] + 1)
        model.addConstr(crane_l_positions[t + 1] >= crane_l_positions[t] - 1)
        model.addConstr(crane_l_positions[t + 1] <= crane_l_positions[t] + 1)
        model.addConstr(crane_l_positions[t] - crane_w_positions[t] >= 1)

    seaside_drop_time = model.addVars(n, name='seaside_drop_time', vtype=GRB.INTEGER)
    seaside_drop_slot = model.addVars(n, name='seaside_drop_slot', vtype=GRB.INTEGER)
    for i in range(n):
        container = seaside_containers[i]
        target_slot = container['target_slot']
        model.addConstr(seaside_drop_time[i] >= 0)
        model.addConstr(seaside_drop_slot[i] >= 1)
        model.addConstr(seaside_drop_slot[i] <= S + 1)
        if i > 0:
            model.addConstr(seaside_drop_time[i] >= seaside_drop_time[i - 1])

    landside_lift_time = model.addVars(n, name='landside_lift_time', vtype=GRB.INTEGER)
    landside_lift_slot = model.addVars(n, name='landside_lift_slot', vtype=GRB.INTEGER)
    landside_drop_time = model.addVars(n, name='landside_drop_time', vtype=GRB.INTEGER)
    is_cooperative = model.addVars(n, name='is_cooperative', vtype=GRB.BINARY)
    for i in range(n):
        model.addConstr(landside_lift_time[i] >= seaside_drop_time[i])
        model.addConstr(is_cooperative[i] * (landside_lift_slot[i] - seaside_drop_slot[i]) == 0)
        model.addConstr(landside_drop_time[i] >= landside_lift_time[i] + p)
        model.addConstr(makespan >= seaside_drop_time[i] + p)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {
            'objective_value': model.getObjective().getValue(),
            'crane_w_positions': {t: crane_w_positions[t].X for t in range(T)},
            'crane_l_positions': {t: crane_l_positions[t].X for t in range(T)},
            'seaside_events': [],
            'landside_events': []
        }
        for i in range(n):
            seaside_event = {
                'container_id': i + 1,
                'target_slot': seaside_containers[i]['target_slot'],
                'seaside_drop_time': seaside_drop_time[i].X,
                'seaside_drop_slot': seaside_drop_slot[i].X,
            }
            if is_cooperative[i].X == 1:
                seaside_event['landside_lift_time'] = landside_lift_time[i].X
                seaside_event['landside_lift_slot'] = landside_lift_slot[i].X
                seaside_event['landside_drop_time'] = landside_drop_time[i].X
            solution['seaside_events'].append(seaside_event)
        return solution
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, args.log_path)
    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("No solution found.")

if __name__ == '__main__':
    main()