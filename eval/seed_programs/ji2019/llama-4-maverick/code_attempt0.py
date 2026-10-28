import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    num_ships = instance['num_ships']
    num_berths = instance['num_berths']
    chamber_width = instance['chamber_width_W']
    chamber_length = instance['chamber_length_L']
    lockage_duration = instance['lockage_duration_Du']
    min_time_between_lockages = instance['min_time_between_lockages_MT']
    lock_start_availability = instance['lock_start_availability_sc']
    ships = instance['ships']
    berths = instance['berths']
    handling_times = instance['handling_times']

    model = Model('InlandWaterwayDamOptimization')
    model.modelSense = GRB.MINIMIZE

    # Decision Variables
    modes = model.addVars(num_ships, vtype=GRB.BINARY, name='mode')  # 1 for transshipment, 0 for lock
    lockage_assignments = model.addVars(num_ships, num_ships, vtype=GRB.BINARY, name='lockage_assignment')
    lockage_used = model.addVars(num_ships, vtype=GRB.BINARY, name='lockage_used')
    lockage_completion_times = model.addVars(num_ships, vtype=GRB.CONTINUOUS, name='lockage_completion_time')
    berth_assignments = model.addVars(num_ships, num_berths, num_ships, vtype=GRB.BINARY, name='berth_assignment')
    ship_positions_x = model.addVars(num_ships, vtype=GRB.INTEGER, name='ship_position_x')
    ship_positions_y = model.addVars(num_ships, vtype=GRB.INTEGER, name='ship_position_y')
    ship_departure_times = model.addVars(num_ships, vtype=GRB.CONTINUOUS, name='ship_departure_time')
    berth_service_start_times = model.addVars(num_ships, num_berths, vtype=GRB.CONTINUOUS, name='berth_service_start_time')
    berth_idle_times = model.addVars(num_ships, num_berths, num_ships, vtype=GRB.CONTINUOUS, name='berth_idle_time')

    # Objective Function
    total_delay_lock_mode = quicksum((ship_departure_times[i] - ships[i]['arrival_time']) * (1 - modes[i]) for i in range(num_ships))
    total_delay_transshipment_mode = quicksum(
        (berth_service_start_times[i, k] + handling_times[i][k] - ships[i]['arrival_time']) * modes[i] * berth_assignments[i, k, o]
        for i in range(num_ships)
        for k in range(num_berths)
        for o in range(num_ships)
    )
    total_transshipment_penalty = quicksum(ships[i]['transshipment_penalty'] * modes[i] for i in range(num_ships))
    model.setObjective(total_delay_lock_mode + total_delay_transshipment_mode + total_transshipment_penalty)

    # Constraints
    # 1. Mode selection and assignment
    for i in range(num_ships):
        model.addConstr(quicksum(lockage_assignments[i, l] for l in range(num_ships)) + quicksum(berth_assignments[i, k, o] for k in range(num_berths) for o in range(num_ships)) == 1)

    # 2. Lockage assignments and completion times
    for l in range(num_ships):
        model.addConstr(lockage_used[l] >= quicksum(lockage_assignments[i, l] for i in range(num_ships)))
        model.addConstr(lockage_completion_times[l] >= lock_start_availability + lockage_duration + l * min_time_between_lockages)
        for i in range(num_ships):
            model.addConstr(lockage_completion_times[l] >= ships[i]['arrival_time'] + lockage_duration - (1 - lockage_assignments[i, l]) * instance['C_max'])

    for i in range(num_ships):
        for l in range(num_ships):
            model.addConstr(ship_departure_times[i] >= lockage_completion_times[l] - (1 - lockage_assignments[i, l]) * instance['C_max'])

    # 3. Berth assignments and service times
    for i in range(num_ships):
        for k in range(num_berths):
            for o in range(num_ships):
                model.addConstr(berth_service_start_times[i, k] >= berths[k]['start_availability'] + quicksum(handling_times[j][k] + berth_idle_times[j, k, o2] for j in range(num_ships) for o2 in range(o)) - (1 - berth_assignments[i, k, o]) * instance['C_max'])

    # 4. Non-overlap and mooring constraints within lockage
    for l in range(num_ships):
        for i in range(num_ships):
            for j in range(i + 1, num_ships):
                model.addConstr(ship_positions_x[i] + ships[i]['width'] <= ship_positions_x[j] + (1 - lockage_assignments[i, l]) * chamber_width)
                model.addConstr(ship_positions_y[i] + ships[i]['length'] <= ship_positions_y[j] + (1 - lockage_assignments[i, l]) * chamber_length)

    # 5. First-come-first-served and lockage order constraints
    for i in range(num_ships):
        for j in range(num_ships):
            if ships[i]['arrival_time'] < ships[j]['arrival_time']:
                model.addConstr(ship_departure_times[i] <= ship_departure_times[j] + (modes[i] + modes[j]) * instance['C_max'])
                for l in range(num_ships):
                    model.addConstr(quicksum(lockage_assignments[i, l2] for l2 in range(l)) >= lockage_assignments[i, l] - (1 - lockage_assignments[j, l]) * num_ships)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {
            'objective_value': model.objVal,
            'modes': {i: int(modes[i].x) for i in range(num_ships)},
            'completion_times': {i: ship_departure_times[i].x for i in range(num_ships)},
            'lockage_assignments': {i: int(lockage_assignments[i, l].x) for i in range(num_ships) for l in range(num_ships) if lockage_assignments[i, l].x > 0.5},
            'lockage_completion_times': {l: lockage_completion_times[l].x for l in range(num_ships) if lockage_used[l].x > 0.5},
            'berth_assignments': {}
        }
        for i in range(num_ships):
            for k in range(num_berths):
                for o in range(num_ships):
                    if berth_assignments[i, k, o].x > 0.5:
                        solution['berth_assignments'][i] = {'berth': k, 'order': o}
        return solution
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    instance = read_instance(args.instance_path)
    model = Model('InlandWaterwayDamOptimization')
    model.Params.TimeLimit = args.time_limit

    solution = solve(instance)

    if solution is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
        if logger:
            logger.log(solution['objective_value'])

if __name__ == '__main__':
    main()