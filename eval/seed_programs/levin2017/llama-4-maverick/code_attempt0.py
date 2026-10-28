import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Solve the SAV optimization problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path where the solution JSON file will be written.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, help='Path to the log JSONL file for intermediate solutions.')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    model, solution_vars = build_model(instance_data)

    model.setParam('TimeLimit', args.time_limit)
    model.optimize(lambda model, where: callback(model, where, logger))

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = extract_solution(instance_data, solution_vars)
        solution['objective_value'] = model.objVal
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("No feasible solution found.")

def build_model(instance_data):
    model = Model('SAV_Optimization')
    model.modelSense = GRB.MINIMIZE

    num_nodes = instance_data['network']['num_nodes']
    num_links = instance_data['network']['num_links']
    num_centroids = instance_data['network']['num_centroids']
    time_horizon_T = instance_data['time_parameters']['time_horizon_T']
    nodes = instance_data['network']['nodes']
    links = instance_data['network']['links']
    centroids = [node['id'] for node in nodes if node['type'] == 'centroid']
    od_pairs = instance_data['demand']['od_pairs']
    initial_parked = instance_data['fleet']['initial_distribution']

    # Create dictionaries for quick lookup
    link_dict = {(link['from'], link['to']): link for link in links}
    node_dict = {node['id']: node for node in nodes}

    # Variables
    y = model.addVars([((link['from'], link['to']), l['to'], d, t) for link in links for l in links if l['from'] == link['to'] for d in centroids for t in range(time_horizon_T)], name='y', lb=0)
    y_centroid = model.addVars([(c, link['to'], d, t) for c in centroids for link in links if link['from'] == c for d in centroids for t in range(time_horizon_T)], name='y_centroid', lb=0)
    N_U = model.addVars([((link['from'], link['to']), d, t) for link in links for d in centroids for t in range(time_horizon_T+1)], name='N_U', lb=0)
    N_D = model.addVars([((link['from'], link['to']), d, t) for link in links for d in centroids for t in range(time_horizon_T+1)], name='N_D', lb=0)
    p = model.addVars([(c, t) for c in centroids for t in range(time_horizon_T+1)], name='p', lb=0)
    e = model.addVars([(o, d, t) for od in od_pairs for o, d in [(od['origin'], od['destination'])] for t in range(time_horizon_T)], name='e', lb=0)
    omega = model.addVars([(o, d, t) for od in od_pairs for o, d in [(od['origin'], od['destination'])] for t in range(time_horizon_T+1)], name='omega', lb=0)

    # Objective Function
    model.setObjective(quicksum((N_U[(link['from'], link['to']), d, t] - N_D[(link['from'], link['to']), d, t]) for link in links for d in centroids for t in range(time_horizon_T)) +
                       quicksum(omega[o, d, t] for o, d in [(od['origin'], od['destination']) for od in od_pairs] for t in range(time_horizon_T)), GRB.MINIMIZE)

    # Constraints
    # 1. Cumulative Count Updates
    for link in links:
        for d in centroids:
            for t in range(time_horizon_T):
                model.addConstr(N_U[(link['from'], link['to']), d, t+1] == N_U[(link['from'], link['to']), d, t] + quicksum(y[((l['from'], l['to']), link['from'], d, t)] for l in links if l['to'] == link['from']))
                model.addConstr(N_D[(link['from'], link['to']), d, t+1] == N_D[(link['from'], link['to']), d, t] + quicksum(y[((link['from'], link['to']), l['to'], d, t)] for l in links if l['from'] == link['to']))

    for c in centroids:
        for d in centroids:
            for t in range(time_horizon_T):
                model.addConstr(N_D[(c, link['to']), d, t+1] == N_U[(c, link['to']), d, t] for link in links if link['from'] == c)

    # 2. Sending Flow and Capacity Constraints
    for link in links:
        for d in centroids:
            for t in range(link['free_flow_travel_time_steps'], time_horizon_T):
                sending_flow = N_U[(link['from'], link['to']), d, t-link['free_flow_travel_time_steps']+1] - N_D[(link['from'], link['to']), d, t]
                model.addConstr(quicksum(y[((link['from'], link['to']), l['to'], d, t)] for l in links if l['from'] == link['to']) <= sending_flow)

    for link in links:
        for t in range(time_horizon_T):
            model.addConstr(quicksum(y[((link['from'], link['to']), l['to'], d, t)] for l in links if l['from'] == link['to'] for d in centroids) <= link['capacity_vph'] / 3600 * instance_data['time_parameters']['time_step_duration_seconds'])
            model.addConstr(quicksum(y[((l['from'], l['to']), link['from'], d, t)] for l in links if l['to'] == link['from'] for d in centroids) <= link['capacity_vph'] / 3600 * instance_data['time_parameters']['time_step_duration_seconds'])

    # 3. Congested Wave Constraint
    for link in links:
        for t in range(time_horizon_T):
            congested_wave_bound = quicksum(N_U[(link['from'], link['to']), d, t-link['congested_travel_time_steps']+1] - N_D[(link['from'], link['to']), d, t] for d in centroids) + link['jam_density_vehicles'] * link['length_ft']
            model.addConstr(quicksum(y[((l['from'], l['to']), link['from'], d, t)] for l in links if l['to'] == link['from'] for d in centroids) <= congested_wave_bound)

    # 4. Parked Vehicles Update
    for c in centroids:
        for t in range(time_horizon_T):
            arriving_vehicles = quicksum(N_D[(link['from'], c), c, t+1] - N_U[(link['from'], c), c, t+1] for link in links if link['to'] == c)
            departing_vehicles = quicksum(y_centroid[c, link['to'], d, t] for link in links if link['from'] == c for d in centroids)
            model.addConstr(p[c, t+1] == p[c, t] + arriving_vehicles - departing_vehicles)

    # 5. Vehicle Departure from Centroids
    for c in centroids:
        for t in range(time_horizon_T):
            model.addConstr(quicksum(y_centroid[c, link['to'], d, t] for link in links if link['from'] == c for d in centroids) <= p[c, t])

    # 6. Traveler Pickup and Waiting Demand Update
    for od in od_pairs:
        o, d = od['origin'], od['destination']
        for t in range(time_horizon_T):
            model.addConstr(e[o, d, t] <= omega[o, d, t])
            model.addConstr(e[o, d, t] <= quicksum(y_centroid[o, l['to'], d, t] for l in links if l['from'] == o))
            model.addConstr(omega[o, d, t+1] == omega[o, d, t] + od['num_travelers'][t] - e[o, d, t])

    # Initial Conditions
    for link in links:
        for d in centroids:
            model.addConstr(N_U[(link['from'], link['to']), d, 0] == 0)
            model.addConstr(N_D[(link['from'], link['to']), d, 0] == 0)

    for c in centroids:
        model.addConstr(p[c, 0] == initial_parked.get(c, 0))

    for od in od_pairs:
        o, d = od['origin'], od['destination']
        model.addConstr(omega[o, d, 0] == 0)

    # Final Conditions
    for c in centroids:
        model.addConstr(p[c, time_horizon_T] == initial_parked.get(c, 0))

    for od in od_pairs:
        o, d = od['origin'], od['destination']
        model.addConstr(omega[o, d, time_horizon_T] == 0)

    return model, (y, y_centroid, N_U, N_D, p, e, omega)

def callback(model, where, logger):
    if where == GRB.Callback.MIP:
        obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
        if model._logger:
            model._logger.log(obj_best)

def extract_solution(instance_data, solution_vars):
    y, y_centroid, N_U, N_D, p, e, omega = solution_vars
    time_horizon_T = instance_data['time_parameters']['time_horizon_T']
    links = instance_data['network']['links']
    nodes = instance_data['network']['nodes']
    centroids = [node['id'] for node in nodes if node['type'] == 'centroid']
    od_pairs = instance_data['demand']['od_pairs']

    solution = {
        'y': {(l['from'] + '|' + l['to'] + '|' + l2['to'] + '|' + d + '|' + str(t)): y[(l['from'], l['to']), l2['to'], d, t].X for l in links for l2 in links if l2['from'] == l['to'] for d in centroids for t in range(time_horizon_T)},
        'y_centroid': {(c + '|' + l['to'] + '|' + d + '|' + str(t)): y_centroid[c, l['to'], d, t].X for c in centroids for l in links if l['from'] == c for d in centroids for t in range(time_horizon_T)},
        'N_U': {((l['from'] + '|' + l['to'] + '|' + d + '|' + str(t)): N_U[(l['from'], l['to']), d, t].X for l in links for d in centroids for t in range(time_horizon_T+1))},
        'N_D': {((l['from'] + '|' + l['to'] + '|' + d + '|' + str(t)): N_D[(l['from'], l['to']), d, t].X for l in links for d in centroids for t in range(time_horizon_T+1))},
        'p': {(c + '|' + str(t)): p[c, t].X for c in centroids for t in range(time_horizon_T+1)},
        'e': {(o + '|' + d + '|' + str(t)): e[o, d, t].X for od in od_pairs for o, d in [(od['origin'], od['destination'])] for t in range(time_horizon_T)},
        'omega': {(o + '|' + d + '|' + str(t)): omega[o, d, t].X for od in od_pairs for o, d in [(od['origin'], od['destination'])] for t in range(time_horizon_T+1)}
    }
    return solution

if __name__ == '__main__':
    main()