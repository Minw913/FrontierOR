import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
import time
import networkx as nx

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def generate_candidate_lines(instance):
    candidate_lines = {}
    for mode in instance['modes']:
        mode_name = mode['name']
        candidate_lines[mode_name] = []
        G_mode = nx.Graph()
        for edge_idx in mode['edge_indices']:
            edge = instance['network']['edges'][edge_idx]
            G_mode.add_edge(edge['endpoints'][0], edge['endpoints'][1])
        terminals = set(mode['terminals'])
        for terminal1 in terminals:
            for terminal2 in terminals:
                if terminal1 == terminal2:
                    continue
                try:
                    paths = list(nx.all_simple_paths(G_mode, terminal1, terminal2, cutoff=instance['global_parameters']['max_line_length_edges']))
                    for path in paths:
                        line_edges = []
                        for i in range(len(path) - 1):
                            node1 = path[i]
                            node2 = path[i+1]
                            edge_idx = next((e['id'] for e in instance['network']['edges'] if sorted(e['endpoints']) == sorted([node1, node2])), None)
                            if edge_idx is not None:
                                line_edges.append(edge_idx)
                        candidate_lines[mode_name].append({'nodes': path, 'edges': line_edges})
                except nx.NetworkXNoPath:
                    continue
    return candidate_lines

def build_model(instance, candidate_lines):
    model = Model('TransitNetworkDesign')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    
    # Line activation and frequency variables
    line_vars = {}
    for mode, lines in candidate_lines.items():
        line_vars[mode] = {}
        for i, line in enumerate(lines):
            line_var = model.addVar(vtype=GRB.BINARY, name=f'line_{mode}_{i}_active')
            freq_var = model.addVar(lb=0, ub=instance['global_parameters']['frequency_upper_bound_F'], name=f'line_{mode}_{i}_freq')
            line_vars[mode][i] = (line_var, freq_var)
    
    # Passenger path flow variables
    passenger_flow_vars = {}
    od_pairs = [(od['origin'], od['destination']) for od in instance['od_matrix']]
    for origin, destination in od_pairs:
        passenger_flow_vars[(origin, destination)] = {}
        paths = get_passenger_paths(instance, origin, destination)
        for path_idx, path in enumerate(paths):
            flow_var = model.addVar(lb=0, name=f'passenger_flow_{origin}_{destination}_{path_idx}')
            passenger_flow_vars[(origin, destination)][path_idx] = flow_var
    
    # Objective: weighted combination of total line cost and total passenger travel time
    lambda_ = instance['global_parameters']['lambda']
    total_line_cost = get_total_line_cost(instance, candidate_lines, line_vars)
    total_passenger_travel_time = get_total_passenger_travel_time(instance, passenger_flow_vars)
    model.setObjective(lambda_ * total_line_cost + (1 - lambda_) * total_passenger_travel_time, GRB.MINIMIZE)
    
    # Constraints
    add_line_frequency_constraints(model, instance, candidate_lines, line_vars)
    add_edge_capacity_constraints(model, instance, candidate_lines, line_vars)
    add_passenger_flow_constraints(model, instance, passenger_flow_vars, od_pairs)
    add_passenger_capacity_constraints(model, instance, candidate_lines, line_vars, passenger_flow_vars)
    
    return model, line_vars, passenger_flow_vars

def get_passenger_paths(instance, origin, destination):
    G_passenger = nx.DiGraph()
    for edge in instance['network']['edges']:
        G_passenger.add_edge(edge['endpoints'][0], edge['endpoints'][1], traveling_time=edge['traveling_time_seconds'])
        G_passenger.add_edge(edge['endpoints'][1], edge['endpoints'][0], traveling_time=edge['traveling_time_seconds'])
    try:
        paths = list(nx.all_simple_paths(G_passenger, origin, destination))
        return paths
    except nx.NetworkXNoPath:
        return []

def get_total_line_cost(instance, candidate_lines, line_vars):
    total_line_cost = 0
    for mode, lines in candidate_lines.items():
        mode_fixed_cost = instance['modes'][instance['modes_names'].index(mode)]['fixed_cost_per_line']
        for i, line in enumerate(lines):
            line_var, freq_var = line_vars[mode][i]
            line_operating_cost = sum(instance['network']['edges'][edge_idx]['operating_cost'] for edge_idx in line['edges'])
            total_line_cost += mode_fixed_cost * line_var + line_operating_cost * freq_var
    return total_line_cost

def get_total_passenger_travel_time(instance, passenger_flow_vars):
    total_passenger_travel_time = 0
    od_pairs = [(od['origin'], od['destination']) for od in instance['od_matrix']]
    G_passenger = nx.DiGraph()
    for edge in instance['network']['edges']:
        G_passenger.add_edge(edge['endpoints'][0], edge['endpoints'][1], traveling_time=edge['traveling_time_seconds'])
        G_passenger.add_edge(edge['endpoints'][1], edge['endpoints'][0], traveling_time=edge['traveling_time_seconds'])
    for origin, destination in od_pairs:
        for path_idx, flow_var in passenger_flow_vars[(origin, destination)].items():
            path = get_passenger_path(instance, origin, destination, path_idx, G_passenger)
            path_travel_time = sum(G_passenger[u][v]['traveling_time'] for u, v in zip(path, path[1:]))
            total_passenger_travel_time += path_travel_time * flow_var
    return total_passenger_travel_time

def get_passenger_path(instance, origin, destination, path_idx, G_passenger):
    paths = list(nx.all_simple_paths(G_passenger, origin, destination))
    return paths[path_idx]

def add_line_frequency_constraints(model, instance, candidate_lines, line_vars):
    for mode, lines in candidate_lines.items():
        for i, line in enumerate(lines):
            line_var, freq_var = line_vars[mode][i]
            model.addConstr(freq_var <= instance['global_parameters']['frequency_upper_bound_F'] * line_var)

def add_edge_capacity_constraints(model, instance, candidate_lines, line_vars):
    for edge in instance['network']['edges']:
        edge_idx = edge['id']
        total_freq = 0
        for mode, lines in candidate_lines.items():
            for i, line in enumerate(lines):
                if edge_idx in line['edges']:
                    _, freq_var = line_vars[mode][i]
                    total_freq += freq_var
        model.addConstr(total_freq <= edge['edge_capacity'])

def add_passenger_flow_constraints(model, instance, passenger_flow_vars, od_pairs):
    for origin, destination in od_pairs:
        demand = next(od['demand'] for od in instance['od_matrix'] if od['origin'] == origin and od['destination'] == destination)
        total_flow = quicksum(flow_var for flow_var in passenger_flow_vars[(origin, destination)].values())
        model.addConstr(total_flow == demand)

def add_passenger_capacity_constraints(model, instance, candidate_lines, line_vars, passenger_flow_vars):
    od_pairs = [(od['origin'], od['destination']) for od in instance['od_matrix']]
    G_passenger = nx.DiGraph()
    edge_map = {}
    for edge in instance['network']['edges']:
        edge_map[(edge['endpoints'][0], edge['endpoints'][1])] = edge['id']
        edge_map[(edge['endpoints'][1], edge['endpoints'][0])] = edge['id']
        G_passenger.add_edge(edge['endpoints'][0], edge['endpoints'][1], edge_id=edge['id'])
        G_passenger.add_edge(edge['endpoints'][1], edge['endpoints'][0], edge_id=edge['id'])
    for edge in instance['network']['edges']:
        edge_id = edge['id']
        edge_capacity = 0
        for mode, lines in candidate_lines.items():
            vehicle_capacity = next(m['vehicle_capacity'] for m in instance['modes'] if m['name'] == mode)
            for i, line in enumerate(lines):
                if edge_id in line['edges']:
                    _, freq_var = line_vars[mode][i]
                    edge_capacity += vehicle_capacity * freq_var
        total_passenger_flow = 0
        for origin, destination in od_pairs:
            for path_idx, flow_var in passenger_flow_vars[(origin, destination)].items():
                path = get_passenger_path(instance, origin, destination, path_idx, G_passenger)
                for u, v in zip(path, path[1:]):
                    if G_passenger[u][v]['edge_id'] == edge_id:
                        total_passenger_flow += flow_var
        model.addConstr(total_passenger_flow <= edge_capacity)

def solve_model(model, time_limit):
    model.Params.TimeLimit = time_limit
    model.optimize()

def extract_solution(instance, model, line_vars, passenger_flow_vars, candidate_lines):
    solution = {'objective_value': model.ObjVal, 'active_lines': [], 'active_passenger_paths': []}
    for mode, lines in candidate_lines.items():
        for i, line in enumerate(lines):
            line_var, freq_var = line_vars[mode][i]
            if freq_var.X > 1e-6:
                solution['active_lines'].append({
                    'line_index': i,
                    'mode': mode,
                    'nodes': line['nodes'],
                    'edges': line['edges'],
                    'frequency': freq_var.X
                })
    od_pairs = [(od['origin'], od['destination']) for od in instance['od_matrix']]
    G_passenger = nx.DiGraph()
    for edge in instance['network']['edges']:
        G_passenger.add_edge(edge['endpoints'][0], edge['endpoints'][1])
        G_passenger.add_edge(edge['endpoints'][1], edge['endpoints'][0])
    for origin, destination in od_pairs:
        for path_idx, flow_var in passenger_flow_vars[(origin, destination)].items():
            if flow_var.X > 1e-6:
                path = get_passenger_path(instance, origin, destination, path_idx, G_passenger)
                arcs = [(path[i], path[i+1]) for i in range(len(path)-1)]
                solution['active_passenger_paths'].append({
                    'origin': origin,
                    'destination': destination,
                    'arcs': arcs,
                    'flow': flow_var.X
                })
    return solution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()
    
    instance = read_instance(args.instance_path)
    instance['modes_names'] = [mode['name'] for mode in instance['modes']]
    
    candidate_lines = generate_candidate_lines(instance)
    model, line_vars, passenger_flow_vars = build_model(instance, candidate_lines)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    solve_model(model, args.time_limit)
    end_time = time.time()
    
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        solution = extract_solution(instance, model, line_vars, passenger_flow_vars, candidate_lines)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        if logger:
            logger.log_solution(model.ObjVal, solution)
    else:
        print("No feasible solution found.")

if __name__ == '__main__':
    main()