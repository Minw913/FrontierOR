import json
import argparse
import networkx as nx
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def preprocess_instance(instance):
    n = instance['parameters']['n']
    num_arcs = instance['parameters']['num_arcs']
    num_od_pairs = instance['parameters']['num_od_pairs']
    R = instance['parameters']['R']
    p = instance['parameters']['p']
    nodes = instance['nodes']
    arcs = instance['arcs']
    od_nodes = instance['od_nodes']
    od_pairs = instance['od_pairs']

    G = nx.Graph()
    for arc in arcs:
        G.add_edge(arc['from'], arc['to'], weight=arc['distance'])

    od_pair_data = []
    for i, od_pair in enumerate(od_pairs):
        origin = od_pair['origin']
        destination = od_pair['destination']
        flow = od_pair['flow']
        shortest_path = od_pair['shortest_path']
        path_distance = od_pair['path_distance']

        round_trip_path = shortest_path + shortest_path[::-1][1:]
        round_trip_distance = 2 * path_distance

        od_pair_data.append({
            'origin': origin,
            'destination': destination,
            'flow': flow,
            'round_trip_path': round_trip_path,
            'round_trip_distance': round_trip_distance
        })

    return n, num_arcs, num_od_pairs, R, p, nodes, G, od_nodes, od_pair_data

def solve_refueling_station_location(n, p, nodes, G, od_pair_data, R, time_limit, logger):
    model = Model('RefuelingStationLocation')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    # Decision variables
    station_vars = {node['id']: model.addVar(vtype=GRB.BINARY, name=f"station_{node['id']}") for node in nodes}
    od_pair_vars = {i: model.addVar(vtype=GRB.BINARY, name=f"od_pair_{i}") for i in range(len(od_pair_data))}

    # Objective function
    model.setObjective(quicksum(od_pair['flow'] * od_pair_vars[i] for i, od_pair in enumerate(od_pair_data)), GRB.MAXIMIZE)

    # Constraints
    for i, od_pair in enumerate(od_pair_data):
        round_trip_path = od_pair['round_trip_path']
        round_trip_distance = od_pair['round_trip_distance']

        if round_trip_distance <= R:
            continue  # No need for refueling stations

        cum_dist = 0
        servicing_stations = set()
        for j in range(len(round_trip_path) - 1):
            cum_dist += G.get_edge_data(round_trip_path[j], round_trip_path[j+1])['weight']
            if cum_dist > R:
                break
        remaining_dist = cum_dist - R
        for k in range(j, len(round_trip_path) - 1):
            edge_dist = G.get_edge_data(round_trip_path[k], round_trip_path[k+1])['weight']
            for node in nodes:
                node_id = node['id']
                if node_id in round_trip_path[j:k+1]:
                    continue
                dist_to_node = nx.shortest_path_length(G, source=round_trip_path[k], target=node_id, weight='weight')
                dist_from_node = nx.shortest_path_length(G, source=node_id, target=round_trip_path[k+1], weight='weight')
                if dist_to_node + dist_from_node <= remaining_dist:
                    servicing_stations.add(node_id)
                    remaining_dist = edge_dist
                    break
            else:
                model.addConstr(od_pair_vars[i] == 0)
                break
        else:
            model.addConstr(quicksum(station_vars[node_id] for node_id in servicing_stations) >= od_pair_vars[i])

    model.addConstr(quicksum(station_vars[node['id']] for node in nodes) == p)

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        stations = [node_id for node_id, var in station_vars.items() if var.x > 0.5]
        refueled_od_pairs = [i for i, var in od_pair_vars.items() if var.x > 0.5]

        if logger:
            logger.log(objective_value)

        return {
            'objective_value': objective_value,
            'stations': stations,
            'refueled_od_pairs': refueled_od_pairs
        }
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    n, num_arcs, num_od_pairs, R, p, nodes, G, od_nodes, od_pair_data = preprocess_instance(instance)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    solution = solve_refueling_station_location(n, p, nodes, G, od_pair_data, R, args.time_limit, logger)

    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()