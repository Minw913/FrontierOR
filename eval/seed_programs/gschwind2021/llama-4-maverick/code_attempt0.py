import argparse
import json
import networkx as nx
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def generate_relaxed_cliques(G, relaxed_clique_type, parameter_s, connectivity_required):
    relaxed_cliques = []
    for node in G.nodes():
        for other_node in G.nodes():
            if node != other_node:
                try:
                    path = nx.shortest_path(G, source=node, target=other_node)
                    if len(path) <= parameter_s + 1:
                        clique = set(path)
                        if is_relaxed_clique(G, clique, relaxed_clique_type, parameter_s):
                            if not connectivity_required or is_connected(G, clique):
                                relaxed_cliques.append(clique)
                except nx.NetworkXNoPath:
                    continue
    # Also consider singletons as valid relaxed cliques
    for node in G.nodes():
        relaxed_cliques.append({node})
    return relaxed_cliques

def is_relaxed_clique(G, clique, relaxed_clique_type, parameter_s):
    subgraph = G.subgraph(clique)
    if relaxed_clique_type == 's-plex':
        for node in clique:
            if len(subgraph.neighbors(node)) < len(clique) - parameter_s:
                return False
        return True
    elif relaxed_clique_type == 's-clique':
        for node1 in clique:
            for node2 in clique:
                if node1 != node2 and nx.shortest_path_length(G, source=node1, target=node2) > parameter_s:
                    return False
        return True
    elif relaxed_clique_type == 'gamma-quasi-clique':
        gamma = parameter_s
        num_edges = subgraph.number_of_edges()
        max_edges = len(clique) * (len(clique) - 1) / 2
        if num_edges / max_edges >= gamma:
            return True
        return False
    # Add more types as needed
    else:
        raise NotImplementedError("Relaxed clique type not implemented")

def is_connected(G, clique):
    subgraph = G.subgraph(clique)
    return nx.is_connected(subgraph)

def formulate_mip(G, relaxed_cliques, decomposition_type):
    model = Model()
    model.setParam('OutputFlag', 0)
    
    clique_vars = []
    for i, clique in enumerate(relaxed_cliques):
        var = model.addVar(vtype=GRB.BINARY, name=f'clique_{i}')
        clique_vars.append(var)
    
    model.setObjective(quicksum(clique_vars), GRB.MINIMIZE)
    
    for node in G.nodes():
        node_cliques = [var for clique, var in zip(relaxed_cliques, clique_vars) if node in clique]
        if decomposition_type == 'partitioning':
            model.addConstr(quicksum(node_cliques) == 1, name=f'partition_{node}')
        elif decomposition_type == 'covering':
            model.addConstr(quicksum(node_cliques) >= 1, name=f'cover_{node}')
    
    return model, clique_vars

def main():
    args = parse_args()
    instance = read_instance(args.instance_path)
    
    G = nx.Graph()
    G.add_nodes_from(instance['graph']['vertices'])
    G.add_edges_from(instance['graph']['edges'])
    
    relaxed_clique_type = instance['problem_settings']['relaxed_clique_type']
    parameter_s = instance['problem_settings']['parameter_s']
    decomposition_type = instance['problem_settings']['decomposition_type']
    connectivity_required = instance['problem_settings']['connectivity_required']
    
    relaxed_cliques = generate_relaxed_cliques(G, relaxed_clique_type, parameter_s, connectivity_required)
    
    model, clique_vars = formulate_mip(G, relaxed_cliques, decomposition_type)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    model.setParam('TimeLimit', args.time_limit)
    model.optimize()
    
    selected_cliques = [clique for clique, var in zip(relaxed_cliques, clique_vars) if var.X > 0.5]
    
    solution = {
        "objective_value": model.ObjVal,
        "solution": [list(clique) for clique in selected_cliques]
    }
    
    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)
    
    if logger:
        logger.log(model.ObjVal)

if __name__ == '__main__':
    main()