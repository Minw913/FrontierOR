import json
import argparse
import time
import random
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def greedy_clique(num_vertices, weights, adj):
    best_clique = []
    best_weight = 0
    
    # Try multiple random orders to find a good maximal clique
    for _ in range(20):
        order = list(range(num_vertices))
        # Sort by weight with some random tie-breaking to diversify searches
        order.sort(key=lambda v: (weights[v] + random.uniform(0, 1.0)), reverse=True)
        
        clique = []
        for v in order:
            if all(v in adj[u] for u in clique):
                clique.append(v)
        
        weight = sum(weights[v] for v in clique)
        if weight > best_weight:
            best_weight = weight
            best_clique = clique
            
    return best_clique, best_weight

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    graph = instance['graph']
    num_vertices = graph['num_vertices']
    edges = graph['edges']
    weights = instance['vertex_weights']

    start_time = time.time()
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time <= 0:
        remaining_time = 1

    adj = [set() for _ in range(num_vertices)]
    edge_set = set()
    for e in edges:
        u, v = e[0], e[1]
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        edge_set.add((v, u))

    # Compute initial heuristic solution
    best_clique, best_weight = greedy_clique(num_vertices, weights, adj)
    
    if best_clique:
        sol_dict = {
            "objective_value": best_weight,
            "clique_vertices": best_clique
        }
        if logger:
            logger.log_solution(best_weight, sol_dict)

    m = Model('max_weight_clique')
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = remaining_time
    m.Params.OutputFlag = 0

    x = m.addVars(num_vertices, vtype=GRB.BINARY, name='x')
    m.setObjective(quicksum(weights[i] * x[i] for i in range(num_vertices)), GRB.MAXIMIZE)

    # Add non-edge constraints: x_i + x_j <= 1 for all non-edges
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if (i, j) not in edge_set:
                m.addConstr(x[i] + x[j] <= 1)

    if best_clique:
        start = []
        for i in range(num_vertices):
            start.append(1.0 if i in best_clique else 0.0)
        m.addMipStart(x.keys(), start)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol = model.cbGet(GRB.Callback.MIPSOL_X)
            clique = [i for i in range(num_vertices) if sol[i] > 0.5]
            sol_dict = {
                "objective_value": obj,
                "clique_vertices": clique
            }
            if logger:
                logger.log_solution(obj, sol_dict)

    m.optimize(callback)

    if m.SolCount > 0:
        obj = m.objVal
        clique = [i for i in range(num_vertices) if x[i].X > 0.5]
    else:
        obj = best_weight
        clique = best_clique

    solution = {
        "objective_value": obj,
        "clique_vertices": clique
    }

    if logger:
        logger.log_solution(obj, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()