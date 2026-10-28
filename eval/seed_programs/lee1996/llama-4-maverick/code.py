import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger
import itertools

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve(instance, time_limit, log_path):
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    m = instance['parameters']['m']
    n = instance['parameters']['n']
    target_nodes = instance['target_nodes']['coordinates']
    steiner_nodes = instance['steiner_nodes']['coordinates']
    node_weights = instance['steiner_nodes']['node_weights']
    assignment_costs = instance['cost_matrices']['assignment_costs']['matrix']
    interconnection_costs = instance['cost_matrices']['interconnection_costs']['matrix']

    model = gp.Model()
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  
    
    x = model.addVars(m, n, vtype=GRB.BINARY, name="x")  
    y = model.addVars(n, vtype=GRB.BINARY, name="y")  
    z = model.addVars(n, n, vtype=GRB.BINARY, name="z", symmetric=True)  
    
    model.setObjective(
        gp.quicksum(assignment_costs[i][j] * x[i, j] for i in range(m) for j in range(n)) +
        gp.quicksum(interconnection_costs[j1][j2] * z[j1, j2] for j1 in range(n) for j2 in range(j1 + 1, n)) +
        gp.quicksum(node_weights[j] * y[j] for j in range(n)),
        GRB.MINIMIZE
    )
    
    for i in range(m):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f"target_assignment_{i}")
    for i in range(m):
        for j in range(n):
            model.addConstr(x[i, j] <= y[j], name=f"assignment_activation_{i}_{j}")
    for j1 in range(n):
        for j2 in range(j1 + 1, n):
            model.addConstr(z[j1, j2] <= y[j1], name=f"edge_activation_start_{j1}_{j2}")
            model.addConstr(z[j1, j2] <= y[j2], name=f"edge_activation_end_{j1}_{j2}")
    model.addConstr(gp.quicksum(z[j1, j2] for j1 in range(n) for j2 in range(j1 + 1, n)) == gp.quicksum(y[j] for j in range(n)) - 1, name="spanning_tree_edges")
    for subset_size in range(3, n + 1):
        for subset in itertools.combinations(range(n), subset_size):
            for excluded_node in subset:
                included_nodes = [node for node in subset if node != excluded_node]
                model.addConstr(
                    gp.quicksum(z[min(j1, j2), max(j1, j2)] for j1 in included_nodes for j2 in included_nodes if j1 < j2) <= 
                    gp.quicksum(y[j] for j in included_nodes) - 1, 
                    name=f"subtour_elimination_{subset}_{excluded_node}"
                )
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            active_steiner_nodes = [j for j in range(n) if y[j].X > 0.5]
            assignments = {i: np.argmax([x[i, j].X for j in range(n)]) for i in range(m)}
            steiner_tree_edges = [[j1, j2] for j1 in range(n) for j2 in range(j1 + 1, n) if z[j1, j2].X > 0.5]
            objective_value = model.ObjVal
            
            if logger:
                logger.log(objective_value)
            
            return {
                "objective_value": objective_value,
                "active_steiner_nodes": active_steiner_nodes,
                "assignments": {str(i): int(assignments[i]) for i in range(m)},
                "steiner_tree_edges": steiner_tree_edges
            }
        else:
            raise Exception("No solution found")
    else:
        raise Exception(f"Model status: {model.status}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()
    
    instance = read_instance(args.instance_path)
    try:
        solution = solve(instance, args.time_limit, args.log_path)
        with open(args.solution_path, 'w') as file:
            json.dump(solution, file, indent=4)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()