import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        return json.load(file)

def solve_clique(instance_data, time_limit, logger):
    graph_data = instance_data['graph']
    num_vertices = graph_data['num_vertices']
    adjacency_list = graph_data['adjacency_list']

    model = gp.Model()
    model.setParam('TimeLimit', time_limit)
    model.setParam('LogToConsole', 0)  # Disable console logging

    # Create decision variables
    x = model.addVars(num_vertices, vtype=GRB.BINARY, name='x')

    # Objective: Maximize the number of selected vertices
    model.setObjective(gp.quicksum(x), GRB.MAXIMIZE)

    # Constraints: For every pair of vertices not connected by an edge, at most one can be selected
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if str(j) not in adjacency_list[str(i)]:
                model.addConstr(x[i] + x[j] <= 1, f'edge_constraint_{i}_{j}')

    # Optimize the model
    model.optimize()

    # Retrieve the solution
    objective_value = model.getObjective().getValue()
    clique_vertices = [i for i in range(num_vertices) if x[i].X > 0.5]

    # Log intermediate solutions
    if logger:
        for i in range(model.SolCount):
            model.setParam('SolutionNumber', i)
            obj_val = model.getAttr('PoolObjVal')
            logger.log(obj_val)

    return objective_value, clique_vertices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', required=True, help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds.')
    parser.add_argument('--log_path', help='Path to log intermediate solutions (optional).')
    args = parser.parse_args()

    instance_data = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    objective_value, clique_vertices = solve_clique(instance_data, args.time_limit, logger)

    solution = {
        "objective_value": int(objective_value),
        "clique_vertices": clique_vertices
    }

    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == "__main__":
    main()