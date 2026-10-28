import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    num_vertices = instance['num_vertices']
    num_edges = instance['num_edges']
    interdiction_budget_k = instance['interdiction_budget_k']
    vertices = instance['vertices']
    edges = instance['edges']

    edge_dict = {(min(edge[0], edge[1]), max(edge[0], edge[1])): 1 for edge in edges}

    model = gp.Model('clique_interdiction')
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0
    model.Params.LazyConstraints = 1

    # Variables
    x = model.addVars(vertices, vtype=GRB.BINARY, name='x')
    z = model.addVars([(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges], vtype=GRB.BINARY, name='z')
    clique_size = model.addVar(vtype=GRB.INTEGER, name='clique_size')

    # Objective
    model.setObjective(clique_size, GRB.MINIMIZE)

    # Constraints
    model.addConstr(gp.quicksum(z[min(edge[0], edge[1]), max(edge[0], edge[1])] for edge in edges) <= interdiction_budget_k, name='interdiction_budget')

    model.addConstr(clique_size == gp.quicksum(x[v] for v in vertices), name='clique_size_definition')

    for edge in edges:
        v1, v2 = edge
        model.addConstr(x[v1] + x[v2] <= 1 + z[min(v1, v2), max(v1, v2)], name=f'edge_{v1}_{v2}')

    # Lazy constraints for clique definition
    def lazy_clique_constraint(model, where):
        if where == GRB.Callback.MIPSOL:
            solution_x = {v: model.cbGetSolution(x[v]) for v in vertices}
            selected_vertices = [v for v in vertices if solution_x[v] > 0.5]
            for i in range(len(selected_vertices)):
                for j in range(i+1, len(selected_vertices)):
                    v1, v2 = selected_vertices[i], selected_vertices[j]
                    if (min(v1, v2), max(v1, v2)) not in edge_dict or model.cbGetSolution(z[min(v1, v2), max(v1, v2)]) > 0.5:
                        model.cbLazy(x[v1] + x[v2] <= 1)

    model.Params.PreCrush = 1
    model.optimize(lazy_clique_constraint)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.objVal
        interdicted_edges = [[e[0], e[1]] for e in edges if z[min(e[0], e[1]), max(e[0], e[1])].x > 0.5]
        return objective_value, interdicted_edges
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)

    objective_value, interdicted_edges = solve(instance, args.time_limit, logger)

    if objective_value is not None and interdicted_edges is not None:
        solution = {
            'objective_value': int(objective_value),
            'interdicted_edges': interdicted_edges
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
    else:
        print("No feasible solution found.")

if __name__ == '__main__':
    main()