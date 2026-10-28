import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def write_solution(solution_path, objective_value, selected_edges, vertex_costs):
    solution = {
        "objective_value": objective_value,
        "selected_edges": selected_edges,
        "vertex_costs": vertex_costs
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n = instance['n']
    vertices = instance['vertices']
    edges = []
    weights = {}
    for i in range(n):
        for j in range(i+1, n):
            weight = (vertices[i]['x'] - vertices[j]['x'])**2 + (vertices[i]['y'] - vertices[j]['y'])**2
            edges.append((i, j))
            weights[i, j] = weight

    model = gp.Model()
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = False

    x = model.addVars(edges, vtype=GRB.BINARY, name='x')
    z = model.addVars(range(n), vtype=GRB.CONTINUOUS, name='z')

    model.setObjective(gp.quicksum(z[i] for i in range(n)), GRB.MINIMIZE)

    for i, j in edges:
        model.addConstr(x[i, j] <= 1, name=f'x_{i}_{j}_upper_bound')

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if (i, j) in edges or (j, i) in edges) >= 1, name=f'degree_{i}')

    for i in range(n):
        for j in range(n):
            if (i, j) in edges:
                model.addConstr(z[i] >= weights[i, j] * x[i, j], name=f'z_{i}_{j}')

    model.addConstr(gp.quicksum(x[i, j] for i, j in edges) >= n - 1, name='spanning_tree_size')

    # Callback to find and log feasible solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            solution = model.cbGetSolution(model._vars)
            x_sol = {edge: solution[x[edge]] for edge in edges}
            selected_edges = [(i, j) for i, j in edges if x_sol[i, j] > 0.5]
            vertex_costs = {}
            for i in range(n):
                max_weight = max([weights[i, j] * x_sol.get((i, j), 0) for j in range(n) if (i, j) in edges] + 
                                 [weights[j, i] * x_sol.get((j, i), 0) for j in range(n) if (j, i) in edges])
                vertex_costs[i] = max_weight
            objective_value = sum(vertex_costs.values())
            if logger:
                logger.log(objective_value)

    model._vars = model.getVars()
    model.optimize(callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        x_sol = {edge: x[edge].x for edge in edges}
        selected_edges = [[int(vertices[i]['id']), int(vertices[j]['id'])] for i, j in edges if x_sol[i, j] > 0.5]
        vertex_costs = {str(int(vertices[i]['id'])): z[i].x for i in range(n)}
        objective_value = sum(z[i].x for i in range(n))
        write_solution(args.solution_path, objective_value, selected_edges, vertex_costs)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()