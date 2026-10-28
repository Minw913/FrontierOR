import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_tsp(instance, time_limit, logger):
    n = instance['n']
    depot = instance['depot'] - 1  # Adjust depot index
    t0 = instance['t0']
    prizes = {i: int(instance['prizes'].get(str(i+1), 0)) for i in range(n)}  # Adjust city index
    travel_times = {(i, j): int(instance['travel_times'].get(f'({min(i+1, j+1)},{max(i+1, j+1)})', 0)) for i in range(n) for j in range(n) if i != j}  # Adjust city index

    model = gp.Model()
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = time_limit

    x = model.addVars([(i, j) for i in range(n) for j in range(n) if i != j], vtype=GRB.BINARY, name='x')
    y = model.addVars(range(n), vtype=GRB.BINARY, name='y')
    u = model.addVars(range(n), vtype=GRB.INTEGER, name='u')

    model.setObjective(gp.quicksum(prizes[i] * y[i] for i in range(n)), GRB.MAXIMIZE)

    model.addConstr(y[depot] == 1)
    model.addConstr(gp.quicksum(y[i] for i in range(n)) >= 3)
    model.addConstr(gp.quicksum(travel_times[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j) <= t0)

    for i in range(n):
        model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == y[i])
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == y[i])

    for i in range(n):
        for j in range(n):
            if i != j and (i, j) in x:
                model.addConstr(u[i] - u[j] + n*x[i, j] <= n - 1)
    model.addConstr(u[depot] == 0)  # Fix u[depot] to avoid symmetry
    for i in range(n):
        if i != depot:
            model.addConstr(u[i] >= 1)
            model.addConstr(u[i] <= n-1)

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objective_value)

    model.optimize(mycallback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        edges = [(i+1, j+1) for (i, j) in x if x[i, j].X > 0.5]  # Adjust city index
        visited_nodes = [i+1 for i in range(n) if y[i].X > 0.5]  # Adjust city index
        tour = construct_tour(edges, depot+1)  # Adjust depot index
        objective_value = model.getObjective().getValue()
        return objective_value, visited_nodes, edges, tour
    else:
        return None, None, None, None

def construct_tour(edges, depot):
    edge_dict = {i: j for (i, j) in edges}
    tour = [depot]
    current = depot
    while True:
        next_node = edge_dict.get(current)
        if next_node is None:
            break
        tour.append(next_node)
        current = next_node
        if current == depot:
            break
    return tour

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    objective_value, visited_nodes, edges, tour = solve_tsp(instance, args.time_limit, logger)

    if objective_value is not None:
        solution = {
            'objective_value': objective_value,
            'visited_nodes': visited_nodes,
            'edges': edges,
            'tour': tour
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()