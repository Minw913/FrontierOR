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
    depot = instance['depot']
    t0 = instance['t0']
    prizes = instance['prizes']
    travel_times = instance['travel_times']

    model = gp.Model()
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = time_limit

    # Decision Variables
    x = model.addVars([(i, j) for i in range(n) for j in range(n) if i != j], vtype=GRB.BINARY, name='x')
    y = model.addVars(range(n), vtype=GRB.BINARY, name='y')

    # Objective Function
    model.setObjective(gp.quicksum(prizes[str(i)] * y[i] for i in range(n)), GRB.MAXIMIZE)

    # Constraints
    model.addConstr(y[depot] == 1)  # Depot is always visited
    model.addConstr(gp.quicksum(x[i, j] for i in range(n) for j in range(n) if i != j) == gp.quicksum(y[i] for i in range(n)))  # Number of edges equals number of visited cities
    model.addConstr(gp.quicksum(y[i] for i in range(n)) >= 3)  # At least three cities are visited
    model.addConstr(gp.quicksum(travel_times.get(f'({min(i, j)},{max(i, j)})', 0) * x[i, j] for i in range(n) for j in range(n) if i != j) <= t0)  # Total travel time does not exceed t0

    for i in range(n):
        model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == y[i])  # In-degree equals y[i]
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == y[i])  # Out-degree equals y[i]

    # Subtour elimination constraints
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n) if j != i) <= gp.quicksum(x[j, i] for j in range(S) if j != i) for i in S for S in [{k for k in range(n) if k not in [depot] and y[k] == 1}]))

    # Correct implementation of subtour elimination
    def subtour(edges, n):
        visited = [False]*n
        cycles = []
        for i in range(n):
            if not visited[i]:
                cycle = []
                current = i
                while not visited[current]:
                    visited[current] = True
                    next_node = None
                    for j in range(n):
                        if edges[current, j] == 1:
                            next_node = j
                            break
                    if next_node is not None:
                        current = next_node
                    else:
                        break
                    cycle.append(current)
                if len(cycle) > 0 and cycle[0] != depot:
                    cycles.append(cycle)
        return cycles

    def add_subtour_elimination(model, x, y, n):
        while True:
            model.optimize()
            if model.status != GRB.OPTIMAL:
                break
            edges = model.getAttr('X', x)
            cycles = subtour(edges, n)
            if len(cycles) == 0:
                break
            for cycle in cycles:
                model.addConstr(gp.quicksum(x[i, j] for i in cycle for j in cycle if i != j) <= len(cycle) - 1)
            model.update()

    # Callback for logging intermediate solutions
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objective_value)

    model.optimize(mycallback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        edges = model.getAttr('X', x)
        visited_nodes = [i for i in range(n) if y[i].X > 0.5]
        tour_edges = [(i, j) for i in range(n) for j in range(n) if i != j and edges[i, j] > 0.5]
        tour = construct_tour(tour_edges, depot)
        objective_value = model.getObjective().getValue()
        return objective_value, visited_nodes, tour_edges, tour
    else:
        return None, None, None, None

def construct_tour(edges, depot):
    tour = [depot]
    current = depot
    edge_dict = {i: j for (i, j) in edges}
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