import json
import argparse
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, log_path=None):
    num_vertices = instance['num_vertices']
    num_markets = instance['num_markets']
    num_products = instance['num_products']
    depot = instance['depot']
    travel_costs = instance['travel_costs']
    product_demands = instance['product_demands']
    product_markets = instance['product_markets']
    purchase_costs = instance['purchase_costs']
    supplies = instance['supplies']

    # Determine mandatory markets
    mandatory_markets = set()
    for product in range(num_products):
        total_supply = sum(supplies.get(f'{product},{market}', 0) for market in product_markets.get(str(product), []))
        for market in product_markets.get(str(product), []):
            supply_without_market = total_supply - supplies.get(f'{product},{market}', 0)
            if supply_without_market < product_demands[product]:
                mandatory_markets.add(market)

    # Initialize model
    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi logging to console

    # Decision variables
    visit = model.addVars(num_markets, vtype=GRB.BINARY, name='visit')
    x = model.addVars(num_vertices, num_vertices, vtype=GRB.BINARY, name='x')
    u = model.addVars(num_markets, vtype=GRB.CONTINUOUS, name='u')
    purchase = model.addVars(num_products, num_markets, vtype=GRB.CONTINUOUS, name='purchase', lb=0)

    # Objective function
    travel_cost = gp.quicksum(travel_costs.get(f'{i},{j}', 0) * x[i, j] for i in range(num_vertices) for j in range(num_vertices) if i != j)
    purchase_cost = gp.quicksum(purchase_costs.get(f'{product},{market}', 0) * purchase[product, market] for product in range(num_products) for market in range(num_markets))
    model.setObjective(travel_cost + purchase_cost, GRB.MINIMIZE)

    # Constraints
    # 1. Visit mandatory markets
    for market in mandatory_markets:
        visit[market] = 1

    # 2. Purchase constraints
    for product in range(num_products):
        model.addConstr(gp.quicksum(purchase[product, market] for market in product_markets.get(str(product), [])) == product_demands[product])
        for market in range(num_markets):
            purchase[product, market].ub = supplies.get(f'{product},{market}', 0) * visit[market]
            if market not in product_markets.get(str(product), []):
                purchase[product, market].ub = 0

    # 3. Routing constraints
    model.addConstr(gp.quicksum(x[depot, j] for j in range(num_vertices) if j != depot) == 1)
    model.addConstr(gp.quicksum(x[i, depot] for i in range(num_vertices) if i != depot) == 1)
    for market in range(num_markets):
        model.addConstr(gp.quicksum(x[i, market] for i in range(num_vertices) if i != market) == visit[market])
        model.addConstr(gp.quicksum(x[market, j] for j in range(num_vertices) if j != market) == visit[market])

    # MTZ subtour elimination constraints
    for i in range(num_markets):
        for j in range(num_markets):
            if i != j:
                model.addConstr(u[i] - u[j] + num_markets * x[i + (i >= depot), j + (j >= depot)] <= num_markets - 1)
    for i in range(num_markets):
        model.addConstr(u[i] >= 0)
        model.addConstr(u[i] <= num_markets - 1)

    # Symmetry breaking: depot is always the first vertex visited
    for i in range(num_vertices):
        if i != depot:
            model.addConstr(x[depot, i] + x[i, depot] <= 1)

    # Callback for logging incumbent solutions
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        visited_markets = [market for market in range(num_markets) if visit[market].X > 0.5]
        tour = []
        current = depot
        tour.append(current)
        while True:
            next_vertex = None
            for j in range(num_vertices):
                if x[current, j].X > 0.5:
                    next_vertex = j
                    break
            if next_vertex is None or next_vertex == depot:
                break
            tour.append(next_vertex)
            current = next_vertex
        tour.append(depot)
        tour_edges = [(tour[i], tour[i+1]) for i in range(len(tour)-1)]
        purchases = {}
        for product in range(num_products):
            purchases[product] = {}
            for market in range(num_markets):
                quantity = purchase[product, market].X
                if quantity > 0:
                    purchases[product][market] = quantity
        objective_value = model.ObjVal
    else:
        objective_value = float('inf')
        visited_markets = []
        tour = []
        tour_edges = []
        purchases = {}

    return {
        'objective_value': objective_value,
        'visited_markets': visited_markets,
        'tour': tour,
        'tour_edges': tour_edges,
        'purchases': purchases
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = solve_instance(instance, args.time_limit, args.log_path)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()