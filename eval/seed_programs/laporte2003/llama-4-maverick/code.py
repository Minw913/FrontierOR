import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import itertools

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
    product_markets = {int(k): v for k, v in instance['product_markets'].items()}
    purchase_costs = {(int(k.split(',')[0]), int(k.split(',')[1])): v for k, v in instance['purchase_costs'].items()}
    supplies = {(int(k.split(',')[0]), int(k.split(',')[1])): v for k, v in instance['supplies'].items()}

    mandatory_markets = set()
    for product in range(num_products):
        total_supply = sum(supplies.get((product, market), 0) for market in range(num_markets))
        for market in range(num_markets):
            supply_without_market = total_supply - supplies.get((product, market), 0)
            if supply_without_market < product_demands[product] and supply_without_market >= 0:
                mandatory_markets.add(market)

    model = gp.Model()
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  

    visit = model.addVars(num_markets, vtype=GRB.BINARY, name='visit')
    for market in mandatory_markets:
        visit[market].LB = 1

    x = model.addVars(num_vertices, num_vertices, vtype=GRB.BINARY, name='x')
    purchase = model.addVars(num_products, num_markets, vtype=GRB.CONTINUOUS, name='purchase', lb=0)

    travel_cost = gp.quicksum(float(travel_costs.get(f'{min(i, j)},{max(i, j)}', float('inf'))) * (x[i, j] + x[j, i]) for i in range(num_vertices) for j in range(i+1, num_vertices))
    purchase_cost = gp.quicksum(purchase_costs.get((product, market), 0) * purchase[product, market] for product in range(num_products) for market in range(num_markets))
    model.setObjective(travel_cost + purchase_cost, GRB.MINIMIZE)

    for product in range(num_products):
        model.addConstr(gp.quicksum(purchase[product, market] for market in range(num_markets)) == product_demands[product])
        for market in range(num_markets):
            purchase[product, market].ub = supplies.get((product, market), 0) * visit[market]
            if market not in product_markets.get(product, []):
                purchase[product, market].ub = 0

    model.addConstr(gp.quicksum(x[depot, j] for j in range(num_vertices) if j != depot) == 1)
    model.addConstr(gp.quicksum(x[i, depot] for i in range(num_vertices) if i != depot) == 1)
    for market in range(num_markets):
        model.addConstr(gp.quicksum(x[i, market] for i in range(num_vertices) if i != market) == visit[market])
        model.addConstr(gp.quicksum(x[market, j] for j in range(num_vertices) if j != market) == visit[market])

    for S in range(1, 2**num_markets):
        subset = [i for i in range(num_markets) if (S & (1 << i))]
        if len(subset) > 0 and len(subset) < num_markets:
            LHS = gp.quicksum(x[i, j] + x[j, i] for i in [v for v in subset] + [depot] for j in range(num_vertices) if j not in subset + [depot])
            model.addConstr(LHS >= 2)

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(callback)

    if model.SolCount > 0:
        visited_markets = [market for market in range(num_markets) if visit[market].X > 0.5]
        tour = [depot]
        current = depot
        x_values = model.getAttr('X', x)
        while True:
            next_vertex = None
            for j in range(num_vertices):
                if x_values[current, j] > 0.5:
                    next_vertex = j
                    break
            if next_vertex is not None:
                tour.append(next_vertex)
                current = next_vertex
            if next_vertex == depot:
                break
        tour_edges = [(min(tour[i], tour[i+1]), max(tour[i], tour[i+1])) for i in range(len(tour)-1)]
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
        'purchases': {int(k): v for k, v in purchases.items()}
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