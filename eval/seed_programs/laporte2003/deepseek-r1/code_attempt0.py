import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

class CallbackWrapper:
    def __init__(self, logger):
        self.logger = logger
    
    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if self.logger:
                self.logger.log(obj)

def main(instance_path, solution_path, time_limit, log_path):
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_vertices = data['num_vertices']
    num_products = data['num_products']
    depot = data['depot']
    travel_costs_dict = data['travel_costs']
    product_demands = data['product_demands']
    product_markets = data['product_markets']
    purchase_costs_dict = data['purchase_costs']
    supplies_dict = data['supplies']
    
    cost = [[0] * num_vertices for _ in range(num_vertices)]
    for key, c_val in travel_costs_dict.items():
        i, j = map(int, key.split(','))
        cost[i][j] = c_val
        cost[j][i] = c_val
    
    mandatory_markets = set()
    total_supply = [0] * num_products
    for p in range(num_products):
        markets_p = product_markets[str(p)]
        total = 0
        for k in markets_p:
            s = supplies_dict[f"{p},{k}"]
            total += s
        total_supply[p] = total
        
        for k in markets_p:
            if total - supplies_dict[f"{p},{k}"] < product_demands[p]:
                mandatory_markets.add(k)
    
    try:
        model = gp.Model("TPP")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)
        
        y = model.addVars(num_vertices, vtype=GRB.BINARY, name="y")
        a = model.addVars(num_vertices, num_vertices, vtype=GRB.BINARY, name="a")
        u = model.addVars(num_vertices, vtype=GRB.CONTINUOUS, name="u")
        
        z = {}
        for p in range(num_products):
            markets_p = product_markets[str(p)]
            for k in markets_p:
                z[p, k] = model.addVar(vtype=GRB.CONTINUOUS, name=f"z_{p}_{k}")
        
        for p in range(num_products):
            markets_p = product_markets[str(p)]
            model.addConstr(gp.quicksum(z[p, k] for k in markets_p) == product_demands[p], name=f"demand_{p}")
        
        for p in range(num_products):
            markets_p = product_markets[str(p)]
            for k in markets_p:
                model.addConstr(z[p, k] <= supplies_dict[f"{p},{k}"] * y[k], name=f"supply_{p}_{k}")
        
        model.addConstr(y[depot] == 1, name="depot_visited")
        for k in mandatory_markets:
            model.addConstr(y[k] == 1, name=f"mandatory_{k}")
        
        for i in range(num_vertices):
            model.addConstr(gp.quicksum(a[i, j] for j in range(num_vertices) if j != i) == y[i], name=f"out_deg_{i}")
            model.addConstr(gp.quicksum(a[j, i] for j in range(num_vertices) if j != i) == y[i], name=f"in_deg_{i}")
        
        model.addConstr(u[depot] == 0, name="u_depot")
        n = num_vertices
        for i in range(num_vertices):
            if i != depot:
                model.addConstr(u[i] >= y[i])
                model.addConstr(u[i] <= (n - 1) * y[i])
        
        for i in range(num_vertices):
            for j in range(num_vertices):
                if i != j and j != depot:
                    model.addConstr(u[i] - u[j] + n * a[i, j] <= n - 1, name=f"mtz_{i}_{j}")
        
        travel_cost = gp.quicksum(cost[i][j] * a[i, j] for i in range(num_vertices) for j in range(num_vertices) if i != j)
        purchase_cost = gp.quicksum(purchase_costs_dict[f"{p},{k}"] * z[p, k] for p in range(num_products) for k in product_markets[str(p)])
        
        total_cost = travel_cost + purchase_cost
        model.setObjective(total_cost, GRB.MINIMIZE)
        
        cb = CallbackWrapper(logger)
        model.optimize(cb)
        
        if model.SolCount > 0:
            visited_vertices = [i for i in range(num_vertices) if y[i].X > 0.5]
            visited_markets = [i for i in visited_vertices if i != depot]
            
            tour = [depot]
            current = depot
            while True:
                next_vertex = None
                for j in range(num_vertices):
                    if j != current and a[current, j].X > 0.5:
                        next_vertex = j
                        break
                if next_vertex is None:
                    break
                if next_vertex == depot:
                    break
                tour.append(next_vertex)
                current = next_vertex
            tour.append(depot)
            
            tour_edges = []
            for i in range(len(tour) - 1):
                u_val = tour[i]
                v_val = tour[i + 1]
                tour_edges.append([min(u_val, v_val), max(u_val, v_val)])
            
            purchases = {}
            for p in range(num_products):
                purchases_p = {}
                markets_p = product_markets[str(p)]
                for k in markets_p:
                    if k in visited_markets and abs(z[p, k].X) > 1e-6:
                        purchases_p[str(k)] = z[p, k].X
                purchases[str(p)] = purchases_p
            
            obj_value = model.ObjVal
        else:
            visited_vertices = [depot] + list(mandatory_markets)
            visited_markets = [v for v in visited_vertices if v != depot]
            
            remaining_demands = product_demands.copy()
            purchases = {}
            for p in range(num_products):
                purchases_p = {}
                markets_p = [k for k in product_markets[str(p)] if k in visited_markets]
                market_list = []
                for k in markets_p:
                    c = purchase_costs_dict[f"{p},{k}"]
                    s = supplies_dict[f"{p},{k}"]
                    market_list.append((c, k, s))
                market_list.sort(key=lambda x: x[0])
                rem = remaining_demands[p]
                for (c, k, s) in market_list:
                    buy = min(rem, s)
                    if buy > 0:
                        purchases_p[str(k)] = buy
                        rem -= buy
                purchases[str(p)] = purchases_p
            
            unvisited = set(visited_vertices)
            unvisited.remove(depot)
            tour = [depot]
            current = depot
            while unvisited:
                next_vertex = None
                min_cost = float('inf')
                for j in unvisited:
                    if cost[current][j] < min_cost:
                        min_cost = cost[current][j]
                        next_vertex = j
                tour.append(next_vertex)
                unvisited.remove(next_vertex)
                current = next_vertex
            tour.append(depot)
            
            tour_edges = []
            for i in range(len(tour) - 1):
                u_val = tour[i]
                v_val = tour[i + 1]
                tour_edges.append([min(u_val, v_val), max(u_val, v_val)])
            
            travel_cost_value = 0
            for i in range(len(tour) - 1):
                travel_cost_value += cost[tour[i]][tour[i + 1]]
            
            purchase_cost_value = 0
            for p in range(num_products):
                for k, q in purchases[str(p)].items():
                    k_int = int(k)
                    c = purchase_costs_dict[f"{p},{k_int}"]
                    purchase_cost_value += c * q
            
            obj_value = travel_cost_value + purchase_cost_value
        
        solution = {
            "objective_value": obj_value,
            "visited_markets": visited_markets,
            "tour": tour,
            "tour_edges": tour_edges,
            "purchases": purchases
        }
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
    
    except Exception as e:
        visited_vertices = [depot] + list(mandatory_markets)
        visited_markets = [v for v in visited_vertices if v != depot]
        
        remaining_demands = product_demands.copy()
        purchases = {}
        for p in range(num_products):
            purchases_p = {}
            markets_p = [k for k in product_markets[str(p)] if k in visited_markets]
            market_list = []
            for k in markets_p:
                c = purchase_costs_dict[f"{p},{k}"]
                s = supplies_dict[f"{p},{k}"]
                market_list.append((c, k, s))
            market_list.sort(key=lambda x: x[0])
            rem = remaining_demands[p]
            for (c, k, s) in market_list:
                buy = min(rem, s)
                if buy > 0:
                    purchases_p[str(k)] = buy
                    rem -= buy
            purchases[str(p)] = purchases_p
        
        unvisited = set(visited_vertices)
        unvisited.remove(depot)
        tour = [depot]
        current = depot
        while unvisited:
            next_vertex = None
            min_cost = float('inf')
            for j in unvisited:
                if cost[current][j] < min_cost:
                    min_cost = cost[current][j]
                    next_vertex = j
            tour.append(next_vertex)
            unvisited.remove(next_vertex)
            current = next_vertex
        tour.append(depot)
        
        tour_edges = []
        for i in range(len(tour) - 1):
            u_val = tour[i]
            v_val = tour[i + 1]
            tour_edges.append([min(u_val, v_val), max(u_val, v_val)])
        
        travel_cost_value = 0
        for i in range(len(tour) - 1):
            travel_cost_value += cost[tour[i]][tour[i + 1]]
        
        purchase_cost_value = 0
        for p in range(num_products):
            for k, q in purchases[str(p)].items():
                k_int = int(k)
                c = purchase_costs_dict[f"{p},{k_int}"]
                purchase_cost_value += c * q
        
        obj_value = travel_cost_value + purchase_cost_value
        
        solution = {
            "objective_value": obj_value,
            "visited_markets": visited_markets,
            "tour": tour,
            "tour_edges": tour_edges,
            "purchases": purchases
        }
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)