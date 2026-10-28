import argparse
import json
import time
import math
from gurobipy import Model, GRB, quicksum, Env
from solution_logger import SolutionLogger

class SubTourCallback:
    def __init__(self, model, x, y, n, depot=1):
        self.model = model
        self.x = x
        self.y = y
        self.n = n
        self.depot = depot
        
    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            x_vals = model.cbGetSolution(self.x)
            y_vals = model.cbGetSolution(self.y)
            
            visited_nodes = []
            for i in range(1, self.n+1):
                if y_vals[i] > 0.5:
                    visited_nodes.append(i)
                    
            if not visited_nodes:
                return
                
            graph = {i: [] for i in visited_nodes}
            for (i, j) in self.x:
                if x_vals[(i, j)] > 0.5:
                    if i in visited_nodes and j in visited_nodes:
                        graph[i].append(j)
                        graph[j].append(i)
            
            visited_set = set()
            comps = []
            for node in visited_nodes:
                if node not in visited_set:
                    comp = []
                    stack = [node]
                    visited_set.add(node)
                    while stack:
                        cur = stack.pop()
                        comp.append(cur)
                        for neighbor in graph[cur]:
                            if neighbor not in visited_set:
                                visited_set.add(neighbor)
                                stack.append(neighbor)
                    comps.append(comp)
            
            if len(comps) == 1:
                return
                
            for comp in comps:
                if self.depot in comp:
                    continue
                if len(comp) == 0:
                    continue
                    
                i0 = min(comp)
                comp_set = set(comp)
                expr = 0.0
                for i in comp:
                    for j in range(1, i):
                        if j not in comp_set:
                            if (j, i) in self.x:
                                expr += self.x[(j, i)]
                    for j in range(i+1, self.n+1):
                        if j not in comp_set:
                            if (i, j) in self.x:
                                expr += self.x[(i, j)]
                model.cbLazy(expr >= 2 * self.y[i0])

def build_tour(edges_used, n, depot=1):
    if not edges_used:
        return [depot]
        
    graph = {}
    for edge in edges_used:
        i, j = edge
        if i not in graph:
            graph[i] = []
        graph[i].append(j)
        if j not in graph:
            graph[j] = []
        graph[j].append(i)
        
    for node in graph:
        if len(graph[node]) != 2:
            return None
                
    start = depot
    tour = [start]
    current = start
    prev = None
    while True:
        neighbors = graph[current]
        next_node = None
        for nb in neighbors:
            if nb == prev:
                continue
            next_node = nb
            break
        if next_node is None:
            break
        tour.append(next_node)
        prev = current
        current = next_node
        if current == start and len(tour) > 1:
            break
    return tour

def main(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
        
    n = data['n']
    depot = data['depot']
    t0 = data['t0']
    prizes_dict = data['prizes']
    travel_times_dict = data['travel_times']
    
    prizes = {}
    for city_str, prize_val in prizes_dict.items():
        city = int(city_str)
        prizes[city] = prize_val
        
    tt = {}
    for edge_str, time_val in travel_times_dict.items():
        edge_str_clean = edge_str.strip('()')
        parts = edge_str_clean.split(',')
        i = int(parts[0])
        j = int(parts[1])
        tt[(i, j)] = time_val
        tt[(j, i)] = time_val
        
    initial_solution = None
    best_two_city_prize = -1
    best_two_city_pair = None
    for i in range(2, n+1):
        for j in range(i+1, n+1):
            time_ij = tt.get((depot, i), 0) + tt.get((i, j), 0) + tt.get((j, depot), 0)
            if time_ij > t0:
                continue
            prize_ij = prizes.get(i, 0) + prizes.get(j, 0)
            if prize_ij > best_two_city_prize:
                best_two_city_prize = prize_ij
                best_two_city_pair = (i, j)
                
    if best_two_city_pair is not None:
        a, b = best_two_city_pair
        initial_solution = {
            'y': {1: 1, a: 1, b: 1},
            'x': {(min(1, a), max(1, a)): 1, 
                  (min(1, b), max(1, b)): 1, 
                  (min(a, b), max(a, b)): 1}
        }
    
    model = Model("PC-TSP")
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 0)
    model.setParam('LogToConsole', 0)
    
    y = {}
    for i in range(1, n+1):
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        
    x = {}
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            x[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    model.setObjective(quicksum(prizes[i] * y[i] for i in range(1, n+1)), GRB.MAXIMIZE)
    
    model.addConstr(y[depot] == 1, "depot_visited")
    
    for i in range(1, n+1):
        expr = 0
        for j in range(i+1, n+1):
            expr += x.get((i, j), 0)
        for j in range(1, i):
            if (j, i) in x:
                expr += x[(j, i)]
        model.addConstr(expr == 2 * y[i], f"degree_{i}")
    
    time_expr = 0
    for (i, j), var in x.items():
        time_val = tt.get((i, j), 0)
        time_expr += time_val * var
    model.addConstr(time_expr <= t0, "time_budget")
    
    if initial_solution is not None:
        for i, val in initial_solution['y'].items():
            y[i].Start = val
        for edge, val in initial_solution['x'].items():
            if edge in x:
                x[edge].Start = val
                
    time_elapsed = time.time() - start_time
    time_remaining = max(1, time_limit - time_elapsed)
    model.setParam('TimeLimit', time_remaining)
    
    callback = SubTourCallback(model, x, y, n, depot)
    model.optimize(lambda model, where: callback(model, where))
    
    status = model.Status
    if status == GRB.INFEASIBLE:
        if initial_solution is not None:
            visited_nodes = [1, best_two_city_pair[0], best_two_city_pair[1]]
            edges_used = []
            edges_used.append([min(1, best_two_city_pair[0]), max(1, best_two_city_pair[0])])
            edges_used.append([min(1, best_two_city_pair[1]), max(1, best_two_city_pair[1])])
            edges_used.append([min(best_two_city_pair[0], best_two_city_pair[1]), 
                               max(best_two_city_pair[0], best_two_city_pair[1])])
            tour = [1, best_two_city_pair[0], best_two_city_pair[1], 1]
            objective_value = prizes[best_two_city_pair[0]] + prizes[best_two_city_pair[1]]
        else:
            visited_nodes = [1]
            edges_used = []
            tour = [1]
            objective_value = 0
    else:
        if status == GRB.OPTIMAL or status == GRB.TIME_LIMIT:
            if model.SolCount > 0:
                visited_nodes = []
                for i in range(1, n+1):
                    if y[i].X > 0.5:
                        visited_nodes.append(i)
                edges_used = []
                for edge in x:
                    if x[edge].X > 0.5:
                        edges_used.append(list(edge))
                tour_sequence = build_tour(edges_used, n, depot)
                if tour_sequence is None:
                    tour_sequence = [1]
                objective_value = model.objVal
                if logger:
                    logger.log(objective_value)
            else:
                if initial_solution is not None:
                    visited_nodes = [1, best_two_city_pair[0], best_two_city_pair[1]]
                    edges_used = []
                    edges_used.append([min(1, best_two_city_pair[0]), max(1, best_two_city_pair[0])])
                    edges_used.append([min(1, best_two_city_pair[1]), max(1, best_two_city_pair[1])])
                    edges_used.append([min(best_two_city_pair[0], best_two_city_pair[1]), 
                                       max(best_two_city_pair[0], best_two_city_pair[1])])
                    tour_sequence = [1, best_two_city_pair[0], best_two_city_pair[1], 1]
                    objective_value = prizes[best_two_city_pair[0]] + prizes[best_two_city_pair[1]]
                else:
                    visited_nodes = [1]
                    edges_used = []
                    tour_sequence = [1]
                    objective_value = 0
        else:
            visited_nodes = [1]
            edges_used = []
            tour_sequence = [1]
            objective_value = 0
            
    solution = {
        "objective_value": float(objective_value),
        "visited_nodes": visited_nodes,
        "edges": edges_used,
        "tour": tour_sequence
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