import argparse
import json
import time
import random
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve_tsp(dist_matrix, depot_index, time_limit):
    n = len(dist_matrix)
    if n == 1:
        return [depot_index]
    nodes = list(range(n))
    
    m = gp.Model("TSP")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    if time_limit:
        m.Params.TimeLimit = time_limit
    
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    for i in nodes:
        x[i, i].ub = 0
        
    m.addConstrs(x.sum(i, '*') == 1 for i in nodes)
    m.addConstrs(x.sum('*', i) == 1 for i in nodes)
    m.setObjective(gp.quicksum(dist_matrix[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j), GRB.MINIMIZE)
    
    def subtour_elimination(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._vars)
            next_node = {}
            for i in nodes:
                for j in nodes:
                    if i != j and vals[i, j] > 0.5:
                        next_node[i] = j
            visited = set()
            subtours = []
            for i in nodes:
                if i not in visited:
                    curr = i
                    subtour = []
                    while curr not in visited:
                        visited.add(curr)
                        subtour.append(curr)
                        curr = next_node[curr]
                    if len(subtour) < n:
                        subtours.append(subtour)
            for tour in subtours:
                if len(tour) >= 2:
                    m.cbLazy(gp.quicksum(x[i, j] for i in tour for j in tour if i != j) <= len(tour) - 1)
                    
    m._vars = x
    m.Params.LazyConstraints = 1
    m.optimize(subtour_elimination)
    
    if m.SolCount == 0:
        return None
        
    vals = m.getAttr('X', x)
    next_node = {}
    for i in nodes:
        for j in nodes:
            if i != j and vals[i, j] > 0.5:
                next_node[i] = j
                
    curr = depot_index
    tour = [curr]
    while True:
        curr = next_node[curr]
        if curr == depot_index:
            break
        tour.append(curr)
        
    return tour

def calc_cost(tour, dist_matrix):
    cost = 0
    for i in range(len(tour) - 1):
        cost += dist_matrix[tour[i]][tour[i+1]]
    return float(cost)

def is_compatible(pickup_order, delivery_order, s, l):
    n = len(pickup_order)
    if n == 0:
        return True
    pos_P = {req: i for i, req in enumerate(pickup_order)}
    pos_D = {req: i for i, req in enumerate(delivery_order)}
    
    adj = {u: set() for u in pickup_order}
    for i in range(n):
        for j in range(i+1, n):
            u = pickup_order[i]
            v = pickup_order[j]
            if pos_D[u] > pos_D[v]:
                adj[u].add(v)
                adj[v].add(u)
                
    nodes_list = list(pickup_order)
    for _ in range(100):
        random.shuffle(nodes_list)
        nodes = sorted(nodes_list, key=lambda x: len(adj[x]), reverse=True)
        stacks = [[] for _ in range(s)]
        possible = True
        for u in nodes:
            assigned = False
            for stack in stacks:
                if len(stack) < l:
                    if all(v not in adj[u] for v in stack):
                        stack.append(u)
                        assigned = True
                        break
            if not assigned:
                possible = False
                break
        if possible:
            return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    s = instance['parameters']['num_stacks_s']
    l = instance['parameters']['stack_height_l']
    n = instance['parameters']['num_requests_n']
    depot_p = instance['pickup_region']['depot_index']
    depot_d = instance['delivery_region']['depot_index']
    pickup_dist = instance['pickup_region']['distance_matrix']
    delivery_dist = instance['delivery_region']['distance_matrix']
    requests = instance['requests']
    
    start_time = time.time()
    
    p_to_req = {}
    d_to_req = {}
    for i, req in enumerate(requests):
        p_to_req[req['pickup_vertex']] = i
        d_to_req[req['delivery_vertex']] = i
    
    best_pickup_tour = None
    best_delivery_tour = None
    best_cost = float('inf')
    
    # 1. Solve independent TSPs
    t1 = max(1, args.time_limit / 3)
    pickup_tour = solve_tsp(pickup_dist, depot_p, t1)
    
    elapsed = time.time() - start_time
    t2 = max(1, (args.time_limit - elapsed) / 2)
    delivery_tour = solve_tsp(delivery_dist, depot_d, t2)
    
    if pickup_tour and delivery_tour:
        pickup_reqs = [p_to_req[v] for v in pickup_tour if v != depot_p]
        delivery_reqs = [d_to_req[v] for v in delivery_tour if v != depot_d]
                    
        if is_compatible(pickup_reqs, delivery_reqs, s, l):
            best_pickup_tour = pickup_tour + [depot_p]
            best_delivery_tour = delivery_tour + [depot_d]
            best_cost = calc_cost(best_pickup_tour, pickup_dist) + calc_cost(best_delivery_tour, delivery_dist)
            
            if logger:
                sol = {
                    "objective_value": best_cost,
                    "pickup_tour": best_pickup_tour,
                    "delivery_tour": best_delivery_tour
                }
                logger.log_solution(best_cost, sol)
                
    # 2. Solve ATSP if independent TSPs not compatible or failed
    if best_pickup_tour is None:
        N = n + 1
        W = [[0] * N for _ in range(N)]
        p_vert = [depot_p] + [req['pickup_vertex'] for req in requests]
        d_vert = [depot_d] + [req['delivery_vertex'] for req in requests]
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    W[i][j] = 0
                else:
                    W[i][j] = pickup_dist[p_vert[i]][p_vert[j]] + delivery_dist[d_vert[j]][d_vert[i]]
                    
        elapsed = time.time() - start_time
        t3 = max(1, args.time_limit - elapsed)
        atsp_tour = solve_tsp(W, 0, t3)
        
        if atsp_tour:
            req_order = atsp_tour[1:] # remove dummy 0
            best_pickup_tour = [depot_p] + [requests[i-1]['pickup_vertex'] for i in req_order] + [depot_p]
            best_delivery_tour = [depot_d] + [requests[i-1]['delivery_vertex'] for i in reversed(req_order)] + [depot_d]
            best_cost = calc_cost(best_pickup_tour, pickup_dist) + calc_cost(best_delivery_tour, delivery_dist)
            
            if logger:
                sol = {
                    "objective_value": best_cost,
                    "pickup_tour": best_pickup_tour,
                    "delivery_tour": best_delivery_tour
                }
                logger.log_solution(best_cost, sol)
                
    # 3. Fallback to trivial reverse order if no solution found
    if best_pickup_tour is None:
        best_pickup_tour = [depot_p] + [req['pickup_vertex'] for req in requests] + [depot_p]
        best_delivery_tour = [depot_d] + [req['delivery_vertex'] for req in reversed(requests)] + [depot_d]
        best_cost = calc_cost(best_pickup_tour, pickup_dist) + calc_cost(best_delivery_tour, delivery_dist)
        if logger:
            sol = {
                "objective_value": best_cost,
                "pickup_tour": best_pickup_tour,
                "delivery_tour": best_delivery_tour
            }
            logger.log_solution(best_cost, sol)

    solution = {
        "objective_value": best_cost,
        "pickup_tour": best_pickup_tour,
        "delivery_tour": best_delivery_tour
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()