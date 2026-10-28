import json
import argparse
import random
import math
import os
import gurobipy as gp
from gurobipy import GRB

def solve_heuristic(nodes, depot, customer_ids, K, Q, Gamma, dist, nominal_demands, deviations):
    def get_robust_weight(r):
        nom = sum(nominal_demands[c] for c in r)
        devs = sorted([deviations[c] for c in r], reverse=True)
        return nom + sum(devs[:Gamma])

    best_cost = float('inf')
    best_routes = None

    strategies = []
        
    # Strategy 1: Deterministic largest worst-case node first
    s1 = sorted(customer_ids, key=lambda x: nominal_demands[x] + deviations[x], reverse=True)
    strategies.append(s1)
    
    # Random strategies
    n = len(customer_ids)
    iterations = min(2000, max(50, 100000 // (n + 1)))
    for _ in range(iterations):
        seq = list(customer_ids)
        random.shuffle(seq)
        strategies.append(seq)
        
    for seq in strategies:
        unassigned = seq[:]
        if len(unassigned) < K:
            continue
        
        routes = [[] for _ in range(K)]
        for k in range(K):
            routes[k].append(unassigned.pop(0))
            
        possible = True
        for cust in unassigned:
            best_ins = float('inf')
            best_r = -1
            best_p = -1
            
            for r_idx, r in enumerate(routes):
                if get_robust_weight(r + [cust]) <= Q:
                    for p in range(len(r) + 1):
                        prv = r[p-1] if p > 0 else depot
                        nxt = r[p] if p < len(r) else depot
                        diff = dist[prv][cust] + dist[cust][nxt] - dist[prv][nxt]
                        if diff < best_ins:
                            best_ins = diff
                            best_r = r_idx
                            best_p = p
            if best_r == -1:
                possible = False
                break
            else:
                routes[best_r].insert(best_p, cust)
                
        if possible:
            cost = 0
            for r in routes:
                curr = depot
                for node in r:
                    cost += dist[curr][node]
                    curr = node
                cost += dist[curr][depot]
                
            if cost < best_cost:
                best_cost = cost
                best_routes = [list(r) for r in routes]
                
    return best_routes, best_cost


class RobustVRPCallback:
    def __init__(self, nodes, depot, customer_ids, K, Q, Gamma, nominals, deviations, logger):
        self.nodes = nodes
        self.depot = depot
        self.customer_ids = customer_ids
        self.K = K
        self.Q = Q
        self.Gamma = Gamma
        self.nominals = nominals
        self.deviations = deviations
        self.logger = logger
        self.best_obj = float('inf')

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                x_vals = model.cbGetSolution(model._x)
                edges = []
                for i in self.nodes:
                    for j in self.nodes:
                        if i != j and x_vals[i, j] > 0.5:
                            edges.append((i, j))
                            
                succ = {i: [] for i in self.nodes}
                for i, j in edges:
                    succ[i].append(j)
                    
                visited = set()
                routes = []
                if self.depot in succ:
                    for first_cust in succ[self.depot]:
                        route = []
                        curr = first_cust
                        while curr != self.depot:
                            route.append(curr)
                            visited.add(curr)
                            if len(succ[curr]) > 0:
                                curr = succ[curr][0]
                            else:
                                break
                        routes.append(route)
                        
                viable = True
                
                # Check for detached subtours
                unvisited = [v for v in self.customer_ids if v not in visited and len(succ[v]) > 0]
                while unvisited:
                    viable = False
                    start = unvisited[0]
                    cycle = []
                    curr = start
                    while curr not in cycle:
                        cycle.append(curr)
                        visited.add(curr)
                        if len(succ[curr]) > 0:
                            curr = succ[curr][0]
                        else:
                            break
                    idx = cycle.index(curr) if curr in cycle else 0
                    actual_cycle = cycle[idx:]
                    c_set = set(actual_cycle)
                    expr = gp.quicksum(model._x[i, j] for i in c_set for j in c_set if i != j)
                    model.cbLazy(expr <= len(c_set) - 1)
                    unvisited = [v for v in self.customer_ids if v not in visited and len(succ[v]) > 0]
                        
                # Check route robust capacity
                if viable:
                    for r in routes:
                        nom = sum(self.nominals[c] for c in r)
                        devs = sorted([self.deviations[c] for c in r], reverse=True)
                        w = nom + sum(devs[:self.Gamma])
                        
                        if w > self.Q:
                            viable = False
                            r_set = set(r)
                            expr = gp.quicksum(
                                model._x[i, j] 
                                for i in self.nodes if i not in r_set
                                for j in r_set
                            )
                            model.cbLazy(expr >= math.ceil(w / self.Q))
                            
                if viable:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < self.best_obj - 1e-4:
                        self.best_obj = obj
                        if self.logger:
                            self.logger.log(float(obj))
            except Exception as e:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, default=300)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load problem instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    Gamma = instance['uncertainty_set']['Gamma']
    distance_matrix = instance['distance_matrix']
    
    depot = instance['depot']['id']
    customer_ids = [c['id'] for c in instance['customers']]
    nodes = [depot] + customer_ids

    # Safe mapping of IDs to matrix indices avoiding OutOfBounds on non-continguous mapping arrays
    dim = len(distance_matrix)
    max_id = max(nodes)
    node_to_matrix_idx = {}
    
    if dim > len(instance['customers']) + 1 and dim > max_id:
        for node in nodes:
            node_to_matrix_idx[node] = node
    else:
        node_to_matrix_idx[depot] = 0
        for i, c in enumerate(instance['customers']):
            node_to_matrix_idx[c['id']] = i + 1

    mapped_dist = {i: {} for i in nodes}
    for i in nodes:
        for j in nodes:
            mapped_dist[i][j] = distance_matrix[node_to_matrix_idx[i]][node_to_matrix_idx[j]]

    nominal_demands = {depot: 0}
    deviations = {depot: 0}
    for c in instance['customers']:
        nominal_demands[c['id']] = c['nominal_demand']
        deviations[c['id']] = c['demand_deviation']

    # Initialize Logger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Solve initial heuristic
    best_routes, best_cost = solve_heuristic(
        nodes, depot, customer_ids, num_vehicles, vehicle_capacity, Gamma, 
        mapped_dist, nominal_demands, deviations
    )
    
    if best_routes and logger:
        logger.log(float(best_cost))

    # Gurobi Environment and Model
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    
    model = gp.Model("RobustVRP", env=env)
    model.Params.TimeLimit = args.time_limit
    model.Params.LazyConstraints = 1
    model.Params.Threads = 1  # Strictly restricted to single CPU core

    # Variables
    x = {}
    f = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
                f[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"f_{i}_{j}")
    model._x = x
                
    # Objective
    model.setObjective(
        gp.quicksum(mapped_dist[i][j] * x[i, j] for i in nodes for j in nodes if i != j), 
        GRB.MINIMIZE
    )
    
    # 1. Degree constraints
    for i in customer_ids:
        model.addConstr(gp.quicksum(x[i, j] for j in nodes if i != j) == 1)
        model.addConstr(gp.quicksum(x[j, i] for j in nodes if i != j) == 1)
        
    # 2. Depot degree constraints
    model.addConstr(gp.quicksum(x[depot, j] for j in customer_ids) == num_vehicles)
    model.addConstr(gp.quicksum(x[j, depot] for j in customer_ids) == num_vehicles)
    
    # 3. Flow constraints for nominal demand single commodity flow (prevents capacity and basic subtour issues natively)
    for j in customer_ids:
        model.addConstr(f[depot, j] == 0)
        
    for i in customer_ids:
        model.addConstr(
            gp.quicksum(f[i, j] for j in nodes if i != j) -
            gp.quicksum(f[j, i] for j in nodes if i != j) == nominal_demands[i]
        )
        
    for i in nodes:
        for j in customer_ids:
            if i != j:
                model.addConstr(f[i, j] <= (vehicle_capacity - nominal_demands[j]) * x[i, j])
                
    for i in customer_ids:
        model.addConstr(f[i, depot] <= vehicle_capacity * x[i, depot])
        for j in nodes:
            if i != j:
                model.addConstr(f[i, j] >= nominal_demands[i] * x[i, j])

    # Basic pre-processing bounds
    for i in customer_ids:
        for j in customer_ids:
            if i != j:
                if nominal_demands[i] + nominal_demands[j] > vehicle_capacity:
                    model.addConstr(x[i, j] == 0)

    # Warm-start from heuristic solution
    if best_routes:
        for i in nodes:
            for j in nodes:
                if i != j:
                    x[i, j].Start = 0
                    f[i, j].Start = 0
        for r in best_routes:
            load = 0
            curr = depot
            for node in r:
                x[curr, node].Start = 1
                if curr != depot:
                    f[curr, node].Start = load
                load += nominal_demands[node]
                curr = node
            x[curr, depot].Start = 1
            f[curr, depot].Start = load

    # Optimize with callback
    cb_handler = RobustVRPCallback(nodes, depot, customer_ids, num_vehicles, vehicle_capacity, Gamma, nominal_demands, deviations, logger)
    if best_routes:
        cb_handler.best_obj = best_cost
        
    model.optimize(cb_handler)

    # Post-process and extract best answer
    out_routes = []
    total_cost = 0.0
    
    if model.SolCount > 0:
        edges = []
        for i in nodes:
            for j in nodes:
                if i != j and x[i, j].X > 0.5:
                    edges.append((i, j))
                    
        succ = {i: [] for i in nodes}
        for i, j in edges:
            succ[i].append(j)
            
        if depot in succ:
            for first_cust in succ[depot]:
                route = []
                curr = first_cust
                r_cost = mapped_dist[depot][first_cust]
                while curr != depot:
                    route.append(curr)
                    if len(succ[curr]) > 0:
                        nxt = succ[curr][0]
                    else:
                        break  # Failsafe
                    r_cost += mapped_dist[curr][nxt]
                    curr = nxt
                    
                w = sum(nominal_demands[c] for c in route)
                d_sorted = sorted([deviations[c] for c in route], reverse=True)
                w += sum(d_sorted[:Gamma])
                
                out_routes.append({
                    "customers": route,
                    "cost": int(round(r_cost)),
                    "worst_case_demand": w
                })
                total_cost += r_cost
    elif best_routes is not None:
        for r in best_routes:
            r_cost = 0
            curr = depot
            for node in r:
                r_cost += mapped_dist[curr][node]
                curr = node
            r_cost += mapped_dist[curr][depot]
            
            w = sum(nominal_demands[c] for c in r)
            d_sorted = sorted([deviations[c] for c in r], reverse=True)
            w += sum(d_sorted[:Gamma])
            
            out_routes.append({
                "customers": r,
                "cost": int(round(r_cost)),
                "worst_case_demand": w
            })
            total_cost += r_cost
    else:
        total_cost = -1.0
        
    output = {
        "objective_value": float(total_cost),
        "routes": out_routes
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)