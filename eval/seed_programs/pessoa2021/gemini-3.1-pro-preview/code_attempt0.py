import json
import argparse
import random
import math
import os
import gurobipy as gp
from gurobipy import GRB

def solve_heuristic(n, K, Q, Gamma, dist, nominal_demands, deviations):
    def get_robust_weight(r):
        nom = sum(nominal_demands[c] for c in r)
        devs = sorted([deviations[c] for c in r], reverse=True)
        return nom + sum(devs[:Gamma])

    best_cost = float('inf')
    best_routes = None

    strategies = []
        
    # Strategy 1: Deterministic largest worst-case node first
    s1 = sorted(list(range(1, n+1)), key=lambda x: nominal_demands[x] + deviations[x], reverse=True)
    strategies.append(s1)
    
    # Random strategies
    iterations = min(2000, max(50, 100000 // (n + 1)))
    for _ in range(iterations):
        seq = list(range(1, n+1))
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
                        prv = r[p-1] if p > 0 else 0
                        nxt = r[p] if p < len(r) else 0
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
                curr = 0
                for node in r:
                    cost += dist[curr][node]
                    curr = node
                cost += dist[curr][0]
                
            if cost < best_cost:
                best_cost = cost
                best_routes = [list(r) for r in routes]
                
    return best_routes, best_cost

class RobustVRPCallback:
    def __init__(self, n, K, Q, Gamma, nominals, deviations, logger):
        self.n = n
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
                for i in range(self.n + 1):
                    for j in range(self.n + 1):
                        if i != j and x_vals[i, j] > 0.5:
                            edges.append((i, j))
                            
                succ = {i: [] for i in range(self.n + 1)}
                for i, j in edges:
                    succ[i].append(j)
                    
                visited = set()
                routes = []
                if 0 in succ:
                    for first_cust in succ[0]:
                        route = []
                        curr = first_cust
                        while curr != 0:
                            route.append(curr)
                            visited.add(curr)
                            if len(succ[curr]) > 0:
                                curr = succ[curr][0]
                            else:
                                break
                        routes.append(route)
                        
                viable = True
                
                # Check for detached subtours
                unvisited = [v for v in range(1, self.n + 1) if v not in visited and len(succ[v]) > 0]
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
                    unvisited = [v for v in range(1, self.n + 1) if v not in visited and len(succ[v]) > 0]
                        
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
                                for i in range(self.n + 1) if i not in r_set
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
        
    num_customers = instance['num_customers']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    Gamma = instance['uncertainty_set']['Gamma']
    distance_matrix = instance['distance_matrix']
    
    nominal_demands = [0] * (num_customers + 1)
    deviations = [0] * (num_customers + 1)
    for c in instance['customers']:
        idx = c['id']
        nominal_demands[idx] = c['nominal_demand']
        deviations[idx] = c['demand_deviation']

    # Initialize Logger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Solve initial heuristic
    best_routes, best_cost = solve_heuristic(
        num_customers, num_vehicles, vehicle_capacity, Gamma, 
        distance_matrix, nominal_demands, deviations
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
    for i in range(num_customers + 1):
        for j in range(num_customers + 1):
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
                f[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"f_{i}_{j}")
    model._x = x
                
    # Objective
    model.setObjective(
        gp.quicksum(distance_matrix[i][j] * x[i, j] for i in range(num_customers + 1) for j in range(num_customers + 1) if i != j), 
        GRB.MINIMIZE
    )
    
    # 1. Degree constraints
    for i in range(1, num_customers + 1):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_customers + 1) if i != j) == 1)
        model.addConstr(gp.quicksum(x[j, i] for j in range(num_customers + 1) if i != j) == 1)
        
    # 2. Depot degree constraints
    model.addConstr(gp.quicksum(x[0, j] for j in range(1, num_customers + 1)) == num_vehicles)
    model.addConstr(gp.quicksum(x[j, 0] for j in range(1, num_customers + 1)) == num_vehicles)
    
    # 3. Flow constraints for nominal demand single commodity flow (prevents capacity and basic subtour issues natively)
    for j in range(1, num_customers + 1):
        model.addConstr(f[0, j] == 0)
        
    for i in range(1, num_customers + 1):
        model.addConstr(
            gp.quicksum(f[i, j] for j in range(num_customers + 1) if i != j) -
            gp.quicksum(f[j, i] for j in range(num_customers + 1) if i != j) == nominal_demands[i]
        )
        
    for i in range(num_customers + 1):
        for j in range(1, num_customers + 1):
            if i != j:
                model.addConstr(f[i, j] <= (vehicle_capacity - nominal_demands[j]) * x[i, j])
                
    for i in range(1, num_customers + 1):
        model.addConstr(f[i, 0] <= vehicle_capacity * x[i, 0])
        for j in range(num_customers + 1):
            if i != j:
                model.addConstr(f[i, j] >= nominal_demands[i] * x[i, j])

    # Basic pre-processing bounds
    for i in range(1, num_customers + 1):
        for j in range(1, num_customers + 1):
            if i != j:
                if nominal_demands[i] + nominal_demands[j] > vehicle_capacity:
                    model.addConstr(x[i, j] == 0)

    # Warm-start from heuristic solution
    if best_routes:
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j:
                    x[i, j].Start = 0
                    f[i, j].Start = 0
        for r in best_routes:
            load = 0
            curr = 0
            for node in r:
                x[curr, node].Start = 1
                if curr != 0:
                    f[curr, node].Start = load
                load += nominal_demands[node]
                curr = node
            x[curr, 0].Start = 1
            f[curr, 0].Start = load

    # Optimize with callback
    cb_handler = RobustVRPCallback(num_customers, num_vehicles, vehicle_capacity, Gamma, nominal_demands, deviations, logger)
    if best_routes:
        cb_handler.best_obj = best_cost
        
    model.optimize(cb_handler)

    # Post-process and extract best answer
    out_routes = []
    total_cost = 0.0
    
    if model.SolCount > 0:
        edges = []
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                if i != j and x[i, j].X > 0.5:
                    edges.append((i, j))
                    
        succ = {i: [] for i in range(num_customers + 1)}
        for i, j in edges:
            succ[i].append(j)
            
        for first_cust in succ[0]:
            route = []
            curr = first_cust
            r_cost = distance_matrix[0][first_cust]
            while curr != 0:
                route.append(curr)
                nxt = succ[curr][0]
                r_cost += distance_matrix[curr][nxt]
                curr = nxt
                
            w = sum(nominal_demands[c] for c in route)
            d_sorted = sorted([deviations[c] for c in route], reverse=True)
            w += sum(d_sorted[:Gamma])
            
            out_routes.append({
                "customers": route,
                "cost": r_cost,
                "worst_case_demand": w
            })
            total_cost += r_cost
    elif best_routes is not None:
        for r in best_routes:
            r_cost = 0
            curr = 0
            for node in r:
                r_cost += distance_matrix[curr][node]
                curr = node
            r_cost += distance_matrix[curr][0]
            
            w = sum(nominal_demands[c] for c in r)
            d_sorted = sorted([deviations[c] for c in r], reverse=True)
            w += sum(d_sorted[:Gamma])
            
            out_routes.append({
                "customers": r,
                "cost": r_cost,
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