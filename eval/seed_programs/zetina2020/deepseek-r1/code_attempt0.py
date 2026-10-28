import argparse
import json
import itertools
import math
import random
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve_tsp(n, distance_matrix, time_limit, logger):
    if n <= 10:
        best_tour = None
        best_cost = float('inf')
        cities = list(range(n))
        for perm in itertools.permutations(cities[1:]):
            tour = [cities[0]] + list(perm)
            cost = 0
            for i in range(n):
                j = (i + 1) % n
                cost += distance_matrix[tour[i]][tour[j]]
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
                if logger:
                    sol_dict = {}
                    for t in range(n):
                        city_at_t = best_tour[t]
                        for city in range(n):
                            sol_dict[f"x_{city}_{t}"] = 1 if city == city_at_t else 0
                    logger.log_solution(best_cost, sol_dict)
        sol_dict = {}
        for t in range(n):
            city_at_t = best_tour[t]
            for city in range(n):
                sol_dict[f"x_{city}_{t}"] = 1 if city == city_at_t else 0
        return best_cost, sol_dict, best_tour
    else:
        model = gp.Model("TSP")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', time_limit)
        
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        
        u = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=n-1, name="u")
        model.addConstr(u[0] == 0)
        
        obj = gp.quicksum(distance_matrix[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)
        model.setObjective(obj, GRB.MINIMIZE)
        
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1)
            model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1)
        
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)
        
        model._x = x
        model._n = n
        model._distance_matrix = distance_matrix
        model._logger = logger
        
        def tsp_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                x_vals = {}
                for (i, j), var in model._x.items():
                    x_vals[i, j] = model.cbGetSolution(var)
                
                next_city = {}
                for i in range(n):
                    for j in range(n):
                        if i != j and (i, j) in x_vals and x_vals[i, j] > 0.5:
                            next_city[i] = j
                
                tour = [0]
                current = 0
                visited = {0}
                for _ in range(n-1):
                    if current not in next_city:
                        break
                    j = next_city[current]
                    if j in visited:
                        break
                    tour.append(j)
                    visited.add(j)
                    current = j
                
                if len(visited) == n:
                    cost = 0
                    for i in range(n):
                        j = (i + 1) % n
                        cost += model._distance_matrix[tour[i]][tour[j]]
                    sol_dict = {}
                    for t, city in enumerate(tour):
                        for c in range(n):
                            sol_dict[f"x_{c}_{t}"] = 1 if c == city else 0
                    if model._logger:
                        model._logger.log_solution(cost, sol_dict)
        
        model.optimize(tsp_callback if logger else None)
        
        if model.SolCount > 0:
            x_vals = model.getAttr('X', x)
            next_city = {}
            for (i, j), var in x.items():
                if x_vals[var] > 0.5:
                    next_city[i] = j
            
            tour = [0]
            current = 0
            visited = {0}
            for _ in range(n-1):
                j = next_city.get(current)
                if j is None or j in visited:
                    break
                tour.append(j)
                visited.add(j)
                current = j
            
            if len(visited) < n:
                tour = list(range(n))
                random.shuffle(tour)
            
            cost = 0
            for i in range(n):
                j = (i + 1) % n
                cost += distance_matrix[tour[i]][tour[j]]
            
            sol_dict = {}
            for t, city in enumerate(tour):
                for c in range(n):
                    sol_dict[f"x_{c}_{t}"] = 1 if c == city else 0
            return cost, sol_dict, tour
        else:
            tour = list(range(n))
            random.shuffle(tour)
            cost = 0
            for i in range(n):
                j = (i + 1) % n
                cost += distance_matrix[tour[i]][tour[j]]
            sol_dict = {}
            for t, city in enumerate(tour):
                for c in range(n):
                    sol_dict[f"x_{c}_{t}"] = 1 if c == city else 0
            return cost, sol_dict, tour

def solve_maxcut(n_nodes, edges, time_limit, logger):
    model = gp.Model("MaxCut")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('NonConvex', 2)
    
    x = model.addVars(n_nodes, vtype=GRB.BINARY, name="x")
    
    obj = 0
    for edge in edges:
        if len(edge) == 2:
            i, j = edge
            w = 1
        else:
            i, j, w = edge
        obj += w * (x[i] + x[j] - 2 * x[i] * x[j])
    model.setObjective(obj, GRB.MAXIMIZE)
    
    model._x = x
    model._logger = logger
    
    def maxcut_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol_vals = model.cbGetSolution(model._x)
            sol_dict = {f"x_{i}": int(round(sol_vals[i])) for i in range(n_nodes)}
            if model._logger:
                model._logger.log_solution(obj_val, sol_dict)
    
    model.optimize(maxcut_callback if logger else None)
    
    if model.SolCount > 0:
        obj_val = model.ObjVal
        sol_vals = model.getAttr('X', x)
        sol_dict = {f"x_{i}": int(round(sol_vals[i])) for i in range(n_nodes)}
        return obj_val, sol_dict
    else:
        sol_dict = {f"x_{i}": random.randint(0, 1) for i in range(n_nodes)}
        obj_val = 0
        for edge in edges:
            if len(edge) == 2:
                i, j = edge
                w = 1
            else:
                i, j, w = edge
            if sol_dict[f"x_{i}"] != sol_dict[f"x_{j}"]:
                obj_val += w
        return obj_val, sol_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = None
    if "coordinates" in instance.get("data", {}) and "distance_matrix" in instance.get("data", {}):
        n_cities = instance["parameters"]["n_cities"]
        distance_matrix = instance["data"]["distance_matrix"]
        if args.log_path:
            logger = SolutionLogger(args.log_path, sense="minimize")
        obj_val, sol_dict, tour = solve_tsp(n_cities, distance_matrix, args.time_limit, logger)
        solution = {
            "objective_value": obj_val,
            "solution": sol_dict,
            "tour": tour
        }
    elif "edges" in instance.get("data", {}):
        edges = instance["data"]["edges"]
        if "parameters" in instance and "n_nodes" in instance["parameters"]:
            n_nodes = instance["parameters"]["n_nodes"]
        else:
            nodes = set()
            for edge in edges:
                nodes.add(edge[0])
                nodes.add(edge[1])
            n_nodes = len(nodes)
        if args.log_path:
            logger = SolutionLogger(args.log_path, sense="maximize")
        obj_val, sol_dict = solve_maxcut(n_nodes, edges, args.time_limit, logger)
        solution = {
            "objective_value": obj_val,
            "solution": sol_dict
        }
    else:
        raise ValueError("Unrecognized problem type in instance data")
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()