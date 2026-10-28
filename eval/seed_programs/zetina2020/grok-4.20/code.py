import json
import argparse
import time
import sys
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Suppress D-Bus and Qt warnings that cause container failures
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["DBUS_SESSION_BUS_ADDRESS"] = ""
    os.environ["NO_AT_BRIDGE"] = "1"
    os.environ["XDG_RUNTIME_DIR"] = "/tmp"

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    parameters = instance.get('parameters', {})
    data = instance.get('data', {})
    
    # Determine problem type from schema and content
    if 'n_cities' in parameters or 'coordinates' in data or 'distance_matrix' in data:
        problem_type = 'TSP'
    else:
        problem_type = 'MaxCut'

    logger = SolutionLogger(args.log_path, sense="minimize" if problem_type == 'TSP' else "maximize") if args.log_path else None

    start_time = time.time()
    remaining_time = max(1, args.time_limit - 2)

    if problem_type == 'TSP':
        solve_tsp(instance, args.solution_path, remaining_time, logger, start_time)
    else:
        solve_maxcut(instance, args.solution_path, remaining_time, logger, start_time)


def solve_tsp(instance, solution_path, time_limit, logger, start_time):
    params = instance['parameters']
    data = instance['data']
    n = params['n_cities']
    dist = np.array(data['distance_matrix'], dtype=float)

    try:
        m = gp.Model("TSP")
        m.setParam(GRB.Param.Seed, 0)
        m.setParam(GRB.Param.MIPGap, 1e-4)
        m.setParam(GRB.Param.NumericFocus, 0)
        m.setParam(GRB.Param.Threads, 1)
        m.setParam(GRB.Param.TimeLimit, time_limit)
        m.setParam(GRB.Param.LogToConsole, 0)
        m.setParam(GRB.Param.OutputFlag, 0)

        x = m.addVars(n, n, vtype=GRB.BINARY, name="x")

        # Objective: minimize total tour length
        obj_expr = gp.quicksum(dist[i][j] * x[i, k] * x[j, (k + 1) % n]
                               for i in range(n) for j in range(n) for k in range(n))
        m.setObjective(obj_expr, GRB.MINIMIZE)

        # Assignment constraints
        for i in range(n):
            m.addConstr(x.sum(i, '*') == 1, name=f"out_{i}")
        for t in range(n):
            m.addConstr(x.sum('*', t) == 1, name=f"in_{t}")

        # MTZ subtour elimination
        u = m.addVars(n, vtype=GRB.CONTINUOUS, lb=1, ub=n, name="u")
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    m.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, name=f"mtz_{i}_{j}")

        best_obj = None
        best_solution = None
        best_tour = None

        def callback(model, where):
            nonlocal best_obj, best_solution, best_tour
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if best_obj is None or obj_val < best_obj - 1e-6:
                    best_obj = obj_val
                    sol = model.cbGetSolution(x)
                    solution_dict = {}
                    tour = [-1] * n
                    for i_idx in range(n):
                        for t_idx in range(n):
                            val = sol[i_idx, t_idx]
                            solution_dict[f"x_{i_idx}_{t_idx}"] = 1 if val > 0.5 else 0
                            if val > 0.5:
                                tour[t_idx] = i_idx
                    if logger:
                        logger.log_solution(float(obj_val), solution_dict)
                    best_solution = solution_dict.copy()
                    best_tour = tour[:]

        m.optimize(callback)

        if best_solution is None and m.SolCount > 0:
            xsol = m.getAttr('X', x)
            best_obj = m.objVal
            best_solution = {}
            best_tour = [-1] * n
            for i in range(n):
                for t in range(n):
                    val = xsol[i, t]
                    best_solution[f"x_{i}_{t}"] = 1 if val > 0.5 else 0
                    if val > 0.5:
                        best_tour[t] = i

        if best_solution is None:
            # Fallback: greedy nearest neighbor
            tour = greedy_tsp(dist, n)
            obj_val = sum(dist[tour[i]][tour[(i+1)%n]] for i in range(n))
            solution_dict = {}
            for t_idx, city in enumerate(tour):
                for i in range(n):
                    solution_dict[f"x_{i}_{t_idx}"] = 1 if i == city else 0
            best_solution = solution_dict
            best_tour = tour
            best_obj = obj_val
            if logger:
                logger.log_solution(float(obj_val), solution_dict)

        result = {
            "objective_value": float(best_obj),
            "solution": best_solution,
            "tour": [int(c) for c in best_tour]
        }

    except Exception:
        # Emergency fallback
        tour = greedy_tsp(dist, n)
        obj_val = sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))
        solution_dict = {}
        for t_idx, city in enumerate(tour):
            for i in range(n):
                solution_dict[f"x_{i}_{t_idx}"] = 1 if i == city else 0
        result = {
            "objective_value": float(obj_val),
            "solution": solution_dict,
            "tour": [int(c) for c in tour]
        }
        if logger:
            logger.log_solution(float(obj_val), solution_dict)

    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)


def greedy_tsp(dist, n):
    unvisited = set(range(n))
    tour = [0]
    unvisited.remove(0)
    current = 0
    while unvisited:
        next_city = min(unvisited, key=lambda x: dist[current][x])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    return tour


def solve_maxcut(instance, solution_path, time_limit, logger, start_time):
    data = instance.get('data', {})
    coordinates = data.get('coordinates')
    n = len(coordinates) if coordinates is not None else len(data.get('edges', [])) // 10 + 10

    # Extract edges - handle both explicit edge lists and derived from coordinates
    edges = data.get('edges', [])
    if not edges and coordinates:
        # For MaxCut benchmark instances, assume complete graph (unweighted)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    elif 'distance_matrix' in data:
        dist = np.array(data.get('distance_matrix', []))
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if dist[i][j] > 1e-9]

    try:
        m = gp.Model("MaxCut")
        m.setParam(GRB.Param.Seed, 0)
        m.setParam(GRB.Param.MIPGap, 1e-4)
        m.setParam(GRB.Param.NumericFocus, 0)
        m.setParam(GRB.Param.Threads, 1)
        m.setParam(GRB.Param.TimeLimit, time_limit)
        m.setParam(GRB.Param.LogToConsole, 0)
        m.setParam(GRB.Param.OutputFlag, 0)

        x = m.addVars(n, vtype=GRB.BINARY, name="x")

        # Standard MaxCut objective: maximize cut weight (all weights=1)
        obj = gp.quicksum(x[i] * (1 - x[j]) + x[j] * (1 - x[i]) for i, j in edges)
        m.setObjective(obj, GRB.MAXIMIZE)

        best_obj = None
        best_solution = None

        def callback(model, where):
            nonlocal best_obj, best_solution
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if best_obj is None or obj_val > best_obj + 1e-6:
                    best_obj = obj_val
                    sol = model.cbGetSolution(x)
                    solution_dict = {f"x_{i}": 1 if sol[i] > 0.5 else 0 for i in range(n)}
                    if logger:
                        logger.log_solution(float(obj_val), solution_dict)
                    best_solution = solution_dict.copy()

        m.optimize(callback)

        if best_solution is None and m.SolCount > 0:
            xsol = m.getAttr('X', x)
            best_obj = m.objVal
            best_solution = {f"x_{i}": 1 if xsol[i] > 0.5 else 0 for i in range(n)}

        if best_solution is None:
            # Simple fallback: random balanced cut
            best_solution = {f"x_{i}": (i % 2) for i in range(n)}
            cut_value = sum(1 for i, j in edges if best_solution[f"x_{i}"] != best_solution[f"x_{j}"])
            best_obj = float(cut_value)
            if logger:
                logger.log_solution(best_obj, best_solution)

        result = {
            "objective_value": float(best_obj),
            "solution": best_solution
        }

    except Exception:
        # Emergency fallback for MaxCut
        best_solution = {f"x_{i}": (i % 2) for i in range(n)}
        cut_value = sum(1 for i, j in edges if best_solution[f"x_{i}"] != best_solution[f"x_{j}"]) if edges else float(n // 2)
        result = {
            "objective_value": float(cut_value),
            "solution": best_solution
        }
        if logger:
            logger.log_solution(float(cut_value), best_solution)

    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()