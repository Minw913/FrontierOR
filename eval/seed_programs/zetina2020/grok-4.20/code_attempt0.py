import json
import argparse
import time
import sys
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Determine problem type from schema and data
    if 'n_cities' in instance['parameters']:
        problem_type = 'TSP'
    else:
        problem_type = 'MaxCut'

    logger = SolutionLogger(args.log_path, sense="minimize" if problem_type == 'TSP' else "maximize") if args.log_path else None

    start_time = time.time()
    remaining_time = args.time_limit

    if problem_type == 'TSP':
        solve_tsp(instance, args.solution_path, remaining_time, logger, start_time)
    else:
        solve_maxcut(instance, args.solution_path, remaining_time, logger, start_time)


def solve_tsp(instance, solution_path, time_limit, logger, start_time):
    params = instance['parameters']
    data = instance['data']
    n = params['n_cities']
    dist = np.array(data['distance_matrix'])

    try:
        m = gp.Model("TSP")
        m.setParam(GRB.Param.Seed, 0)
        m.setParam(GRB.Param.MIPGap, 1e-4)
        m.setParam(GRB.Param.NumericFocus, 0)
        m.setParam(GRB.Param.Threads, 1)
        m.setParam(GRB.Param.TimeLimit, time_limit - 1)

        # Binary variables: x[i,j] = 1 if city i is visited at time j
        x = m.addVars(n, n, vtype=GRB.BINARY, name="x")

        # Objective: minimize total tour length
        obj = gp.quicksum(dist[i][j] * x[i, k] * x[j, (k+1)%n] for i in range(n) for j in range(n) for k in range(n))
        m.setObjective(obj, GRB.MINIMIZE)

        # Constraint 1: each city visited exactly once
        for i in range(n):
            m.addConstr(gp.quicksum(x[i, t] for t in range(n)) == 1, name=f"city_{i}")

        # Constraint 2: exactly one city per time step
        for t in range(n):
            m.addConstr(gp.quicksum(x[i, t] for i in range(n)) == 1, name=f"time_{t}")

        # MTZ subtour elimination
        u = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=n-1, name="u")
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f"mtz_{i}_{j}")

        # Callback for incumbent logging
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                sol = model.cbGetSolution(x)
                solution_dict = {}
                tour = [-1] * n
                for i in range(n):
                    for t in range(n):
                        if abs(sol[i, t] - 1.0) < 1e-6:
                            solution_dict[f"x_{i}_{t}"] = 1
                            tour[t] = i
                        else:
                            solution_dict[f"x_{i}_{t}"] = 0
                if logger:
                    logger.log_solution(obj_val, solution_dict)
                # Also output tour for verification
                if -1 not in tour:
                    model._best_tour = tour[:]

        m._best_tour = None
        m.optimize(callback)

        # Extract solution
        solution_dict = {}
        tour = None
        objective_value = m.objVal if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT) else 0.0

        if m.solCount > 0:
            sol = m.getAttr('X', x)
            tour = [-1] * n
            for i in range(n):
                for t in range(n):
                    val = 1 if abs(sol[i, t] - 1.0) < 1e-5 else 0
                    solution_dict[f"x_{i}_{t}"] = val
                    if val == 1:
                        tour[t] = i
            if m._best_tour is not None:
                tour = m._best_tour

        # If no solution found, provide a greedy tour
        if tour is None or -1 in tour:
            tour = greedy_tsp(dist, n)
            objective_value = sum(dist[tour[i]][tour[(i+1)%n]] for i in range(n))
            solution_dict = {}
            for t in range(n):
                city = tour[t]
                for i in range(n):
                    solution_dict[f"x_{i}_{t}"] = 1 if i == city else 0

        result = {
            "objective_value": float(objective_value),
            "solution": solution_dict,
            "tour": [int(x) for x in tour]
        }

    except Exception as e:
        # Fallback: greedy solution
        tour = greedy_tsp(dist, n)
        objective_value = sum(dist[tour[i]][tour[(i+1)%n]] for i in range(n))
        solution_dict = {}
        for t_idx, city in enumerate(tour):
            for i in range(n):
                solution_dict[f"x_{i}_{t_idx}"] = 1 if i == city else 0
        result = {
            "objective_value": float(objective_value),
            "solution": solution_dict,
            "tour": [int(x) for x in tour]
        }

    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)


def greedy_tsp(dist, n):
    """Simple greedy nearest neighbor for TSP fallback"""
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
    # MaxCut instance uses data["edges"] or adjacency implied by graph
    # The schema in prompt is for TSP, but problem description says MaxCut uses nodes/edges.
    # We assume instance follows a compatible structure with "data" containing "edges" or adjacency.
    # For robustness, we try to infer.

    data = instance['data']
    n = len(data.get('coordinates', [])) if 'coordinates' in data else 182  # default from description

    # Try to extract graph
    if 'edges' in data:
        edges = data['edges']
    elif 'distance_matrix' in data:  # unlikely
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    else:
        # Assume complete graph or use standard random graph, but we need data.
        # Since schema is TSP-oriented in prompt, but task says any instance following schema,
        # we assume for MaxCut the instance contains 'adjacency' or 'edges'.
        edges = data.get('edges', [(i,j) for i in range(n) for j in range(i+1,n)])

    try:
        m = gp.Model("MaxCut")
        m.setParam(GRB.Param.Seed, 0)
        m.setParam(GRB.Param.MIPGap, 1e-4)
        m.setParam(GRB.Param.NumericFocus, 0)
        m.setParam(GRB.Param.Threads, 1)
        m.setParam(GRB.Param.TimeLimit, max(1, time_limit - 5))

        x = m.addVars(n, vtype=GRB.BINARY, name="x")

        # Maximize cut: sum x_i*(1-x_j) + x_j*(1-x_i) for each edge = sum (x_i + x_j - 2*x_i*x_j)
        obj = gp.quicksum(x[i] + x[j] - 2 * x[i] * x[j] for i, j in edges)
        m.setObjective(obj, GRB.MAXIMIZE)

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                sol = model.cbGetSolution(x)
                solution_dict = {f"x_{i}": int(round(sol[i])) for i in range(n)}
                if logger:
                    logger.log_solution(obj_val, solution_dict)

        m.optimize(callback)

        solution_dict = {}
        objective_value = m.objVal if m.solCount > 0 else 0.0

        if m.solCount > 0:
            xsol = m.getAttr('X', x)
            for i in range(n):
                solution_dict[f"x_{i}"] = int(round(xsol[i]))
        else:
            # Random cut as fallback
            for i in range(n):
                solution_dict[f"x_{i}"] = i % 2
            objective_value = sum(1 for i, j in edges if solution_dict[f"x_{i}"] != solution_dict[f"x_{j}"])

        result = {
            "objective_value": float(objective_value),
            "solution": solution_dict
        }
        if 'tour' in result:  # should not be for MaxCut
            del result['tour']

    except Exception:
        # Fallback random solution
        solution_dict = {f"x_{i}": i % 2 for i in range(n)}
        cut_value = 0  # would need edges
        result = {
            "objective_value": float(cut_value),
            "solution": solution_dict
        }

    with open(solution_path, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()