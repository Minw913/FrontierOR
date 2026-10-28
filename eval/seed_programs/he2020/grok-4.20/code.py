import json
import argparse
import time
import sys
from typing import List, Dict, Any
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Extract data
    n = data['num_customers']
    K = data['num_vehicles']
    Q = data['vehicle_capacity']
    T = data['planning_horizon']
    service_time_default = data.get('service_time', 0)
    depot_start = 0
    depot_end = n + 1
    N = list(range(1, n + 1))  # customers
    V = [0] + N + [n + 1]      # all nodes: 0=start, 1..n=customers, n+1=end

    # Coordinates and attributes
    x = [0] * (n + 2)
    y = [0] * (n + 2)
    demand = [0] * (n + 2)
    a = [0] * (n + 2)  # earliest time
    b = [0] * (n + 2)  # latest time
    preferred = [0.0] * (n + 2)
    rho = data['inconvenience_cost_function']['rho']

    # Depot start
    d0 = data['depot']
    x[0] = d0['x']
    y[0] = d0['y']
    a[0] = d0['time_window'][0]
    b[0] = d0['time_window'][1]
    demand[0] = 0

    # Depot end
    de = data['depot_end']
    x[n+1] = de['x']
    y[n+1] = de['y']
    a[n+1] = de['time_window'][0]
    b[n+1] = de['time_window'][1]
    demand[n+1] = 0

    # Customers
    cust_map = {}  # id -> index
    for cust in data['customers']:
        idx = cust['id']
        cust_map[idx] = idx
        x[idx] = cust['x']
        y[idx] = cust['y']
        demand[idx] = cust['demand']
        a[idx] = cust['time_window'][0]
        b[idx] = cust['time_window'][1]
        preferred[idx] = cust['preferred_time']

    dist = data['distance_matrix']  # travel time = cost

    # Build model
    model = gp.Model("VRPTW_with_inconvenience")
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, args.time_limit - 5)  # reserve time for output

    # Variables
    xvar = model.addVars(V, V, vtype=GRB.BINARY, name="x")
    tvar = model.addVars(V, vtype=GRB.CONTINUOUS, name="t")  # service start / departure time
    uvar = model.addVars(N, vtype=GRB.CONTINUOUS, name="u")  # load

    # Inconvenience is quadratic: rho * (t_i - preferred_i)^2
    # We introduce auxiliary variables for linearization via piecewise or direct quadratic obj
    # Since Gurobi supports quadratic objectives, we will add quadratic terms directly

    # Objective: routing cost + inconvenience
    routing_cost = gp.quicksum(dist[i][j] * xvar[i, j] for i in V for j in V if i != j)
    
    inconvenience = gp.quicksum(
        rho * (tvar[i] - preferred[i]) * (tvar[i] - preferred[i]) for i in N
    )
    
    model.setObjective(routing_cost + inconvenience, GRB.MINIMIZE)

    # Constraints
    # 1. Each customer visited exactly once
    for i in N:
        model.addConstr(gp.quicksum(xvar[i, j] for j in V if j != i) == 1, f"out_{i}")
        model.addConstr(gp.quicksum(xvar[j, i] for j in V if j != i) == 1, f"in_{i}")

    # 2. Flow conservation at depot
    model.addConstr(gp.quicksum(xvar[0, j] for j in N) == K, "depot_out")
    model.addConstr(gp.quicksum(xvar[i, n+1] for i in N) == K, "depot_in")

    # 3. Capacity
    for i in N:
        model.addConstr(uvar[i] >= demand[i], f"load_lower_{i}")
        model.addConstr(uvar[i] <= Q, f"load_upper_{i}")
    for i in N:
        for j in N:
            if i != j:
                model.addConstr(uvar[j] >= uvar[i] + demand[j] - Q * (1 - xvar[i, j]), f"mtz_load_{i}_{j}")

    # 4. Time windows and sequencing
    M = T + 1
    for i in V:
        model.addConstr(tvar[i] >= a[i], f"tw_early_{i}")
        model.addConstr(tvar[i] <= b[i], f"tw_late_{i}")

    for i in V:
        for j in V:
            if i != j and i != n+1 and j != 0:
                model.addConstr(
                    tvar[j] >= tvar[i] + service_time_default + dist[i][j] - M * (1 - xvar[i, j]),
                    f"time_{i}_{j}"
                )

    # 5. No self loops
    for i in V:
        model.addConstr(xvar[i, i] == 0, f"no_self_{i}")

    # 6. Depot time
    model.addConstr(tvar[0] >= a[0])
    model.addConstr(tvar[n+1] <= b[n+1])

    # Callback for logging incumbents
    incumbent_obj = float('inf')
    best_solution = None

    def callback(model, where):
        nonlocal incumbent_obj, best_solution
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < incumbent_obj - 1e-6:
                incumbent_obj = obj
                # Extract solution
                x_sol = model.cbGetSolution(xvar)
                t_sol = model.cbGetSolution(tvar)
                
                routes = []
                schedule = {}
                
                # Reconstruct routes
                for k in range(K):
                    route = [0]
                    current = 0
                    while True:
                        found = False
                        for j in V:
                            if j != current and x_sol[current, j] > 0.5:
                                if j == n+1:
                                    route.append(n+1)
                                    routes.append(route)
                                    found = True
                                    break
                                else:
                                    route.append(j)
                                    schedule[str(j)] = round(float(t_sol[j]), 6)
                                    current = j
                                    found = True
                                    break
                        if not found or current == n+1:
                            break
                    if len(route) <= 2:
                        break  # no more routes
                
                solution_dict = {
                    "objective_value": float(obj),
                    "routes": routes,
                    "schedule": schedule
                }
                
                if logger:
                    logger.log_solution(float(obj), solution_dict)
                
                best_solution = solution_dict

    # Optimize
    model.optimize(callback)

    # Final solution extraction
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        if best_solution is None:
            # Extract from final solution if callback didn't capture
            try:
                obj = model.objVal
                x_sol = model.getAttr('X', xvar)
                t_sol = model.getAttr('X', tvar)
                
                routes = []
                schedule = {}
                visited = set()
                
                for _ in range(K):
                    route = [0]
                    current = 0
                    while True:
                        found = False
                        for j_idx in V:
                            if j_idx != current and x_sol[current, j_idx] > 0.5:
                                if j_idx == n + 1:
                                    route.append(n + 1)
                                    if len(route) > 2:
                                        routes.append(route)
                                    found = True
                                    break
                                elif j_idx not in visited:
                                    route.append(j_idx)
                                    schedule[str(j_idx)] = round(float(t_sol[j_idx]), 6)
                                    visited.add(j_idx)
                                    current = j_idx
                                    found = True
                                    break
                        if not found:
                            break
                    if len(route) <= 2:
                        break
                
                best_solution = {
                    "objective_value": float(obj),
                    "routes": routes,
                    "schedule": schedule
                }
                if logger:
                    logger.log_solution(float(obj), best_solution)
            except:
                # Minimal feasible fallback
                best_solution = {
                    "objective_value": float('inf'),
                    "routes": [],
                    "schedule": {}
                }
    else:
        best_solution = {
            "objective_value": float('inf'),
            "routes": [],
            "schedule": {}
        }

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)

    if logger and best_solution["objective_value"] < float('inf'):
        logger.log_solution(best_solution["objective_value"], best_solution)

if __name__ == "__main__":
    main()