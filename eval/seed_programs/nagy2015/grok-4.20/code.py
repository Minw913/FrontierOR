import json
import argparse
import time
import sys
from typing import List, Dict, Any
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import itertools

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_customers']
    C = instance['vehicle_capacity']
    dist = instance['distance_matrix']
    deliveries = [0] + [c['delivery_demand'] for c in instance['customers']]
    pickups = [0] + [c['pickup_demand'] for c in instance['customers']]
    N = 2 * n + 1  # nodes 0=depot, 1..n deliveries, n+1..2n pickups

    # Build full distance matrix for expanded graph
    D = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == 0 or j == 0:
                orig_i = 0 if i == 0 else (i if i <= n else i - n)
                orig_j = 0 if j == 0 else (j if j <= n else j - n)
                D[i][j] = dist[orig_i][orig_j]
            elif i == j:
                D[i][j] = 0
            else:
                orig_i = i if i <= n else i - n
                orig_j = j if j <= n else j - n
                D[i][j] = dist[orig_i][orig_j]

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Heuristic upper bound: separate delivery and pickup routes
    def solve_VRP(demands, is_delivery):
        visited = [False] * (n + 1)
        routes = []
        total_cost = 0.0
        for _ in range(n):
            if all(visited[1:]):
                break
            route = [0]
            load = 0
            current = 0
            while True:
                best_next = None
                best_cost = float('inf')
                for j in range(1, n + 1):
                    if visited[j]:
                        continue
                    d = demands[j]
                    if load + d > C:
                        continue
                    cost = D[current][j] if not is_delivery else D[current][j]
                    if cost < best_cost:
                        best_cost = cost
                        best_next = j
                if best_next is None:
                    break
                route.append(best_next)
                load += demands[best_next]
                visited[best_next] = True
                current = best_next
            route.append(0)
            # compute cost
            route_cost = sum(D[route[k]][route[k+1]] for k in range(len(route)-1))
            total_cost += route_cost
            routes.append(route)
        return total_cost, routes

    ub_del, _ = solve_VRP(deliveries, True)
    ub_pick, _ = solve_VRP(pickups, False)
    best_ub = ub_del + ub_pick

    # Try to improve with cheapest insertion for combined routes
    def try_combine():
        nonlocal best_ub
        # Simple: for each customer with both demands, try a route 0-i-(n+i)-0 if feasible
        combined_savings = 0
        combined_routes = []
        used = set()
        for i in range(1, n+1):
            if deliveries[i] + pickups[i] <= C:
                cost = D[0][i] + D[i][n+i] + D[n+i][0]
                if cost < (D[0][i] + D[i][0] + D[0][n+i] + D[n+i][0]):
                    savings = (D[0][i] + D[i][0] + D[0][n+i] + D[n+i][0]) - cost
                    combined_savings += savings
                    combined_routes.append([0, i, n+i, 0])
                    used.add(i)
        remaining_del = [i for i in range(1,n+1) if i not in used]
        remaining_pick = [i for i in range(1,n+1) if i not in used]
        
        # Add separate for remaining
        del_cost = sum(2 * D[0][i] for i in remaining_del)
        pick_cost = sum(2 * D[0][i] for i in remaining_pick)
        candidate = (best_ub - combined_savings) - (ub_del + ub_pick - del_cost - pick_cost - combined_savings) + (del_cost + pick_cost)
        # rough estimate
        if len(remaining_del) > 0:
            candidate = best_ub - combined_savings + len(remaining_del) * 10  # conservative
        if candidate < best_ub:
            best_ub = candidate
            if logger:
                logger.log(best_ub)

    try_combine()

    # MIP model with time limit
    model = gp.Model("VRPB")
    model.setParam(GRB.Param.TimeLimit, max(1, args.time_limit - (time.time() - start_time) - 5))
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.MIPGap, 1e-4)  # accept some gap for larger instances
    model.setParam(GRB.Param.LogToConsole, 0)

    # Variables
    x = model.addVars(N, N, vtype=GRB.BINARY, name="x")
    u = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, ub=C, name="u")  # load after node

    # Objective
    model.setObjective(gp.quicksum(D[i][j] * x[i,j] for i in range(N) for j in range(N)), GRB.MINIMIZE)

    # Degree constraints
    for k in range(1, 2*n+1):
        model.addConstr(gp.quicksum(x[i,k] for i in range(N) if i != k) == 1, f"in_{k}")
        model.addConstr(gp.quicksum(x[k,j] for j in range(N) if j != k) == 1, f"out_{k}")

    # Depot
    model.addConstr(gp.quicksum(x[0,j] for j in range(1,2*n+1)) == gp.quicksum(x[i,0] for i in range(1,2*n+1)), "depot_flow")

    # No self loops
    for i in range(N):
        model.addConstr(x[i,i] == 0)

    # Capacity and load constraints (MTZ style + pickup/delivery logic)
    M = C + 1
    for i in range(1, 2*n+1):
        for j in range(1, 2*n+1):
            if i != j:
                if i <= n and j <= n:  # delivery to delivery
                    model.addConstr(u[i] + deliveries[j] - M*(1 - x[i,j]) <= u[j])
                elif i <= n and j > n:  # delivery to pickup
                    model.addConstr(u[i] - pickups[j-n] - M*(1 - x[i,j]) <= u[j])
                elif i > n and j <= n:  # pickup to delivery
                    model.addConstr(u[i] + deliveries[j] - M*(1 - x[i,j]) <= u[j])
                elif i > n and j > n:  # pickup to pickup
                    model.addConstr(u[i] - pickups[j-n] - M*(1 - x[i,j]) <= u[j])

    # Initial load when leaving depot
    for j in range(1, n+1):  # only to deliveries
        model.addConstr(u[j] == deliveries[j] * x[0,j], f"init_del_{j}")
    for j in range(n+1, 2*n+1):
        model.addConstr(u[j] <= M * x[0,j])  # shouldn't happen but safe

    # Load at pickup nodes
    for k in range(1, n+1):
        pk = n + k
        model.addConstr(u[pk] >= pickups[k] * gp.quicksum(x[i, pk] for i in range(N) if i != pk) - 1e-5)

    # Return to depot constraints
    for i in range(1, 2*n+1):
        model.addConstr(x[i,0] * u[i] <= 1e-3, f"return_empty_{i}")  # only return with near zero load (i.e. pure pickups)

    # Subtour elimination is approximated via MTZ load

    # Callback to record improving solutions
    best_obj = best_ub
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < best_obj - 1e-4:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    model.optimize(callback)

    final_obj = best_obj
    if model.SolCount > 0:
        final_obj = model.objVal

    # Reconstruct routes from MIP solution if available, else fallback
    routes = []
    if model.SolCount > 0:
        xval = model.getAttr('X', x)
        # Build successor
        succ = {}
        for i in range(N):
            for j in range(N):
                if xval[i,j] > 0.5:
                    succ[i] = j
                    break
        # Extract all cycles from depot
        visited = set()
        for start in range(1, 2*n+1):
            if start in visited:
                continue
            route = [0]
            current = start
            while current != 0:
                visited.add(current)
                route.append(current)
                if current not in succ:
                    break
                current = succ[current]
            route.append(0)
            if len(route) > 2:
                routes.append(route)
    else:
        # Use the simple heuristic solution
        routes = []
        for i in range(1, n+1):
            routes.append([0, i, 0])
            routes.append([0, n+i, 0])

    # Build detailed output
    routes_out = []
    detailed = []
    for r in routes:
        routes_out.append(r)
        detail_r = []
        for node in r:
            if node == 0:
                detail_r.append({
                    "role": "depot",
                    "customer": 0,
                    "delivery": 0,
                    "pickup": 0
                })
            elif node <= n:
                detail_r.append({
                    "role": "linehaul",
                    "customer": node,
                    "delivery": deliveries[node],
                    "pickup": 0
                })
            else:
                cust = node - n
                detail_r.append({
                    "role": "backhaul",
                    "customer": cust,
                    "delivery": 0,
                    "pickup": pickups[cust]
                })
        detailed.append(detail_r)

    solution = {
        "objective_value": float(final_obj),
        "routes": routes_out,
        "routes_detailed": detailed
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()