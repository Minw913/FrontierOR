import json
import argparse
import time
import math
from itertools import combinations
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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

    depot = instance['depot']
    customers = instance['customers']
    capacity = instance['vehicle_capacity']
    n = len(customers)
    N = list(range(n))
    depot_id = depot['id']
    depot_ready = depot['ready_time']
    depot_due = depot['due_date']

    # Precompute distances and travel times (travel time = distance)
    points = [(depot['x'], depot['y'])] + [(c['x'], c['y']) for c in customers]
    dist = [[0.0 for _ in range(n+1)] for _ in range(n+1)]
    for i in range(n+1):
        for j in range(n+1):
            if i != j:
                dist[i][j] = euclidean_distance(points[i][0], points[i][1], points[j][0], points[j][1])

    demands = [0] + [c['demand'] for c in customers]
    ready = [depot_ready] + [c['ready_time'] for c in customers]
    due = [depot_due] + [c['due_date'] for c in customers]
    service = [0] + [c['service_time'] for c in customers]

    # Build feasible routes using dynamic programming for VRPTW with capacity
    # Label: (position, time, load, path, cost)
    # Since n is small in typical VRPTW benchmarks for exact methods, we generate all feasible routes
    feasible_routes = []
    route_costs = []

    # Use DP to generate all feasible paths
    # State: subset, last_customer, but with time and capacity it's heavy.
    # For practicality, since single core and time limit, we use heuristic column generation like approach or MIP with time limit.

    # We will formulate the Set Covering formulation with all possible feasible routes if possible, but generate them via recursion with pruning.

    def is_feasible_route(path):
        # path is list of customer indices (0-based, excluding depot)
        if not path:
            return False
        current_load = sum(demands[i+1] for i in path)
        if current_load > capacity:
            return False
        current_time = depot_ready
        prev = 0  # depot
        for cust_idx in path:
            node = cust_idx + 1
            travel = dist[prev][node]
            arrival = current_time + travel
            if arrival > due[node]:
                return False
            start_service = max(arrival, ready[node])
            current_time = start_service + service[node]
            prev = node
        # Return to depot
        travel_back = dist[prev][0]
        arrival_depot = current_time + travel_back
        if arrival_depot > depot_due:
            return False
        return True

    def route_distance(path):
        if not path:
            return 0.0
        d = dist[0][path[0]+1]
        prev = path[0] + 1
        for i in range(1, len(path)):
            nxt = path[i] + 1
            d += dist[prev][nxt]
            prev = nxt
        d += dist[prev][0]
        return d

    # Generate all feasible routes by recursion with pruning (branch and bound style)
    def generate_routes(current_path, current_load, current_time, last_node, visited, start_time):
        if len(current_path) > 0:
            # Check if we can return to depot from here
            arrival_depot = current_time + dist[last_node][0]
            if arrival_depot <= depot_due:
                feasible_routes.append(list(current_path))
                route_costs.append(route_distance(current_path))
        
        for i in N:
            if i in visited:
                continue
            node = i + 1
            if current_load + demands[node] > capacity:
                continue
            travel = dist[last_node][node]
            arrival = current_time + travel
            if arrival > due[node]:
                continue
            start_serv = max(arrival, ready[node])
            new_time = start_serv + service[node]
            # Early pruning: if even going directly back would be too late, but we already check on add
            current_path.append(i)
            visited.add(i)
            generate_routes(current_path, current_load + demands[node], new_time, node, visited, start_time)
            current_path.pop()
            visited.remove(i)
    
    start_gen = time.time()
    generate_routes([], 0, depot_ready, 0, set(), depot_ready)
    gen_time = time.time() - start_gen
    
    print(f"Generated {len(feasible_routes)} feasible routes in {gen_time:.2f} seconds.")

    if len(feasible_routes) == 0:
        print("No feasible routes found. Problem might be infeasible.")
        # Write empty/infeasible placeholder
        solution = {
            "objective_value": 0.0,
            "routes": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # Now solve Set Covering Problem with Gurobi, with time limit
    model = gp.Model("VRPTW_SetCover")
    model.setParam('TimeLimit', max(1, args.time_limit - int(time.time() - start_gen) - 5))  # reserve some time
    model.setParam('MIPGap', 0.01)  # Allow 1% gap if needed
    model.setParam('Threads', 1)
    model.setParam('LogToConsole', 0)

    # Variables: x_r = 1 if route r is used
    x = model.addVars(len(feasible_routes), vtype=GRB.BINARY, name="x")

    # Objective: minimize sum cost_r * x_r
    obj = gp.quicksum(route_costs[r] * x[r] for r in range(len(feasible_routes)))
    model.setObjective(obj, GRB.MINIMIZE)

    # Each customer covered at least once
    for i in N:
        covering_routes = [r for r in range(len(feasible_routes)) if i in feasible_routes[r]]
        model.addConstr(gp.quicksum(x[r] for r in covering_routes) >= 1, f"cover_{i}")

    # Callback to log incumbent solutions
    best_obj = float('inf')
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj - 1e-6:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    start_opt = time.time()
    model.optimize(callback)
    opt_time = time.time() - start_opt

    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        print("Warning: Solver did not find a feasible solution.")
        # Fallback: use cheapest routes or single customer routes
        selected = []
        objective = 0.0
        covered = set()
        # Sort routes by cost efficiency
        idxs = list(range(len(feasible_routes)))
        idxs.sort(key=lambda k: route_costs[k])
        for k in idxs:
            route = feasible_routes[k]
            if any(i not in covered for i in route):
                selected.append(k)
                objective += route_costs[k]
                for i in route:
                    covered.add(i)
                if len(covered) == n:
                    break
    else:
        selected = [r for r in range(len(feasible_routes)) if x[r].X > 0.5]
        objective = model.ObjVal

    print(f"Final objective: {objective}")

    # Build solution
    routes_out = []
    for r_idx in selected:
        cust_path = feasible_routes[r_idx]
        # customer ids are original from instance
        cust_ids = [customers[i]['id'] for i in cust_path]
        full_path = [depot_id] + cust_ids + [depot_id]
        routes_out.append({
            "customer_ids": cust_ids,
            "path": full_path
        })

    solution = {
        "objective_value": float(objective),
        "routes": routes_out
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger and best_obj < float('inf'):
        logger.log(float(objective))

if __name__ == "__main__":
    main()