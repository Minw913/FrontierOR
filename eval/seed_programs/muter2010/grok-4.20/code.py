import json
import argparse
import time
import math
from itertools import combinations
from solution_logger import SolutionLogger

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
    dist = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                dist[i][j] = euclidean_distance(points[i][0], points[i][1], points[j][0], points[j][1])

    demands = [0] + [c['demand'] for c in customers]
    ready_times = [depot_ready] + [c['ready_time'] for c in customers]
    due_times = [depot_due] + [c['due_date'] for c in customers]
    service_times = [0] + [c['service_time'] for c in customers]

    def is_feasible(path):
        if not path:
            return False, 0.0
        load = sum(demands[i + 1] for i in path)
        if load > capacity:
            return False, 0.0
        t = depot_ready
        prev = 0
        for idx in path:
            node = idx + 1
            arrival = t + dist[prev][node]
            if arrival > due_times[node]:
                return False, 0.0
            t = max(arrival, ready_times[node]) + service_times[node]
            prev = node
        arrival_depot = t + dist[prev][0]
        if arrival_depot > depot_due:
            return False, 0.0
        cost = dist[0][path[0] + 1]
        prev = path[0] + 1
        for k in range(1, len(path)):
            cost += dist[prev][path[k] + 1]
            prev = path[k] + 1
        cost += dist[prev][0]
        return True, cost

    # Generate all feasible routes via recursion with pruning
    feasible_routes = []
    route_costs = []
    customer_ids = [c['id'] for c in customers]

    def generate(current_path, load, t, last, visited):
        feasible, cost = is_feasible(current_path)
        if feasible:
            feasible_routes.append(list(current_path))
            route_costs.append(cost)
        
        for i in N:
            if i in visited:
                continue
            node = i + 1
            if load + demands[node] > capacity:
                continue
            arrival = t + dist[last][node]
            if arrival > due_times[node]:
                continue
            start_s = max(arrival, ready_times[node])
            new_t = start_s + service_times[node]
            current_path.append(i)
            visited.add(i)
            generate(current_path, load + demands[node], new_t, node, visited)
            current_path.pop()
            visited.remove(i)

    start_gen = time.time()
    generate([], 0, depot_ready, 0, set())
    gen_time = time.time() - start_gen
    print(f"Generated {len(feasible_routes)} feasible routes in {gen_time:.2f}s")

    # If too many routes, keep only a subset (cheapest per customer)
    if len(feasible_routes) > 5000:
        print("Too many routes, filtering...")
        route_list = list(zip(route_costs, feasible_routes))
        route_list.sort()
        seen = set()
        filtered_routes = []
        filtered_costs = []
        for cost, path in route_list:
            for cust in path:
                if cust not in seen:
                    seen.add(cust)
                    filtered_routes.append(path)
                    filtered_costs.append(cost)
                    break
            if len(seen) == n and len(filtered_routes) > 50:
                break
        feasible_routes = filtered_routes
        route_costs = filtered_costs
        print(f"Filtered to {len(feasible_routes)} routes")

    # Greedy set cover like algorithm with local improvement
    start_time = time.time()
    best_objective = float('inf')
    best_selection = []
    
    # Sort routes by cost
    route_indices = list(range(len(feasible_routes)))
    route_indices.sort(key=lambda k: route_costs[k])

    # Simple greedy + 2-opt like perturbation
    covered = [False] * n
    selection = []
    current_cost = 0.0
    for r in route_indices:
        improved = False
        for cust in feasible_routes[r]:
            if not covered[cust]:
                improved = True
                break
        if improved:
            selection.append(r)
            current_cost += route_costs[r]
            for cust in feasible_routes[r]:
                covered[cust] = True
        if all(covered):
            break

    if all(covered):
        best_objective = current_cost
        best_selection = list(selection)
        if logger:
            logger.log(float(best_objective))
        print(f"Initial solution: {best_objective}")

    # Local search: try to replace routes
    improved = True
    iterations = 0
    max_iter = 1000 if n <= 30 else 200
    time_limit = args.time_limit - 5

    while improved and iterations < max_iter and (time.time() - start_time) < time_limit:
        improved = False
        iterations += 1
        for i in range(len(best_selection)):
            r1 = best_selection[i]
            for r2_idx in route_indices:
                if r2_idx in best_selection:
                    continue
                if route_costs[r2_idx] >= route_costs[r1]:
                    continue
                new_selection = best_selection[:i] + best_selection[i+1:] + [r2_idx]
                new_covered = [False] * n
                new_cost = sum(route_costs[k] for k in new_selection)
                if new_cost >= best_objective:
                    continue
                for k in new_selection:
                    for cust in feasible_routes[k]:
                        new_covered[cust] = True
                if all(new_covered):
                    best_selection = new_selection
                    best_objective = new_cost
                    improved = True
                    if logger:
                        logger.log(float(best_objective))
                    print(f"Improved to {best_objective} at iteration {iterations}")
                    break
            if improved:
                break

    # Final reconstruction
    selected_routes = []
    for r_idx in best_selection:
        path = feasible_routes[r_idx]
        cust_list = [customer_ids[i] for i in path]
        full_path = [depot_id] + cust_list + [depot_id]
        selected_routes.append({
            "customer_ids": cust_list,
            "path": full_path
        })

    # Sort routes by first customer id for consistency
    selected_routes.sort(key=lambda r: r["customer_ids"][0] if r["customer_ids"] else 0)

    solution = {
        "objective_value": float(best_objective),
        "routes": selected_routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Final solution cost: {best_objective:.4f} with {len(selected_routes)} routes")

if __name__ == "__main__":
    main()