import argparse
import json
import math
import time
import random
import sys
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    num_customers = data['num_customers']
    num_vehicles = data['num_vehicles']
    vehicle_capacity = data['vehicle_capacity']
    gamma = data['uncertainty_set']['Gamma']
    depot = data['depot']
    customers = data['customers']
    dist_matrix = data['distance_matrix']
    
    # Build customer info
    cust_ids = [c['id'] for c in customers]
    nominal = {c['id']: c['nominal_demand'] for c in customers}
    deviation = {c['id']: c['demand_deviation'] for c in customers}
    
    def worst_case_demand(route_customers):
        """
        Compute worst-case demand for a set of customers under the uncertainty set.
        The worst case is: sum of nominal demands + max over subsets S with |fractional| <= Gamma
        of sum of deviations. This equals:
        sum(nominal) + sum of top-Gamma deviations (sorted descending),
        but Gamma can be fractional effectively. Since Gamma is int here,
        it's sum of nominal + sum of top min(Gamma, len(route)) largest deviations.
        """
        if not route_customers:
            return 0
        nom_sum = sum(nominal[c] for c in route_customers)
        devs = sorted([deviation[c] for c in route_customers], reverse=True)
        k = min(gamma, len(devs))
        dev_sum = sum(devs[:k])
        return nom_sum + dev_sum
    
    def route_cost(route_customers):
        """Compute cost of route: depot -> c1 -> c2 -> ... -> cn -> depot"""
        if not route_customers:
            return 0
        cost = dist_matrix[0][route_customers[0]]
        for i in range(len(route_customers) - 1):
            cost += dist_matrix[route_customers[i]][route_customers[i+1]]
        cost += dist_matrix[route_customers[-1]][0]
        return cost
    
    def is_feasible_route(route_customers):
        return worst_case_demand(route_customers) <= vehicle_capacity
    
    # --- Initial solution using nearest neighbor heuristic with capacity check ---
    def build_initial_solution():
        unvisited = set(cust_ids)
        routes = []
        
        for v in range(num_vehicles):
            if not unvisited:
                routes.append([])
                continue
            route = []
            current = 0  # depot
            while unvisited:
                # Find nearest feasible customer
                best_cust = None
                best_dist = float('inf')
                for c in unvisited:
                    test_route = route + [c]
                    if is_feasible_route(test_route):
                        d = dist_matrix[current][c]
                        if d < best_dist:
                            best_dist = d
                            best_cust = c
                if best_cust is None:
                    break
                route.append(best_cust)
                unvisited.remove(best_cust)
                current = best_cust
            routes.append(route)
        
        # If there are still unvisited customers, try to fit them
        for c in list(unvisited):
            placed = False
            for r in routes:
                test_route = r + [c]
                if is_feasible_route(test_route):
                    r.append(c)
                    unvisited.remove(c)
                    placed = True
                    break
            if not placed:
                # Force into smallest route
                min_route = min(routes, key=lambda r: worst_case_demand(r))
                min_route.append(c)
                unvisited.remove(c)
        
        return routes
    
    # --- Clarke-Wright Savings heuristic ---
    def build_cw_solution():
        # Start with each customer in its own route
        routes = {c: [c] for c in cust_ids}
        route_of = {c: c for c in cust_ids}  # customer -> route key
        
        # Compute savings
        savings = []
        for i in cust_ids:
            for j in cust_ids:
                if i != j:
                    s = dist_matrix[0][i] + dist_matrix[j][0] - dist_matrix[i][j]
                    savings.append((s, i, j))
        savings.sort(reverse=True)
        
        for s, i, j in savings:
            if s <= 0:
                break
            ri = route_of[i]
            rj = route_of[j]
            if ri == rj:
                continue
            if ri not in routes or rj not in routes:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            # i must be last in its route, j must be first in its route
            if route_i[-1] != i or route_j[0] != j:
                continue
            # Check capacity
            merged = route_i + route_j
            if not is_feasible_route(merged):
                continue
            # Check vehicle limit: total routes should not go below num_vehicles
            # Actually we can merge as long as we don't have too few routes at the end
            # We'll handle that later
            # Merge
            routes[ri] = merged
            del routes[rj]
            for c in route_j:
                route_of[c] = ri
        
        result = list(routes.values())
        
        # If we have more routes than vehicles, we need to merge more
        while len(result) > num_vehicles:
            # Try to merge the two smallest routes
            result.sort(key=lambda r: worst_case_demand(r))
            merged = False
            for i in range(len(result)):
                for j in range(i+1, len(result)):
                    test = result[i] + result[j]
                    if is_feasible_route(test):
                        result[i] = test
                        result.pop(j)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                # Force merge the two smallest
                result[0] = result[0] + result[1]
                result.pop(1)
        
        # If fewer routes than vehicles, split or add empty
        while len(result) < num_vehicles:
            result.append([])
        
        return result
    
    def total_cost(routes):
        return sum(route_cost(r) for r in routes)
    
    def solution_feasible(routes):
        all_custs = []
        for r in routes:
            if not is_feasible_route(r):
                return False
            all_custs.extend(r)
        return set(all_custs) == set(cust_ids) and len(all_custs) == len(cust_ids)
    
    # Build initial solutions and pick best
    sol1 = build_initial_solution()
    sol2 = build_cw_solution()
    
    if solution_feasible(sol1) and solution_feasible(sol2):
        routes = sol1 if total_cost(sol1) < total_cost(sol2) else sol2
    elif solution_feasible(sol1):
        routes = sol1
    elif solution_feasible(sol2):
        routes = sol2
    else:
        routes = sol1  # fallback
    
    best_routes = [list(r) for r in routes]
    best_cost = total_cost(best_routes)
    
    if logger:
        logger.log(best_cost)
    
    # --- Optimize each route with 2-opt ---
    def two_opt_route(route):
        if len(route) <= 2:
            return route
        improved = True
        route = list(route)
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    # Reverse segment [i, j]
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_cost(new_route) < route_cost(route):
                        route = new_route
                        improved = True
        return route
    
    # Apply 2-opt to each route
    for idx in range(len(best_routes)):
        best_routes[idx] = two_opt_route(best_routes[idx])
    
    current_cost = total_cost(best_routes)
    if current_cost < best_cost:
        best_cost = current_cost
        if logger:
            logger.log(best_cost)
    
    # --- Local search: inter-route moves ---
    def try_relocate(routes):
        """Try moving a customer from one route to another."""
        best_improvement = 0
        best_move = None
        
        for r1_idx in range(len(routes)):
            if len(routes[r1_idx]) == 0:
                continue
            for pos1 in range(len(routes[r1_idx])):
                c = routes[r1_idx][pos1]
                # Cost of removing c from r1
                r1 = routes[r1_idx]
                # Current cost contribution of c in r1
                prev1 = 0 if pos1 == 0 else r1[pos1-1]
                if pos1 == 0:
                    prev1 = 0  # depot
                else:
                    prev1 = r1[pos1-1]
                next1 = 0 if pos1 == len(r1)-1 else r1[pos1+1]
                if pos1 == len(r1)-1:
                    next1_node = 0
                else:
                    next1_node = r1[pos1+1]
                if pos1 == 0:
                    prev1_node = 0
                else:
                    prev1_node = r1[pos1-1]
                
                remove_saving = (dist_matrix[prev1_node][c] + dist_matrix[c][next1_node] 
                                - dist_matrix[prev1_node][next1_node])
                
                for r2_idx in range(len(routes)):
                    if r2_idx == r1_idx:
                        continue
                    r2 = routes[r2_idx]
                    # Check if adding c to r2 is feasible
                    test_r2_custs = r2 + [c]
                    if not is_feasible_route(test_r2_custs):
                        continue
                    # Also check r1 without c is still ok (always feasible since demand decreases)
                    
                    # Find best insertion position in r2
                    best_insert_cost = float('inf')
                    best_pos2 = 0
                    for pos2 in range(len(r2) + 1):
                        if pos2 == 0:
                            prev2 = 0
                        else:
                            prev2 = r2[pos2-1]
                        if pos2 == len(r2):
                            next2 = 0
                        else:
                            next2 = r2[pos2]
                        insert_cost = (dist_matrix[prev2][c] + dist_matrix[c][next2] 
                                      - dist_matrix[prev2][next2])
                        if insert_cost < best_insert_cost:
                            best_insert_cost = insert_cost
                            best_pos2 = pos2
                    
                    improvement = remove_saving - best_insert_cost
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (r1_idx, pos1, r2_idx, best_pos2, c)
        
        return best_move, best_improvement
    
    def try_swap(routes):
        """Try swapping a customer between two routes."""
        best_improvement = 0
        best_move = None
        
        for r1_idx in range(len(routes)):
            r1 = routes[r1_idx]
            if not r1:
                continue
            for r2_idx in range(r1_idx + 1, len(routes)):
                r2 = routes[r2_idx]
                if not r2:
                    continue
                for p1 in range(len(r1)):
                    for p2 in range(len(r2)):
                        c1 = r1[p1]
                        c2 = r2[p2]
                        
                        # Check feasibility of swap
                        new_r1 = r1[:p1] + [c2] + r1[p1+1:]
                        new_r2 = r2[:p2] + [c1] + r2[p2+1:]
                        
                        if not is_feasible_route(new_r1) or not is_feasible_route(new_r2):
                            continue
                        
                        old_cost = route_cost(r1) + route_cost(r2)
                        new_cost = route_cost(new_r1) + route_cost(new_r2)
                        
                        improvement = old_cost - new_cost
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (r1_idx, p1, r2_idx, p2)
        
        return best_move, best_improvement
    
    def try_or_opt(routes):
        """Try moving a segment of 1, 2, or 3 customers from one route to another."""
        best_improvement = 0
        best_move = None
        
        for seg_len in [1, 2, 3]:
            for r1_idx in range(len(routes)):
                r1 = routes[r1_idx]
                if len(r1) < seg_len:
                    continue
                for pos1 in range(len(r1) - seg_len + 1):
                    segment = r1[pos1:pos1+seg_len]
                    
                    # Cost of removing segment
                    if pos1 == 0:
                        prev_node = 0
                    else:
                        prev_node = r1[pos1-1]
                    if pos1 + seg_len >= len(r1):
                        next_node = 0
                    else:
                        next_node = r1[pos1+seg_len]
                    
                    # Current cost of segment in route
                    old_seg_cost = dist_matrix[prev_node][segment[0]]
                    for k in range(seg_len - 1):
                        old_seg_cost += dist_matrix[segment[k]][segment[k+1]]
                    old_seg_cost += dist_matrix[segment[-1]][next_node]
                    new_direct = dist_matrix[prev_node][next_node]
                    remove_saving = old_seg_cost - new_direct
                    
                    for r2_idx in range(len(routes)):
                        if r2_idx == r1_idx:
                            continue
                        r2 = routes[r2_idx]
                        # Check feasibility
                        test_custs = r2 + segment
                        if not is_feasible_route(test_custs):
                            continue
                        
                        # Best insertion in r2
                        for pos2 in range(len(r2) + 1):
                            if pos2 == 0:
                                prev2 = 0
                            else:
                                prev2 = r2[pos2-1]
                            if pos2 == len(r2):
                                next2 = 0
                            else:
                                next2 = r2[pos2]
                            
                            insert_cost = dist_matrix[prev2][segment[0]]
                            for k in range(seg_len - 1):
                                insert_cost += dist_matrix[segment[k]][segment[k+1]]
                            insert_cost += dist_matrix[segment[-1]][next2]
                            insert_cost -= dist_matrix[prev2][next2]
                            
                            improvement = remove_saving - insert_cost
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_move = (r1_idx, pos1, seg_len, r2_idx, pos2)
        
        return best_move, best_improvement

    # Local search loop
    def local_search(routes, time_budget):
        current_routes = [list(r) for r in routes]
        improved = True
        
        while improved and time.time() - start_time < time_budget:
            improved = False
            
            # Try relocate
            move, imp = try_relocate(current_routes)
            if move and imp > 0:
                r1_idx, pos1, r2_idx, pos2, c = move
                current_routes[r1_idx].pop(pos1)
                current_routes[r2_idx].insert(pos2, c)
                # Apply 2-opt to affected routes
                current_routes[r1_idx] = two_opt_route(current_routes[r1_idx])
                current_routes[r2_idx] = two_opt_route(current_routes[r2_idx])
                improved = True
                continue
            
            # Try swap
            move, imp = try_swap(current_routes)
            if move and imp > 0:
                r1_idx, p1, r2_idx, p2 = move
                current_routes[r1_idx][p1], current_routes[r2_idx][p2] = current_routes[r2_idx][p2], current_routes[r1_idx][p1]
                current_routes[r1_idx] = two_opt_route(current_routes[r1_idx])
                current_routes[r2_idx] = two_opt_route(current_routes[r2_idx])
                improved = True
                continue
            
            # Try or-opt (inter-route)
            move, imp = try_or_opt(current_routes)
            if move and imp > 0:
                r1_idx, pos1, seg_len, r2_idx, pos2 = move
                segment = current_routes[r1_idx][pos1:pos1+seg_len]
                del current_routes[r1_idx][pos1:pos1+seg_len]
                for k, c in enumerate(segment):
                    current_routes[r2_idx].insert(pos2 + k, c)
                current_routes[r1_idx] = two_opt_route(current_routes[r1_idx])
                current_routes[r2_idx] = two_opt_route(current_routes[r2_idx])
                improved = True
                continue
        
        return current_routes
    
    time_budget = start_time + time_limit * 0.7
    best_routes = local_search(best_routes, time_budget)
    current_cost = total_cost(best_routes)
    if current_cost < best_cost:
        best_cost = current_cost
        if logger:
            logger.log(best_cost)
    
    # --- Simulated Annealing / Perturbation phase ---
    def perturb(routes):
        """Random perturbation: relocate or swap a random customer."""
        new_routes = [list(r) for r in routes]
        non_empty = [i for i in range(len(new_routes)) if len(new_routes[i]) > 0]
        
        if len(non_empty) < 1:
            return new_routes
        
        move_type = random.choice(['relocate', 'swap', 'or_opt_intra'])
        
        if move_type == 'relocate' and len(non_empty) >= 1:
            r1_idx = random.choice(non_empty)
            if len(new_routes[r1_idx]) == 0:
                return new_routes
            pos1 = random.randrange(len(new_routes[r1_idx]))
            c = new_routes[r1_idx][pos1]
            
            # Pick a different route (can be empty)
            candidates = [i for i in range(len(new_routes)) if i != r1_idx]
            if not candidates:
                return new_routes
            r2_idx = random.choice(candidates)
            
            # Check feasibility
            test = new_routes[r2_idx] + [c]
            if is_feasible_route(test):
                new_routes[r1_idx].pop(pos1)
                pos2 = random.randrange(len(new_routes[r2_idx]) + 1)
                new_routes[r2_idx].insert(pos2, c)
        
        elif move_type == 'swap' and len(non_empty) >= 2:
            r1_idx, r2_idx = random.sample(non_empty, 2)
            if len(new_routes[r1_idx]) == 0 or len(new_routes[r2_idx]) == 0:
                return new_routes
            p1 = random.randrange(len(new_routes[r1_idx]))
            p2 = random.randrange(len(new_routes[r2_idx]))
            
            c1, c2 = new_routes[r1_idx][p1], new_routes[r2_idx][p2]
            test_r1 = new_routes[r1_idx][:p1] + [c2] + new_routes[r1_idx][p1+1:]
            test_r2 = new_routes[r2_idx][:p2] + [c1] + new_routes[r2_idx][p2+1:]
            
            if is_feasible_route(test_r1) and is_feasible_route(test_r2):
                new_routes[r1_idx] = test_r1
                new_routes[r2_idx] = test_r2
        
        elif move_type == 'or_opt_intra':
            r_idx = random.choice(non_empty)
            r = new_routes[r_idx]
            if len(r) >= 4:
                # Random 2-opt within route
                i = random.randrange(len(r) - 1)
                j = random.randrange(i + 1, len(r))
                new_routes[r_idx] = r[:i] + r[i:j+1][::-1] + r[j+1:]
        
        return new_routes
    
    # SA phase
    current_routes = [list(r) for r in best_routes]
    current_cost = best_cost
    temperature = best_cost * 0.02
    cooling_rate = 0.9999
    
    iterations = 0
    time_budget_sa = start_time + time_limit * 0.95
    
    while time.time() < time_budget_sa:
        iterations += 1
        
        new_routes = perturb(current_routes)
        
        # Quick validation
        all_custs = []
        for r in new_routes:
            all_custs.extend(r)
        if set(all_custs) != set(cust_ids) or len(all_custs) != len(cust_ids):
            continue
        
        feasible = all(is_feasible_route(r) for r in new_routes)
        if not feasible:
            continue
        
        new_cost = total_cost(new_routes)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
            current_routes = new_routes
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_routes = [list(r) for r in current_routes]
                best_cost = current_cost
                if logger:
                    logger.log(best_cost)
        
        temperature *= cooling_rate
        
        # Periodically run local search on current solution
        if iterations % 5000 == 0 and time.time() < time_budget_sa - 2:
            # Quick 2-opt on all routes
            for idx in range(len(current_routes)):
                current_routes[idx] = two_opt_route(current_routes[idx])
            current_cost = total_cost(current_routes)
            if current_cost < best_cost:
                best_routes = [list(r) for r in current_routes]
                best_cost = current_cost
                if logger:
                    logger.log(best_cost)
    
    # Final 2-opt on best routes
    for idx in range(len(best_routes)):
        best_routes[idx] = two_opt_route(best_routes[idx])
    best_cost = total_cost(best_routes)
    
    # Final local search with remaining time
    remaining = start_time + time_limit * 0.99 - time.time()
    if remaining > 1:
        best_routes = local_search(best_routes, start_time + time_limit * 0.99)
        best_cost = total_cost(best_routes)
        if logger:
            logger.log(best_cost)
    
    # Build output
    output_routes = []
    for r in best_routes:
        output_routes.append({
            "customers": r,
            "cost": route_cost(r),
            "worst_case_demand": worst_case_demand(r)
        })
    
    solution = {
        "objective_value": best_cost,
        "routes": output_routes
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)