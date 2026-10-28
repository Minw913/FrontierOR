import json
import argparse
import time
import math
import random
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_customers = instance['num_customers']
    vehicle_capacity = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    distance_matrix = instance['distance_matrix']
    scheduling_horizon = instance['scheduling_horizon']
    
    n = num_customers + 1
    
    # Parse node data
    ready = [0] * n
    due = [0] * n
    serv = [0] * n
    dem = [0] * n
    
    ready[0] = depot['ready_time']
    due[0] = depot['due_date']
    serv[0] = depot['service_time']
    dem[0] = depot['demand']
    
    for c in customers:
        idx = c['id']
        ready[idx] = c['ready_time']
        due[idx] = c['due_date']
        serv[idx] = c['service_time']
        dem[idx] = c['demand']
    
    # Compute effective travel times
    eff = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            base = distance_matrix[i][j]
            e = base + serv[i]
            d = ready[j] - due[i]
            if d > e:
                e = d
            eff[i][j] = e
    
    def time_remaining():
        return args.time_limit - (time.time() - start_time)
    
    def check_route(route):
        """Check feasibility and compute departure times and loads."""
        if not route:
            return {}, {}, True
        
        dep_times = {}
        loads = {}
        t = ready[0]
        load = 0
        prev = 0
        
        for cust in route:
            t = t + eff[prev][cust]
            if t < ready[cust]:
                t = ready[cust]
            if t > due[cust]:
                return dep_times, loads, False
            load += dem[cust]
            if load > vehicle_capacity:
                return dep_times, loads, False
            dep_times[cust] = t
            loads[cust] = load
            prev = cust
        
        t_return = t + eff[prev][0]
        if t_return > due[0]:
            return dep_times, loads, False
        
        return dep_times, loads, True
    
    def route_dist(route):
        if not route:
            return 0.0
        d = distance_matrix[0][route[0]]
        for i in range(len(route) - 1):
            d += distance_matrix[route[i]][route[i+1]]
        d += distance_matrix[route[-1]][0]
        return d
    
    def total_dist(routes):
        return sum(route_dist(r) for r in routes)
    
    def make_solution(routes):
        all_dep_times = {}
        all_loads = {}
        total_d = 0.0
        full_routes = []
        
        for route in routes:
            full_route = [0] + route + [0]
            full_routes.append(full_route)
            dep_times, loads, feas = check_route(route)
            if not feas:
                return None
            for k, v in dep_times.items():
                all_dep_times[str(k)] = v
            for k, v in loads.items():
                all_loads[str(k)] = v
            total_d += route_dist(route)
        
        return {
            "objective_value": float(len(routes)),
            "num_vehicles": len(routes),
            "routes": full_routes,
            "total_distance": round(total_d, 1),
            "departure_times": all_dep_times,
            "loads": all_loads
        }
    
    # Construction: cheapest insertion
    def construct_greedy(order):
        routes = []
        for cust in order:
            best_r = -1
            best_pos = -1
            best_cost = float('inf')
            
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [cust] + route[pos:]
                    _, _, feas = check_route(new_route)
                    if feas:
                        cost = route_dist(new_route) - route_dist(route)
                        if cost < best_cost:
                            best_cost = cost
                            best_r = r_idx
                            best_pos = pos
            
            if best_r >= 0:
                routes[best_r] = routes[best_r][:best_pos] + [cust] + routes[best_r][best_pos:]
            else:
                routes.append([cust])
        
        return routes
    
    # Regret-2 insertion
    def construct_regret(order):
        routes = []
        unassigned = list(order)
        
        while unassigned:
            best_cust = -1
            best_r = -1
            best_pos = -1
            best_regret = -float('inf')
            
            for cust in unassigned:
                # Find best and second best insertion
                insertions = []
                for r_idx in range(len(routes)):
                    route = routes[r_idx]
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [cust] + route[pos:]
                        _, _, feas = check_route(new_route)
                        if feas:
                            cost = route_dist(new_route) - route_dist(route)
                            insertions.append((cost, r_idx, pos))
                
                # Also consider creating a new route
                new_route_cost = route_dist([cust])
                insertions.append((new_route_cost, -1, 0))
                
                insertions.sort()
                
                if len(insertions) >= 2:
                    regret = insertions[1][0] - insertions[0][0]
                else:
                    regret = insertions[0][0] if insertions else 0
                
                if regret > best_regret or (regret == best_regret and insertions[0][0] < best_regret):
                    best_regret = regret
                    best_cust = cust
                    best_r = insertions[0][1]
                    best_pos = insertions[0][2]
            
            unassigned.remove(best_cust)
            if best_r >= 0:
                routes[best_r] = routes[best_r][:best_pos] + [best_cust] + routes[best_r][best_pos:]
            else:
                routes.append([best_cust])
        
        return routes
    
    # Route elimination
    def eliminate_routes(routes):
        improved = True
        while improved:
            if time_remaining() < 10:
                break
            improved = False
            route_indices = sorted(range(len(routes)), key=lambda x: len(routes[x]))
            
            for r_idx in route_indices:
                if time_remaining() < 10:
                    break
                if r_idx >= len(routes):
                    continue
                
                route_to_remove = routes[r_idx]
                other_routes = [routes[i] for i in range(len(routes)) if i != r_idx]
                
                # Try different orderings of customers to insert
                orderings = [
                    sorted(route_to_remove, key=lambda c: due[c]),
                    sorted(route_to_remove, key=lambda c: ready[c]),
                    sorted(route_to_remove, key=lambda c: -dem[c]),
                    sorted(route_to_remove, key=lambda c: due[c] - ready[c]),
                    list(route_to_remove),
                ]
                
                success = False
                best_result = None
                
                for ordering in orderings:
                    temp_routes = [list(r) for r in other_routes]
                    all_inserted = True
                    
                    for cust in ordering:
                        best_r_i = -1
                        best_pos_i = -1
                        best_cost_i = float('inf')
                        
                        for ri in range(len(temp_routes)):
                            for pos in range(len(temp_routes[ri]) + 1):
                                new_route = temp_routes[ri][:pos] + [cust] + temp_routes[ri][pos:]
                                _, _, feas = check_route(new_route)
                                if feas:
                                    cost = route_dist(new_route) - route_dist(temp_routes[ri])
                                    if cost < best_cost_i:
                                        best_cost_i = cost
                                        best_r_i = ri
                                        best_pos_i = pos
                        
                        if best_r_i >= 0:
                            temp_routes[best_r_i] = temp_routes[best_r_i][:best_pos_i] + [cust] + temp_routes[best_r_i][best_pos_i:]
                        else:
                            all_inserted = False
                            break
                    
                    if all_inserted:
                        success = True
                        best_result = temp_routes
                        break
                
                if success:
                    routes = best_result
                    improved = True
                    break
        
        return [r for r in routes if r]
    
    # Local search: relocate
    def ls_relocate(routes):
        improved = True
        while improved:
            if time_remaining() < 5:
                break
            improved = False
            for r1 in range(len(routes)):
                if improved:
                    break
                for i in range(len(routes[r1])):
                    if improved:
                        break
                    cust = routes[r1][i]
                    for r2 in range(len(routes)):
                        if r1 == r2:
                            continue
                        for pos in range(len(routes[r2]) + 1):
                            new_r1 = routes[r1][:i] + routes[r1][i+1:]
                            new_r2 = routes[r2][:pos] + [cust] + routes[r2][pos:]
                            _, _, f1 = check_route(new_r1)
                            _, _, f2 = check_route(new_r2)
                            if f1 and f2:
                                old_d = route_dist(routes[r1]) + route_dist(routes[r2])
                                new_d = route_dist(new_r1) + route_dist(new_r2)
                                if new_d < old_d - 1e-9:
                                    routes[r1] = new_r1
                                    routes[r2] = new_r2
                                    improved = True
                                    break
                        if improved:
                            break
            routes = [r for r in routes if r]
        return routes
    
    # Local search: 2-opt intra
    def ls_2opt(routes):
        for r_idx in range(len(routes)):
            if time_remaining() < 5:
                break
            route = routes[r_idx]
            improved = True
            while improved:
                improved = False
                for i in range(len(route) - 1):
                    if improved:
                        break
                    for j in range(i + 1, len(route)):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        _, _, feas = check_route(new_route)
                        if feas and route_dist(new_route) < route_dist(route) - 1e-9:
                            route = new_route
                            improved = True
                            break
            routes[r_idx] = route
        return routes
    
    # Or-opt
    def ls_oropt(routes):
        for r_idx in range(len(routes)):
            if time_remaining() < 5:
                break
            route = routes[r_idx]
            improved = True
            while improved:
                improved = False
                for seg_len in [1, 2, 3]:
                    if improved:
                        break
                    for i in range(len(route) - seg_len + 1):
                        if improved:
                            break
                        seg = route[i:i+seg_len]
                        rem = route[:i] + route[i+seg_len:]
                        for pos in range(len(rem) + 1):
                            new_route = rem[:pos] + seg + rem[pos:]
                            _, _, feas = check_route(new_route)
                            if feas and route_dist(new_route) < route_dist(route) - 1e-9:
                                route = new_route
                                improved = True
                                break
            routes[r_idx] = route
        return routes
    
    # Inter-route exchange (swap)
    def ls_exchange(routes):
        improved = True
        while improved:
            if time_remaining() < 5:
                break
            improved = False
            for r1 in range(len(routes)):
                if improved:
                    break
                for r2 in range(r1 + 1, len(routes)):
                    if improved:
                        break
                    for i in range(len(routes[r1])):
                        if improved:
                            break
                        for j in range(len(routes[r2])):
                            c1 = routes[r1][i]
                            c2 = routes[r2][j]
                            new_r1 = routes[r1][:i] + [c2] + routes[r1][i+1:]
                            new_r2 = routes[r2][:j] + [c1] + routes[r2][j+1:]
                            _, _, f1 = check_route(new_r1)
                            _, _, f2 = check_route(new_r2)
                            if f1 and f2:
                                old_d = route_dist(routes[r1]) + route_dist(routes[r2])
                                new_d = route_dist(new_r1) + route_dist(new_r2)
                                if new_d < old_d - 1e-9:
                                    routes[r1] = new_r1
                                    routes[r2] = new_r2
                                    improved = True
                                    break
        return routes
    
    # Cross exchange: move segment from one route to another
    def ls_cross_exchange(routes):
        improved = True
        while improved:
            if time_remaining() < 5:
                break
            improved = False
            for r1 in range(len(routes)):
                if improved:
                    break
                for r2 in range(r1 + 1, len(routes)):
                    if improved:
                        break
                    for i in range(len(routes[r1])):
                        if improved:
                            break
                        for seg_len in [1, 2]:
                            if i + seg_len > len(routes[r1]):
                                break
                            seg = routes[r1][i:i+seg_len]
                            rem1 = routes[r1][:i] + routes[r1][i+seg_len:]
                            for pos in range(len(routes[r2]) + 1):
                                new_r2 = routes[r2][:pos] + seg + routes[r2][pos:]
                                _, _, f1 = check_route(rem1)
                                _, _, f2 = check_route(new_r2)
                                if f1 and f2:
                                    old_d = route_dist(routes[r1]) + route_dist(routes[r2])
                                    new_d = route_dist(rem1) + route_dist(new_r2)
                                    if new_d < old_d - 1e-9:
                                        routes[r1] = rem1
                                        routes[r2] = new_r2
                                        improved = True
                                        break
                            if improved:
                                break
            routes = [r for r in routes if r]
        return routes
    
    # LNS for route reduction
    def lns_reduce(routes, max_iter=500):
        best_routes = [list(r) for r in routes]
        best_num = len(routes)
        best_d = total_dist(routes)
        
        for it in range(max_iter):
            if time_remaining() < 15:
                break
            
            current = [list(r) for r in best_routes]
            if len(current) <= 1:
                break
            
            # Pick target route to eliminate
            target = random.randint(0, len(current) - 1)
            destroyed = list(current[target])
            others = [current[i] for i in range(len(current)) if i != target]
            
            # Also remove some from other routes
            num_extra = random.randint(0, min(4, sum(len(r) for r in others)))
            for _ in range(num_extra):
                if not others:
                    break
                non_empty = [i for i in range(len(others)) if others[i]]
                if not non_empty:
                    break
                ri = random.choice(non_empty)
                ci = random.randint(0, len(others[ri]) - 1)
                destroyed.append(others[ri][ci])
                others[ri] = others[ri][:ci] + others[ri][ci+1:]
            
            # Reinsert
            random.shuffle(destroyed)
            temp = [list(r) for r in others]
            success = True
            
            # Sort by urgency for better insertion
            destroyed.sort(key=lambda c: due[c])
            
            for cust in destroyed:
                best_ri = -1
                best_pos_i = -1
                best_cost_i = float('inf')
                
                for ri in range(len(temp)):
                    for pos in range(len(temp[ri]) + 1):
                        new_route = temp[ri][:pos] + [cust] + temp[ri][pos:]
                        _, _, feas = check_route(new_route)
                        if feas:
                            cost = route_dist(new_route) - route_dist(temp[ri])
                            if cost < best_cost_i:
                                best_cost_i = cost
                                best_ri = ri
                                best_pos_i = pos
                
                if best_ri >= 0:
                    temp[best_ri] = temp[best_ri][:best_pos_i] + [cust] + temp[best_ri][best_pos_i:]
                else:
                    success = False
                    break
            
            if success:
                temp = [r for r in temp if r]
                cur_num = len(temp)
                cur_d = total_dist(temp)
                if cur_num < best_num or (cur_num == best_num and cur_d < best_d - 1e-9):
                    best_routes = temp
                    best_num = cur_num
                    best_d = cur_d
        
        return best_routes
    
    # LNS for distance improvement (same number of routes)
    def lns_improve(routes, max_iter=300):
        best_routes = [list(r) for r in routes]
        best_num = len(routes)
        best_d = total_dist(routes)
        
        for it in range(max_iter):
            if time_remaining() < 10:
                break
            
            current = [list(r) for r in best_routes]
            
            # Remove random customers
            all_custs = [c for r in current for c in r]
            num_remove = min(max(2, len(all_custs) // 5), len(all_custs))
            removed = random.sample(all_custs, num_remove)
            
            for cust in removed:
                for ri in range(len(current)):
                    if cust in current[ri]:
                        current[ri].remove(cust)
                        break
            
            current = [r for r in current if r]
            
            # Reinsert
            random.shuffle(removed)
            temp = [list(r) for r in current]
            success = True
            
            for cust in removed:
                best_ri = -1
                best_pos_i = -1
                best_cost_i = float('inf')
                
                for ri in range(len(temp)):
                    for pos in range(len(temp[ri]) + 1):
                        new_route = temp[ri][:pos] + [cust] + temp[ri][pos:]
                        _, _, feas = check_route(new_route)
                        if feas:
                            cost = route_dist(new_route) - route_dist(temp[ri])
                            if cost < best_cost_i:
                                best_cost_i = cost
                                best_ri = ri
                                best_pos_i = pos
                
                if best_ri >= 0:
                    temp[best_ri] = temp[best_ri][:best_pos_i] + [cust] + temp[best_ri][best_pos_i:]
                else:
                    success = False
                    break
            
            if success:
                temp = [r for r in temp if r]
                cur_num = len(temp)
                cur_d = total_dist(temp)
                if cur_num < best_num or (cur_num == best_num and cur_d < best_d - 1e-9):
                    best_routes = temp
                    best_num = cur_num
                    best_d = cur_d
        
        return best_routes
    
    # Initialize logger
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    random.seed(42)
    
    best_solution = None
    best_obj = float('inf')
    
    def update_best(routes):
        nonlocal best_solution, best_obj
        sol = make_solution(routes)
        if sol is not None:
            if sol["objective_value"] < best_obj or (sol["objective_value"] == best_obj and sol["total_distance"] < (best_solution["total_distance"] if best_solution else float('inf'))):
                best_solution = sol
                best_obj = sol["objective_value"]
                if logger:
                    logger.log_solution(sol["objective_value"], sol)
    
    # Phase 1: Multiple constructions
    orders = [
        sorted(range(1, num_customers + 1), key=lambda x: (due[x], ready[x])),
        sorted(range(1, num_customers + 1), key=lambda x: (ready[x], due[x])),
        sorted(range(1, num_customers + 1), key=lambda x: (-dem[x], due[x])),
        sorted(range(1, num_customers + 1), key=lambda x: (due[x] - ready[x], due[x])),
    ]
    
    best_routes = None
    best_num_routes = float('inf')
    best_total_d = float('inf')
    
    for order in orders:
        if time_remaining() < 30:
            break
        r = construct_greedy(order)
        d = total_dist(r)
        if len(r) < best_num_routes or (len(r) == best_num_routes and d < best_total_d):
            best_routes = r
            best_num_routes = len(r)
            best_total_d = d
    
    # Try regret construction
    if time_remaining() > 30:
        order_regret = sorted(range(1, num_customers + 1), key=lambda x: (due[x], ready[x]))
        r = construct_regret(order_regret)
        d = total_dist(r)
        if len(r) < best_num_routes or (len(r) == best_num_routes and d < best_total_d):
            best_routes = r
            best_num_routes = len(r)
            best_total_d = d
    
    routes = best_routes
    update_best(routes)
    
    # Phase 2: Route elimination
    routes = eliminate_routes(routes)
    update_best(routes)
    
    # Phase 3: Local search
    routes = ls_relocate(routes)
    routes = ls_2opt(routes)
    routes = ls_oropt(routes)
    routes = ls_exchange(routes)
    routes = ls_cross_exchange(routes)
    update_best(routes)
    
    # Phase 4: More elimination
    routes = eliminate_routes(routes)
    update_best(routes)
    
    # Phase 5: LNS route reduction
    if time_remaining() > 30:
        routes = lns_reduce(routes, max_iter=1000)
        update_best(routes)
    
    # Phase 6: Local search again
    routes = ls_relocate(routes)
    routes = ls_2opt(routes)
    routes = ls_oropt(routes)
    routes = ls_exchange(routes)
    update_best(routes)
    
    # Phase 7: More elimination after LNS
    routes = eliminate_routes(routes)
    update_best(routes)
    
    # Phase 8: LNS distance improvement
    if time_remaining() > 20:
        routes = lns_improve(routes, max_iter=500)
        update_best(routes)
    
    # Phase 9: Final local search
    routes = ls_relocate(routes)
    routes = ls_2opt(routes)
    routes = ls_oropt(routes)
    update_best(routes)
    
    # Phase 10: Final elimination attempt
    routes = eliminate_routes(routes)
    update_best(routes)
    
    # Write final solution
    if best_solution is None:
        best_solution = make_solution(routes)
    
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)

if __name__ == '__main__':
    main()