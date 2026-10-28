import sys
import json
import math
import random
import time
import argparse

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

class Solution:
    def __init__(self, d, t, m, n):
        self.routes = [[[ [] for _ in range(m)] for _ in range(t)] for _ in range(d)]
        self.route_demand = [[[ 0 for _ in range(m)] for _ in range(t)] for _ in range(d)]
        self.route_duration = [[[ 0.0 for _ in range(m)] for _ in range(t)] for _ in range(d)]
        self.assignment = [None] * n
        self.unassigned = set(range(n))
        self.cost = 0.0

def copy_solution(sol, d, t, m, n):
    new_sol = Solution(d, t, m, n)
    new_sol.cost = sol.cost
    for D in range(d):
        for k in range(t):
            for v in range(m):
                new_sol.routes[D][k][v] = list(sol.routes[D][k][v])
                new_sol.route_demand[D][k][v] = sol.route_demand[D][k][v]
                new_sol.route_duration[D][k][v] = sol.route_duration[D][k][v]
    new_sol.assignment = list(sol.assignment)
    new_sol.unassigned = set(sol.unassigned)
    return new_sol

def evaluate_insertion_strict(sol, i, D, pattern, m, n, dist, customers, vehicle_capacity, route_duration_limit):
    total_cost = 0
    best_moves = {}
    
    for k in pattern:
        best_day_cost = float('inf')
        best_day_move = None
        
        for v in range(m):
            if sol.route_demand[D][k][v] + customers[i]['demand'] > vehicle_capacity:
                continue
                
            route = sol.routes[D][k][v]
            for idx in range(len(route) + 1):
                prev_node = route[idx-1] if idx > 0 else D + n
                nxt_node = route[idx] if idx < len(route) else D + n
                
                detour = dist[prev_node][i] + dist[i][nxt_node] - dist[prev_node][nxt_node]
                added_dur = detour + customers[i]['service_duration']
                
                if sol.route_duration[D][k][v] + added_dur <= route_duration_limit + 1e-5:
                    if detour < best_day_cost:
                        best_day_cost = detour
                        best_day_move = (v, idx, customers[i]['demand'], added_dur, detour)
                        
        if best_day_cost == float('inf'):
            return float('inf'), None
            
        total_cost += best_day_cost
        best_moves[k] = best_day_move
        
    return total_cost, best_moves

def apply_insertion(sol, i, move, customers):
    D, p_idx, moves = move
    pattern = customers[i]['pattern_list'][p_idx]
    
    for k in pattern:
        v, idx, added_dem, added_dur, detour = moves[k]
        sol.routes[D][k][v].insert(idx, i)
        sol.route_demand[D][k][v] += added_dem
        sol.route_duration[D][k][v] += added_dur
        sol.cost += detour
        
    sol.assignment[i] = (D, p_idx)
    if i in sol.unassigned:
        sol.unassigned.remove(i)

def build_solution_regret(seed, d, t, m, n, dist, customers, vehicle_capacity, route_duration_limit):
    random.seed(seed)
    sol = Solution(d, t, m, n)
    unassigned = list(range(n))
    
    while unassigned:
        best_regret = -1
        best_customer = None
        best_customer_move = None
        
        for i in unassigned:
            valid_moves = []
            for D in range(d):
                for p_idx, pattern in enumerate(customers[i]['pattern_list']):
                    cost, moves = evaluate_insertion_strict(sol, i, D, pattern, m, n, dist, customers, vehicle_capacity, route_duration_limit)
                    if cost != float('inf'):
                        valid_moves.append((cost, D, p_idx, moves))
            
            if not valid_moves:
                return None 
            
            valid_moves.sort(key=lambda x: x[0])
            cost1 = valid_moves[0][0]
            if len(valid_moves) > 1:
                cost2 = valid_moves[1][0]
            else:
                cost2 = cost1
            
            regret = cost2 - cost1
            regret += random.random() * 1e-4
            
            if regret > best_regret:
                best_regret = regret
                best_customer = i
                best_customer_move = (valid_moves[0][1], valid_moves[0][2], valid_moves[0][3])
                
        apply_insertion(sol, best_customer, best_customer_move, customers)
        unassigned.remove(best_customer)
        
    return sol

def remove_customer(sol, cust, n, d, dist, customers, m):
    if sol.assignment[cust] is None:
        return
    D, p_idx = sol.assignment[cust]
    pattern = customers[cust]['pattern_list'][p_idx]
    
    for k in pattern:
        for v in range(m):
            if cust in sol.routes[D][k][v]:
                idx = sol.routes[D][k][v].index(cust)
                prev_node = n + D if idx == 0 else sol.routes[D][k][v][idx-1]
                nxt_node = n + D if idx == len(sol.routes[D][k][v])-1 else sol.routes[D][k][v][idx+1]
                
                rem_dist = dist[prev_node][nxt_node] - (dist[prev_node][cust] + dist[cust][nxt_node])
                
                sol.routes[D][k][v].pop(idx)
                sol.route_demand[D][k][v] -= customers[cust]['demand']
                sol.route_duration[D][k][v] += rem_dist - customers[cust]['service_duration']
                sol.cost += rem_dist
                break
                
    sol.assignment[cust] = None
    sol.unassigned.add(cust)

def repair_regret(sol, d, t, m, n, dist, customers, vehicle_capacity, route_duration_limit):
    unassigned = list(sol.unassigned)
    
    while unassigned:
        best_regret = -1
        best_customer = None
        best_customer_move = None
        
        for i in unassigned:
            valid_moves = []
            for D in range(d):
                for p_idx, pattern in enumerate(customers[i]['pattern_list']):
                    cost, moves = evaluate_insertion_strict(sol, i, D, pattern, m, n, dist, customers, vehicle_capacity, route_duration_limit)
                    if cost != float('inf'):
                        valid_moves.append((cost, D, p_idx, moves))
            
            if not valid_moves:
                return False 
            
            valid_moves.sort(key=lambda x: x[0])
            cost1 = valid_moves[0][0]
            if len(valid_moves) > 1:
                cost2 = valid_moves[1][0]
            else:
                cost2 = cost1
            
            regret = cost2 - cost1
            regret += random.random() * 1e-4
            
            if regret > best_regret:
                best_regret = regret
                best_customer = i
                best_customer_move = (valid_moves[0][1], valid_moves[0][2], valid_moves[0][3])
                
        apply_insertion(sol, best_customer, best_customer_move, customers)
        unassigned.remove(best_customer)
    return True

def apply_2opt(sol, d, t, m, n, dist):
    improved = False
    for D in range(d):
        for k in range(t):
            for v in range(m):
                route = sol.routes[D][k][v]
                if len(route) < 2:
                    continue
                depot_node = n + D
                
                local_improved = True
                while local_improved:
                    local_improved = False
                    for i in range(len(route) - 1):
                        for j in range(i + 1, len(route)):
                            prev = depot_node if i == 0 else route[i-1]
                            nxt = depot_node if j == len(route)-1 else route[j+1]
                            
                            old_cost = dist[prev][route[i]] + dist[route[j]][nxt]
                            new_cost = dist[prev][route[j]] + dist[route[i]][nxt]
                            
                            if new_cost < old_cost - 1e-5:
                                route[i:j+1] = reversed(route[i:j+1])
                                delta = new_cost - old_cost
                                sol.route_duration[D][k][v] += delta
                                sol.cost += delta
                                local_improved = True
                                improved = True
    return improved

def apply_intra_relocate(sol, d, t, m, n, dist):
    improved = False
    for D in range(d):
        for k in range(t):
            for v in range(m):
                route = sol.routes[D][k][v]
                if len(route) < 3:
                    continue
                depot_node = n + D
                
                local_improved = True
                while local_improved:
                    local_improved = False
                    for i in range(len(route)):
                        for j in range(len(route)):
                            if i == j: continue
                            
                            cust = route[i]
                            temp_route = route[:]
                            temp_route.pop(i)
                            temp_route.insert(j, cust)
                            
                            old_cost_eval = 0.0
                            curr = depot_node
                            for node in route:
                                old_cost_eval += dist[curr][node]
                                curr = node
                            old_cost_eval += dist[curr][depot_node]
                            
                            new_cost_eval = 0.0
                            curr = depot_node
                            for node in temp_route:
                                new_cost_eval += dist[curr][node]
                                curr = node
                            new_cost_eval += dist[curr][depot_node]
                            
                            delta = new_cost_eval - old_cost_eval
                            if delta < -1e-5:
                                route[:] = temp_route
                                sol.route_duration[D][k][v] += delta
                                sol.cost += delta
                                local_improved = True
                                improved = True
                                break
                        if local_improved:
                            break
    return improved

def inter_route_relocate(sol, d, t, m, n, dist, customers, vehicle_capacity, route_duration_limit):
    improved = False
    for D in range(d):
        for k in range(t):
            local_improved = True
            while local_improved:
                local_improved = False
                for v1 in range(m):
                    route1 = sol.routes[D][k][v1]
                    if not route1: continue
                    for i_idx, cust in enumerate(route1):
                        for v2 in range(m):
                            if v1 == v2: continue
                            route2 = sol.routes[D][k][v2]
                            
                            prev1 = n + D if i_idx == 0 else route1[i_idx-1]
                            nxt1 = n + D if i_idx == len(route1)-1 else route1[i_idx+1]
                            rem_dist = dist[prev1][nxt1] - (dist[prev1][cust] + dist[cust][nxt1])
                            
                            for j_idx in range(len(route2) + 1):
                                prev2 = n + D if j_idx == 0 else route2[j_idx-1]
                                nxt2 = n + D if j_idx == len(route2) else route2[j_idx]
                                add_dist = dist[prev2][cust] + dist[cust][nxt2] - dist[prev2][nxt2]
                                
                                delta = rem_dist + add_dist
                                if delta < -1e-5:
                                    if sol.route_demand[D][k][v2] + customers[cust]['demand'] <= vehicle_capacity:
                                        if sol.route_duration[D][k][v2] + add_dist + customers[cust]['service_duration'] <= route_duration_limit + 1e-5:
                                            route2.insert(j_idx, cust)
                                            route1.pop(i_idx)
                                            
                                            sol.route_demand[D][k][v1] -= customers[cust]['demand']
                                            sol.route_demand[D][k][v2] += customers[cust]['demand']
                                            
                                            sol.route_duration[D][k][v1] += rem_dist - customers[cust]['service_duration']
                                            sol.route_duration[D][k][v2] += add_dist + customers[cust]['service_duration']
                                            
                                            sol.cost += delta
                                            local_improved = True
                                            improved = True
                                            break
                        if local_improved: break
                    if local_improved: break
    return improved

def inter_route_swap(sol, d, t, m, n, dist, customers, vehicle_capacity, route_duration_limit):
    improved = False
    for D in range(d):
        for k in range(t):
            local_improved = True
            while local_improved:
                local_improved = False
                for v1 in range(m):
                    route1 = sol.routes[D][k][v1]
                    for i_idx, c1 in enumerate(route1):
                        for v2 in range(v1 + 1, m):
                            route2 = sol.routes[D][k][v2]
                            for j_idx, c2 in enumerate(route2):
                                dem1 = customers[c1]['demand']
                                dem2 = customers[c2]['demand']
                                if sol.route_demand[D][k][v1] - dem1 + dem2 > vehicle_capacity: continue
                                if sol.route_demand[D][k][v2] - dem2 + dem1 > vehicle_capacity: continue
                                
                                p1 = n + D if i_idx == 0 else route1[i_idx-1]
                                n1 = n + D if i_idx == len(route1)-1 else route1[i_idx+1]
                                rem1 = dist[p1][n1] - (dist[p1][c1] + dist[c1][n1])
                                add1 = dist[p1][c2] + dist[c2][n1] - dist[p1][n1]
                                
                                p2 = n + D if j_idx == 0 else route2[j_idx-1]
                                n2 = n + D if j_idx == len(route2)-1 else route2[j_idx+1]
                                rem2 = dist[p2][n2] - (dist[p2][c2] + dist[c2][n2])
                                add2 = dist[p2][c1] + dist[c1][n2] - dist[p2][n2]
                                
                                delta1 = rem1 + add1
                                delta2 = rem2 + add2
                                delta = delta1 + delta2
                                
                                if delta < -1e-5:
                                    dur1 = sol.route_duration[D][k][v1] + delta1 - customers[c1]['service_duration'] + customers[c2]['service_duration']
                                    dur2 = sol.route_duration[D][k][v2] + delta2 - customers[c2]['service_duration'] + customers[c1]['service_duration']
                                    if dur1 <= route_duration_limit + 1e-5 and dur2 <= route_duration_limit + 1e-5:
                                        route1[i_idx] = c2
                                        route2[j_idx] = c1
                                        
                                        sol.route_demand[D][k][v1] += dem2 - dem1
                                        sol.route_demand[D][k][v2] += dem1 - dem2
                                        
                                        sol.route_duration[D][k][v1] = dur1
                                        sol.route_duration[D][k][v2] = dur2
                                        
                                        sol.cost += delta
                                        local_improved = True
                                        improved = True
                                        break
                            if local_improved: break
                        if local_improved: break
                    if local_improved: break
    return improved

def recalculate_solution(sol, d, t, m, n, dist, customers):
    sol.cost = 0.0
    for D in range(d):
        for k in range(t):
            for v in range(m):
                route = sol.routes[D][k][v]
                if not route:
                    sol.route_demand[D][k][v] = 0
                    sol.route_duration[D][k][v] = 0.0
                    continue
                
                depot_node = n + D
                curr = depot_node
                dem = 0
                dur = 0.0
                
                for node in route:
                    sol.cost += dist[curr][node]
                    dur += dist[curr][node] + customers[node]['service_duration']
                    dem += customers[node]['demand']
                    curr = node
                
                sol.cost += dist[curr][depot_node]
                dur += dist[curr][depot_node]
                
                sol.route_demand[D][k][v] = dem
                sol.route_duration[D][k][v] = dur

def build_output(sol, d, t, m, n, customers):
    out = {
        "objective_value": sol.cost,
        "routes": [],
        "assignments": {}
    }
    
    for D in range(d):
        for k in range(t):
            for v in range(m):
                if sol.routes[D][k][v]:
                    out["routes"].append({
                        "depot": D,
                        "period": k + 1,
                        "vehicle": v + 1,
                        "customers": list(sol.routes[D][k][v])
                    })
                    
    for i in range(n):
        if sol.assignment[i] is not None:
            D, p_idx = sol.assignment[i]
            pat = [day + 1 for day in customers[i]['pattern_list'][p_idx]]
            out["assignments"][str(i)] = {
                "depot": D,
                "pattern": pat
            }
        
    return out

def lns_optimize(seed, time_limit, start_time, logger, d, t, m, n, dist, customers, vehicle_capacity, route_duration_limit):
    random.seed(seed)
    
    best_sol = None
    best_cost = float('inf')
    
    for attempt in range(100):
        if time.time() - start_time > time_limit * 0.1:
            break
        sol = build_solution_regret(seed + attempt, d, t, m, n, dist, customers, vehicle_capacity, route_duration_limit)
        if sol is not None:
            best_sol = copy_solution(sol, d, t, m, n)
            best_cost = best_sol.cost
            if logger:
                logger.log_solution(best_cost, build_output(best_sol, d, t, m, n, customers))
            break
            
    if best_sol is None:
        return None
        
    curr_sol = copy_solution(best_sol, d, t, m, n)
    
    T = max(best_cost * 0.05, 1.0)
    T_min = 0.1
    est_iters = max(10, int((time_limit - 1.5) * 20))
    ratio = T_min / max(T, T_min + 1e-3)
    alpha = math.exp(math.log(ratio) / est_iters)
    
    while time.time() - start_time < time_limit - 1.5:
        temp_sol = copy_solution(curr_sol, d, t, m, n)
        
        q = random.randint(1, max(2, min(n // 4, 15)))
        destroy_method = random.choice(['random', 'worst', 'depot', 'pattern', 'route'])
        
        to_remove = set()
        
        if destroy_method == 'random':
            to_remove.update(random.sample(list(range(n)), q))
        elif destroy_method == 'worst':
            costs = []
            for i in range(n):
                if temp_sol.assignment[i] is None: continue
                cost_i = 0
                D, p_idx = temp_sol.assignment[i]
                for k in customers[i]['pattern_list'][p_idx]:
                    for v in range(m):
                        if i in temp_sol.routes[D][k][v]:
                            idx = temp_sol.routes[D][k][v].index(i)
                            prv = n + D if idx == 0 else temp_sol.routes[D][k][v][idx-1]
                            nxt = n + D if idx == len(temp_sol.routes[D][k][v])-1 else temp_sol.routes[D][k][v][idx+1]
                            cost_i += dist[prv][i] + dist[i][nxt] - dist[prv][nxt]
                costs.append((cost_i, i))
            costs.sort(reverse=True)
            pool = [i for _, i in costs[:2*q]]
            to_remove.update(random.sample(pool, min(q, len(pool))))
        elif destroy_method == 'depot':
            dep = random.randint(0, d-1)
            candidates = [i for i in range(n) if temp_sol.assignment[i] is not None and temp_sol.assignment[i][0] == dep]
            if candidates:
                to_remove.update(random.sample(candidates, min(q*2, len(candidates))))
            else:
                to_remove.update(random.sample(list(range(n)), q))
        elif destroy_method == 'pattern':
            day = random.randint(0, t-1)
            candidates = []
            for i in range(n):
                if temp_sol.assignment[i] is None: continue
                _, p_idx = temp_sol.assignment[i]
                if day in customers[i]['pattern_list'][p_idx]:
                    candidates.append(i)
            if candidates:
                to_remove.update(random.sample(candidates, min(q*2, len(candidates))))
            else:
                to_remove.update(random.sample(list(range(n)), q))
        elif destroy_method == 'route':
            non_empty_routes = []
            for D in range(d):
                for k in range(t):
                    for v in range(m):
                        if temp_sol.routes[D][k][v]:
                            non_empty_routes.append((D, k, v))
            if non_empty_routes:
                sel_D, sel_k, sel_v = random.choice(non_empty_routes)
                for cust in temp_sol.routes[sel_D][sel_k][sel_v]:
                    to_remove.add(cust)
            if len(to_remove) < q:
                extra = set(random.sample(list(range(n)), q))
                to_remove.update(extra)
        
        for i in to_remove:
            if temp_sol.assignment[i] is not None:
                remove_customer(temp_sol, i, n, d, dist, customers, m)
            
        success = repair_regret(temp_sol, d, t, m, n, dist, customers, vehicle_capacity, route_duration_limit)
        
        if success:
            apply_2opt(temp_sol, d, t, m, n, dist)
            apply_intra_relocate(temp_sol, d, t, m, n, dist)
            inter_route_relocate(temp_sol, d, t, m, n, dist, customers, vehicle_capacity, route_duration_limit)
            inter_route_swap(temp_sol, d, t, m, n, dist, customers, vehicle_capacity, route_duration_limit)
            recalculate_solution(temp_sol, d, t, m, n, dist, customers)
            
            delta = temp_sol.cost - curr_sol.cost
            
            if delta < -1e-5 or random.random() < math.exp(-delta / T):
                curr_sol = temp_sol
                if curr_sol.cost < best_cost - 1e-5:
                    best_cost = curr_sol.cost
                    best_sol = copy_solution(curr_sol, d, t, m, n)
                    if logger:
                        logger.log_solution(best_cost, build_output(best_sol, d, t, m, n, customers))
            
        T = max(T_min, T * alpha)
        
    return best_sol

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    d = instance['d']
    t = instance['t']
    m = instance['m']
    vehicle_capacity = instance['vehicle_capacity']
    route_duration_limit = instance['route_duration_limit']
    
    customers = instance['customers']
    depots = instance['depots']
    
    customers.sort(key=lambda c: c['id'])
    depots.sort(key=lambda dp: dp['id'])
    
    for cust in customers:
        cust['pattern_list'] = [[p - 1 for p in pat] for pat in cust['pattern_list']]

    nodes = customers + depots
    dist = [[0.0] * len(nodes) for _ in range(len(nodes))]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            dx = nodes[i]['x'] - nodes[j]['x']
            dy = nodes[i]['y'] - nodes[j]['y']
            dist[i][j] = math.hypot(dx, dy)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    best_sol = lns_optimize(42, args.time_limit, start_time, logger, d, t, m, n, dist, customers, vehicle_capacity, route_duration_limit)

    if best_sol is None:
        out = {
            "objective_value": 0.0,
            "routes": [],
            "assignments": {}
        }
        for i in range(n):
            out["assignments"][str(i)] = {
                "depot": 0,
                "pattern": [day + 1 for day in customers[i]['pattern_list'][0]]
            }
    else:
        out = build_output(best_sol, d, t, m, n, customers)

    with open(args.solution_path, 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()