import json
import argparse
import time
import math
import copy
import random
from solution_logger import SolutionLogger

def solve(instance, time_limit, logger=None):
    start_time = time.time()
    
    num_customers = instance["num_customers"]
    capacity = instance["vehicle_capacity"]
    horizon = instance["scheduling_horizon"]
    a_cost = instance["cost_function"]["a"]
    b_cost = instance["cost_function"]["b"]
    depot = instance["depot"]
    customers = instance["customers"]
    dist_matrix = instance["distance_matrix"]
    
    time_matrix = dist_matrix
    
    depot_id = depot["id"]
    depot_tw = depot["time_window"]
    
    cust_demand = {}
    cust_tw = {}
    cust_st = {}
    for c in customers:
        cust_demand[c["id"]] = c["demand"]
        cust_tw[c["id"]] = c["time_window"]
        cust_st[c["id"]] = c["service_time"]
    
    def edge_cost(i, j, load):
        return dist_matrix[i][j] * (a_cost * load + b_cost)
    
    def route_cost(route, deliveries):
        cost = 0.0
        load = sum(deliveries.get(c, 0) for c in route if c != depot_id)
        prev = depot_id
        for c in route[1:]:
            cost += edge_cost(prev, c, load)
            if c != depot_id:
                load -= deliveries.get(c, 0)
            prev = c
        return cost
    
    def check_time_feasibility(route):
        current_time = depot_tw[0]
        prev = depot_id
        for c in route[1:]:
            arrival = current_time + time_matrix[prev][c]
            if c == depot_id:
                if arrival > depot_tw[1]:
                    return None
            else:
                if arrival > cust_tw[c][1]:
                    return None
                service_start = max(arrival, cust_tw[c][0])
                current_time = service_start + cust_st[c]
            prev = c
        return True
    
    def insertion_cost_increase(route, deliveries, cust, delivery_qty, pos):
        old_cost = route_cost(route, deliveries)
        new_route = route[:pos] + [cust] + route[pos:]
        new_deliveries = dict(deliveries)
        new_deliveries[cust] = new_deliveries.get(cust, 0) + delivery_qty
        new_cost = route_cost(new_route, new_deliveries)
        return new_cost - old_cost
    
    # State
    routes = []  # list of (route, deliveries)
    remaining_demand = dict(cust_demand)
    
    def greedy_construction():
        nonlocal remaining_demand
        remaining_demand = dict(cust_demand)
        routes.clear()
        
        unserved = set(c["id"] for c in customers)
        
        while unserved and time.time() - start_time < time_limit * 0.3:
            route = [depot_id, depot_id]
            deliveries = {}
            current_load = 0
            
            improved = True
            while improved and time.time() - start_time < time_limit * 0.3:
                improved = False
                best_ratio = float('inf')
                best_cust = None
                best_pos = None
                best_qty = None
                
                candidates = set()
                for c_id in remaining_demand:
                    if remaining_demand[c_id] > 0:
                        # Check not already in this route
                        if c_id not in [x for x in route if x != depot_id]:
                            candidates.add(c_id)
                
                for cust in candidates:
                    qty = min(remaining_demand[cust], capacity - current_load)
                    if qty <= 0:
                        continue
                    
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        tf = check_time_feasibility(new_route)
                        if tf is None:
                            continue
                        
                        cost_inc = insertion_cost_increase(route, deliveries, cust, qty, pos)
                        ratio = cost_inc / qty if qty > 0 else float('inf')
                        
                        if ratio < best_ratio:
                            best_ratio = ratio
                            best_cust = cust
                            best_pos = pos
                            best_qty = qty
                
                if best_cust is not None:
                    route = route[:best_pos] + [best_cust] + route[best_pos:]
                    deliveries[best_cust] = deliveries.get(best_cust, 0) + best_qty
                    current_load += best_qty
                    remaining_demand[best_cust] -= best_qty
                    if remaining_demand[best_cust] <= 0:
                        unserved.discard(best_cust)
                    improved = True
            
            if len(route) > 2:
                routes.append((route, deliveries))
            elif unserved:
                best_c = None
                best_d = float('inf')
                for c in unserved:
                    d = dist_matrix[depot_id][c]
                    if d < best_d:
                        best_d = d
                        best_c = c
                if best_c is not None:
                    qty = min(remaining_demand[best_c], capacity)
                    new_route = [depot_id, best_c, depot_id]
                    tf = check_time_feasibility(new_route)
                    if tf is not None:
                        routes.append((new_route, {best_c: qty}))
                        remaining_demand[best_c] -= qty
                        if remaining_demand[best_c] <= 0:
                            unserved.discard(best_c)
                    else:
                        unserved.discard(best_c)
        
        for c in customers:
            cid = c["id"]
            while remaining_demand.get(cid, 0) > 0:
                qty = min(remaining_demand[cid], capacity)
                new_route = [depot_id, cid, depot_id]
                tf = check_time_feasibility(new_route)
                if tf is not None:
                    routes.append((new_route, {cid: qty}))
                    remaining_demand[cid] -= qty
                else:
                    remaining_demand[cid] = 0
                    break
    
    def compute_total_cost():
        total = 0.0
        for route, deliveries in routes:
            total += route_cost(route, deliveries)
        return total
    
    def build_solution_dict():
        sol_routes = []
        total_cost = 0.0
        
        for vid, (route, deliveries) in enumerate(routes):
            sol_route = list(route)
            sol_route[-1] = num_customers + 1
            
            edges = []
            load = sum(deliveries.get(c, 0) for c in route if c != depot_id)
            prev = depot_id
            for idx in range(1, len(route)):
                c = route[idx]
                from_node = prev
                to_node = num_customers + 1 if idx == len(route) - 1 else c
                
                cost = edge_cost(prev, c, load)
                total_cost += cost
                
                edges.append({
                    "from": from_node,
                    "to": to_node,
                    "load_weight": load
                })
                
                if c != depot_id:
                    load -= deliveries.get(c, 0)
                prev = c
            
            del_dict = {}
            for cid, qty in deliveries.items():
                if qty > 0:
                    del_dict[str(cid)] = qty
            
            sol_routes.append({
                "vehicle": vid,
                "route": sol_route,
                "deliveries": del_dict,
                "edges": edges
            })
        
        return {
            "objective_value": total_cost,
            "routes": sol_routes
        }
    
    def two_opt_route(route_idx):
        route, deliveries = routes[route_idx]
        if len(route) <= 4:
            return False
        
        n = len(route)
        best_cost = route_cost(route, deliveries)
        
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                tf = check_time_feasibility(new_route)
                if tf is None:
                    continue
                new_cost = route_cost(new_route, deliveries)
                if new_cost < best_cost - 1e-6:
                    routes[route_idx] = (new_route, deliveries)
                    return True
        return False
    
    def or_opt_route(route_idx):
        route, deliveries = routes[route_idx]
        n = len(route)
        if n <= 4:
            return False
        
        best_cost = route_cost(route, deliveries)
        
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if j == i or j == i - 1 or j == i + 1:
                    continue
                cust = route[i]
                new_route = route[:i] + route[i+1:]
                insert_pos = j if j < i else j - 1
                if insert_pos < 1:
                    insert_pos = 1
                if insert_pos >= len(new_route):
                    insert_pos = len(new_route) - 1
                new_route = new_route[:insert_pos] + [cust] + new_route[insert_pos:]
                
                tf = check_time_feasibility(new_route)
                if tf is None:
                    continue
                new_cost = route_cost(new_route, deliveries)
                if new_cost < best_cost - 1e-6:
                    routes[route_idx] = (new_route, deliveries)
                    return True
        return False
    
    def relocate_between_routes():
        for r1_idx in range(len(routes)):
            route1, del1 = routes[r1_idx]
            for pos1 in range(1, len(route1) - 1):
                cust = route1[pos1]
                qty = del1.get(cust, 0)
                
                for r2_idx in range(len(routes)):
                    if r1_idx == r2_idx:
                        continue
                    route2, del2 = routes[r2_idx]
                    
                    load2 = sum(del2.get(c, 0) for c in route2 if c != depot_id)
                    if load2 + qty > capacity:
                        continue
                    
                    if cust in route2:
                        continue
                    
                    old_cost = route_cost(route1, del1) + route_cost(route2, del2)
                    
                    best_pos2 = None
                    best_new_cost = old_cost
                    
                    for pos2 in range(1, len(route2)):
                        new_route2 = route2[:pos2] + [cust] + route2[pos2:]
                        tf = check_time_feasibility(new_route2)
                        if tf is None:
                            continue
                        new_del2 = dict(del2)
                        new_del2[cust] = qty
                        
                        new_route1 = route1[:pos1] + route1[pos1+1:]
                        new_del1 = dict(del1)
                        if cust in new_del1:
                            del new_del1[cust]
                        
                        tf1 = check_time_feasibility(new_route1)
                        if tf1 is None:
                            continue
                        
                        new_cost = route_cost(new_route1, new_del1) + route_cost(new_route2, new_del2)
                        if new_cost < best_new_cost - 1e-6:
                            best_new_cost = new_cost
                            best_pos2 = pos2
                    
                    if best_pos2 is not None:
                        new_route1 = route1[:pos1] + route1[pos1+1:]
                        new_del1 = dict(del1)
                        if cust in new_del1:
                            del new_del1[cust]
                        
                        new_route2 = route2[:best_pos2] + [cust] + route2[best_pos2:]
                        new_del2 = dict(del2)
                        new_del2[cust] = qty
                        
                        if len(new_route1) <= 2:
                            routes[r1_idx] = None
                            routes[r2_idx] = (new_route2, new_del2)
                            # Remove None entries
                            while None in routes:
                                routes.remove(None)
                        else:
                            routes[r1_idx] = (new_route1, new_del1)
                            routes[r2_idx] = (new_route2, new_del2)
                        return True
        return False
    
    def swap_between_routes():
        for r1_idx in range(len(routes)):
            route1, del1 = routes[r1_idx]
            for r2_idx in range(r1_idx + 1, len(routes)):
                route2, del2 = routes[r2_idx]
                
                for pos1 in range(1, len(route1) - 1):
                    cust1 = route1[pos1]
                    qty1 = del1.get(cust1, 0)
                    
                    for pos2 in range(1, len(route2) - 1):
                        cust2 = route2[pos2]
                        qty2 = del2.get(cust2, 0)
                        
                        if cust1 == cust2:
                            continue
                        
                        if cust2 in route1 or cust1 in route2:
                            continue
                        
                        load1 = sum(del1.get(c, 0) for c in route1 if c != depot_id)
                        load2 = sum(del2.get(c, 0) for c in route2 if c != depot_id)
                        
                        new_load1 = load1 - qty1 + qty2
                        new_load2 = load2 - qty2 + qty1
                        
                        if new_load1 > capacity or new_load2 > capacity:
                            continue
                        
                        new_route1 = list(route1)
                        new_route1[pos1] = cust2
                        new_del1 = dict(del1)
                        del new_del1[cust1]
                        new_del1[cust2] = qty2
                        
                        new_route2 = list(route2)
                        new_route2[pos2] = cust1
                        new_del2 = dict(del2)
                        del new_del2[cust2]
                        new_del2[cust1] = qty1
                        
                        tf1 = check_time_feasibility(new_route1)
                        if tf1 is None:
                            continue
                        tf2 = check_time_feasibility(new_route2)
                        if tf2 is None:
                            continue
                        
                        old_cost = route_cost(route1, del1) + route_cost(route2, del2)
                        new_cost = route_cost(new_route1, new_del1) + route_cost(new_route2, new_del2)
                        
                        if new_cost < old_cost - 1e-6:
                            routes[r1_idx] = (new_route1, new_del1)
                            routes[r2_idx] = (new_route2, new_del2)
                            return True
        return False
    
    def merge_split_deliveries():
        cust_routes = {}
        for ridx, (route, deliveries) in enumerate(routes):
            for c, qty in deliveries.items():
                if qty > 0:
                    if c not in cust_routes:
                        cust_routes[c] = []
                    cust_routes[c].append(ridx)
        
        for cust, ridxs in cust_routes.items():
            if len(ridxs) <= 1:
                continue
            
            for i in range(len(ridxs)):
                for j in range(len(ridxs)):
                    if i == j:
                        continue
                    r_from = ridxs[i]
                    r_to = ridxs[j]
                    
                    if r_from >= len(routes) or r_to >= len(routes):
                        continue
                    
                    route_from, del_from = routes[r_from]
                    route_to, del_to = routes[r_to]
                    
                    qty_from = del_from.get(cust, 0)
                    qty_to = del_to.get(cust, 0)
                    
                    if qty_from <= 0:
                        continue
                    
                    load_to = sum(del_to.get(c, 0) for c in route_to if c != depot_id)
                    can_add = capacity - load_to
                    
                    move_qty = min(qty_from, can_add)
                    if move_qty <= 0:
                        continue
                    
                    new_del_from = dict(del_from)
                    new_del_to = dict(del_to)
                    
                    new_del_from[cust] = qty_from - move_qty
                    new_del_to[cust] = qty_to + move_qty
                    
                    new_route_from = list(route_from)
                    if new_del_from[cust] <= 0:
                        if cust in new_route_from:
                            pos = new_route_from.index(cust)
                            new_route_from = new_route_from[:pos] + new_route_from[pos+1:]
                        if cust in new_del_from:
                            del new_del_from[cust]
                    
                    tf_from = check_time_feasibility(new_route_from)
                    if tf_from is None:
                        continue
                    
                    old_cost = route_cost(route_from, del_from) + route_cost(route_to, del_to)
                    new_cost = route_cost(new_route_from, new_del_from) + route_cost(route_to, new_del_to)
                    
                    if new_cost < old_cost - 1e-6:
                        routes[r_to] = (route_to, new_del_to)
                        if len(new_route_from) <= 2:
                            routes[r_from] = None
                            while None in routes:
                                routes.remove(None)
                        else:
                            routes[r_from] = (new_route_from, new_del_from)
                        return True
        return False
    
    def adjust_delivery_quantities():
        cust_routes = {}
        for ridx, (route, deliveries) in enumerate(routes):
            for c, qty in deliveries.items():
                if qty > 0:
                    if c not in cust_routes:
                        cust_routes[c] = []
                    cust_routes[c].append(ridx)
        
        improved = False
        for cust, ridxs in cust_routes.items():
            if len(ridxs) <= 1:
                continue
            
            for i in range(len(ridxs)):
                for j in range(i + 1, len(ridxs)):
                    r1_idx = ridxs[i]
                    r2_idx = ridxs[j]
                    
                    if r1_idx >= len(routes) or r2_idx >= len(routes):
                        continue
                    
                    route1, del1 = routes[r1_idx]
                    route2, del2 = routes[r2_idx]
                    
                    qty1 = del1.get(cust, 0)
                    qty2 = del2.get(cust, 0)
                    
                    load1 = sum(del1.get(c, 0) for c in route1 if c != depot_id)
                    load2 = sum(del2.get(c, 0) for c in route2 if c != depot_id)
                    
                    for delta in [1, -1]:
                        new_qty1 = qty1 - delta
                        new_qty2 = qty2 + delta
                        if new_qty1 < 0 or new_qty2 < 0:
                            continue
                        new_load1 = load1 - delta
                        new_load2 = load2 + delta
                        if new_load1 > capacity or new_load2 > capacity:
                            continue
                        if new_load1 < 0 or new_load2 < 0:
                            continue
                        
                        new_del1 = dict(del1)
                        new_del2 = dict(del2)
                        new_del1[cust] = new_qty1
                        new_del2[cust] = new_qty2
                        
                        old_cost = route_cost(route1, del1) + route_cost(route2, del2)
                        new_cost = route_cost(route1, new_del1) + route_cost(route2, new_del2)
                        
                        if new_cost < old_cost - 1e-6:
                            routes[r1_idx] = (route1, new_del1)
                            routes[r2_idx] = (route2, new_del2)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        return improved
    
    # Build initial solution
    greedy_construction()
    
    best_cost = compute_total_cost()
    best_routes = copy.deepcopy(routes)
    best_sol = build_solution_dict()
    
    if logger:
        logger.log_solution(best_cost, best_sol)
    
    # Local search
    max_time = time_limit * 0.9
    
    while time.time() - start_time < max_time:
        improved = False
        
        for ridx in range(len(routes)):
            if time.time() - start_time >= max_time:
                break
            if two_opt_route(ridx):
                improved = True
            if time.time() - start_time >= max_time:
                break
            if or_opt_route(ridx):
                improved = True
        
        if time.time() - start_time >= max_time:
            break
        
        if relocate_between_routes():
            improved = True
        
        if time.time() - start_time >= max_time:
            break
        
        if swap_between_routes():
            improved = True
        
        if time.time() - start_time >= max_time:
            break
        
        if merge_split_deliveries():
            improved = True
        
        if time.time() - start_time >= max_time:
            break
        
        if adjust_delivery_quantities():
            improved = True
        
        current_cost = compute_total_cost()
        if current_cost < best_cost - 1e-6:
            best_cost = current_cost
            best_routes = copy.deepcopy(routes)
            best_sol = build_solution_dict()
            if logger:
                logger.log_solution(best_cost, best_sol)
        
        if not improved:
            if len(routes) >= 2 and time.time() - start_time < max_time:
                r_idx = random.randint(0, len(routes) - 1)
                route, dels = routes[r_idx]
                if len(route) > 3:
                    pos = random.randint(1, len(route) - 2)
                    cust = route[pos]
                    qty = dels.get(cust, 0)
                    
                    other_routes = [i for i in range(len(routes)) if i != r_idx]
                    if other_routes:
                        r2_idx = random.choice(other_routes)
                        route2, del2 = routes[r2_idx]
                        load2 = sum(del2.get(c, 0) for c in route2 if c != depot_id)
                        
                        if load2 + qty <= capacity and cust not in route2:
                            done = False
                            for pos2 in range(1, len(route2)):
                                new_route2 = route2[:pos2] + [cust] + route2[pos2:]
                                tf = check_time_feasibility(new_route2)
                                if tf is not None:
                                    new_route1 = route[:pos] + route[pos+1:]
                                    new_del1 = dict(dels)
                                    if cust in new_del1:
                                        del new_del1[cust]
                                    tf1 = check_time_feasibility(new_route1)
                                    if tf1 is not None:
                                        new_del2 = dict(del2)
                                        new_del2[cust] = qty
                                        
                                        if len(new_route1) <= 2:
                                            routes[r_idx] = None
                                            routes[r2_idx] = (new_route2, new_del2)
                                            while None in routes:
                                                routes.remove(None)
                                        else:
                                            routes[r_idx] = (new_route1, new_del1)
                                            routes[r2_idx] = (new_route2, new_del2)
                                        done = True
                                        break
                
                current_cost = compute_total_cost()
                if current_cost < best_cost - 1e-6:
                    best_cost = current_cost
                    best_routes = copy.deepcopy(routes)
                    best_sol = build_solution_dict()
                    if logger:
                        logger.log_solution(best_cost, best_sol)
            else:
                break
    
    # Restore best solution
    routes.clear()
    routes.extend(best_routes)
    
    return build_solution_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    solution = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()