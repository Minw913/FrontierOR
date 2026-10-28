import json
import argparse
import time
import random
import math
from copy import deepcopy
from itertools import product

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()


class Instance:
    def __init__(self, data):
        self.n = data['parameters']['n_customers']
        self.H = data['vehicle']['H']
        self.W = data['vehicle']['W']
        self.Q = data['vehicle']['Q']
        self.K = data['vehicle']['K']
        self.depot_id = data['depot']['id']
        self.recourse_cost = data['parameters']['recourse_cost_cf']
        self.dist = data['distance_matrix']
        self.customers = data['customers']
        
        self.customer_ids = [c['id'] for c in self.customers]
        self.customer_map = {c['id']: c for c in self.customers}
        
        self.expected_area = {}
        self.expected_weight = {}
        self.det_items = {}
        self.stoch_items = {}
        self.worst_case_height = {}
        
        for c in self.customers:
            cid = c['id']
            total_area = 0.0
            total_weight = 0.0
            det = []
            stoch = []
            
            for item in c['items']:
                if item['is_stochastic']:
                    stoch.append(item)
                    for r in item['realizations']:
                        total_area += r['probability'] * r['height'] * r['width']
                        total_weight += r['probability'] * r['weight']
                else:
                    r = item['realizations'][0]
                    det.append((r['height'], r['width'], r['weight']))
                    total_area += r['height'] * r['width']
                    total_weight += r['weight']
            
            self.expected_area[cid] = total_area
            self.expected_weight[cid] = total_weight
            self.det_items[cid] = det
            self.stoch_items[cid] = stoch
        
        # Precompute worst-case packing height for each customer
        for cid in self.customer_ids:
            self.worst_case_height[cid] = self._compute_worst_case_height(cid)
    
    def _compute_worst_case_height(self, cid):
        """Compute worst-case packing height for a customer."""
        det = [(h, w) for h, w, _ in self.det_items[cid]]
        stoch = self.stoch_items[cid]
        
        if not stoch:
            return pack_items_ffdh(det, self.W)
        
        # For stochastic items, use worst-case dimensions
        max_h_items = list(det)
        for item in stoch:
            worst_h = max(r['height'] for r in item['realizations'])
            worst_w = max(r['width'] for r in item['realizations'])
            max_h_items.append((worst_h, worst_w))
        
        return pack_items_ffdh(max_h_items, self.W)


def pack_items_ffdh(items, W):
    """Pack items (h, w) in strip of width W using First-Fit Decreasing Height."""
    if not items:
        return 0
    items_sorted = sorted(items, key=lambda x: (-x[0], -x[1]))
    shelves = []  # [width_used, shelf_height]
    
    for h, w in items_sorted:
        if w > W:
            return float('inf')
        placed = False
        for i in range(len(shelves)):
            if shelves[i][0] + w <= W and h <= shelves[i][1]:
                shelves[i][0] += w
                placed = True
                break
        if not placed:
            shelves.append([w, h])
    
    return sum(s[1] for s in shelves)


def check_route_packing(inst, route):
    """Check if a route satisfies the packing constraint (worst-case)."""
    total_height = 0
    for cid in route:
        total_height += inst.worst_case_height[cid]
        if total_height > inst.H:
            return False
    return True


def check_route_weight(inst, route):
    """Check weight capacity using expected weights."""
    total = sum(inst.expected_weight[cid] for cid in route)
    return total <= inst.Q + 1e-9


def check_route_area(inst, route):
    """Check area capacity using expected areas."""
    total = sum(inst.expected_area[cid] for cid in route)
    return total <= inst.H * inst.W + 1e-9


def route_travel_cost(inst, route):
    """Compute travel cost for a route."""
    if not route:
        return 0
    cost = inst.dist[inst.depot_id][route[0]]
    for i in range(len(route) - 1):
        cost += inst.dist[route[i]][route[i + 1]]
    cost += inst.dist[route[-1]][inst.depot_id]
    return cost


def route_is_feasible(inst, route):
    """Check all feasibility constraints for a route."""
    if not route:
        return True
    if not check_route_weight(inst, route):
        return False
    if not check_route_area(inst, route):
        return False
    if not check_route_packing(inst, route):
        return False
    return True


def compute_recourse_cost(inst, route):
    """Compute expected recourse cost for a route."""
    if not route:
        return 0.0
    
    stoch_items_on_route = []
    for cid in route:
        for item in inst.stoch_items[cid]:
            stoch_items_on_route.append((cid, item))
    
    if not stoch_items_on_route:
        if check_route_packing(inst, route):
            return 0.0
        else:
            return float(inst.recourse_cost) * len(route)
    
    num_scenarios = 1
    for _, item in stoch_items_on_route:
        num_scenarios *= item['num_realizations']
    
    if num_scenarios > 3000:
        return estimate_recourse_sampling(inst, route, stoch_items_on_route, 200)
    
    total_expected_unserved = 0.0
    realizations_list = [range(item['num_realizations']) for _, item in stoch_items_on_route]
    
    # Map each stochastic item to its index in stoch_items_on_route
    item_to_idx = {}
    for idx, (cid, item) in enumerate(stoch_items_on_route):
        item_to_idx[id(item)] = idx
    
    for combo in product(*realizations_list):
        prob = 1.0
        for idx, (cid, item) in enumerate(stoch_items_on_route):
            r = item['realizations'][combo[idx]]
            prob *= r['probability']
        
        # Build items for this scenario
        total_height = 0
        served = 0
        for cid in route:
            items = [(h, w) for h, w, _ in inst.det_items[cid]]
            for item in inst.stoch_items[cid]:
                si = item_to_idx[id(item)]
                r = item['realizations'][combo[si]]
                items.append((r['height'], r['width']))
            
            h = pack_items_ffdh(items, inst.W)
            if total_height + h <= inst.H:
                total_height += h
                served += 1
            else:
                break
        
        unserved = len(route) - served
        total_expected_unserved += prob * unserved
    
    return inst.recourse_cost * total_expected_unserved


def estimate_recourse_sampling(inst, route, stoch_items_on_route, num_samples):
    """Estimate recourse cost by sampling."""
    total_unserved = 0.0
    
    for _ in range(num_samples):
        total_height = 0
        served = 0
        for cid in route:
            items = [(h, w) for h, w, _ in inst.det_items[cid]]
            for item in inst.stoch_items[cid]:
                probs = [r['probability'] for r in item['realizations']]
                idx = random.choices(range(len(probs)), weights=probs, k=1)[0]
                r = item['realizations'][idx]
                items.append((r['height'], r['width']))
            
            h = pack_items_ffdh(items, inst.W)
            if total_height + h <= inst.H:
                total_height += h
                served += 1
            else:
                break
        
        total_unserved += (len(route) - served)
    
    return inst.recourse_cost * total_unserved / num_samples


def compute_objective(inst, routes):
    """Compute total objective value."""
    travel = sum(route_travel_cost(inst, r) for r in routes)
    recourse = sum(compute_recourse_cost(inst, r) for r in routes)
    return travel + recourse


def build_savings_solution(inst):
    """Build initial solution using Clarke-Wright savings algorithm."""
    n = len(inst.customer_ids)
    K = inst.K
    
    # Start with each customer on their own route
    routes = [[cid] for cid in inst.customer_ids]
    route_of = {cid: i for i, cid in enumerate(inst.customer_ids)}
    
    # Compute savings
    savings = []
    for i in range(n):
        for j in range(i + 1, n):
            ci = inst.customer_ids[i]
            cj = inst.customer_ids[j]
            s = (inst.dist[inst.depot_id][ci] + inst.dist[inst.depot_id][cj]
                 - inst.dist[ci][cj])
            savings.append((s, ci, cj))
    
    savings.sort(key=lambda x: -x[0])
    
    for s, ci, cj in savings:
        if len([r for r in routes if r]) <= K:
            break
        
        ri = route_of[ci]
        rj = route_of[cj]
        
        if ri == rj or not routes[ri] or not routes[rj]:
            continue
        
        route_i = routes[ri]
        route_j = routes[rj]
        
        # Try to merge
        merged = None
        if route_i[-1] == ci and route_j[0] == cj:
            merged = route_i + route_j
        elif route_j[-1] == cj and route_i[0] == ci:
            merged = route_j + route_i
        elif route_i[-1] == ci and route_j[-1] == cj:
            merged = route_i + route_j[::-1]
        elif route_i[0] == ci and route_j[0] == cj:
            merged = route_i[::-1] + route_j
        else:
            continue
        
        if not route_is_feasible(inst, merged):
            continue
        
        routes[ri] = merged
        routes[rj] = []
        for cid in merged:
            route_of[cid] = ri
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    # Ensure exactly K routes
    while len(routes) < K:
        # Split the longest route
        longest_idx = max(range(len(routes)), key=lambda i: len(routes[i]))
        if len(routes[longest_idx]) < 2:
            routes.append([])
            continue
        route = routes[longest_idx]
        mid = len(route) // 2
        r1 = route[:mid]
        r2 = route[mid:]
        routes[longest_idx] = r1
        routes.append(r2)
    
    while len(routes) > K:
        # Merge two shortest routes
        routes.sort(key=len)
        if route_is_feasible(inst, routes[0] + routes[1]):
            merged = routes[0] + routes[1]
            routes = [merged] + routes[2:]
        else:
            # Try other pair
            found = False
            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    if route_is_feasible(inst, routes[i] + routes[j]):
                        merged = routes[i] + routes[j]
                        routes = [r for k, r in enumerate(routes) if k != i and k != j]
                        routes.append(merged)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
    
    # Pad with empty routes if needed
    while len(routes) < K:
        routes.append([])
    
    return routes[:K]


def build_nn_solution(inst):
    """Build solution using nearest neighbor heuristic."""
    unvisited = set(inst.customer_ids)
    routes = []
    
    for k in range(inst.K):
        if not unvisited:
            routes.append([])
            continue
        
        route = []
        current = inst.depot_id
        current_weight = 0.0
        current_height = 0
        
        while unvisited:
            best_cid = None
            best_dist = float('inf')
            
            for cid in unvisited:
                d = inst.dist[current][cid]
                if d < best_dist:
                    new_weight = current_weight + inst.expected_weight[cid]
                    new_height = current_height + inst.worst_case_height[cid]
                    if new_weight <= inst.Q + 1e-9 and new_height <= inst.H:
                        best_cid = cid
                        best_dist = d
            
            if best_cid is None:
                break
            
            route.append(best_cid)
            unvisited.remove(best_cid)
            current = best_cid
            current_weight += inst.expected_weight[best_cid]
            current_height += inst.worst_case_height[best_cid]
        
        routes.append(route)
    
    # Assign remaining customers
    for cid in list(unvisited):
        best_route_idx = -1
        best_cost = float('inf')
        for i in range(len(routes)):
            test_route = routes[i] + [cid]
            if route_is_feasible(inst, test_route):
                cost = route_travel_cost(inst, test_route) - route_travel_cost(inst, routes[i])
                if cost < best_cost:
                    best_cost = cost
                    best_route_idx = i
        
        if best_route_idx == -1:
            # Force insert into least-cost route
            for i in range(len(routes)):
                cost = route_travel_cost(inst, routes[i] + [cid]) - route_travel_cost(inst, routes[i])
                if cost < best_cost:
                    best_cost = cost
                    best_route_idx = i
        
        if best_route_idx >= 0:
            routes[best_route_idx].append(cid)
        else:
            routes[0].append(cid)
        unvisited.discard(cid)
    
    return routes


def two_opt_route(inst, route):
    """Apply 2-opt improvement to a single route."""
    if len(route) < 3:
        return route
    
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                if route_travel_cost(inst, new_route) < route_travel_cost(inst, route) - 1e-10:
                    route = new_route
                    improved = True
                    break
            if improved:
                break
    return route


def or_opt_route(inst, route):
    """Apply or-opt improvement to a single route."""
    if len(route) < 4:
        return route
    
    improved = True
    while improved:
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(len(route) - seg_len + 1):
                segment = route[i:i + seg_len]
                remaining = route[:i] + route[i + seg_len:]
                
                for j in range(len(remaining) + 1):
                    new_route = remaining[:j] + segment + remaining[j:]
                    if route_travel_cost(inst, new_route) < route_travel_cost(inst, route) - 1e-10:
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return route


def local_search_intra(inst, routes, time_limit):
    """Intra-route improvements."""
    start_time = time.time()
    
    for i in range(len(routes)):
        if time.time() - start_time > time_limit:
            break
        routes[i] = two_opt_route(inst, routes[i])
        if time.time() - start_time > time_limit:
            break
        routes[i] = or_opt_route(inst, routes[i])
    
    return routes


def local_search_inter(inst, routes, time_limit):
    """Inter-route improvements (relocate and swap)."""
    start_time = time.time()
    improved = True
    
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Relocate
        for i in range(len(routes)):
            if time.time() - start_time > time_limit:
                return routes
            for pos_i in range(len(routes[i])):
                cid = routes[i][pos_i]
                new_route_i = routes[i][:pos_i] + routes[i][pos_i + 1:]
                
                for j in range(len(routes)):
                    if i == j:
                        continue
                    for pos_j in range(len(routes[j]) + 1):
                        new_route_j = routes[j][:pos_j] + [cid] + routes[j][pos_j:]
                        
                        if not route_is_feasible(inst, new_route_j):
                            continue
                        
                        old_cost = route_travel_cost(inst, routes[i]) + route_travel_cost(inst, routes[j])
                        new_cost = route_travel_cost(inst, new_route_i) + route_travel_cost(inst, new_route_j)
                        
                        if new_cost < old_cost - 1e-10:
                            routes[i] = new_route_i
                            routes[j] = new_route_j
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if time.time() - start_time > time_limit:
            break
        
        # Swap
        if not improved:
            for i in range(len(routes)):
                if time.time() - start_time > time_limit:
                    return routes
                for j in range(i + 1, len(routes)):
                    for pi in range(len(routes[i])):
                        for pj in range(len(routes[j])):
                            ci = routes[i][pi]
                            cj = routes[j][pj]
                            
                            new_route_i = routes[i][:pi] + [cj] + routes[i][pi + 1:]
                            new_route_j = routes[j][:pj] + [ci] + routes[j][pj + 1:]
                            
                            if not route_is_feasible(inst, new_route_i):
                                continue
                            if not route_is_feasible(inst, new_route_j):
                                continue
                            
                            old_cost = route_travel_cost(inst, routes[i]) + route_travel_cost(inst, routes[j])
                            new_cost = route_travel_cost(inst, new_route_i) + route_travel_cost(inst, new_route_j)
                            
                            if new_cost < old_cost - 1e-10:
                                routes[i] = new_route_i
                                routes[j] = new_route_j
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return routes


def perturb_solution(inst, routes, rng):
    """Perturb solution for diversification."""
    routes = deepcopy(routes)
    
    # Random relocate
    non_empty = [i for i in range(len(routes)) if len(routes[i]) > 1]
    if len(non_empty) >= 2:
        i = rng.choice(non_empty)
        pos = rng.randint(0, len(routes[i]) - 1)
        cid = routes[i][pos]
        new_route_i = routes[i][:pos] + routes[i][pos + 1:]
        
        j = rng.choice([x for x in range(len(routes)) if x != i])
        pos_j = rng.randint(0, len(routes[j]))
        new_route_j = routes[j][:pos_j] + [cid] + routes[j][pos_j:]
        
        if route_is_feasible(inst, new_route_j):
            routes[i] = new_route_i
            routes[j] = new_route_j
    
    return routes


def main():
    args = parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    random.seed(42)
    rng = random.Random(42)
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    inst = Instance(data)
    
    # Build initial solution - try savings first, fall back to NN
    routes = build_savings_solution(inst)
    
    # Verify all customers are assigned
    assigned = set()
    for r in routes:
        assigned.update(r)
    missing = set(inst.customer_ids) - assigned
    for cid in missing:
        best_idx = 0
        best_cost = float('inf')
        for i in range(len(routes)):
            test = routes[i] + [cid]
            c = route_travel_cost(inst, test)
            if c < best_cost:
                best_cost = c
                best_idx = i
        routes[best_idx].append(cid)
    
    # Ensure we have exactly K routes
    while len(routes) < inst.K:
        routes.append([])
    routes = routes[:inst.K]
    
    # Initial intra-route optimization
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 2:
        routes = local_search_intra(inst, routes, remaining * 0.3)
    
    # Compute initial objective
    obj = compute_objective(inst, routes)
    best_obj = obj
    best_routes = deepcopy(routes)
    
    solution = {"objective_value": obj, "routes": routes}
    if logger:
        logger.log_solution(obj, solution)
    
    # Inter-route optimization
    remaining = args.time_limit - (time.time() - start_time)
    if remaining > 2:
        routes = local_search_inter(inst, routes, remaining * 0.4)
        
        obj = compute_objective(inst, routes)
        if obj < best_obj:
            best_obj = obj
            best_routes = deepcopy(routes)
            solution = {"objective_value": obj, "routes": routes}
            if logger:
                logger.log_solution(obj, solution)
    
    # Iterative improvement with perturbation
    remaining = args.time_limit - (time.time() - start_time)
    iteration = 0
    
    while time.time() - start_time < args.time_limit - 5:
        iteration += 1
        
        # Perturb
        perturbed = perturb_solution(inst, best_routes, rng)
        
        # Local search on perturbed solution
        time_budget = min(5.0, (args.time_limit - (time.time() - start_time)) * 0.5)
        if time_budget < 1:
            break
        
        perturbed = local_search_intra(inst, perturbed, time_budget * 0.5)
        
        remaining = args.time_limit - (time.time() - start_time)
        if remaining > 1:
            perturbed = local_search_inter(inst, perturbed, min(3.0, remaining * 0.5))
        
        obj = compute_objective(inst, perturbed)
        if obj < best_obj:
            best_obj = obj
            best_routes = deepcopy(perturbed)
            solution = {"objective_value": obj, "routes": best_routes}
            if logger:
                logger.log_solution(obj, solution)
    
    # Final solution
    solution = {"objective_value": best_obj, "routes": best_routes}
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()