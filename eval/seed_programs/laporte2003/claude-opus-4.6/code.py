import argparse
import json
import time
import math
import itertools
from collections import defaultdict
import sys

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        inst = json.load(f)

    num_vertices = inst['num_vertices']
    num_markets = inst['num_markets']
    num_products = inst['num_products']
    depot = inst['depot']
    travel_costs = {}
    for key, val in inst['travel_costs'].items():
        i, j = map(int, key.split(','))
        travel_costs[(i, j)] = val
        travel_costs[(j, i)] = val

    product_demands = inst['product_demands']
    product_markets = {}
    for key, val in inst['product_markets'].items():
        product_markets[int(key)] = val

    purchase_costs = {}
    for key, val in inst['purchase_costs'].items():
        p, m = map(int, key.split(','))
        purchase_costs[(p, m)] = val

    supplies = {}
    for key, val in inst['supplies'].items():
        p, m = map(int, key.split(','))
        supplies[(p, m)] = val

    # All markets (vertices except depot)
    all_markets = [v for v in range(num_vertices) if v != depot]

    def get_cost(i, j):
        if i == j:
            return 0
        a, b = min(i, j), max(i, j)
        return travel_costs.get((a, b), travel_costs.get((b, a), 0))

    # Determine mandatory markets
    mandatory_markets = set()
    for p in range(num_products):
        markets_for_p = product_markets[p]
        total_supply = sum(supplies[(p, m)] for m in markets_for_p)
        demand_p = product_demands[p]
        for m in markets_for_p:
            # If removing m makes total supply < demand, m is mandatory
            if total_supply - supplies[(p, m)] < demand_p:
                mandatory_markets.add(m)

    # Compute minimum purchase cost for each product given a set of visited markets
    def compute_min_purchase_cost(visited_set):
        """Given a set of visited markets, compute minimum purchase cost for all products.
        Returns (total_cost, purchases_dict) or (None, None) if infeasible."""
        total_cost = 0.0
        purchases = {}
        for p in range(num_products):
            demand_p = product_demands[p]
            # Available markets for this product that are visited
            available = [(purchase_costs[(p, m)], m) for m in product_markets[p] if m in visited_set]
            # Sort by cost
            available.sort()
            remaining = demand_p
            total_available = sum(supplies[(p, m)] for _, m in available)
            if total_available < demand_p:
                return None, None
            p_purchases = {}
            for cost, m in available:
                if remaining <= 0:
                    break
                buy = min(remaining, supplies[(p, m)])
                p_purchases[m] = buy
                total_cost += cost * buy
                remaining -= buy
            if remaining > 0:
                return None, None
            purchases[p] = p_purchases
        return total_cost, purchases

    # For small instances, try exact approach with Gurobi
    # For larger instances, use heuristic approach

    def solve_tsp_nearest_neighbor(vertices_list):
        """Solve TSP approximately using nearest neighbor heuristic starting from depot."""
        if len(vertices_list) <= 1:
            return vertices_list, 0
        
        visited = [depot]
        remaining = set(vertices_list) - {depot}
        total_cost = 0
        
        while remaining:
            current = visited[-1]
            nearest = min(remaining, key=lambda x: get_cost(current, x))
            total_cost += get_cost(current, nearest)
            visited.append(nearest)
            remaining.remove(nearest)
        
        total_cost += get_cost(visited[-1], depot)
        return visited, total_cost

    def solve_tsp_2opt(tour):
        """Improve a tour using 2-opt."""
        improved = True
        n = len(tour)
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Cost of current edges
                    if j == n - 1:
                        old_cost = get_cost(tour[i-1], tour[i]) + get_cost(tour[j], tour[0])
                        new_cost = get_cost(tour[i-1], tour[j]) + get_cost(tour[i], tour[0])
                    else:
                        old_cost = get_cost(tour[i-1], tour[i]) + get_cost(tour[j], tour[j+1])
                        new_cost = get_cost(tour[i-1], tour[j]) + get_cost(tour[i], tour[j+1])
                    
                    if new_cost < old_cost - 1e-10:
                        tour[i:j+1] = tour[i:j+1][::-1]
                        improved = True
        return tour

    def tour_cost(tour):
        """Compute total travel cost of a tour."""
        cost = 0
        for i in range(len(tour)):
            cost += get_cost(tour[i], tour[(i+1) % len(tour)])
        return cost

    def evaluate_solution(visited_markets_set):
        """Evaluate a solution: compute purchase cost + TSP tour cost."""
        purchase_cost, purchases = compute_min_purchase_cost(visited_markets_set)
        if purchase_cost is None:
            return None, None, None, None
        
        vertices = [depot] + list(visited_markets_set)
        if len(vertices) <= 3:
            # For 2 or 3 vertices, tour is trivial
            t = vertices
            tc = tour_cost(t)
        else:
            t, _ = solve_tsp_nearest_neighbor(vertices)
            # Remove depot from beginning if present, we'll handle it
            # Actually nearest_neighbor starts from depot
            # Make sure tour starts at depot
            if t[0] != depot:
                idx = t.index(depot)
                t = t[idx:] + t[:idx]
            t = solve_tsp_2opt(t)
            tc = tour_cost(t)
        
        total = tc + purchase_cost
        return total, t, tc, purchases

    # Strategy: Start with mandatory markets, then greedily add beneficial markets
    
    # First check if mandatory markets alone are sufficient
    best_obj = float('inf')
    best_tour = None
    best_purchases = None
    best_visited = None

    def try_solution(visited_set):
        nonlocal best_obj, best_tour, best_purchases, best_visited
        if time.time() - start_time > args.time_limit - 2:
            return False
        obj, t, tc, purchases = evaluate_solution(visited_set)
        if obj is not None and obj < best_obj:
            best_obj = obj
            best_tour = t
            best_purchases = purchases
            best_visited = visited_set.copy()
            if logger:
                logger.log(best_obj)
        return True

    # Check feasibility with all markets
    all_markets_set = set(all_markets)
    pc_all, _ = compute_min_purchase_cost(all_markets_set)
    if pc_all is None:
        # Should not happen per problem statement
        pass

    # Start with mandatory markets
    current_visited = set(mandatory_markets)
    
    # Check if mandatory markets are sufficient for all products
    def is_feasible(visited_set):
        for p in range(num_products):
            available_supply = sum(supplies[(p, m)] for m in product_markets[p] if m in visited_set)
            if available_supply < product_demands[p]:
                return False
        return True

    # If not feasible with mandatory, add cheapest markets to make feasible
    if not is_feasible(current_visited):
        # For each product, check if we need more markets
        for p in range(num_products):
            available_supply = sum(supplies[(p, m)] for m in product_markets[p] if m in current_visited)
            if available_supply < product_demands[p]:
                # Need to add markets for this product
                remaining_markets = [(purchase_costs[(p, m)], m) for m in product_markets[p] if m not in current_visited]
                remaining_markets.sort()
                for cost, m in remaining_markets:
                    current_visited.add(m)
                    available_supply += supplies[(p, m)]
                    if available_supply >= product_demands[p]:
                        break

    if is_feasible(current_visited):
        try_solution(current_visited)

    # Try with all markets
    try_solution(all_markets_set)

    # Now try to improve by removing non-mandatory markets one at a time
    if best_visited is not None:
        improved = True
        while improved and time.time() - start_time < args.time_limit - 5:
            improved = False
            removable = [m for m in best_visited if m not in mandatory_markets]
            for m in removable:
                if time.time() - start_time > args.time_limit - 5:
                    break
                test_set = best_visited - {m}
                if is_feasible(test_set):
                    old_obj = best_obj
                    try_solution(test_set)
                    if best_obj < old_obj:
                        improved = True
                        break

    # Try greedy addition from mandatory set
    if time.time() - start_time < args.time_limit - 10:
        # Start fresh from mandatory
        base_visited = set(mandatory_markets)
        # Make feasible
        if not is_feasible(base_visited):
            for p in range(num_products):
                available_supply = sum(supplies[(p, m)] for m in product_markets[p] if m in base_visited)
                if available_supply < product_demands[p]:
                    remaining_markets = [(purchase_costs[(p, m)], m) for m in product_markets[p] if m not in base_visited]
                    remaining_markets.sort()
                    for cost, m in remaining_markets:
                        base_visited.add(m)
                        available_supply += supplies[(p, m)]
                        if available_supply >= product_demands[p]:
                            break

        if is_feasible(base_visited):
            try_solution(base_visited)

        # Try adding each non-visited market to see if it improves
        improved = True
        current = set(base_visited)
        while improved and time.time() - start_time < args.time_limit - 5:
            improved = False
            non_visited = [m for m in all_markets if m not in current]
            best_addition = None
            best_addition_obj = best_obj
            for m in non_visited:
                if time.time() - start_time > args.time_limit - 5:
                    break
                test_set = current | {m}
                obj, t, tc, purchases = evaluate_solution(test_set)
                if obj is not None and obj < best_addition_obj:
                    best_addition = m
                    best_addition_obj = obj
            if best_addition is not None:
                current.add(best_addition)
                try_solution(current)
                improved = True

    # Try a more sophisticated approach: enumerate subsets for small instances
    if num_markets <= 15 and time.time() - start_time < args.time_limit - 10:
        # We can try more combinations
        optional_markets = [m for m in all_markets if m not in mandatory_markets]
        n_opt = len(optional_markets)
        
        if n_opt <= 15:
            # Try all subsets of optional markets
            for size in range(n_opt + 1):
                if time.time() - start_time > args.time_limit - 5:
                    break
                for combo in itertools.combinations(optional_markets, size):
                    if time.time() - start_time > args.time_limit - 5:
                        break
                    test_set = mandatory_markets | set(combo)
                    if is_feasible(test_set):
                        try_solution(test_set)

    # Try Gurobi MIP for moderate instances
    remaining_time = args.time_limit - (time.time() - start_time) - 2
    if remaining_time > 5 and num_markets <= 50:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            # Formulate as MIP
            # Variables:
            # y[m] = 1 if market m is visited
            # x[i,j] = 1 if edge (i,j) is in tour
            # q[p,m] = quantity of product p bought at market m
            # u[v] = position in tour (MTZ formulation)
            
            model = gp.Model("purchasing_tour")
            model.setParam('TimeLimit', max(remaining_time - 1, 1))
            model.setParam('OutputFlag', 0)
            model.setParam('Threads', 1)
            
            vertices = list(range(num_vertices))
            markets = all_markets
            
            # Binary: visit market m
            y = {}
            for m in markets:
                y[m] = model.addVar(vtype=GRB.BINARY, name=f"y_{m}")
                if m in mandatory_markets:
                    y[m].lb = 1.0
            
            # Binary: edge in tour
            x = {}
            for i in range(num_vertices):
                for j in range(i+1, num_vertices):
                    x[(i,j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            
            def xvar(i, j):
                if i < j:
                    return x[(i,j)]
                else:
                    return x[(j,i)]
            
            # Continuous: quantity purchased
            q = {}
            for p in range(num_products):
                for m in product_markets[p]:
                    q[(p,m)] = model.addVar(lb=0, ub=supplies[(p,m)], name=f"q_{p}_{m}")
            
            # MTZ position variables
            n_v = num_vertices
            u = {}
            for v in vertices:
                if v != depot:
                    u[v] = model.addVar(lb=1, ub=n_v - 1, name=f"u_{v}")
            
            model.update()
            
            # Objective: minimize travel + purchase cost
            travel_obj = gp.quicksum(get_cost(i,j) * x[(i,j)] for i,j in x)
            purchase_obj = gp.quicksum(purchase_costs[(p,m)] * q[(p,m)] for p,m in q)
            model.setObjective(travel_obj + purchase_obj, GRB.MINIMIZE)
            
            # Degree constraints: depot always has degree 2
            model.addConstr(gp.quicksum(xvar(depot, j) for j in vertices if j != depot) == 2)
            
            # Each visited market has degree 2, unvisited has degree 0
            for m in markets:
                model.addConstr(gp.quicksum(xvar(m, j) for j in vertices if j != m) == 2 * y[m])
            
            # Demand satisfaction
            for p in range(num_products):
                model.addConstr(gp.quicksum(q[(p,m)] for m in product_markets[p]) == product_demands[p])
            
            # Can only purchase at visited markets
            for p in range(num_products):
                for m in product_markets[p]:
                    model.addConstr(q[(p,m)] <= supplies[(p,m)] * y[m])
            
            # MTZ subtour elimination
            # For edges between non-depot vertices
            M = n_v
            for m1 in markets:
                for m2 in markets:
                    if m1 < m2:
                        model.addConstr(u[m1] - u[m2] + M * xvar(m1, m2) <= M - 1)
                        model.addConstr(u[m2] - u[m1] + M * xvar(m1, m2) <= M - 1)
            
            # Warm start with best solution found so far
            if best_visited is not None and best_tour is not None:
                for m in markets:
                    y[m].Start = 1.0 if m in best_visited else 0.0
                
                # Set edge variables from best tour
                for i, j in x:
                    x[(i,j)].Start = 0.0
                
                for idx in range(len(best_tour)):
                    i = best_tour[idx]
                    j = best_tour[(idx+1) % len(best_tour)]
                    a, b = min(i,j), max(i,j)
                    if (a,b) in x:
                        x[(a,b)].Start = 1.0
                
                # Set purchase variables
                for p, m_dict in best_purchases.items():
                    for m, qty in m_dict.items():
                        if (p, m) in q:
                            q[(p,m)].Start = qty
                
                # Set MTZ variables
                if len(best_tour) > 1:
                    depot_idx = best_tour.index(depot)
                    tour_ordered = best_tour[depot_idx:] + best_tour[:depot_idx]
                    for pos, v in enumerate(tour_ordered):
                        if v != depot and v in u:
                            u[v].Start = pos
            
            model.optimize()
            
            if model.SolCount > 0:
                obj_val = model.ObjVal
                
                # Extract solution
                visited_set = set()
                for m in markets:
                    if y[m].X > 0.5:
                        visited_set.add(m)
                
                # Extract tour edges
                tour_edges = []
                for (i,j), var in x.items():
                    if var.X > 0.5:
                        tour_edges.append((i,j))
                
                # Reconstruct tour from edges
                adj = defaultdict(list)
                for i, j in tour_edges:
                    adj[i].append(j)
                    adj[j].append(i)
                
                if depot in adj and len(adj[depot]) == 2:
                    tour_seq = [depot]
                    prev = depot
                    current = adj[depot][0]
                    while current != depot:
                        tour_seq.append(current)
                        neighbors = adj[current]
                        next_v = neighbors[0] if neighbors[0] != prev else neighbors[1]
                        prev = current
                        current = next_v
                    
                    # Extract purchases
                    purchases_extracted = {}
                    for p in range(num_products):
                        p_dict = {}
                        for m in product_markets[p]:
                            if (p,m) in q and q[(p,m)].X > 0.5:
                                p_dict[m] = round(q[(p,m)].X)
                        purchases_extracted[p] = p_dict
                    
                    # Compute objective
                    tc = tour_cost(tour_seq)
                    pc = sum(purchase_costs[(p,m)] * amt for p, mdict in purchases_extracted.items() for m, amt in mdict.items())
                    total_obj = tc + pc
                    
                    if total_obj < best_obj:
                        best_obj = total_obj
                        best_tour = tour_seq
                        best_purchases = purchases_extracted
                        best_visited = visited_set
                        if logger:
                            logger.log(best_obj)
        except Exception as e:
            pass

    # If no solution found yet, use all markets
    if best_tour is None:
        try_solution(all_markets_set)

    # Final 2-opt improvement on best tour
    if best_tour is not None and len(best_tour) > 3:
        # Ensure starts at depot
        if best_tour[0] != depot:
            idx = best_tour.index(depot)
            best_tour = best_tour[idx:] + best_tour[:idx]
        
        old_tc = tour_cost(best_tour)
        best_tour = solve_tsp_2opt(best_tour)
        new_tc = tour_cost(best_tour)
        if new_tc < old_tc:
            # Recompute objective
            pc, purchases = compute_min_purchase_cost(best_visited)
            best_obj = new_tc + pc
            best_purchases = purchases
            if logger:
                logger.log(best_obj)

    # Or-opt improvement
    if best_tour is not None and len(best_tour) > 4:
        improved = True
        while improved and time.time() - start_time < args.time_limit - 1:
            improved = False
            n = len(best_tour)
            for seg_len in [1, 2, 3]:
                if improved:
                    break
                for i in range(n):
                    if improved:
                        break
                    if time.time() - start_time > args.time_limit - 1:
                        break
                    # Don't move depot
                    # Check if segment contains depot
                    segment_indices = [(i + k) % n for k in range(seg_len)]
                    segment_vertices = [best_tour[si] for si in segment_indices]
                    if depot in segment_vertices:
                        continue
                    
                    # Remove segment
                    before = (i - 1) % n
                    after = (i + seg_len) % n
                    
                    old_cost = get_cost(best_tour[before], best_tour[i % n])
                    last_seg = (i + seg_len - 1) % n
                    old_cost += get_cost(best_tour[last_seg], best_tour[after])
                    
                    new_direct = get_cost(best_tour[before], best_tour[after])
                    removal_saving = old_cost - new_direct
                    
                    # Try inserting segment at each other position
                    remaining_tour = []
                    idx_map = []
                    skip = set(segment_indices)
                    for k in range(n):
                        if k not in skip:
                            remaining_tour.append(best_tour[k])
                    
                    rn = len(remaining_tour)
                    best_insert_cost = float('inf')
                    best_insert_pos = -1
                    
                    for j in range(rn):
                        j_next = (j + 1) % rn
                        insert_cost = (get_cost(remaining_tour[j], segment_vertices[0]) +
                                       get_cost(segment_vertices[-1], remaining_tour[j_next]) -
                                       get_cost(remaining_tour[j], remaining_tour[j_next]))
                        if insert_cost < best_insert_cost:
                            best_insert_cost = insert_cost
                            best_insert_pos = j
                    
                    if removal_saving + (-best_insert_cost) > 1e-10:
                        # Beneficial move: removal_saving > best_insert_cost
                        if removal_saving > best_insert_cost + 1e-10:
                            new_tour = remaining_tour[:best_insert_pos+1] + segment_vertices + remaining_tour[best_insert_pos+1:]
                            # Rotate to start at depot
                            di = new_tour.index(depot)
                            new_tour = new_tour[di:] + new_tour[:di]
                            new_tc = tour_cost(new_tour)
                            if new_tc < tour_cost(best_tour) - 1e-10:
                                best_tour = new_tour
                                improved = True
                                pc, purchases = compute_min_purchase_cost(best_visited)
                                best_obj = new_tc + pc
                                best_purchases = purchases
                                if logger:
                                    logger.log(best_obj)

    # Prepare output
    if best_tour is None:
        # Fallback
        best_visited = all_markets_set
        best_tour = [depot] + all_markets
        best_obj = float('inf')
        _, best_purchases = compute_min_purchase_cost(best_visited)
        best_obj = tour_cost(best_tour) + sum(purchase_costs[(p,m)] * amt for p, mdict in best_purchases.items() for m, amt in mdict.items())

    # Ensure tour starts at depot
    if best_tour[0] != depot:
        idx = best_tour.index(depot)
        best_tour = best_tour[idx:] + best_tour[:idx]

    visited_markets_list = sorted([m for m in best_visited])
    
    tour_edges = []
    for i in range(len(best_tour)):
        j = (i + 1) % len(best_tour)
        tour_edges.append([best_tour[i], best_tour[j]])

    # Format purchases
    purchases_out = {}
    for p in range(num_products):
        p_dict = {}
        if p in best_purchases:
            for m, qty in best_purchases[p].items():
                if qty > 0:
                    p_dict[str(m)] = qty
        purchases_out[str(p)] = p_dict

    # Recompute objective value accurately
    travel_total = sum(get_cost(e[0], e[1]) for e in tour_edges)
    purchase_total = sum(purchase_costs[(p, int(m))] * qty 
                        for p_str, mdict in purchases_out.items() 
                        for m, qty in mdict.items()
                        for p in [int(p_str)])
    final_obj = travel_total + purchase_total

    solution = {
        "objective_value": final_obj,
        "visited_markets": visited_markets_list,
        "tour": best_tour,
        "tour_edges": tour_edges,
        "purchases": purchases_out
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    solve()