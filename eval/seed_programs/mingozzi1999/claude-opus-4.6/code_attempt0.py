import argparse
import json
import time
import random
import math
from copy import deepcopy

from solution_logger import SolutionLogger


def solve(instance, time_limit, logger=None):
    start_time = time.time()
    
    depot_idx = instance["depot"]["index"]
    num_linehaul = instance["num_linehaul_customers"]
    num_backhaul = instance["num_backhaul_customers"]
    num_vehicles = instance["num_vehicles"]
    capacity = instance["vehicle_capacity"]
    dist = instance["distance_matrix"]
    
    linehaul_customers = [c["index"] for c in instance["linehaul_customers"]]
    backhaul_customers = [c["index"] for c in instance["backhaul_customers"]]
    
    demand = {}
    for c in instance["linehaul_customers"]:
        demand[c["index"]] = c["demand"]
    for c in instance["backhaul_customers"]:
        demand[c["index"]] = c["demand"]
    
    lh_set = set(linehaul_customers)
    bh_set = set(backhaul_customers)
    
    def route_cost(route):
        """Calculate cost of a route (list of nodes including depot at start and end)."""
        cost = 0
        for i in range(len(route) - 1):
            cost += dist[route[i]][route[i+1]]
        return cost
    
    def validate_route(lh_nodes, bh_nodes):
        """Check capacity constraints for a route."""
        lh_demand = sum(demand[n] for n in lh_nodes)
        bh_demand = sum(demand[n] for n in bh_nodes)
        return lh_demand <= capacity and bh_demand <= capacity
    
    def build_route(lh_nodes, bh_nodes):
        """Build a route: depot -> linehaul nodes -> backhaul nodes -> depot."""
        return [depot_idx] + list(lh_nodes) + list(bh_nodes) + [depot_idx]
    
    def total_cost(routes):
        return sum(route_cost(r) for r in routes)
    
    # ---- Nearest Neighbor Construction Heuristic ----
    def nearest_neighbor_construction():
        """Build initial solution using nearest neighbor heuristic."""
        unvisited_lh = set(linehaul_customers)
        unvisited_bh = set(backhaul_customers)
        routes = []
        
        while unvisited_lh or unvisited_bh:
            if not unvisited_lh and unvisited_bh:
                # We need linehaul customers on each route, but we've run out
                # Assign remaining backhaul to existing routes or create dummy routes
                # Actually, backhaul-only routes are not allowed per problem description
                # So we need to attach remaining backhaul to existing routes
                break
            
            lh_route = []
            bh_route = []
            lh_load = 0
            bh_load = 0
            current = depot_idx
            
            # Build linehaul portion
            while unvisited_lh:
                best_next = None
                best_dist = float('inf')
                for n in unvisited_lh:
                    if lh_load + demand[n] <= capacity:
                        d = dist[current][n]
                        if d < best_dist:
                            best_dist = d
                            best_next = n
                if best_next is None:
                    break
                lh_route.append(best_next)
                lh_load += demand[best_next]
                unvisited_lh.remove(best_next)
                current = best_next
            
            if not lh_route:
                break
            
            # Build backhaul portion
            while unvisited_bh:
                best_next = None
                best_dist = float('inf')
                for n in unvisited_bh:
                    if bh_load + demand[n] <= capacity:
                        d = dist[current][n]
                        if d < best_dist:
                            best_dist = d
                            best_next = n
                if best_next is None:
                    break
                bh_route.append(best_next)
                bh_load += demand[best_next]
                unvisited_bh.remove(best_next)
                current = best_next
            
            routes.append((lh_route, bh_route))
        
        # Attach remaining backhaul customers to existing routes
        remaining_bh = list(unvisited_bh)
        for n in remaining_bh:
            best_route = None
            best_cost_increase = float('inf')
            for i, (lh, bh) in enumerate(routes):
                bh_load = sum(demand[b] for b in bh)
                if bh_load + demand[n] <= capacity:
                    # Try inserting at best position in backhaul portion
                    for pos in range(len(bh) + 1):
                        new_bh = list(bh)
                        new_bh.insert(pos, n)
                        old_route = build_route(lh, bh)
                        new_route = build_route(lh, new_bh)
                        cost_inc = route_cost(new_route) - route_cost(old_route)
                        if cost_inc < best_cost_increase:
                            best_cost_increase = cost_inc
                            best_route = i
                            best_pos = pos
            if best_route is not None:
                lh, bh = routes[best_route]
                new_bh = list(bh)
                new_bh.insert(best_pos, n)
                routes[best_route] = (lh, new_bh)
                unvisited_bh.remove(n)
        
        # If still remaining, force them
        for n in list(unvisited_bh):
            for i, (lh, bh) in enumerate(routes):
                bh_load = sum(demand[b] for b in bh)
                if bh_load + demand[n] <= capacity:
                    new_bh = list(bh) + [n]
                    routes[i] = (lh, new_bh)
                    unvisited_bh.remove(n)
                    break
        
        # Pad with empty routes if needed
        while len(routes) < num_vehicles:
            routes.append(([], []))
        
        return routes
    
    # ---- Savings-based Construction ----
    def savings_construction():
        """Clarke-Wright savings heuristic adapted for VRPB."""
        # Start with individual routes for each linehaul customer
        routes_lh = [[c] for c in linehaul_customers]
        route_of = {c: i for i, c in enumerate(linehaul_customers)}
        lh_loads = [demand[c] for c in linehaul_customers]
        
        # Compute savings
        savings = []
        for i in linehaul_customers:
            for j in linehaul_customers:
                if i != j:
                    s = dist[i][depot_idx] + dist[depot_idx][j] - dist[i][j]
                    if s > 0:
                        savings.append((s, i, j))
        savings.sort(reverse=True)
        
        for s, i, j in savings:
            ri = route_of.get(i)
            rj = route_of.get(j)
            if ri is None or rj is None or ri == rj:
                continue
            if routes_lh[ri] is None or routes_lh[rj] is None:
                continue
            # Check if i is last in its route and j is first in its route
            if routes_lh[ri][-1] == i and routes_lh[rj][0] == j:
                if lh_loads[ri] + lh_loads[rj] <= capacity:
                    # Merge
                    new_route = routes_lh[ri] + routes_lh[rj]
                    new_load = lh_loads[ri] + lh_loads[rj]
                    new_idx = ri
                    routes_lh[new_idx] = new_route
                    lh_loads[new_idx] = new_load
                    for c in new_route:
                        route_of[c] = new_idx
                    routes_lh[rj] = None
                    lh_loads[rj] = 0
        
        # Collect non-None routes
        final_lh_routes = [r for r in routes_lh if r is not None]
        
        # Limit to num_vehicles by merging smallest routes
        while len(final_lh_routes) > num_vehicles:
            # Find two routes that can merge with minimum cost increase
            best_merge = None
            best_cost = float('inf')
            for i in range(len(final_lh_routes)):
                for j in range(i+1, len(final_lh_routes)):
                    li = sum(demand[c] for c in final_lh_routes[i])
                    lj = sum(demand[c] for c in final_lh_routes[j])
                    if li + lj <= capacity:
                        # Cost of merging i then j
                        cost = dist[final_lh_routes[i][-1]][final_lh_routes[j][0]]
                        if cost < best_cost:
                            best_cost = cost
                            best_merge = (i, j)
            if best_merge is None:
                break
            i, j = best_merge
            final_lh_routes[i] = final_lh_routes[i] + final_lh_routes[j]
            final_lh_routes.pop(j)
        
        # Now assign backhaul customers to routes
        routes = [(list(r), []) for r in final_lh_routes]
        
        # Pad with empty routes
        while len(routes) < num_vehicles:
            routes.append(([], []))
        
        # Assign backhaul customers greedily
        unvisited_bh = list(backhaul_customers)
        random.shuffle(unvisited_bh)
        
        for n in unvisited_bh:
            best_route = None
            best_cost_increase = float('inf')
            for i, (lh, bh) in enumerate(routes):
                if not lh:
                    continue  # Can't add backhaul to empty linehaul route
                bh_load = sum(demand[b] for b in bh)
                if bh_load + demand[n] <= capacity:
                    for pos in range(len(bh) + 1):
                        new_bh = list(bh)
                        new_bh.insert(pos, n)
                        old_route = build_route(lh, bh)
                        new_route = build_route(lh, new_bh)
                        cost_inc = route_cost(new_route) - route_cost(old_route)
                        if cost_inc < best_cost_increase:
                            best_cost_increase = cost_inc
                            best_route = i
                            best_pos = pos
            if best_route is not None:
                lh, bh = routes[best_route]
                new_bh = list(bh)
                new_bh.insert(best_pos, n)
                routes[best_route] = (lh, new_bh)
        
        return routes
    
    # ---- Solution representation: list of (lh_list, bh_list) tuples ----
    
    def solution_to_routes(sol):
        """Convert solution representation to route format."""
        result = []
        for lh, bh in sol:
            if lh or bh:
                result.append(build_route(lh, bh))
            else:
                result.append([depot_idx, depot_idx])
        return result
    
    def solution_cost(sol):
        return sum(route_cost(build_route(lh, bh)) for lh, bh in sol)
    
    # ---- Local Search Operators ----
    
    def or_opt_within_route(sol):
        """Move a segment of 1-3 nodes within the same route portion."""
        improved = True
        while improved:
            improved = False
            for ri in range(len(sol)):
                lh, bh = sol[ri]
                # Try within linehaul portion
                for segment_len in [1, 2, 3]:
                    for portion_type in ['lh', 'bh']:
                        portion = list(lh) if portion_type == 'lh' else list(bh)
                        if len(portion) < segment_len + 1:
                            continue
                        best_improvement = 0
                        best_move = None
                        for i in range(len(portion) - segment_len + 1):
                            segment = portion[i:i+segment_len]
                            remaining = portion[:i] + portion[i+segment_len:]
                            for j in range(len(remaining) + 1):
                                if j == i:
                                    continue
                                new_portion = remaining[:j] + segment + remaining[j:]
                                if portion_type == 'lh':
                                    new_cost = route_cost(build_route(new_portion, bh))
                                    old_cost = route_cost(build_route(lh, bh))
                                else:
                                    new_cost = route_cost(build_route(lh, new_portion))
                                    old_cost = route_cost(build_route(lh, bh))
                                improvement = old_cost - new_cost
                                if improvement > best_improvement:
                                    best_improvement = improvement
                                    best_move = (portion_type, new_portion)
                        if best_move:
                            ptype, new_portion = best_move
                            if ptype == 'lh':
                                sol[ri] = (new_portion, bh)
                            else:
                                sol[ri] = (lh, new_portion)
                            improved = True
        return sol
    
    def two_opt_within(sol):
        """2-opt within each portion of each route."""
        improved = True
        while improved:
            improved = False
            for ri in range(len(sol)):
                lh, bh = sol[ri]
                for portion_type in ['lh', 'bh']:
                    portion = list(lh) if portion_type == 'lh' else list(bh)
                    if len(portion) < 2:
                        continue
                    # Build the relevant sequence with depot boundaries
                    if portion_type == 'lh':
                        prev_node = depot_idx
                        next_node = bh[0] if bh else depot_idx
                    else:
                        prev_node = lh[-1] if lh else depot_idx
                        next_node = depot_idx
                    
                    best_improvement = 0
                    best_ij = None
                    n = len(portion)
                    for i in range(n - 1):
                        for j in range(i + 1, n):
                            # Reverse portion[i:j+1]
                            new_portion = portion[:i] + portion[i:j+1][::-1] + portion[j+1:]
                            # Quick cost comparison for affected edges
                            node_before_i = prev_node if i == 0 else portion[i-1]
                            node_after_j = next_node if j == n-1 else portion[j+1]
                            
                            old_cost = dist[node_before_i][portion[i]] + dist[portion[j]][node_after_j]
                            new_cost = dist[node_before_i][portion[j]] + dist[portion[i]][node_after_j]
                            
                            # Also need to account for internal reversal cost changes
                            # For simplicity, compute full route cost difference
                            if new_cost < old_cost:
                                if portion_type == 'lh':
                                    full_old = route_cost(build_route(portion, bh))
                                    full_new = route_cost(build_route(new_portion, bh))
                                else:
                                    full_old = route_cost(build_route(lh, portion))
                                    full_new = route_cost(build_route(lh, new_portion))
                                imp = full_old - full_new
                                if imp > best_improvement:
                                    best_improvement = imp
                                    best_ij = (i, j, new_portion)
                    
                    if best_ij:
                        i, j, new_portion = best_ij
                        if portion_type == 'lh':
                            sol[ri] = (new_portion, bh)
                        else:
                            sol[ri] = (lh, new_portion)
                        improved = True
        return sol
    
    def relocate_customer(sol):
        """Move a single customer from one route to another."""
        improved = True
        while improved:
            improved = False
            best_improvement = 0
            best_move = None
            
            for ri in range(len(sol)):
                lh_r, bh_r = sol[ri]
                
                # Try relocating linehaul customers
                for ci in range(len(lh_r)):
                    cust = lh_r[ci]
                    # Remove from route ri
                    new_lh_r = lh_r[:ci] + lh_r[ci+1:]
                    if not new_lh_r and bh_r:
                        continue  # Can't leave backhaul without linehaul
                    
                    old_cost_ri = route_cost(build_route(lh_r, bh_r))
                    new_cost_ri = route_cost(build_route(new_lh_r, bh_r))
                    saving = old_cost_ri - new_cost_ri
                    
                    for rj in range(len(sol)):
                        if ri == rj:
                            continue
                        lh_s, bh_s = sol[rj]
                        lh_load_s = sum(demand[c] for c in lh_s)
                        if lh_load_s + demand[cust] > capacity:
                            continue
                        
                        old_cost_rj = route_cost(build_route(lh_s, bh_s))
                        
                        for pos in range(len(lh_s) + 1):
                            new_lh_s = list(lh_s)
                            new_lh_s.insert(pos, cust)
                            new_cost_rj = route_cost(build_route(new_lh_s, bh_s))
                            insertion_cost = new_cost_rj - old_cost_rj
                            
                            total_imp = saving - insertion_cost
                            if total_imp > best_improvement:
                                best_improvement = total_imp
                                best_move = ('lh', ri, ci, rj, pos)
                
                # Try relocating backhaul customers
                for ci in range(len(bh_r)):
                    cust = bh_r[ci]
                    new_bh_r = bh_r[:ci] + bh_r[ci+1:]
                    
                    old_cost_ri = route_cost(build_route(lh_r, bh_r))
                    new_cost_ri = route_cost(build_route(lh_r, new_bh_r))
                    saving = old_cost_ri - new_cost_ri
                    
                    for rj in range(len(sol)):
                        if ri == rj:
                            continue
                        lh_s, bh_s = sol[rj]
                        if not lh_s:
                            continue  # Can't add backhaul to route without linehaul
                        bh_load_s = sum(demand[c] for c in bh_s)
                        if bh_load_s + demand[cust] > capacity:
                            continue
                        
                        old_cost_rj = route_cost(build_route(lh_s, bh_s))
                        
                        for pos in range(len(bh_s) + 1):
                            new_bh_s = list(bh_s)
                            new_bh_s.insert(pos, cust)
                            new_cost_rj = route_cost(build_route(lh_s, new_bh_s))
                            insertion_cost = new_cost_rj - old_cost_rj
                            
                            total_imp = saving - insertion_cost
                            if total_imp > best_improvement:
                                best_improvement = total_imp
                                best_move = ('bh', ri, ci, rj, pos)
            
            if best_move and best_improvement > 0.5:
                ptype, ri, ci, rj, pos = best_move
                if ptype == 'lh':
                    lh_r, bh_r = sol[ri]
                    cust = lh_r[ci]
                    new_lh_r = lh_r[:ci] + lh_r[ci+1:]
                    sol[ri] = (new_lh_r, bh_r)
                    lh_s, bh_s = sol[rj]
                    new_lh_s = list(lh_s)
                    new_lh_s.insert(pos, cust)
                    sol[rj] = (new_lh_s, bh_s)
                else:
                    lh_r, bh_r = sol[ri]
                    cust = bh_r[ci]
                    new_bh_r = bh_r[:ci] + bh_r[ci+1:]
                    sol[ri] = (lh_r, new_bh_r)
                    lh_s, bh_s = sol[rj]
                    new_bh_s = list(bh_s)
                    new_bh_s.insert(pos, cust)
                    sol[rj] = (lh_s, new_bh_s)
                improved = True
        return sol
    
    def swap_customers(sol):
        """Swap customers between routes."""
        improved = True
        while improved:
            improved = False
            best_improvement = 0
            best_swap = None
            
            for ri in range(len(sol)):
                for rj in range(ri + 1, len(sol)):
                    lh_i, bh_i = sol[ri]
                    lh_j, bh_j = sol[rj]
                    
                    # Swap linehaul customers
                    for ci in range(len(lh_i)):
                        for cj in range(len(lh_j)):
                            ci_cust = lh_i[ci]
                            cj_cust = lh_j[cj]
                            
                            # Check capacity
                            new_load_i = sum(demand[c] for c in lh_i) - demand[ci_cust] + demand[cj_cust]
                            new_load_j = sum(demand[c] for c in lh_j) - demand[cj_cust] + demand[ci_cust]
                            if new_load_i > capacity or new_load_j > capacity:
                                continue
                            
                            new_lh_i = list(lh_i)
                            new_lh_i[ci] = cj_cust
                            new_lh_j = list(lh_j)
                            new_lh_j[cj] = ci_cust
                            
                            old_cost = route_cost(build_route(lh_i, bh_i)) + route_cost(build_route(lh_j, bh_j))
                            new_cost = route_cost(build_route(new_lh_i, bh_i)) + route_cost(build_route(new_lh_j, bh_j))
                            
                            imp = old_cost - new_cost
                            if imp > best_improvement:
                                best_improvement = imp
                                best_swap = ('lh', ri, ci, rj, cj, new_lh_i, new_lh_j)
                    
                    # Swap backhaul customers
                    for ci in range(len(bh_i)):
                        for cj in range(len(bh_j)):
                            ci_cust = bh_i[ci]
                            cj_cust = bh_j[cj]
                            
                            new_load_i = sum(demand[c] for c in bh_i) - demand[ci_cust] + demand[cj_cust]
                            new_load_j = sum(demand[c] for c in bh_j) - demand[cj_cust] + demand[ci_cust]
                            if new_load_i > capacity or new_load_j > capacity:
                                continue
                            
                            new_bh_i = list(bh_i)
                            new_bh_i[ci] = cj_cust
                            new_bh_j = list(bh_j)
                            new_bh_j[cj] = ci_cust
                            
                            old_cost = route_cost(build_route(lh_i, bh_i)) + route_cost(build_route(lh_j, bh_j))
                            new_cost = route_cost(build_route(lh_i, new_bh_i)) + route_cost(build_route(lh_j, new_bh_j))
                            
                            imp = old_cost - new_cost
                            if imp > best_improvement:
                                best_improvement = imp
                                best_swap = ('bh', ri, ci, rj, cj, new_bh_i, new_bh_j)
            
            if best_swap and best_improvement > 0.5:
                ptype = best_swap[0]
                ri, ci, rj, cj = best_swap[1], best_swap[2], best_swap[3], best_swap[4]
                if ptype == 'lh':
                    new_lh_i, new_lh_j = best_swap[5], best_swap[6]
                    sol[ri] = (new_lh_i, sol[ri][1])
                    sol[rj] = (new_lh_j, sol[rj][1])
                else:
                    new_bh_i, new_bh_j = best_swap[5], best_swap[6]
                    sol[ri] = (sol[ri][0], new_bh_i)
                    sol[rj] = (sol[rj][0], new_bh_j)
                improved = True
        return sol
    
    def cross_exchange(sol):
        """Exchange tails of linehaul or backhaul portions between routes."""
        improved = True
        while improved:
            improved = False
            best_improvement = 0
            best_exchange = None
            
            for ri in range(len(sol)):
                for rj in range(ri + 1, len(sol)):
                    lh_i, bh_i = sol[ri]
                    lh_j, bh_j = sol[rj]
                    
                    # Cross exchange on linehaul tails
                    for ci in range(len(lh_i) + 1):
                        for cj in range(len(lh_j) + 1):
                            new_lh_i = lh_i[:ci] + lh_j[cj:]
                            new_lh_j = lh_j[:cj] + lh_i[ci:]
                            
                            # Check: no empty linehaul with non-empty backhaul
                            if not new_lh_i and bh_i:
                                continue
                            if not new_lh_j and bh_j:
                                continue
                            
                            load_i = sum(demand[c] for c in new_lh_i)
                            load_j = sum(demand[c] for c in new_lh_j)
                            if load_i > capacity or load_j > capacity:
                                continue
                            
                            old_cost = route_cost(build_route(lh_i, bh_i)) + route_cost(build_route(lh_j, bh_j))
                            new_cost = route_cost(build_route(new_lh_i, bh_i)) + route_cost(build_route(new_lh_j, bh_j))
                            
                            imp = old_cost - new_cost
                            if imp > best_improvement:
                                best_improvement = imp
                                best_exchange = ('lh', ri, rj, new_lh_i, new_lh_j)
                    
                    # Cross exchange on backhaul tails
                    for ci in range(len(bh_i) + 1):
                        for cj in range(len(bh_j) + 1):
                            new_bh_i = bh_i[:ci] + bh_j[cj:]
                            new_bh_j = bh_j[:cj] + bh_i[ci:]
                            
                            # Check: no backhaul without linehaul
                            if new_bh_i and not lh_i:
                                continue
                            if new_bh_j and not lh_j:
                                continue
                            
                            load_i = sum(demand[c] for c in new_bh_i)
                            load_j = sum(demand[c] for c in new_bh_j)
                            if load_i > capacity or load_j > capacity:
                                continue
                            
                            old_cost = route_cost(build_route(lh_i, bh_i)) + route_cost(build_route(lh_j, bh_j))
                            new_cost = route_cost(build_route(lh_i, new_bh_i)) + route_cost(build_route(lh_j, new_bh_j))
                            
                            imp = old_cost - new_cost
                            if imp > best_improvement:
                                best_improvement = imp
                                best_exchange = ('bh', ri, rj, new_bh_i, new_bh_j)
            
            if best_exchange and best_improvement > 0.5:
                ptype = best_exchange[0]
                ri, rj = best_exchange[1], best_exchange[2]
                if ptype == 'lh':
                    new_lh_i, new_lh_j = best_exchange[3], best_exchange[4]
                    sol[ri] = (new_lh_i, sol[ri][1])
                    sol[rj] = (new_lh_j, sol[rj][1])
                else:
                    new_bh_i, new_bh_j = best_exchange[3], best_exchange[4]
                    sol[ri] = (sol[ri][0], new_bh_i)
                    sol[rj] = (sol[rj][0], new_bh_j)
                improved = True
        return sol
    
    def local_search(sol):
        """Apply local search operators."""
        prev_cost = solution_cost(sol)
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit * 0.95:
                break
            
            sol = two_opt_within(sol)
            if time.time() - start_time > time_limit * 0.95:
                break
            
            sol = relocate_customer(sol)
            if time.time() - start_time > time_limit * 0.95:
                break
            
            sol = swap_customers(sol)
            if time.time() - start_time > time_limit * 0.95:
                break
            
            sol = cross_exchange(sol)
            if time.time() - start_time > time_limit * 0.95:
                break
            
            new_cost = solution_cost(sol)
            if new_cost < prev_cost - 0.5:
                improved = True
                prev_cost = new_cost
        return sol
    
    def is_valid_solution(sol):
        """Check if solution is valid."""
        all_lh = []
        all_bh = []
        for lh, bh in sol:
            all_lh.extend(lh)
            all_bh.extend(bh)
            if bh and not lh:
                return False
            if sum(demand[c] for c in lh) > capacity:
                return False
            if sum(demand[c] for c in bh) > capacity:
                return False
        if set(all_lh) != lh_set or len(all_lh) != len(lh_set):
            return False
        if set(all_bh) != bh_set or len(all_bh) != len(bh_set):
            return False
        return True
    
    def perturbation(sol, strength=3):
        """Perturb solution by random moves."""
        sol = deepcopy(sol)
        for _ in range(strength):
            move_type = random.choice(['relocate_lh', 'relocate_bh', 'swap_lh', 'swap_bh'])
            
            if move_type == 'relocate_lh':
                # Pick random linehaul customer and move to random route
                non_empty_lh = [(i, sol[i]) for i in range(len(sol)) if len(sol[i][0]) > 0]
                if not non_empty_lh:
                    continue
                ri, (lh_r, bh_r) = random.choice(non_empty_lh)
                # Don't remove last linehaul if there are backhaul
                if len(lh_r) == 1 and bh_r:
                    continue
                ci = random.randint(0, len(lh_r) - 1)
                cust = lh_r[ci]
                
                possible_targets = []
                for rj in range(len(sol)):
                    if rj == ri:
                        continue
                    lh_s, bh_s = sol[rj]
                    if sum(demand[c] for c in lh_s) + demand[cust] <= capacity:
                        possible_targets.append(rj)
                
                if not possible_targets:
                    continue
                
                rj = random.choice(possible_targets)
                new_lh_r = lh_r[:ci] + lh_r[ci+1:]
                sol[ri] = (new_lh_r, bh_r)
                lh_s, bh_s = sol[rj]
                pos = random.randint(0, len(lh_s))
                new_lh_s = list(lh_s)
                new_lh_s.insert(pos, cust)
                sol[rj] = (new_lh_s, bh_s)
            
            elif move_type == 'relocate_bh':
                non_empty_bh = [(i, sol[i]) for i in range(len(sol)) if len(sol[i][1]) > 0]
                if not non_empty_bh:
                    continue
                ri, (lh_r, bh_r) = random.choice(non_empty_bh)
                ci = random.randint(0, len(bh_r) - 1)
                cust = bh_r[ci]
                
                possible_targets = []
                for rj in range(len(sol)):
                    if rj == ri:
                        continue
                    lh_s, bh_s = sol[rj]
                    if not lh_s:
                        continue
                    if sum(demand[c] for c in bh_s) + demand[cust] <= capacity:
                        possible_targets.append(rj)
                
                if not possible_targets:
                    continue
                
                rj = random.choice(possible_targets)
                new_bh_r = bh_r[:ci] + bh_r[ci+1:]
                sol[ri] = (lh_r, new_bh_r)
                lh_s, bh_s = sol[rj]
                pos = random.randint(0, len(bh_s))
                new_bh_s = list(bh_s)
                new_bh_s.insert(pos, cust)
                sol[rj] = (lh_s, new_bh_s)
            
            elif move_type == 'swap_lh':
                non_empty_lh = [(i, sol[i]) for i in range(len(sol)) if len(sol[i][0]) > 0]
                if len(non_empty_lh) < 2:
                    continue
                (ri, (lh_i, bh_i)), (rj, (lh_j, bh_j)) = random.sample(non_empty_lh, 2)
                ci = random.randint(0, len(lh_i) - 1)
                cj = random.randint(0, len(lh_j) - 1)
                
                new_load_i = sum(demand[c] for c in lh_i) - demand[lh_i[ci]] + demand[lh_j[cj]]
                new_load_j = sum(demand[c] for c in lh_j) - demand[lh_j[cj]] + demand[lh_i[ci]]
                
                if new_load_i <= capacity and new_load_j <= capacity:
                    new_lh_i = list(lh_i)
                    new_lh_j = list(lh_j)
                    new_lh_i[ci], new_lh_j[cj] = new_lh_j[cj], new_lh_i[ci]
                    sol[ri] = (new_lh_i, bh_i)
                    sol[rj] = (new_lh_j, bh_j)
            
            elif move_type == 'swap_bh':
                non_empty_bh = [(i, sol[i]) for i in range(len(sol)) if len(sol[i][1]) > 0]
                if len(non_empty_bh) < 2:
                    continue
                (ri, (lh_i, bh_i)), (rj, (lh_j, bh_j)) = random.sample(non_empty_bh, 2)
                ci = random.randint(0, len(bh_i) - 1)
                cj = random.randint(0, len(bh_j) - 1)
                
                new_load_i = sum(demand[c] for c in bh_i) - demand[bh_i[ci]] + demand[bh_j[cj]]
                new_load_j = sum(demand[c] for c in bh_j) - demand[bh_j[cj]] + demand[bh_i[ci]]
                
                if new_load_i <= capacity and new_load_j <= capacity:
                    new_bh_i = list(bh_i)
                    new_bh_j = list(bh_j)
                    new_bh_i[ci], new_bh_j[cj] = new_bh_j[cj], new_bh_i[ci]
                    sol[ri] = (lh_i, new_bh_i)
                    sol[rj] = (lh_j, new_bh_j)
        
        return sol
    
    # ---- Main solving logic ----
    
    # Try multiple construction heuristics
    best_sol = None
    best_cost = float('inf')
    
    # Construction 1: Nearest neighbor
    try:
        sol1 = nearest_neighbor_construction()
        if is_valid_solution(sol1):
            cost1 = solution_cost(sol1)
            if cost1 < best_cost:
                best_cost = cost1
                best_sol = deepcopy(sol1)
                if logger:
                    logger.log(best_cost)
    except:
        pass
    
    # Construction 2: Savings
    try:
        sol2 = savings_construction()
        if is_valid_solution(sol2):
            cost2 = solution_cost(sol2)
            if cost2 < best_cost:
                best_cost = cost2
                best_sol = deepcopy(sol2)
                if logger:
                    logger.log(best_cost)
    except:
        pass
    
    # Multiple random constructions
    for _ in range(5):
        if time.time() - start_time > time_limit * 0.1:
            break
        try:
            # Randomized nearest neighbor
            unvisited_lh = set(linehaul_customers)
            unvisited_bh = set(backhaul_customers)
            routes = []
            
            while unvisited_lh:
                lh_route = []
                bh_route = []
                lh_load = 0
                current = depot_idx
                
                while unvisited_lh:
                    candidates = [n for n in unvisited_lh if lh_load + demand[n] <= capacity]
                    if not candidates:
                        break
                    # Randomized selection from top-k nearest
                    candidates.sort(key=lambda n: dist[current][n])
                    k = min(3, len(candidates))
                    chosen = random.choice(candidates[:k])
                    lh_route.append(chosen)
                    lh_load += demand[chosen]
                    unvisited_lh.remove(chosen)
                    current = chosen
                
                if lh_route:
                    routes.append((lh_route, bh_route))
            
            while len(routes) < num_vehicles:
                routes.append(([], []))
            
            # Assign backhaul
            remaining_bh = list(backhaul_customers)
            random.shuffle(remaining_bh)
            for n in remaining_bh:
                best_route_idx = None
                best_ci = float('inf')
                best_pos = 0
                for i, (lh, bh) in enumerate(routes):
                    if not lh:
                        continue
                    bh_load = sum(demand[b] for b in bh)
                    if bh_load + demand[n] > capacity:
                        continue
                    for pos in range(len(bh) + 1):
                        new_bh = list(bh)
                        new_bh.insert(pos, n)
                        cost_inc = route_cost(build_route(lh, new_bh)) - route_cost(build_route(lh, bh))
                        if cost_inc < best_ci:
                            best_ci = cost_inc
                            best_route_idx = i
                            best_pos = pos
                if best_route_idx is not None:
                    lh, bh = routes[best_route_idx]
                    new_bh = list(bh)
                    new_bh.insert(best_pos, n)
                    routes[best_route_idx] = (lh, new_bh)
            
            if is_valid_solution(routes):
                cost = solution_cost(routes)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = deepcopy(routes)
                    if logger:
                        logger.log(best_cost)
        except:
            pass
    
    if best_sol is None:
        # Fallback: simple construction
        routes = []
        lh_sorted = sorted(linehaul_customers, key=lambda c: demand[c], reverse=True)
        bh_sorted = sorted(backhaul_customers, key=lambda c: demand[c], reverse=True)
        
        # Bin-pack linehaul
        bins_lh = [[] for _ in range(num_vehicles)]
        bin_loads_lh = [0] * num_vehicles
        for c in lh_sorted:
            # First fit decreasing
            placed = False
            for i in range(num_vehicles):
                if bin_loads_lh[i] + demand[c] <= capacity:
                    bins_lh[i].append(c)
                    bin_loads_lh[i] += demand[c]
                    placed = True
                    break
            if not placed:
                bins_lh[0].append(c)
                bin_loads_lh[0] += demand[c]
        
        routes = [(list(b), []) for b in bins_lh]
        
        # Assign backhaul
        for c in bh_sorted:
            for i in range(len(routes)):
                lh, bh = routes[i]
                if not lh:
                    continue
                bh_load = sum(demand[b] for b in bh)
                if bh_load + demand[c] <= capacity:
                    routes[i] = (lh, bh + [c])
                    break
        
        if is_valid_solution(routes):
            best_sol = deepcopy(routes)
            best_cost = solution_cost(best_sol)
            if logger:
                logger.log(best_cost)
    
    # Apply local search to best solution
    if best_sol is not None and time.time() - start_time < time_limit * 0.9:
        best_sol = local_search(best_sol)
        new_cost = solution_cost(best_sol)
        if new_cost < best_cost:
            best_cost = new_cost
            if logger:
                logger.log(best_cost)
    
    # Iterated Local Search
    if best_sol is not None:
        current_sol = deepcopy(best_sol)
        current_cost = best_cost
        no_improve_count = 0
        
        while time.time() - start_time < time_limit * 0.95:
            strength = random.randint(2, max(3, min(num_linehaul + num_backhaul, 8)))
            perturbed = perturbation(current_sol, strength)
            
            if not is_valid_solution(perturbed):
                no_improve_count += 1
                if no_improve_count > 50:
                    current_sol = deepcopy(best_sol)
                    current_cost = best_cost
                    no_improve_count = 0
                continue
            
            if time.time() - start_time > time_limit * 0.95:
                break
            
            perturbed = local_search(perturbed)
            perturbed_cost = solution_cost(perturbed)
            
            # Accept with probability (simulated annealing acceptance)
            if perturbed_cost < current_cost:
                current_sol = perturbed
                current_cost = perturbed_cost
                no_improve_count = 0
            else:
                # Accept worse solutions with some probability
                delta = perturbed_cost - current_cost
                temp = max(1, best_cost * 0.02)
                if random.random() < math.exp(-delta / temp):
                    current_sol = perturbed
                    current_cost = perturbed_cost
                no_improve_count += 1
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_sol = deepcopy(current_sol)
                if logger:
                    logger.log(best_cost)
            
            if no_improve_count > 50:
                current_sol = deepcopy(best_sol)
                current_cost = best_cost
                no_improve_count = 0
    
    # Convert to output format
    output_routes = []
    for lh, bh in best_sol:
        route = build_route(lh, bh)
        output_routes.append(route)
    
    return {
        "objective_value": float(best_cost),
        "routes": output_routes
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    solution = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()