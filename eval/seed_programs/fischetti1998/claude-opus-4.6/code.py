import argparse
import json
import time
import random
import math
from itertools import combinations

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    depot = instance['depot']
    t0 = instance['t0']
    prizes = {}
    for k, v in instance['prizes'].items():
        prizes[int(k)] = v
    
    travel_times = {}
    for k, v in instance['travel_times'].items():
        # Parse key like "(1,2)"
        k = k.strip()
        if k.startswith('(') and k.endswith(')'):
            k = k[1:-1]
        parts = k.split(',')
        i, j = int(parts[0].strip()), int(parts[1].strip())
        travel_times[(i, j)] = v
        travel_times[(j, i)] = v

    cities = list(range(1, n + 1))
    
    def dist(i, j):
        if i == j:
            return 0
        a, b = min(i, j), max(i, j)
        return travel_times.get((a, b), travel_times.get((i, j), float('inf')))

    def tour_cost(tour):
        """Calculate total travel time of a tour (list starting and ending at depot)."""
        total = 0
        for idx in range(len(tour) - 1):
            total += dist(tour[idx], tour[idx + 1])
        return total

    def tour_prize(tour):
        """Calculate total prize of a tour."""
        visited = set(tour)
        return sum(prizes.get(c, 0) for c in visited)

    start_time = time.time()
    
    best_tour = None
    best_prize = 0
    best_cost = 0

    def update_best(tour):
        nonlocal best_tour, best_prize, best_cost
        if tour is None or len(tour) < 4:  # need at least depot + 2 cities + depot
            return
        cost = tour_cost(tour)
        if cost > t0:
            return
        prize = tour_prize(tour)
        if prize > best_prize:
            best_tour = tour[:]
            best_prize = prize
            best_cost = cost
            if logger:
                logger.log(best_prize)

    # For small instances, try MIP approach
    if n <= 60:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining_time = args.time_limit - (time.time() - start_time) - 1
            if remaining_time > 2:
                model = gp.Model()
                model.setParam('TimeLimit', remaining_time)
                model.setParam('OutputFlag', 0)
                model.setParam('Threads', 1)
                
                # Variables
                x = {}  # edge variables
                y = {}  # node visit variables
                
                for c in cities:
                    y[c] = model.addVar(vtype=GRB.BINARY, name=f'y_{c}')
                
                # Depot must be visited
                model.addConstr(y[depot] == 1)
                
                edge_list = []
                for i in cities:
                    for j in cities:
                        if i < j:
                            edge_list.append((i, j))
                            x[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
                
                # Objective: maximize prizes
                model.setObjective(
                    gp.quicksum(prizes.get(c, 0) * y[c] for c in cities),
                    GRB.MAXIMIZE
                )
                
                # Travel time budget
                model.addConstr(
                    gp.quicksum(dist(i, j) * x[(i, j)] for i, j in edge_list) <= t0
                )
                
                # Degree constraints: each visited node has degree 2
                for c in cities:
                    incident = []
                    for i, j in edge_list:
                        if i == c or j == c:
                            incident.append(x[(i, j)])
                    if incident:
                        model.addConstr(gp.quicksum(incident) == 2 * y[c])
                
                # At least 3 cities visited (including depot)
                model.addConstr(gp.quicksum(y[c] for c in cities) >= 3)
                
                # MTZ subtour elimination using auxiliary variables
                # We use flow-based or MTZ formulation
                # MTZ: u[i] gives position in tour, depot is position 0
                u = {}
                non_depot = [c for c in cities if c != depot]
                for c in non_depot:
                    u[c] = model.addVar(lb=1, ub=n-1, vtype=GRB.CONTINUOUS, name=f'u_{c}')
                
                # We need directed edge variables for MTZ
                # Let's use a callback-based approach instead for subtour elimination
                
                # Remove MTZ variables - use lazy constraints instead
                model.remove([u[c] for c in non_depot])
                
                model.setParam('LazyConstraints', 1)
                
                def subtour_callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        x_val = {}
                        y_val = {}
                        for i, j in edge_list:
                            x_val[(i, j)] = model.cbGetSolution(x[(i, j)])
                        for c in cities:
                            y_val[c] = model.cbGetSolution(y[c])
                        
                        # Find connected components among visited nodes
                        visited = [c for c in cities if y_val[c] > 0.5]
                        if len(visited) < 3:
                            return
                        
                        # Build adjacency
                        adj = {c: [] for c in visited}
                        for i, j in edge_list:
                            if x_val[(i, j)] > 0.5:
                                if i in adj:
                                    adj[i].append(j)
                                if j in adj:
                                    adj[j].append(i)
                        
                        # BFS from depot
                        seen = set()
                        queue = [depot]
                        seen.add(depot)
                        while queue:
                            node = queue.pop()
                            for nb in adj.get(node, []):
                                if nb not in seen:
                                    seen.add(nb)
                                    queue.append(nb)
                        
                        # If all visited nodes are connected to depot, no subtour
                        visited_set = set(visited)
                        if seen >= visited_set:
                            return
                        
                        # Find subtours not containing depot
                        remaining = visited_set - seen
                        while remaining:
                            # BFS to find a component
                            start = next(iter(remaining))
                            component = set()
                            q = [start]
                            component.add(start)
                            while q:
                                node = q.pop()
                                for nb in adj.get(node, []):
                                    if nb not in component and nb in remaining:
                                        component.add(nb)
                                        q.append(nb)
                            remaining -= component
                            
                            # Add subtour elimination constraint for this component
                            # Sum of edges within component <= |component| - 1
                            inner_edges = []
                            comp_list = list(component)
                            for idx_a in range(len(comp_list)):
                                for idx_b in range(idx_a + 1, len(comp_list)):
                                    a, b = comp_list[idx_a], comp_list[idx_b]
                                    key = (min(a, b), max(a, b))
                                    if key in x:
                                        inner_edges.append(x[key])
                            
                            if inner_edges:
                                model.cbLazy(
                                    gp.quicksum(inner_edges) <= len(component) - 1
                                )
                
                model.optimize(subtour_callback)
                
                if model.SolCount > 0:
                    # Extract solution
                    selected_edges = []
                    for i, j in edge_list:
                        if x[(i, j)].X > 0.5:
                            selected_edges.append((i, j))
                    
                    visited_nodes = [c for c in cities if y[c].X > 0.5]
                    
                    # Build tour from edges
                    adj = {c: [] for c in visited_nodes}
                    for i, j in selected_edges:
                        adj[i].append(j)
                        adj[j].append(i)
                    
                    # Trace tour starting from depot
                    tour = [depot]
                    current = depot
                    prev = -1
                    if adj[depot]:
                        next_node = adj[depot][0]
                        tour.append(next_node)
                        prev = current
                        current = next_node
                        while current != depot:
                            neighbors = adj[current]
                            next_n = None
                            for nb in neighbors:
                                if nb != prev:
                                    next_n = nb
                                    break
                            if next_n is None:
                                break
                            tour.append(next_n)
                            prev = current
                            current = next_n
                    
                    if len(tour) >= 4 and tour[-1] == depot:
                        update_best(tour)
        except Exception as e:
            pass
    
    # If MIP didn't work or for larger instances, use greedy + local search
    if best_tour is None or n > 60:
        # Greedy construction
        # Sort non-depot cities by prize/nearest_distance ratio
        non_depot = [c for c in cities if c != depot]
        
        # Build nearest neighbor tour greedily by prize density
        def greedy_construct():
            # Start with just the depot
            # Greedily insert the city that gives best prize per added cost
            current_tour = [depot, depot]  # placeholder
            
            # Find the best initial triangle
            best_init_prize = 0
            best_init_tour = None
            
            # Try inserting cities one by one
            candidates = sorted(non_depot, key=lambda c: prizes.get(c, 0), reverse=True)
            
            # Try top candidates for initial triangle
            top_k = min(len(candidates), 50)
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    c1, c2 = candidates[i], candidates[j]
                    cost = dist(depot, c1) + dist(c1, c2) + dist(c2, depot)
                    if cost <= t0:
                        p = prizes.get(c1, 0) + prizes.get(c2, 0)
                        if p > best_init_prize:
                            best_init_prize = p
                            best_init_tour = [depot, c1, c2, depot]
            
            if best_init_tour is None:
                # Try any pair
                for c1 in candidates:
                    for c2 in candidates:
                        if c1 != c2:
                            cost = dist(depot, c1) + dist(c1, c2) + dist(c2, depot)
                            if cost <= t0:
                                p = prizes.get(c1, 0) + prizes.get(c2, 0)
                                if p > best_init_prize:
                                    best_init_prize = p
                                    best_init_tour = [depot, c1, c2, depot]
                    if best_init_tour is not None:
                        break
            
            if best_init_tour is None:
                return None
            
            tour = best_init_tour
            in_tour = set(tour)
            
            # Iteratively insert best city
            improved = True
            while improved and time.time() - start_time < args.time_limit - 2:
                improved = False
                best_ratio = -1
                best_insertion = None
                
                current_cost = tour_cost(tour)
                
                for c in candidates:
                    if c in in_tour:
                        continue
                    p = prizes.get(c, 0)
                    if p == 0:
                        continue
                    
                    # Find best position to insert
                    best_pos_cost = float('inf')
                    best_pos = -1
                    for pos in range(1, len(tour)):
                        added = dist(tour[pos-1], c) + dist(c, tour[pos]) - dist(tour[pos-1], tour[pos])
                        if added < best_pos_cost:
                            best_pos_cost = added
                            best_pos = pos
                    
                    if current_cost + best_pos_cost <= t0:
                        ratio = p / max(best_pos_cost, 0.001)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_insertion = (c, best_pos, best_pos_cost)
                
                if best_insertion is not None:
                    c, pos, added_cost = best_insertion
                    tour.insert(pos, c)
                    in_tour.add(c)
                    improved = True
            
            return tour
        
        greedy_tour = greedy_construct()
        if greedy_tour is not None:
            update_best(greedy_tour)
        
        # Local search improvements
        def two_opt(tour):
            """Apply 2-opt improvements."""
            if tour is None or len(tour) < 5:
                return tour
            improved = True
            while improved and time.time() - start_time < args.time_limit - 1:
                improved = False
                n_tour = len(tour) - 1  # exclude last (duplicate of first)
                for i in range(1, n_tour - 1):
                    for j in range(i + 1, n_tour):
                        if time.time() - start_time >= args.time_limit - 1:
                            return tour
                        # Reverse segment from i to j
                        d1 = dist(tour[i-1], tour[i]) + dist(tour[j], tour[j+1 if j+1 < len(tour) else 0])
                        d2 = dist(tour[i-1], tour[j]) + dist(tour[i], tour[j+1 if j+1 < len(tour) else 0])
                        if d2 < d1:
                            tour[i:j+1] = reversed(tour[i:j+1])
                            improved = True
            return tour
        
        def or_opt(tour):
            """Try moving segments of 1, 2, or 3 cities to better positions."""
            if tour is None or len(tour) < 5:
                return tour
            improved = True
            while improved and time.time() - start_time < args.time_limit - 1:
                improved = False
                n_tour = len(tour) - 1
                for seg_len in [1, 2, 3]:
                    for i in range(1, n_tour - seg_len + 1):
                        if time.time() - start_time >= args.time_limit - 1:
                            return tour
                        # Cost of removing segment [i, i+seg_len-1]
                        seg_end = i + seg_len - 1
                        if seg_end >= n_tour:
                            continue
                        remove_cost = (dist(tour[i-1], tour[i]) + 
                                      dist(tour[seg_end], tour[seg_end + 1]) -
                                      dist(tour[i-1], tour[seg_end + 1]))
                        
                        # Segment internal cost
                        seg_internal = sum(dist(tour[i+k], tour[i+k+1]) for k in range(seg_len - 1))
                        
                        for j in range(1, n_tour):
                            if j >= i and j <= seg_end + 1:
                                continue
                            # Cost of inserting segment between j-1 and j
                            insert_cost = (dist(tour[j-1], tour[i]) + 
                                         dist(tour[seg_end], tour[j]) -
                                         dist(tour[j-1], tour[j]))
                            
                            if insert_cost < remove_cost - 0.001:
                                # Perform the move
                                segment = tour[i:seg_end+1]
                                new_tour = tour[:i] + tour[seg_end+1:]
                                # Find new insertion point
                                if j > seg_end:
                                    new_j = j - seg_len
                                else:
                                    new_j = j
                                new_tour = new_tour[:new_j] + segment + new_tour[new_j:]
                                
                                if tour_cost(new_tour) <= t0:
                                    tour = new_tour
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            return tour
        
        if best_tour is not None:
            improved_tour = two_opt(best_tour[:])
            update_best(improved_tour)
            improved_tour = or_opt(improved_tour)
            update_best(improved_tour)
        
        # Try to add more cities to the tour
        def try_insertions(tour):
            if tour is None:
                return tour
            in_tour = set(tour)
            improved = True
            while improved and time.time() - start_time < args.time_limit - 1:
                improved = False
                current_cost = tour_cost(tour)
                best_ratio = -1
                best_insertion = None
                
                for c in non_depot:
                    if c in in_tour:
                        continue
                    p = prizes.get(c, 0)
                    if p == 0:
                        continue
                    
                    best_pos_cost = float('inf')
                    best_pos = -1
                    for pos in range(1, len(tour)):
                        added = dist(tour[pos-1], c) + dist(c, tour[pos]) - dist(tour[pos-1], tour[pos])
                        if added < best_pos_cost:
                            best_pos_cost = added
                            best_pos = pos
                    
                    if current_cost + best_pos_cost <= t0:
                        ratio = p / max(best_pos_cost, 0.001)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_insertion = (c, best_pos)
                
                if best_insertion is not None:
                    c, pos = best_insertion
                    tour.insert(pos, c)
                    in_tour.add(c)
                    improved = True
            return tour
        
        if best_tour is not None:
            tour = try_insertions(best_tour[:])
            update_best(tour)
            tour = two_opt(tour)
            update_best(tour)
        
        # Swap: try replacing a visited city with an unvisited one
        def try_swaps(tour):
            if tour is None or len(tour) < 4:
                return tour
            improved = True
            while improved and time.time() - start_time < args.time_limit - 1:
                improved = False
                in_tour = set(tour)
                n_tour = len(tour) - 1
                
                for i in range(1, n_tour):
                    if time.time() - start_time >= args.time_limit - 1:
                        return tour
                    old_city = tour[i]
                    old_cost = dist(tour[i-1], old_city) + dist(old_city, tour[i+1])
                    old_prize = prizes.get(old_city, 0)
                    
                    for c in non_depot:
                        if c in in_tour:
                            continue
                        new_cost = dist(tour[i-1], c) + dist(c, tour[i+1])
                        new_prize = prizes.get(c, 0)
                        
                        if new_prize > old_prize and tour_cost(tour) - old_cost + new_cost <= t0:
                            tour[i] = c
                            in_tour.remove(old_city)
                            in_tour.add(c)
                            improved = True
                            break
                        elif new_prize >= old_prize and new_cost < old_cost:
                            tour[i] = c
                            in_tour.remove(old_city)
                            in_tour.add(c)
                            improved = True
                            break
                    if improved:
                        break
            return tour
        
        if best_tour is not None and time.time() - start_time < args.time_limit - 2:
            tour = try_swaps(best_tour[:])
            update_best(tour)
            tour = try_insertions(tour)
            update_best(tour)
            tour = two_opt(tour)
            update_best(tour)
        
        # Perturbation + local search (iterated local search)
        def perturb(tour):
            if tour is None or len(tour) < 6:
                return tour
            tour = tour[:]
            n_tour = len(tour) - 1
            # Double bridge or random removal+reinsertion
            # Remove a random city and try to add another
            if n_tour <= 3:
                return tour
            
            # Remove 1-2 random cities
            removable = list(range(1, n_tour))
            if len(removable) < 1:
                return tour
            
            num_remove = min(random.randint(1, 3), len(removable))
            to_remove = sorted(random.sample(removable, num_remove), reverse=True)
            removed_cities = []
            for idx in to_remove:
                removed_cities.append(tour[idx])
                tour.pop(idx)
            
            if len(tour) < 3:
                # Need at least depot + 1 city + depot; try to keep valid
                return None
            
            return tour
        
        iterations = 0
        while time.time() - start_time < args.time_limit - 2:
            iterations += 1
            if best_tour is None:
                break
            
            ptour = perturb(best_tour[:])
            if ptour is None or len(ptour) < 4:
                continue
            
            if tour_cost(ptour) > t0:
                # Try 2-opt to fix
                ptour = two_opt(ptour)
            
            if tour_cost(ptour) <= t0:
                ptour = try_insertions(ptour)
                ptour = two_opt(ptour)
                ptour = try_swaps(ptour)
                ptour = try_insertions(ptour)
                update_best(ptour)

    # Format output
    if best_tour is not None and len(best_tour) >= 4:
        visited_nodes = list(set(best_tour))
        edges = []
        for i in range(len(best_tour) - 1):
            edges.append([best_tour[i], best_tour[i+1]])
        
        solution = {
            "objective_value": float(best_prize),
            "visited_nodes": visited_nodes,
            "edges": edges,
            "tour": best_tour
        }
    else:
        solution = {
            "objective_value": 0.0,
            "visited_nodes": [],
            "edges": [],
            "tour": []
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    solve()