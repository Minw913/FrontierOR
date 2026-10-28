import json
import argparse
import time
import random
import math
import sys

from solution_logger import SolutionLogger

def load_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def compute_tour_distance(tour, dist):
    """Compute total distance of a tour (list of node indices forming a cycle)."""
    total = 0
    for i in range(len(tour)):
        total += dist[tour[i]][tour[(i+1) % len(tour)]]
    return total

def compute_tour_score(tour, scores):
    return sum(scores[v] for v in tour)

def extract_edges(tour):
    """Extract edges from a tour (cyclic order)."""
    edges = []
    for i in range(len(tour)):
        edges.append([tour[i], tour[(i+1) % len(tour)]])
    return edges

def make_solution(tour, scores, dist):
    obj = compute_tour_score(tour, scores)
    edges = extract_edges(tour)
    return {
        "objective_value": float(obj),
        "visited_vertices": list(tour),
        "tour_edges": edges
    }

def greedy_construct(data, strategy='ratio'):
    """Greedy construction heuristic for orienteering problem."""
    n = data['num_nodes']
    d0 = data['distance_limitation_d0']
    depot = data['depot_vertex']
    scores = data['scores']
    dist = data['distance_matrix']
    
    tour = [depot]
    unvisited = set(range(n)) - {depot}
    
    while unvisited:
        best_node = None
        best_pos = None
        best_value = -float('inf')
        
        current_dist = compute_tour_distance(tour, dist)
        
        for node in unvisited:
            for pos in range(len(tour)):
                i = tour[pos]
                j = tour[(pos + 1) % len(tour)]
                added_dist = dist[i][node] + dist[node][j] - dist[i][j]
                new_dist = current_dist + added_dist
                
                if new_dist <= d0:
                    if strategy == 'ratio':
                        value = scores[node] / max(added_dist, 1)
                    elif strategy == 'score':
                        value = scores[node]
                    elif strategy == 'min_dist':
                        value = -added_dist
                    else:
                        value = scores[node] / max(added_dist, 1)
                    
                    if value > best_value:
                        best_value = value
                        best_node = node
                        best_pos = pos + 1
        
        if best_node is None:
            break
        
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
    
    return tour

def two_opt(tour, dist, d0):
    """Apply 2-opt improvement."""
    improved = True
    while improved:
        improved = False
        n = len(tour)
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                old_dist = compute_tour_distance(tour, dist)
                new_dist = compute_tour_distance(new_tour, dist)
                if new_dist <= d0 and new_dist < old_dist:
                    tour = new_tour
                    improved = True
                    break
            if improved:
                break
    return tour

def try_insert_unvisited(tour, data):
    """Try to insert unvisited nodes into the tour."""
    n = data['num_nodes']
    d0 = data['distance_limitation_d0']
    depot = data['depot_vertex']
    scores = data['scores']
    dist = data['distance_matrix']
    
    visited = set(tour)
    unvisited = [v for v in range(n) if v not in visited]
    
    current_dist = compute_tour_distance(tour, dist)
    
    for node in sorted(unvisited, key=lambda v: scores[v], reverse=True):
        best_pos = None
        best_added = float('inf')
        
        for pos in range(len(tour)):
            i = tour[pos]
            j = tour[(pos + 1) % len(tour)]
            added = dist[i][node] + dist[node][j] - dist[i][j]
            if current_dist + added <= d0 and added < best_added:
                best_added = added
                best_pos = pos + 1
        
        if best_pos is not None:
            tour.insert(best_pos, node)
            current_dist += best_added
    
    return tour

def try_remove_and_reinsert(tour, data):
    """Try removing low-score nodes and reinserting high-score ones."""
    n = data['num_nodes']
    d0 = data['distance_limitation_d0']
    depot = data['depot_vertex']
    scores = data['scores']
    dist = data['distance_matrix']
    
    improved = True
    while improved:
        improved = False
        visited = set(tour)
        unvisited = [v for v in range(n) if v not in visited and v != depot]
        
        if not unvisited:
            break
        
        for unode in sorted(unvisited, key=lambda v: scores[v], reverse=True):
            for idx in range(1, len(tour)):
                vnode = tour[idx]
                if scores[unode] > scores[vnode]:
                    new_tour = tour[:idx] + [unode] + tour[idx+1:]
                    if compute_tour_distance(new_tour, dist) <= d0:
                        tour = new_tour
                        improved = True
                        break
            if improved:
                break
    
    return tour

def local_search(tour, data):
    """Apply comprehensive local search."""
    dist = data['distance_matrix']
    d0 = data['distance_limitation_d0']
    scores = data['scores']
    
    tour = two_opt(tour, dist, d0)
    tour = try_insert_unvisited(tour, data)
    tour = two_opt(tour, dist, d0)
    tour = try_remove_and_reinsert(tour, data)
    tour = two_opt(tour, dist, d0)
    tour = try_insert_unvisited(tour, data)
    
    return tour

def perturbation(tour, data, strength=3):
    """Perturb the tour by removing random nodes."""
    depot = data['depot_vertex']
    
    removable = [i for i in range(1, len(tour)) if tour[i] != depot]
    if len(removable) <= strength:
        strength = max(1, len(removable) // 2)
    
    to_remove = set(random.sample(removable, min(strength, len(removable))))
    new_tour = [tour[i] for i in range(len(tour)) if i not in to_remove]
    
    new_tour = try_insert_unvisited(new_tour, data)
    
    return new_tour

def iterated_local_search(data, time_limit, logger=None):
    """Iterated Local Search metaheuristic."""
    start_time = time.time()
    n = data['num_nodes']
    scores = data['scores']
    dist = data['distance_matrix']
    
    best_tour = greedy_construct(data, strategy='ratio')
    best_tour = local_search(best_tour, data)
    best_score = compute_tour_score(best_tour, scores)
    
    if logger:
        logger.log_solution(best_score, make_solution(best_tour, scores, dist))
    
    current_tour = list(best_tour)
    current_score = best_score
    
    no_improve = 0
    
    while time.time() - start_time < time_limit - 0.5:
        strength = min(3 + no_improve // 10, max(2, len(current_tour) // 4))
        
        candidate_tour = perturbation(current_tour, data, strength)
        candidate_tour = local_search(candidate_tour, data)
        candidate_score = compute_tour_score(candidate_tour, scores)
        
        if candidate_score > current_score:
            current_tour = candidate_tour
            current_score = candidate_score
            no_improve = 0
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_tour = list(candidate_tour)
                if logger:
                    logger.log_solution(best_score, make_solution(best_tour, scores, dist))
        else:
            no_improve += 1
            if random.random() < 0.1:
                current_tour = candidate_tour
                current_score = candidate_score
    
    return best_tour, best_score

def reconstruct_tour_from_edges(visited, edges, depot):
    """Reconstruct a tour (ordered list) from a set of directed edges."""
    if not edges:
        return [depot]
    
    adj = {}
    for i, j in edges:
        if i not in adj:
            adj[i] = []
        adj[i].append(j)
    
    tour = [depot]
    current = depot
    visited_set = {depot}
    
    while True:
        neighbors = adj.get(current, [])
        next_node = None
        for nb in neighbors:
            if nb not in visited_set:
                next_node = nb
                break
        
        if next_node is None:
            break
        
        tour.append(next_node)
        visited_set.add(next_node)
        current = next_node
    
    if len(tour) == len(visited):
        return tour
    return None

def reconstruct_tour_undirected(visited, edges, depot):
    """Reconstruct tour from undirected edges."""
    if not edges:
        return [depot]
    
    adj = {v: [] for v in visited}
    for i, j in edges:
        if i in adj:
            adj[i].append(j)
        if j in adj:
            adj[j].append(i)
    
    tour = [depot]
    visited_set = {depot}
    current = depot
    
    while True:
        next_node = None
        for nb in adj.get(current, []):
            if nb not in visited_set:
                next_node = nb
                break
        
        if next_node is None:
            break
        
        tour.append(next_node)
        visited_set.add(next_node)
        current = next_node
    
    if len(tour) == len(visited):
        return tour
    return None

def solve_mip(data, time_limit, logger=None):
    """Solve using Gurobi MIP formulation for small instances."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    
    n = data['num_nodes']
    d0 = data['distance_limitation_d0']
    depot = data['depot_vertex']
    scores = data['scores']
    dist = data['distance_matrix']
    
    model = gp.Model("orienteering")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    
    # Binary variables
    y = {}
    for i in range(n):
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
    
    # x[i,j] = 1 if directed arc (i,j) is in the tour
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # MTZ position variables
    u = {}
    for i in range(n):
        if i != depot:
            u[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=n-1, name=f"u_{i}")
    
    model.update()
    
    # Objective: maximize total score
    model.setObjective(gp.quicksum(scores[i] * y[i] for i in range(n)), GRB.MAXIMIZE)
    
    # Depot is always visited
    model.addConstr(y[depot] == 1)
    
    # Flow conservation
    for i in range(n):
        model.addConstr(gp.quicksum(x[j,i] for j in range(n) if j != i) == y[i], name=f"in_{i}")
        model.addConstr(gp.quicksum(x[i,j] for j in range(n) if j != i) == y[i], name=f"out_{i}")
    
    # Distance constraint
    model.addConstr(
        gp.quicksum(dist[i][j] * x[i,j] for i in range(n) for j in range(n) if i != j) <= d0,
        name="distance"
    )
    
    # MTZ subtour elimination constraints
    for i in range(n):
        if i != depot:
            for j in range(n):
                if j != depot and j != i:
                    model.addConstr(u[i] - u[j] + (n - 1) * x[i,j] <= n - 2, name=f"mtz_{i}_{j}")
    
    model.optimize()
    
    if model.SolCount > 0:
        y_vals = {}
        x_vals = {}
        for i in range(n):
            y_vals[i] = y[i].X
        for i in range(n):
            for j in range(n):
                if i != j:
                    x_vals[i,j] = x[i,j].X
        
        visited = [i for i in range(n) if y_vals[i] > 0.5]
        edges = [(i,j) for i in range(n) for j in range(n) if i != j and x_vals[i,j] > 0.5]
        
        tour = reconstruct_tour_from_edges(visited, edges, depot)
        if tour:
            obj = compute_tour_score(tour, scores)
            if compute_tour_distance(tour, dist) <= d0:
                if logger:
                    logger.log_solution(obj, make_solution(tour, scores, dist))
                return tour, obj
    
    return None

def solve_with_gurobi_large(data, time_limit, logger=None):
    """Use Gurobi with lazy constraint callback for medium instances."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    
    n = data['num_nodes']
    d0 = data['distance_limitation_d0']
    depot = data['depot_vertex']
    scores = data['scores']
    dist = data['distance_matrix']
    
    model = gp.Model("orienteering")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    model.setParam('LazyConstraints', 1)
    
    # y[i] = 1 if node i is visited
    y = {}
    for i in range(n):
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
    
    # x[i,j] for undirected edges (i < j)
    x = {}
    for i in range(n):
        for j in range(i+1, n):
            x[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    model.update()
    
    # Objective
    model.setObjective(gp.quicksum(scores[i] * y[i] for i in range(n)), GRB.MAXIMIZE)
    
    # Depot always visited
    model.addConstr(y[depot] == 1)
    
    # Degree constraints
    for i in range(n):
        edges_i = []
        for j in range(n):
            if j < i:
                edges_i.append(x[j,i])
            elif j > i:
                edges_i.append(x[i,j])
        model.addConstr(gp.quicksum(edges_i) == 2 * y[i])
    
    # Distance constraint
    model.addConstr(
        gp.quicksum(dist[i][j] * x[i,j] for i in range(n) for j in range(i+1, n)) <= d0
    )
    
    # Subtour elimination using lazy constraints
    def subtour_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            x_vals = model.cbGetSolution(x)
            y_vals = model.cbGetSolution(y)
            
            visited = [i for i in range(n) if y_vals[i] > 0.5]
            if len(visited) <= 2:
                return
            
            adj = {v: [] for v in visited}
            for i in range(n):
                for j in range(i+1, n):
                    if x_vals[i,j] > 0.5:
                        if i in adj and j in adj:
                            adj[i].append(j)
                            adj[j].append(i)
            
            found_component = set()
            for start in visited:
                if start not in found_component:
                    component = set()
                    stack = [start]
                    while stack:
                        node = stack.pop()
                        if node not in component:
                            component.add(node)
                            for nb in adj[node]:
                                if nb not in component:
                                    stack.append(nb)
                    found_component.update(component)
                    
                    if len(component) < len(visited):
                        edges_in_component = []
                        for i in component:
                            for j in component:
                                if i < j:
                                    edges_in_component.append(x[i,j])
                        model.cbLazy(
                            gp.quicksum(edges_in_component) <= len(component) - 1
                        )
    
    model.optimize(subtour_callback)
    
    if model.SolCount > 0:
        y_vals = {i: y[i].X for i in range(n)}
        x_vals = {}
        for i in range(n):
            for j in range(i+1, n):
                x_vals[i,j] = x[i,j].X
        
        visited = [i for i in range(n) if y_vals[i] > 0.5]
        edges = [(i,j) for i in range(n) for j in range(i+1, n) if x_vals[i,j] > 0.5]
        
        tour = reconstruct_tour_undirected(visited, edges, depot)
        if tour:
            obj = compute_tour_score(tour, scores)
            if compute_tour_distance(tour, dist) <= d0:
                if logger:
                    logger.log_solution(obj, make_solution(tour, scores, dist))
                return tour, obj
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Orienteering Problem Solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    data = load_instance(args.instance_path)
    n = data['num_nodes']
    d0 = data['distance_limitation_d0']
    depot = data['depot_vertex']
    scores = data['scores']
    dist = data['distance_matrix']
    
    start_time = time.time()
    
    best_tour = None
    best_score = 0
    
    # Phase 1: Quick heuristic to get initial solution
    random.seed(0)
    
    for strategy in ['ratio', 'score', 'min_dist']:
        if time.time() - start_time > args.time_limit - 1:
            break
        tour = greedy_construct(data, strategy=strategy)
        tour = local_search(tour, data)
        score = compute_tour_score(tour, scores)
        if score > best_score:
            best_score = score
            best_tour = list(tour)
            if logger:
                logger.log_solution(best_score, make_solution(best_tour, scores, dist))
    
    remaining_time = args.time_limit - (time.time() - start_time)
    
    # Phase 2: For small instances, try MIP
    if n <= 25 and remaining_time > 5:
        mip_time = min(remaining_time * 0.6, remaining_time - 2)
        result = solve_mip(data, mip_time, logger)
        if result is not None:
            tour, score = result
            if score > best_score:
                best_score = score
                best_tour = tour
    elif n <= 60 and remaining_time > 10:
        mip_time = min(remaining_time * 0.5, remaining_time - 3)
        result = solve_with_gurobi_large(data, mip_time, logger)
        if result is not None:
            tour, score = result
            if score > best_score:
                best_score = score
                best_tour = tour
    
    remaining_time = args.time_limit - (time.time() - start_time)
    
    # Phase 3: Iterated Local Search for remaining time
    if remaining_time > 2:
        ils_tour, ils_score = iterated_local_search(data, remaining_time, logger)
        if ils_score > best_score:
            best_score = ils_score
            best_tour = ils_tour
    
    # Fallback: if no solution found, use depot only
    if best_tour is None:
        best_tour = [depot]
        best_score = 0
    
    # Final local search
    best_tour = local_search(best_tour, data)
    final_score = compute_tour_score(best_tour, scores)
    if final_score > best_score:
        best_score = final_score
        if logger:
            logger.log_solution(best_score, make_solution(best_tour, scores, dist))
    
    # Write solution
    solution = make_solution(best_tour, scores, dist)
    solution["objective_value"] = float(best_score)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    
    print(f"Objective value: {best_score}")
    print(f"Tour length: {len(best_tour)} nodes")
    print(f"Tour distance: {compute_tour_distance(best_tour, dist)}")

if __name__ == '__main__':
    main()