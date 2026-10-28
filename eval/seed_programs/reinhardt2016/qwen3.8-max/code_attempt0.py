import json
import argparse
import time
import math
import random
import sys
from copy import deepcopy
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    time_limit = args.time_limit

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_customers = instance['num_customers']
    num_vehicles = instance['num_vehicles']
    vehicle_capacity = instance['vehicle_capacity']
    num_edge_sets = instance['num_edge_sets']
    schedule_horizon = instance['schedule_horizon']
    nodes = instance['nodes']
    edge_sets_data = instance['edge_sets']
    free_edges_list = instance['free_edges']
    dist_matrix = instance['distance_matrix']

    n = num_customers + 1
    random.seed(0)

    # Node info
    demands = [nodes[i]['demand'] for i in range(n)]
    tw_start = [nodes[i]['time_window_start'] for i in range(n)]
    tw_end = [nodes[i]['time_window_end'] for i in range(n)]
    service_time = [nodes[i]['service_time'] for i in range(n)]

    # Edge classification
    edge_to_set = {}
    for es in edge_sets_data:
        for edge in es['edges']:
            key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
            edge_to_set[key] = es['id']

    edge_set_costs = {}
    for es in edge_sets_data:
        edge_set_costs[es['id']] = es['access_cost']

    def get_edge_set_id(i, j):
        key = (min(i, j), max(i, j))
        return edge_to_set.get(key, None)

    def compute_solution(routes):
        """Compute total cost and used edge sets for a set of routes."""
        total_traversal = 0.0
        all_used_sets = set()
        for route in routes:
            for k in range(len(route) - 1):
                i, j = route[k], route[k + 1]
                total_traversal += dist_matrix[i][j]
                es = get_edge_set_id(i, j)
                if es is not None:
                    all_used_sets.add(es)
        access_cost = sum(edge_set_costs[s] for s in all_used_sets)
        return total_traversal + access_cost, all_used_sets

    def get_route_demand(route):
        return sum(demands[node] for node in route[1:-1])

    def get_visit_times(route):
        times = {}
        current_time = tw_start[0]
        times[route[0]] = current_time
        for k in range(1, len(route)):
            prev_node = route[k - 1]
            curr_node = route[k]
            arrival = current_time + dist_matrix[prev_node][curr_node] + service_time[prev_node]
            visit_time = max(arrival, tw_start[curr_node])
            times[curr_node] = visit_time
            current_time = visit_time
        return times

    def check_route_time_feasibility(route):
        if len(route) <= 2:
            return True
        current_time = tw_start[0]
        for k in range(1, len(route)):
            prev_node = route[k - 1]
            curr_node = route[k]
            arrival = current_time + dist_matrix[prev_node][curr_node] + service_time[prev_node]
            visit_time = max(arrival, tw_start[curr_node])
            if visit_time > tw_end[curr_node]:
                return False
            current_time = visit_time
        return True

    def check_insertion_feasible(route, pos, customer):
        """Check if inserting customer at position pos in route is feasible."""
        new_route = route[:pos] + [customer] + route[pos:]
        if get_route_demand(new_route) > vehicle_capacity:
            return False
        return check_route_time_feasibility(new_route)

    # ==================== CONSTRUCTIVE HEURISTIC ====================
    def construct_nearest_neighbor():
        """Build routes using nearest neighbor heuristic."""
        unvisited = set(range(1, n))
        routes = []
        
        while unvisited and len(routes) < num_vehicles:
            route = [0]
            current_demand = 0
            current_node = 0
            current_time = tw_start[0]
            
            while True:
                best_next = None
                best_cost = float('inf')
                
                for c in unvisited:
                    if current_demand + demands[c] > vehicle_capacity:
                        continue
                    arrival = current_time + dist_matrix[current_node][c] + service_time[current_node]
                    visit = max(arrival, tw_start[c])
                    if visit > tw_end[c]:
                        continue
                    # Consider edge set cost
                    es = get_edge_set_id(current_node, c)
                    cost = dist_matrix[current_node][c]
                    best_next_candidate = c
                    if cost < best_cost:
                        best_cost = cost
                        best_next = c
                
                if best_next is None:
                    break
                
                arrival = current_time + dist_matrix[current_node][best_next] + service_time[current_node]
                visit = max(arrival, tw_start[best_next])
                route.append(best_next)
                current_demand += demands[best_next]
                current_time = visit
                current_node = best_next
                unvisited.remove(best_next)
            
            route.append(0)
            routes.append(route)
        
        # Handle remaining unvisited (shouldn't happen if feasible)
        while unvisited and len(routes) < num_vehicles:
            c = unvisited.pop()
            routes.append([0, c, 0])
        
        return routes

    def construct_cheapest_insertion():
        """Build routes using cheapest insertion heuristic."""
        unvisited = list(range(1, n))
        random.shuffle(unvisited)
        routes = [[0, 0] for _ in range(min(num_vehicles, num_customers))]
        
        # Sort customers by demand (descending) or time window tightness
        unvisited.sort(key=lambda c: demands[c], reverse=True)
        
        for customer in unvisited:
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            
            for r_idx, route in enumerate(routes):
                if get_route_demand(route) + demands[customer] > vehicle_capacity:
                    continue
                for pos in range(1, len(route)):
                    if check_insertion_feasible(route, pos, customer):
                        # Cost of insertion
                        prev_node = route[pos - 1]
                        next_node = route[pos]
                        cost_increase = (dist_matrix[prev_node][customer] + 
                                        dist_matrix[customer][next_node] - 
                                        dist_matrix[prev_node][next_node])
                        if cost_increase < best_cost:
                            best_cost = cost_increase
                            best_route_idx = r_idx
                            best_pos = pos
            
            if best_route_idx >= 0:
                routes[best_route_idx].insert(best_pos, customer)
            else:
                # Try to create a new route
                if len(routes) < num_vehicles:
                    routes.append([0, customer, 0])
                else:
                    # Force insert into route with most slack
                    for r_idx, route in enumerate(routes):
                        for pos in range(1, len(route)):
                            if check_insertion_feasible(route, pos, customer):
                                routes[r_idx].insert(pos, customer)
                                break
                        else:
                            continue
                        break
        
        # Remove empty routes
        routes = [r for r in routes if len(r) > 2]
        return routes

    # ==================== LOCAL SEARCH ====================
    def intra_route_2opt(routes):
        """Apply 2-opt within each route."""
        improved = True
        while improved:
            improved = False
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                if len(route) <= 4:
                    continue
                best_improvement = 0
                best_i, best_j = -1, -1
                for i in range(1, len(route) - 2):
                    for j in range(i + 2, len(route) - 1):
                        new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        if not check_route_time_feasibility(new_route):
                            continue
                        old_cost = (dist_matrix[route[i-1]][route[i]] + 
                                   dist_matrix[route[j]][route[j+1]])
                        new_cost = (dist_matrix[route[i-1]][route[j]] + 
                                   dist_matrix[route[i]][route[j+1]])
                        improvement = old_cost - new_cost
                        if improvement > best_improvement + 1e-10:
                            best_improvement = improvement
                            best_i, best_j = i, j
                if best_i >= 0:
                    routes[r_idx] = route[:best_i] + route[best_i:best_j + 1][::-1] + route[best_j + 1:]
                    improved = True
        return routes

    def or_opt(routes):
        """Move segments of 1-3 customers within or between routes."""
        improved = True
        while improved:
            improved = False
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                if len(route) <= 3:
                    continue
                for seg_len in [1, 2, 3]:
                    for i in range(1, len(route) - seg_len):
                        segment = route[i:i + seg_len]
                        remaining = route[:i] + route[i + seg_len:]
                        
                        # Try reinserting in same route
                        for pos in range(1, len(remaining)):
                            new_route = remaining[:pos] + segment + remaining[pos:]
                            if get_route_demand(new_route) > vehicle_capacity:
                                continue
                            if not check_route_time_feasibility(new_route):
                                continue
                            old_cost = sum(dist_matrix[route[k]][route[k+1]] for k in range(len(route)-1))
                            new_cost = sum(dist_matrix[new_route[k]][new_route[k+1]] for k in range(len(new_route)-1))
                            if new_cost < old_cost - 1e-10:
                                routes[r_idx] = new_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def relocate(routes):
        """Move single customer from one route to another."""
        improved = True
        while improved:
            improved = False
            for r1 in range(len(routes)):
                for i in range(1, len(routes[r1]) - 1):
                    customer = routes[r1][i]
                    route_without = routes[r1][:i] + routes[r1][i+1:]
                    
                    for r2 in range(len(routes)):
                        if r2 == r1:
                            continue
                        if get_route_demand(routes[r2]) + demands[customer] > vehicle_capacity:
                            continue
                        for pos in range(1, len(routes[r2])):
                            new_route = routes[r2][:pos] + [customer] + routes[r2][pos:]
                            if not check_route_time_feasibility(new_route):
                                continue
                            # Evaluate total cost change
                            old_cost_r1 = sum(dist_matrix[routes[r1][k]][routes[r1][k+1]] for k in range(len(routes[r1])-1))
                            new_cost_r1 = sum(dist_matrix[route_without[k]][route_without[k+1]] for k in range(len(route_without)-1))
                            old_cost_r2 = sum(dist_matrix[routes[r2][k]][routes[r2][k+1]] for k in range(len(routes[r2])-1))
                            new_cost_r2 = sum(dist_matrix[new_route[k]][new_route[k+1]] for k in range(len(new_route)-1))
                            
                            # Also consider edge set cost change
                            old_sets = set()
                            for k in range(len(routes[r1])-1):
                                es = get_edge_set_id(routes[r1][k], routes[r1][k+1])
                                if es: old_sets.add(es)
                            for k in range(len(routes[r2])-1):
                                es = get_edge_set_id(routes[r2][k], routes[r2][k+1])
                                if es: old_sets.add(es)
                            
                            new_sets = set()
                            for k in range(len(route_without)-1):
                                es = get_edge_set_id(route_without[k], route_without[k+1])
                                if es: new_sets.add(es)
                            for k in range(len(new_route)-1):
                                es = get_edge_set_id(new_route[k], new_route[k+1])
                                if es: new_sets.add(es)
                            
                            old_total = (old_cost_r1 + old_cost_r2 + 
                                        sum(edge_set_costs[s] for s in old_sets))
                            new_total = (new_cost_r1 + new_cost_r2 + 
                                        sum(edge_set_costs[s] for s in new_sets))
                            
                            if new_total < old_total - 1e-10:
                                routes[r1] = route_without
                                routes[r2] = new_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            # Remove empty routes
            routes = [r for r in routes if len(r) > 2]
        return routes

    def swap_customers(routes):
        """Swap customers between routes."""
        improved = True
        while improved:
            improved = False
            for r1 in range(len(routes)):
                for i in range(1, len(routes[r1]) - 1):
                    for r2 in range(r1 + 1, len(routes)):
                        for j in range(1, len(routes[r2]) - 1):
                            c1 = routes[r1][i]
                            c2 = routes[r2][j]
                            
                            new_r1 = routes[r1][:i] + [c2] + routes[r1][i+1:]
                            new_r2 = routes[r2][:j] + [c1] + routes[r2][j+1:]
                            
                            if get_route_demand(new_r1) > vehicle_capacity:
                                continue
                            if get_route_demand(new_r2) > vehicle_capacity:
                                continue
                            if not check_route_time_feasibility(new_r1):
                                continue
                            if not check_route_time_feasibility(new_r2):
                                continue
                            
                            old_cost = (sum(dist_matrix[routes[r1][k]][routes[r1][k+1]] for k in range(len(routes[r1])-1)) +
                                       sum(dist_matrix[routes[r2][k]][routes[r2][k+1]] for k in range(len(routes[r2])-1)))
                            new_cost = (sum(dist_matrix[new_r1[k]][new_r1[k+1]] for k in range(len(new_r1)-1)) +
                                       sum(dist_matrix[new_r2[k]][new_r2[k+1]] for k in range(len(new_r2)-1)))
                            
                            if new_cost < old_cost - 1e-10:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def local_search(routes, max_time=None):
        """Apply all local search operators iteratively."""
        if max_time is None:
            max_time = time_limit
        
        best_routes = deepcopy(routes)
        best_cost, _ = compute_solution(best_routes)
        
        iterations = 0
        while time.time() - start_time < max_time - 5:
            iterations += 1
            old_cost = best_cost
            
            routes = intra_route_2opt(routes)
            routes = or_opt(routes)
            routes = relocate(routes)
            routes = swap_customers(routes)
            
            current_cost, _ = compute_solution(routes)
            if current_cost < best_cost - 1e-10:
                best_cost = current_cost
                best_routes = deepcopy(routes)
            
            if abs(current_cost - old_cost) < 1e-10:
                break
        
        return best_routes

    # ==================== EDGE SET OPTIMIZATION ====================
    def optimize_edge_sets(routes):
        """Try to reduce edge set usage by rerouting."""
        best_routes = deepcopy(routes)
        best_cost, best_sets = compute_solution(best_routes)
        
        # Find expensive edge sets that are used
        for es_id in sorted(best_sets, key=lambda s: edge_set_costs[s], reverse=True):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Find edges in this set that are used
            es_edges = set()
            for es in edge_sets_data:
                if es['id'] == es_id:
                    for edge in es['edges']:
                        es_edges.add((min(edge[0], edge[1]), max(edge[0], edge[1])))
            
            # Try to avoid using these edges
            for r_idx in range(len(best_routes)):
                route = best_routes[r_idx]
                modified = False
                for k in range(len(route) - 1):
                    key = (min(route[k], route[k+1]), max(route[k], route[k+1]))
                    if key in es_edges:
                        # Try 2-opt to avoid this edge
                        for i in range(1, len(route) - 2):
                            for j in range(i + 2, len(route) - 1):
                                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                                if not check_route_time_feasibility(new_route):
                                    continue
                                # Check if this avoids the edge set
                                new_es_used = set()
                                for m in range(len(new_route) - 1):
                                    ek = (min(new_route[m], new_route[m+1]), max(new_route[m], new_route[m+1]))
                                    if ek in es_edges:
                                        new_es_used.add(es_id)
                                if len(new_es_used) == 0:
                                    # This avoids the edge set
                                    test_routes = deepcopy(best_routes)
                                    test_routes[r_idx] = new_route
                                    test_cost, _ = compute_solution(test_routes)
                                    if test_cost < best_cost:
                                        best_routes = test_routes
                                        best_cost = test_cost
                                        route = new_route
                                        modified = True
                                        break
                            if modified:
                                break
                    if modified:
                        break
        
        return best_routes

    # ==================== GUROBI MIP ====================
    def solve_gurobi():
        """Solve using Gurobi MIP for small instances."""
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return None
        
        if num_customers > 25:
            return None
        
        remaining = time_limit - (time.time() - start_time)
        if remaining < 30:
            return None
        
        try:
            model = gp.Model("VRPTW_EdgeSets")
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('TimeLimit', max(1, remaining - 5))
            model.setParam('OutputFlag', 0)
            
            V = min(num_vehicles, num_customers)
            
            # Decision variables
            x = {}
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    for v in range(V):
                        x[i, j, v] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{v}")
            
            y = {}
            for es in edge_sets_data:
                y[es['id']] = model.addVar(vtype=GRB.BINARY, name=f"y_{es['id']}")
            
            t = {}
            for i in range(n):
                for v in range(V):
                    t[i, v] = model.addVar(lb=tw_start[i], ub=tw_end[i], 
                                          vtype=GRB.CONTINUOUS, name=f"t_{i}_{v}")
            
            model.update()
            
            # Objective
            obj = gp.LinExpr()
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    for v in range(V):
                        obj += dist_matrix[i][j] * x[i, j, v]
            for es in edge_sets_data:
                obj += es['access_cost'] * y[es['id']]
            model.setObjective(obj, GRB.MINIMIZE)
            
            # Each customer visited exactly once
            for j in range(1, n):
                model.addConstr(
                    gp.quicksum(x[i, j, v] for i in range(n) if i != j for v in range(V)) == 1,
                    name=f"visit_{j}"
                )
            
            # Flow conservation
            for j in range(1, n):
                for v in range(V):
                    model.addConstr(
                        gp.quicksum(x[i, j, v] for i in range(n) if i != j) ==
                        gp.quicksum(x[j, k, v] for k in range(n) if k != j),
                        name=f"flow_{j}_{v}"
                    )
            
            # Vehicle leaves depot at most once
            for v in range(V):
                model.addConstr(
                    gp.quicksum(x[0, j, v] for j in range(1, n)) <= 1,
                    name=f"depot_out_{v}"
                )
                model.addConstr(
                    gp.quicksum(x[j, 0, v] for j in range(1, n)) ==
                    gp.quicksum(x[0, j, v] for j in range(1, n)),
                    name=f"depot_balance_{v}"
                )
            
            # Capacity
            for v in range(V):
                model.addConstr(
                    gp.quicksum(demands[j] * gp.quicksum(x[i, j, v] for i in range(n) if i != j) 
                               for j in range(1, n)) <= vehicle_capacity,
                    name=f"cap_{v}"
                )
            
            # Time consistency
            M = schedule_horizon + max(service_time) + max(max(row) for row in dist_matrix)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    for v in range(V):
                        model.addConstr(
                            t[j, v] >= t[i, v] + dist_matrix[i][j] + service_time[i] - M * (1 - x[i, j, v]),
                            name=f"time_{i}_{j}_{v}"
                        )
            
            # Edge set constraints
            for es in edge_sets_data:
                es_id = es['id']
                for edge in es['edges']:
                    i, j = edge[0], edge[1]
                    for v in range(V):
                        model.addConstr(x[i, j, v] <= y[es_id], name=f"es_{es_id}_{i}_{j}_{v}_fwd")
                        model.addConstr(x[j, i, v] <= y[es_id], name=f"es_{es_id}_{j}_{i}_{v}_rev")
            
            # Symmetry breaking: vehicle v can only be used if vehicle v-1 is used
            for v in range(1, V):
                model.addConstr(
                    gp.quicksum(x[0, j, v] for j in range(1, n)) <=
                    gp.quicksum(x[0, j, v-1] for j in range(1, n)),
                    name=f"sym_{v}"
                )
            
            model.update()
            model.optimize()
            
            if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
                if model.SolCount > 0:
                    # Extract solution
                    routes = []
                    for v in range(V):
                        if gp.quicksum(x[0, j, v].X for j in range(1, n)) > 0.5:
                            route = [0]
                            current = 0
                            visited = set()
                            while True:
                                next_node = None
                                for j in range(n):
                                    if j != current and j not in visited:
                                        if x[current, j, v].X > 0.5:
                                            next_node = j
                                            break
                                if next_node is None or next_node == 0:
                                    break
                                route.append(next_node)
                                visited.add(next_node)
                                current = next_node
                            route.append(0)
                            if len(route) > 2:
                                routes.append(route)
                    
                    if routes:
                        return routes
            return None
        except Exception as e:
            return None

    # ==================== MAIN SOLVER ====================
    # Phase 1: Constructive heuristic
    routes_nn = construct_nearest_neighbor()
    routes_ci = construct_cheapest_insertion()
    
    cost_nn, _ = compute_solution(routes_nn)
    cost_ci, _ = compute_solution(routes_ci)
    
    if cost_nn <= cost_ci:
        routes = routes_nn
    else:
        routes = routes_ci
    
    best_cost, _ = compute_solution(routes)
    
    # Log initial solution
    if logger:
        solution_dict = build_solution_dict(routes)
        logger.log_solution(best_cost, solution_dict)
    
    # Phase 2: Local search
    if time.time() - start_time < time_limit - 10:
        routes = local_search(routes, max_time=time_limit - 15)
        current_cost, _ = compute_solution(routes)
        if current_cost < best_cost:
            best_cost = current_cost
            if logger:
                solution_dict = build_solution_dict(routes)
                logger.log_solution(best_cost, solution_dict)
    
    # Phase 3: Edge set optimization
    if time.time() - start_time < time_limit - 10:
        routes = optimize_edge_sets(routes)
        current_cost, _ = compute_solution(routes)
        if current_cost < best_cost:
            best_cost = current_cost
            if logger:
                solution_dict = build_solution_dict(routes)
                logger.log_solution(best_cost, solution_dict)
    
    # Phase 4: Gurobi for small instances
    if num_customers <= 20 and time.time() - start_time < time_limit - 20:
        gurobi_routes = solve_gurobi()
        if gurobi_routes is not None:
            gurobi_cost, _ = compute_solution(gurobi_routes)
            if gurobi_cost < best_cost:
                routes = gurobi_routes
                best_cost = gurobi_cost
                if logger:
                    solution_dict = build_solution_dict(routes)
                    logger.log_solution(best_cost, solution_dict)
    
    # Phase 5: Additional random restarts with local search
    while time.time() - start_time < time_limit - 10:
        random_routes = construct_nearest_neighbor()
        random_routes = local_search(random_routes, max_time=min(time.time() - start_time + 20, time_limit - 5))
        current_cost, _ = compute_solution(random_routes)
        if current_cost < best_cost:
            best_cost = current_cost
            routes = random_routes
            if logger:
                solution_dict = build_solution_dict(routes)
                logger.log_solution(best_cost, solution_dict)
        else:
            break
    
    # Build final solution
    solution_dict = build_solution_dict(routes)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)

    def build_solution_dict(routes):
        """Build the solution dictionary in the required output format."""
        total_cost, used_sets = compute_solution(routes)
        
        solution_routes = []
        for v_idx, route in enumerate(routes):
            visit_times = {}
            route_times = get_visit_times(route)
            
            # Assign visit times for all nodes
            for node_id in range(n):
                if node_id in route_times:
                    visit_times[str(node_id)] = route_times[node_id]
                else:
                    # Assign a feasible time within the time window
                    visit_times[str(node_id)] = tw_start[node_id]
            
            arcs = []
            for k in range(len(route) - 1):
                arcs.append([route[k], route[k + 1]])
            
            solution_routes.append({
                "vehicle": v_idx,
                "sequence": route,
                "arcs": arcs,
                "visit_times": visit_times
            })
        
        return {
            "objective_value": total_cost,
            "routes": solution_routes,
            "edge_sets_used": sorted(list(used_sets))
        }

    # Need to define build_solution_dict before use - restructure
    # Actually let me restructure the code properly

if __name__ == "__main__":
    main()