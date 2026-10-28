import json
import argparse
import time
import math
import sys
from copy import deepcopy

def solve_mip(instance, time_limit, logger, start_time):
    """Solve using Gurobi MIP formulation for small instances."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None
    
    n = instance['num_customers']
    capacity = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']
    num_vehicles_ub = instance['num_vehicles_upper_bound']
    
    K = min(num_vehicles_ub, n + 5)
    N = n + 2
    ret = n + 1
    
    tw = [None] * N
    st_arr = [0] * N
    dem = [0] * N
    
    tw[0] = depot['time_window']
    st_arr[0] = depot['service_time']
    tw[ret] = depot['time_window']
    st_arr[ret] = 0
    
    for c in customers:
        tw[c['id']] = c['time_window']
        st_arr[c['id']] = c['service_time']
        dem[c['id']] = c['demand']
    
    # Feasible arcs
    arcs = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if i == ret:
                continue
            if j == 0:
                continue
            if i == 0 and j == ret:
                arcs.append((i, j))
                continue
            if tw[i][0] + st_arr[i] + dist[i][j] <= tw[j][1]:
                arcs.append((i, j))
    
    arc_set = set(arcs)
    
    if len(arcs) * K > 80000:
        return None
    
    model = gp.Model("VRPSDTW")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    
    x = {}
    for (i, j) in arcs:
        for k in range(K):
            x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
    
    q = {}
    for c in customers:
        ci = c['id']
        for k in range(K):
            q[ci, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                     ub=min(c['demand'], capacity),
                                     name=f"q_{ci}_{k}")
    
    t = {}
    for i in range(N):
        for k in range(K):
            t[i, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=tw[i][0], ub=tw[i][1],
                                    name=f"t_{i}_{k}")
    
    model.update()
    
    obj = gp.quicksum(dist[i][j] * x[i, j, k] for (i, j) in arcs for k in range(K))
    model.setObjective(obj, GRB.MINIMIZE)
    
    for c in customers:
        ci = c['id']
        for k in range(K):
            inflow = gp.quicksum(x[j, ci, k] for j in range(N) if (j, ci) in arc_set)
            outflow = gp.quicksum(x[ci, j, k] for j in range(N) if (ci, j) in arc_set)
            model.addConstr(inflow == outflow, name=f"flow_{ci}_{k}")
    
    for k in range(K):
        leaves = gp.quicksum(x[0, j, k] for j in range(N) if (0, j) in arc_set)
        enters = gp.quicksum(x[i, ret, k] for i in range(N) if (i, ret) in arc_set)
        model.addConstr(leaves <= 1, name=f"leave_{k}")
        model.addConstr(enters <= 1, name=f"enter_{k}")
        model.addConstr(leaves == enters, name=f"bal_{k}")
    
    for c in customers:
        ci = c['id']
        model.addConstr(gp.quicksum(q[ci, k] for k in range(K)) >= c['demand'],
                       name=f"dem_{ci}")
    
    for c in customers:
        ci = c['id']
        for k in range(K):
            visited = gp.quicksum(x[ci, j, k] for j in range(N) if (ci, j) in arc_set)
            model.addConstr(q[ci, k] <= min(c['demand'], capacity) * visited,
                           name=f"dv_{ci}_{k}")
    
    for k in range(K):
        model.addConstr(gp.quicksum(q[c['id'], k] for c in customers) <= capacity,
                       name=f"cap_{k}")
    
    for (i, j) in arcs:
        if i == 0 and j == ret:
            continue
        for k in range(K):
            M = tw[i][1] + st_arr[i] + dist[i][j] - tw[j][0]
            if M > 0:
                model.addConstr(t[j, k] >= t[i, k] + st_arr[i] + dist[i][j] - M * (1 - x[i, j, k]),
                               name=f"time_{i}_{j}_{k}")
    
    total_demand = sum(c['demand'] for c in customers)
    min_veh = math.ceil(total_demand / capacity)
    veh_used = gp.quicksum(x[0, j, k] for j in range(N) if (0, j) in arc_set for k in range(K))
    model.addConstr(veh_used >= min_veh, name="min_veh")
    
    for k in range(1, K):
        use_k = gp.quicksum(x[0, j, k] for j in range(N) if (0, j) in arc_set)
        use_km1 = gp.quicksum(x[0, j, k - 1] for j in range(N) if (0, j) in arc_set)
        model.addConstr(use_k <= use_km1, name=f"sym_{k}")
    
    model.optimize()
    
    if model.SolCount > 0:
        obj_val = model.ObjVal
        routes = []
        deliveries_summary = {str(c['id']): {'demand': c['demand'], 'total_delivered': 0.0}
                             for c in customers}
        
        for k in range(K):
            used = any(x[0, j, k].X > 0.5 for j in range(N) if (0, j) in arc_set)
            if not used:
                continue
            
            route_nodes = [0]
            current = 0
            while current != ret:
                next_node = None
                for j in range(N):
                    if (current, j) in arc_set and x[current, j, k].X > 0.5:
                        next_node = j
                        break
                if next_node is None:
                    break
                route_nodes.append(next_node)
                current = next_node
            
            vehicle_deliveries = {}
            for c in customers:
                ci = c['id']
                qval = q[ci, k].X
                if qval > 1e-6:
                    vehicle_deliveries[str(ci)] = round(qval, 4)
                    deliveries_summary[str(ci)]['total_delivered'] += qval
            
            routes.append({
                'vehicle': k,
                'route': route_nodes,
                'deliveries': vehicle_deliveries
            })
        
        solution = {
            'objective_value': round(obj_val, 4),
            'num_vehicles_used': len(routes),
            'routes': routes,
            'deliveries': deliveries_summary
        }
        
        if logger:
            logger.log_solution(solution['objective_value'], solution)
        
        return solution
    
    return None


def solve_heuristic(instance, time_limit, logger, start_time):
    """Solve using construction heuristic + local search."""
    n = instance['num_customers']
    capacity = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']
    num_vehicles_ub = instance['num_vehicles_upper_bound']
    
    tw = {0: depot['time_window']}
    st = {0: depot['service_time']}
    dem = {}
    
    for c in customers:
        tw[c['id']] = c['time_window']
        st[c['id']] = c['service_time']
        dem[c['id']] = c['demand']
    
    depot_tw = depot['time_window']
    all_customers = [c['id'] for c in customers]
    
    def is_route_feasible(route):
        if not route:
            return True
        current_time = depot_tw[0]
        prev = 0
        for node in route:
            arrival = current_time + st[prev] + dist[prev][node]
            service_start = max(arrival, tw[node][0])
            if service_start > tw[node][1]:
                return False
            current_time = service_start
            prev = node
        arrival = current_time + st[prev] + dist[prev][0]
        if arrival > depot_tw[1]:
            return False
        return True
    
    def route_cost(route):
        if not route:
            return 0.0
        cost = dist[0][route[0]]
        for i in range(len(route) - 1):
            cost += dist[route[i]][route[i + 1]]
        cost += dist[route[-1]][0]
        return cost
    
    def compute_deliveries(routes):
        """Compute delivery quantities given routes."""
        remaining = {cid: dem[cid] for cid in all_customers}
        route_deliveries = []
        
        for route in routes:
            deliveries = {}
            remaining_cap = capacity
            for cid in route:
                if remaining[cid] > 0 and remaining_cap > 0:
                    qty = min(remaining[cid], remaining_cap)
                    deliveries[cid] = qty
                    remaining[cid] -= qty
                    remaining_cap -= qty
                else:
                    deliveries[cid] = 0
            route_deliveries.append(deliveries)
        
        # Check if all demand met
        for cid in all_customers:
            if remaining[cid] > 0:
                # Add extra routes for unmet demand
                while remaining[cid] > 0:
                    routes.append([cid])
                    qty = min(remaining[cid], capacity)
                    route_deliveries.append({cid: qty})
                    remaining[cid] -= qty
        
        return route_deliveries
    
    def total_cost(routes):
        return sum(route_cost(r) for r in routes)
    
    def build_solution(routes, route_deliveries):
        """Build solution dictionary from routes and deliveries."""
        obj_val = total_cost(routes)
        sol_routes = []
        deliveries_summary = {str(cid): {'demand': dem[cid], 'total_delivered': 0.0}
                             for cid in all_customers}
        
        for idx, (route, deliveries) in enumerate(zip(routes, route_deliveries)):
            if not route:
                continue
            route_nodes = [0] + route + [n + 1]
            veh_del = {}
            for cid, qty in deliveries.items():
                if qty > 0:
                    veh_del[str(cid)] = qty
                    deliveries_summary[str(cid)]['total_delivered'] += qty
            
            sol_routes.append({
                'vehicle': idx,
                'route': route_nodes,
                'deliveries': veh_del
            })
        
        return {
            'objective_value': round(obj_val, 4),
            'num_vehicles_used': len(sol_routes),
            'routes': sol_routes,
            'deliveries': deliveries_summary
        }
    
    # Construction: cheapest insertion
    remaining_demand = {cid: dem[cid] for cid in all_customers}
    routes = []
    
    while any(remaining_demand[cid] > 0 for cid in all_customers):
        if time.time() - start_time > time_limit - 2:
            break
        
        route = []
        remaining_cap = capacity
        in_route = set()
        
        while remaining_cap > 0:
            best_cost = float('inf')
            best_cid = None
            best_pos = None
            
            for cid in all_customers:
                if remaining_demand[cid] <= 0:
                    continue
                if cid in in_route:
                    continue
                
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [cid] + route[pos:]
                    if is_route_feasible(new_route):
                        if pos == 0:
                            prev = 0
                        else:
                            prev = route[pos - 1]
                        if pos == len(route):
                            nxt = 0
                        else:
                            nxt = route[pos]
                        cost = dist[prev][cid] + dist[cid][nxt] - dist[prev][nxt]
                        if cost < best_cost:
                            best_cost = cost
                            best_cid = cid
                            best_pos = pos
            
            if best_cid is None:
                break
            
            qty = min(remaining_demand[best_cid], remaining_cap)
            route.insert(best_pos, best_cid)
            in_route.add(best_cid)
            remaining_demand[best_cid] -= qty
            remaining_cap -= qty
        
        if route:
            routes.append(route)
        else:
            # Fallback: assign one customer per route
            for cid in all_customers:
                if remaining_demand[cid] > 0:
                    routes.append([cid])
                    remaining_demand[cid] -= min(remaining_demand[cid], capacity)
                    break
    
    # Ensure all demand is met
    for cid in all_customers:
        while remaining_demand[cid] > 0:
            routes.append([cid])
            remaining_demand[cid] -= min(remaining_demand[cid], capacity)
    
    route_deliveries = compute_deliveries(routes)
    
    best_routes = deepcopy(routes)
    best_deliveries = deepcopy(route_deliveries)
    best_cost = total_cost(routes)
    
    if logger:
        sol = build_solution(routes, route_deliveries)
        logger.log_solution(sol['objective_value'], sol)
    
    # Local search
    def try_2opt(routes, route_deliveries):
        improved = False
        for ri in range(len(routes)):
            route = routes[ri]
            if len(route) < 3:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    if is_route_feasible(new_route):
                        old_cost = route_cost(route)
                        new_cost = route_cost(new_route)
                        if new_cost < old_cost - 1e-10:
                            routes[ri] = new_route
                            route = new_route
                            improved = True
        return improved
    
    def try_relocate(routes, route_deliveries):
        improved = False
        for ri in range(len(routes)):
            if not routes[ri]:
                continue
            for ci in range(len(routes[ri])):
                cid = routes[ri][ci]
                # Try moving cid to another route
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    for pos in range(len(routes[rj]) + 1):
                        new_rj = routes[rj][:pos] + [cid] + routes[rj][pos:]
                        if not is_route_feasible(new_rj):
                            continue
                        new_ri = routes[ri][:ci] + routes[ri][ci + 1:]
                        
                        old_cost = route_cost(routes[ri]) + route_cost(routes[rj])
                        new_cost = route_cost(new_ri) + route_cost(new_rj)
                        
                        if new_cost < old_cost - 1e-10:
                            routes[ri] = new_ri
                            routes[rj] = new_rj
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        return improved
    
    def try_swap(routes, route_deliveries):
        improved = False
        for ri in range(len(routes)):
            for rj in range(ri + 1, len(routes)):
                if not routes[ri] or not routes[rj]:
                    continue
                for ci in range(len(routes[ri])):
                    for cj in range(len(routes[rj])):
                        cid_i = routes[ri][ci]
                        cid_j = routes[rj][cj]
                        new_ri = routes[ri][:ci] + [cid_j] + routes[ri][ci + 1:]
                        new_rj = routes[rj][:cj] + [cid_i] + routes[rj][cj + 1:]
                        
                        if not is_route_feasible(new_ri) or not is_route_feasible(new_rj):
                            continue
                        
                        old_cost = route_cost(routes[ri]) + route_cost(routes[rj])
                        new_cost = route_cost(new_ri) + route_cost(new_rj)
                        
                        if new_cost < old_cost - 1e-10:
                            routes[ri] = new_ri
                            routes[rj] = new_rj
                            improved = True
                            return improved
        return improved
    
    def try_remove_route(routes, route_deliveries):
        """Try to remove a route by redistributing its customers."""
        improved = False
        for ri in range(len(routes)):
            if not routes[ri]:
                continue
            if len(routes[ri]) > 3:
                continue
            
            route_to_remove = routes[ri][:]
            success = True
            temp_routes = deepcopy(routes)
            
            for cid in route_to_remove:
                placed = False
                for rj in range(len(temp_routes)):
                    if rj == ri:
                        continue
                    for pos in range(len(temp_routes[rj]) + 1):
                        new_rj = temp_routes[rj][:pos] + [cid] + temp_routes[rj][pos:]
                        if is_route_feasible(new_rj):
                            # Check capacity
                            temp_routes[rj] = new_rj
                            placed = True
                            break
                    if placed:
                        break
                
                if not placed:
                    success = False
                    break
            
            if success:
                old_cost = total_cost(routes)
                temp_routes[ri] = []
                new_cost = total_cost(temp_routes)
                if new_cost < old_cost - 1e-10:
                    routes[:] = temp_routes
                    improved = True
                    break
        
        # Clean up empty routes
        routes[:] = [r for r in routes if r]
        return improved
    
    # Apply local search iteratively
    iteration = 0
    while time.time() - start_time < time_limit - 2:
        iteration += 1
        any_improved = False
        
        if try_2opt(routes, route_deliveries):
            any_improved = True
        
        if time.time() - start_time > time_limit - 2:
            break
        
        if try_relocate(routes, route_deliveries):
            any_improved = True
        
        if time.time() - start_time > time_limit - 2:
            break
        
        if try_swap(routes, route_deliveries):
            any_improved = True
        
        if time.time() - start_time > time_limit - 2:
            break
        
        if try_remove_route(routes, route_deliveries):
            any_improved = True
        
        current_cost = total_cost(routes)
        if current_cost < best_cost:
            best_cost = current_cost
            best_routes = deepcopy(routes)
            best_deliveries = compute_deliveries(best_routes)
            if logger:
                sol = build_solution(best_routes, best_deliveries)
                logger.log_solution(sol['objective_value'], sol)
        
        if not any_improved:
            break
    
    # Recompute deliveries for best routes
    best_deliveries = compute_deliveries(best_routes)
    
    solution = build_solution(best_routes, best_deliveries)
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    start_time = time.time()
    time_limit = args.time_limit
    
    n = instance['num_customers']
    capacity = instance['vehicle_capacity']
    customers = instance['customers']
    num_vehicles_ub = instance['num_vehicles_upper_bound']
    
    # Run heuristic first for quick feasible solution
    solution = solve_heuristic(instance, time_limit, logger, start_time)
    
    # Try MIP for small instances if time permits
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 3
    if remaining > 30 and n <= 22 and num_vehicles_ub <= 25:
        mip_sol = solve_mip(instance, remaining, logger, start_time)
        if mip_sol is not None:
            if solution is None or mip_sol['objective_value'] < solution['objective_value']:
                solution = mip_sol
    
    if solution is None:
        # Fallback: create a trivial solution
        depot = instance['depot']
        dist = instance['distance_matrix']
        routes = []
        deliveries_summary = {}
        remaining_demand = {}
        for c in customers:
            remaining_demand[c['id']] = c['demand']
            deliveries_summary[str(c['id'])] = {'demand': c['demand'], 'total_delivered': 0.0}
        
        obj_val = 0.0
        veh_id = 0
        for c in customers:
            while remaining_demand[c['id']] > 0:
                qty = min(remaining_demand[c['id']], capacity)
                routes.append({
                    'vehicle': veh_id,
                    'route': [0, c['id'], n + 1],
                    'deliveries': {str(c['id']): qty}
                })
                deliveries_summary[str(c['id'])]['total_delivered'] += qty
                remaining_demand[c['id']] -= qty
                obj_val += 2 * dist[0][c['id']]
                veh_id += 1
        
        solution = {
            'objective_value': round(obj_val, 4),
            'num_vehicles_used': len(routes),
            'routes': routes,
            'deliveries': deliveries_summary
        }
    
    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()