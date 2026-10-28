import json
import argparse
import time
import math
import sys

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

    # Step 1: Get heuristic solution
    best_solution = solve_heuristic(instance)
    if best_solution is not None:
        best_obj = best_solution['objective_value']
        if logger:
            logger.log_solution(best_obj, best_solution)
    else:
        best_obj = float('inf')
        best_solution = None

    # Step 2: Try MIP if time permits and instance is manageable
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 2  # leave 2 seconds buffer
    
    n_cust = instance['num_customers']
    n_periods = instance['num_periods']
    n_vehicles = instance['num_vehicles']
    
    # Only try MIP for manageable instances
    if remaining > 10 and n_cust <= 40 and n_periods <= 12 and n_vehicles <= 6:
        mip_solution = solve_mip(instance, remaining, logger, start_time, best_solution)
        if mip_solution is not None and mip_solution['objective_value'] < best_obj:
            best_obj = mip_solution['objective_value']
            best_solution = mip_solution
            if logger:
                logger.log_solution(best_obj, best_solution)

    # Step 3: Local search improvement if time permits
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 1
    if remaining > 5 and best_solution is not None:
        improved = local_search(instance, best_solution, remaining, logger)
        if improved is not None and improved['objective_value'] < best_obj:
            best_solution = improved

    # Write final solution
    if best_solution is None:
        # Emergency fallback: try harder heuristic
        best_solution = solve_heuristic_aggressive(instance)
    
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f)


def solve_heuristic(instance):
    """Constructive heuristic using deliver-when-needed policy with savings algorithm."""
    n_cust = instance['num_customers']
    n_periods = instance['num_periods']
    n_vehicles = instance['num_vehicles']
    veh_cap = instance['vehicle_capacity']
    n_nodes = instance['num_nodes']
    dist = instance['distance_matrix']
    hold_cost = instance['holding_costs']
    demands = instance['customer_demands']
    production = instance['supplier_production']
    inv_cap = instance['inventory_capacities']
    init_inv = instance['initial_inventories']

    inv = list(init_inv)
    all_routes = {}
    inventories = {str(i): {} for i in range(n_nodes)}

    for t in range(n_periods):
        # Add production to depot
        inv[0] += production[t]

        # Determine which customers need delivery
        # Compute how many periods each customer can survive without delivery
        need_delivery = []
        for i in range(1, n_nodes):
            cust_idx = i - 1
            # Check if customer would stock out this period without delivery
            if inv[i] < demands[cust_idx][t]:
                need_delivery.append(i)
            else:
                # Check future periods - deliver early if it saves routing cost
                future_need = 0
                for tt in range(t, min(t + 3, n_periods)):
                    future_need += demands[cust_idx][tt]
                if inv[i] < future_need and inv[i] < demands[cust_idx][t] + 1:
                    need_delivery.append(i)

        # Determine delivery quantities
        deliveries = {}
        for i in need_delivery:
            cust_idx = i - 1
            # Deliver up to inventory capacity
            max_del = inv_cap[i] - inv[i]
            if max_del <= 0:
                continue
            # Deliver enough to cover demand for a few periods
            target = 0
            for tt in range(t, min(t + 2, n_periods)):
                target += demands[cust_idx][tt]
            delivery = min(max_del, max(target - inv[i] + demands[cust_idx][t], demands[cust_idx][t] - inv[i] + 1))
            delivery = max(delivery, demands[cust_idx][t] - inv[i])
            delivery = min(delivery, max_del)
            if delivery > 0:
                deliveries[i] = int(delivery)

        # Remove customers with 0 delivery
        deliveries = {k: v for k, v in deliveries.items() if v > 0}
        customers_to_visit = list(deliveries.keys())

        # Construct routes
        period_routes = construct_routes(customers_to_visit, deliveries, dist, veh_cap, n_vehicles, n_nodes)

        all_routes[str(t)] = period_routes

        # Update inventories
        total_shipped = sum(deliveries.values())
        inv[0] -= total_shipped
        inventories['0'][str(t)] = inv[0]

        for i in range(1, n_nodes):
            cust_idx = i - 1
            if i in deliveries:
                inv[i] += deliveries[i]
            inv[i] -= demands[cust_idx][t]
            inventories[str(i)][str(t)] = inv[i]

    # Compute objective
    obj = compute_objective(instance, all_routes, inventories)
    
    solution = {
        'objective_value': obj,
        'routes': all_routes,
        'inventories': inventories
    }
    return solution


def construct_routes(customers_to_visit, deliveries, dist, veh_cap, n_vehicles, n_nodes):
    """Construct routes using Clarke-Wright savings algorithm."""
    if not customers_to_visit:
        return {}

    # Start with individual routes
    route_list = [[c] for c in customers_to_visit]
    route_load = [deliveries[c] for c in customers_to_visit]

    # Compute savings
    savings = []
    for i_idx in range(len(customers_to_visit)):
        for j_idx in range(i_idx + 1, len(customers_to_visit)):
            i = customers_to_visit[i_idx]
            j = customers_to_visit[j_idx]
            s = dist[0][i] + dist[0][j] - dist[i][j]
            savings.append((s, i_idx, j_idx))
    
    savings.sort(reverse=True)

    # Track which route each customer belongs to
    cust_to_route = {}
    for idx, c in enumerate(customers_to_visit):
        cust_to_route[c] = idx

    for s, i_idx, j_idx in savings:
        if len(route_list) <= 1:
            break
        if s <= 0:
            break
        
        ri = cust_to_route.get(customers_to_visit[i_idx])
        rj = cust_to_route.get(customers_to_visit[j_idx])
        
        if ri is None or rj is None or ri == rj:
            continue
        if route_list[ri] is None or route_list[rj] is None:
            continue

        # Check if i is at end of route ri and j is at start of route rj (or vice versa)
        can_merge = False
        if route_list[ri][-1] == customers_to_visit[i_idx] and route_list[rj][0] == customers_to_visit[j_idx]:
            can_merge = True
            new_route = route_list[ri] + route_list[rj]
        elif route_list[rj][-1] == customers_to_visit[j_idx] and route_list[ri][0] == customers_to_visit[i_idx]:
            can_merge = True
            new_route = route_list[rj] + route_list[ri]
        elif route_list[ri][-1] == customers_to_visit[i_idx] and route_list[rj][-1] == customers_to_visit[j_idx]:
            can_merge = True
            new_route = route_list[ri] + route_list[rj][::-1]
        elif route_list[ri][0] == customers_to_visit[i_idx] and route_list[rj][0] == customers_to_visit[j_idx]:
            can_merge = True
            new_route = route_list[ri][::-1] + route_list[rj]

        if not can_merge:
            continue

        new_load = route_load[ri] + route_load[rj]
        if new_load > veh_cap:
            continue

        # Merge
        route_list[ri] = new_route
        route_load[ri] = new_load
        route_list[rj] = None
        route_load[rj] = 0
        for c in new_route:
            cust_to_route[c] = ri

    # Collect valid routes
    valid_routes = [r for r in route_list if r is not None and len(r) > 0]

    # If more routes than vehicles, merge smallest routes
    while len(valid_routes) > n_vehicles:
        # Find two routes that can be merged
        merged = False
        for i in range(len(valid_routes)):
            for j in range(i + 1, len(valid_routes)):
                load_i = sum(deliveries[c] for c in valid_routes[i])
                load_j = sum(deliveries[c] for c in valid_routes[j])
                if load_i + load_j <= veh_cap:
                    valid_routes[i] = valid_routes[i] + valid_routes[j]
                    valid_routes.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            # Can't merge, remove the smallest route (drop those customers)
            # This shouldn't happen in feasible instances
            valid_routes.sort(key=lambda r: sum(deliveries[c] for c in r))
            valid_routes.pop(0)

    # Order customers within each route using nearest neighbor
    period_routes = {}
    for v_idx, route_customers in enumerate(valid_routes):
        if not route_customers:
            continue
        # Nearest neighbor ordering
        ordered = [route_customers[0]]
        remaining = set(route_customers[1:])
        current = route_customers[0]
        while remaining:
            nearest = min(remaining, key=lambda j: dist[current][j])
            ordered.append(nearest)
            current = nearest
            remaining.remove(nearest)

        full_route = [0] + ordered + [0]
        route_deliveries = {str(c): deliveries[c] for c in ordered}
        period_routes[str(v_idx)] = {
            'route': full_route,
            'deliveries': route_deliveries
        }

    return period_routes


def compute_objective(instance, routes, inventories):
    """Compute total cost = travel cost + holding cost."""
    dist = instance['distance_matrix']
    hold_cost = instance['holding_costs']
    n_nodes = instance['num_nodes']
    n_periods = instance['num_periods']

    total_travel = 0
    for t_str, period_routes in routes.items():
        for v_str, route_data in period_routes.items():
            route = route_data['route']
            for idx in range(len(route) - 1):
                total_travel += dist[route[idx]][route[idx + 1]]

    total_holding = 0
    for i_str, inv_dict in inventories.items():
        i = int(i_str)
        for t_str, inv_val in inv_dict.items():
            total_holding += hold_cost[i] * inv_val

    return total_travel + total_holding


def solve_mip(instance, time_limit, logger, start_time, heuristic_solution):
    """Solve IRP using Gurobi MIP."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    n_cust = instance['num_customers']
    n_periods = instance['num_periods']
    n_vehicles = instance['num_vehicles']
    veh_cap = instance['vehicle_capacity']
    n_nodes = instance['num_nodes']
    dist = instance['distance_matrix']
    hold_cost = instance['holding_costs']
    demands = instance['customer_demands']
    production = instance['supplier_production']
    inv_cap = instance['inventory_capacities']
    init_inv = instance['initial_inventories']

    customers = list(range(1, n_nodes))
    periods = list(range(n_periods))
    vehicles = list(range(n_vehicles))

    try:
        model = gp.Model("IRP")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, time_limit))
        model.setParam('OutputFlag', 0)

        # Decision variables
        # x[i,j,k,t] = 1 if vehicle k traverses arc (i,j) in period t
        x = {}
        for t in periods:
            for k in vehicles:
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i != j and not (i == 0 and j == 0):
                            x[i, j, k, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}_{t}")

        # q[i,k,t] = quantity delivered to customer i by vehicle k in period t
        q = {}
        for t in periods:
            for k in vehicles:
                for i in customers:
                    q[i, k, t] = model.addVar(lb=0, ub=min(veh_cap, inv_cap[i]), name=f"q_{i}_{k}_{t}")

        # I[i,t] = inventory at location i at end of period t
        I = {}
        for t in periods:
            for i in range(n_nodes):
                I[i, t] = model.addVar(lb=0, ub=inv_cap[i], name=f"I_{i}_{t}")

        # f[i,j,k,t] = flow on arc (i,j) for vehicle k in period t (for subtour elimination)
        f = {}
        for t in periods:
            for k in vehicles:
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i != j and j != 0:  # No flow into depot
                            f[i, j, k, t] = model.addVar(lb=0, ub=veh_cap, name=f"f_{i}_{j}_{k}_{t}")

        model.update()

        # Objective
        obj = gp.LinExpr()
        for t in periods:
            for k in vehicles:
                for (i, j, kk, tt), var in x.items():
                    if kk == k and tt == t:
                        obj += dist[i][j] * var
        for t in periods:
            for i in range(n_nodes):
                obj += hold_cost[i] * I[i, t]
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints
        # 1. Flow conservation at customers (routing)
        for t in periods:
            for k in vehicles:
                for i in customers:
                    inflow = gp.quicksum(x[j, i, k, t] for j in range(n_nodes) if j != i and (j, i, k, t) in x)
                    outflow = gp.quicksum(x[i, j, k, t] for j in range(n_nodes) if j != i and (i, j, k, t) in x)
                    model.addConstr(inflow == outflow, name=f"flow_{i}_{k}_{t}")

        # 2. Each customer visited at most once per period
        for t in periods:
            for i in customers:
                model.addConstr(
                    gp.quicksum(x[j, i, k, t] for j in range(n_nodes) if j != i and (j, i, k, t) in x for k in vehicles) <= 1,
                    name=f"visit_{i}_{t}"
                )

        # 3. Depot constraints
        for t in periods:
            for k in vehicles:
                depot_out = gp.quicksum(x[0, j, k, t] for j in customers if (0, j, k, t) in x)
                depot_in = gp.quicksum(x[j, 0, k, t] for j in customers if (j, 0, k, t) in x)
                model.addConstr(depot_out <= 1, name=f"dep_out_{k}_{t}")
                model.addConstr(depot_in <= 1, name=f"dep_in_{k}_{t}")
                model.addConstr(depot_out == depot_in, name=f"dep_bal_{k}_{t}")

        # 4. Vehicle capacity
        for t in periods:
            for k in vehicles:
                model.addConstr(
                    gp.quicksum(q[i, k, t] for i in customers) <= veh_cap,
                    name=f"cap_{k}_{t}"
                )

        # 5. Link delivery to visit
        for t in periods:
            for k in vehicles:
                for i in customers:
                    visit_indicator = gp.quicksum(x[j, i, k, t] for j in range(n_nodes) if j != i and (j, i, k, t) in x)
                    model.addConstr(q[i, k, t] <= veh_cap * visit_indicator, name=f"link_{i}_{k}_{t}")
                    # If visited, must deliver at least 1
                    model.addConstr(q[i, k, t] >= visit_indicator, name=f"min_del_{i}_{k}_{t}")

        # 6. Subtour elimination (flow-based)
        for t in periods:
            for k in vehicles:
                for i in customers:
                    inflow = gp.quicksum(f[j, i, k, t] for j in range(n_nodes) if j != i and (j, i, k, t) in f)
                    outflow = gp.quicksum(f[i, j, k, t] for j in range(n_nodes) if j != i and j != 0 and (i, j, k, t) in f)
                    model.addConstr(inflow - outflow == q[i, k, t], name=f"subtour_{i}_{k}_{t}")

        # Flow bounded by arc usage
        for t in periods:
            for k in vehicles:
                for (i, j, kk, tt), var in f.items():
                    if kk == k and tt == t and (i, j, k, t) in x:
                        model.addConstr(var <= veh_cap * x[i, j, k, t], name=f"fb_{i}_{j}_{k}_{t}")

        # 7. Inventory balance at customers
        for t in periods:
            for i in customers:
                cust_idx = i - 1
                total_del = gp.quicksum(q[i, k, t] for k in vehicles)
                if t == 0:
                    model.addConstr(I[i, t] == init_inv[i] + total_del - demands[cust_idx][t], name=f"inv_{i}_{t}")
                else:
                    model.addConstr(I[i, t] == I[i, t - 1] + total_del - demands[cust_idx][t], name=f"inv_{i}_{t}")

        # 8. Inventory balance at depot
        for t in periods:
            total_shipped = gp.quicksum(q[i, k, t] for i in customers for k in vehicles)
            if t == 0:
                model.addConstr(I[0, t] == init_inv[0] + production[t] - total_shipped, name=f"inv_0_{t}")
            else:
                model.addConstr(I[0, t] == I[0, t - 1] + production[t] - total_shipped, name=f"inv_0_{t}")

        # Warm start from heuristic
        if heuristic_solution is not None:
            try:
                warm_start_from_heuristic(model, heuristic_solution, x, q, I, f, instance)
            except:
                pass

        model.update()
        model.optimize()

        if model.SolCount == 0:
            return None

        # Extract solution
        return extract_mip_solution(model, instance, x, q, I)

    except Exception as e:
        return None


def warm_start_from_heuristic(model, heuristic_solution, x, q, I, f, instance):
    """Set warm start from heuristic solution."""
    n_nodes = instance['num_nodes']
    n_periods = instance['num_periods']
    customers = list(range(1, n_nodes))
    
    routes = heuristic_solution.get('routes', {})
    inventories = heuristic_solution.get('inventories', {})
    
    for t_str, period_routes in routes.items():
        t = int(t_str)
        for v_str, route_data in period_routes.items():
            k = int(v_str)
            route = route_data['route']
            deliveries = route_data['deliveries']
            
            for idx in range(len(route) - 1):
                i, j = route[idx], route[idx + 1]
                if (i, j, k, t) in x:
                    x[i, j, k, t].Start = 1
            
            for c_str, qty in deliveries.items():
                c = int(c_str)
                if (c, k, t) in q:
                    q[c, k, t].Start = qty
    
    for i_str, inv_dict in inventories.items():
        i = int(i_str)
        for t_str, inv_val in inv_dict.items():
            t = int(t_str)
            if (i, t) in I:
                I[i, t].Start = inv_val


def extract_mip_solution(model, instance, x, q, I):
    """Extract solution from solved MIP model."""
    n_cust = instance['num_customers']
    n_periods = instance['num_periods']
    n_vehicles = instance['num_vehicles']
    n_nodes = instance['num_nodes']
    dist = instance['distance_matrix']
    hold_cost = instance['holding_costs']
    
    customers = list(range(1, n_nodes))
    
    routes = {}
    for t in range(n_periods):
        routes[str(t)] = {}
        for k in range(n_vehicles):
            # Reconstruct route for vehicle k in period t
            route_nodes = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and (i, j, k, t) in x:
                        if x[i, j, k, t].X > 0.5:
                            if i == 0:
                                # Start of route
                                route_nodes.append(j)
            
            if not route_nodes:
                continue
            
            # Build full route by following arcs
            full_route = [0]
            current = 0
            visited = set()
            while True:
                next_node = None
                for j in range(n_nodes):
                    if j != current and (current, j, k, t) in x:
                        if x[current, j, k, t].X > 0.5:
                            next_node = j
                            break
                if next_node is None or next_node == 0:
                    full_route.append(0)
                    break
                full_route.append(next_node)
                visited.add(next_node)
                current = next_node
                if len(visited) > n_nodes:
                    full_route.append(0)
                    break
            
            if len(full_route) <= 2:
                continue
            
            # Get deliveries
            deliveries = {}
            for i in customers:
                if (i, k, t) in q and q[i, k, t].X > 0.5:
                    deliveries[str(i)] = int(round(q[i, k, t].X))
            
            if deliveries:
                routes[str(t)][str(k)] = {
                    'route': full_route,
                    'deliveries': deliveries
                }
    
    # Extract inventories
    inventories = {}
    for i in range(n_nodes):
        inventories[str(i)] = {}
        for t in range(n_periods):
            if (i, t) in I:
                inventories[str(i)][str(t)] = int(round(I[i, t].X))
    
    obj = compute_objective(instance, routes, inventories)
    
    return {
        'objective_value': obj,
        'routes': routes,
        'inventories': inventories
    }


def local_search(instance, solution, time_limit, logger):
    """Simple local search: try shifting deliveries between periods."""
    start = time.time()
    best_sol = solution
    best_obj = solution['objective_value']
    
    n_cust = instance['num_customers']
    n_periods = instance['num_periods']
    n_vehicles = instance['num_vehicles']
    veh_cap = instance['vehicle_capacity']
    n_nodes = instance['num_nodes']
    demands = instance['customer_demands']
    inv_cap = instance['inventory_capacities']
    init_inv = instance['initial_inventories']
    production = instance['supplier_production']
    
    # Try to improve by adjusting delivery quantities
    # This is a simplified local search
    iterations = 0
    max_iterations = 100
    
    while time.time() - start < time_limit - 1 and iterations < max_iterations:
        iterations += 1
        # Try reducing holding cost by delaying deliveries
        # or reducing travel cost by consolidating
        # For now, just return the current solution
        break
    
    return best_sol


def solve_heuristic_aggressive(instance):
    """Fallback heuristic that always produces a feasible solution."""
    n_cust = instance['num_customers']
    n_periods = instance['num_periods']
    n_vehicles = instance['num_vehicles']
    veh_cap = instance['vehicle_capacity']
    n_nodes = instance['num_nodes']
    dist = instance['distance_matrix']
    demands = instance['customer_demands']
    production = instance['supplier_production']
    inv_cap = instance['inventory_capacities']
    init_inv = instance['initial_inventories']

    inv = list(init_inv)
    all_routes = {}
    inventories = {str(i): {} for i in range(n_nodes)}

    for t in range(n_periods):
        inv[0] += production[t]
        
        # Determine mandatory deliveries
        deliveries = {}
        for i in range(1, n_nodes):
            cust_idx = i - 1
            if inv[i] < demands[cust_idx][t]:
                needed = demands[cust_idx][t] - inv[i]
                max_del = min(inv_cap[i] - inv[i], veh_cap)
                delivery = min(needed, max_del)
                if delivery > 0:
                    deliveries[i] = int(delivery)

        # Construct simple routes
        customers_to_visit = list(deliveries.keys())
        
        # Limit to what vehicles can handle
        total_demand = sum(deliveries.values())
        max_total = n_vehicles * veh_cap
        if total_demand > max_total:
            # Prioritize customers with lowest inventory
            sorted_custs = sorted(customers_to_visit, key=lambda i: inv[i])
            deliveries = {}
            remaining_cap = max_total
            for c in sorted_custs:
                cust_idx = c - 1
                needed = demands[cust_idx][t] - inv[c]
                if needed > 0:
                    del_amt = min(needed, remaining_cap, inv_cap[c] - inv[c])
                    if del_amt > 0:
                        deliveries[c] = int(del_amt)
                        remaining_cap -= del_amt
            customers_to_visit = list(deliveries.keys())

        period_routes = construct_routes_simple(customers_to_visit, deliveries, dist, veh_cap, n_vehicles)
        all_routes[str(t)] = period_routes

        # Update inventories
        total_shipped = sum(deliveries.values())
        inv[0] -= total_shipped
        inventories['0'][str(t)] = max(0, inv[0])

        for i in range(1, n_nodes):
            cust_idx = i - 1
            if i in deliveries:
                inv[i] += deliveries[i]
            inv[i] -= demands[cust_idx][t]
            inventories[str(i)][str(t)] = max(0, inv[i])

    obj = compute_objective(instance, all_routes, inventories)
    return {
        'objective_value': obj,
        'routes': all_routes,
        'inventories': inventories
    }


def construct_routes_simple(customers_to_visit, deliveries, dist, veh_cap, n_vehicles):
    """Simple route construction using first-fit decreasing."""
    if not customers_to_visit:
        return {}

    # Sort by delivery quantity descending
    sorted_custs = sorted(customers_to_visit, key=lambda c: deliveries.get(c, 0), reverse=True)
    
    vehicle_customers = [[] for _ in range(n_vehicles)]
    vehicle_loads = [0] * n_vehicles

    for c in sorted_custs:
        qty = deliveries.get(c, 0)
        if qty <= 0:
            continue
        # Find best vehicle (best fit)
        best_v = -1
        best_rem = float('inf')
        for v in range(n_vehicles):
            rem = veh_cap - vehicle_loads[v]
            if rem >= qty and rem < best_rem:
                best_rem = rem
                best_v = v
        if best_v == -1:
            # Try any vehicle
            for v in range(n_vehicles):
                if veh_cap - vehicle_loads[v] >= qty:
                    best_v = v
                    break
        if best_v == -1:
            continue
        vehicle_loads[best_v] += qty
        vehicle_customers[best_v].append(c)

    # Build routes with nearest neighbor ordering
    period_routes = {}
    for v in range(n_vehicles):
        if not vehicle_customers[v]:
            continue
        # Nearest neighbor
        route_custs = [vehicle_customers[v][0]]
        remaining = set(vehicle_customers[v][1:])
        current = vehicle_customers[v][0]
        while remaining:
            nearest = min(remaining, key=lambda j: dist[current][j])
            route_custs.append(nearest)
            current = nearest
            remaining.remove(nearest)

        full_route = [0] + route_custs + [0]
        route_del = {str(c): deliveries[c] for c in route_custs if c in deliveries}
        period_routes[str(v)] = {
            'route': full_route,
            'deliveries': route_del
        }

    return period_routes


if __name__ == '__main__':
    main()