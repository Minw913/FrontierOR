import json
import argparse
import time
import sys

def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_objective(instance, routes_per_period, deliveries_per_period, inventories_per_period):
    """Compute total cost from solution components."""
    T = instance['num_periods']
    dist = instance['distance_matrix']
    depot = instance['depot']
    customers = instance['customers']
    
    total_cost = 0.0
    for t in range(T):
        # Travel cost
        for route in routes_per_period[t]:
            for idx in range(len(route) - 1):
                total_cost += dist[route[idx]][route[idx + 1]]
        # Holding cost
        total_cost += depot['holding_cost'] * inventories_per_period[t][0]
        for i in range(1, instance['num_customers'] + 1):
            total_cost += customers[i-1]['holding_cost'] * inventories_per_period[t][i]
    return total_cost

def build_solution_dict(instance, routes_per_period, deliveries_per_period, inventories_per_period):
    """Build the solution dictionary in the required output format."""
    T = instance['num_periods']
    
    objective_value = compute_objective(instance, routes_per_period, deliveries_per_period, inventories_per_period)
    
    solution = {
        "objective_value": objective_value,
        "solution_details": {
            "periods": {}
        }
    }
    
    for t in range(T):
        period_key = str(t + 1)
        
        routes = routes_per_period[t]
        
        deliveries = {}
        for cust_id, qty in deliveries_per_period[t].items():
            if qty > 1e-9:
                deliveries[str(cust_id)] = round(qty, 6)
        
        inventories = {"depot": round(inventories_per_period[t][0], 6)}
        for i in range(1, instance['num_customers'] + 1):
            inventories[str(i)] = round(inventories_per_period[t][i], 6)
        
        solution["solution_details"]["periods"][period_key] = {
            "routes": routes,
            "deliveries": deliveries,
            "inventories": inventories
        }
    
    return solution

def solve_mip(instance, time_limit, logger=None):
    """Solve using MIP formulation with Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    
    n = instance['num_customers']
    T = instance['num_periods']
    K = instance['num_vehicles']
    Q = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']
    
    N = n + 1
    
    # Check if problem is too large for MIP
    num_x_vars = N * (N - 1) * K * T
    if num_x_vars > 200000:
        return None
    
    model = gp.Model("IRP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0
    
    # Decision variables
    x = {}
    for t in range(T):
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    if i != j:
                        x[i, j, k, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}_{t}")
    
    y = {}
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                y[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}_{t}")
    
    q = {}
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                q[i, k, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, 
                                           ub=customers[i-1]['max_inventory'],
                                           name=f"q_{i}_{k}_{t}")
    
    I = {}
    for t in range(T):
        I[0, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=depot['max_inventory'], name=f"I_0_{t}")
        for i in range(1, N):
            I[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=customers[i-1]['min_inventory'],
                                    ub=customers[i-1]['max_inventory'], name=f"I_{i}_{t}")
    
    # MTZ position variables
    u = {}
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                u[i, k, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=n, name=f"u_{i}_{k}_{t}")
    
    # Objective
    obj = gp.LinExpr()
    for t in range(T):
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    if i != j:
                        obj += dist[i][j] * x[i, j, k, t]
    
    for t in range(T):
        obj += depot['holding_cost'] * I[0, t]
        for i in range(1, N):
            obj += customers[i-1]['holding_cost'] * I[i, t]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    # Flow conservation at customers
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                model.addConstr(
                    gp.quicksum(x[j, i, k, t] for j in range(N) if j != i) == y[i, k, t],
                    name=f"flow_in_{i}_{k}_{t}"
                )
                model.addConstr(
                    gp.quicksum(x[i, j, k, t] for j in range(N) if j != i) == y[i, k, t],
                    name=f"flow_out_{i}_{k}_{t}"
                )
    
    # Route starts and ends at depot
    for t in range(T):
        for k in range(K):
            model.addConstr(
                gp.quicksum(x[0, j, k, t] for j in range(1, N)) <= 1,
                name=f"route_start_{k}_{t}"
            )
            model.addConstr(
                gp.quicksum(x[i, 0, k, t] for i in range(1, N)) ==
                gp.quicksum(x[0, j, k, t] for j in range(1, N)),
                name=f"route_end_{k}_{t}"
            )
    
    # Vehicle capacity
    for t in range(T):
        for k in range(K):
            model.addConstr(
                gp.quicksum(q[i, k, t] for i in range(1, N)) <= Q,
                name=f"capacity_{k}_{t}"
            )
    
    # Each customer visited at most once per period
    for t in range(T):
        for i in range(1, N):
            model.addConstr(
                gp.quicksum(y[i, k, t] for k in range(K)) <= 1,
                name=f"visit_once_{i}_{t}"
            )
    
    # Delivery only if visited
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                model.addConstr(
                    q[i, k, t] <= customers[i-1]['max_inventory'] * y[i, k, t],
                    name=f"deliver_visit_{i}_{k}_{t}"
                )
    
    # Inventory balance at depot
    for t in range(T):
        prev_inv = depot['initial_inventory'] if t == 0 else I[0, t-1]
        total_delivered = gp.quicksum(q[i, k, t] for i in range(1, N) for k in range(K))
        model.addConstr(
            I[0, t] == prev_inv + depot['production_per_period'] - total_delivered,
            name=f"depot_bal_{t}"
        )
    
    # Inventory balance at customers
    for t in range(T):
        for i in range(1, N):
            prev_inv = customers[i-1]['initial_inventory'] if t == 0 else I[i, t-1]
            total_delivered_i = gp.quicksum(q[i, k, t] for k in range(K))
            model.addConstr(
                I[i, t] == prev_inv + total_delivered_i - customers[i-1]['demand_per_period'],
                name=f"cust_bal_{i}_{t}"
            )
    
    # MTZ subtour elimination
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                for j in range(1, N):
                    if i != j:
                        model.addConstr(
                            u[i, k, t] - u[j, k, t] + n * x[i, j, k, t] <= n - 1,
                            name=f"mtz_{i}_{j}_{k}_{t}"
                        )
    
    # Callback for logging incumbent solutions
    best_solutions = []
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            try:
                sol = extract_solution_cb(model, instance, x, y, q, I, T, K, N, n)
                if sol is not None:
                    best_solutions.append((obj_val, sol))
                    if logger:
                        logger.log_solution(obj_val, sol)
            except:
                if logger:
                    logger.log(obj_val)
    
    model.optimize(callback)
    
    # Extract final solution
    if model.SolCount > 0:
        try:
            sol = extract_solution_final(model, instance, x, y, q, I, T, K, N, n)
            if sol is not None:
                return sol
        except:
            pass
    
    # Return best from callback
    if best_solutions:
        best_solutions.sort(key=lambda x: x[0])
        return best_solutions[0][1]
    
    return None

def extract_solution_cb(model, instance, x, y, q, I, T, K, N, n):
    """Extract solution during callback using cbGetSolution."""
    routes_per_period = []
    deliveries_per_period = []
    inventories_per_period = []
    
    for t in range(T):
        routes = []
        deliveries = {}
        
        for k in range(K):
            # Check if vehicle k is used
            route_start_val = sum(model.cbGetSolution(x[0, j, k, t]) for j in range(1, N))
            if route_start_val < 0.5:
                continue
            
            # Trace route
            route = [0]
            current = 0
            visited_nodes = set()
            while True:
                next_node = None
                for j in range(N):
                    if j != current and j not in visited_nodes:
                        if (current, j, k, t) in x and model.cbGetSolution(x[current, j, k, t]) > 0.5:
                            next_node = j
                            break
                if next_node is None or next_node == 0:
                    route.append(0)
                    break
                route.append(next_node)
                visited_nodes.add(next_node)
                current = next_node
            
            if len(route) > 2:
                routes.append(route)
        
        for i in range(1, N):
            total_q = sum(model.cbGetSolution(q[i, k, t]) for k in range(K))
            if total_q > 1e-6:
                deliveries[i] = round(total_q, 6)
        
        inventories = {0: round(model.cbGetSolution(I[0, t]), 6)}
        for i in range(1, N):
            inventories[i] = round(model.cbGetSolution(I[i, t]), 6)
        
        routes_per_period.append(routes)
        deliveries_per_period.append(deliveries)
        inventories_per_period.append(inventories)
    
    return build_solution_dict(instance, routes_per_period, deliveries_per_period, inventories_per_period)

def extract_solution_final(model, instance, x, y, q, I, T, K, N, n):
    """Extract solution after optimization using .X attribute."""
    routes_per_period = []
    deliveries_per_period = []
    inventories_per_period = []
    
    for t in range(T):
        routes = []
        deliveries = {}
        
        for k in range(K):
            route_start_val = sum(x[0, j, k, t].X for j in range(1, N))
            if route_start_val < 0.5:
                continue
            
            route = [0]
            current = 0
            visited_nodes = set()
            while True:
                next_node = None
                for j in range(N):
                    if j != current and j not in visited_nodes:
                        if (current, j, k, t) in x and x[current, j, k, t].X > 0.5:
                            next_node = j
                            break
                if next_node is None or next_node == 0:
                    route.append(0)
                    break
                route.append(next_node)
                visited_nodes.add(next_node)
                current = next_node
            
            if len(route) > 2:
                routes.append(route)
        
        for i in range(1, N):
            total_q = sum(q[i, k, t].X for k in range(K))
            if total_q > 1e-6:
                deliveries[i] = round(total_q, 6)
        
        inventories = {0: round(I[0, t].X, 6)}
        for i in range(1, N):
            inventories[i] = round(I[i, t].X, 6)
        
        routes_per_period.append(routes)
        deliveries_per_period.append(deliveries)
        inventories_per_period.append(inventories)
    
    return build_solution_dict(instance, routes_per_period, deliveries_per_period, inventories_per_period)

def solve_heuristic(instance, time_limit, logger=None):
    """Constructive heuristic for IRP."""
    n = instance['num_customers']
    T = instance['num_periods']
    K = instance['num_vehicles']
    Q = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']
    
    routes_per_period = []
    deliveries_per_period = []
    inventories_per_period = []
    
    depot_inv = depot['initial_inventory']
    cust_inv = [customers[i]['initial_inventory'] for i in range(n)]
    
    for t in range(T):
        depot_inv += depot['production_per_period']
        
        # Determine customers needing delivery
        need_list = []
        for i in range(n):
            # Minimum delivery to satisfy: end_inv >= min_inventory
            # end_inv = cust_inv[i] + delivery - demand >= min_inventory
            min_delivery = customers[i]['demand_per_period'] + customers[i]['min_inventory'] - cust_inv[i]
            if min_delivery > 0:
                need_list.append((min_delivery, i))
        
        # Sort by urgency (descending)
        need_list.sort(reverse=True)
        
        routes = []
        deliveries = {}
        remaining_cap = [Q] * K
        vehicle_used = [False] * K
        visited = set()
        
        for min_del, cust_idx in need_list:
            if cust_idx in visited:
                continue
            
            # Desired delivery: fill to max inventory
            desired_delivery = customers[cust_idx]['max_inventory'] + customers[cust_idx]['demand_per_period'] - cust_inv[cust_idx]
            desired_delivery = max(desired_delivery, min_del)
            
            # Find best insertion
            best_k = -1
            best_pos = -1
            best_cost = float('inf')
            
            for k in range(K):
                if not vehicle_used[k]:
                    if remaining_cap[k] >= min_del:
                        cost = dist[0][cust_idx + 1] + dist[cust_idx + 1][0]
                        if cost < best_cost:
                            best_cost = cost
                            best_k = k
                            best_pos = -1  # new route
                else:
                    route = routes[k]
                    for pos in range(1, len(route)):
                        insert_cost = (dist[route[pos-1]][cust_idx + 1] + 
                                      dist[cust_idx + 1][route[pos]] - 
                                      dist[route[pos-1]][route[pos]])
                        if insert_cost < best_cost and remaining_cap[k] >= min_del:
                            best_cost = insert_cost
                            best_k = k
                            best_pos = pos
            
            if best_k >= 0:
                actual_delivery = min(desired_delivery, remaining_cap[best_k], depot_inv)
                actual_delivery = max(actual_delivery, min_del)
                
                if actual_delivery <= remaining_cap[best_k] and actual_delivery <= depot_inv:
                    if best_pos == -1:
                        routes.append([0, cust_idx + 1, 0])
                        vehicle_used[best_k] = True
                    else:
                        routes[best_k].insert(best_pos, cust_idx + 1)
                    
                    deliveries[cust_idx + 1] = actual_delivery
                    remaining_cap[best_k] -= actual_delivery
                    depot_inv -= actual_delivery
                    visited.add(cust_idx)
        
        # Update inventories
        for i in range(n):
            cust_inv[i] = cust_inv[i] + deliveries.get(i + 1, 0) - customers[i]['demand_per_period']
        
        routes_per_period.append(routes)
        deliveries_per_period.append(deliveries)
        inventories = {0: depot_inv}
        for i in range(n):
            inventories[i + 1] = cust_inv[i]
        inventories_per_period.append(inventories)
    
    return build_solution_dict(instance, routes_per_period, deliveries_per_period, inventories_per_period)

def improve_heuristic(instance, solution_dict, time_limit, logger=None):
    """Try to improve heuristic solution with local search."""
    start_time = time.time()
    n = instance['num_customers']
    T = instance['num_periods']
    K = instance['num_vehicles']
    Q = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']
    
    best_obj = solution_dict['objective_value']
    best_solution = solution_dict
    
    # Extract current solution into working format
    routes_per_period = []
    deliveries_per_period = []
    inventories_per_period = []
    
    for t in range(T):
        pk = str(t + 1)
        pd = solution_dict['solution_details']['periods'][pk]
        routes_per_period.append(pd['routes'])
        deliv = {}
        for k_str, v in pd['deliveries'].items():
            deliv[int(k_str)] = v
        deliveries_per_period.append(deliv)
        inv = {}
        inv[0] = pd['inventories']['depot']
        for i in range(1, n + 1):
            inv[i] = pd['inventories'][str(i)]
        inventories_per_period.append(inv)
    
    # Try 2-opt improvement on each route
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.8:
        improved = False
        for t in range(T):
            for r_idx in range(len(routes_per_period[t])):
                route = routes_per_period[t][r_idx]
                if len(route) <= 4:  # depot + at least 2 customers + depot
                    continue
                
                best_route = route[:]
                best_saving = 0
                
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        # Reverse segment between i and j
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        old_cost = sum(dist[route[k]][route[k+1]] for k in range(len(route)-1))
                        new_cost = sum(dist[new_route[k]][new_route[k+1]] for k in range(len(new_route)-1))
                        saving = old_cost - new_cost
                        if saving > best_saving + 1e-9:
                            best_saving = saving
                            best_route = new_route[:]
                
                if best_saving > 1e-9:
                    routes_per_period[t][r_idx] = best_route
                    improved = True
    
    # Rebuild solution
    new_solution = build_solution_dict(instance, routes_per_period, deliveries_per_period, inventories_per_period)
    
    if new_solution['objective_value'] < best_obj:
        return new_solution
    return best_solution

def solve_mip_enhanced(instance, time_limit, logger=None):
    """Enhanced MIP with additional valid inequalities."""
    import gurobipy as gp
    from gurobipy import GRB
    
    n = instance['num_customers']
    T = instance['num_periods']
    K = instance['num_vehicles']
    Q = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']
    
    N = n + 1
    
    num_x_vars = N * (N - 1) * K * T
    if num_x_vars > 150000:
        return None
    
    model = gp.Model("IRP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0
    model.Params.Cuts = 1
    
    # Decision variables
    x = {}
    for t in range(T):
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    if i != j:
                        x[i, j, k, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}_{t}")
    
    y = {}
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                y[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}_{t}")
    
    q = {}
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                q[i, k, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                           ub=customers[i-1]['max_inventory'],
                                           name=f"q_{i}_{k}_{t}")
    
    I = {}
    for t in range(T):
        I[0, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=depot['max_inventory'], name=f"I_0_{t}")
        for i in range(1, N):
            I[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=customers[i-1]['min_inventory'],
                                    ub=customers[i-1]['max_inventory'], name=f"I_{i}_{t}")
    
    # MTZ variables
    u = {}
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                u[i, k, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=1, ub=n, name=f"u_{i}_{k}_{t}")
    
    # Objective
    obj = gp.LinExpr()
    for t in range(T):
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    if i != j:
                        obj += dist[i][j] * x[i, j, k, t]
    
    for t in range(T):
        obj += depot['holding_cost'] * I[0, t]
        for i in range(1, N):
            obj += customers[i-1]['holding_cost'] * I[i, t]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Flow conservation
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                model.addConstr(
                    gp.quicksum(x[j, i, k, t] for j in range(N) if j != i) == y[i, k, t])
                model.addConstr(
                    gp.quicksum(x[i, j, k, t] for j in range(N) if j != i) == y[i, k, t])
    
    # Route structure
    for t in range(T):
        for k in range(K):
            model.addConstr(
                gp.quicksum(x[0, j, k, t] for j in range(1, N)) <= 1)
            model.addConstr(
                gp.quicksum(x[i, 0, k, t] for i in range(1, N)) ==
                gp.quicksum(x[0, j, k, t] for j in range(1, N)))
    
    # Vehicle capacity
    for t in range(T):
        for k in range(K):
            model.addConstr(
                gp.quicksum(q[i, k, t] for i in range(1, N)) <= Q)
    
    # Visit at most once
    for t in range(T):
        for i in range(1, N):
            model.addConstr(
                gp.quicksum(y[i, k, t] for k in range(K)) <= 1)
    
    # Link delivery to visit
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                model.addConstr(q[i, k, t] <= customers[i-1]['max_inventory'] * y[i, k, t])
    
    # Inventory balance - depot
    for t in range(T):
        prev_inv = depot['initial_inventory'] if t == 0 else I[0, t-1]
        total_del = gp.quicksum(q[i, k, t] for i in range(1, N) for k in range(K))
        model.addConstr(I[0, t] == prev_inv + depot['production_per_period'] - total_del)
    
    # Inventory balance - customers
    for t in range(T):
        for i in range(1, N):
            prev_inv = customers[i-1]['initial_inventory'] if t == 0 else I[i, t-1]
            del_i = gp.quicksum(q[i, k, t] for k in range(K))
            model.addConstr(I[i, t] == prev_inv + del_i - customers[i-1]['demand_per_period'])
    
    # MTZ subtour elimination
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                for j in range(1, N):
                    if i != j:
                        model.addConstr(
                            u[i, k, t] - u[j, k, t] + n * x[i, j, k, t] <= n - 1)
    
    # Additional valid inequalities: link MTZ to y
    for t in range(T):
        for k in range(K):
            for i in range(1, N):
                model.addConstr(u[i, k, t] <= n * y[i, k, t])
                model.addConstr(u[i, k, t] >= y[i, k, t])
    
    # Callback
    best_solutions = []
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            try:
                sol = extract_solution_cb(model, instance, x, y, q, I, T, K, N, n)
                if sol is not None:
                    best_solutions.append((obj_val, sol))
                    if logger:
                        logger.log_solution(obj_val, sol)
            except:
                if logger:
                    logger.log(obj_val)
    
    model.optimize(callback)
    
    if model.SolCount > 0:
        try:
            sol = extract_solution_final(model, instance, x, y, q, I, T, K, N, n)
            if sol is not None:
                return sol
        except:
            pass
    
    if best_solutions:
        best_solutions.sort(key=lambda x: x[0])
        return best_solutions[0][1]
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Inventory Routing Problem Solver')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    start_time = time.time()
    instance = load_instance(args.instance_path)
    
    solution = None
    
    # Phase 1: Try MIP
    elapsed = time.time() - start_time
    mip_time = max(args.time_limit - elapsed - 2, 5)  # Reserve time for heuristic
    
    if mip_time > 3:
        try:
            solution = solve_mip_enhanced(instance, mip_time, logger)
        except Exception as e:
            solution = None
    
    # Phase 2: If MIP failed, use heuristic
    if solution is None:
        try:
            solution = solve_heuristic(instance, args.time_limit, logger)
            if logger and solution:
                logger.log_solution(solution['objective_value'], solution)
        except Exception as e:
            pass
    
    # Phase 3: Try to improve heuristic solution
    if solution is not None:
        elapsed = time.time() - start_time
        remaining = args.time_limit - elapsed
        if remaining > 2:
            try:
                improved = improve_heuristic(instance, solution, remaining, logger)
                if improved['objective_value'] < solution['objective_value']:
                    solution = improved
                    if logger:
                        logger.log_solution(solution['objective_value'], solution)
            except:
                pass
    
    # Fallback: ensure we have some solution
    if solution is None:
        solution = solve_heuristic(instance, 1, None)
        if logger and solution:
            logger.log_solution(solution['objective_value'], solution)
    
    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()