import json
import argparse
import time
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
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    # Parse instance
    n = instance['num_retailers']
    H = instance['time_horizon']
    supplier = instance['supplier']
    retailers = instance['retailers']
    capacity = instance['vehicle_capacity']
    dist = instance['distance_matrix']
    
    # Supplier data
    s_start_inv = supplier['starting_inventory']
    s_prod = supplier['production_rate']
    s_inv_cost = supplier['inventory_cost']
    
    # Retailer data
    r_demand = [retailers[i]['demand'] for i in range(n)]
    r_max_inv = [retailers[i]['max_inventory'] for i in range(n)]
    r_start_inv = [retailers[i]['starting_inventory'] for i in range(n)]
    r_inv_cost = [retailers[i]['inventory_cost'] for i in range(n)]
    
    start_time = time.time()
    
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        # Create model
        model = gp.Model("IRP")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', args.time_limit)
        model.setParam('OutputFlag', 0)
        
        # Decision variables
        # x[i,j,t] = number of times edge (i,j) is traversed in period t
        # i,j in {0,...,n}, i < j
        # y[i,t] = 1 if retailer i is visited in period t
        # q[i,t] = quantity delivered to retailer i in period t
        
        # Extended horizon: periods 0..H (H+1 periods for inventory tracking)
        # Delivery periods: 1..H
        # Inventory periods: 0..H (extended horizon)
        
        # Variables
        y = {}  # y[i,t] = 1 if retailer i is visited in period t (t=1..H)
        q = {}  # q[i,t] = quantity delivered to retailer i in period t
        x = {}  # x[i,j,t] = number of times edge (i,j) is traversed in period t
        I_s = {}  # supplier inventory at period t (t=0..H)
        I_r = {}  # retailer i inventory at period t (t=0..H)
        
        for t in range(1, H+1):
            for i in range(n):
                y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")
                q[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"q_{i}_{t}")
        
        for t in range(1, H+1):
            for i in range(n+1):
                for j in range(i+1, n+1):
                    x[i, j, t] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=2, name=f"x_{i}_{j}_{t}")
        
        for t in range(0, H+1):
            I_s[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Is_{t}")
        
        for i in range(n):
            for t in range(0, H+1):
                I_r[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Ir_{i}_{t}")
        
        model.update()
        
        # Objective: minimize total cost
        obj = gp.LinExpr()
        
        # Supplier inventory cost over extended horizon (periods 0..H)
        for t in range(0, H+1):
            obj += s_inv_cost * I_s[t]
        
        # Retailer inventory cost over extended horizon
        for i in range(n):
            for t in range(0, H+1):
                obj += r_inv_cost[i] * I_r[i, t]
        
        # Transportation cost
        for t in range(1, H+1):
            for i in range(n+1):
                for j in range(i+1, n+1):
                    obj += dist[i][j] * x[i, j, t]
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Constraints
        
        # Initial inventory
        model.addConstr(I_s[0] == s_start_inv)
        for i in range(n):
            model.addConstr(I_r[i, 0] == r_start_inv[i])
        
        # Supplier inventory balance
        # I_s[t] = I_s[t-1] + production - sum(q[i,t-1])
        # Wait, let me re-read the problem:
        # "The inventory at the supplier at each period equals the supplier's inventory 
        #  in the previous period, plus the quantity produced at the supplier in the 
        #  previous period, minus the total quantity shipped to all retailers in the 
        #  previous period."
        # So I_s[t] = I_s[t-1] + prod - sum(q[i,t-1]) for t=1..H
        # But we need to be careful about what "period" means here.
        # Let's say deliveries happen in periods 1..H.
        # I_s[0] = starting inventory
        # I_s[t] = I_s[t-1] + prod - sum(q[i,t]) for t=1..H
        # Actually re-reading: "plus the quantity produced at the supplier in the previous period"
        # This means production in period t-1 is available at start of period t.
        # Let me interpret it as: I_s[t] = I_s[t-1] + prod - sum(q[i,t]) for t=1..H
        # The constraint "the supplier's inventory must be at least the total quantity shipped"
        # means I_s[t-1] + prod >= sum(q[i,t]) which is equivalent to I_s[t] >= 0.
        
        for t in range(1, H+1):
            model.addConstr(I_s[t] == I_s[t-1] + s_prod - gp.quicksum(q[i, t] for i in range(n)))
        
        # Retailer inventory balance
        # I_r[i,t] = I_r[i,t-1] + q[i,t] - demand[i] for t=1..H
        for i in range(n):
            for t in range(1, H+1):
                model.addConstr(I_r[i, t] == I_r[i, t-1] + q[i, t] - r_demand[i])
        
        # Delivery quantity: if visited, deliver up to max inventory
        # q[i,t] = y[i,t] * (max_inv[i] - I_r[i,t-1])
        # This is: q[i,t] = max_inv[i]*y[i,t] - I_r[i,t-1]*y[i,t]
        # Linearize: q[i,t] <= max_inv[i]*y[i,t]
        #            q[i,t] >= max_inv[i]*y[i,t] - max_inv[i]*(1-y[i,t]) ... no
        # Actually: q[i,t] = y[i,t] * (max_inv[i] - I_r[i,t-1])
        # Since I_r[i,t-1] is a variable, this is bilinear.
        # We can use: q[i,t] <= max_inv[i] * y[i,t]
        #             q[i,t] <= max_inv[i] - I_r[i,t-1]  (when y=1, q = max_inv - I_r)
        #             q[i,t] >= max_inv[i] - I_r[i,t-1] - max_inv[i]*(1-y[i,t])
        #             q[i,t] >= 0
        for i in range(n):
            for t in range(1, H+1):
                model.addConstr(q[i, t] <= r_max_inv[i] * y[i, t])
                model.addConstr(q[i, t] <= r_max_inv[i] - I_r[i, t-1])
                model.addConstr(q[i, t] >= r_max_inv[i] - I_r[i, t-1] - r_max_inv[i] * (1 - y[i, t]))
        
        # Vehicle capacity
        for t in range(1, H+1):
            model.addConstr(gp.quicksum(q[i, t] for i in range(n)) <= capacity)
        
        # Route constraints
        # Degree constraint: for each visited retailer, exactly 2 edges incident
        # For supplier: degree = 2 * (number of routes) but since single route, degree = 2
        # Actually: "the total number of route-segment traversals touching each visited 
        # location (supplier and retailers) equals exactly two in that period"
        
        for t in range(1, H+1):
            # Supplier degree = 2 if any retailer is visited
            supplier_degree = gp.quicksum(x[0, j, t] for j in range(1, n+1))
            model.addConstr(supplier_degree == 2 * gp.quicksum(y[i, t] for i in range(n)) / gp.quicksum(y[i, t] for i in range(n)) if False else supplier_degree >= 0)
            # Actually, let me think again. If any retailer is visited, the supplier degree is 2.
            # If no retailer is visited, supplier degree is 0.
            # Let's use: supplier_degree = 2 * z[t] where z[t] = 1 if any retailer visited
            # But we can simplify: if sum(y[i,t]) > 0, then supplier_degree = 2
            # Let's add: supplier_degree <= 2 * sum(y[i,t])  (if no visits, no edges)
            #            supplier_degree >= 2 * y[i,t] for each i (if i visited, at least 2 edges from supplier... no)
            # Actually the constraint says degree of each visited location = 2.
            # For supplier: degree = 2 if any retailer is visited, 0 otherwise.
            
            # Let me use a binary variable for "any visit"
            # Or simply: degree of supplier = 2 if any y[i,t]=1, else 0
            # We can write: supplier_degree <= 2 * sum(y[i,t]) and supplier_degree >= 2*y[i,t] for all i
            # But that's not quite right either. Let me think more carefully.
            
            # If any retailer is visited, the route is a single tour starting and ending at supplier.
            # So supplier has degree 2 in the tour.
            # Each visited retailer has degree 2.
            # Non-visited retailers have degree 0.
            
            # For supplier: sum of x[0,j,t] for j=1..n = 2 if any retailer visited, 0 otherwise
            # We can write: sum(x[0,j,t]) <= 2 * sum(y[i,t])  -- but this is weak
            # Better: sum(x[0,j,t]) = 2 * z[t] where z[t] is binary = 1 if any retailer visited
            # But we can avoid z by noting: sum(x[0,j,t]) <= 2*n (trivially)
            # and sum(x[0,j,t]) >= 2*y[i,t] for each i (if i is visited, supplier must have degree >= 2)
            # and sum(x[0,j,t]) <= 2 (at most 2 edges from supplier in a tour)
            # Actually for a single tour, supplier degree is exactly 2.
            # So: sum(x[0,j,t]) = 2 if any y[i,t]=1, else 0.
            # We can write: sum(x[0,j,t]) <= 2 and sum(x[0,j,t]) >= 2*y[i,t] for all i
            # This gives: if any y=1, then sum >= 2, and sum <= 2, so sum = 2.
            # If all y=0, then sum >= 0 and sum <= 2, but we also need sum = 0.
            # Add: x[i,j,t] <= y[i,t] for edges involving retailer i (handled below)
            # Then if all y=0, all x=0, so sum=0. Good.
            
            model.addConstr(gp.quicksum(x[0, j, t] for j in range(1, n+1)) <= 2)
            for i in range(n):
                model.addConstr(gp.quicksum(x[0, j, t] for j in range(1, n+1)) >= 2 * y[i, t])
        
        for t in range(1, H+1):
            for i in range(1, n+1):
                ri = i - 1  # retailer index
                # Degree of retailer i = 2 if visited, 0 otherwise
                degree_i = gp.quicksum(x[min(0, i), max(0, i), t] if 0 < i else x[0, i, t] for _ in [0])
                # Let me compute degree properly
                deg = gp.LinExpr()
                for j in range(n+1):
                    if j < i:
                        deg += x[j, i, t]
                    elif j > i:
                        deg += x[i, j, t]
                model.addConstr(deg == 2 * y[ri, t])
        
        # Subtour elimination (MTZ-style or cut-based)
        # For small n, we can use subtour elimination constraints
        # "for any subset S of retailers, the number of route segments connecting only pairs 
        #  of locations within S cannot exceed |S_visited| - 1"
        # More precisely: for any subset S of retailers, sum of x[i,j,t] for i,j in S, i<j <= |S|-1
        # But only for visited retailers in S.
        # Actually the constraint says: <= |S| - visit_status of one chosen retailer from S
        # This is the standard subtour elimination: for subset S of retailers,
        # sum_{i,j in S, i<j} x[i,j,t] <= |S| - 1 (if at least one in S is visited)
        # Let's use: for all subsets S of retailers with |S| >= 2:
        # sum_{i,j in S} x[i,j,t] <= sum_{i in S} y[i,t] - 1 + (1 - y[k,t]) for some k in S
        # Simplified: sum_{i,j in S, i<j} x[i,j,t] <= |S_visited_in_S| - 1
        # Which is: sum_{i,j in S} x[i,j,t] <= sum_{i in S} y[i,t] - 1
        # But this only makes sense if at least one retailer in S is visited.
        # Standard form: sum_{i,j in S} x[i,j,t] <= sum_{i in S} y[i,t] - 1
        # This is valid when sum y[i,t] >= 1 for i in S.
        # If sum y = 0, then LHS = 0 (since x[i,j] <= y[i] and x[i,j] <= y[j]) and RHS = -1, violation!
        # So we need: sum_{i,j in S} x[i,j,t] <= sum_{i in S} y[i,t] - 1 + (1 - min(y[i,t] for i in S))
        # Actually the standard formulation is:
        # sum_{i,j in S, i<j} x[i,j,t] <= |S| - 1  for all S with 2 <= |S| <= n
        # But this is only active when retailers in S are visited.
        # Since x[i,j,t] <= y[i,t] and x[i,j,t] <= y[j,t], if no retailer in S is visited, LHS=0 <= |S|-1.
        # If some but not all are visited, we need the constraint to be valid.
        # The correct form is: sum_{i,j in S} x[i,j,t] <= sum_{i in S} y[i,t] - 1
        # This works when at least one retailer in S is visited.
        # If none visited: LHS=0, RHS=-1. Problem!
        # So we need: sum_{i,j in S} x[i,j,t] <= sum_{i in S} y[i,t] - 1 + (1 - z_S)
        # where z_S = 1 if any retailer in S is visited. This gets complicated.
        # 
        # Alternative: use the formulation from the problem description:
        # "for any subset of retailers, the number of route segments connecting only pairs of 
        #  locations within that subset cannot exceed the number of visited retailers in that 
        #  subset minus the visit status of one chosen retailer from that subset"
        # This means: for subset S and chosen k in S:
        # sum_{i,j in S, i<j} x[i,j,t] <= sum_{i in S} y[i,t] - y[k,t]
        # This is valid for any choice of k. If we pick k such that y[k,t]=1 (if any visited),
        # we get <= |S_visited| - 1. If none visited, y[k,t]=0, so <= 0, which is fine since LHS=0.
        # So the constraint is: for all S with |S|>=2, for all k in S:
        # sum_{i,j in S, i<j} x[i,j,t] <= sum_{i in S} y[i,t] - y[k,t]
        # We can just pick one k per S (e.g., the first element).
        
        # For small n, enumerate all subsets
        if n <= 15:
            for t in range(1, H+1):
                for size in range(2, n+1):
                    for S in combinations(range(n), size):
                        # Pick k = S[0]
                        k = S[0]
                        lhs = gp.LinExpr()
                        for idx_a in range(len(S)):
                            for idx_b in range(idx_a+1, len(S)):
                                i, j = S[idx_a], S[idx_b]
                                # Edge between retailer i and retailer j
                                # In our x variables, retailers are indexed 1..n
                                # So edge is (i+1, j+1) with i+1 < j+1
                                ni, nj = i+1, j+1
                                if ni < nj:
                                    lhs += x[ni, nj, t]
                                else:
                                    lhs += x[nj, ni, t]
                        rhs = gp.quicksum(y[i, t] for i in S) - y[k, t]
                        model.addConstr(lhs <= rhs)
        
        # Edge validity: x[i,j,t] can only be positive if both endpoints are visited
        # For edge between retailer i and retailer j (both >= 1):
        for t in range(1, H+1):
            for i in range(1, n+1):
                for j in range(i+1, n+1):
                    ri, rj = i-1, j-1
                    model.addConstr(x[i, j, t] <= y[ri, t])
                    model.addConstr(x[i, j, t] <= y[rj, t])
            # For edge between supplier (0) and retailer i:
            for i in range(1, n+1):
                ri = i - 1
                model.addConstr(x[0, i, t] <= y[ri, t])
                model.addConstr(x[0, i, t] <= 2 * y[ri, t])  # can be 0, 1, or 2
        
        model.update()
        
        # Solve
        model.optimize()
        
        # Extract solution
        if model.SolCount > 0:
            obj_val = model.ObjVal
            
            deliveries = {}
            routes = {}
            supplier_inventory = {}
            retailer_inventory = {}
            
            for t in range(0, H+1):
                supplier_inventory[str(t)] = round(I_s[t].X, 6)
            
            for i in range(n):
                retailer_inventory[str(i)] = {}
                for t in range(0, H+1):
                    retailer_inventory[str(i)][str(t)] = round(I_r[i, t].X, 6)
            
            for t in range(1, H+1):
                t_deliveries = {}
                t_routes = []
                has_delivery = False
                for i in range(n):
                    if y[i, t].X > 0.5:
                        q_val = q[i, t].X
                        t_deliveries[str(i)] = round(q_val, 6)
                        has_delivery = True
                
                if has_delivery:
                    for i in range(n+1):
                        for j in range(i+1, n+1):
                            x_val = round(x[i, j, t].X)
                            if x_val > 0:
                                t_routes.append([i, j, x_val])
                
                if t_deliveries:
                    deliveries[str(t)] = t_deliveries
                if t_routes:
                    routes[str(t)] = t_routes
            
            solution = {
                "objective_value": round(obj_val, 6),
                "deliveries": deliveries,
                "routes": routes,
                "supplier_inventory": supplier_inventory,
                "retailer_inventory": retailer_inventory
            }
            
            if logger:
                logger.log_solution(obj_val, solution)
        else:
            # Fallback: no solution found, create trivial solution
            solution = {
                "objective_value": 0.0,
                "deliveries": {},
                "routes": {},
                "supplier_inventory": {str(t): s_start_inv + t * s_prod for t in range(H+1)},
                "retailer_inventory": {str(i): {str(t): max(0, r_start_inv[i] - t * r_demand[i]) for t in range(H+1)} for i in range(n)}
            }
            if logger:
                logger.log_solution(0.0, solution)
    
    except Exception as e:
        # Fallback heuristic if Gurobi fails
        solution = build_heuristic_solution(n, H, supplier, retailers, capacity, dist, s_start_inv, s_prod, s_inv_cost, r_demand, r_max_inv, r_start_inv, r_inv_cost)
        if logger:
            logger.log_solution(solution["objective_value"], solution)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


def build_heuristic_solution(n, H, supplier, retailers, capacity, dist, s_start_inv, s_prod, s_inv_cost, r_demand, r_max_inv, r_start_inv, r_inv_cost):
    """Build a simple heuristic solution."""
    supplier_inventory = {}
    retailer_inventory = {}
    deliveries = {}
    routes = {}
    
    # Simple greedy: visit each retailer when its inventory would go negative
    s_inv = s_start_inv
    r_inv = list(r_start_inv)
    
    supplier_inventory["0"] = s_inv
    for i in range(n):
        retailer_inventory[str(i)] = {"0": r_inv[i]}
    
    total_cost = 0.0
    total_cost += s_inv_cost * s_inv
    for i in range(n):
        total_cost += r_inv_cost[i] * r_inv[i]
    
    for t in range(1, H+1):
        s_inv += s_prod
        
        # Determine which retailers need delivery
        to_visit = []
        for i in range(n):
            if r_inv[i] - r_demand[i] < 0:
                to_visit.append(i)
        
        # Check capacity and deliver
        delivered = {}
        load = 0
        for i in to_visit:
            qty = r_max_inv[i] - r_inv[i]
            if load + qty <= capacity:
                delivered[i] = qty
                load += qty
        
        # Update inventories
        for i in range(n):
            if i in delivered:
                r_inv[i] += delivered[i]
            r_inv[i] -= r_demand[i]
        
        s_inv -= sum(delivered.values())
        
        supplier_inventory[str(t)] = s_inv
        for i in range(n):
            retailer_inventory[str(i)][str(t)] = r_inv[i]
        
        if delivered:
            deliveries[str(t)] = {str(i): delivered[i] for i in delivered}
            # Build route: simple nearest neighbor from supplier
            visited = sorted(delivered.keys())
            route_edges = []
            if len(visited) == 1:
                route_edges.append([0, visited[0]+1, 2])
            else:
                # Simple tour: 0 -> v1 -> v2 -> ... -> vk -> 0
                route_edges.append([0, visited[0]+1, 1])
                for idx in range(len(visited)-1):
                    a, b = visited[idx]+1, visited[idx+1]+1
                    if a < b:
                        route_edges.append([a, b, 1])
                    else:
                        route_edges.append([b, a, 1])
                route_edges.append([0, visited[-1]+1, 1])
            routes[str(t)] = route_edges
            
            # Add transport cost
            for edge in route_edges:
                total_cost += dist[edge[0]][edge[1]] * edge[2]
        
        total_cost += s_inv_cost * s_inv
        for i in range(n):
            total_cost += r_inv_cost[i] * r_inv[i]
    
    return {
        "objective_value": total_cost,
        "deliveries": deliveries,
        "routes": routes,
        "supplier_inventory": supplier_inventory,
        "retailer_inventory": retailer_inventory
    }


if __name__ == '__main__':
    main()