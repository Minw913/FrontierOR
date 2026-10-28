import argparse
import json
import time
import math
import itertools
from copy import deepcopy

def solve_irp(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    n = data['num_retailers']
    H = data['time_horizon']
    supplier = data['supplier']
    retailers = data['retailers']
    C = data['vehicle_capacity']
    dist = data['distance_matrix']
    
    sup_id = supplier['id']
    sup_start_inv = supplier['starting_inventory']
    sup_prod = supplier['production_rate']
    sup_cost = supplier['inventory_cost']
    
    ret_ids = [r['id'] for r in retailers]
    ret_demand = [r['demand'] for r in retailers]
    ret_max_inv = [r['max_inventory'] for r in retailers]
    ret_start_inv = [r['starting_inventory'] for r in retailers]
    ret_cost = [r['inventory_cost'] for r in retailers]
    
    # Extended horizon: periods 1..H+1 for inventory tracking
    # Deliveries happen in periods 1..H
    # Period 0 is the starting state
    
    # Try Gurobi MIP formulation for small instances, heuristic for larger
    
    def solve_tsp_route(visited_retailers, dist_matrix):
        """Solve TSP for a set of visited retailers + supplier (node 0)."""
        if not visited_retailers:
            return [], 0
        if len(visited_retailers) == 1:
            r = visited_retailers[0]
            # Direct round trip: supplier -> retailer -> supplier
            cost = 2 * dist_matrix[0][r]
            return [[0, r, 2]], cost
        
        # For small sets, enumerate or use nearest neighbor
        nodes = [0] + list(visited_retailers)
        
        if len(visited_retailers) <= 10:
            # Try nearest neighbor + 2-opt
            best_tour, best_cost = nearest_neighbor_tsp(nodes, dist_matrix)
            best_tour, best_cost = two_opt(best_tour, dist_matrix)
            return tour_to_segments(best_tour, dist_matrix), best_cost
        else:
            best_tour, best_cost = nearest_neighbor_tsp(nodes, dist_matrix)
            best_tour, best_cost = two_opt(best_tour, dist_matrix)
            return tour_to_segments(best_tour, dist_matrix), best_cost
    
    def nearest_neighbor_tsp(nodes, dist_matrix):
        """Nearest neighbor heuristic starting from depot (node 0)."""
        unvisited = set(nodes[1:])  # exclude depot
        tour = [0]
        current = 0
        total_cost = 0
        while unvisited:
            nearest = min(unvisited, key=lambda x: dist_matrix[current][x])
            total_cost += dist_matrix[current][nearest]
            tour.append(nearest)
            current = nearest
            unvisited.remove(nearest)
        total_cost += dist_matrix[current][0]
        tour.append(0)
        return tour, total_cost
    
    def two_opt(tour, dist_matrix):
        """2-opt improvement for TSP tour."""
        improved = True
        best_cost = sum(dist_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1))
        while improved:
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour) - 1):
                    old_cost = dist_matrix[tour[i-1]][tour[i]] + dist_matrix[tour[j]][tour[j+1]]
                    new_cost = dist_matrix[tour[i-1]][tour[j]] + dist_matrix[tour[i]][tour[j+1]]
                    if new_cost < old_cost - 1e-9:
                        tour[i:j+1] = reversed(tour[i:j+1])
                        best_cost -= (old_cost - new_cost)
                        improved = True
        return tour, best_cost
    
    def tour_to_segments(tour, dist_matrix):
        """Convert a tour [0, r1, r2, ..., 0] to route segments."""
        segments = []
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i+1]
            if a == b:
                continue
            segments.append([a, b, 1])
        
        # Merge segments: if there are two segments involving the same pair, merge
        # For supplier segments (involving 0), we can have k=2
        seg_dict = {}
        for seg in segments:
            a, b, k = seg
            key = (min(a, b), max(a, b))
            if key in seg_dict:
                seg_dict[key] += k
            else:
                seg_dict[key] = k
        
        result = []
        for (a, b), k in seg_dict.items():
            result.append([a, b, k])
        
        return result
    
    def route_cost(segments, dist_matrix):
        total = 0
        for seg in segments:
            a, b, k = seg
            total += dist_matrix[a][b] * k
        return total
    
    # Greedy/heuristic approach
    # For each period, decide which retailers to visit
    
    def compute_greedy_solution():
        """Greedy heuristic: visit retailers when their inventory would go negative."""
        sup_inv = [0] * (H + 2)  # periods 0..H+1
        ret_inv = [[0] * (H + 2) for _ in range(n)]
        
        sup_inv[0] = sup_start_inv
        for i in range(n):
            ret_inv[i][0] = ret_start_inv[i]
        
        deliveries = {}  # period -> {retailer_id: quantity}
        routes = {}  # period -> list of segments
        total_transport_cost = 0
        
        for t in range(1, H + 1):
            # Supplier inventory at start of period t (before shipping)
            # sup_inv[t] = sup_inv[t-1] + sup_prod - total_shipped[t-1]
            # For period 1: sup_inv[1] = sup_inv[0] + sup_prod - shipped[0]
            # But shipping happens in period t, not t-1
            # Let me re-read the problem...
            
            # "The inventory at the supplier at each period equals the supplier's inventory 
            # in the previous period, plus the quantity produced at the supplier in the 
            # previous period, minus the total quantity shipped to all retailers in the 
            # previous period."
            
            # So sup_inv[t] = sup_inv[t-1] + production[t-1] - shipped[t-1]
            # This means production and shipping happen in the previous period
            
            # "The inventory at each retailer at each period equals the retailer's inventory 
            # in the previous period, plus the quantity received from the supplier in the 
            # previous period, minus the quantity consumed by the retailer in the previous period."
            
            # ret_inv[i][t] = ret_inv[i][t-1] + received[i][t-1] - demand[i]
            # (demand consumed each period)
            
            pass
        
        # Let me redo this more carefully
        # Period 0: starting inventories
        # Decisions are made for periods 1..H (delivery periods)
        # But the inventory updates use "previous period" production/shipping/consumption
        
        # Actually, re-reading: deliveries happen in periods, and inventory is tracked.
        # Let me think of it as:
        # At period t (for t=1..H):
        #   - Decide which retailers to visit and ship product
        #   - sup_inv[t] = sup_inv[t-1] + production_in_period_(t-1) - shipped_in_period_(t-1)
        
        # Wait, this is confusing. Let me use a different indexing.
        # Let's say decisions (deliveries) are indexed by period t = 1..H
        # shipped[t] = total shipped in period t
        # received[i][t] = quantity received by retailer i in period t
        
        # sup_inv[t] = sup_inv[t-1] + sup_prod - shipped[t-1]  for t >= 1
        # But what about shipped[0]? There's no period 0 delivery, so shipped[0] = 0
        
        # Hmm, let me re-read once more:
        # "The inventory at the supplier at each period equals the supplier's inventory 
        # in the previous period, plus the quantity produced at the supplier in the 
        # previous period, minus the total quantity shipped to all retailers in the 
        # previous period."
        
        # So for period t: sup_inv[t] = sup_inv[t-1] + prod[t-1] - ship[t-1]
        # And the constraint: sup_inv[t] >= ship[t] for t=1..H
        
        # For retailers: ret_inv[i][t] = ret_inv[i][t-1] + recv[i][t-1] - demand[i]
        # And ret_inv[i][t] >= 0 for t=1..H+1
        
        # The "extended horizon" includes period H+1 for inventory.
        # Inventory costs are assessed for periods 1..H+1.
        
        # Actually wait - "inventory costs at both the supplier and all retailers are 
        # assessed over this extended horizon" - what periods exactly?
        # "No retailer's inventory may ever fall below zero at any period from one 
        # through the end of the extended horizon, which includes one period beyond 
        # the last delivery period."
        # So extended horizon = periods 1..H+1
        
        # Let me re-index. Period 0 = initial state.
        # Periods 1..H = delivery periods.
        # Period H+1 = one period beyond.
        # Inventory is tracked for periods 0..H+1.
        # Inventory costs assessed for periods 1..H+1.
        
        # For t = 1..H+1:
        #   sup_inv[t] = sup_inv[t-1] + sup_prod - ship[t-1]
        #   ret_inv[i][t] = ret_inv[i][t-1] + recv[i][t-1] - demand[i]
        # where ship[t] and recv[i][t] are only defined for t=1..H (delivery periods)
        # ship[0] = 0, recv[i][0] = 0
        # For t = H+1: ship[H] is the shipping in period H, which is a delivery period
        
        # Wait, I think the production and shipping happen "in" the period, and the 
        # inventory update formula tells us the inventory "at" the next period.
        
        # Let me just be very explicit:
        # sup_inv[0] = starting_inventory
        # For t = 1, 2, ..., H+1:
        #   sup_inv[t] = sup_inv[t-1] + sup_prod - total_shipped_in_period_(t-1)
        #   where total_shipped_in_period_0 = 0 (no delivery before period 1)
        #   and total_shipped_in_period_t for t=1..H are our decision variables
        #   For t=H+1: we need shipped in period H
        
        # Hmm wait. Deliveries happen in periods 1..H. But what about period 0?
        # There are no deliveries in period 0. And for period H+1, there are no deliveries either.
        
        # Actually I think the formulation is:
        # ship[t] for t=1..H (delivery decision in period t)
        # sup_inv[1] = sup_inv[0] + sup_prod - ship[1]  ?? 
        # No, that doesn't match "previous period" language.
        
        # Let me try: production and consumption happen continuously. 
        # At the START of period t, we have sup_inv[t] and ret_inv[i][t].
        # During period t, we deliver ship[t] to retailers (decision for t=1..H).
        # At the END of period t / START of period t+1:
        #   sup_inv[t+1] = sup_inv[t] + sup_prod - ship[t]
        #   ret_inv[i][t+1] = ret_inv[i][t] + recv[i][t] - demand[i]
        
        # And constraints:
        #   sup_inv[t] >= ship[t] for t=1..H (can't ship more than available AT the period)
        #   ret_inv[i][t] >= 0 for t=1..H+1
        
        # Inventory costs: sum over t=1..H+1 of sup_inv[t] * sup_cost + sum_i ret_inv[i][t] * ret_cost[i]
        # Wait, "assessed over this extended horizon" - does it include period 0?
        # "Inventory costs at both the supplier and all retailers are assessed over this 
        # extended horizon" and the extended horizon is periods 1..H+1? Or 0..H+1?
        
        # The problem says "from one through the end of the extended horizon" for the 
        # non-negativity constraint. And inventory costs are assessed "over this extended 
        # horizon". I'll assume periods 1 to H+1 for costs.
        
        # Actually, I think a more standard interpretation:
        # sup_inv[t] for t=0..H+1
        # Costs: sum_{t=0}^{H} sup_inv[t] * sup_cost  (or t=1..H+1)
        # Let me just go with t=1..H+1 as stated.
        
        # Alright, let me just code it up properly.
        pass
    
    def try_gurobi_solution():
        """Try to solve with Gurobi MIP."""
        import gurobipy as gp
        from gurobipy import GRB
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            return None
        
        model = gp.Model("IRP")
        model.Params.TimeLimit = remaining_time - 2
        model.Params.Threads = 1
        model.Params.MIPGap = 1e-4
        
        # Decision variables
        # y[i][t] = 1 if retailer i is visited in period t
        y = {}
        for i in range(n):
            for t in range(1, H + 1):
                y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")
        
        # q[i][t] = quantity shipped to retailer i in period t
        q = {}
        for i in range(n):
            for t in range(1, H + 1):
                q[i, t] = model.addVar(lb=0, ub=ret_max_inv[i], name=f"q_{i}_{t}")
        
        # Supplier inventory
        Is = {}
        for t in range(0, H + 2):
            Is[t] = model.addVar(lb=0, name=f"Is_{t}")
        
        # Retailer inventory
        Ir = {}
        for i in range(n):
            for t in range(0, H + 2):
                Ir[i, t] = model.addVar(lb=0, name=f"Ir_{i}_{t}")
        
        # Routing variables
        # x[i][j][t] = number of times edge (i,j) is traversed in period t
        # Nodes: 0 = supplier, 1..n = retailers (using their position index, not id)
        # We use dist_matrix with supplier at index 0, retailers at indices 1..n
        
        # For edges between retailers: x is binary (0 or 1)
        # For edges involving supplier: x can be 0, 1, or 2
        
        x = {}
        for t in range(1, H + 1):
            for i in range(n + 1):
                for j in range(i + 1, n + 1):
                    if i == 0:
                        # supplier to retailer j (retailer index j-1 in our arrays, but j in dist matrix)
                        x[i, j, t] = model.addVar(lb=0, ub=2, vtype=GRB.INTEGER, name=f"x_{i}_{j}_{t}")
                    else:
                        x[i, j, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{t}")
        
        model.update()
        
        # Initial inventories
        model.addConstr(Is[0] == sup_start_inv)
        for i in range(n):
            model.addConstr(Ir[i, 0] == ret_start_inv[i])
        
        # Supplier inventory dynamics
        for t in range(1, H + 2):
            if t <= H:
                ship_t_minus_1 = gp.quicksum(q[i, t] for i in range(n)) if t >= 1 else 0
                # Wait, I need to be careful about the timing.
                # sup_inv[t] = sup_inv[t-1] + sup_prod - total_shipped_in_period_(t-1)
                # But we said shipping happens in period t-1 (for t-1 = 1..H)
                # For t=1: sup_inv[1] = sup_inv[0] + sup_prod - shipped_in_period_0
                # shipped_in_period_0 = 0 (no deliveries in period 0)
                
                # Hmm, let me reconsider. The way I defined it above:
                # During period t (t=1..H), we ship q[i,t] to retailer i
                # sup_inv[t+1] = sup_inv[t] + sup_prod - sum_i q[i,t]
                # sup_inv[t] >= sum_i q[i,t]
                
                # For period H+1: no shipping
                # sup_inv[H+1] = sup_inv[H] + sup_prod - sum_i q[i,H]
                
                # But then what about sup_inv[1]?
                # sup_inv[1] = sup_inv[0] + sup_prod - sum_i q[i,0]
                # There's no q[i,0], so:
                # sup_inv[1] = sup_inv[0] + sup_prod
                
                # Wait, that doesn't seem right either. Let me re-read the problem once more.
                
                # "The inventory at the supplier at each period equals the supplier's 
                # inventory in the previous period, plus the quantity produced at the 
                # supplier in the previous period, minus the total quantity shipped to 
                # all retailers in the previous period."
                
                # OK so this is:
                # sup_inv[t] = sup_inv[t-1] + production_rate - shipped[t-1]
                # where shipped[t] = sum_i q[i,t] for t=1..H, and shipped[t]=0 otherwise
                
                # For t=1: sup_inv[1] = sup_inv[0] + sup_prod - shipped[0] = sup_start_inv + sup_prod - 0
                # For t=2: sup_inv[2] = sup_inv[1] + sup_prod - shipped[1]
                # ...
                # For t=H+1: sup_inv[H+1] = sup_inv[H] + sup_prod - shipped[H]
                pass
            pass
        
        # Let me redo this cleanly
        # Remove all constraints and redo
        model.remove(model.getConstrs())
        model.update()
        
        # Re-add initial inventory constraints
        model.addConstr(Is[0] == sup_start_inv)
        for i in range(n):
            model.addConstr(Ir[i, 0] == ret_start_inv[i])
        
        # Supplier inventory dynamics: sup_inv[t] = sup_inv[t-1] + sup_prod - shipped[t-1]
        # shipped[0] = 0 (no delivery in period 0)
        for t in range(1, H + 2):
            if t - 1 >= 1 and t - 1 <= H:
                shipped_prev = gp.quicksum(q[i, t-1] for i in range(n))
            else:
                shipped_prev = 0
            model.addConstr(Is[t] == Is[t-1] + sup_prod - shipped_prev)
        
        # Supplier capacity constraint: sup_inv[t] >= shipped[t] for t=1..H
        for t in range(1, H + 1):
            model.addConstr(Is[t] >= gp.quicksum(q[i, t] for i in range(n)))
        
        # Retailer inventory dynamics: ret_inv[i][t] = ret_inv[i][t-1] + recv[i][t-1] - demand[i]
        # recv[i][0] = 0
        for i in range(n):
            for t in range(1, H + 2):
                if t - 1 >= 1 and t - 1 <= H:
                    recv_prev = q[i, t-1]
                else:
                    recv_prev = 0
                model.addConstr(Ir[i, t] == Ir[i, t-1] + recv_prev - ret_demand[i])
        
        # Retailer inventory non-negativity (already set lb=0, but explicit)
        # Already handled by lb=0 on Ir variables
        
        # Delivery quantity: if visited, ship max_inv - current_inv
        # q[i,t] = (max_inv[i] - Ir[i,t]) * y[i,t]
        # But Ir[i,t] is a variable, so this is nonlinear.
        # However, q[i,t] = max_inv[i] * y[i,t] - Ir[i,t] * y[i,t]
        # We need linearization.
        
        # Actually, the problem says: "Whenever a retailer is visited in a given period, 
        # the quantity shipped must exactly equal the difference between the retailer's 
        # maximum inventory level and its current inventory level at that time"
        
        # So q[i,t] = (max_inv[i] - Ir[i,t]) * y[i,t]
        # If y[i,t] = 0: q[i,t] = 0
        # If y[i,t] = 1: q[i,t] = max_inv[i] - Ir[i,t]
        
        # This means: q[i,t] <= max_inv[i] * y[i,t]
        # q[i,t] >= (max_inv[i] - Ir[i,t]) - max_inv[i] * (1 - y[i,t])
        # q[i,t] <= (max_inv[i] - Ir[i,t]) + max_inv[i] * (1 - y[i,t])
        # q[i,t] >= 0
        
        # Simpler: 
        # If y[i,t]=1: q[i,t] = max_inv[i] - Ir[i,t], i.e., Ir[i,t] + q[i,t] = max_inv[i]
        # If y[i,t]=0: q[i,t] = 0
        
        # We can write: q[i,t] + Ir[i,t] <= max_inv[i]  (always true)
        # q[i,t] + Ir[i,t] >= max_inv[i] - max_inv[i]*(1-y[i,t])  
        # q[i,t] <= max_inv[i] * y[i,t]
        
        for i in range(n):
            for t in range(1, H + 1):
                model.addConstr(q[i, t] <= ret_max_inv[i] * y[i, t])
                model.addConstr(q[i, t] + Ir[i, t] >= ret_max_inv[i] - ret_max_inv[i] * (1 - y[i, t]))
                model.addConstr(q[i, t] + Ir[i, t] <= ret_max_inv[i] + ret_max_inv[i] * (1 - y[i, t]))
        
        # Actually, since Ir[i,t] <= max_inv[i] always (retailer can't hold more), 
        # and q[i,t] >= 0, we have q[i,t] + Ir[i,t] <= max_inv[i] + something only if we allow
        # But actually we also need Ir[i,t] <= max_inv[i] always
        for i in range(n):
            for t in range(0, H + 2):
                model.addConstr(Ir[i, t] <= ret_max_inv[i])
        
        # Vehicle capacity
        for t in range(1, H + 1):
            model.addConstr(gp.quicksum(q[i, t] for i in range(n)) <= C)
        
        # Routing constraints
        # Node 0 = supplier, nodes 1..n = retailers (in dist matrix order)
        # y[i,t] corresponds to retailer with dist matrix index i+1 (since retailers[i] has id=i+1 typically)
        # Actually, the dist matrix is indexed as: row/col 0 = supplier, row/col 1..n = retailers
        # And retailers are listed in order, so retailer index i (0-based) corresponds to dist matrix index i+1
        
        # Degree constraints: for each visited node, the number of incident edge traversals = 2
        # For supplier: sum_j x[0,j,t] = 2 * (1 if any retailer visited)
        # Actually: sum_j x[0,j,t] = 2 * z[t] where z[t] = 1 if any delivery in period t
        # No, that's not right either. The degree = 2 for each visited node.
        
        # "the total number of route-segment traversals touching each visited location 
        # (supplier and retailers) equals exactly two in that period"
        
        # So for the supplier, if any retailer is visited: degree = 2
        # For retailer i+1 (dist matrix index), if visited: degree = 2
        
        # z[t] = 1 if any retailer is visited in period t
        z = {}
        for t in range(1, H + 1):
            z[t] = model.addVar(vtype=GRB.BINARY, name=f"z_{t}")
            model.addConstr(z[t] <= gp.quicksum(y[i, t] for i in range(n)))
            for i in range(n):
                model.addConstr(z[t] >= y[i, t])
        
        model.update()
        
        # Degree constraints
        for t in range(1, H + 1):
            # Supplier degree
            model.addConstr(
                gp.quicksum(x[0, j, t] for j in range(1, n + 1)) == 2 * z[t]
            )
            
            # Retailer degree
            for i in range(1, n + 1):  # dist matrix index
                ret_idx = i - 1  # 0-based retailer index
                incident = []
                # Edges from supplier to this retailer
                incident.append(x[0, i, t])
                # Edges from other retailers
                for j in range(1, n + 1):
                    if j != i:
                        if i < j:
                            incident.append(x[i, j, t])
                        else:
                            incident.append(x[j, i, t])
                model.addConstr(gp.quicksum(incident) == 2 * y[ret_idx, t])
        
        # Edge can only be used if both endpoints are visited
        for t in range(1, H + 1):
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    model.addConstr(x[i, j, t] <= y[i - 1, t])
                    model.addConstr(x[i, j, t] <= y[j - 1, t])
            for j in range(1, n + 1):
                model.addConstr(x[0, j, t] <= 2 * y[j - 1, t])
        
        # Subtour elimination constraints
        # For any subset S of retailers (dist matrix indices 1..n), 
        # sum of edges within S <= sum of y[i,t] for i in S - y[k,t] for some k in S
        # This is exponential, so we use lazy constraints or MTZ formulation
        
        # Use MTZ formulation for subtour elimination
        # u[i,t] = position of retailer i in the tour in period t
        u = {}
        for i in range(1, n + 1):
            for t in range(1, H + 1):
                u[i, t] = model.addVar(lb=0, ub=n, name=f"u_{i}_{t}")
        
        model.update()
        
        # MTZ constraints: for each edge between retailers i and j (i < j):
        # u[i,t] - u[j,t] + n * x[i,j,t] <= n - 1 (if both visited)
        # u[j,t] - u[i,t] + n * x[i,j,t] <= n - 1 (if both visited)
        # But this doesn't quite work for undirected TSP with MTZ.
        
        # Actually, for undirected TSP with MTZ, we need to orient edges.
        # Let me use a different approach: SEC via callback.
        
        # For small n, we can add all subtour elimination constraints explicitly
        # For larger n, use callback
        
        if n <= 15:
            # Add subtour elimination constraints for all subsets of size 2..n-1
            # This is exponential but manageable for small n
            # Actually even for n=15 this is 2^15 = 32768 subsets per period, times H periods
            # That could be a lot. Let's use callback instead.
            pass
        
        # Use lazy callback for subtour elimination
        model.Params.LazyConstraints = 1
        
        def subtour_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                for t in range(1, H + 1):
                    # Get current solution
                    y_val = {}
                    x_val = {}
                    for i in range(n):
                        y_val[i] = model.cbGetSolution(y[i, t])
                    
                    visited = [i + 1 for i in range(n) if y_val[i] > 0.5]
                    if len(visited) <= 1:
                        continue
                    
                    for i in range(n + 1):
                        for j in range(i + 1, n + 1):
                            x_val[i, j] = model.cbGetSolution(x[i, j, t])
                    
                    # Build adjacency for visited retailers
                    adj = {v: [] for v in [0] + visited}
                    for i in range(n + 1):
                        for j in range(i + 1, n + 1):
                            if x_val[i, j] > 0.5:
                                if i in adj and j in adj:
                                    adj[i].append(j)
                                    adj[j].append(i)
                    
                    # Find connected components among visited retailers (without supplier)
                    # Check if all visited retailers are connected through the supplier
                    visited_set = set(visited)
                    seen = set()
                    
                    # BFS from supplier
                    queue = [0]
                    seen.add(0)
                    while queue:
                        node = queue.pop(0)
                        for neighbor in adj.get(node, []):
                            if neighbor not in seen:
                                seen.add(neighbor)
                                queue.append(neighbor)
                    
                    not_connected = visited_set - seen
                    if not_connected:
                        # Add subtour elimination constraint for the disconnected component
                        S = list(not_connected)
                        # Sum of edges within S <= |S| - 1
                        edges_in_S = []
                        for i_idx in range(len(S)):
                            for j_idx in range(i_idx + 1, len(S)):
                                a, b = min(S[i_idx], S[j_idx]), max(S[i_idx], S[j_idx])
                                edges_in_S.append(x[a, b, t])
                        
                        # Also need to exclude just cycling among these retailers
                        # Standard SEC: sum of edges in S <= |S| - 1
                        model.cbLazy(
                            gp.quicksum(edges_in_S) <= len(S) - 1
                        )
                    else:
                        # All connected, but check for subtours not through depot
                        # Find subtours among retailers only
                        ret_adj = {v: [] for v in visited}
                        for i in range(1, n + 1):
                            for j in range(i + 1, n + 1):
                                if x_val.get((i, j), 0) > 0.5:
                                    if i in ret_adj and j in ret_adj:
                                        ret_adj[i].append(j)
                                        ret_adj[j].append(i)
                        
                        # Check for cycles among retailers
                        ret_seen = set()
                        for start in visited:
                            if start in ret_seen:
                                continue
                            # BFS/DFS to find connected component
                            comp = set()
                            stack = [start]
                            while stack:
                                node = stack.pop()
                                if node in comp:
                                    continue
                                comp.add(node)
                                for nb in ret_adj.get(node, []):
                                    if nb not in comp:
                                        stack.append(nb)
                            ret_seen.update(comp)
                            
                            # Check if this component forms a cycle (all degrees = 2 in retailer subgraph)
                            # and doesn't include supplier connection
                            is_cycle = True
                            for node in comp:
                                deg_ret = len([nb for nb in ret_adj[node] if nb in comp])
                                deg_sup = 1 if x_val.get((0, node) if 0 < node else (node, 0), 0) > 0.5 else 0
                                if 0 < node:
                                    deg_sup = 1 if x_val.get((0, node), 0) > 0.5 else 0
                                if deg_sup > 0:
                                    is_cycle = False
                                    break
                            
                            if is_cycle and len(comp) >= 2:
                                S = list(comp)
                                edges_in_S = []
                                for i_idx in range(len(S)):
                                    for j_idx in range(i_idx + 1, len(S)):
                                        a, b = min(S[i_idx], S[j_idx]), max(S[i_idx], S[j_idx])
                                        edges_in_S.append(x[a, b, t])
                                model.cbLazy(
                                    gp.quicksum(edges_in_S) <= len(S) - 1
                                )
        
        # Objective function
        obj = gp.LinExpr()
        
        # Supplier inventory cost for periods 1..H+1
        for t in range(1, H + 2):
            obj += sup_cost * Is[t]
        
        # Retailer inventory costs for periods 1..H+1
        for i in range(n):
            for t in range(1, H + 2):
                obj += ret_cost[i] * Ir[i, t]
        
        # Transportation costs
        for t in range(1, H + 1):
            for i in range(n + 1):
                for j in range(i + 1, n + 1):
                    obj += dist[i][j] * x[i, j, t]
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        model.optimize(subtour_callback)
        
        if model.SolCount == 0:
            return None
        
        # Extract solution
        result = {}
        result['objective_value'] = model.ObjVal
        
        if logger:
            logger.log(model.ObjVal)
        
        deliveries = {}
        for t in range(1, H + 1):
            period_deliveries = {}
            for i in range(n):
                qval = q[i, t].X
                if qval > 0.5:  # some delivery
                    ret_id = retailers[i]['id']
                    period_deliveries[str(ret_id)] = round(qval)
            if period_deliveries:
                deliveries[str(t)] = period_deliveries
        result['deliveries'] = deliveries
        
        routes = {}
        for t in range(1, H + 1):
            period_routes = []
            for i in range(n + 1):
                for j in range(i + 1, n + 1):
                    xval = round(x[i, j, t].X)
                    if xval > 0:
                        # Map indices to location ids
                        loc_i = supplier['id'] if i == 0 else retailers[i - 1]['id']
                        loc_j = supplier['id'] if j == 0 else retailers[j - 1]['id']
                        period_routes.append([loc_i, loc_j, xval])
            if period_routes:
                routes[str(t)] = period_routes
        result['routes'] = routes
        
        supplier_inventory = {}
        for t in range(0, H + 2):
            supplier_inventory[str(t)] = round(Is[t].X, 2)
        result['supplier_inventory'] = supplier_inventory
        
        retailer_inventory = {}
        for i in range(n):
            ret_id = str(retailers[i]['id'])
            retailer_inventory[ret_id] = {}
            for t in range(0, H + 2):
                retailer_inventory[ret_id][str(t)] = round(Ir[i, t].X, 2)
        result['retailer_inventory'] = retailer_inventory
        
        return result
    
    # First try heuristic to have a feasible solution, then try Gurobi
    
    def greedy_heuristic():
        """Simple greedy heuristic."""
        # Simulate forward, visiting retailers when they would run out of inventory
        sup_inv = [0] * (H + 2)
        ret_inv = [[0] * (H + 2) for _ in range(n)]
        
        sup_inv[0] = sup_start_inv
        for i in range(n):
            ret_inv[i][0] = ret_start_inv[i]
        
        deliveries = {}
        routes = {}
        total_cost = 0
        
        for t in range(1, H + 2):
            # Update inventories from previous period
            if t == 1:
                shipped_prev = 0
                sup_inv[t] = sup_inv[t-1] + sup_prod
            else:
                shipped_prev = sum(deliveries.get(str(t-1), {}).values())
                sup_inv[t] = sup_inv[t-1] + sup_prod - shipped_prev
            
            for i in range(n):
                if t == 1:
                    ret_inv[i][t] = ret_inv[i][t-1] - ret_demand[i]
                else:
                    recv_prev = deliveries.get(str(t-1), {}).get(str(retailers[i]['id']), 0)
                    ret_inv[i][t] = ret_inv[i][t-1] + recv_prev - ret_demand[i]
        
        # This naive forward simulation doesn't make any deliveries, so retailer inventories
        # may go negative. We need to plan deliveries.
        
        # Reset and plan properly
        sup_inv[0] = sup_start_inv
        for i in range(n):
            ret_inv[i][0] = ret_start_inv[i]
        deliveries = {}
        routes = {}
        
        for t in range(1, H + 1):
            # Update supplier inventory
            if t == 1:
                sup_inv[t] = sup_inv[t-1] + sup_prod
            else:
                shipped_prev = sum(deliveries.get(str(t-1), {}).values())
                sup_inv[t] = sup_inv[t-1] + sup_prod - shipped_prev
            
            # Update retailer inventories (from previous period deliveries and consumption)
            for i in range(n):
                if t == 1:
                    ret_inv[i][t] = ret_inv[i][t-1] - ret_demand[i]
                else:
                    recv_prev = deliveries.get(str(t-1), {}).get(str(retailers[i]['id']), 0)
                    ret_inv[i][t] = ret_inv[i][t-1] + recv_prev - ret_demand[i]
            
            # Determine which retailers need delivery
            # A retailer needs delivery if it will run out before the next possible delivery
            urgent = []
            for i in range(n):
                # How many periods can retailer i survive without delivery?
                # Current inventory at period t is ret_inv[i][t]
                # After delivery in period t, inventory becomes ret_max_inv[i] (at period t, effectively)
                # Wait, delivery in period t: q[i,t] is decided, ret_inv[i][t+1] = ret_inv[i][t] + q[i,t] - demand[i]
                
                # If no delivery, ret_inv[i][t+1] = ret_inv[i][t] - demand[i]
                # Retailer runs out when inventory < 0
                periods_left = ret_inv[i][t] // ret_demand[i] if ret_demand[i] > 0 else H + 2
                
                # Need delivery if would go negative by next period
                if ret_inv[i][t] < ret_demand[i]:
                    # URGENT: would go negative next period without delivery
                    urgent.append((0, i))  # priority 0 = most urgent
                elif periods_left <= 1:
                    urgent.append((1, i))
                else:
                    # Check if delivering now saves routing cost (combine with other urgent deliveries)
                    urgent.append((periods_left, i))
            
            urgent.sort()
            
            # Select retailers to visit: must visit all with priority 0, then others if capacity allows
            to_visit = []
            total_shipped = 0
            available = sup_inv[t]
            
            for priority, i in urgent:
                if priority > 2:
                    continue  # Not urgent enough
                
                qty = ret_max_inv[i] - ret_inv[i][t]
                if qty <= 0:
                    continue
                if total_shipped + qty > C:
                    continue
                if total_shipped + qty > available:
                    continue
                
                to_visit.append(i)
                total_shipped += qty
            
            # Make sure truly urgent retailers are included
            for priority, i in urgent:
                if priority > 0:
                    break
                if i not in to_visit:
                    qty = ret_max_inv[i] - ret_inv[i][t]
                    # Must include even if over capacity (shouldn't happen in feasible instance)
                    if qty > 0 and total_shipped + qty <= C and total_shipped + qty <= available:
                        to_visit.append(i)
                        total_shipped += qty
            
            if to_visit:
                period_deliveries = {}
                for i in to_visit:
                    qty = ret_max_inv[i] - ret_inv[i][t]
                    ret_id = retailers[i]['id']
                    period_deliveries[str(ret_id)] = qty
                deliveries[str(t)] = period_deliveries
                
                # Compute route
                visited_dist_indices = [i + 1 for i in to_visit]  # dist matrix indices
                segments, cost = solve_tsp_route(visited_dist_indices, dist)
                
                # Convert segments to use location ids
                route_segments = []
                for seg in segments:
                    a, b, k = seg
                    loc_a = supplier['id'] if a == 0 else retailers[a - 1]['id']
                    loc_b = supplier['id'] if b == 0 else retailers[b - 1]['id']
                    route_segments.append([loc_a, loc_b, k])
                
                routes[str(t)] = route_segments
        
        # Compute period H+1 inventory
        t = H + 1
        shipped_prev = sum(deliveries.get(str(t-1), {}).values())
        sup_inv[t] = sup_inv[t-1] + sup_prod - shipped_prev
        for i in range(n):
            recv_prev = deliveries.get(str(t-1), {}).get(str(retailers[i]['id']), 0)
            ret_inv[i][t] = ret_inv[i][t-1] + recv_prev - ret_demand[i]
        
        # Check feasibility
        feasible = True
        for i in range(n):
            for t in range(1, H + 2):
                if ret_inv[i][t] < -0.001:
                    feasible = False
                    break
        for t in range(1, H + 1):
            shipped_t = sum(deliveries.get(str(t), {}).values())
            if sup_inv[t] < shipped_t - 0.001:
                feasible = False
        
        if not feasible:
            return None
        
        # Compute objective
        total_cost = 0
        for t in range(1, H + 2):
            total_cost += sup_cost * sup_inv[t]
        for i in range(n):
            for t in range(1, H + 2):
                total_cost += ret_cost[i] * ret_inv[i][t]
        for t_str, segs in routes.items():
            for seg in segs:
                a, b, k = seg
                # Find dist matrix indices
                a_idx = 0 if a == supplier['id'] else a  # assuming ids match dist matrix indices
                b_idx = 0 if b == supplier['id'] else b
                # Actually, need to map back to dist matrix indices
                # supplier id -> 0, retailer id -> their position in dist matrix
                # The dist matrix uses row/col 0 for supplier, 1..n for retailers
                # retailer[i].id is the dist matrix index for that retailer
                total_cost += dist[a_idx][b_idx] * k
        
        result = {
            'objective_value': total_cost,
            'deliveries': deliveries,
            'routes': routes,
            'supplier_inventory': {str(t): sup_inv[t] for t in range(0, H + 2)},
            'retailer_inventory': {
                str(retailers[i]['id']): {str(t): ret_inv[i][t] for t in range(0, H + 2)}
                for i in range(n)
            }
        }
        
        return result
    
    def improved_greedy_heuristic():
        """Better greedy that looks ahead to determine optimal delivery timing."""
        # For each possible delivery schedule, evaluate cost
        # This is a more careful greedy that considers:
        # 1. Must deliver before retailer runs out
        # 2. Try to batch deliveries to reduce routing cost
        
        sup_inv = [0] * (H + 2)
        ret_inv = [[0] * (H + 2) for _ in range(n)]
        
        sup_inv[0] = sup_start_inv
        for i in range(n):
            ret_inv[i][0] = ret_start_inv[i]
        
        # First, compute the latest period by which each retailer must receive a delivery
        # to avoid stockout
        def compute_latest_delivery_periods(current_inv_list):
            """For each retailer, compute the set of periods where delivery is mandatory."""
            mandatory = [[] for _ in range(n)]
            for i in range(n):
                inv = current_inv_list[i]
                for t in range(1, H + 2):
                    inv -= ret_demand[i]
                    if inv < 0:
                        # Must have been delivered before or at period t-1
                        # (delivery at t-1 would update inventory at t)
                        # Actually, delivery at period t affects ret_inv[i][t+1]
                        # So if inv at t is negative, we need delivery at t-1 or earlier
                        mandatory[i].append(t)
                        inv = ret_max_inv[i] - ret_demand[i]  # assume delivery fills to max
            return mandatory
        
        # Simulate forward with smarter decisions
        deliveries = {}
        routes = {}
        
        for t in range(1, H + 1):
            # Update supplier inventory
            if t == 1:
                sup_inv[t] = sup_inv[t-1] + sup_prod
            else:
                shipped_prev = sum(deliveries.get(str(t-1), {}).values())
                sup_inv[t] = sup_inv[t-1] + sup_prod - shipped_prev
            
            # Update retailer inventories
            for i in range(n):
                if t == 1:
                    ret_inv[i][t] = ret_inv[i][t-1] - ret_demand[i]
                else:
                    recv_prev = deliveries.get(str(t-1), {}).get(str(retailers[i]['id']), 0)
                    ret_inv[i][t] = ret_inv[i][t-1] + recv_prev - ret_demand[i]
            
            # Determine which retailers to visit
            # Must visit: retailers whose inventory at t+1 would be negative without delivery at t
            # (delivery at t gives: ret_inv[i][t+1] = ret_inv[i][t] + q[i,t] - demand[i])
            
            must_visit = []
            can_visit = []
            
            for i in range(n):
                # Without delivery at t: ret_inv[i][t+1] = ret_inv[i][t] - demand[i]
                future_inv = ret_inv[i][t]
                will_go_negative = False
                periods_until_negative = H + 2
                
                for s in range(t + 1, H + 2):
                    future_inv -= ret_demand[i]
                    if future_inv < 0:
                        periods_until_negative = s - t
                        break
                
                qty = ret_max_inv[i] - ret_inv[i][t]
                if qty <= 0:
                    continue
                
                if periods_until_negative <= 1:
                    must_visit.append(i)
                elif periods_until_negative <= 3:
                    can_visit.append((periods_until_negative, i))
            
            can_visit.sort()
            
            # Start with must_visit
            to_visit = list(must_visit)
            total_shipped = sum(ret_max_inv[i] - ret_inv[i][t] for i in to_visit)
            
            # Add can_visit if they fit
            for _, i in can_visit:
                qty = ret_max_inv[i] - ret_inv[i][t]
                if total_shipped + qty <= C and total_shipped + qty <= sup_inv[t]:
                    to_visit.append(i)
                    total_shipped += qty
            
            if to_visit:
                period_deliveries = {}
                for i in to_visit:
                    qty = ret_max_inv[i] - ret_inv[i][t]
                    if qty > 0:
                        ret_id = retailers[i]['id']
                        period_deliveries[str(ret_id)] = qty
                
                # Verify capacity and supplier constraints
                total = sum(period_deliveries.values())
                if total > C or total > sup_inv[t]:
                    # Remove least urgent retailers until feasible
                    urgency = []
                    for i in to_visit:
                        qty = ret_max_inv[i] - ret_inv[i][t]
                        future_inv = ret_inv[i][t]
                        ptun = H + 2
                        for s in range(t + 1, H + 2):
                            future_inv -= ret_demand[i]
                            if future_inv < 0:
                                ptun = s - t
                                break
                        urgency.append((ptun, qty, i))
                    
                    urgency.sort(reverse=True)  # least urgent first
                    
                    while total > C or total > sup_inv[t]:
                        if not urgency:
                            break
                        ptun, qty, i = urgency.pop(0)
                        if ptun <= 1:
                            continue  # can't remove must-visit
                        to_visit.remove(i)
                        ret_id = retailers[i]['id']
                        del period_deliveries[str(ret_id)]
                        total -= qty
                
                if period_deliveries:
                    deliveries[str(t)] = period_deliveries
                    
                    visited_dist_indices = [i + 1 for i in to_visit if str(retailers[i]['id']) in period_deliveries]
                    segments, cost = solve_tsp_route(visited_dist_indices, dist)
                    
                    route_segments = []
                    for seg in segments:
                        a, b, k = seg
                        loc_a = supplier['id'] if a == 0 else retailers[a - 1]['id']
                        loc_b = supplier['id'] if b == 0 else retailers[b - 1]['id']
                        route_segments.append([loc_a, loc_b, k])
                    
                    routes[str(t)] = route_segments
        
        # Compute period H+1 inventory
        t = H + 1
        shipped_prev = sum(deliveries.get(str(t-1), {}).values())
        sup_inv[t] = sup_inv[t-1] + sup_prod - shipped_prev
        for i in range(n):
            recv_prev = deliveries.get(str(t-1), {}).get(str(retailers[i]['id']), 0)
            ret_inv[i][t] = ret_inv[i][t-1] + recv_prev - ret_demand[i]
        
        # Check feasibility
        feasible = True
        for i in range(n):
            for t in range(1, H + 2):
                if ret_inv[i][t] < -0.001:
                    feasible = False
                    break
            if not feasible:
                break
        
        if not feasible:
            return None
        
        # Compute objective
        total_cost = 0
        for t in range(1, H + 2):
            total_cost += sup_cost * sup_inv[t]
        for i in range(n):
            for t in range(1, H + 2):
                total_cost += ret_cost[i] * ret_inv[i][t]
        
        for t_str, segs in routes.items():
            for seg in segs:
                a, b, k = seg
                a_idx = 0 if a == supplier['id'] else a
                b_idx = 0 if b == supplier['id'] else b
                total_cost += dist[a_idx][b_idx] * k
        
        result = {
            'objective_value': total_cost,
            'deliveries': deliveries,
            'routes': routes,
            'supplier_inventory': {str(t): sup_inv[t] for t in range(0, H + 2)},
            'retailer_inventory': {
                str(retailers[i]['id']): {str(t): ret_inv[i][t] for t in range(0, H + 2)}
                for i in range(n)
            }
        }
        
        return result
    
    # Try Gurobi for exact solution
    best_result = None
    
    # First get a heuristic solution
    heur_result = improved_greedy_heuristic()
    if heur_result is not None:
        best_result = heur_result
        if logger:
            logger.log(heur_result['objective_value'])
    
    remaining_time = time_limit - (time.time() - start_time)
    
    if remaining_time > 10 and n * H <= 500:  # Only try Gurobi for reasonable sizes
        try:
            gurobi_result = try_gurobi_solution()
            if gurobi_result is not None:
                if best_result is None or gurobi_result['objective_value'] < best_result['objective_value']:
                    best_result = gurobi_result
        except Exception as e:
            print(f"Gurobi failed: {e}")
    
    if best_result is None:
        # Emergency: return a minimal solution
        # Try the simple greedy
        best_result = greedy_heuristic()
    
    if best_result is None:
        # Truly emergency: create empty solution (probably infeasible)
        best_result = {
            'objective_value': float('inf'),
            'deliveries': {},
            'routes': {},
            'supplier_inventory': {str(t): 0 for t in range(0, H + 2)},
            'retailer_inventory': {
                str(retailers[i]['id']): {str(t): 0 for t in range(0, H + 2)}
                for i in range(n)
            }
        }
    
    with open(solution_path, 'w') as f:
        json.dump(best_result, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_irp(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()