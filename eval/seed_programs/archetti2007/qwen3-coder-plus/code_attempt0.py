import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def solve_inventory_routing(instance_path, time_limit, log_path=None):
    # Load instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    num_retailers = instance['num_retailers']
    H = instance['time_horizon']
    
    # Supplier data
    supplier_id = instance['supplier']['id']
    prod_rate = instance['supplier']['production_rate']
    init_supplier_inv = instance['supplier']['starting_inventory']
    supp_inv_cost = instance['supplier']['inventory_cost']
    
    # Retailer data
    demands = []
    max_invs = []
    init_retailer_invs = []
    retailer_inv_costs = []
    retailer_ids = []
    
    for r in instance['retailers']:
        demands.append(r['demand'])
        max_invs.append(r['max_inventory'])
        init_retailer_invs.append(r['starting_inventory'])
        retailer_inv_costs.append(r['inventory_cost'])
        retailer_ids.append(r['id'])
        
    vehicle_cap = instance['vehicle_capacity']
    dist_mat = np.array(instance['distance_matrix'])

    model = gp.Model("inventory_routing")
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)  # Focus more on finding feasible solutions
    
    # Decision Variables
    # x[r,t] = 1 if retailer r is visited in period t
    x = model.addVars(num_retailers, H, vtype=GRB.BINARY, name="visit")
    
    # y[i,j,t] = number of times arc (i,j) is used in period t
    y = {}
    for t in range(H):
        for i in range(num_retailers + 1):  # 0 is supplier
            for j in range(i + 1, num_retailers + 1):  # Since matrix is symmetric
                # We need to distinguish between visits to retailer and to supplier
                y[(i, j, t)] = model.addVar(vtype=GRB.INTEGER, name=f"arc_{i}_{j}_{t}")

    # Quantities shipped: Fixed due to "exact-fill" policy
    ship_quantities = {}  # Retailer fills up to max in stock
    for r in range(num_retailers):
        for t in range(H):
            # How much inventory the retailer has just before delivery
            # This depends on previous inventory and consumption
            initial_inventory_at_r = init_retailer_invs[r]
            for prev_t in range(t):
                if x[r, prev_t].X > 0.5 if hasattr(x[r, prev_t], 'X') else True:  # Use a default guess initially
                    delivered = max_invs[r] - initial_inventory_at_r - (prev_t - 0) * demands[r]
                    initial_inventory_at_r += delivered
                initial_inventory_at_r -= demands[r]
            
            needed = max_invs[r] - max(0, initial_inventory_at_r - t * demands[r])
            ship_quantities[(r, t)] = min(needed, max_invs[r])

    # Ship quantities based on inventory and max level policy
    q = model.addVars(num_retailers, H, vtype=GRB.CONTINUOUS, lb=0, name="ship_quantity")

    # Inventories
    supp_inv = model.addVars(range(H + 1), vtype=GRB.CONTINUOUS, lb=0, name="supplier_inventory")
    retail_inv = model.addVars(num_retailers, range(H + 1), vtype=GRB.CONTINUOUS, lb=0, name="retailer_inventory")
    
    # Route connectivity constraints (MTZ-based subtour elimination)
    u = model.addVars(range(1, num_retailers + 1), range(H), vtype=GRB.CONTINUOUS,
                      lb=1, ub=num_retailers, name="u")

    # Constraints
    # Initial inventory constraints
    model.addConstr(supp_inv[0] == init_supplier_inv)

    for r in range(num_retailers):
        model.addConstr(retail_inv[r, 0] == init_retailer_invs[r])

    # Exact-fill policy - when retailer r is visited in period t, it's filled to max level
    for r in range(num_retailers):
        for t in range(H):
            model.addConstr(q[r, t] == (max_invs[r] - retail_inv[r, t]) * x[r, t])

    # Inventory balance constraints
    for t in range(H):
        # Supplier inventory: prev + prod - shipped
        model.addConstr(supp_inv[t+1] == supp_inv[t] + prod_rate -
                        gp.quicksum(q[r, t] for r in range(num_retailers)))

        for r in range(num_retailers):
            # Retailer inventory: prev + shipped - demand
            model.addConstr(retail_inv[r, t+1] == retail_inv[r, t] + q[r, t] - demands[r])

    # Vehicle capacity constraint
    for t in range(H):
        model.addConstr(gp.quicksum(q[r, t] for r in range(num_retailers)) <= vehicle_cap)

    # Connectivity and routing constraints
    for t in range(H):
        if num_retailers > 0:
            # Each selected retailer connected to the tour
            for r in range(1, num_retailers + 1):
                model.addConstr(
                    gp.quicksum(y[(0, r, t)]) + 
                    gp.quicksum(y[(s, r, t)] + y[(r, s, t)] for s in range(1, num_retailers + 1) if s != r) 
                    >= 2 * x[r-1, t])  # 2 edges incident to each visited retailer

            # Outgoing from supplier constraint (if anything is visited)
            visited_retailers = gp.quicksum(x[r, t] for r in range(num_retailers))
            model.addConstr(
                gp.quicksum(y[(0, r, t)] for r in range(1, num_retailers + 1))
                == visited_retailers + visited_retailers if visited_retailers > 0 else 0)

            # Subtour elimination constraints
            for r1 in range(1, num_retailers + 1):
                for r2 in range(1, num_retailers + 1):
                    if r1 != r2:
                        # Only apply if both retailers are visited
                        model.addConstr(u[r1, t] - u[r2, t] + num_retailers * (
                            y[(r1, r2, t)] + y[(r2, r1, t)]
                        ) <= num_retailers - 1)

    # Objective function parts
    supplier_holding_cost = gp.quicksum(supp_inv[t] * supp_inv_cost for t in range(1, H + 1))
    retailer_holding_cost = gp.quicksum(
        retail_inv[r, t] * retailer_inv_costs[r] 
        for r in range(num_retailers) for t in range(1, H + 1)
    )
    transport_cost = gp.quicksum(
        y[(i, j, t)] * dist_mat[i][j] if (i,j,t) in y else 0
        for i in range(num_retailers+1) for j in range(i+1, num_retailers+1) for t in range(H)
    )

    model.setObjective(supplier_holding_cost + retailer_holding_cost + transport_cost, GRB.MINIMIZE)

    # Optimize
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if log_path:
                logger.log(obj_val)

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    model.optimize(callback)
    
    # Extract solution
    if model.solCount == 0:
        print("No solution found!")
        return {
            "objective_value": float('inf'),
            "deliveries": {},
            "routes": {},
            "supplier_inventory": {str(t): 0 for t in range(H + 1)},
            "retailer_inventory": {str(r): {str(t): 0 for t in range(H + 1)} for r in range(num_retailers)}
        }

    objective_value = model.objVal

    deliveries = {}
    routes = {}
    supplier_inventory = {}
    retailer_inventory = {}

    # Fill delivery decisions (x indicates when a retailer is visited)
    for t in range(H):
        if any(x[r, t].x > 0.9 for r in range(num_retailers)):
            deliveries[str(t)] = {}
            for r in range(num_retailers):
                if x[r, t].x > 0.9:
                    deliveries[str(t)][str(r)] = max_invs[r] - (
                        retail_inv[r, t].x if retail_inv[r, t].x > 0 else 
                        max_invs[r] - demands[r] * t + init_retailer_invs[r])

    # Build routes
    for t in range(H):
        if any(x[r, t].x > 0.9 for r in range(num_retailers)):
            route_segments = []
            for i in range(num_retailers + 1):
                for j in range(i + 1, num_retailers + 1):
                    if y.get((i, j, t)):
                        count = int(round(y[i, j, t].x))
                        if count > 0:
                            route_segments.append([i, j, count])
            routes[str(t)] = route_segments
            
            # If no route segments added, try to reconstruct from x values
            if len(route_segments) == 0:
                visited = [r for r in range(num_retailers) if x[r, t].x > 0.9]
                if visited:
                    # Create basic route visiting all retailers from supplier back to supplier
                    basic_route = [0]  # Start from supplier
                    for vr in visited:
                        basic_route.extend([vr+1, 0])  # Go from supplier to retailer and back
                    # Represent as segments
                    route_segments = [[0, r_idx+1, 1] for r_idx in visited]
                    routes[str(t)] = route_segments

    # Fill inventory levels
    for t in range(H + 1):
        supplier_inventory[str(t)] = supp_inv[t].x

    for r in range(num_retailers):
        retailer_inventory[str(r)] = {}
        for t in range(H + 1):
            retailer_inventory[str(r)][str(t)] = retail_inv[r, t].x

    solution = {
        "objective_value": objective_value,
        "deliveries": deliveries,
        "routes": routes,
        "supplier_inventory": supplier_inventory,
        "retailer_inventory": retailer_inventory
    }

    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)

    args = parser.parse_args()

    solution = solve_inventory_routing(args.instance_path, args.time_limit, args.log_path)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()