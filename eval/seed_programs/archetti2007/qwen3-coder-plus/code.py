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

    # Create Gurobi model
    model = gp.Model("inventory_routing") 
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Heuristics', 0.1)
    model.setParam('Cuts', 2)
    model.setParam('MIPFocus', 1)
    model.setParam('OutputFlag', 1)

    # Helper functions for consistent indexing
    def get_retailer_index_from_id(retailer_id_val):
        """Convert retailer id to index in our arrays"""
        for idx, r_id in enumerate(retailer_ids):
            if r_id == retailer_id_val:
                return idx
        return -1

    # Decision Variables
    # Binary variables indicating if retailer r is visited in period t
    x = model.addVars(num_retailers, H, vtype=GRB.BINARY, name="visit")
    
    # Quantities shipped to each retailer
    q = model.addVars(num_retailers, H, vtype=GRB.CONTINUOUS, lb=0, name="ship_quantity")

    # Supplier & Retailer inventories over time 
    supp_inv = model.addVars(range(H + 1), vtype=GRB.CONTINUOUS, lb=0, name="supplier_inventory")
    retail_inv = model.addVars(num_retailers, range(H + 1), vtype=GRB.CONTINUOUS, lb=0, name="retailer_inventory")

    # Constraint on initial inventories
    model.addConstr(supp_inv[0] == init_supplier_inv, name="init_supp_inv")
    for r in range(num_retailers):
        model.addConstr(retail_inv[r, 0] == init_retailer_invs[r], name=f"init_retail_{r}_inv")

    # Exact fill-up policy: when a retailer is visited, ship exactly to fill up to max inventory
    for r in range(num_retailers):
        for t in range(H):
            # Upper bound: don't ship more than needed to reach max
            model.addConstr(q[r, t] <= max_invs[r] - retail_inv[r, t] + max_invs[r]*(1 - x[r, t]), name=f"quantity_ub_{r}_{t}")
            # Lower bound: if visited, ensure max is reached
            model.addConstr(q[r, t] >= max_invs[r] - retail_inv[r, t] - max_invs[r]*(1 - x[r, t]),
                           name=f"exact_fill_{r}_{t}")  # This enforces exact fill-up when x[r,t]=1

    # Supplier inventory dynamics
    for t in range(H):
        model.addConstr(supp_inv[t+1] == supp_inv[t] + prod_rate - quicksum(q[r, t] for r in range(num_retailers)), 
                       name=f"supp_balance_{t}")

    # Retailer inventory dynamics
    for t in range(H):
        for r in range(num_retailers):
            model.addConstr(retail_inv[r, t+1] == retail_inv[r, t] + q[r, t] - demands[r],
                           name=f"retail_balance_{r}_{t}")

    # Vehicle capacity constraint
    for t in range(H):
        model.addConstr(quicksum(q[r, t] for r in range(num_retailers)) <= vehicle_cap,
                       name=f"vehicle_capacity_{t}")

    # Routing variables
    # y[i, j, t] = number of times edge between i and j is used in period t
    Y = {}
    # Edges between suppliers and retailers and between retailers
    for t in range(H):
        for i in range(num_retailers + 1):
            for j in range(i + 1, num_retailers + 1):  # Undirected graph with i < j
                Y[(i, j, t)] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=2, name=f"Y_{i}_{j}_{t}")

    # Degree constraints for VRP
    # Each visited retailer has degree = 2 (incoming = outgoing = 1)
    for r_idx in range(num_retailers):
        r_node = r_idx + 1  # Retailer node index (starting from 1 since 0=supplier)
        for t in range(H):
            degree_sum = gp.LinExpr()
            # Add edges where r_node appears as either endpoint
            for i in range(r_node):  # Edges where r_node is second vertex
                if (i, r_node, t) in Y:
                    degree_sum += Y[i, r_node, t]
            for j in range(r_node+1, num_retailers + 1):  # Edges where r_node is first vertex
                if (r_node, j, t) in Y:
                    degree_sum += Y[r_node, j, t]

            # Degree of a visited retailer is 2, unvisited retailer is 0
            model.addConstr(degree_sum == 2 * x[r_idx, t], name=f"retailer_deg_{r_idx}t{t}")

    # Depot (supplier 0) - has even positive degree when any retailer is served
    for t in range(H):
        depot_deg = gp.LinExpr()
        for j in range(1, num_retailers + 1):
            if (0, j, t) in Y:
                depot_deg += Y[0, j, t]
        # If any retailers visited, depot must have degree >= 2, <= 2*num_retailers
        num_retailers_visited = quicksum(x[r, t] for r in range(num_retailers))
        model.addConstr(depot_deg >= 2 * (num_retailers_visited > 0) if num_retailers_visited > 0 else 0,
                       name=f"depot_deg_min_{t}")
        model.addConstr(depot_deg <= 2 * num_retailers_visited, name=f"depot_deg_max_{t}")

    # Subtour elimination using Miller-Tucker-Zemlin (MTZ) type formulation
    u = {}  # Auxiliary variable for MTZ subtour elimination
    for t in range(H):
        for i in range(1, num_retailers + 1):  # Retailer nodes (excluding supplier)
            u[(i, t)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
    
    for t in range(H):
        # MTZ constraints: eliminate subtours
        for i in range(1, num_retailers + 1):
            for j in range(1, num_retailers + 1):
                if i != j:
                    # If edge (i,j) is used, enforce ordering u_i - u_j <= n-1-n*y_ij
                    edge_var = Y.get((min(i, j), max(i, j), t), 0)  # Handle symmetric edge
                    if isinstance(edge_var, (int, float)):
                        continue
                    
                    c = model.addConstr(
                        u[i, t] - u[j, t] <= (num_retailers-1) - (num_retailers)*(Y[min(i,j),max(i,j),t]) + 0.1,
                        name=f"MTZ_{i}_{j}_t{t}"
                    )

    # OBJ: Minimize total inventory and transport costs
    total_transport_cost = gp.LinExpr()
    
    for (i, j, t), var in Y.items():
        total_transport_cost += var * dist_mat[i][j]

    total_supp_inv_cost = gp.quicksum(supp_inv[t] * supp_inv_cost for t in range(1, H+1))
    total_ret_inv_cost = gp.quicksum(retail_inv[r, t] * retailer_inv_costs[r] 
                                   for r in range(num_retailers) for t in range(1, H+1))

    total_cost = total_transport_cost + total_supp_inv_cost + total_ret_inv_cost
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Callback setup
    logger_instance = None
    if log_path:
        logger_instance = SolutionLogger(log_path, sense="minimize")

    def callback_log_and_stop(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger_instance:
                logger_instance.log(obj)

    # Solve
    model.optimize(callback_log_and_stop)

    # Process result
    solution = {"objective_value": float('inf'), "deliveries": {}, "routes": {}, 
                "supplier_inventory": {}, "retailer_inventory": {}}

    # Check solution existences
    if model.SolCount == 0:
        print("No feasible solution found.")
        return solution

    obj_val = model.objVal
    solution["objective_value"] = obj_val

    # Deliveries reconstruction
    solution["deliveries"] = {}
    for t in range(H):
        sol_delivs = {}
        for r in range(num_retailers):
            ship_qty = round(q[r, t].x, 2)
            if ship_qty > 0.01:
                sol_delivs[str(r)] = ship_qty
        if sol_delivs:
            solution["deliveries"][str(t)] = sol_delivs

    # Routes reconstruction
    solution["routes"] = {}
    for t in range(H):
        segs = []
        for (i, j, time_period), var in Y.items():
            if time_period == t:
                count = round(var.x)
                if count > 0:
                    segs.append([i, j, count])
        if segs:  # Only add routes that have segments
            solution["routes"][str(t)] = segs

    # Supplier inventory
    solution["supplier_inventory"] = {}
    for t in range(H + 1):
        solution["supplier_inventory"][str(t)] = round(supp_inv[t].x, 2)

    # Retailer inventory
    solution["retailer_inventory"] = {}
    for r in range(num_retailers):
        inv_dict = {}
        for t in range(H+1):
            inv_dict[str(t)] = round(retail_inv[r, t].x, 2)
        solution["retailer_inventory"][str(r)] = inv_dict

    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)

    args = parser.parse_args()

    # Adding a small helper function
    def quicksum(exprs):
        res = gp.LinExpr()
        for e in exprs:
            res += e
        return res

    solution = solve_inventory_routing(args.instance_path, args.time_limit, args.log_path)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()