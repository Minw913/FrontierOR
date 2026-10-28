import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Extract data
    n = instance['num_retailers']
    H = instance['time_horizon']
    supplier = instance['supplier']
    S = 0
    retailers_list = list(range(1, n+1))
    V = [S] + retailers_list
    C = instance['vehicle_capacity']
    dist = instance['distance_matrix']
    prod = supplier['production_rate']
    h_s = supplier['inventory_cost']
    start_s = supplier['starting_inventory']

    demands = [0] + [r['demand'] for r in instance['retailers']]
    max_inv_r = [0] + [r['max_inventory'] for r in instance['retailers']]
    start_inv_r = [0] + [r['starting_inventory'] for r in instance['retailers']]
    h_r = [0] + [r['inventory_cost'] for r in instance['retailers']]

    T = list(range(H + 1))
    delivery_T = list(range(H))

    try:
        m = gp.Model("IRP_FillUp")
        m.Params.TimeLimit = args.time_limit - 5 if args.time_limit > 10 else args.time_limit
        m.Params.Threads = 1
        m.Params.LogToConsole = 0
        m.Params.MIPGap = 1e-4
        m.Params.Heuristics = 0.2
        m.Params.MIPFocus = 1

        # Variables
        y = m.addVars(retailers_list, delivery_T, vtype=GRB.BINARY, name="y")
        x = {}
        for t in delivery_T:
            for i_idx in range(len(V)):
                for j_idx in range(i_idx + 1, len(V)):
                    i = V[i_idx]
                    j = V[j_idx]
                    if i == S or j == S:
                        x[i, j, t] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=2, name=f"x_{i}_{j}_{t}")
                    else:
                        x[i, j, t] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{t}")

        I_s = m.addVars(T, lb=0, name="I_s")
        I_r = m.addVars(retailers_list, T, lb=0, name="I_r")
        d = m.addVars(retailers_list, delivery_T, lb=0, ub=C, name="d")

        # Objective
        inv_cost = h_s * gp.quicksum(I_s[t] for t in T)
        for i in retailers_list:
            inv_cost += h_r[i] * gp.quicksum(I_r[i, t] for t in T)
        transp = gp.quicksum(dist[i][j] * x[i, j, t] for i, j, t in x)
        m.setObjective(inv_cost + transp, GRB.MINIMIZE)

        # Supplier inventory
        m.addConstr(I_s[0] == start_s, "start_sup")
        for t in range(H):
            shipped = gp.quicksum(d[i, t] for i in retailers_list)
            m.addConstr(I_s[t] >= shipped, f"sup_cap_{t}")
            m.addConstr(I_s[t + 1] == I_s[t] + prod - shipped, f"sup_flow_{t}")

        # Retailer inventory and max inventory
        for i in retailers_list:
            m.addConstr(I_r[i, 0] == start_inv_r[i], f"start_r_{i}")
            for t in range(H):
                m.addConstr(I_r[i, t + 1] == I_r[i, t] + d[i, t] - demands[i], f"rflow_{i}_{t}")
            for tt in T:
                m.addConstr(I_r[i, tt] <= max_inv_r[i], f"rmax_{i}_{tt}")

        # Delivery policy: fill-up to max when visited
        for i in retailers_list:
            for t in delivery_T:
                m.addConstr(d[i, t] <= max_inv_r[i] * y[i, t], f"d_ub_{i}_{t}")
                m.addConstr(I_r[i, t] + d[i, t] >= max_inv_r[i] - max_inv_r[i] * (1 - y[i, t]), f"fill_low_{i}_{t}")
                m.addConstr(I_r[i, t] + d[i, t] <= max_inv_r[i] + max_inv_r[i] * (1 - y[i, t]), f"fill_up_{i}_{t}")

        # Vehicle capacity
        for t in delivery_T:
            m.addConstr(gp.quicksum(d[i, t] for i in retailers_list) <= C, f"veh_cap_{t}")

        # Routing: degree constraints
        for t in delivery_T:
            for i in retailers_list:
                deg = gp.quicksum(x[min(i, j), max(i, j), t] for j in V if j != i)
                m.addConstr(deg == 2 * y[i, t], f"degree_{i}_{t}")

            visits = gp.quicksum(y[i, t] for i in retailers_list)
            sup_deg = gp.quicksum(x[S, j, t] for j in retailers_list)
            m.addConstr(sup_deg == 2 * visits, f"sup_degree_{t}")

        # MTZ subtour elimination
        u = m.addVars(retailers_list, delivery_T, lb=0, ub=n, vtype=GRB.CONTINUOUS, name="u")
        for t in delivery_T:
            for i in retailers_list:
                for j_idx in range(1, len(V)):
                    j = V[j_idx]
                    if i != j:
                        arc_var = x[min(i, j), max(i, j), t]
                        m.addConstr(u[i, t] - u[j, t] + (n + 1) * arc_var <= n, f"mtz_{i}_{j}_{t}")
                m.addConstr(u[i, t] <= n * y[i, t] + 0.1 * (1 - y[i, t]), f"u_link_{i}_{t}")
                m.addConstr(u[i, t] >= y[i, t], f"u_low_{i}_{t}")

        m.optimize()

        solution = {
            "objective_value": 0.0,
            "deliveries": {},
            "routes": {},
            "supplier_inventory": {},
            "retailer_inventory": {str(i): {} for i in retailers_list}
        }

        if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT) and m.SolCount >= 1:
            obj_val = m.objVal
            deliveries = {}
            routes = {}
            sup_inv = {}
            ret_inv = {str(i): {} for i in retailers_list}

            for t in T:
                sup_inv[str(t)] = round(I_s[t].X)
                for i in retailers_list:
                    ret_inv[str(i)][str(t)] = round(I_r[i, t].X)

            for t in delivery_T:
                deliv_t = {}
                route_list = []
                visit_count = 0
                for i in retailers_list:
                    if y[i, t].X > 0.5:
                        qty = d[i, t].X
                        if qty > 0.1:
                            deliv_t[str(i)] = int(round(qty))
                            visit_count += 1

                if visit_count > 0:
                    deliveries[str(t)] = deliv_t
                    # Extract used arcs
                    for ii in range(len(V)):
                        for jj in range(ii + 1, len(V)):
                            i_node = V[ii]
                            j_node = V[jj]
                            key = (i_node, j_node, t)
                            if key in x:
                                val = x[key].X
                                if val > 0.4:
                                    k = int(round(val))
                                    route_list.append([i_node, j_node, k])

                if visit_count > 0:
                    routes[str(t)] = route_list

            solution = {
                "objective_value": float(obj_val),
                "deliveries": deliveries,
                "routes": routes,
                "supplier_inventory": sup_inv,
                "retailer_inventory": ret_inv
            }

            if logger:
                logger.log(float(obj_val))

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        fallback = {
            "objective_value": 0.0,
            "deliveries": {},
            "routes": {},
            "supplier_inventory": {str(t): 0 for t in range(H+1)},
            "retailer_inventory": {str(i): {str(t): 0 for t in range(H+1)} for i in range(1, n+1)}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(fallback, f, indent=2)

if __name__ == "__main__":
    main()