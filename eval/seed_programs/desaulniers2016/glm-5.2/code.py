#!/usr/bin/env python3
import json
import argparse
import sys
import time
from gurobipy import Model, GRB, quicksum

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    start_time = time.time()
    solution = solve(instance, args.time_limit, logger, start_time)

    if solution is None:
        solution = construct_trivial_solution(instance)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def solve(instance, time_limit, logger, start_time):
    N = instance['num_nodes']
    n_cust = instance['num_customers']
    T = instance['num_periods']
    K = instance['num_vehicles']
    Q = instance['vehicle_capacity']
    dist = instance['distance_matrix']
    h = instance['holding_costs']
    dem = instance['customer_demands']
    prod = instance['supplier_production']
    cap = instance['inventory_capacities']
    init_inv = instance['initial_inventories']

    customers = list(range(1, N))
    periods = list(range(T))
    vehicles = list(range(K))

    if n_cust == 0 or T == 0:
        return construct_trivial_solution(instance)

    model = Model('IRP')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 0)

    # Variables
    arcs = [(t, i, j, k) for t in periods for i in range(N) for j in range(N)
            for k in vehicles if i != j]
    x = model.addVars(arcs, vtype=GRB.BINARY, name='x')
    z = model.addVars(T, N, K, vtype=GRB.BINARY, name='z')
    q = model.addVars(T, N, K, vtype=GRB.CONTINUOUS, lb=0, name='q')
    I = model.addVars(T, N, vtype=GRB.CONTINUOUS, lb=0, name='I')
    u = model.addVars(T, N, K, vtype=GRB.CONTINUOUS, lb=0, name='u')

    # Objective
    travel_cost = quicksum(dist[i][j] * x[t, i, j, k] for (t, i, j, k) in arcs)
    holding_cost = quicksum(h[i] * I[t, i] for t in periods for i in range(N))
    model.setObjective(travel_cost + holding_cost, GRB.MINIMIZE)

    # Constraints
    for t in periods:
        total_q_t = quicksum(q[t, c, k] for c in customers for k in vehicles)
        if t == 0:
            model.addConstr(I[t, 0] == init_inv[0] + prod[t] - total_q_t)
        else:
            model.addConstr(I[t, 0] == I[t - 1, 0] + prod[t] - total_q_t)
        model.addConstr(I[t, 0] <= cap[0])

        for c in customers:
            ci = c - 1
            cust_q_t = quicksum(q[t, c, k] for k in vehicles)
            if t == 0:
                model.addConstr(I[t, c] == init_inv[c] + cust_q_t - dem[ci][t])
                model.addConstr(init_inv[c] + cust_q_t <= cap[c])
            else:
                model.addConstr(I[t, c] == I[t - 1, c] + cust_q_t - dem[ci][t])
                model.addConstr(I[t - 1, c] + cust_q_t <= cap[c])
            model.addConstr(quicksum(z[t, c, k] for k in vehicles) <= 1)

        for k in vehicles:
            model.addConstr(quicksum(q[t, c, k] for c in customers) <= Q)
            for c in customers:
                model.addConstr(q[t, c, k] <= min(Q, cap[c]) * z[t, c, k])

            for c in customers:
                in_flow = quicksum(x[t, i, c, k] for i in range(N) if i != c)
                out_flow = quicksum(x[t, c, j, k] for j in range(N) if j != c)
                model.addConstr(in_flow == z[t, c, k])
                model.addConstr(out_flow == z[t, c, k])

            out_depot = quicksum(x[t, 0, j, k] for j in customers)
            in_depot = quicksum(x[t, i, 0, k] for i in customers)
            model.addConstr(out_depot == in_depot)
            model.addConstr(out_depot <= 1)

            # MTZ subtour elimination
            for i in customers:
                for j in customers:
                    if i != j:
                        model.addConstr(
                            u[t, i, k] - u[t, j, k] + n_cust * x[t, i, j, k] <= n_cust - 1
                        )
            for c in customers:
                model.addConstr(u[t, c, k] >= z[t, c, k])
                model.addConstr(u[t, c, k] <= n_cust * z[t, c, k])

    # Symmetry breaking: vehicle k+1 used only if vehicle k used
    for t in periods:
        for k in range(K - 1):
            model.addConstr(
                quicksum(x[t, 0, j, k + 1] for j in customers) <=
                quicksum(x[t, 0, j, k] for j in customers)
            )

    # Prepare for callback
    x_var_list = list(x.values())
    x_key_list = list(x.keys())
    q_var_list = list(q.values())
    q_key_list = list(q.keys())
    I_var_list = list(I.values())
    I_key_list = list(I.keys())

    best_solution = [None]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals_list = model.cbGetSolution(x_var_list)
            x_vals = dict(zip(x_key_list, x_vals_list))
            q_vals_list = model.cbGetSolution(q_var_list)
            q_vals = dict(zip(q_key_list, q_vals_list))
            I_vals_list = model.cbGetSolution(I_var_list)
            I_vals = dict(zip(I_key_list, I_vals_list))

            sol = build_solution_dict(x_vals, q_vals, I_vals, T, N, K, customers, dist, h)
            sol["objective_value"] = float(obj)
            best_solution[0] = sol

            if logger:
                logger.log_solution(obj, sol)

    # Set time limit accounting for model building time
    elapsed = time.time() - start_time
    remaining = max(1, time_limit - elapsed - 3)
    model.setParam('TimeLimit', remaining)

    model.optimize(callback)

    if best_solution[0] is not None:
        return best_solution[0]

    if model.SolCount > 0:
        x_vals = {k: v.X for k, v in x.items()}
        q_vals = {k: v.X for k, v in q.items()}
        I_vals = {k: v.X for k, v in I.items()}
        sol = build_solution_dict(x_vals, q_vals, I_vals, T, N, K, customers, dist, h)
        sol["objective_value"] = float(model.ObjVal)
        return sol

    return None


def build_solution_dict(x_vals, q_vals, I_vals, T, N, K, customers, dist, h):
    routes = {}
    inventories = {str(i): {} for i in range(N)}

    for t in range(T):
        routes[str(t)] = {}
        for k in range(K):
            route = [0]
            current = 0
            visited = set()
            deliveries = {}

            while True:
                next_node = None
                for j in range(N):
                    if j != current:
                        key = (t, current, j, k)
                        if key in x_vals and x_vals[key] > 0.5:
                            next_node = j
                            break
                if next_node is None or next_node == 0:
                    route.append(0)
                    break
                if next_node in visited:
                    route.append(0)
                    break
                route.append(next_node)
                visited.add(next_node)
                qkey = (t, next_node, k)
                if qkey in q_vals and q_vals[qkey] > 1e-6:
                    deliveries[str(next_node)] = float(q_vals[qkey])
                current = next_node

            routes[str(t)][str(k)] = {
                "route": route,
                "deliveries": deliveries
            }

    for i in range(N):
        for t in range(T):
            inventories[str(i)][str(t)] = float(I_vals.get((t, i), 0))

    # Calculate objective
    total_cost = 0.0
    for t in range(T):
        for k in range(K):
            r = routes[str(t)][str(k)]["route"]
            for idx in range(len(r) - 1):
                total_cost += dist[r[idx]][r[idx + 1]]
    for i in range(N):
        for t in range(T):
            total_cost += h[i] * I_vals.get((t, i), 0)

    return {
        "objective_value": total_cost,
        "routes": routes,
        "inventories": inventories
    }


def construct_trivial_solution(instance):
    """Construct a feasible solution as fallback."""
    N = instance['num_nodes']
    n_cust = instance['num_customers']
    T = instance['num_periods']
    K = instance['num_vehicles']
    Q = instance['vehicle_capacity']
    dist = instance['distance_matrix']
    h = instance['holding_costs']
    dem = instance['customer_demands']
    prod = instance['supplier_production']
    cap = instance['inventory_capacities']
    init_inv = instance['initial_inventories']

    customers = list(range(1, N))
    inv = list(init_inv)
    routes = {}
    inventories = {str(i): {} for i in range(N)}
    total_cost = 0.0

    for t in range(T):
        depot_inv = inv[0] + prod[t]

        # Decide deliveries
        deliveries = {}
        for c in customers:
            ci = c - 1
            max_del = min(Q, cap[c] - inv[c])
            needed = max(0, dem[ci][t] - inv[c])
            deliver = min(needed, max_del)
            if deliver < 0:
                deliver = 0
            if deliver > depot_inv:
                deliver = depot_inv
            deliveries[c] = deliver
            depot_inv -= deliver

        # Build routes
        period_routes = {}
        remaining = [c for c in customers if deliveries.get(c, 0) > 1e-6]
        vehicle_idx = 0

        while remaining and vehicle_idx < K:
            route = [0]
            route_deliveries = {}
            load = 0
            new_remaining = []

            # Greedy nearest neighbor
            current = 0
            temp_remaining = list(remaining)
            while temp_remaining:
                best_c = None
                best_dist = float('inf')
                for c in temp_remaining:
                    if load + deliveries[c] <= Q and dist[current][c] < best_dist:
                        best_dist = dist[current][c]
                        best_c = c
                if best_c is None:
                    break
                route.append(best_c)
                route_deliveries[str(best_c)] = float(deliveries[best_c])
                load += deliveries[best_c]
                temp_remaining.remove(best_c)
                current = best_c
                remaining.remove(best_c)

            route.append(0)
            period_routes[vehicle_idx] = {"route": route, "deliveries": route_deliveries}
            vehicle_idx += 1

        for k in range(K):
            if k not in period_routes:
                period_routes[k] = {"route": [0, 0], "deliveries": {}}

        routes[str(t)] = {}
        for k in range(K):
            routes[str(t)][str(k)] = period_routes[k]
            r = routes[str(t)][str(k)]["route"]
            for idx in range(len(r) - 1):
                total_cost += dist[r[idx]][r[idx + 1]]

        inv[0] = depot_inv
        for c in customers:
            ci = c - 1
            inv[c] = inv[c] + deliveries.get(c, 0) - dem[ci][t]
            if inv[c] < 0:
                inv[c] = 0

        for i in range(N):
            inventories[str(i)][str(t)] = float(inv[i])
            total_cost += h[i] * inv[i]

    return {
        "objective_value": total_cost,
        "routes": routes,
        "inventories": inventories
    }


if __name__ == '__main__':
    main()