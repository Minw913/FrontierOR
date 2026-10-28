import argparse
import json
import sys

import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Solution logger
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    # Parse instance
    n = instance['num_customers']
    m = instance['num_periods']
    k = instance['num_vehicles']
    Q = instance['vehicle_capacity']
    depot = instance['depot']
    customers = instance['customers']
    dist = instance['distance_matrix']

    N = list(range(n + 1))
    C = list(range(1, n + 1))
    T = list(range(1, m + 1))

    prod = depot['production_per_period']
    I0_0 = depot['initial_inventory']
    Imax0 = depot['max_inventory']
    h0 = depot['holding_cost']

    I0_i = {i: customers[i - 1]['initial_inventory'] for i in C}
    Imax_i = {i: customers[i - 1]['max_inventory'] for i in C}
    Imin_i = {i: customers[i - 1]['min_inventory'] for i in C}
    d_i = {i: customers[i - 1]['demand_per_period'] for i in C}
    h_i = {i: customers[i - 1]['holding_cost'] for i in C}

    # Create Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.setParam('NumericFocus', 0)
    env.start()
    model = gp.Model(env=env)
    model.setParam('MIPGap', 1e-4)
    model.setParam('TimeLimit', max(1, args.time_limit))
    model.setParam('OutputFlag', 0)

    # Variables
    x = {}
    for i in N:
        for j in N:
            if i != j:
                for t in T:
                    x[i, j, t] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{t}')

    z = {}
    for i in C:
        for t in T:
            z[i, t] = model.addVar(vtype=GRB.BINARY, name=f'z_{i}_{t}')

    q = {}
    for i in C:
        for t in T:
            q[i, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'q_{i}_{t}')

    I = {}
    for i in N:
        for t in T:
            I[i, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'I_{i}_{t}')

    f = {}
    for i in N:
        for j in C:
            if i != j:
                for t in T:
                    f[i, j, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'f_{i}_{j}_{t}')

    w = {}
    for i in C:
        for t in T:
            w[i, t] = model.addVar(lb=1, ub=n, vtype=GRB.CONTINUOUS, name=f'w_{i}_{t}')

    model.update()

    # Objective
    obj = gp.LinExpr()
    for t in T:
        for i in N:
            for j in N:
                if i != j:
                    obj += dist[i][j] * x[i, j, t]
        obj += h0 * I[0, t]
        for i in C:
            obj += h_i[i] * I[i, t]
    model.setObjective(obj, GRB.MINIMIZE)

    # Depot inventory balance
    for t in T:
        if t == 1:
            model.addConstr(I[0, t] == I0_0 + prod - gp.quicksum(q[i, t] for i in C))
        else:
            model.addConstr(I[0, t] == I[0, t - 1] + prod - gp.quicksum(q[i, t] for i in C))
        model.addConstr(I[0, t] <= Imax0)
        model.addConstr(I[0, t] >= 0)

    # Customer inventory balance and capacity
    for i in C:
        for t in T:
            if t == 1:
                model.addConstr(I[i, t] == I0_i[i] + q[i, t] - d_i[i])
                model.addConstr(q[i, t] <= Imax_i[i] - I0_i[i])
            else:
                model.addConstr(I[i, t] == I[i, t - 1] + q[i, t] - d_i[i])
                model.addConstr(q[i, t] <= Imax_i[i] - I[i, t - 1])
            model.addConstr(I[i, t] <= Imax_i[i])
            model.addConstr(I[i, t] >= Imin_i[i])
            M = Imax_i[i]
            model.addConstr(q[i, t] <= M * z[i, t])

    # Routing constraints
    for t in T:
        # Vehicle count
        model.addConstr(gp.quicksum(x[0, j, t] for j in C) <= k)
        # Degree
        for i in C:
            model.addConstr(gp.quicksum(x[j, i, t] for j in N if j != i) == z[i, t])
            model.addConstr(gp.quicksum(x[i, j, t] for j in N if j != i) == z[i, t])
        # Flow conservation
        model.addConstr(gp.quicksum(f[0, j, t] for j in C) == gp.quicksum(q[i, t] for i in C))
        for i in C:
            model.addConstr(
                gp.quicksum(f[j, i, t] for j in N if j != i) - gp.quicksum(f[i, j, t] for j in C if j != i) == q[i, t]
            )
        # Capacity on arcs
        for i in N:
            for j in C:
                if i != j:
                    model.addConstr(f[i, j, t] <= Q * x[i, j, t])
        # Total capacity
        model.addConstr(gp.quicksum(q[i, t] for i in C) <= k * Q)
        # MTZ subtour elimination
        for i in C:
            for j in C:
                if i != j:
                    model.addConstr(w[j, t] >= w[i, t] + 1 - n * (1 - x[i, j, t]))

    # Simple warm start: deliver exactly demand each period if feasible
    total_demand = sum(d_i.values())
    feasible_ws = True
    if total_demand > k * Q:
        feasible_ws = False
    else:
        for t in T:
            depot_end = I0_0 + prod * t - t * total_demand
            if depot_end < 0 or depot_end > Imax0:
                feasible_ws = False
                break
    if feasible_ws:
        for i in C:
            for t in T:
                z[i, t].Start = 1.0
                q[i, t].Start = d_i[i]
                I[i, t].Start = I0_i[i]
        for t in T:
            I[0, t].Start = I0_0 + prod * t - t * total_demand

    model.optimize()

    # Build solution
    solution = {
        "objective_value": float('inf'),
        "solution_details": {
            "periods": {}
        }
    }

    if model.SolCount > 0:
        obj_val = model.ObjVal
        periods = {}
        for t in T:
            routes = []
            remaining = set(i for i in C if z[i, t].X > 0.5)
            while remaining:
                route = [0]
                curr = None
                for i in C:
                    if i in remaining and x[0, i, t].X > 0.5:
                        curr = i
                        break
                if curr is None and remaining:
                    curr = next(iter(remaining))
                while curr != 0:
                    route.append(curr)
                    remaining.discard(curr)
                    nxt = None
                    for j in N:
                        if j != curr and x[curr, j, t].X > 0.5:
                            nxt = j
                            break
                    if nxt is None:
                        nxt = 0
                    curr = nxt
                route.append(0)
                routes.append(route)

            deliveries = {}
            for i in C:
                if q[i, t].X > 1e-6:
                    deliveries[str(i)] = q[i, t].X

            inventories = {"depot": I[0, t].X}
            for i in C:
                inventories[str(i)] = I[i, t].X

            periods[str(t)] = {
                "routes": routes,
                "deliveries": deliveries,
                "inventories": inventories
            }

        solution["objective_value"] = obj_val
        solution["solution_details"]["periods"] = periods

        if logger:
            logger.log_solution(obj_val, solution)
    else:
        if logger:
            logger.log(float('inf'))

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()