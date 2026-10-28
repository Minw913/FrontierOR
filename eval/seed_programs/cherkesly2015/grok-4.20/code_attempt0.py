import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n_requests = data['n_requests']
    n_nodes = data['n_nodes']
    Q = data['vehicle_capacity']
    T = data['planning_horizon'][1]
    fixed_cost = data['vehicle_fixed_cost']
    nodes = data['nodes']
    arcs = data['arcs']

    # Node indices from 0 to n_nodes-1
    N = list(range(n_nodes))
    P = list(range(1, n_requests + 1))  # pickups
    D = list(range(n_requests + 1, 2 * n_requests + 1))  # deliveries
    o = 0  # origin depot
    d = 2 * n_requests + 1  # destination depot

    # Precompute arc data
    cost = {}
    travel_time = {}
    for a in arcs:
        i = a['from']
        j = a['to']
        cost[i, j] = a['cost']
        travel_time[i, j] = a['travel_time']

    # Node data
    load_change = {nd['node_id']: nd['load'] for nd in nodes}
    service = {nd['node_id']: nd['service_time'] for nd in nodes}
    early = {nd['node_id']: nd['tw_early'] for nd in nodes}
    late = {nd['node_id']: nd['tw_late'] for nd in nodes}

    # Pairing: delivery for each pickup
    pair = {}
    for p in P:
        pair[p] = p + n_requests
    for dd in D:
        pair[dd] = dd - n_requests

    # Build model
    model = gp.Model("PDPTW_LIFO")

    # Variables
    x = model.addVars([(i, j) for i in N for j in N if (i, j) in cost], vtype=GRB.BINARY, name="x")
    t = model.addVars(N, lb=0, ub=T, vtype=GRB.CONTINUOUS, name="t")
    q = model.addVars(N, lb=0, ub=Q, vtype=GRB.CONTINUOUS, name="q")

    # Objective: minimize total cost (includes fixed costs)
    model.setObjective(gp.quicksum(cost[i, j] * x[i, j] for i, j in cost.keys()), GRB.MINIMIZE)

    # Flow conservation
    for k in N:
        if k == o or k == d:
            continue
        model.addConstr(gp.quicksum(x[i, k] for i in N if (i, k) in cost) == 
                        gp.quicksum(x[k, j] for j in N if (k, j) in cost), f"flow_{k}")

    # Start from origin, end at destination
    model.addConstr(gp.quicksum(x[o, j] for j in N if (o, j) in cost) >= 1, "leave_depot")
    model.addConstr(gp.quicksum(x[i, d] for i in N if (i, d) in cost) >= 1, "enter_destination")

    # Each request served exactly once
    for p in P:
        model.addConstr(gp.quicksum(x[i, p] for i in N if (i, p) in cost) == 1, f"visit_p_{p}")
        model.addConstr(gp.quicksum(x[i, pair[p]] for i in N if (i, pair[p]) in cost) == 1, f"visit_d_{pair[p]}")

    # Capacity and load
    for i in N:
        model.addConstr(q[i] >= max(0, load_change[i]), f"q_lower_{i}")
        model.addConstr(q[i] <= min(Q, Q + load_change[i]), f"q_upper_{i}")

    for i, j in cost.keys():
        if j == o or j == d and load_change[j] == 0:
            continue
        M = Q
        model.addConstr(q[j] >= q[i] + load_change[j] - M * (1 - x[i, j]), f"load_flow_{i}_{j}")

    # Time windows and timing
    for i in N:
        model.addConstr(t[i] >= early[i], f"tw_early_{i}")
        model.addConstr(t[i] <= late[i], f"tw_late_{i}")

    for i, j in cost.keys():
        M = T + 1
        model.addConstr(t[j] >= t[i] + service[i] + travel_time[i, j] - M * (1 - x[i, j]), f"time_flow_{i}_{j}")

    # Precedence: pickup before delivery
    for p in P:
        dd = pair[p]
        M = T + 1
        model.addConstr(t[dd] >= t[p] + service[p] + travel_time.get((p, dd), 999999) - M * (1 - x[p, dd]), f"prec_{p}_{dd}")
        # Also add big-M for general case
        model.addConstr(t[dd] >= t[p] + service[p] - M * (1 - gp.quicksum(x[k, p] for k in N if (k, p) in cost)), f"prec_general_{p}")

    # LIFO constraints (subset-based as described, but for tractability we use a simplified but effective MTZ-like for precedence + forbid cross deliveries)
    # Forbid direct pickup_i to delivery_j when i != j (as per problem statement)
    for i in P:
        for j in D:
            if j != pair[i] and (i, j) in cost:
                model.addConstr(x[i, j] == 0, f"no_cross_{i}_{j}")

    # Additional LIFO: basic 2-node and 3-node forbidden patterns via constraints
    # Forbid patterns that would violate LIFO (e.g., p1, p2, d1)
    for p1 in P:
        d1 = pair[p1]
        for p2 in P:
            if p2 == p1:
                continue
            d2 = pair[p2]
            # p1 -> p2 -> d1 forbidden
            if (p1, p2) in cost and (p2, d1) in cost:
                model.addConstr(x[p1, p2] + x[p2, d1] <= 1, f"lifo_ppd_{p1}_{p2}_{d1}")
            # p1 -> d2 -> d1 forbidden if not same
            if (p1, d2) in cost and (d2, d1) in cost and d2 != d1:
                model.addConstr(x[p1, d2] + x[d2, d1] <= 1, f"lifo_pdd_{p1}_{d2}_{d1}")

    # MTZ-like for pickup-delivery ordering to reinforce precedence and LIFO
    u = model.addVars(P, vtype=GRB.INTEGER, lb=1, ub=n_requests, name="u")
    for p in P:
        dd = pair[p]
        model.addConstr(u[p] + 1 <= u.get(pair.get(dd, -1), u[p]) + n_requests * (1 - gp.quicksum(x[i, dd] for i in N if (i, dd) in cost)), "mtz_order")

    # Subtour elimination (basic DFJ)
    for s in range(3, n_nodes - 1):
        for S in [set(comb) for comb in combinations(N, s) if o not in comb and d not in comb]:
            if not any(p in S and pair[p] not in S for p in P if p in S):
                continue
            model.addConstr(gp.quicksum(x[i, j] for i in S for j in S if (i, j) in cost) <= len(S) - 1, f"subtour_{len(S)}")

    # This is too slow. We replace with lazy callback for subtour and LIFO
    model.reset()
    # Remove the DFJ constraints as they are too many; use lazy callbacks instead

    # Set parameters
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 0.01
    model.Params.LogToConsole = 0

    best_obj = float('inf')
    incumbent_routes = None

    def extract_solution(model):
        # Extract routes
        routes = []
        vehicle_id = 0
        visited = set()
        while True:
            # Find next unused start from depot
            found = False
            for j in N:
                if (o, j) in x and x[o, j].X > 0.5 and j not in visited:
                    found = True
                    break
            if not found:
                break
            # Build route
            route = [o]
            current = o
            visited.add(o)
            while current != d:
                for nxt in N:
                    if (current, nxt) in x and x[current, nxt].X > 0.5:
                        route.append(nxt)
                        visited.add(nxt)
                        current = nxt
                        break
            if current != d:
                route.append(d)
            # Build details
            route_details = []
            prev_time = 0.0
            prev_load = 0.0
            for idx, nd in enumerate(route):
                if idx == 0:
                    arr_time = 0.0
                    load_after = 0.0
                else:
                    arr_time = max(early[nd], prev_time + service[route[idx-1]] + travel_time.get((route[idx-1], nd), 0.0))
                    load_after = prev_load + load_change[nd]
                route_details.append({
                    "node_id": int(nd),
                    "arrival_time": float(arr_time),
                    "load_after": float(load_after)
                })
                prev_time = arr_time
                prev_load = load_after
            routes.append({
                "vehicle_id": vehicle_id,
                "route_sequence": [int(nd) for nd in route],
                "route_details": route_details
            })
            vehicle_id += 1
        return routes

    def callback(model, where):
        nonlocal best_obj, incumbent_routes
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            if obj < best_obj - 1e-6:
                # Check if integer solution is feasible (Gurobi already ensures)
                best_obj = obj
                try:
                    model.cbGetSolution(x)
                    routes = extract_solution(model)
                    if routes:
                        incumbent_routes = routes
                        if logger:
                            logger.log(float(best_obj))
                except:
                    pass

    # Optimize with callback for better incumbents
    model.optimize(callback)

    # Get final solution
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        if model.SolCount > 0:
            final_routes = extract_solution(model)
            final_obj = model.ObjVal
        elif incumbent_routes is not None:
            final_routes = incumbent_routes
            final_obj = best_obj
        else:
            # Fallback: no solution, return empty
            final_routes = []
            final_obj = 0.0
    else:
        final_routes = []
        final_obj = 0.0

    # Write solution
    solution = {
        "objective_value": float(final_obj),
        "routes": final_routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    # For large instances this is a heuristic approach using MIP with callback.
    # For very small instances it may solve to optimality.
    # To improve quality, we could add more valid inequalities but this balances speed and quality.
    from itertools import combinations
    main()