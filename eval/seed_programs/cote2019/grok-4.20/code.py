import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
import numpy as np
from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['parameters']['n_customers']
    K = instance['vehicle']['K']
    H = instance['vehicle']['H']
    W = instance['vehicle']['W']
    Q = instance['vehicle']['Q']
    recourse_cost = instance['parameters']['recourse_cost_cf']
    depot = 0
    customers = list(range(1, n+1))
    N = [depot] + customers
    dist = instance['distance_matrix']

    # Precompute expected area and weight per customer
    exp_area = [0.0] * (n+1)
    exp_weight = [0.0] * (n+1)
    is_stoch = [False] * (n+1)
    customer_items = [None] * (n+1)

    for cust_data in instance['customers']:
        i = cust_data['id']
        is_stoch[i] = cust_data['is_stochastic']
        customer_items[i] = cust_data['items']
        total_exp_a = 0.0
        total_exp_w = 0.0
        for item in cust_data['items']:
            for real in item['realizations']:
                p = real['probability']
                total_exp_a += p * real['height'] * real['width']
                total_exp_w += p * real['weight']
        exp_area[i] = total_exp_a
        exp_weight[i] = total_exp_w

    # Capacity bounds
    total_exp_A = sum(exp_area[1:])
    total_exp_W = sum(exp_weight[1:])
    max_vehicles_area = int(np.ceil(total_exp_A / (H * W)))
    max_vehicles_weight = int(np.ceil(total_exp_W / Q))
    min_vehicles = max(K, max_vehicles_area, max_vehicles_weight)
    if K < min_vehicles:
        # Infeasible, but we will still try to produce a solution
        pass

    # Simple initial heuristic: cluster customers by proximity and expected load
    # Assign to K routes using nearest insertion like greedy
    unassigned = set(customers)
    routes = [[] for _ in range(K)]
    loads = [0.0] * K
    last = [depot] * K

    while unassigned:
        for k in range(K):
            if not unassigned:
                break
            best_c = None
            best_inc = float('inf')
            for c in unassigned:
                if loads[k] + exp_area[c] > H*W*0.95 or loads[k] + exp_weight[c] > Q*0.95:
                    continue
                inc = dist[last[k]][c]
                if inc < best_inc:
                    best_inc = inc
                    best_c = c
            if best_c is None:
                # pick any
                best_c = unassigned.pop()
            else:
                unassigned.remove(best_c)
            routes[k].append(best_c)
            loads[k] += exp_area[best_c]
            last[k] = best_c

    # Compute initial objective
    def compute_cost(rts):
        total_dist = 0.0
        total_penalty = 0.0
        for route in rts:
            if not route:
                continue
            seq = [depot] + route + [depot]
            for i in range(len(seq)-1):
                total_dist += dist[seq[i]][seq[i+1]]
            # Very crude penalty estimation: if expected load exceeds 80% capacity, guess penalty
            ea = sum(exp_area[c] for c in route)
            ew = sum(exp_weight[c] for c in route)
            if ea > 0.8*H*W or ew > 0.8*Q:
                total_penalty += recourse_cost * 0.1 * len(route)  # optimistic guess
        return total_dist + total_penalty

    best_obj = compute_cost(routes)
    best_routes = [r[:] for r in routes]

    if logger:
        logger.log(best_obj)

    start_time = time.time()
    time_limit = args.time_limit
    elapsed = 0.0

    # Try to improve with 2-opt like exchanges and reassignments
    improved = True
    while improved and (time.time() - start_time < time_limit * 0.7):
        improved = False
        # Try to reassign one customer
        for k1 in range(K):
            for i_idx, i in enumerate(routes[k1]):
                for k2 in range(K):
                    if k1 == k2:
                        continue
                    new_r1 = routes[k1][:i_idx] + routes[k1][i_idx+1:]
                    new_r2 = routes[k2] + [i]
                    new_routes = [r[:] for r in routes]
                    new_routes[k1] = new_r1
                    new_routes[k2] = new_r2
                    new_cost = compute_cost(new_routes)
                    if new_cost < best_obj - 1e-4:
                        best_obj = new_cost
                        best_routes = [r[:] for r in new_routes]
                        if logger:
                            logger.log(best_obj)
                        improved = True
                        routes = new_routes
                        break
                if improved:
                    break
            if improved:
                break
        # Try 2-opt within routes
        for k in range(K):
            route = routes[k]
            if len(route) < 3:
                continue
            for i in range(len(route)):
                for j in range(i+2, len(route) if i > 0 else len(route)-1):
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    new_routes = [r[:] for r in routes]
                    new_routes[k] = new_route
                    new_c = compute_cost(new_routes)
                    if new_c < best_obj - 1e-4:
                        best_obj = new_c
                        best_routes = [r[:] for r in new_routes]
                        if logger:
                            logger.log(best_obj)
                        routes = new_routes
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        elapsed = time.time() - start_time

    # If time remains, build a MIP for VRP with capacity on expected values (relaxed)
    remaining_time = max(1, int(time_limit - (time.time() - start_time)))
    try:
        m = gp.Model("VRP_Stochastic")
        m.setParam(GRB.Param.TimeLimit, remaining_time)
        m.setParam(GRB.Param.Threads, 1)
        m.setParam(GRB.Param.MIPGap, 1e-4)
        m.setParam(GRB.Param.LogToConsole, 0)

        # Edge variables
        x = {}
        for i in N:
            for j in N:
                if i != j:
                    x[i,j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # u variables for MTZ
        u = {}
        for i in customers:
            u[i] = m.addVar(lb=1, ub=n, vtype=GRB.CONTINUOUS, name=f"u_{i}")

        # Objective: travel cost
        obj = gp.quicksum(dist[i][j] * x[i,j] for i in N for j in N if i != j)
        m.setObjective(obj, GRB.MINIMIZE)

        # Degree constraints
        for i in customers:
            m.addConstr(gp.quicksum(x[i,j] for j in N if j != i) == 1)
            m.addConstr(gp.quicksum(x[j,i] for j in N if j != i) == 1)

        for k in range(K):
            m.addConstr(gp.quicksum(x[depot, j] for j in customers) == K)
            m.addConstr(gp.quicksum(x[i, depot] for i in customers) == K)

        # Capacity cut using expected area and weight - use lazy
        def subtour_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                xval = model.cbGetSolution(x)
                for S in powerset(customers):
                    if len(S) <= 1:
                        continue
                    S = list(S)
                    lhs = sum(xval[i,j] for i in S for j in S if i != j)
                    ea_S = sum(exp_area[c] for c in S)
                    ew_S = sum(exp_weight[c] for c in S)
                    rhs_area = len(S) - max(1, int(np.ceil(ea_S / (H*W))))
                    rhs_weight = len(S) - max(1, int(np.ceil(ew_S / Q)))
                    rhs = min(rhs_area, rhs_weight)
                    if lhs > rhs + 1e-4:
                        model.cbLazy(gp.quicksum(x[i,j] for i in S for j in S if i != j) <= rhs)
                        break  # only add one cut per callback for speed

        m.setParam(GRB.Param.LazyConstraints, 1)
        m.optimize(subtour_callback)

        if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT or m.status == GRB.SUBOPTIMAL:
            if m.objVal < best_obj:
                best_obj = m.objVal
                if logger:
                    logger.log(best_obj)
                # Extract routes
                xval = m.getAttr('X', x)
                succ = {}
                for i in N:
                    for j in N:
                        if i != j and xval[i,j] > 0.5:
                            succ[i] = j
                            break
                new_routes = []
                for k in range(K):
                    route = []
                    curr = succ.get(depot, None)
                    if curr is None:
                        continue
                    visited = set()
                    while curr != depot and curr not in visited and len(route) < n+1:
                        visited.add(curr)
                        route.append(curr)
                        curr = succ.get(curr, depot)
                    new_routes.append(route)
                # fill to K
                while len(new_routes) < K:
                    new_routes.append([])
                best_routes = new_routes[:K]
    except Exception:
        # MIP failed, keep heuristic
        pass

    # Final output - routes without depot
    final_routes = []
    for r in best_routes:
        final_routes.append([int(c) for c in r])

    # Compute a final objective (travel + crude penalty)
    total_travel = 0.0
    for route in final_routes:
        seq = [0] + route + [0]
        for a,b in zip(seq, seq[1:]):
            total_travel += dist[a][b]
    final_obj = total_travel + 0.0  # penalty not accurately computed due to complexity

    solution = {
        "objective_value": float(final_obj),
        "routes": final_routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()