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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_customers']
    K = instance['num_vehicles']
    Q = instance['vehicle_capacity']
    Gamma = instance['uncertainty_set']['Gamma']
    depot = instance['depot']['id']
    customers = [c['id'] for c in instance['customers']]
    nominal = {c['id']: c['nominal_demand'] for c in instance['customers']}
    dev = {c['id']: c['demand_deviation'] for c in instance['customers']}
    dist = instance['distance_matrix']
    N = list(range(1, n+1))  # customer indices 1..n
    V = [0] + N
    all_subsets = []

    # Precompute relevant deviation subsets (only those with size up to Gamma)
    for r in range(Gamma + 1):
        for comb in combinations(N, r):
            all_subsets.append(comb)

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Build MIP for CVRP with robust capacity constraints
    m = gp.Model("RobustCVRP")
    m.setParam('TimeLimit', args.time_limit)
    m.setParam('Threads', 1)
    m.setParam('MIPGap', 0.01)
    m.setParam('LogToConsole', 0)

    # Variables
    x = m.addVars(V, V, vtype=GRB.BINARY, name="x")
    u = m.addVars(N, vtype=GRB.INTEGER, lb=0, ub=n, name="u")  # MTZ

    # Objective: minimize total cost
    obj = gp.quicksum(dist[i][j] * x[i,j] for i in V for j in V if i != j)
    m.setObjective(obj, GRB.MINIMIZE)

    # Degree constraints
    for i in N:
        m.addConstr(gp.quicksum(x[i,j] for j in V if j != i) == 1, name=f"out_{i}")
        m.addConstr(gp.quicksum(x[j,i] for j in V if j != i) == 1, name=f"in_{i}")

    # Exactly K vehicles
    m.addConstr(gp.quicksum(x[0,j] for j in N) == K, name="out_depot")
    m.addConstr(gp.quicksum(x[i,0] for i in N) == K, name="in_depot")

    # MTZ subtour elimination
    for i in N:
        for j in N:
            if i != j:
                m.addConstr(u[i] - u[j] + (n+1)*x[i,j] <= n, name=f"mtz_{i}_{j}")

    # Robust capacity constraints
    # For each possible deviation subset S with |S| <= Gamma, the total nominal + dev on route <= Q
    # But since routes are not explicitly listed, we use a cut generation like approach in callback
    # Instead we add a relaxed version and use lazy constraints for violation

    # Initial relaxed capacity: total nominal demand per route <= Q - min possible protection
    min_protect = 0
    for i in N:
        m.addConstr(gp.quicksum((nominal[i] + dev[i]) * x[i,j] for j in V if j != i) <= Q, name=f"naive_{i}")

    # Better: add constraints for full deviation on Gamma largest
    # But to be exact, we will use a lazy constraint callback

    best_obj = float('inf')
    best_routes = None
    best_value = None

    def extract_routes(x_sol):
        # Build adjacency
        adj = {i: [] for i in V}
        for i in V:
            for j in V:
                if i != j and abs(x_sol[i,j] - 1) < 1e-5:
                    adj[i].append(j)
        routes = []
        visited = set()
        for start in N:
            if start in visited:
                continue
            route = []
            curr = start
            while True:
                visited.add(curr)
                route.append(curr)
                next_nodes = adj[curr]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                if nxt == 0:
                    break
                curr = nxt
                if curr in visited:
                    break
            if route:
                routes.append(route)
        return routes

    def is_feasible(routes):
        # Check robust feasibility for a set of routes
        for route in routes:
            route_cust = [c for c in route]
            if not route_cust:
                continue
            # Compute nominal total
            nom_tot = sum(nominal[c] for c in route_cust)
            devs = sorted([dev[c] for c in route_cust], reverse=True)
            extra = sum(devs[:Gamma]) if len(devs) > Gamma else sum(devs)
            if nom_tot + extra > Q:
                return False
        return True

    def get_worst_case_demand(route_cust):
        if not route_cust:
            return 0
        nom_tot = sum(nominal[c] for c in route_cust)
        devs = sorted([dev[c] for c in route_cust], reverse=True)
        extra = sum(devs[:Gamma]) if len(devs) >= Gamma else sum(devs)
        return nom_tot + extra

    def callback(model, where):
        nonlocal best_obj, best_routes, best_value
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val >= best_obj:
                return
            x_sol = model.cbGetSolution(x)
            routes = extract_routes(x_sol)
            if len(routes) != K:
                return
            if is_feasible(routes):
                best_obj = obj_val
                best_value = obj_val
                best_routes = routes
                if logger:
                    logger.log(float(obj_val))
                # Could add more cuts but for time we accept
            else:
                # Add a violating cut - find the most violated route and add constraint
                for route in routes:
                    route_cust = [c for c in route]
                    if not route_cust: continue
                    nom_sum = sum(nominal[c] for c in route_cust)
                    # Find the Gamma largest deviation customers in this route
                    route_devs = sorted([(dev[c], c) for c in route_cust], reverse=True)
                    extra = sum(d[0] for d in route_devs[:Gamma])
                    if nom_sum + extra > Q:
                        # Add a no-good like cut or a capacity cut on these variables
                        # Simple: forbid this exact set of customers on one vehicle by adding a cut
                        # sum x[0,j] for j in route >= 2 or use a more MIP friendly cut
                        # For efficiency, add a knapsack cut on nominal + selected devs
                        S = route_cust
                        dev_list = sorted([dev[c] for c in S], reverse=True)
                        max_extra = sum(dev_list[:Gamma])
                        if max_extra + sum(nominal[c] for c in S) > Q:
                            # Add constraint that total "protected" flow cannot be 1 for this S
                            # A simple but weak cut: at least one customer in S cannot be in same route as others
                            # Better: use the fact that if this set is together, it violates
                            # We add a cut that sum_{i in S, j not in S union {0}} x[i,j] >= 1 if |S|>=2
                            if len(S) >= 2:
                                lhs = gp.quicksum(x[i,j] for i in S for j in V if j not in S and j != 0)
                                model.cbLazy(lhs >= 1)
                            break

    m.optimize(callback)

    # If we have a solution from MIP, use it; otherwise try construction heuristic
    if best_routes is None or best_value is None:
        # Heuristic fallback: solve TSP like with savings or random clustering
        # Simple cheapest insertion with sorted customers by angle
        def angle_key(c):
            cx = instance['customers'][c-1]['x'] - instance['depot']['x']
            cy = instance['customers'][c-1]['y'] - instance['depot']['y']
            return np.arctan2(cy, cx)
        sorted_cust = sorted(N, key=angle_key)
        routes = [[] for _ in range(K)]
        demands = [0] * K
        for c in sorted_cust:
            best_k = -1
            best_inc = float('inf')
            for k in range(K):
                new_nom = demands[k] + nominal[c]
                extra = 0
                if len(routes[k]) > 0:
                    dlist = sorted([dev[cust] for cust in routes[k]] + [dev[c]], reverse=True)
                    extra = sum(dlist[:Gamma])
                if new_nom + extra <= Q:
                    inc_cost = 0
                    if not routes[k]:
                        inc_cost = 2 * dist[0][c]
                    else:
                        last = routes[k][-1]
                        inc_cost = dist[last][c] + dist[c][0] - dist[last][0]
                    if inc_cost < best_inc:
                        best_inc = inc_cost
                        best_k = k
            if best_k != -1:
                routes[best_k].append(c)
                demands[best_k] += nominal[c]
            else:
                # If cannot fit, put in its own route if possible (but may violate K)
                pass

        feasible_routes = [r for r in routes if r]
        if len(feasible_routes) <= K and is_feasible(feasible_routes):
            best_routes = feasible_routes
            total_cost = 0
            for r in best_routes:
                if not r: continue
                cost = dist[0][r[0]]
                for i in range(len(r)-1):
                    cost += dist[r[i]][r[i+1]]
                cost += dist[r[-1]][0]
                total_cost += cost
            best_value = total_cost
            if logger:
                logger.log(float(best_value))

    # Final output
    if best_routes is None:
        # Last resort: assign each customer to its own route if K >= n (unlikely)
        best_routes = [[i] for i in N[:K]]
        best_value = sum(2 * dist[0][i] for i in range(1, K+1))

    output_routes = []
    total_cost = 0.0
    for route in best_routes:
        if not route:
            continue
        seq = route[:]
        route_cost = dist[0][seq[0]]
        for i in range(len(seq)-1):
            route_cost += dist[seq[i]][seq[i+1]]
        route_cost += dist[seq[-1]][0]
        total_cost += route_cost
        wcd = get_worst_case_demand(seq)
        output_routes.append({
            "customers": seq,
            "cost": int(route_cost),
            "worst_case_demand": int(wcd)
        })

    result = {
        "objective_value": float(total_cost),
        "routes": output_routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()