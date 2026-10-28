import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import itertools
import sys
from collections import defaultdict
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Extract instance data
    n_cust = instance['num_customers']
    depot_start = 0
    depot_end = n_cust + 1
    N = list(range(n_cust + 2))
    customers = list(range(1, n_cust + 1))
    vehicle_capacity = instance['vehicle_capacity']
    tw_width = instance['endogenous_time_window_width']
    scenarios = instance['scenarios']
    prob = {s['scenario_id']: s['probability'] for s in scenarios}
    scenario_ids = [s['scenario_id'] for s in scenarios]
    demands = {}
    for s in scenarios:
        sid = s['scenario_id']
        demands[sid] = {i+1: d for i, d in enumerate(s['demands'])}
    
    dist = instance['distance_matrix']  # cost = time
    exogenous = {}
    exogenous[depot_start] = instance['depot']['exogenous_time_window']
    exogenous[depot_end] = instance['depot']['exogenous_time_window']
    for cust in instance['customers']:
        exogenous[cust['id']] = cust['exogenous_time_window']
    
    # Compute feasible time windows for each customer
    earliest_start = {}
    latest_start = {}
    for i in customers:
        e_start, e_end = exogenous[i]
        earliest_start[i] = e_start
        latest_start[i] = e_end - tw_width
        if latest_start[i] < earliest_start[i]:
            latest_start[i] = earliest_start[i]
    
    # For depot
    depot_earliest = exogenous[depot_start][0]
    depot_latest = exogenous[depot_end][1]
    
    # Precompute possible predecessor/successor for time feasibility (ignoring endogenous for now)
    # But endogenous will be decided
    
    # Since n is small in these instances (assume <=12 customers), we can enumerate feasible routes per scenario
    # But first let's see how large n is
    if n_cust > 10:
        # For larger, we need heuristic. But for now assume we can do exact with discretization or MIP
        pass
    
    # Approach: Use two-stage stochastic MIP with time window discretization or big-M with continuous start times
    # But continuous start times + route variables is hard.
    # Since we need exact and limited time, let's use route enumeration per scenario with compatible time windows.
    
    # First, discretize possible start times for each customer.
    # But to make it tractable, note that optimal start times will be such that they align with some arrival times from previous.
    # A common way is to have start time continuous but use big-M for route based model.
    
    # Let's try to formulate a MIP.
    # But with unlimited vehicles, per scenario it's like VRPTW but with shared time windows across scenarios.
    
    # The endogenous windows are first-stage, routes per scenario second-stage.
    
    # To solve, we can use Benders or just a big MIP if n small.
    # Let's assume n<=8 or so for exact.
    
    # Better approach: since time windows are to be chosen, but service time must be in [s_i, s_i + w], and s_i chosen in [ES, LS].
    # For a fixed set of s_i, each scenario becomes a VRPTW with tight time windows [s_i, s_i+w].
    # Solving VRPTW for each is still hard but if n small we can enumerate all partitions into feasible routes.
    
    # Let's check typical size by code but since unknown, let's implement a route-based column generation like but with Gurobi.
    # But for simplicity and since single core and time limit, let's implement an MIP with arc variables per scenario and continuous start time variables.
    
    # Variables:
    # - For each customer i, start time S_i continuous in [earliest_start[i], latest_start[i]]
    # - For each scenario s, for each arc (i,j), binary x_{ij}^s if arc used in s
    # - For each scenario s, for each node i, service time T_i^s
    # - Flow and capacity are tricky because capacity is per route but with unlimited, we need MTZ or something for capacity? No, capacity is on sum demand per route but to enforce with arc vars we need additional.
    # Enforcing capacity with arc flow without route is difficult without 3-index or MTZ like for capacity.
    # Standard is to use 2-index flow with subtour elimination and capacity cut or use MTZ for load.
    
    # This will be a large MIP with many big-M constraints for time and load.
    # For n=10, 12 customers, ~14 nodes, per scenario ~ 14^2 = 200 arcs * 5 scen = 1000 binaries, plus times.
    # Subtour and capacity make it weak but Gurobi might solve if we add lazy cuts or good formulation.
    
    # Let's try to build it.
    
    m = gp.Model("StochasticTW")
    m.setParam('TimeLimit', args.time_limit - 5)  # leave margin
    m.setParam('MIPGap', 0.01)
    m.setParam('Threads', 1)
    
    # First stage: start times S_i
    S = {}
    for i in customers:
        S[i] = m.addVar(lb=earliest_start[i], ub=latest_start[i], name=f"S_{i}")
    
    # For each scenario
    x = {}
    T = {}
    U = {}  # for load to break subtours and capacity
    for s in scenario_ids:
        for i in N:
            T[(s, i)] = m.addVar(lb=0, ub=depot_latest, name=f"T_{s}_{i}")
            U[(s, i)] = m.addVar(lb=0, ub=vehicle_capacity, name=f"U_{s}_{i}")
        for i in N:
            for j in N:
                if i != j:
                    x[(s, i, j)] = m.addVar(vtype=GRB.BINARY, name=f"x_{s}_{i}_{j}")
    
    # Objective
    obj = gp.quicksum(prob[s] * gp.quicksum(dist[i][j] * x[(s,i,j)] for i in N for j in N if i != j) for s in scenario_ids)
    m.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    
    # 1. Each customer visited exactly once per scenario
    for s in scenario_ids:
        for i in customers:
            m.addConstr(gp.quicksum(x[(s, k, i)] for k in N if k != i) == 1, name=f"visit_in_{s}_{i}")
            m.addConstr(gp.quicksum(x[(s, i, k)] for k in N if k != i) == 1, name=f"visit_out_{s}_{i}")
    
    # Depot flow
    for s in scenario_ids:
        m.addConstr(gp.quicksum(x[(s, depot_start, j)] for j in customers) == gp.quicksum(x[(s, i, depot_end)] for i in customers), name=f"depot_flow_{s}")
        # At most one return? No, multiple vehicles ok, the above is number of vehicles out = in.
    
    # No self loops
    for s in scenario_ids:
        for i in N:
            m.addConstr(x.get((s,i,i),0) == 0)
    
    # Time windows: service in exogenous and in endogenous for customers
    for s in scenario_ids:
        # Depot start
        m.addConstr(T[(s, depot_start)] >= exogenous[depot_start][0])
        m.addConstr(T[(s, depot_start)] <= exogenous[depot_start][1])
        # Customers exogenous + endogenous
        for i in customers:
            m.addConstr(T[(s, i)] >= S[i])
            m.addConstr(T[(s, i)] <= S[i] + tw_width)
            m.addConstr(T[(s, i)] >= exogenous[i][0])
            m.addConstr(T[(s, i)] <= exogenous[i][1])
        # Depot end
        m.addConstr(T[(s, depot_end)] >= exogenous[depot_end][0])
        m.addConstr(T[(s, depot_end)] <= exogenous[depot_end][1])
    
    # Time propagation with big-M
    M = 10000  # large enough, since times are usually < 1000
    for s in scenario_ids:
        for i in N:
            for j in N:
                if i != j:
                    m.addConstr(T[(s,j)] >= T[(s,i)] + dist[i][j] - M * (1 - x[(s,i,j)]), name=f"time_{s}_{i}_{j}")
    
    # Load / Capacity
    for s in scenario_ids:
        # Depot start load 0
        m.addConstr(U[(s, depot_start)] == 0)
        for i in customers:
            m.addConstr(U[(s, i)] >= demands[s][i])
            m.addConstr(U[(s, i)] <= vehicle_capacity)
        for i in N:
            for j in customers:  # to customer
                if i != j:
                    m.addConstr(U[(s,j)] >= U[(s,i)] + demands[s][j] - vehicle_capacity * (1 - x[(s,i,j)]), name=f"load_{s}_{i}_{j}")
        # No load on end depot needed but can set
        m.addConstr(U[(s, depot_end)] == 0)
    
    # Subtour elimination / connectivity: since we have load increasing, and start from 0, it actually prevents subtours on customers as load strictly increases by positive demand.
    # Since all demands >0, the load constraints with U increasing prevent cycles not touching depot.
    # Good.
    
    # We need at least to prevent disconnected, but with visit exactly once and flow conservation at customers implicit by in=out=1, and from depot, it should be paths from start to end.
    # Yes, the model is correct.
    
    # To strengthen, we can add some valid inequalities but let's try.
    
    # Optimize
    if logger:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if objval < model._best_obj - 1e-4:
                    model._best_obj = objval
                    if logger:
                        logger.log(float(objval))
            elif where == GRB.Callback.MIP:
                runtime = time.time() - start_time
                if runtime > args.time_limit:
                    model.terminate()
        
        m._best_obj = float('inf')
        m.optimize(callback)
    else:
        def callback(model, where):
            if where == GRB.Callback.MIP:
                runtime = time.time() - start_time
                if runtime > args.time_limit:
                    model.terminate()
        m.optimize(callback)
    
    # If infeasible or no solution
    if m.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL]:
        # Try to find a feasible solution heuristically - but for now return a simple one
        # Simple heuristic: set all S_i to middle of exogenous
        print("Warning: Solver did not find feasible solution. Constructing heuristic solution.")
        solution = construct_heuristic(instance)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return
    
    # Extract solution
    objective = m.objVal
    time_windows = {}
    for i in customers:
        si = S[i].X
        time_windows[str(i)] = [float(si), float(si + tw_width)]
    
    routes_per_scen = {}
    for s_idx, s in enumerate(scenario_ids):
        # Reconstruct routes from x
        arcs = []
        for i in N:
            for j in N:
                if i != j and x[(s, i, j)].X > 0.5:
                    arcs.append((i, j))
        # Build paths from depot_start
        succ = {i: j for i,j in arcs}
        routes = []
        visited = set()
        current = depot_start
        while len(visited) < n_cust:
            route = [depot_start]
            current = depot_start
            while True:
                next_node = succ.get(current)
                if next_node is None:
                    break
                route.append(next_node)
                if next_node == depot_end:
                    break
                if next_node in customers:
                    visited.add(next_node)
                current = next_node
            if len(route) > 2:  # has customers
                routes.append(route)
            # To handle multiple, remove used
            # Since may have multiple components, better build all paths
        # Better way: since each has one successor
        # Start from each possible after depot
        from_depot = [j for i,j in arcs if i == depot_start]
        for start_c in from_depot:
            route = [depot_start]
            current = start_c
            while current != depot_end:
                route.append(current)
                current = succ[current]
            route.append(depot_end)
            routes.append(route)
        
        # Now for each route compute arrival times (service times)
        scen_routes = []
        for r in routes:
            route_dict = {"route": [int(node) for node in r]}
            arrival = {}
            # Compute cumulative time, but use the T values from solver
            for node in r:
                arrival[str(node)] = float(T[(s, node)].X)
            # cost
            route_cost = 0.0
            for k in range(len(r)-1):
                route_cost += dist[r[k]][r[k+1]]
            scen_routes.append({
                "route": route_dict["route"],
                "arrival_times": arrival,
                "cost": float(route_cost)
            })
        routes_per_scen[str(s)] = scen_routes
    
    solution = {
        "objective_value": float(objective),
        "time_windows": time_windows,
        "routes": routes_per_scen
    }
    
    # If we have logger and improved, but since at end
    if logger and m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        logger.log(float(objective))
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def construct_heuristic(instance):
    # Very basic heuristic: set time windows to earliest possible, solve each scenario independently as CVRP with time windows using greedy.
    # But to make valid, at least set S_i to mid point, and create one big route if possible or separate.
    n_cust = instance['num_customers']
    customers = list(range(1, n_cust+1))
    depot_start = 0
    depot_end = n_cust + 1
    exogenous = {}
    exogenous[0] = instance['depot']['exogenous_time_window']
    exogenous[depot_end] = instance['depot']['exogenous_time_window']
    for c in instance['customers']:
        exogenous[c['id']] = c['exogenous_time_window']
    tw_width = instance['endogenous_time_window_width']
    
    time_windows = {}
    for i in customers:
        e = exogenous[i]
        start = (e[0] + e[1] - tw_width) / 2.0
        time_windows[str(i)] = [float(start), float(start + tw_width)]
    
    # For routes, for each scenario make one route per customer (feasible but expensive)
    scenarios = instance['scenarios']
    routes_dict = {}
    dist = instance['distance_matrix']
    for scen in scenarios:
        sid = str(scen['scenario_id'])
        scen_routes = []
        for cust in customers:
            route = [depot_start, cust, depot_end]
            cost = dist[depot_start][cust] + dist[cust][depot_end]
            arrival_times = {
                str(depot_start): float(exogenous[0][0]),
                str(cust): float(time_windows[str(cust)][0]),
                str(depot_end): float(max(exogenous[depot_end][0], time_windows[str(cust)][0] + dist[cust][depot_end]))
            }
            scen_routes.append({
                "route": route,
                "arrival_times": arrival_times,
                "cost": float(cost)
            })
        routes_dict[sid] = scen_routes
    
    # Compute objective
    total = 0.0
    for s in scenarios:
        prob = s['probability']
        scen_cost = sum(r['cost'] for r in routes_dict[str(s['scenario_id'])])
        total += prob * scen_cost
    
    return {
        "objective_value": float(total),
        "time_windows": time_windows,
        "routes": routes_dict
    }


if __name__ == "__main__":
    main()