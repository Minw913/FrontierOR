import json
import argparse
import time
from typing import List, Dict, Any
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path: str) -> Dict[str, Any]:
    with open(instance_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    num_vehicles = instance['num_vehicles']
    capacity = instance['vehicle_capacity']
    depot = instance['depot']['index']
    linehauls = [cust['index'] for cust in instance['linehaul_customers']]
    backhauls = [cust['index'] for cust in instance['backhaul_customers']]
    n_line = len(linehauls)
    n_back = len(backhauls)
    customers = linehauls + backhauls
    all_nodes = [depot] + customers
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    dist = instance['distance_matrix']

    # Precompute demands
    demand_l = {cust['index']: cust['demand'] for cust in instance['linehaul_customers']}
    demand_b = {cust['index']: cust['demand'] for cust in instance['backhaul_customers']}

    # Map to 0-based for variables
    L = list(range(n_line))
    B = list(range(n_back))
    V = list(range(num_vehicles))

    model = gp.Model("VRPB")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)
    model.setParam('Heuristics', 0.8)
    model.setParam('Cuts', 2)
    model.setParam('LogToConsole', 0)

    # Variables
    x = model.addVars(L, L, V, vtype=GRB.BINARY, name="x")
    y = model.addVars(B, B, V, vtype=GRB.BINARY, name="y")
    u = model.addVars(L, V, vtype=GRB.BINARY, name="u")
    w = model.addVars(B, V, vtype=GRB.BINARY, name="w")
    t = model.addVars(L, B, V, vtype=GRB.BINARY, name="t")
    s = model.addVars(L, V, vtype=GRB.BINARY, name="s")
    z = model.addVars(B, V, vtype=GRB.BINARY, name="z")
    dummy_back = model.addVars(V, vtype=GRB.BINARY, name="dummy")

    # Linearization variables for last_line_to_depot
    aux = model.addVars(L, V, vtype=GRB.BINARY, name="aux")

    # Objective components (built as LinExpr)
    obj_expr = gp.LinExpr()

    # Linehaul internal arcs
    for i_idx in L:
        i = linehauls[i_idx]
        for j_idx in L:
            if i_idx != j_idx:
                j = linehauls[j_idx]
                for v in V:
                    obj_expr.addTerms(dist[i][j], x[i_idx, j_idx, v])

    # Linehaul to depot (pure linehaul routes)
    for i_idx in L:
        i = linehauls[i_idx]
        for v in V:
            obj_expr.addTerms(dist[i][depot], aux[i_idx, v])

    # Depot to first linehaul
    for i_idx in L:
        i = linehauls[i_idx]
        for v in V:
            obj_expr.addTerms(dist[depot][i], s[i_idx, v])

    # Last linehaul to first backhaul
    for i_idx in L:
        i = linehauls[i_idx]
        for j_idx in B:
            j = backhauls[j_idx]
            for v in V:
                obj_expr.addTerms(dist[i][j], t[i_idx, j_idx, v])

    # Backhaul internal arcs
    for i_idx in B:
        i = backhauls[i_idx]
        for j_idx in B:
            if i_idx != j_idx:
                j = backhauls[j_idx]
                for v in V:
                    obj_expr.addTerms(dist[i][j], y[i_idx, j_idx, v])

    # Depot to first backhaul
    for j_idx in B:
        j = backhauls[j_idx]
        for v in V:
            obj_expr.addTerms(dist[depot][j], w[j_idx, v])

    # Last backhaul to depot
    for j_idx in B:
        j = backhauls[j_idx]
        for v in V:
            obj_expr.addTerms(dist[j][depot], z[j_idx, v])

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Constraints

    # Each linehaul visited exactly once
    for i_idx in L:
        lhs = gp.quicksum(x[i_idx, j_idx, v] for j_idx in L if i_idx != j_idx) + gp.quicksum(u[i_idx, v] for v in V)
        model.addConstr(lhs == 1, name=f"line_visit_{i_idx}")

    # Each backhaul visited exactly once
    for j_idx in B:
        lhs = gp.quicksum(y[i_idx, j_idx, v] for i_idx in B if i_idx != j_idx) + gp.quicksum(w[j_idx, v] for v in V)
        model.addConstr(lhs == 1, name=f"back_visit_{j_idx}")

    # Flow conservation - linehauls
    for v in V:
        for k_idx in L:
            inflow = gp.quicksum(x[i_idx, k_idx, v] for i_idx in L if i_idx != k_idx) + s[k_idx, v]
            outflow = gp.quicksum(x[k_idx, j_idx, v] for j_idx in L if k_idx != j_idx) + u[k_idx, v]
            model.addConstr(inflow == outflow, name=f"line_flow_{v}_{k_idx}")

    # Flow conservation - backhauls
    for v in V:
        for k_idx in B:
            inflow = (gp.quicksum(y[i_idx, k_idx, v] for i_idx in B if i_idx != k_idx) +
                      w[k_idx, v] +
                      gp.quicksum(t[i_idx, k_idx, v] for i_idx in L))
            outflow = (gp.quicksum(y[k_idx, j_idx, v] for j_idx in B if k_idx != j_idx) +
                       z[k_idx, v])
            model.addConstr(inflow == outflow, name=f"back_flow_{v}_{k_idx}")

    # Vehicle start constraint
    for v in V:
        model.addConstr(gp.quicksum(s[i_idx, v] for i_idx in L) <= 1, name=f"start_line_{v}")
        model.addConstr(gp.quicksum(w[j_idx, v] for j_idx in B) <= 1 - dummy_back[v], name=f"start_back_{v}")

    # Link dummy_back to linehaul usage
    for v in V:
        model.addConstr(dummy_back[v] <= gp.quicksum(u[i_idx, v] for i_idx in L), name=f"dummy_ub_{v}")

    # Transition linking
    for v in V:
        model.addConstr(gp.quicksum(u[i_idx, v] for i_idx in L) == 
                        gp.quicksum(t[i_idx, j_idx, v] for i_idx in L for j_idx in B) + dummy_back[v],
                        name=f"last_line_trans_{v}")
        model.addConstr(gp.quicksum(t[i_idx, j_idx, v] for i_idx in L for j_idx in B) ==
                        gp.quicksum(w[j_idx, v] for j_idx in B),
                        name=f"trans_back_start_{v}")

    # Linearization of aux = u * dummy_back
    for i_idx in L:
        for v in V:
            model.addConstr(aux[i_idx, v] <= u[i_idx, v], name=f"aux1_{i_idx}_{v}")
            model.addConstr(aux[i_idx, v] <= dummy_back[v], name=f"aux2_{i_idx}_{v}")
            model.addConstr(aux[i_idx, v] >= u[i_idx, v] + dummy_back[v] - 1, name=f"aux3_{i_idx}_{v}")

    # Capacity constraints
    for v in V:
        # Linehaul capacity
        line_load = gp.quicksum(
            demand_l[linehauls[i_idx]] * (
                s[i_idx, v] +
                gp.quicksum(x[i_idx, j_idx, v] for j_idx in L if i_idx != j_idx) +
                gp.quicksum(t[i_idx, k_idx, v] for k_idx in B)
            ) for i_idx in L
        )
        model.addConstr(line_load <= capacity, name=f"cap_line_{v}")

        # Backhaul capacity
        back_load = gp.quicksum(
            demand_b[backhauls[j_idx]] * (
                w[j_idx, v] +
                gp.quicksum(y[i_idx, j_idx, v] for i_idx in B if i_idx != j_idx) +
                gp.quicksum(t[k_idx, j_idx, v] for k_idx in L)
            ) for j_idx in B
        )
        model.addConstr(back_load <= capacity, name=f"cap_back_{v}")

    # Subtour elimination - linehauls (MTZ)
    if n_line > 1:
        pos_l = model.addVars(L, V, lb=0, ub=n_line, vtype=GRB.INTEGER, name="pos_l")
        for v in V:
            for i_idx in L:
                for j_idx in L:
                    if i_idx != j_idx:
                        model.addConstr(
                            pos_l[i_idx, v] - pos_l[j_idx, v] + (n_line + 1) * x[i_idx, j_idx, v] <= n_line,
                            name=f"mtz_l_{v}_{i_idx}_{j_idx}"
                        )
            # Link position to start
            for i_idx in L:
                model.addConstr(pos_l[i_idx, v] >= s[i_idx, v], name=f"pos_start_l_{v}_{i_idx}")

    # Subtour elimination - backhauls (MTZ)
    if n_back > 1:
        pos_b = model.addVars(B, V, lb=0, ub=n_back, vtype=GRB.INTEGER, name="pos_b")
        for v in V:
            for i_idx in B:
                for j_idx in B:
                    if i_idx != j_idx:
                        model.addConstr(
                            pos_b[i_idx, v] - pos_b[j_idx, v] + (n_back + 1) * y[i_idx, j_idx, v] <= n_back,
                            name=f"mtz_b_{v}_{i_idx}_{j_idx}"
                        )
            for j_idx in B:
                model.addConstr(pos_b[j_idx, v] >= w[j_idx, v] + 
                                gp.quicksum(t[i_idx, j_idx, v] for i_idx in L),
                                name=f"pos_start_b_{v}_{j_idx}")

    # Callback
    best_obj = float('inf')
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj - 1e-4:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    model.optimize(callback)

    # Extract solution
    routes = []
    if model.SolCount >= 1:
        objective_value = model.objVal
        for v in V:
            route = [depot]
            # Find starting linehaul
            start_line = None
            for i_idx in L:
                if s[i_idx, v].X > 0.5:
                    start_line = linehauls[i_idx]
                    break

            if start_line is None:
                # Check for pure backhaul (should not occur)
                start_back = None
                for j_idx in B:
                    if w[j_idx, v].X > 0.5:
                        start_back = backhauls[j_idx]
                        break
                if start_back:
                    route.append(start_back)
                    current = start_back
                    curr_idx = B[backhauls.index(current)]
                    visited = {current}
                    while True:
                        next_c = None
                        for nxt_idx in B:
                            nxt_node = backhauls[nxt_idx]
                            if nxt_node not in visited and y[curr_idx, nxt_idx, v].X > 0.5:
                                next_c = nxt_node
                                break
                        if next_c is None:
                            break
                        route.append(next_c)
                        visited.add(next_c)
                        current = next_c
                        curr_idx = B[backhauls.index(current)]
                    route.append(depot)
                    routes.append(route)
                continue

            # Build linehaul segment
            current = start_line
            route.append(current)
            visited_line = {current}
            curr_l_idx = L[linehauls.index(current)]

            while True:
                next_node = None
                for nxt_idx in L:
                    nxt_node_cand = linehauls[nxt_idx]
                    if nxt_node_cand not in visited_line and x[curr_l_idx, nxt_idx, v].X > 0.5:
                        next_node = nxt_node_cand
                        break
                if next_node is None:
                    break
                route.append(next_node)
                visited_line.add(next_node)
                current = next_node
                curr_l_idx = L[linehauls.index(current)]

            # Transition to backhaul or depot
            transitioned = False
            for b_idx in B:
                if t[curr_l_idx, b_idx, v].X > 0.5:
                    next_back = backhauls[b_idx]
                    route.append(next_back)
                    transitioned = True
                    current = next_back
                    curr_b_idx = b_idx
                    break

            if not transitioned:
                # Linehaul-only route
                route.append(depot)
                routes.append(route)
                continue

            # Build backhaul segment
            visited_back = {current}
            while True:
                next_node = None
                for nxt_idx in B:
                    nxt_node_cand = backhauls[nxt_idx]
                    if nxt_node_cand not in visited_back and y[curr_b_idx, nxt_idx, v].X > 0.5:
                        next_node = nxt_node_cand
                        break
                if next_node is None:
                    break
                route.append(next_node)
                visited_back.add(next_node)
                current = next_node
                curr_b_idx = nxt_idx
            route.append(depot)
            routes.append(route)

        # Ensure we have exactly num_vehicles routes (add empty if necessary)
        while len(routes) < num_vehicles:
            routes.append([depot, depot])
        if len(routes) > num_vehicles:
            routes = routes[:num_vehicles]
    else:
        objective_value = 0.0
        routes = [[depot, depot] for _ in range(num_vehicles)]

    solution = {
        "objective_value": float(objective_value),
        "routes": routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger and 'objective_value' in solution:
        logger.log(float(solution["objective_value"]))

if __name__ == "__main__":
    main()