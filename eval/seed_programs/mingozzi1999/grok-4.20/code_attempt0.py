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
    L = list(range(n_line))  # 0 to n_line-1 for linehauls
    B = list(range(n_back))  # 0 to n_back-1 for backhauls
    V = list(range(num_vehicles))

    model = gp.Model("VRPB")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)
    model.setParam('Heuristics', 0.8)
    model.setParam('Cuts', 2)

    # Variables
    x = model.addVars(L, L, V, vtype=GRB.BINARY, name="x")  # line i to line j on v
    y = model.addVars(B, B, V, vtype=GRB.BINARY, name="y")  # back i to back j on v
    u = model.addVars(L, V, vtype=GRB.BINARY, name="u")     # line i to depot on v
    w = model.addVars(B, V, vtype=GRB.BINARY, name="w")     # depot to back i on v
    t = model.addVars(L, B, V, vtype=GRB.BINARY, name="t")  # last line to first back on v
    s = model.addVars(L, V, vtype=GRB.BINARY, name="s")     # depot to first line on v
    z = model.addVars(B, V, vtype=GRB.BINARY, name="z")     # last back to depot on v

    # Dummy for empty backhaul (line-only routes)
    dummy_back = model.addVars(V, vtype=GRB.BINARY, name="dummy_back")

    # Objective
    obj_expr = gp.LinExpr()

    # Linehaul edges
    for i_idx in L:
        i = linehauls[i_idx]
        for j_idx in L:
            if i != linehauls[j_idx]:
                j = linehauls[j_idx]
                for v in V:
                    obj_expr.add(x[i_idx, j_idx, v] * dist[i][j])

    # Line to depot
    for i_idx in L:
        i = linehauls[i_idx]
        for v in V:
            obj_expr.add(u[i_idx, v] * dist[i][depot])

    # Depot to first line
    for i_idx in L:
        i = linehauls[i_idx]
        for v in V:
            obj_expr.add(s[i_idx, v] * dist[depot][i])

    # Transition: last line to first back
    for i_idx in L:
        i = linehauls[i_idx]
        for j_idx in B:
            j = backhauls[j_idx]
            for v in V:
                obj_expr.add(t[i_idx, j_idx, v] * dist[i][j])

    # Backhaul edges
    for i_idx in B:
        i = backhauls[i_idx]
        for j_idx in B:
            if i != backhauls[j_idx]:
                j = backhauls[j_idx]
                for v in V:
                    obj_expr.add(y[i_idx, j_idx, v] * dist[i][j])

    # Depot to first back
    for j_idx in B:
        j = backhauls[j_idx]
        for v in V:
            obj_expr.add(w[j_idx, v] * dist[depot][j])

    # Last back to depot
    for j_idx in B:
        j = backhauls[j_idx]
        for v in V:
            obj_expr.add(z[j_idx, v] * dist[j][depot])

    # Dummy transition: last line to depot for line-only
    for i_idx in L:
        i = linehauls[i_idx]
        for v in V:
            obj_expr.add(dummy_back[v] * u[i_idx, v] * dist[i][depot])  # will be linearized later

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Constraints

    # 1. Each linehaul visited exactly once
    for i_idx in L:
        model.addConstr(gp.quicksum(x[i_idx, j_idx, v] for j_idx in L if linehauls[i_idx] != linehauls[j_idx]) +
                        gp.quicksum(u[i_idx, v] for v in V) == 1)

    # 2. Each backhaul visited exactly once
    for j_idx in B:
        model.addConstr(gp.quicksum(y[i_idx, j_idx, v] for i_idx in B if backhauls[i_idx] != backhauls[j_idx]) +
                        gp.quicksum(w[j_idx, v] for v in V) == 1)

    # 3. Flow conservation - linehaul
    for v in V:
        for k_idx in L:
            inflow = gp.quicksum(x[i_idx, k_idx, v] for i_idx in L if linehauls[i_idx] != linehauls[k_idx]) + s[k_idx, v]
            outflow = gp.quicksum(x[k_idx, j_idx, v] for j_idx in L if linehauls[k_idx] != linehauls[j_idx]) + u[k_idx, v]
            model.addConstr(inflow == outflow)

    # 4. Flow conservation - backhaul
    for v in V:
        for k_idx in B:
            inflow = (gp.quicksum(y[i_idx, k_idx, v] for i_idx in B if backhauls[i_idx] != backhauls[k_idx]) +
                      w[k_idx, v] + gp.quicksum(t[i_idx, k_idx, v] for i_idx in L))
            outflow = (gp.quicksum(y[k_idx, j_idx, v] for j_idx in B if backhauls[k_idx] != backhauls[j_idx]) +
                       z[k_idx, v])
            model.addConstr(inflow == outflow)

    # 5. At most one start per vehicle
    for v in V:
        model.addConstr(gp.quicksum(s[i_idx, v] for i_idx in L) + dummy_back[v] <= 1)

    # 6. Link dummy_back with usage
    for v in V:
        model.addConstr(dummy_back[v] <= gp.quicksum(u[i_idx, v] for i_idx in L))
        model.addConstr(dummy_back[v] <= 1 - gp.quicksum(w[j_idx, v] for j_idx in B))

    # 7. Transition constraints
    for v in V:
        # If a linehaul is last (u), then must go to a backhaul or be dummy
        model.addConstr(gp.quicksum(u[i_idx, v] for i_idx in L) == 
                        gp.quicksum(t[i_idx, j_idx, v] for i_idx in L for j_idx in B) + dummy_back[v])

        # If a backhaul is first from line, must have a last line
        model.addConstr(gp.quicksum(t[i_idx, j_idx, v] for i_idx in L for j_idx in B) ==
                        gp.quicksum(w[j_idx, v] for j_idx in B))

    # 8. Capacity constraints
    for v in V:
        # Linehaul capacity
        model.addConstr(gp.quicksum(demand_l[linehauls[i_idx]] * 
                                    (s[i_idx, v] + gp.quicksum(x[i_idx, j_idx, v] for j_idx in L if linehauls[i_idx] != linehauls[j_idx]) +
                                     gp.quicksum(t[i_idx, k_idx, v] for k_idx in B)) for i_idx in L) <= capacity)
        
        # Backhaul capacity
        model.addConstr(gp.quicksum(demand_b[backhauls[j_idx]] * 
                                    (w[j_idx, v] + gp.quicksum(y[i_idx, j_idx, v] for i_idx in B if backhauls[i_idx] != backhauls[j_idx]) +
                                     gp.quicksum(t[k_idx, j_idx, v] for k_idx in L)) for j_idx in B) <= capacity)

    # 9. Subtour elimination for linehauls (MTZ)
    if n_line > 1:
        pos_l = model.addVars(L, V, lb=0, ub=n_line, vtype=GRB.INTEGER, name="pos_l")
        for v in V:
            for i_idx in L:
                for j_idx in L:
                    if i_idx != j_idx:
                        i_node = linehauls[i_idx]
                        j_node = linehauls[j_idx]
                        model.addConstr(pos_l[i_idx, v] - pos_l[j_idx, v] + (n_line + 1) * x[i_idx, j_idx, v] <= n_line)

    # 10. Subtour elimination for backhauls (MTZ)
    if n_back > 1:
        pos_b = model.addVars(B, V, lb=0, ub=n_back, vtype=GRB.INTEGER, name="pos_b")
        for v in V:
            for i_idx in B:
                for j_idx in B:
                    if i_idx != j_idx:
                        model.addConstr(pos_b[i_idx, v] - pos_b[j_idx, v] + (n_back + 1) * y[i_idx, j_idx, v] <= n_back)

    # Linearize dummy terms if any (in this formulation, the dummy term in obj is linearized by big-M or by substitution)
    # Actually, since we added dummy_back * u in objective, we need to linearize it.
    # Replace the dummy term with auxiliary variables
    aux = model.addVars(L, V, vtype=GRB.BINARY, name="aux")
    for i_idx in L:
        for v in V:
            model.addConstr(aux[i_idx, v] <= u[i_idx, v])
            model.addConstr(aux[i_idx, v] <= dummy_back[v])
            model.addConstr(aux[i_idx, v] >= u[i_idx, v] + dummy_back[v] - 1)
            # Add to objective the transition to depot
            i = linehauls[i_idx]
            obj_expr.add(aux[i_idx, v] * dist[i][depot])  # This is added after model.setObjective, so update

    # Re-set objective with aux
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Callback for logging incumbents
    best_obj = float('inf')
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj - 1e-6:
                best_obj = obj_val
                if logger:
                    logger.log(float(obj_val))

    # Optimize
    model.optimize(callback)

    # Extract solution
    routes = []
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        objective_value = model.objVal if model.objVal < float('inf') else 0.0

        for v in V:
            route = [depot]
            # Find first linehaul
            first_line = None
            for i_idx in L:
                if s[i_idx, v].X > 0.5:
                    first_line = linehauls[i_idx]
                    break
            if first_line is None:
                # Check if dummy (but we disallow pure backhaul)
                has_back = False
                for j_idx in B:
                    if w[j_idx, v].X > 0.5:
                        has_back = True
                        break
                if has_back:
                    # This should not happen due to constraints, but skip
                    continue
                else:
                    continue  # empty vehicle - allowed if not all used, but problem says use exactly num_vehicles?

            if first_line:
                route.append(first_line)
                current = first_line
                current_idx = L[linehauls.index(current)]
                visited = set([current])

                # Follow linehaul chain
                while True:
                    found = False
                    for j_idx in L:
                        j_node = linehauls[j_idx]
                        if j_node not in visited and x[current_idx, j_idx, v].X > 0.5:
                            route.append(j_node)
                            visited.add(j_node)
                            current = j_node
                            current_idx = j_idx
                            found = True
                            break
                    if not found:
                        break

                # Transition to backhaul or depot
                transitioned = False
                for j_idx in B:
                    if t[current_idx, j_idx, v].X > 0.5:
                        next_node = backhauls[j_idx]
                        route.append(next_node)
                        transitioned = True
                        current = next_node
                        current_idx_b = j_idx
                        break

                if not transitioned:
                    # Go back to depot (dummy case)
                    route.append(depot)
                    routes.append(route)
                    continue

                # Follow backhaul chain
                visited_b = set([current])
                while True:
                    found = False
                    for j_idx in B:
                        j_node = backhauls[j_idx]
                        if j_node not in visited_b and y[current_idx_b, j_idx, v].X > 0.5:
                            route.append(j_node)
                            visited_b.add(j_node)
                            current = j_node
                            current_idx_b = j_idx
                            found = True
                            break
                    if not found:
                        break

                route.append(depot)
                routes.append(route)

            else:
                # Pure backhaul - not allowed per problem statement
                continue

        # If fewer routes than vehicles, add empty routes (though problem requires exactly num_vehicles)
        while len(routes) < num_vehicles:
            routes.append([depot, depot])

    else:
        objective_value = 0.0
        routes = [[depot, depot] for _ in range(num_vehicles)]

    # Write solution
    solution = {
        "objective_value": float(objective_value),
        "routes": routes
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # Log final
    if logger and objective_value < float('inf'):
        logger.log(float(objective_value))

if __name__ == "__main__":
    main()