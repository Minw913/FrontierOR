import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="Capacitated Single Allocation Hub Location optimal solver")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to output the solution JSON")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum solve time in seconds")
    parser.add_argument('--log_path', type=str, default=None, help="Optional path for logging incumbents")
    args = parser.parse_args()

    # Setup logger if provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data['n']
    chi = data['cost_parameters']['collection_cost_chi']
    alpha = data['cost_parameters']['transfer_cost_alpha']
    delta = data['cost_parameters']['distribution_cost_delta']
    D = data['total_flow_D']
    W = data['flow_matrix']
    dist = data['distance_matrix']
    O = data['outgoing_flow']
    S = data['setup_costs']
    C = data['capacities']

    # Compute total incoming flow to each node
    I = [sum(W[j][i] for j in range(N)) for i in range(N)]

    # ---------------------------------------------------------
    # Initial Start Solution Heuristic (Greedy capacity-aware)
    # ---------------------------------------------------------
    open_hubs = set()
    hub_used_cap = [0.0] * N
    assigned = [None] * N
    z_start = {}

    # Valid candidates for hubs: a node can open itself only if it can handle its own flow
    valid_potential_hubs = [h for h in range(N) if O[h] <= C[h] + 1e-6]
    sorted_nodes = sorted(valid_potential_hubs, key=lambda x: C[x], reverse=True)
    nodes_to_assign = sorted(range(N), key=lambda x: O[x], reverse=True)

    feasible_init = True
    for i in nodes_to_assign:
        if assigned[i] is not None:
            continue
        best_h = None
        best_dist = float('inf')
        for h in open_hubs:
            if hub_used_cap[h] + O[i] <= C[h] + 1e-6:
                if dist[i][h] < best_dist:
                    best_dist = dist[i][h]
                    best_h = h
        
        if best_h is not None:
            assigned[i] = best_h
            hub_used_cap[best_h] += O[i]
        else:
            found_h = None
            for h in sorted_nodes:
                if h not in open_hubs:
                    flow_needed = O[h] + (O[i] if i != h else 0)
                    if flow_needed <= C[h] + 1e-6:
                        found_h = h
                        break
            if found_h is not None:
                old_h = assigned[found_h]
                if old_h is not None and old_h != found_h:
                    hub_used_cap[old_h] -= O[found_h]
                open_hubs.add(found_h)
                assigned[found_h] = found_h
                if i != found_h:
                    assigned[i] = found_h
                    hub_used_cap[found_h] += O[i] + O[found_h]
                else:
                    hub_used_cap[found_h] += O[found_h]
            else:
                feasible_init = False
                break

    # Meet the total minimum capacity requirement
    if feasible_init:
        total_opened_cap = sum(C[h] for h in open_hubs)
        for h in sorted_nodes:
            if total_opened_cap >= D - 1e-6:
                break
            if h not in open_hubs:
                open_hubs.add(h)
                total_opened_cap += C[h]
                old_h = assigned[h]
                if old_h is not None and old_h != h:
                    hub_used_cap[old_h] -= O[h]
                assigned[h] = h
                if old_h is None or old_h != h:
                    hub_used_cap[h] += O[h]

    if feasible_init and sum(C[h] for h in open_hubs) >= D - 1e-6:
        for i in range(N):
            z_start[i, assigned[i]] = 1
    else:
        feasible_init = False

    # ---------------------------------------------------------
    # Exact MIP Formulation 
    # ---------------------------------------------------------
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()
    model = gp.Model("CHLP", env=env)
    
    # Restrict to single CPU core
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.MIPFocus = 1 # Focus on finding feasible solutions efficiently
    
    if args.time_limit >= 60:
        model.Params.NoRelHeurTime = 15
    elif args.time_limit >= 30:
        model.Params.NoRelHeurTime = min(args.time_limit * 0.2, 10)
    elif args.time_limit > 5:
        model.Params.NoRelHeurTime = 1

    # Z[i, k] = 1 if node i is assigned to hub k
    Z = model.addVars(N, N, vtype=GRB.BINARY, name="Z")
    
    # F[i, k, m]: flow of commodity from orig i transitioning from hub k to hub m (k != m)
    F = model.addVars([(i, k, m) for i in range(N) for k in range(N) for m in range(N) if k != m], 
                      vtype=GRB.CONTINUOUS, name="F")

    # Apply heuristic starts
    if feasible_init:
        for i in range(N):
            for k in range(N):
                Z[i, k].Start = 1 if (i, k) in z_start else 0

    # Assignment Constraints
    model.addConstrs((Z[i, k] <= Z[k, k] for i in range(N) for k in range(N) if i != k), name="assign_open")
    model.addConstrs((Z.sum(i, '*') == 1 for i in range(N)), name="single_assign")

    # Capacity Constraints
    model.addConstrs((gp.quicksum(O[i] * Z[i, k] for i in range(N)) <= C[k] * Z[k, k] for k in range(N)), name="capacity")
    model.addConstr(gp.quicksum(C[k] * Z[k, k] for k in range(N)) >= D, name="total_capacity")

    # Flow Balance Constraints
    for i in range(N):
        for k in range(N):
            model.addConstr(
                F.sum(i, k, '*') - F.sum(i, '*', k) == 
                O[i] * Z[i, k] - gp.quicksum(W[i][j] * Z[j, k] for j in range(N) if W[i][j] > 0),
                name=f"bal_{i}_{k}"
            )

    # Valid Inequalities
    model.addConstrs((F.sum(i, k, '*') <= O[i] * Z[i, k] for i in range(N) for k in range(N)), name="vi1")
    model.addConstrs((F.sum(i, '*', k) <= gp.quicksum(W[i][j] * Z[j, k] for j in range(N) if W[i][j] > 0) for i in range(N) for k in range(N)), name="vi2")

    # Objective Function
    obj = gp.quicksum(S[k] * Z[k, k] for k in range(N))
    obj += gp.quicksum((chi * O[i] + delta * I[i]) * dist[i][k] * Z[i, k] for i in range(N) for k in range(N))
    obj += gp.quicksum(alpha * dist[k][m] * F[i, k, m] for i, k, m in F.keys())

    model.setObjective(obj, GRB.MINIMIZE)

    # Callback definition for logger
    def make_callback(logger_ref):
        best_obj = [float('inf')]
        def callback(mdl, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    current_obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if current_obj < best_obj[0] - 1e-5:
                        best_obj[0] = current_obj
                        if logger_ref:
                            logger_ref.log(current_obj)
                except Exception:
                    pass
        return callback

    # Optimize Model
    model.optimize(make_callback(logger))

    # Helper function to rigidly compute exact objective match
    def compute_objective(hubs_list, assignments):
        val = sum(S[k] for k in hubs_list)
        for i in range(N):
            val += (chi * O[i] + delta * I[i]) * dist[i][assignments[i]]
            k = assignments[i]
            for j in range(N):
                if W[i][j] > 0:
                    m = assignments[j]
                    if k != m:
                        val += alpha * dist[k][m] * W[i][j]
        return val

    # Solution Extraction Logic
    if model.SolCount > 0:
        assignment_res = [0] * N
        for i in range(N):
            best_k = 0
            best_val = -1.0
            for k in range(N):
                if Z[i, k].X > best_val:
                    best_val = Z[i, k].X
                    best_k = k
            assignment_res[i] = best_k
    elif feasible_init:
        assignment_res = assigned
    else:
        # Failsafe fallback 
        assignment_res = []

    if len(assignment_res) > 0:
        opened_hubs = sorted(list(set(assignment_res)))
        obj_val = compute_objective(opened_hubs, assignment_res)
    else:
        opened_hubs = []
        obj_val = 0.0

    # Build and write final JSON complying strictly with schema
    out_data = {
        "objective_value": obj_val,
        "hubs": opened_hubs,
        "assignment": assignment_res
    }

    with open(args.solution_path, 'w') as f:
        json.dump(out_data, f, indent=4)

if __name__ == '__main__':
    main()