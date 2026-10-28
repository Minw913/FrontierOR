import argparse
import json
import time

def get_heuristic_cost(H, unique_dests, dists, tau, U, fixed_costs):
    if not H:
        return float('inf')
    cost = sum(fixed_costs[h] for h in H)
    M = {}
    for i in H:
        M[i] = {}
        for d in unique_dests:
            M[i][d] = min(tau * dists[i][j] + dists[j][d] for j in H)
            
    for u, w in U.items():
        o, d = u
        best_r = min(dists[o][i] + M[i][d] for i in H)
        cost += w * best_r
    return cost

def evaluate_H_exact(H, U, C, fixed_costs):
    if not H:
        return float('inf'), {}
    cost = sum(fixed_costs[h] for h in H)
    u_best_e = {}
    E_H = [ (i, j) for i in H for j in H if i <= j ]
    for u in U:
        best_cost_e = float('inf')
        best_e = None
        C_u = C[u]
        for e in E_H:
            if C_u[e] < best_cost_e:
                best_cost_e = C_u[e]
                best_e = e
        cost += best_cost_e
        u_best_e[u] = best_e
    return cost, u_best_e

def main():
    parser = argparse.ArgumentParser(description="Uncapacitated Hub Location Problem")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    start_time = time.time()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    N = instance['num_nodes']
    num_commodities = instance['num_commodities']
    tau = instance['parameters']['tau']
    nodes = instance['nodes']
    distances = instance['distances']

    fixed_costs = [nodes[i]['fixed_cost'] for i in range(N)]

    U = {}
    commodity_map = []
    unique_dests = set()
    for idx, c in enumerate(instance['commodities']):
        o = c['origin']
        d = c['destination']
        w = c['flow']
        u = (o, d)
        if u not in U:
            U[u] = 0.0
        U[u] += w
        unique_dests.add(d)
        commodity_map.append({
            'commodity': idx,
            'origin': o,
            'destination': d,
            'cost': 0.0,
            'hub_edge': [],
            'w': w
        })

    # Start Fast Heuristic
    best_H = [0]
    best_cost = float('inf')
    for i in range(N):
        c = get_heuristic_cost([i], unique_dests, distances, tau, U, fixed_costs)
        if c < best_cost:
            best_cost = c
            best_H = [i]

    if logger:
        logger.log(best_cost)

    # Greedy Add
    while time.time() - start_time < args.time_limit - 1:
        improved = False
        for i in range(N):
            if i not in best_H:
                 cand_H = best_H + [i]
                 c = get_heuristic_cost(cand_H, unique_dests, distances, tau, U, fixed_costs)
                 if c < best_cost - 1e-5:
                     best_cost = c
                     best_H = cand_H
                     improved = True
                     if logger: logger.log(best_cost)
        if not improved:
            break

    # Greedy Drop
    while time.time() - start_time < args.time_limit - 1:
        improved = False
        for i in best_H:
            if len(best_H) == 1:
                continue
            cand_H = [h for h in best_H if h != i]
            c = get_heuristic_cost(cand_H, unique_dests, distances, tau, U, fixed_costs)
            if c < best_cost - 1e-5:
                best_cost = c
                best_H = cand_H
                improved = True
                if logger: logger.log(best_cost)
        if not improved:
            break

    # Greedy Swap
    improved = True
    while improved and time.time() - start_time < args.time_limit - 1:
        improved = False
        for i in best_H:
            for j in range(N):
                if j not in best_H:
                    cand_H = [h for h in best_H if h != i] + [j]
                    c = get_heuristic_cost(cand_H, unique_dests, distances, tau, U, fixed_costs)
                    if c < best_cost - 1e-5:
                        best_cost = c
                        best_H = cand_H
                        improved = True
                        if logger: logger.log(best_cost)
                        break
            if improved:
                break

    # Prepare MIP
    C = {}
    E_list = []
    for i in range(N):
        for j in range(i, N):
            E_list.append((i, j))

    for u, w in U.items():
        o, d = u
        C_u = {}
        for i, j in E_list:
            cost1 = w * (distances[o][i] + tau * distances[i][j] + distances[j][d])
            cost2 = w * (distances[o][j] + tau * distances[j][i] + distances[i][d])
            C_u[(i, j)] = min(cost1, cost2)
        C[u] = C_u

    candidate_pairs = {u: set() for u in U}
    P = max(50, 2000000 // max(len(U), 1))
    if P > len(E_list):
        P = len(E_list)

    for u in U:
        sorted_E = sorted(E_list, key=lambda e: C[u][e])
        for i in range(P):
            candidate_pairs[u].add(sorted_E[i])

    best_global_cost = best_cost
    best_global_H = best_H

    import gurobipy as gp
    from gurobipy import GRB

    # Column Generation (Iterative RMP)
    for iteration in range(5):
        rem_time = args.time_limit - (time.time() - start_time)
        if rem_time <= 1:
            break

        # Ensure that ALL pairs formed by the incumbent are open to evaluation
        E_H_best = [(i, j) for i in best_global_H for j in best_global_H if i <= j]
        for u in U:
            for e in E_H_best:
                candidate_pairs[u].add(e)

        model = gp.Model("UHLP_RMP")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = max(1.0, rem_time)
        model.Params.Threads = 1

        y = model.addVars(N, vtype=GRB.BINARY, obj=fixed_costs, name="y")
        for h in best_global_H:
             y[h].Start = 1
        for h in range(N):
             if h not in best_global_H:
                  y[h].Start = 0

        needed_Y = set()
        for pairs in candidate_pairs.values():
             needed_Y.update(pairs)

        Y = model.addVars(needed_Y, vtype=GRB.CONTINUOUS, obj=0, name="Y")
        for e in needed_Y:
            i, j = e
            model.addConstr(Y[e] <= y[i])
            if i != j:
                model.addConstr(Y[e] <= y[j])

        for u, pairs in candidate_pairs.items():
            var_dict = model.addVars(pairs, vtype=GRB.CONTINUOUS, name=f"x_{u}")
            for e in pairs:
                var_dict[e].Obj = C[u][e]
            model.addConstr(gp.quicksum(var_dict.values()) == 1)
            for e in pairs:
                model.addConstr(var_dict[e] <= Y[e])

        if logger is not None:
            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj)
            model.optimize(cb)
        else:
            model.optimize()

        if model.SolCount > 0:
            H_star = [i for i in range(N) if y[i].X > 0.5]
        else:
            break

        true_cost, u_best_e = evaluate_H_exact(H_star, U, C, fixed_costs)
        
        if true_cost < best_global_cost:
            best_global_cost = true_cost
            best_global_H = H_star
            if logger:
                logger.log(best_global_cost)

        improved = False
        for u in U:
            if u_best_e[u] not in candidate_pairs[u]:
                candidate_pairs[u].add(u_best_e[u])
                improved = True

        if not improved:
            break

    # Final Extraction
    _, u_best_e = evaluate_H_exact(best_global_H, U, C, fixed_costs)

    assignments = []
    total_assigned_cost = 0.0

    for k in range(num_commodities):
        c = commodity_map[k]
        o, d = c['origin'], c['destination']
        u = (o, d)
        best_e = u_best_e[u]
        w = c['w']
        
        dist1 = distances[o][best_e[0]] + tau * distances[best_e[0]][best_e[1]] + distances[best_e[1]][d]
        dist2 = distances[o][best_e[1]] + tau * distances[best_e[1]][best_e[0]] + distances[best_e[0]][d]
        
        cost1 = w * dist1
        cost2 = w * dist2
        
        if cost1 <= cost2:
            final_cost = cost1
        else:
            final_cost = cost2
            
        if best_e[0] == best_e[1]:
            hub_edge = [best_e[0]]
        else:
            hub_edge = [best_e[0], best_e[1]]
            
        c['hub_edge'] = hub_edge
        c['cost'] = final_cost
        total_assigned_cost += final_cost

        assignments.append({
            "commodity": c['commodity'],
            "origin": o,
            "destination": d,
            "hub_edge": hub_edge,
            "cost": final_cost
        })

    final_objective_value = sum(fixed_costs[h] for h in best_global_H) + total_assigned_cost

    solution = {
        "objective_value": final_objective_value,
        "open_hubs": best_global_H,
        "assignments": assignments
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()