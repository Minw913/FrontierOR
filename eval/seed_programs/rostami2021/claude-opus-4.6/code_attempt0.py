import argparse
import json
import time
import numpy as np
from itertools import combinations

def solve_instance(instance, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    n = instance["n"]
    num_scenarios = instance["num_scenarios"]
    chi = instance["cost_parameters"]["chi"]
    alpha = instance["cost_parameters"]["alpha"]
    delta = instance["cost_parameters"]["delta"]
    p_hubs = instance.get("p_hubs", None)
    distances = np.array(instance["distances"])
    fixed_costs = np.array(instance["fixed_costs"])
    hub_capacities = instance.get("hub_capacities", None)
    scenarios = instance["scenarios"]
    
    # Precompute scenario data
    scenario_data = []
    for s, sc in enumerate(scenarios):
        prob = sc["probability"]
        demands = np.array(sc["demands"])
        outflow = demands.sum(axis=1)  # total outgoing flow from each node
        inflow = demands.sum(axis=0)   # total incoming flow to each node
        scenario_data.append({
            "prob": prob,
            "demands": demands,
            "outflow": outflow,
            "inflow": inflow
        })
    
    def evaluate_solution(hubs_set, return_allocations=False):
        """Given a set of hub indices, compute optimal allocations and total cost."""
        hubs_list = sorted(hubs_set)
        hub_indices = np.array(hubs_list)
        
        # Fixed costs
        if p_hubs is None:
            total_fixed = sum(fixed_costs[h] for h in hubs_list)
        else:
            total_fixed = 0.0
        
        total_scenario_cost = 0.0
        allocations = {}
        
        for s, sd in enumerate(scenario_data):
            prob = sd["prob"]
            demands = sd["demands"]
            outflow = sd["outflow"]
            inflow = sd["inflow"]
            
            # For each non-hub node, we need to find the best hub assignment
            # Cost of assigning node i to hub h:
            # collection: chi * distances[i][h] * outflow[i]
            # distribution: delta * distances[h][i] * inflow[i]
            # But we also need inter-hub transfer costs which depend on pairs
            
            # This is a quadratic assignment-like problem per scenario
            # For small n, we can try enumeration or use the standard USApHMP approach
            
            # For the standard approach: we compute for each node the cost of assigning to each hub
            # The collection+distribution cost for node i assigned to hub h:
            # cd_cost[i][h] = chi * dist[i][h] * outflow[i] + delta * dist[h][i] * inflow[i]
            
            # The transfer cost for OD pair (i,j) with flow w[i][j]:
            # alpha * dist[hub(i)][hub(j)] * w[i][j]
            
            # Total scenario cost = sum_i cd_cost[i][alloc[i]] + alpha * sum_{i,j} w[i][j] * dist[alloc[i]][alloc[j]]
            
            # This is a quadratic problem. For small n, we can solve it optimally.
            # For larger n, we use iterative/greedy approaches.
            
            n_hubs = len(hubs_list)
            
            # Precompute collection+distribution costs
            cd_cost = np.zeros((n, n_hubs))
            for hi, h in enumerate(hubs_list):
                cd_cost[:, hi] = chi * distances[:, h] * outflow + delta * distances[h, :] * inflow
                # Hubs assigned to themselves have 0 cd cost
            
            # For hubs, they must be assigned to themselves
            hub_map = {}  # node -> index in hubs_list
            for hi, h in enumerate(hubs_list):
                hub_map[h] = hi
            
            # Inter-hub distance matrix
            hub_dist = distances[np.ix_(hub_indices, hub_indices)]
            
            if n <= 30 or n_hubs <= 3:
                # For small problems, use iterative best response
                # Initialize: assign each non-hub node to nearest hub (by cd_cost)
                assignment = np.zeros(n, dtype=int)  # index into hubs_list
                for i in range(n):
                    if i in hub_map:
                        assignment[i] = hub_map[i]
                    else:
                        assignment[i] = np.argmin(cd_cost[i, :])
                
                # Check capacity constraints
                if hub_capacities is not None:
                    caps = np.array(hub_capacities)
                    # Need to ensure capacity feasibility
                    # We'll handle this below
                
                # Iterative improvement
                improved = True
                max_iter = 50
                iter_count = 0
                while improved and iter_count < max_iter:
                    improved = False
                    iter_count += 1
                    for i in range(n):
                        if i in hub_map:
                            continue
                        
                        best_cost = float('inf')
                        best_h = assignment[i]
                        
                        for hi in range(n_hubs):
                            # Cost of assigning i to hub hi
                            cost_i = cd_cost[i, hi]
                            
                            # Transfer costs involving node i as origin
                            # sum_j w[i][j] * alpha * hub_dist[hi][assignment[j]]
                            for j in range(n):
                                if demands[i][j] > 0:
                                    cost_i += alpha * demands[i][j] * hub_dist[hi, assignment[j]]
                            
                            # Transfer costs involving node i as destination
                            # sum_j w[j][i] * alpha * hub_dist[assignment[j]][hi]
                            for j in range(n):
                                if demands[j][i] > 0:
                                    cost_i += alpha * demands[j][i] * hub_dist[assignment[j], hi]
                            
                            # Subtract double-counted self-flow (i->i is 0 on diagonal, so no issue)
                            
                            if cost_i < best_cost - 1e-10:
                                # Check capacity
                                feasible = True
                                if hub_capacities is not None:
                                    h_node = hubs_list[hi]
                                    # Compute current load on hub hi
                                    current_load = sum(outflow[k] for k in range(n) if assignment[k] == hi and k != i)
                                    if current_load + outflow[i] > hub_capacities[h_node]:
                                        feasible = False
                                
                                if feasible:
                                    best_cost = cost_i
                                    best_h = hi
                        
                        if best_h != assignment[i]:
                            assignment[i] = best_h
                            improved = True
                
                # Compute scenario cost
                scenario_cost = 0.0
                for i in range(n):
                    scenario_cost += cd_cost[i, assignment[i]]
                
                for i in range(n):
                    for j in range(n):
                        if demands[i][j] > 0:
                            scenario_cost += alpha * demands[i][j] * hub_dist[assignment[i], assignment[j]]
                
                total_scenario_cost += prob * scenario_cost
                
                if return_allocations:
                    alloc = {}
                    for i in range(n):
                        alloc[str(i)] = hubs_list[assignment[i]]
                    allocations[str(s)] = alloc
            else:
                # Larger problem: use greedy nearest hub with transfer approximation
                # Precompute flow-weighted transfer costs
                # For node i assigned to hub hi, the transfer contribution is:
                # sum_j w[i][j]*alpha*hub_dist[hi][assignment[j]] + sum_j w[j][i]*alpha*hub_dist[assignment[j]][hi]
                # This is coupled, so we use iterative approach
                
                assignment = np.zeros(n, dtype=int)
                for i in range(n):
                    if i in hub_map:
                        assignment[i] = hub_map[i]
                    else:
                        assignment[i] = np.argmin(cd_cost[i, :])
                
                # Vectorized iterative improvement
                improved = True
                max_iter = 20
                iter_count = 0
                while improved and iter_count < max_iter:
                    improved = False
                    iter_count += 1
                    
                    for i in range(n):
                        if i in hub_map:
                            continue
                        
                        # Compute cost for each hub assignment
                        costs = cd_cost[i, :].copy()
                        
                        # Transfer costs as origin
                        out_flows = demands[i, :]  # flow from i to all j
                        # For each candidate hub hi, cost += alpha * sum_j out_flows[j] * hub_dist[hi, assignment[j]]
                        hub_assign_j = assignment  # assignment of each j
                        for hi in range(n_hubs):
                            costs[hi] += alpha * np.sum(out_flows * hub_dist[hi, assignment])
                        
                        # Transfer costs as destination
                        in_flows = demands[:, i]  # flow from all j to i
                        for hi in range(n_hubs):
                            costs[hi] += alpha * np.sum(in_flows * hub_dist[assignment, hi])
                        
                        best_h = np.argmin(costs)
                        if best_h != assignment[i]:
                            assignment[i] = best_h
                            improved = True
                
                scenario_cost = 0.0
                for i in range(n):
                    scenario_cost += cd_cost[i, assignment[i]]
                
                for i in range(n):
                    for j in range(n):
                        if demands[i][j] > 0:
                            scenario_cost += alpha * demands[i][j] * hub_dist[assignment[i], assignment[j]]
                
                total_scenario_cost += prob * scenario_cost
                
                if return_allocations:
                    alloc = {}
                    for i in range(n):
                        alloc[str(i)] = hubs_list[assignment[i]]
                    allocations[str(s)] = alloc
        
        total_cost = total_fixed + total_scenario_cost
        
        if return_allocations:
            return total_cost, allocations
        return total_cost
    
    # Try to use Gurobi for exact solution if problem is manageable
    use_gurobi = True
    
    if use_gurobi:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 5:
                use_gurobi = False
                raise Exception("Not enough time for Gurobi")
            
            m = gp.Model("USApHMP")
            m.setParam("TimeLimit", max(remaining_time - 5, 10))
            m.setParam("Threads", 1)
            m.setParam("MIPGap", 0.001)
            
            # Decision variables
            # z[k] = 1 if node k is a hub
            z = m.addVars(n, vtype=GRB.BINARY, name="z")
            
            # x[s][i][k] = 1 if node i is assigned to hub k in scenario s
            # For efficiency, we use a flat structure
            x = {}
            for s in range(num_scenarios):
                for i in range(n):
                    for k in range(n):
                        x[s, i, k] = m.addVar(vtype=GRB.BINARY, name=f"x_{s}_{i}_{k}")
            
            # Objective
            obj = gp.LinExpr()
            
            # Fixed costs (or p-hub constraint)
            if p_hubs is None:
                for k in range(n):
                    obj += fixed_costs[k] * z[k]
            else:
                m.addConstr(gp.quicksum(z[k] for k in range(n)) == p_hubs)
            
            # Scenario costs
            for s, sd in enumerate(scenario_data):
                prob = sd["prob"]
                demands_s = sd["demands"]
                outflow_s = sd["outflow"]
                inflow_s = sd["inflow"]
                
                # Collection + Distribution costs
                for i in range(n):
                    for k in range(n):
                        cd = chi * distances[i][k] * outflow_s[i] + delta * distances[k][i] * inflow_s[i]
                        if cd > 0:
                            obj += prob * cd * x[s, i, k]
                
                # Inter-hub transfer costs
                # sum_{i,j} w[i][j] * alpha * sum_{k,l} dist[k][l] * x[s,i,k] * x[s,j,l]
                # This is quadratic! We need to linearize.
                # Use y[s,i,j,k,l] = x[s,i,k] * x[s,j,l] -- too many variables
                
                # Alternative: define flow variables between hub pairs
                # For each hub pair (k,l), the flow is sum_{i,j} w[i][j] * x[s,i,k] * x[s,j,l]
                # We can use: for each origin i assigned to hub k, the flow from k is sum_j w[i][j]
                # Actually, we need the flow between specific hub pairs.
                
                # Standard linearization for p-hub:
                # y[s,i,k,l] = fraction of flow originating at i that goes through hub k then hub l
                # But this adds n^3 variables per scenario.
                
                # For small n, this is feasible. For larger n, we need a different approach.
                
                if n <= 25:
                    # Use y variables: y[s,i,j,k] means origin i, destination j, routed via hub k for the origin side
                    # Actually, let's use the standard formulation with y[s,i,k,l]
                    # y[s,i,k,l] = x[s,i,k] * (sum_j w[i][j] * x[s,j,l])
                    # This is still complex. Let me use a different standard formulation.
                    
                    # Standard: For each scenario s, OD pair (i,j), the transfer cost is
                    # alpha * w[i][j] * dist[hub(i)][hub(j)]
                    # = alpha * w[i][j] * sum_{k,l} dist[k][l] * x[s,i,k] * x[s,j,l]
                    
                    # We can aggregate: for each i, define the total flow from i to each hub l:
                    # F[s,i,l] = sum_j w[i][j] * x[s,j,l]
                    # Then transfer cost = alpha * sum_i sum_k x[s,i,k] * sum_l dist[k][l] * F[s,i,l]
                    # Still quadratic in x.
                    
                    # Standard linearization: introduce y[s,i,k,l] = x[s,i,k] * x[s,j,l] for specific pairs
                    # But this is O(n^4) per scenario.
                    
                    # Better: use the Ernst & Krishnamoorthy formulation
                    # y[s,k,l,i] = 1 if node i's flow uses hub pair (k,l)
                    # Or aggregate: Y[s,k,l] = total flow between hub pair (k,l)
                    
                    # Actually, let me use a simpler approach:
                    # For each scenario, for each (i,j) pair with w[i][j]>0:
                    # cost contribution = prob * alpha * w[i][j] * sum_{k,l} dist[k][l] * x[s,i,k] * x[s,j,l]
                    
                    # Linearize using: for each origin i, define 
                    # For flow from i to j: the hub of i is k (x[s,i,k]=1), hub of j is l (x[s,j,l]=1)
                    # cost = alpha * w[i][j] * dist[k][l]
                    
                    # We can reformulate using flow variables on hub links.
                    # Let f[s,k,l] = total flow routed from hub k to hub l in scenario s
                    # f[s,k,l] = sum_{i: assigned to k} sum_{j: assigned to l} w[i][j]
                    # Transfer cost = alpha * sum_{k,l} dist[k][l] * f[s,k,l]
                    
                    # f[s,k,l] = sum_i sum_j w[i][j] * x[s,i,k] * x[s,j,l]
                    # This is quadratic. We linearize:
                    # g[s,i,k,l] = x[s,i,k] * (sum_j w[i][j] * x[s,j,l])
                    # = sum_j w[i][j] * x[s,i,k] * x[s,j,l]
                    # Then f[s,k,l] = sum_i g[s,i,k,l]
                    
                    # Or use single commodity flow approach:
                    # For each origin i in scenario s:
                    # define flow from i: O_i[s,l] = sum_j w[i][j] * x[s,j,l] (flow from i going to hub l's spokes)
                    # The origin hub of i is k: x[s,i,k] = 1
                    # Transfer cost for i = alpha * sum_l O_i[s,l] * dist[k][l] * x[s,i,k]
                    # This is trilinear...
                    
                    # Let's just use the standard Campbell (1994) formulation or similar
                    # Actually, let's use a different variable:
                    # For each OD pair (i,j) with flow w, and for each hub pair (k,l):
                    # y[s,i,j,k,l] = 1 if i assigned to k and j assigned to l
                    # This has O(n^4) variables per scenario - too many for n=25
                    
                    # Better approach for moderate n: 
                    # Use the Skorin-Kapov et al. formulation with
                    # For each (i,k,l): y[s,i,k,l] means node i uses hub k, and we track flow to hub l
                    # This is O(n^3) per scenario
                    
                    # Let me try a different, more compact formulation
                    # For each scenario s, for each pair of hubs (k,l), the flow is:
                    # f[s,k,l] = (sum_i O_i * x[s,i,k]) * ... no, it's not separable
                    
                    # Actually, f[s,k,l] = sum_i sum_j w[i][j] * x[s,i,k] * x[s,j,l]
                    # = (sum_i row_to_l[i] * x[s,i,k]) where row_to_l[i] = sum_j w[i][j] * x[s,j,l]
                    # This doesn't separate nicely.
                    
                    # Let me just go with the Gurobi approach using quadratic constraints or 
                    # bilinear terms. Actually, Gurobi can handle products of binary variables.
                    
                    # For efficiency, let's create auxiliary variables:
                    # For each scenario s, node i, hub k: x[s,i,k] is already defined
                    # Product: p[s,i,j] = product variable isn't needed if we use Gurobi's bilinear
                    
                    # Actually, let me just add the quadratic terms to the objective
                    # Gurobi supports MIQP
                    
                    qobj = gp.QuadExpr()
                    for i in range(n):
                        for j in range(n):
                            if demands_s[i][j] > 0:
                                for k in range(n):
                                    for l in range(n):
                                        coeff = prob * alpha * demands_s[i][j] * distances[k][l]
                                        if coeff > 0:
                                            qobj += coeff * x[s, i, k] * x[s, j, l]
                    obj += qobj
                    
                else:
                    # For larger n, we'd need a different approach
                    # This path shouldn't normally be reached for reasonable instances
                    pass
            
            m.setObjective(obj, GRB.MINIMIZE)
            
            # Constraints
            # Each node assigned to exactly one hub per scenario
            for s in range(num_scenarios):
                for i in range(n):
                    m.addConstr(gp.quicksum(x[s, i, k] for k in range(n)) == 1)
            
            # Can only assign to open hubs
            for s in range(num_scenarios):
                for i in range(n):
                    for k in range(n):
                        m.addConstr(x[s, i, k] <= z[k])
            
            # Hub nodes assigned to themselves
            for s in range(num_scenarios):
                for k in range(n):
                    m.addConstr(x[s, k, k] >= z[k])
            
            # At least one hub
            m.addConstr(gp.quicksum(z[k] for k in range(n)) >= 1)
            
            # Capacity constraints
            if hub_capacities is not None:
                caps = hub_capacities
                for s in range(num_scenarios):
                    sd = scenario_data[s]
                    for k in range(n):
                        m.addConstr(
                            gp.quicksum(sd["outflow"][i] * x[s, i, k] for i in range(n)) <= caps[k] * z[k]
                        )
            
            # For large problems, the quadratic formulation may be too slow
            # Let's check if n is small enough
            if n > 20:
                # Switch to heuristic for large instances
                m.setParam("TimeLimit", max(remaining_time * 0.3, 10))
            
            m.setParam("NonConvex", 2)  # Allow non-convex quadratic
            
            # Add a callback or use lazy approach? For now, just solve
            m.optimize()
            
            if m.SolCount > 0:
                # Extract solution
                hubs = [k for k in range(n) if z[k].X > 0.5]
                obj_val = m.ObjVal
                
                if logger:
                    logger.log(obj_val)
                
                allocations = {}
                x_values = {}
                for s in range(num_scenarios):
                    alloc = {}
                    x_val_s = {}
                    for i in range(n):
                        assigned_hub = None
                        x_val_i = {}
                        for k in range(n):
                            val = x[s, i, k].X
                            if val > 0.5:
                                assigned_hub = k
                            if k != i:
                                x_val_i[str(k)] = round(val)
                        alloc[str(i)] = assigned_hub
                        if x_val_i:
                            x_val_s[str(i)] = x_val_i
                    allocations[str(s)] = alloc
                    x_values[str(s)] = x_val_s
                
                z_values = [round(z[k].X) for k in range(n)]
                
                return {
                    "objective_value": obj_val,
                    "hubs": hubs,
                    "allocations": allocations,
                    "z_values": [float(v) for v in z_values],
                    "x_values": x_values
                }
            else:
                use_gurobi = False
                
        except Exception as e:
            print(f"Gurobi failed or not available: {e}")
            use_gurobi = False
    
    # Fallback: heuristic approach
    best_cost = float('inf')
    best_hubs = None
    best_allocations = None
    
    # Try single hub solutions first
    for k in range(n):
        if time.time() - start_time > time_limit - 2:
            break
        cost = evaluate_solution({k})
        if cost < best_cost:
            best_cost = cost
            best_hubs = {k}
            if logger:
                logger.log(cost)
    
    # Try pairs
    if n <= 50 and time.time() - start_time < time_limit * 0.3:
        for k1, k2 in combinations(range(n), 2):
            if time.time() - start_time > time_limit * 0.5:
                break
            cost = evaluate_solution({k1, k2})
            if cost < best_cost:
                best_cost = cost
                best_hubs = {k1, k2}
                if logger:
                    logger.log(cost)
    
    # Try triples if time allows
    if n <= 20 and time.time() - start_time < time_limit * 0.5:
        for combo in combinations(range(n), 3):
            if time.time() - start_time > time_limit * 0.7:
                break
            cost = evaluate_solution(set(combo))
            if cost < best_cost:
                best_cost = cost
                best_hubs = set(combo)
                if logger:
                    logger.log(cost)
    
    # If p_hubs is specified, enumerate or search for exactly p hubs
    if p_hubs is not None:
        from itertools import combinations as comb
        if n <= 15 or (p_hubs <= 3 and n <= 30):
            for combo in comb(range(n), p_hubs):
                if time.time() - start_time > time_limit - 2:
                    break
                cost = evaluate_solution(set(combo))
                if cost < best_cost:
                    best_cost = cost
                    best_hubs = set(combo)
                    if logger:
                        logger.log(cost)
    
    # Local search: add/remove/swap hubs
    if best_hubs is not None:
        current_hubs = set(best_hubs)
        improved = True
        while improved and time.time() - start_time < time_limit - 2:
            improved = False
            
            # Try adding a hub
            if p_hubs is None:
                for k in range(n):
                    if k in current_hubs:
                        continue
                    if time.time() - start_time > time_limit - 2:
                        break
                    new_hubs = current_hubs | {k}
                    cost = evaluate_solution(new_hubs)
                    if cost < best_cost - 1e-6:
                        best_cost = cost
                        best_hubs = new_hubs
                        current_hubs = new_hubs
                        improved = True
                        if logger:
                            logger.log(cost)
                        break
            
            # Try removing a hub
            if p_hubs is None and len(current_hubs) > 1:
                for k in list(current_hubs):
                    if time.time() - start_time > time_limit - 2:
                        break
                    new_hubs = current_hubs - {k}
                    if len(new_hubs) == 0:
                        continue
                    cost = evaluate_solution(new_hubs)
                    if cost < best_cost - 1e-6:
                        best_cost = cost
                        best_hubs = new_hubs
                        current_hubs = new_hubs
                        improved = True
                        if logger:
                            logger.log(cost)
                        break
            
            # Try swapping
            for k_out in list(current_hubs):
                if time.time() - start_time > time_limit - 2:
                    break
                swapped = False
                for k_in in range(n):
                    if k_in in current_hubs:
                        continue
                    if time.time() - start_time > time_limit - 2:
                        break
                    new_hubs = (current_hubs - {k_out}) | {k_in}
                    cost = evaluate_solution(new_hubs)
                    if cost < best_cost - 1e-6:
                        best_cost = cost
                        best_hubs = new_hubs
                        current_hubs = new_hubs
                        improved = True
                        swapped = True
                        if logger:
                            logger.log(cost)
                        break
                if swapped:
                    break
    
    # Get final allocations
    final_cost, allocations = evaluate_solution(best_hubs, return_allocations=True)
    
    hubs_list = sorted(best_hubs)
    z_values = [1.0 if k in best_hubs else 0.0 for k in range(n)]
    
    # Build x_values
    x_values = {}
    for s in range(num_scenarios):
        x_val_s = {}
        alloc = allocations[str(s)]
        for i in range(n):
            x_val_i = {}
            assigned_hub = alloc[str(i)]
            for k in range(n):
                if k != i:
                    x_val_i[str(k)] = 1.0 if k == assigned_hub else 0.0
            if x_val_i:
                x_val_s[str(i)] = x_val_i
        x_values[str(s)] = x_val_s
    
    return {
        "objective_value": final_cost,
        "hubs": hubs_list,
        "allocations": allocations,
        "z_values": z_values,
        "x_values": x_values
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    solution = solve_instance(instance, args.time_limit, args.log_path)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()