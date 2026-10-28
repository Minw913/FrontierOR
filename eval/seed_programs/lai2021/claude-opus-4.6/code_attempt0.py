import argparse
import json
import time
import random
import numpy as np
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    start_time = time.time()
    
    # Initialize logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    N = instance['N']
    m = instance['m']
    group_caps = instance['group_capacities']
    distances_upper = instance['distances_upper_triangular']
    
    # Build distance matrix
    dist = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            dist[i][j] = distances_upper[idx]
            dist[j][i] = distances_upper[idx]
            idx += 1
    
    L = [group_caps[g]['L_g'] for g in range(m)]
    U = [group_caps[g]['U_g'] for g in range(m)]
    
    def compute_objective(assignment):
        """Compute total diversity given assignment array."""
        obj = 0.0
        for g in range(m):
            members = np.where(assignment == g)[0]
            for i_idx in range(len(members)):
                for j_idx in range(i_idx + 1, len(members)):
                    obj += dist[members[i_idx], members[j_idx]]
        return obj
    
    def compute_objective_fast(assignment, group_members):
        """Compute total diversity using precomputed group members."""
        obj = 0.0
        for g in range(m):
            members = group_members[g]
            nm = len(members)
            for i_idx in range(nm):
                for j_idx in range(i_idx + 1, nm):
                    obj += dist[members[i_idx], members[j_idx]]
        return obj
    
    def compute_group_contribution(members):
        """Compute sum of pairwise distances within a group."""
        total = 0.0
        nm = len(members)
        for i_idx in range(nm):
            for j_idx in range(i_idx + 1, nm):
                total += dist[members[i_idx], members[j_idx]]
        return total
    
    def element_contribution_to_group(elem, group_members):
        """Compute sum of distances from elem to all members of a group."""
        total = 0.0
        for member in group_members:
            if member != elem:
                total += dist[elem][member]
        return total
    
    # Precompute sum of distances from each element to each other element
    # dist_sum[i] = sum of distances from element i to all others
    dist_sum_to_all = np.sum(dist, axis=1)
    
    # For small instances, try Gurobi MIP
    use_mip = (N <= 60 and m <= 10)
    
    best_assignment = None
    best_obj = -1
    best_group_members = None
    
    if use_mip:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining = time_limit - (time.time() - start_time)
            if remaining > 1:
                model = gp.Model()
                model.setParam('OutputFlag', 0)
                model.setParam('TimeLimit', remaining - 0.5)
                model.setParam('Threads', 1)
                
                # x[i,g] = 1 if element i assigned to group g
                x = {}
                for i in range(N):
                    for g in range(m):
                        x[i, g] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{g}')
                
                # Each element assigned to exactly one group
                for i in range(N):
                    model.addConstr(gp.quicksum(x[i, g] for g in range(m)) == 1)
                
                # Capacity constraints
                for g in range(m):
                    model.addConstr(gp.quicksum(x[i, g] for i in range(N)) >= L[g])
                    model.addConstr(gp.quicksum(x[i, g] for i in range(N)) <= U[g])
                
                # Objective: maximize sum of d(i,j) * y[i,j,g] where y[i,j,g] = x[i,g]*x[j,g]
                # We can linearize or use quadratic objective
                # For moderate N, use quadratic objective
                
                if N <= 40:
                    # Quadratic objective
                    obj = gp.QuadExpr()
                    for i in range(N):
                        for j in range(i + 1, N):
                            if dist[i][j] > 0:
                                for g in range(m):
                                    obj += dist[i][j] * x[i, g] * x[j, g]
                    model.setObjective(obj, GRB.MAXIMIZE)
                else:
                    # Linearize with auxiliary variables for pairs
                    y = {}
                    obj_expr = gp.LinExpr()
                    for i in range(N):
                        for j in range(i + 1, N):
                            if dist[i][j] > 0:
                                for g in range(m):
                                    y[i, j, g] = model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{j}_{g}')
                                    model.addConstr(y[i, j, g] <= x[i, g])
                                    model.addConstr(y[i, j, g] <= x[j, g])
                                    model.addConstr(y[i, j, g] >= x[i, g] + x[j, g] - 1)
                                    obj_expr += dist[i][j] * y[i, j, g]
                    model.setObjective(obj_expr, GRB.MAXIMIZE)
                
                # Callback for logging
                def callback(model, where):
                    if where == GRB.Callback.MIPSOL:
                        obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if logger:
                            logger.log(obj_val)
                
                model.optimize(callback)
                
                if model.SolCount > 0:
                    assignment = np.zeros(N, dtype=int)
                    for i in range(N):
                        for g in range(m):
                            if x[i, g].X > 0.5:
                                assignment[i] = g
                                break
                    
                    group_members = {g: [] for g in range(m)}
                    for i in range(N):
                        group_members[assignment[i]].append(i)
                    
                    obj_val = compute_objective_fast(assignment, group_members)
                    best_assignment = assignment
                    best_obj = obj_val
                    best_group_members = group_members
                    if logger:
                        logger.log(best_obj)
        except Exception as e:
            pass
    
    # If MIP didn't work or for larger instances, use heuristic approach
    if best_assignment is None or (time.time() - start_time) < time_limit * 0.5:
        # Greedy construction + local search
        
        def greedy_construct(seed=None):
            """Construct a feasible solution greedily."""
            if seed is not None:
                rng = random.Random(seed)
            else:
                rng = random.Random()
            
            assignment = np.full(N, -1, dtype=int)
            group_members = {g: [] for g in range(m)}
            group_sizes = [0] * m
            
            # First, assign elements to meet minimum capacity
            unassigned = list(range(N))
            rng.shuffle(unassigned)
            
            for g in range(m):
                needed = L[g]
                for _ in range(needed):
                    if not unassigned:
                        break
                    elem = unassigned.pop()
                    assignment[elem] = g
                    group_members[g].append(elem)
                    group_sizes[g] += 1
            
            # Assign remaining elements greedily to maximize contribution
            for elem in unassigned:
                best_g = -1
                best_gain = -float('inf')
                for g in range(m):
                    if group_sizes[g] < U[g]:
                        gain = element_contribution_to_group(elem, group_members[g])
                        if gain > best_gain:
                            best_gain = gain
                            best_g = g
                if best_g >= 0:
                    assignment[elem] = best_g
                    group_members[best_g].append(elem)
                    group_sizes[best_g] += 1
            
            return assignment, group_members, group_sizes
        
        def local_search(assignment, group_members, group_sizes, time_remaining):
            """Improve solution by swap moves."""
            ls_start = time.time()
            
            # Precompute contribution of each element to its group
            contrib = np.zeros(N)
            for i in range(N):
                g = assignment[i]
                contrib[i] = element_contribution_to_group(i, group_members[g])
            
            current_obj = sum(compute_group_contribution(group_members[g]) for g in range(m))
            improved = True
            
            while improved and (time.time() - ls_start) < time_remaining:
                improved = False
                
                # Try move: move element i from group g1 to group g2
                elements = list(range(N))
                random.shuffle(elements)
                
                for i in elements:
                    if (time.time() - ls_start) >= time_remaining:
                        break
                    g1 = assignment[i]
                    if group_sizes[g1] <= L[g1]:
                        continue  # Can't remove from this group
                    
                    loss = contrib[i]  # Contribution of i to g1
                    
                    for g2 in range(m):
                        if g2 == g1:
                            continue
                        if group_sizes[g2] >= U[g2]:
                            continue
                        
                        gain = element_contribution_to_group(i, group_members[g2])
                        
                        if gain > loss:
                            # Execute move
                            group_members[g1].remove(i)
                            group_sizes[g1] -= 1
                            group_members[g2].append(i)
                            group_sizes[g2] += 1
                            assignment[i] = g2
                            
                            # Update contributions for affected elements
                            for j in group_members[g1]:
                                contrib[j] -= dist[i][j]
                            for j in group_members[g2]:
                                if j != i:
                                    contrib[j] += dist[i][j]
                            contrib[i] = gain
                            
                            current_obj += gain - loss
                            improved = True
                            break
                
                # Try swap: swap element i (in g1) with element j (in g2)
                if not improved:
                    elements = list(range(N))
                    random.shuffle(elements)
                    
                    for i in elements:
                        if (time.time() - ls_start) >= time_remaining:
                            break
                        g1 = assignment[i]
                        found_swap = False
                        
                        for g2 in range(m):
                            if g2 == g1:
                                continue
                            if found_swap:
                                break
                            
                            members_g2 = list(group_members[g2])
                            random.shuffle(members_g2)
                            
                            for j in members_g2:
                                # Cost of removing i from g1 and j from g2
                                loss_i = contrib[i] - dist[i][j] if j in group_members[g1] else contrib[i]
                                loss_j = contrib[j] - dist[i][j] if i in group_members[g2] else contrib[j]
                                
                                # Actually, i is in g1, j is in g2, they're different groups
                                loss_i_from_g1 = element_contribution_to_group(i, group_members[g1]) - dist[i][j] * 0  # j not in g1
                                # Wait, let me recalculate properly
                                # Removing i from g1: lose contrib[i] (which is sum of dist[i][k] for k in g1, k!=i)
                                # Adding j to g1: gain sum of dist[j][k] for k in g1 after removing i
                                # Removing j from g2: lose contrib[j]
                                # Adding i to g2: gain sum of dist[i][k] for k in g2 after removing j
                                
                                loss_i_g1 = contrib[i]
                                gain_j_g1 = element_contribution_to_group(j, group_members[g1]) - dist[i][j]
                                
                                loss_j_g2 = contrib[j]
                                gain_i_g2 = element_contribution_to_group(i, group_members[g2]) - dist[i][j]
                                
                                net = gain_j_g1 + gain_i_g2 - loss_i_g1 - loss_j_g2
                                
                                if net > 1e-9:
                                    # Execute swap
                                    group_members[g1].remove(i)
                                    group_members[g2].remove(j)
                                    group_members[g1].append(j)
                                    group_members[g2].append(i)
                                    assignment[i] = g2
                                    assignment[j] = g1
                                    
                                    # Recompute contributions for all affected elements
                                    for k in group_members[g1]:
                                        if k == j:
                                            contrib[k] = element_contribution_to_group(k, group_members[g1])
                                        else:
                                            contrib[k] = contrib[k] - dist[k][i] + dist[k][j]
                                    for k in group_members[g2]:
                                        if k == i:
                                            contrib[k] = element_contribution_to_group(k, group_members[g2])
                                        else:
                                            contrib[k] = contrib[k] - dist[k][j] + dist[k][i]
                                    
                                    current_obj += net
                                    improved = True
                                    found_swap = True
                                    break
            
            return assignment, group_members, group_sizes, current_obj
        
        # Multi-start heuristic
        num_starts = 0
        while (time.time() - start_time) < time_limit - 1:
            remaining = time_limit - (time.time() - start_time) - 0.5
            if remaining < 0.5:
                break
            
            assignment, group_members, group_sizes = greedy_construct(seed=num_starts)
            
            # Check feasibility
            feasible = True
            for g in range(m):
                if group_sizes[g] < L[g] or group_sizes[g] > U[g]:
                    feasible = False
                    break
            
            if not feasible:
                num_starts += 1
                continue
            
            ls_time = min(remaining, max(2.0, remaining * 0.5))
            assignment, group_members, group_sizes, obj = local_search(
                assignment, group_members, group_sizes, ls_time
            )
            
            if obj > best_obj:
                best_obj = obj
                best_assignment = assignment.copy()
                best_group_members = {g: list(members) for g, members in group_members.items()}
                if logger:
                    logger.log(best_obj)
            
            num_starts += 1
    
    # If we still don't have a solution, create a simple feasible one
    if best_assignment is None:
        assignment = np.zeros(N, dtype=int)
        group_members = {g: [] for g in range(m)}
        group_sizes = [0] * m
        
        idx = 0
        for g in range(m):
            for _ in range(L[g]):
                if idx < N:
                    assignment[idx] = g
                    group_members[g].append(idx)
                    group_sizes[g] += 1
                    idx += 1
        
        # Assign remaining
        for i in range(idx, N):
            for g in range(m):
                if group_sizes[g] < U[g]:
                    assignment[i] = g
                    group_members[g].append(i)
                    group_sizes[g] += 1
                    break
        
        best_assignment = assignment
        best_group_members = group_members
        best_obj = compute_objective_fast(assignment, group_members)
        if logger:
            logger.log(best_obj)
    
    # Build output
    assignment_dict = {str(i): int(best_assignment[i]) for i in range(N)}
    groups_dict = {}
    for g in range(m):
        if best_group_members is not None:
            groups_dict[str(g)] = sorted(best_group_members[g])
        else:
            groups_dict[str(g)] = sorted([i for i in range(N) if best_assignment[i] == g])
    
    solution = {
        "objective_value": float(best_obj),
        "assignment": assignment_dict,
        "groups": groups_dict
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()