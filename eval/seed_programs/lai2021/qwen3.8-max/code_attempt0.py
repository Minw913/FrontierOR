import argparse
import json
import time
import numpy as np
from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()


def load_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def build_distance_matrix(N, distances):
    dist = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            dist[i][j] = distances[idx]
            dist[j][i] = distances[idx]
            idx += 1
    return dist


def compute_objective(assignment, dist, N, m):
    obj = 0.0
    groups = [[] for _ in range(m)]
    for i in range(N):
        groups[assignment[i]].append(i)
    for g in range(m):
        elems = groups[g]
        for a in range(len(elems)):
            for b in range(a + 1, len(elems)):
                obj += dist[elems[a]][elems[b]]
    return obj


def build_solution_dict(assignment, obj, N, m):
    groups = {}
    for g in range(m):
        groups[str(g)] = sorted([i for i in range(N) if assignment[i] == g])
    assignment_dict = {str(i): assignment[i] for i in range(N)}
    return {
        "objective_value": float(obj),
        "assignment": assignment_dict,
        "groups": groups
    }


def generate_feasible_assignment(N, m, L, U, dist, rng):
    """Generate a feasible assignment respecting capacity constraints."""
    assignment = [-1] * N
    group_sizes = [0] * m
    
    # Shuffle elements
    elements = list(range(N))
    rng.shuffle(elements)
    
    # First satisfy lower bounds
    idx = 0
    for g in range(m):
        for _ in range(L[g]):
            if idx < N:
                assignment[elements[idx]] = g
                group_sizes[g] += 1
                idx += 1
    
    # Distribute remaining elements
    while idx < N:
        assigned = False
        for g in range(m):
            if group_sizes[g] < U[g] and idx < N:
                assignment[elements[idx]] = g
                group_sizes[g] += 1
                idx += 1
                assigned = True
        if not assigned:
            break
    
    return assignment


def greedy_construction(N, m, dist, L, U, rng):
    """Greedy construction maximizing diversity."""
    assignment = [-1] * N
    group_members = [[] for _ in range(m)]
    
    # Sort elements by total distance (descending) - spread high-distance elements
    total_dist = np.sum(dist, axis=1)
    order = list(np.argsort(-total_dist))
    rng.shuffle(order)  # Add some randomness for ties
    
    # First pass: satisfy lower bounds
    # Assign elements to groups that need them most
    remaining = list(order)
    
    for g in range(m):
        needed = L[g]
        while len(group_members[g]) < needed and remaining:
            # Pick the element that adds most diversity to this group
            best_elem = None
            best_gain = -float('inf')
            for e in remaining:
                gain = sum(dist[e][j] for j in group_members[g])
                if gain > best_gain:
                    best_gain = gain
                    best_elem = e
            if best_elem is not None:
                assignment[best_elem] = g
                group_members[g].append(best_elem)
                remaining.remove(best_elem)
    
    # Second pass: assign remaining elements greedily
    for elem in remaining:
        best_gain = -float('inf')
        best_group = -1
        for g in range(m):
            if len(group_members[g]) >= U[g]:
                continue
            gain = sum(dist[elem][j] for j in group_members[g])
            if gain > best_gain:
                best_gain = gain
                best_group = g
        
        if best_group == -1:
            # Fallback
            for g in range(m):
                if len(group_members[g]) < U[g]:
                    best_group = g
                    break
        
        if best_group >= 0:
            assignment[elem] = best_group
            group_members[best_group].append(elem)
    
    return assignment


def local_search(assignment, dist, N, m, L, U, time_limit, start_time, logger=None):
    """Local search using swap moves."""
    best_assignment = assignment[:]
    best_obj = compute_objective(assignment, dist, N, m)
    
    if logger:
        solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
        logger.log_solution(best_obj, solution_dict)
    
    group_members = [[] for _ in range(m)]
    for i in range(N):
        group_members[assignment[i]].append(i)
    
    # Precompute S[i][g] = sum of dist[i][k] for k in group g
    S = np.zeros((N, m))
    for g in range(m):
        for i in range(N):
            S[i][g] = sum(dist[i][k] for k in group_members[g])
    
    max_passes = 100
    for pass_num in range(max_passes):
        if time.time() - start_time > time_limit * 0.7:
            break
        
        improved = False
        
        # Try all swaps
        for i in range(N):
            gi = assignment[i]
            for j in range(i + 1, N):
                gj = assignment[j]
                if gi == gj:
                    continue
                
                # Compute swap gain using precomputed S
                # After swap: i goes to gj, j goes to gi
                # Gain for i moving to gj: S[i][gj] - dist[i][j] - (S[i][gi] - dist[i][i])
                #   = S[i][gj] - dist[i][j] - S[i][gi]  (since dist[i][i]=0)
                # Gain for j moving to gi: S[j][gi] - dist[i][j] - (S[j][gj] - dist[j][j])
                #   = S[j][gi] - dist[i][j] - S[j][gj]
                # Total delta = (S[i][gj] - dist[i][j] - S[i][gi]) + (S[j][gi] - dist[i][j] - S[j][gj])
                
                delta = (S[i][gj] - dist[i][j] - S[i][gi]) + \
                        (S[j][gi] - dist[i][j] - S[j][gj])
                
                if delta > 1e-10:
                    # Perform swap
                    assignment[i] = gj
                    assignment[j] = gi
                    group_members[gi].remove(i)
                    group_members[gi].append(j)
                    group_members[gj].remove(j)
                    group_members[gj].append(i)
                    
                    # Update S values
                    # i moved from gi to gj
                    # j moved from gj to gi
                    for k in range(N):
                        S[k][gi] += -dist[k][i] + dist[k][j]
                        S[k][gj] += -dist[k][j] + dist[k][i]
                    
                    best_obj += delta
                    improved = True
                    
                    if logger and pass_num < 5:
                        solution_dict = build_solution_dict(assignment, best_obj, N, m)
                        logger.log_solution(best_obj, solution_dict)
        
        if not improved:
            break
    
    best_assignment = assignment[:]
    best_obj = compute_objective(assignment, dist, N, m)
    
    if logger:
        solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
        logger.log_solution(best_obj, solution_dict)
    
    return best_assignment, best_obj


def local_search_move(assignment, dist, N, m, L, U, time_limit, start_time, logger=None):
    """Local search using move (relocation) operations."""
    best_assignment = assignment[:]
    best_obj = compute_objective(assignment, dist, N, m)
    
    group_members = [[] for _ in range(m)]
    group_sizes = [0] * m
    for i in range(N):
        group_members[assignment[i]].append(i)
        group_sizes[assignment[i]] += 1
    
    S = np.zeros((N, m))
    for g in range(m):
        for i in range(N):
            S[i][g] = sum(dist[i][k] for k in group_members[g])
    
    for pass_num in range(50):
        if time.time() - start_time > time_limit * 0.75:
            break
        
        improved = False
        for i in range(N):
            gi = assignment[i]
            for g in range(m):
                if g == gi:
                    continue
                if group_sizes[g] >= U[g]:
                    continue
                if group_sizes[gi] <= L[gi]:
                    continue
                
                # Gain of moving i from gi to g
                # Remove i from gi: lose S[i][gi] - dist[i][i] = S[i][gi]
                # Add i to g: gain S[i][g]
                delta = S[i][g] - S[i][gi]
                
                if delta > 1e-10:
                    assignment[i] = g
                    group_members[gi].remove(i)
                    group_members[g].append(i)
                    group_sizes[gi] -= 1
                    group_sizes[g] += 1
                    
                    for k in range(N):
                        S[k][gi] -= dist[k][i]
                        S[k][g] += dist[k][i]
                    
                    best_obj += delta
                    improved = True
        
        if not improved:
            break
    
    best_assignment = assignment[:]
    best_obj = compute_objective(assignment, dist, N, m)
    
    if logger:
        solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
        logger.log_solution(best_obj, solution_dict)
    
    return best_assignment, best_obj


def solve_with_gurobi(N, m, dist, L, U, time_remaining, initial_assignment=None):
    """Solve using Gurobi MIQP formulation."""
    import gurobipy as gp
    from gurobipy import GRB
    
    try:
        model = gp.Model("max_diversity")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, time_remaining))
        model.setParam('OutputFlag', 0)
        
        # Decision variables
        x = {}
        for i in range(N):
            for g in range(m):
                x[i, g] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{g}")
        
        model.update()
        
        # Objective: maximize sum_{i<j} sum_g d(i,j) * x[i,g] * x[j,g]
        obj = gp.QuadExpr()
        for i in range(N):
            for j in range(i + 1, N):
                d_ij = dist[i][j]
                if d_ij > 0:
                    for g in range(m):
                        obj += d_ij * x[i, g] * x[j, g]
        
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # Assignment constraints
        for i in range(N):
            model.addConstr(gp.quicksum(x[i, g] for g in range(m)) == 1, name=f"assign_{i}")
        
        # Capacity constraints
        for g in range(m):
            model.addConstr(gp.quicksum(x[i, g] for i in range(N)) >= L[g], name=f"lower_{g}")
            model.addConstr(gp.quicksum(x[i, g] for i in range(N)) <= U[g], name=f"upper_{g}")
        
        # Warm start from heuristic solution
        if initial_assignment is not None:
            for i in range(N):
                for g in range(m):
                    x[i, g].Start = 1.0 if initial_assignment[i] == g else 0.0
        
        model.optimize()
        
        if model.SolCount > 0:
            assignment = [0] * N
            for i in range(N):
                for g in range(m):
                    if x[i, g].X > 0.5:
                        assignment[i] = g
                        break
            return assignment, model.ObjVal
        
    except Exception as e:
        pass
    
    return None, None


def solve_with_gurobi_linearized(N, m, dist, L, U, time_remaining, initial_assignment=None):
    """Solve using linearized Gurobi MIP formulation for larger instances."""
    import gurobipy as gp
    from gurobipy import GRB
    
    try:
        model = gp.Model("max_diversity_lin")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, time_remaining))
        model.setParam('OutputFlag', 0)
        
        # Decision variables
        x = {}
        for i in range(N):
            for g in range(m):
                x[i, g] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{g}")
        
        # Linearization variables: y[i,j,g] = x[i,g] * x[j,g]
        y = {}
        for i in range(N):
            for j in range(i + 1, N):
                for g in range(m):
                    if dist[i][j] > 0:
                        y[i, j, g] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{g}")
        
        model.update()
        
        # Objective
        obj = gp.LinExpr()
        for (i, j, g), var in y.items():
            obj += dist[i][j] * var
        
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # Assignment constraints
        for i in range(N):
            model.addConstr(gp.quicksum(x[i, g] for g in range(m)) == 1)
        
        # Capacity constraints
        for g in range(m):
            model.addConstr(gp.quicksum(x[i, g] for i in range(N)) >= L[g])
            model.addConstr(gp.quicksum(x[i, g] for i in range(N)) <= U[g])
        
        # Linearization constraints
        for (i, j, g), var in y.items():
            model.addConstr(var <= x[i, g])
            model.addConstr(var <= x[j, g])
            model.addConstr(var >= x[i, g] + x[j, g] - 1)
        
        # Warm start
        if initial_assignment is not None:
            for i in range(N):
                for g in range(m):
                    x[i, g].Start = 1.0 if initial_assignment[i] == g else 0.0
            for (i, j, g), var in y.items():
                val = 1.0 if (initial_assignment[i] == g and initial_assignment[j] == g) else 0.0
                var.Start = val
        
        model.optimize()
        
        if model.SolCount > 0:
            assignment = [0] * N
            for i in range(N):
                for g in range(m):
                    if x[i, g].X > 0.5:
                        assignment[i] = g
                        break
            return assignment, model.ObjVal
        
    except Exception as e:
        pass
    
    return None, None


def main():
    args = parse_args()
    start_time = time.time()
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    # Load instance
    data = load_instance(args.instance_path)
    N = data['N']
    m = data['m']
    L = [gc['L_g'] for gc in data['group_capacities']]
    U = [gc['U_g'] for gc in data['group_capacities']]
    distances = data['distances_upper_triangular']
    
    # Build distance matrix
    dist = build_distance_matrix(N, distances)
    
    rng = np.random.RandomState(42)
    
    best_assignment = None
    best_obj = -float('inf')
    
    # Phase 1: Multiple greedy constructions + local search
    num_restarts = min(5, max(1, 30 // max(1, N // 50)))
    
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > args.time_limit * 0.5:
            break
        
        if restart == 0:
            assignment = greedy_construction(N, m, dist, L, U, rng)
        else:
            assignment = generate_feasible_assignment(N, m, L, U, dist, rng)
        
        obj = compute_objective(assignment, dist, N, m)
        
        if obj > best_obj:
            best_obj = obj
            best_assignment = assignment[:]
            if logger:
                solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
                logger.log_solution(best_obj, solution_dict)
        
        # Local search with swaps
        elapsed = time.time() - start_time
        if elapsed < args.time_limit * 0.6:
            assignment, obj = local_search(assignment, dist, N, m, L, U,
                                           args.time_limit, start_time, logger if restart == 0 else None)
            
            if obj > best_obj:
                best_obj = obj
                best_assignment = assignment[:]
                if logger:
                    solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
                    logger.log_solution(best_obj, solution_dict)
        
        # Local search with moves
        elapsed = time.time() - start_time
        if elapsed < args.time_limit * 0.65:
            assignment, obj = local_search_move(assignment, dist, N, m, L, U,
                                                args.time_limit, start_time, None)
            
            if obj > best_obj:
                best_obj = obj
                best_assignment = assignment[:]
                if logger:
                    solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
                    logger.log_solution(best_obj, solution_dict)
    
    # Phase 2: Try Gurobi for exact/near-exact solution
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed - 2
    
    if remaining_time > 3:
        # Choose formulation based on problem size
        num_y_vars = N * (N - 1) // 2 * m
        
        if N <= 150 and num_y_vars <= 500000:
            # Use quadratic formulation for smaller instances
            gurobi_assignment, gurobi_obj = solve_with_gurobi(
                N, m, dist, L, U, remaining_time, best_assignment)
        elif N <= 100:
            # Use linearized formulation
            gurobi_assignment, gurobi_obj = solve_with_gurobi_linearized(
                N, m, dist, L, U, remaining_time, best_assignment)
        else:
            gurobi_assignment, gurobi_obj = None, None
        
        if gurobi_assignment is not None and gurobi_obj is not None:
            # Verify the Gurobi solution
            gurobi_obj_verify = compute_objective(gurobi_assignment, dist, N, m)
            if gurobi_obj_verify > best_obj:
                best_obj = gurobi_obj_verify
                best_assignment = gurobi_assignment
                if logger:
                    solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
                    logger.log_solution(best_obj, solution_dict)
    
    # Phase 3: Final local search on best solution if time permits
    elapsed = time.time() - start_time
    if elapsed < args.time_limit * 0.9 and best_assignment is not None:
        assignment, obj = local_search(best_assignment, dist, N, m, L, U,
                                       args.time_limit, start_time, None)
        if obj > best_obj:
            best_obj = obj
            best_assignment = assignment
            if logger:
                solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
                logger.log_solution(best_obj, solution_dict)
    
    # Write final solution
    if best_assignment is None:
        # Fallback: generate any feasible solution
        best_assignment = generate_feasible_assignment(N, m, L, U, dist, rng)
        best_obj = compute_objective(best_assignment, dist, N, m)
    
    solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)
    
    if logger:
        logger.log_solution(best_obj, solution_dict)


if __name__ == '__main__':
    main()