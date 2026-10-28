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
        if 0 <= assignment[i] < m:
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
    assignment_dict = {str(i): int(assignment[i]) for i in range(N)}
    return {
        "objective_value": float(obj),
        "assignment": assignment_dict,
        "groups": groups
    }


def is_feasible(assignment, N, m, L, U):
    group_sizes = [0] * m
    for i in range(N):
        if assignment[i] < 0 or assignment[i] >= m:
            return False
        group_sizes[assignment[i]] += 1
    for g in range(m):
        if group_sizes[g] < L[g] or group_sizes[g] > U[g]:
            return False
    return True


def generate_feasible_assignment(N, m, L, U, rng):
    """Generate a feasible assignment respecting capacity constraints."""
    assignment = [-1] * N
    group_sizes = [0] * m
    
    elements = list(range(N))
    rng.shuffle(elements)
    
    idx = 0
    # First satisfy lower bounds
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
    
    total_dist = np.sum(dist, axis=1)
    order = list(np.argsort(-total_dist))
    
    remaining = set(order)
    
    # First pass: satisfy lower bounds
    for g in range(m):
        needed = L[g]
        while len(group_members[g]) < needed and remaining:
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
                remaining.discard(best_elem)
    
    # Second pass: assign remaining elements greedily
    remaining_list = list(remaining)
    rng.shuffle(remaining_list)
    
    for elem in remaining_list:
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
            for g in range(m):
                if len(group_members[g]) < U[g]:
                    best_group = g
                    break
        
        if best_group >= 0:
            assignment[elem] = best_group
            group_members[best_group].append(elem)
    
    return assignment


def local_search_swap(assignment, dist, N, m, L, U, time_limit, start_time, logger=None):
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
            s = 0.0
            for k in group_members[g]:
                s += dist[i][k]
            S[i][g] = s
    
    max_passes = 100
    for pass_num in range(max_passes):
        if time.time() - start_time > time_limit * 0.7:
            break
        
        improved = False
        
        for i in range(N):
            gi = assignment[i]
            for j in range(i + 1, N):
                gj = assignment[j]
                if gi == gj:
                    continue
                
                # Delta for swapping i and j
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
                    for k in range(N):
                        S[k][gi] += -dist[k][i] + dist[k][j]
                        S[k][gj] += -dist[k][j] + dist[k][i]
                    
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
            s = 0.0
            for k in group_members[g]:
                s += dist[i][k]
            S[i][g] = s
    
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
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        model = gp.Model("max_diversity")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, int(time_remaining)))
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
    """Solve using linearized Gurobi MIP formulation."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        model = gp.Model("max_diversity_lin")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, int(time_remaining)))
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
        
        try:
            if restart == 0:
                assignment = greedy_construction(N, m, dist, L, U, rng)
            else:
                assignment = generate_feasible_assignment(N, m, L, U, rng)
            
            # Validate assignment
            if not is_feasible(assignment, N, m, L, U):
                assignment = generate_feasible_assignment(N, m, L, U, rng)
                if not is_feasible(assignment, N, m, L, U):
                    continue
            
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
                assignment_ls, obj_ls = local_search_swap(
                    assignment[:], dist, N, m, L, U,
                    args.time_limit, start_time,
                    logger if restart == 0 else None)
                
                if obj_ls > best_obj:
                    best_obj = obj_ls
                    best_assignment = assignment_ls[:]
                    if logger:
                        solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
                        logger.log_solution(best_obj, solution_dict)
            
            # Local search with moves
            elapsed = time.time() - start_time
            if elapsed < args.time_limit * 0.65:
                assignment_ls, obj_ls = local_search_move(
                    assignment[:], dist, N, m, L, U,
                    args.time_limit, start_time, None)
                
                if obj_ls > best_obj:
                    best_obj = obj_ls
                    best_assignment = assignment_ls[:]
                    if logger:
                        solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
                        logger.log_solution(best_obj, solution_dict)
        except Exception as e:
            continue
    
    # Phase 2: Try Gurobi for exact/near-exact solution
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed - 2
    
    if remaining_time > 3 and best_assignment is not None:
        try:
            num_y_vars = N * (N - 1) // 2 * m
            
            if N <= 150 and num_y_vars <= 500000:
                gurobi_assignment, gurobi_obj = solve_with_gurobi(
                    N, m, dist, L, U, remaining_time, best_assignment)
            elif N <= 100:
                gurobi_assignment, gurobi_obj = solve_with_gurobi_linearized(
                    N, m, dist, L, U, remaining_time, best_assignment)
            else:
                gurobi_assignment, gurobi_obj = None, None
            
            if gurobi_assignment is not None and gurobi_obj is not None:
                if is_feasible(gurobi_assignment, N, m, L, U):
                    gurobi_obj_verify = compute_objective(gurobi_assignment, dist, N, m)
                    if gurobi_obj_verify > best_obj:
                        best_obj = gurobi_obj_verify
                        best_assignment = gurobi_assignment
                        if logger:
                            solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
                            logger.log_solution(best_obj, solution_dict)
        except Exception as e:
            pass
    
    # Phase 3: Final local search on best solution if time permits
    elapsed = time.time() - start_time
    if elapsed < args.time_limit * 0.9 and best_assignment is not None:
        try:
            assignment_ls, obj_ls = local_search_swap(
                best_assignment[:], dist, N, m, L, U,
                args.time_limit, start_time, None)
            if obj_ls > best_obj:
                best_obj = obj_ls
                best_assignment = assignment_ls
                if logger:
                    solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
                    logger.log_solution(best_obj, solution_dict)
        except Exception as e:
            pass
    
    # Write final solution
    if best_assignment is None:
        best_assignment = generate_feasible_assignment(N, m, L, U, rng)
        best_obj = compute_objective(best_assignment, dist, N, m)
    
    solution_dict = build_solution_dict(best_assignment, best_obj, N, m)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)
    
    if logger:
        logger.log_solution(best_obj, solution_dict)


if __name__ == '__main__':
    main()