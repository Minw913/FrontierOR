import argparse
import json
import time
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass

def get_normal(pts, d):
    if len(pts) <= 1:
        w = np.zeros(d)
        w[0] = 1.0
        return w
    pts_centered = pts - pts.mean(axis=0)
    try:
        _, _, Vh = np.linalg.svd(pts_centered, full_matrices=True)
        return Vh[-1]
    except np.linalg.LinAlgError:
        w = np.zeros(d)
        w[0] = 1.0
        return w

def generate_k_subspaces_normals(points, d, target_k):
    n = len(points)
    if target_k >= n:
        return [get_normal(points[[i]], d) for i in range(n)]
    assignments = np.random.randint(0, target_k, size=n)
    pool = []
    for k in range(target_k):
        pts = points[assignments == k]
        if len(pts) > 0:
            pool.append(get_normal(pts, d))
    return pool

def get_initial_w_pool(points, d, n):
    pool = []
    pool.append(get_normal(points, d))
    
    for i in range(d):
        w = np.zeros(d)
        w[i] = 1.0
        pool.append(w)
        
    import math
    if d <= n and d <= 5 and math.comb(n, d) <= 5000:
        import itertools
        for idx in itertools.combinations(range(n), d):
            w = get_normal(points[list(idx)], d)
            pool.append(w)
    else:
        for _ in range(10000):
            if n >= d:
                idx = np.random.choice(n, d, replace=False)
                w = get_normal(points[idx], d)
                pool.append(w)
            
    for _ in range(10000):
        high = min(n, max(2*d, 20))
        m = np.random.randint(min(d+1, n), high + 1) if n > d else n
        idx = np.random.choice(n, m, replace=False)
        w = get_normal(points[idx], d)
        pool.append(w)
        
    for k_guess in [2, 3, 5, 8]:
        for _ in range(200):
            pool.extend(generate_k_subspaces_normals(points, d, k_guess))
            
    for _ in range(5000):
        w = np.random.randn(d)
        w /= np.linalg.norm(w)
        pool.append(w)
        
    return pool

def get_iteration_w_pool(chosen_subsets, points, d, n, best_k):
    pool = []
    num_chosen = len(chosen_subsets)
    if num_chosen > 1:
        for _ in range(3000):
            i, j = np.random.choice(num_chosen, 2, replace=False)
            pts_combined = np.array([points[idx] for idx in (chosen_subsets[i] | chosen_subsets[j])])
            w = get_normal(pts_combined, d)
            pool.append(w)
            
        for _ in range(3000):
            if num_chosen >= 3:
                i, j, k = np.random.choice(num_chosen, 3, replace=False)
                pts_combined = np.array([points[idx] for idx in (chosen_subsets[i] | chosen_subsets[j] | chosen_subsets[k])])
                w = get_normal(pts_combined, d)
                pool.append(w)
                
    for s in chosen_subsets:
        pts = np.array([points[idx] for idx in s])
        base_w = get_normal(pts, d)
        for _ in range(20):
            noise = np.random.randn(d) * 0.1
            w = base_w + noise
            w /= np.linalg.norm(w)
            pool.append(w)
            
        s_list = list(s)
        if len(s_list) > d:
            for _ in range(20):
                sample_idx = np.random.choice(s_list, max(d, len(s_list)//2), replace=False)
                w = get_normal(points[sample_idx], d)
                pool.append(w)
                
    if best_k > 1 and best_k != float('inf'):
        for _ in range(50):
            pool.extend(generate_k_subspaces_normals(points, d, best_k - 1))
            
    for _ in range(1000):
        w = np.random.randn(d)
        w /= np.linalg.norm(w)
        pool.append(w)
        
    return pool

def solve_set_cover(subsets_list, n, time_limit, last_sol_indices):
    m = len(subsets_list)
    row_idx = []
    col_idx = []
    val = []
    
    for j, s in enumerate(subsets_list):
        for i in s:
            row_idx.append(i)
            col_idx.append(j)
            val.append(1)
            
    A = sp.csr_matrix((val, (row_idx, col_idx)), shape=(n, m))
    
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model("SetCover", env=env)
    
    x = model.addMVar(m, vtype=GRB.BINARY)
    model.addConstr(A @ x >= 1)
    
    if last_sol_indices is not None:
        start_vals = np.zeros(m)
        start_vals[last_sol_indices] = 1.0
        x.Start = start_vals
        
    model.setObjective(x.sum(), GRB.MINIMIZE)
    
    if time_limit > 0:
        model.setParam('TimeLimit', time_limit)
        
    model.setParam('MIPGap', 1e-4)
    model.setParam('Threads', 1)
    model.optimize()
    
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        if model.SolCount > 0:
            sol = x.X
            chosen_indices = [j for j, val in enumerate(sol) if val > 0.5]
            return len(chosen_indices), chosen_indices
    return float('inf'), []

def dump_solution(best_k, best_hyperplanes, best_point_assignments, path):
    if best_k == float('inf'):
        return
    out = {
        "objective_value": float(best_k),
        "hyperplanes": best_hyperplanes,
        "point_assignments": best_point_assignments
    }
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance
    with open(args.instance_path, 'r') as f:
        args_data = json.load(f)
        
    d = args_data['d']
    n = args_data['n']
    epsilon = args_data['epsilon']
    points = np.array(args_data['points'])

    global_subsets_list = []
    global_subsets_set = set()
    cluster_params = {}

    best_k = n
    best_hyperplanes = []
    best_point_assignments = [[i] for i in range(n)]

    for i in range(n):
        w = np.zeros(d)
        w[0] = 1.0
        best_hyperplanes.append({
            "w": w.tolist(),
            "w0": float(points[i][0]),
            "assigned_points": [i]
        })
        cluster = frozenset([i])
        if cluster not in global_subsets_set:
            global_subsets_set.add(cluster)
            global_subsets_list.append(cluster)
            cluster_params[cluster] = (w, points[i][0])

    if logger:
        logger.log(best_k)

    last_sol_indices = None

    try:
        while time.time() - start_time < args.time_limit - 1.0:
            old_m = len(global_subsets_list)
            
            if last_sol_indices is not None and len(last_sol_indices) > 0:
                chosen_subsets = [global_subsets_list[i] for i in last_sol_indices]
                pool = get_iteration_w_pool(chosen_subsets, points, d, n, best_k)
            else:
                pool = get_initial_w_pool(points, d, n)
                
            for w in pool:
                w_norm = np.linalg.norm(w)
                if w_norm < 1e-9:
                    continue
                w = w / w_norm
                
                v = points @ w
                idx = np.argsort(v)
                sorted_v = v[idx]
                
                left = 0
                right = 0
                while left < n:
                    while right + 1 < n and sorted_v[right + 1] - sorted_v[left] <= 2 * epsilon + 1e-9:
                        right += 1
                    if right >= left:
                        cluster = frozenset(idx[left:right+1])
                        if cluster not in global_subsets_set:
                            global_subsets_set.add(cluster)
                            global_subsets_list.append(cluster)
                            w0 = (sorted_v[left] + sorted_v[right]) / 2.0
                            cluster_params[cluster] = (w, w0)
                    left += 1

            remaining_time = args.time_limit - (time.time() - start_time) - 1.0
            if remaining_time <= 0:
                break
                
            time_for_gurobi = min(remaining_time, 15 if last_sol_indices is not None else 30)
            
            k, chosen = solve_set_cover(global_subsets_list, n, time_for_gurobi, last_sol_indices)
            
            if k < float('inf') and len(chosen) > 0:
                last_sol_indices = chosen
                if k < best_k:
                    best_k = k
                    best_hyperplanes = []
                    best_point_assignments = [[] for _ in range(n)]
                    
                    for idx_c in chosen:
                        subset = global_subsets_list[idx_c]
                        w_c, w0_c = cluster_params[subset]
                        hk_index = len(best_hyperplanes)
                        best_hyperplanes.append({
                            "w": [float(x) for x in w_c],
                            "w0": float(w0_c),
                            "assigned_points": [int(x) for x in subset]
                        })
                        for pt in subset:
                            best_point_assignments[pt].append(hk_index)
                            
                    if logger:
                        logger.log(best_k)
                        
                    if best_k == 1:
                        break
                        
        dump_solution(best_k, best_hyperplanes, best_point_assignments, args.solution_path)

    except Exception as e:
        dump_solution(best_k, best_hyperplanes, best_point_assignments, args.solution_path)
        raise e

if __name__ == "__main__":
    main()