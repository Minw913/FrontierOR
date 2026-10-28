import argparse
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
import sys

def fit_hyperplane(points_subset, d):
    """Fit a hyperplane to a set of points using PCA (find the direction of least variance)."""
    if len(points_subset) == 0:
        return None, None
    if len(points_subset) == 1:
        # Any hyperplane through this point works; pick arbitrary normal
        w = np.zeros(d)
        w[0] = 1.0
        w0 = np.dot(w, points_subset[0])
        return w, w0
    
    centroid = np.mean(points_subset, axis=0)
    
    if d == 1:
        w = np.array([1.0])
        w0 = centroid[0]
        return w, w0
    
    # Use PCA to find the direction of minimum variance
    centered = points_subset - centroid
    # Covariance matrix
    if len(points_subset) <= d:
        # Use SVD directly
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # The last component of Vt corresponds to smallest singular value
        w = Vt[-1]
    else:
        cov = np.cov(centered.T)
        if cov.ndim == 0:
            w = np.zeros(d)
            w[0] = 1.0
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # eigh returns eigenvalues in ascending order
            w = eigenvectors[:, 0]  # smallest eigenvalue direction
    
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-12:
        w = np.zeros(d)
        w[0] = 1.0
    else:
        w = w / w_norm
    
    w0 = np.dot(w, centroid)
    return w, w0

def point_to_hyperplane_distance(point, w, w0):
    """Compute distance from point to hyperplane w.x = w0."""
    return abs(np.dot(w, point) - w0)

def compute_all_distances(points, w, w0):
    """Compute distances from all points to a hyperplane."""
    return np.abs(points @ w - w0)

def greedy_set_cover(points, hyperplanes_list, epsilon, n):
    """Greedy set cover: repeatedly pick the hyperplane covering the most uncovered points."""
    # Precompute which points each hyperplane covers
    covers = []
    for w, w0 in hyperplanes_list:
        dists = compute_all_distances(points, w, w0)
        covered = set(np.where(dists <= epsilon)[0])
        covers.append(covered)
    
    uncovered = set(range(n))
    selected = []
    
    while uncovered:
        best_idx = -1
        best_count = -1
        for i, cov in enumerate(covers):
            count = len(cov & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_count <= 0:
            break
        
        selected.append(best_idx)
        uncovered -= covers[best_idx]
    
    return selected, uncovered

def solve(instance, time_limit, logger=None):
    start_time = time.time()
    
    n = instance['n']
    d = instance['d']
    epsilon = instance['epsilon']
    upper_bound_K = instance['upper_bound_K']
    points = np.array(instance['points'])
    
    best_solution = None
    best_obj = float('inf')
    
    def update_best(hyperplanes_used, assigned_points_list):
        nonlocal best_solution, best_obj
        obj = len(hyperplanes_used)
        if obj < best_obj:
            best_obj = obj
            best_solution = (hyperplanes_used, assigned_points_list)
            if logger:
                logger.log(obj)
    
    # Trivial solution: one hyperplane per point
    trivial_hyperplanes = []
    trivial_assignments = []
    for i in range(n):
        w = np.zeros(d)
        w[0] = 1.0
        w0 = np.dot(w, points[i])
        trivial_hyperplanes.append((w, w0, [i]))
        trivial_assignments.append([i])
    update_best(trivial_hyperplanes, trivial_assignments)
    
    # Check if all points can be covered by a single hyperplane
    if n <= d:
        # n points in d-dimensional space with n <= d can always be fit exactly
        w, w0 = fit_hyperplane(points, d)
        if w is not None:
            dists = compute_all_distances(points, w, w0)
            if np.all(dists <= epsilon + 1e-10):
                hp = [(w, w0, list(range(n)))]
                assignments = [[0] for _ in range(n)]
                update_best(hp, assignments)
                return format_solution(best_solution, d)
    
    if n > 1:
        w, w0 = fit_hyperplane(points, d)
        if w is not None:
            dists = compute_all_distances(points, w, w0)
            if np.all(dists <= epsilon + 1e-10):
                hp = [(w, w0, list(range(n)))]
                assignments = [[0] for _ in range(n)]
                update_best(hp, assignments)
                return format_solution(best_solution, d)
    
    # Generate candidate hyperplanes using various clustering approaches
    candidate_hyperplanes = []  # list of (w, w0)
    
    def add_candidate(w, w0):
        if w is not None and np.linalg.norm(w) > 1e-12:
            w = w / np.linalg.norm(w)
            candidate_hyperplanes.append((w, w0))
    
    # Strategy 1: Try fitting hyperplanes to clusters from KMeans with various k
    max_k = min(upper_bound_K, n)
    
    # Determine which k values to try
    k_values = set()
    for k in range(1, min(max_k + 1, 30)):
        k_values.add(k)
    # Also try some larger values
    for k in range(30, max_k + 1, max(1, (max_k - 30) // 20)):
        k_values.add(k)
    k_values.add(max_k)
    
    for k in sorted(k_values):
        if time.time() - start_time > time_limit * 0.5:
            break
        if k > n:
            continue
        
        try:
            if k == 1:
                labels = np.zeros(n, dtype=int)
            else:
                n_init = 3 if k < 10 else 1
                kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=100, random_state=42)
                labels = kmeans.fit_predict(points)
            
            hyperplanes_for_k = []
            for c in range(k):
                cluster_points = points[labels == c]
                if len(cluster_points) == 0:
                    continue
                w, w0 = fit_hyperplane(cluster_points, d)
                if w is not None:
                    add_candidate(w, w0)
                    hyperplanes_for_k.append((w, w0))
        except Exception:
            continue
    
    # Strategy 2: Random subsets of d points define a hyperplane exactly
    np.random.seed(42)
    n_random_samples = min(500, n * (n - 1) // 2 if d >= 2 else n)
    for _ in range(n_random_samples):
        if time.time() - start_time > time_limit * 0.6:
            break
        # Pick d random points (or fewer if n < d)
        num_sample = min(d, n)
        if num_sample < 2 and d > 1:
            num_sample = min(2, n)
        indices = np.random.choice(n, size=min(max(d, 2), n), replace=False)
        subset = points[indices]
        w, w0 = fit_hyperplane(subset, d)
        if w is not None:
            add_candidate(w, w0)
    
    # Strategy 3: For each pair of nearby points, fit a hyperplane
    if n <= 200:
        from scipy.spatial.distance import pdist, squareform
        dist_matrix = squareform(pdist(points))
        for i in range(n):
            if time.time() - start_time > time_limit * 0.65:
                break
            # Find points within 2*epsilon
            nearby = np.where(dist_matrix[i] <= 2 * epsilon + 1e-10)[0]
            if len(nearby) > 1:
                w, w0 = fit_hyperplane(points[nearby], d)
                if w is not None:
                    add_candidate(w, w0)
    
    # Strategy 4: For each point, create a hyperplane through it with various orientations
    # This ensures every point can be covered
    for i in range(n):
        for dim in range(d):
            w = np.zeros(d)
            w[dim] = 1.0
            w0 = points[i, dim]
            add_candidate(w, w0)
    
    if not candidate_hyperplanes:
        return format_solution(best_solution, d)
    
    # Remove duplicate candidates
    unique_candidates = []
    seen = set()
    for w, w0 in candidate_hyperplanes:
        # Normalize: ensure first nonzero component of w is positive
        w = np.array(w)
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-12:
            continue
        w = w / w_norm
        # Canonical form: first significant component positive
        for j in range(d):
            if abs(w[j]) > 1e-10:
                if w[j] < 0:
                    w = -w
                    w0 = -w0
                break
        key = tuple(np.round(w, 8)) + (round(w0, 8),)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((w, w0))
    
    candidate_hyperplanes = unique_candidates
    
    # Greedy set cover
    selected_indices, uncovered = greedy_set_cover(points, candidate_hyperplanes, epsilon, n)
    
    # Handle uncovered points: add individual hyperplanes
    extra_hyperplanes = []
    for pt_idx in uncovered:
        w = np.zeros(d)
        w[0] = 1.0
        w0 = points[pt_idx, 0]
        extra_hyperplanes.append((w, w0))
    
    # Combine selected and extra
    all_selected_hps = [candidate_hyperplanes[i] for i in selected_indices] + extra_hyperplanes
    
    # Build assignments
    hp_assigned = [[] for _ in range(len(all_selected_hps))]
    pt_assignments = [[] for _ in range(n)]
    
    for h_idx, (w, w0) in enumerate(all_selected_hps):
        dists = compute_all_distances(points, w, w0)
        for p_idx in range(n):
            if dists[p_idx] <= epsilon + 1e-10:
                hp_assigned[h_idx].append(p_idx)
                pt_assignments[p_idx].append(h_idx)
    
    # Verify all points are assigned
    for p_idx in range(n):
        if not pt_assignments[p_idx]:
            # Emergency: add a trivial hyperplane
            w = np.zeros(d)
            w[0] = 1.0
            w0 = points[p_idx, 0]
            h_idx = len(all_selected_hps)
            all_selected_hps.append((w, w0))
            hp_assigned.append([p_idx])
            pt_assignments[p_idx].append(h_idx)
    
    # Remove empty hyperplanes
    non_empty = [i for i in range(len(all_selected_hps)) if hp_assigned[i]]
    
    final_hps = []
    idx_map = {}
    for new_idx, old_idx in enumerate(non_empty):
        idx_map[old_idx] = new_idx
        w, w0 = all_selected_hps[old_idx]
        final_hps.append((w, w0, hp_assigned[old_idx]))
    
    final_pt_assignments = [[] for _ in range(n)]
    for p_idx in range(n):
        for old_h in pt_assignments[p_idx]:
            if old_h in idx_map:
                final_pt_assignments[p_idx].append(idx_map[old_h])
        if not final_pt_assignments[p_idx]:
            # Shouldn't happen but safety
            w = np.zeros(d)
            w[0] = 1.0
            w0 = points[p_idx, 0]
            h_idx = len(final_hps)
            final_hps.append((w, w0, [p_idx]))
            final_pt_assignments[p_idx].append(h_idx)
    
    update_best(final_hps, final_pt_assignments)
    
    # Try to improve: iterative refinement
    # For each hyperplane, refit using assigned points, then reassign
    if time.time() - start_time < time_limit * 0.8:
        improved = True
        iterations = 0
        current_hps = [(w.copy(), w0_, list(assigned)) for w, w0_, assigned in final_hps]
        
        while improved and iterations < 20 and time.time() - start_time < time_limit * 0.85:
            improved = False
            iterations += 1
            
            # Reassign points to closest hyperplane
            new_assignments = [[] for _ in range(len(current_hps))]
            new_pt_assignments = [[] for _ in range(n)]
            
            for p_idx in range(n):
                best_dist = float('inf')
                best_h = -1
                for h_idx, (w, w0_, _) in enumerate(current_hps):
                    dist = point_to_hyperplane_distance(points[p_idx], w, w0_)
                    if dist <= epsilon + 1e-10:
                        new_assignments[h_idx].append(p_idx)
                        new_pt_assignments[p_idx].append(h_idx)
                    if dist < best_dist:
                        best_dist = dist
                        best_h = h_idx
                
                if not new_pt_assignments[p_idx]:
                    new_assignments[best_h].append(p_idx)
                    new_pt_assignments[p_idx].append(best_h)
            
            # Remove empty hyperplanes
            non_empty_indices = [i for i in range(len(current_hps)) if new_assignments[i]]
            if len(non_empty_indices) < len(current_hps):
                improved = True
            
            # Refit hyperplanes
            new_hps = []
            remap = {}
            for new_i, old_i in enumerate(non_empty_indices):
                remap[old_i] = new_i
                assigned_pts = points[new_assignments[old_i]]
                w, w0_ = fit_hyperplane(assigned_pts, d)
                if w is None:
                    w, w0_ = current_hps[old_i][0], current_hps[old_i][1]
                new_hps.append((w, w0_, new_assignments[old_i]))
            
            current_hps = new_hps
            
            # Rebuild assignments after refit
            final_pt_assign = [[] for _ in range(n)]
            final_hp_assign = [list(hp[2]) for hp in current_hps]
            
            for h_idx, (w, w0_, assigned) in enumerate(current_hps):
                for p_idx in assigned:
                    dist = point_to_hyperplane_distance(points[p_idx], w, w0_)
                    if dist <= epsilon + 1e-10:
                        final_pt_assign[p_idx].append(h_idx)
            
            # Check unassigned
            all_covered = True
            for p_idx in range(n):
                if not final_pt_assign[p_idx]:
                    all_covered = False
                    break
            
            if all_covered:
                update_best(current_hps, final_pt_assign)
    
    # Try smaller k values more carefully with multiple random restarts
    if time.time() - start_time < time_limit * 0.9:
        for k in range(1, min(best_obj, upper_bound_K + 1)):
            if time.time() - start_time > time_limit * 0.95:
                break
            if k >= best_obj:
                break
            
            found = False
            n_restarts = min(20, max(1, int((time_limit * 0.95 - (time.time() - start_time)) / (k + 1))))
            
            for restart in range(n_restarts):
                if time.time() - start_time > time_limit * 0.95:
                    break
                
                if k == 1:
                    labels = np.zeros(n, dtype=int)
                else:
                    try:
                        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=300, random_state=42+ restart)
                        labels = kmeans.fit_predict(points)
                    except:
                        continue
                
                # Fit hyperplane per cluster
                hps = []
                for c in range(k):
                    mask = labels == c
                    if not np.any(mask):
                        continue
                    cluster_pts = points[mask]
                    w, w0_ = fit_hyperplane(cluster_pts, d)
                    if w is not None:
                        hps.append((w, w0_))
                
                if not hps:
                    continue
                
                # Iterative assignment and refit
                for iteration in range(30):
                    if time.time() - start_time > time_limit * 0.95:
                        break
                    
                    # Assign each point to closest hyperplane
                    new_labels = np.zeros(n, dtype=int)
                    all_within = True
                    for p_idx in range(n):
                        best_dist = float('inf')
                        best_h = 0
                        for h_idx, (w, w0_) in enumerate(hps):
                            dist = point_to_hyperplane_distance(points[p_idx], w, w0_)
                            if dist < best_dist:
                                best_dist = dist
                                best_h = h_idx
                        new_labels[p_idx] = best_h
                        if best_dist > epsilon + 1e-10:
                            all_within = False
                    
                    # Refit
                    new_hps = []
                    for h_idx in range(len(hps)):
                        mask = new_labels == h_idx
                        if not np.any(mask):
                            # Keep old
                            new_hps.append(hps[h_idx])
                        else:
                            w, w0_ = fit_hyperplane(points[mask], d)
                            if w is not None:
                                new_hps.append((w, w0_))
                            else:
                                new_hps.append(hps[h_idx])
                    
                    # Check convergence
                    converged = True
                    for h_idx in range(len(hps)):
                        if np.linalg.norm(hps[h_idx][0] - new_hps[h_idx][0]) > 1e-8:
                            converged = False
                            break
                        if abs(hps[h_idx][1] - new_hps[h_idx][1]) > 1e-8:
                            converged = False
                            break
                    
                    hps = new_hps
                    
                    if all_within:
                        # Check if this is valid
                        hp_list = []
                        pt_assign = [[] for _ in range(n)]
                        hp_assign_list = [[] for _ in range(len(hps))]
                        
                        for p_idx in range(n):
                            for h_idx, (w, w0_) in enumerate(hps):
                                dist = point_to_hyperplane_distance(points[p_idx], w, w0_)
                                if dist <= epsilon + 1e-10:
                                    pt_assign[p_idx].append(h_idx)
                                    hp_assign_list[h_idx].append(p_idx)
                        
                        valid = all(len(pa) > 0 for pa in pt_assign)
                        if valid:
                            # Remove empty hyperplanes
                            non_empty_h = [i for i in range(len(hps)) if hp_assign_list[i]]
                            if len(non_empty_h) < best_obj:
                                new_hp_list = []
                                remap = {}
                                for new_i, old_i in enumerate(non_empty_h):
                                    remap[old_i] = new_i
                                    w, w0_ = hps[old_i]
                                    new_hp_list.append((w, w0_, hp_assign_list[old_i]))
                                
                                new_pt_assign = [[] for _ in range(n)]
                                for p_idx in range(n):
                                    for old_h in pt_assign[p_idx]:
                                        if old_h in remap:
                                            new_pt_assign[p_idx].append(remap[old_h])
                                
                                update_best(new_hp_list, new_pt_assign)
                                found = True
                        break
                    
                    if converged:
                        break
                
                if found:
                    break
    
    return format_solution(best_solution, d)

def format_solution(best_solution, d):
    if best_solution is None:
        return {"objective_value": 0, "hyperplanes": [], "point_assignments": []}
    
    hps, pt_assignments = best_solution
    
    hyperplanes = []
    for hp in hps:
        w, w0_, assigned = hp[0], hp[1], hp[2]
        w_list = w.tolist() if isinstance(w, np.ndarray) else list(w)
        hyperplanes.append({
            "w": w_list,
            "w0": float(w0_),
            "assigned_points": sorted(set(assigned))
        })
    
    return {
        "objective_value": len(hyperplanes),
        "hyperplanes": hyperplanes,
        "point_assignments": [list(pa) for pa in pt_assignments]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, default=300)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    solution = solve(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()