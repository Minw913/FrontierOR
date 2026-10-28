import argparse
import json
import math
import random
import time
import sys
from solution_logger import SolutionLogger

try:
    import numpy as np
    use_numpy = True
except ImportError:
    use_numpy = False

def dot(a, b):
    return sum(a_i * b_i for a_i, b_i in zip(a, b))

def norm(a):
    return math.sqrt(sum(x*x for x in a))

def normalize(a):
    nrm = norm(a)
    if nrm < 1e-10:
        return [random.gauss(0, 1) for _ in range(len(a))]
    return [x / nrm for x in a]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    best_total_so_far = float('inf')

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']
    d = instance['d']
    epsilon = instance['epsilon']
    points = instance['points']
    
    start_time = time.time()
    uncovered = set(range(n))
    hyperplanes = []  # each element: (w, w0, cover_set)

    def check_time():
        return time.time() - start_time < args.time_limit - 1.0

    while uncovered and check_time():
        uncovered_list = list(uncovered)
        num_uncovered = len(uncovered_list)
        if num_uncovered == 0:
            break

        if num_uncovered == 1:
            i0 = uncovered_list[0]
            w = [random.gauss(0, 1) for _ in range(d)]
            w = normalize(w)
            w0 = -dot(w, points[i0])
            cover_set = {i0}
            hyperplanes.append((w, w0, cover_set))
            uncovered -= cover_set
            current_total = len(hyperplanes) + len(uncovered)
            if current_total < best_total_so_far:
                best_total_so_far = current_total
                if logger:
                    logger.log(current_total)
            continue

        candidates = []
        if use_numpy and num_uncovered >= 2:
            X_arr = np.array([points[i] for i in uncovered_list])
            center = np.mean(X_arr, axis=0)
            X_centered = X_arr - center
            if num_uncovered == 2:
                v = X_centered[0]
                norm_v = np.linalg.norm(v)
                if norm_v < 1e-10:
                    w_rand = np.random.randn(d)
                    w_rand = w_rand / np.linalg.norm(w_rand)
                    candidates.append(w_rand)
                else:
                    w_cand = v / norm_v
                    candidates.append(w_cand)
            else:
                if not np.allclose(X_centered, 0):
                    cov = np.cov(X_centered, rowvar=False)
                    if not np.allclose(cov, 0):
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        w_cand = eigvecs[:, 0]
                        w_cand = w_cand / np.linalg.norm(w_cand)
                        candidates.append(w_cand)
        
        num_rand = 100 - len(candidates)
        for _ in range(num_rand):
            w_rand = [random.gauss(0, 1) for _ in range(d)]
            w_rand = normalize(w_rand)
            candidates.append(w_rand)
        
        best_cover_count = 0
        best_w = None
        best_w0 = None
        best_cover_set = None

        for w_cand in candidates:
            if not check_time():
                break
            
            a_list = [dot(w_cand, points[i]) for i in uncovered_list]
            events = []
            for idx, a_val in enumerate(a_list):
                low = -a_val - epsilon
                high = -a_val + epsilon
                events.append((low, 1))
                events.append((high, -1))
            
            events.sort(key=lambda x: x[0])
            current_count = 0
            max_count = 0
            best_event_val = 0.0
            for event in events:
                current_count += event[1]
                if current_count > max_count:
                    max_count = current_count
                    best_event_val = event[0]
            
            cover_set = set()
            for idx, a_val in enumerate(a_list):
                low = -a_val - epsilon
                high = -a_val + epsilon
                if low <= best_event_val <= high:
                    cover_set.add(uncovered_list[idx])
            
            cover_count = len(cover_set)
            if cover_count > best_cover_count:
                best_cover_count = cover_count
                best_w = w_cand
                best_w0 = best_event_val
                best_cover_set = cover_set
        
        if best_cover_count == 0:
            i0 = uncovered_list[0]
            w = [random.gauss(0, 1) for _ in range(d)]
            w = normalize(w)
            w0 = -dot(w, points[i0])
            best_cover_set = {i0}
            best_w = w
            best_w0 = w0
            best_cover_count = 1
        
        hyperplanes.append((best_w, best_w0, best_cover_set))
        uncovered -= best_cover_set
        current_total = len(hyperplanes) + len(uncovered)
        if current_total < best_total_so_far:
            best_total_so_far = current_total
            if logger:
                logger.log(current_total)
    
    if uncovered:
        for i in uncovered:
            w = [random.gauss(0, 1) for _ in range(d)]
            w = normalize(w)
            w0 = -dot(w, points[i])
            hyperplanes.append((w, w0, {i}))
    
    hyperplane_objects = []
    point_assignments = [[] for _ in range(n)]
    for idx_hp, (w, w0, cover_set) in enumerate(hyperplanes):
        hyperplane_objects.append({
            "w": w,
            "w0": w0,
            "assigned_points": sorted(cover_set)
        })
        for i in cover_set:
            point_assignments[i].append(idx_hp)
    
    solution = {
        "objective_value": len(hyperplanes),
        "hyperplanes": hyperplane_objects,
        "point_assignments": point_assignments
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
    
    current_total = len(hyperplanes)
    if current_total < best_total_so_far and logger:
        logger.log(current_total)

if __name__ == '__main__':
    main()