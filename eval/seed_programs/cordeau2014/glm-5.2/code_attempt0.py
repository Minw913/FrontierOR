import json
import argparse
import random
import time
from solution_logger import SolutionLogger

def solve(instance, logger, time_limit):
    distances = instance['arcs']['distances']
    speeds = instance['arcs']['speeds_v_ijh']
    boundaries = instance['parameters']['period_boundaries']
    H = len(boundaries)
    
    def calc_t(i, j, start_t):
        d = distances[i][j]
        if d == 0: return 0.0
        t = start_t
        if t >= boundaries[-1]:
            h = H - 1
        else:
            h = 0
            while h < H - 1 and boundaries[h+1] <= t:
                h += 1
        while d > 1e-9:
            if h >= H:
                v = speeds[i][j][H-1]
                if v <= 1e-9: return float('inf')
                t += d / v
                break
            v = speeds[i][j][h]
            if v <= 1e-9: return float('inf')
            t_end = boundaries[h+1]
            time_to_end = t_end - t
            dist_to_end = v * time_to_end
            if d <= dist_to_end:
                t += d / v
                d = 0
            else:
                d -= dist_to_end
                t = t_end
                h += 1
        return t - start_t

    def eval_tour(tour, best_t=float('inf')):
        t = 0.0
        for k in range(len(tour) - 1):
            t += calc_t(tour[k], tour[k+1], t)
            if t >= best_t:
                return float('inf')
        return t

    n = instance['parameters']['n_customers']
    tour = [0]
    visited = [False] * (n + 1)
    visited[0] = True
    curr = 0
    t_curr = 0.0
    for _ in range(n):
        best_j = -1
        best_arr = float('inf')
        for j in range(1, n + 1):
            if not visited[j]:
                arr = t_curr + calc_t(curr, j, t_curr)
                if arr < best_arr:
                    best_arr = arr
                    best_j = j
        tour.append(best_j)
        visited[best_j] = True
        curr = best_j
        t_curr = best_arr
    tour.append(0)
    
    best_tour = list(tour)
    best_t = eval_tour(best_tour)
    overall_best_tour = list(best_tour)
    overall_best_t = best_t
    
    if logger:
        logger.log_solution(overall_best_t, {"objective_value": overall_best_t, "tour": overall_best_tour})
        
    def two_opt(tour, best_t):
        n = len(tour) - 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                t = eval_tour(new_tour, best_t)
                if t < best_t - 1e-5:
                    return new_tour, t
        return tour, best_t

    def or_opt(tour, best_t):
        n = len(tour) - 1
        for seg_len in [1, 2, 3]:
            for i in range(1, max(1, n - seg_len + 1)):
                seg = tour[i : i + seg_len]
                temp_tour = tour[:i] + tour[i+seg_len:]
                for j in range(1, len(temp_tour)):
                    if j == i: continue
                    new_tour = temp_tour[:j] + seg + temp_tour[j:]
                    t = eval_tour(new_tour, best_t)
                    if t < best_t - 1e-5:
                        return new_tour, t
        return tour, best_t

    start_time = time.time()
    while time.time() - start_time < time_limit:
        improved = False
        new_tour, new_t = two_opt(best_tour, best_t)
        if new_t < best_t - 1e-5:
            best_tour = new_tour
            best_t = new_t
            improved = True
            
        if not improved:
            new_tour, new_t = or_opt(best_tour, best_t)
            if new_t < best_t - 1e-5:
                best_tour = new_tour
                best_t = new_t
                improved = True
                
        if improved:
            if best_t < overall_best_t - 1e-5:
                overall_best_tour = list(best_tour)
                overall_best_t = best_t
                if logger:
                    logger.log_solution(overall_best_t, {"objective_value": overall_best_t, "tour": overall_best_tour})
            continue
            
        # Perturbation
        perturbed_tour = list(overall_best_tour)
        n = len(perturbed_tour) - 1
        if n >= 8:
            a = 1 + random.randint(1, max(1, n // 4))
            b = a + 1 + random.randint(1, max(1, n // 4))
            c = b + 1 + random.randint(1, max(1, n // 4))
            if c < n:
                perturbed_tour = perturbed_tour[:a] + perturbed_tour[c:n] + perturbed_tour[b:c] + perturbed_tour[a:b] + [0]
            else:
                i = random.randint(1, n - 1)
                j = random.randint(i + 1, n)
                perturbed_tour = perturbed_tour[:i] + perturbed_tour[i:j+1][::-1] + perturbed_tour[j+1:]
        else:
            if n > 1:
                i = random.randint(1, n - 1)
                j = random.randint(i + 1, n)
                perturbed_tour = perturbed_tour[:i] + perturbed_tour[i:j+1][::-1] + perturbed_tour[j+1:]
            
        t = eval_tour(perturbed_tour)
        if t < float('inf'):
            best_tour = perturbed_tour
            best_t = t

    return overall_best_t, overall_best_tour

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
        
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    obj_val, tour = solve(instance, logger, args.time_limit)
    
    solution = {
        "objective_value": obj_val,
        "tour": tour
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)
        
if __name__ == '__main__':
    main()