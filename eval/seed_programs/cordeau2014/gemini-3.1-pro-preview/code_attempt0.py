import argparse
import json
import math
import random
import time

def traverse_edge(d, u, v, t, p, H, speeds, boundaries):
    if d == 0:
        return t, p
        
    while True:
        sp = speeds[u][v][p]
        if sp <= 0: 
            sp = 1e-6
            
        if p < H - 1:
            rem_t = boundaries[p+1] - t
            if rem_t <= 0:
                p += 1
                t = boundaries[p]
                continue
                
            max_d = sp * rem_t
            if max_d >= d:
                t += d / sp
                break
            else:
                d -= max_d
                p += 1
                t = boundaries[p]
        else:
            t += d / sp
            break
            
    return t, p


def compute_TP(tour, distances, speeds, boundaries, H):
    T = [0.0] * len(tour)
    P = [0] * len(tour)
    
    t = 0.0
    p = 0
    for i in range(len(tour) - 1):
        u = tour[i]
        v = tour[i+1]
        d = distances[u][v]
        
        while p < H - 1 and t >= boundaries[p+1]:
            p += 1
            
        t, p = traverse_edge(d, u, v, t, p, H, speeds, boundaries)
        
        T[i+1] = t
        P[i+1] = p
        
    return T, P


def evaluate_neighbor_sa(new_tour, distances, speeds, boundaries, H, start_idx, t_start, p_start, current_obj, temp, rand_val):
    t = t_start
    p = p_start
    
    if rand_val > 0 and temp > 0:
        max_t = current_obj - temp * math.log(rand_val)
    else:
        max_t = current_obj
        
    for i in range(start_idx, len(new_tour) - 1):
        if t > max_t:
            return float('inf')
            
        u = new_tour[i]
        v = new_tour[i+1]
        d = distances[u][v]
        
        while p < H - 1 and t >= boundaries[p+1]:
            p += 1
            
        t, p = traverse_edge(d, u, v, t, p, H, speeds, boundaries)
        
    if t > max_t:
         return float('inf')
         
    return t


def get_greedy_tour(n, distances, speeds, boundaries, H, randomness=1):
    unvisited = set(range(1, n))
    tour = [0]
    t = 0.0
    p = 0
    
    while unvisited:
        candidates = []
        for v in unvisited:
            d = distances[tour[-1]][v]
            curr_p = p
            curr_t = t
            
            while curr_p < H - 1 and curr_t >= boundaries[curr_p+1]:
                curr_p += 1
                
            curr_t, curr_p = traverse_edge(d, tour[-1], v, curr_t, curr_p, H, speeds, boundaries)
            
            candidates.append((curr_t, v))
            
        candidates.sort(key=lambda x: x[0])
        limit = min(len(candidates), randomness)
        best_arr, best_v = candidates[random.randint(0, limit - 1)]
        
        tour.append(best_v)
        t = best_arr
        
        p = 0
        while p < H - 1 and t >= boundaries[p+1]:
            p += 1
            
    tour.append(0)
    return tour


def get_swap_tour(tour, i, j):
    nt = tour.copy()
    nt[i], nt[j] = nt[j], nt[i]
    return nt, i - 1

def get_relocate_tour(tour, i, j):
    nt = tour.copy()
    val = nt.pop(i)
    nt.insert(j, val)
    return nt, min(i, j) - 1

def get_2opt_tour(tour, i, j):
    nt = tour.copy()
    nt[i:j+1] = reversed(nt[i:j+1])
    return nt, i - 1

def generate_neighbor(tour):
    N = len(tour)
    if N <= 3:
        return tour.copy(), 0
    
    op = random.random()
    if op < 0.2:
        i, j = sorted(random.sample(range(1, N-1), 2))
        return get_swap_tour(tour, i, j)
    elif op < 0.5:
        i, j = random.sample(range(1, N-1), 2)
        return get_relocate_tour(tour, i, j)
    elif op < 0.8:
        i, j = sorted(random.sample(range(1, N-1), 2))
        return get_2opt_tour(tour, i, j)
    else:
        b_len = random.choice([2, 3])
        if N - 1 - b_len < 1:
            i, j = random.sample(range(1, N-1), 2)
            return get_relocate_tour(tour, i, j)
        
        i = random.randint(1, N - 1 - b_len)
        j = i + b_len - 1
        
        valid_k = [x for x in range(1, N-1) if x < i or x > j + 1]
        if not valid_k:
            i, j = random.sample(range(1, N-1), 2)
            return get_relocate_tour(tour, i, j)
        k = random.choice(valid_k)
        
        nt = tour.copy()
        block = nt[i:j+1]
        del nt[i:j+1]
        ins = k if k <= i else k - b_len
        nt[ins:ins] = block
        return nt, min(i, ins) - 1


def init_temp(tour, distances, speeds, boundaries, H):
    deltas = []
    base_obj = evaluate_neighbor_sa(tour, distances, speeds, boundaries, H, 0, 0.0, 0, float('inf'), 0.0, 0.0)
    for _ in range(50):
        nt, _ = generate_neighbor(tour)
        obj = evaluate_neighbor_sa(nt, distances, speeds, boundaries, H, 0, 0.0, 0, float('inf'), 0.0, 0.0)
        if obj > base_obj and obj < float('inf'):
            deltas.append(obj - base_obj)
    if not deltas:
        return 10.0
    return sum(deltas) / len(deltas) / 0.693


def simulated_annealing(distances, speeds, boundaries, H, time_limit, logger):
    start_time = time.time()
    n = len(distances)
    
    safe_time_limit = time_limit - 0.5
    if safe_time_limit <= 0:
        safe_time_limit = time_limit * 0.9
    if safe_time_limit <= 0:
        safe_time_limit = 1.0
        
    best_tour = get_greedy_tour(n, distances, speeds, boundaries, H, randomness=1)
    best_obj = evaluate_neighbor_sa(best_tour, distances, speeds, boundaries, H, 0, 0.0, 0, float('inf'), 0, 0)
    
    for _ in range(5):
        t_tour = get_greedy_tour(n, distances, speeds, boundaries, H, randomness=3)
        t_obj = evaluate_neighbor_sa(t_tour, distances, speeds, boundaries, H, 0, 0.0, 0, float('inf'), 0, 0)
        if t_obj < best_obj:
            best_obj = t_obj
            best_tour = t_tour
            
    if logger:
        logger.log(best_obj)
        
    current_tour = best_tour[:]
    current_obj = best_obj
    
    T_list, P_list = compute_TP(current_tour, distances, speeds, boundaries, H)
    
    T0 = init_temp(current_tour, distances, speeds, boundaries, H)
    Tend = 1e-4
    if T0 <= Tend:
        T0 = 10.0
        
    iter_count = 0
    check_interval = 100
    
    while True:
        if iter_count % check_interval == 0:
            elapsed = time.time() - start_time
            if elapsed >= safe_time_limit:
                break
                
            frac = elapsed / safe_time_limit
            frac = max(0.0, min(1.0, frac))
            temp = T0 * math.exp(frac * math.log(Tend / T0))
            
        nt, s_idx = generate_neighbor(current_tour)
        
        rand_val = max(1e-9, random.random())
        nt_obj = evaluate_neighbor_sa(nt, distances, speeds, boundaries, H, s_idx, T_list[s_idx], P_list[s_idx], current_obj, temp, rand_val)
        
        if nt_obj < float('inf'):
            current_tour = nt
            current_obj = nt_obj
            T_list, P_list = compute_TP(current_tour, distances, speeds, boundaries, H)
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_tour = current_tour[:]
                if logger:
                    logger.log(best_obj)
                    
        iter_count += 1
        
    return best_tour, best_obj


def main():
    parser = argparse.ArgumentParser(description="Solve Time-Dependent TSP.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where final output JSON must be written.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum valid runtime for completion.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to JSONL file for logging iterations.")
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    n = data['parameters']['total_vertices']
    H = data['parameters']['H']
    boundaries = data['parameters']['period_boundaries']
    distances = data['arcs']['distances']
    speeds = data['arcs']['speeds_v_ijh']
    
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
        
    best_tour, best_obj = simulated_annealing(distances, speeds, boundaries, H, args.time_limit, logger)
    
    solution = {
        "objective_value": best_obj,
        "tour": best_tour
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()