import argparse
import json
import random
import time
from solution_logger import SolutionLogger

def compute_travel_time(i, j, departure_time, instance):
    d = instance['arcs']['distances'][i][j]
    if d == 0:
        return 0.0
        
    boundaries = instance['parameters']['period_boundaries']
    H_val = instance['parameters']['H']
    speeds = instance['arcs']['speeds_v_ijh'][i][j]
    
    t_current = departure_time
    remaining = d
    total_time = 0.0

    while remaining > 0:
        if t_current >= boundaries[-1]:
            h = H_val - 1
            time_left_in_period = float('inf')
        else:
            h = 0
            while h < H_val:
                if t_current >= boundaries[h] and t_current < boundaries[h+1]:
                    break
                h += 1
            else:
                h = H_val - 1
            if h < H_val - 1:
                time_left_in_period = boundaries[h+1] - t_current
            else:
                time_left_in_period = float('inf')
        
        speed = speeds[h]
        distance_possible = speed * time_left_in_period
        
        if distance_possible >= remaining:
            time_segment = remaining / speed
            total_time += time_segment
            remaining = 0
        else:
            total_time += time_left_in_period
            remaining -= distance_possible
            t_current = boundaries[h+1] if h < H_val - 1 else boundaries[H_val]
            
    return total_time

def evaluate_tour(tour, instance):
    current_time = 0.0
    for k in range(len(tour) - 1):
        i = tour[k]
        j = tour[k+1]
        tt = compute_travel_time(i, j, current_time, instance)
        current_time += tt
    return current_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n_customers = instance['parameters']['n_customers']
    total_vertices = instance['parameters']['total_vertices']
    H = instance['parameters']['H']
    boundaries = instance['parameters']['period_boundaries']
    b_h = instance['parameters']['b_h']
    distances = instance['arcs']['distances']
    max_speeds = instance['arcs']['max_speeds_u_ij']
    speeds_v_ijh = instance['arcs']['speeds_v_ijh']
    delta_ijh = instance['arcs']['delta_ijh']

    best_speeds = [[0.0] * total_vertices for _ in range(total_vertices)]
    for i in range(total_vertices):
        for j in range(total_vertices):
            best_speed_ij = 0.0
            for h in range(H):
                speed_ijh = speeds_v_ijh[i][j][h]
                if speed_ijh > best_speed_ij:
                    best_speed_ij = speed_ijh
            best_speeds[i][j] = best_speed_ij

    best_travel_time = [[0.0] * total_vertices for _ in range(total_vertices)]
    for i in range(total_vertices):
        for j in range(total_vertices):
            if best_speeds[i][j] > 1e-7:
                best_travel_time[i][j] = distances[i][j] / best_speeds[i][j]
            else:
                best_travel_time[i][j] = float('inf')

    unvisited = set(range(1, total_vertices))
    current = 0
    tour = [0]
    while unvisited:
        next_node = min(unvisited, key=lambda j: best_travel_time[current][j])
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    tour.append(0)

    best_duration = evaluate_tour(tour, instance)
    current_tour = tour
    current_duration = best_duration

    if logger:
        logger.log(best_duration)

    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < args.time_limit:
        improved = False

        non_depot = current_tour[1:-1]
        n = len(non_depot)
        indices = list(range(n))
        random.shuffle(indices)
        for i_idx in indices:
            if time.time() - start_time >= args.time_limit:
                break
            node = non_depot[i_idx]
            j_indices = list(range(n))
            random.shuffle(j_indices)
            for j in j_indices:
                if j == i_idx:
                    continue
                new_non_depot = non_depot[:i_idx] + non_depot[i_idx+1:]
                new_non_depot.insert(j, node)
                new_tour = [current_tour[0]] + new_non_depot + [current_tour[-1]]
                duration = evaluate_tour(new_tour, instance)
                if duration < current_duration - 1e-5:
                    current_tour = new_tour
                    current_duration = duration
                    improved = True
                    if logger:
                        logger.log(current_duration)
                    break
            if improved:
                break

        if improved:
            continue

        non_depot = current_tour[1:-1]
        n = len(non_depot)
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        for (i, j) in pairs:
            if time.time() - start_time >= args.time_limit:
                break
            new_non_depot = non_depot[:]
            new_non_depot[i], new_non_depot[j] = new_non_depot[j], new_non_depot[i]
            new_tour = [current_tour[0]] + new_non_depot + [current_tour[-1]]
            duration = evaluate_tour(new_tour, instance)
            if duration < current_duration - 1e-5:
                current_tour = new_tour
                current_duration = duration
                improved = True
                if logger:
                    logger.log(current_duration)
                break

    solution = {
        "objective_value": current_duration,
        "tour": current_tour
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()