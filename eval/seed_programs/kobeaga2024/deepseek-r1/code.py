import argparse
import json
import time
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    num_nodes = data['num_nodes']
    distance_limitation_d0 = data['distance_limitation_d0']
    depot_vertex = data['depot_vertex']
    scores = data['scores']
    distance_matrix = data['distance_matrix']
    
    start_time = time.time()
    time_limit = args.time_limit

    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")
    else:
        logger = None

    current_tour = [depot_vertex, depot_vertex]
    current_distance = 0
    visited_set = set([depot_vertex])
    current_score = 0
    unvisited = set(range(num_nodes)) - visited_set

    if logger:
        logger.log(current_score)

    while time.time() - start_time < time_limit:
        improved_insertion = False
        best_ratio = -1
        best_candidate = None

        for v in list(unvisited):
            if time.time() - start_time >= time_limit:
                break
            for i in range(1, len(current_tour)):
                a = current_tour[i-1]
                b = current_tour[i]
                add_dist = distance_matrix[a][v] + distance_matrix[v][b] - distance_matrix[a][b]
                if add_dist < 0:
                    add_dist = 0
                new_total_dist = current_distance + add_dist
                if new_total_dist <= distance_limitation_d0:
                    if add_dist == 0:
                        ratio = float('inf')
                    else:
                        ratio = scores[v] / add_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_candidate = (v, i, add_dist, scores[v])
        
        if best_ratio != -1:
            v, i, add_dist, score_inc = best_candidate
            current_tour.insert(i, v)
            current_distance += add_dist
            current_score += score_inc
            visited_set.add(v)
            unvisited.remove(v)
            if logger:
                logger.log(current_score)
            improved_insertion = True
            continue

        improved_swap = False
        for idx in range(1, len(current_tour)-1):
            if time.time() - start_time >= time_limit:
                break
            v = current_tour[idx]
            a = current_tour[idx-1]
            c = current_tour[idx+1]
            saved_dist = distance_matrix[a][v] + distance_matrix[v][c] - distance_matrix[a][c]
            saved_score = scores[v]
            new_tour = current_tour[:idx] + current_tour[idx+1:]
            new_dist = current_distance - saved_dist
            new_score = current_score - saved_score
            for u in list(unvisited):
                if time.time() - start_time >= time_limit:
                    break
                best_j = None
                best_add_dist = float('inf')
                for j in range(1, len(new_tour)):
                    x = new_tour[j-1]
                    y = new_tour[j]
                    add_dist_swap = distance_matrix[x][u] + distance_matrix[u][y] - distance_matrix[x][y]
                    if add_dist_swap < best_add_dist:
                        best_add_dist = add_dist_swap
                        best_j = j
                total_dist = new_dist + best_add_dist
                total_score = new_score + scores[u]
                if total_dist <= distance_limitation_d0 and total_score > current_score:
                    current_tour = new_tour
                    current_tour.insert(best_j, u)
                    current_distance = total_dist
                    current_score = total_score
                    visited_set.remove(v)
                    visited_set.add(u)
                    unvisited.remove(u)
                    unvisited.add(v)
                    if logger:
                        logger.log(current_score)
                    improved_swap = True
                    break
                if time.time() - start_time >= time_limit:
                    break
            if improved_swap:
                break
        if not improved_swap:
            break

    visited_vertices = sorted(list(visited_set))
    tour_edges = []
    for i in range(len(current_tour)-1):
        tour_edges.append([current_tour[i], current_tour[i+1]])
    
    solution = {
        "objective_value": float(current_score),
        "visited_vertices": visited_vertices,
        "tour_edges": tour_edges
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()