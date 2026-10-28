import json
import argparse
import time
import random
import sys
from solution_logger import SolutionLogger


def calculate_dist(tour, dist):
    d = 0
    for i in range(len(tour) - 1):
        d += dist[tour[i]][tour[i+1]]
    if len(tour) > 1:
        d += dist[tour[-1]][tour[0]]
    return d


def build_solution(tour, scores, depot):
    obj = sum(scores[i] for i in tour if i != depot)
    visited_vertices = list(tour)
    edges = []
    for i in range(len(tour) - 1):
        edges.append([min(tour[i], tour[i+1]), max(tour[i], tour[i+1])])
    if len(tour) > 1:
        edges.append([min(tour[0], tour[-1]), max(tour[0], tour[-1])])
    return {
        "objective_value": float(obj),
        "visited_vertices": visited_vertices,
        "tour_edges": edges
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_nodes']
    D = instance['distance_limitation_d0']
    depot = instance['depot_vertex']
    scores = instance['scores']
    dist = instance['distance_matrix']

    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")

    start_time = time.time()
    time_limit = args.time_limit

    if n == 1:
        sol = build_solution([depot], scores, depot)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        if logger:
            logger.log_solution(0.0, sol)
        return

    def get_best_insertion(tour, current_dist, visited):
        best_node = None
        best_pos = -1
        best_extra = float('inf')
        best_val = -1.0
        for i in range(n):
            if i in visited:
                continue
            for k in range(len(tour)):
                u = tour[k]
                v = tour[(k + 1) % len(tour)]
                extra = -dist[u][v] + dist[u][i] + dist[i][v]
                if current_dist + extra <= D:
                    val = scores[i] / (extra + 1e-6)
                    if val > best_val:
                        best_val = val
                        best_node = i
                        best_pos = k
                        best_extra = extra
        return best_node, best_pos, best_extra

    def two_opt(tour, current_dist):
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            L = len(tour)
            for i in range(L - 1):
                u1 = tour[i]
                u2 = tour[i+1]
                for j in range(i + 1, L):
                    if j == i: continue
                    v1 = tour[j]
                    v2 = tour[(j + 1) % L]
                    if i == 0 and j == L - 1: continue
                    delta = dist[u1][v1] + dist[u2][v2] - dist[u1][u2] - dist[v1][v2]
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_i != -1:
                tour[best_i+1:best_j+1] = reversed(tour[best_i+1:best_j+1])
                current_dist += best_delta
                improved = True
                if time.time() - start_time > time_limit:
                    break
        return tour, current_dist

    best_tour = [depot]
    best_score = 0
    best_sol = build_solution(best_tour, scores, depot)

    # Strategy 1: greedy score/distance
    tour = [depot]
    visited = {depot}
    current_dist = 0
    while time.time() - start_time < time_limit * 0.2:
        node, pos, extra = get_best_insertion(tour, current_dist, visited)
        if node is None:
            break
        tour.insert(pos + 1, node)
        current_dist += extra
        visited.add(node)
    score = sum(scores[i] for i in tour if i != depot)
    if score > best_score:
        best_score = score
        best_tour = tour
        best_sol = build_solution(best_tour, scores, depot)
        if logger:
            logger.log_solution(float(best_score), best_sol)

    # Strategy 2: highest absolute score
    if time.time() - start_time < time_limit * 0.2:
        tour = [depot]
        visited = {depot}
        current_dist = 0
        while True:
            best_node = None
            best_pos = -1
            best_extra = 0
            best_val = -1
            for i in range(n):
                if i in visited: continue
                for k in range(len(tour)):
                    u = tour[k]
                    v = tour[(k + 1) % len(tour)]
                    extra = -dist[u][v] + dist[u][i] + dist[i][v]
                    if current_dist + extra <= D:
                        if scores[i] > best_val:
                            best_val = scores[i]
                            best_node = i
                            best_pos = k
                            best_extra = extra
            if best_node is None:
                break
            tour.insert(best_pos + 1, best_node)
            current_dist += best_extra
            visited.add(best_node)
        score = sum(scores[i] for i in tour if i != depot)
        if score > best_score:
            best_score = score
            best_tour = tour
            best_sol = build_solution(best_tour, scores, depot)
            if logger:
                logger.log_solution(float(best_score), best_sol)

    # ILS / Simulated Annealing
    random.seed(0)
    curr_tour = best_tour[:]
    curr_dist = calculate_dist(curr_tour, dist)
    curr_score = best_score

    def local_search(tour, current_dist):
        tour, current_dist = two_opt(tour, current_dist)
        visited = set(tour)
        while time.time() - start_time < time_limit:
            node, pos, extra = get_best_insertion(tour, current_dist, visited)
            if node is None:
                break
            tour.insert(pos + 1, node)
            current_dist += extra
            visited.add(node)
            tour, current_dist = two_opt(tour, current_dist)
        return tour, current_dist

    curr_tour, curr_dist = local_search(curr_tour, curr_dist)
    curr_score = sum(scores[i] for i in curr_tour if i != depot)
    if curr_score > best_score:
        best_score = curr_score
        best_tour = curr_tour[:]
        best_sol = build_solution(best_tour, scores, depot)
        if logger:
            logger.log_solution(float(best_score), best_sol)

    while time.time() - start_time < time_limit:
        pert_tour = curr_tour[:]
        K = random.randint(1, max(1, len(pert_tour) // 4))
        for _ in range(K):
            if len(pert_tour) > 1:
                idx = random.randint(1, len(pert_tour) - 1)
                pert_tour.pop(idx)
            else:
                break
        
        pert_dist = calculate_dist(pert_tour, dist)
        pert_tour, pert_dist = local_search(pert_tour, pert_dist)
        pert_score = sum(scores[i] for i in pert_tour if i != depot)
        
        if pert_score > curr_score:
            curr_tour = pert_tour
            curr_dist = pert_dist
            curr_score = pert_score
            if curr_score > best_score:
                best_score = curr_score
                best_tour = curr_tour[:]
                best_sol = build_solution(best_tour, scores, depot)
                if logger:
                    logger.log_solution(float(best_score), best_sol)
        elif random.random() < 0.2:
            curr_tour = pert_tour
            curr_dist = pert_dist
            curr_score = pert_score

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()