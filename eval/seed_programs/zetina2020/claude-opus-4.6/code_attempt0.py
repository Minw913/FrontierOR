import argparse
import json
import time
import itertools
import sys

def solve_tsp(instance, time_limit, logger):
    n = instance["parameters"]["n_cities"]
    dist = instance["data"]["distance_matrix"]
    
    if n <= 10:
        # For small instances, try all permutations (brute force)
        best_cost = float('inf')
        best_tour = None
        
        cities = list(range(n))
        # Fix city 0 at position 0 to reduce permutations
        remaining = cities[1:]
        
        start_time = time.time()
        for perm in itertools.permutations(remaining):
            if time.time() - start_time > time_limit - 1:
                break
            tour = [0] + list(perm)
            cost = sum(dist[tour[i]][tour[i+1]] for i in range(n-1)) + dist[tour[-1]][tour[0]]
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
                
                # Build solution dict
                solution_dict = {}
                for city in range(n):
                    for t in range(n):
                        solution_dict[f"x_{city}_{t}"] = 0
                for t, city in enumerate(best_tour):
                    solution_dict[f"x_{city}_{t}"] = 1
                
                if logger:
                    logger.log_solution(best_cost, solution_dict)
        
        # Build final solution
        solution = {}
        for city in range(n):
            for t in range(n):
                solution[f"x_{city}_{t}"] = 0
        for t, city in enumerate(best_tour):
            solution[f"x_{city}_{t}"] = 1
        
        return {
            "objective_value": best_cost,
            "solution": solution,
            "tour": best_tour
        }
    else:
        # For larger instances, use nearest neighbor heuristic + 2-opt
        start_time = time.time()
        
        def tour_cost(tour):
            return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))
        
        # Nearest neighbor from each starting city
        best_cost = float('inf')
        best_tour = None
        
        for start in range(n):
            if time.time() - start_time > time_limit * 0.3:
                break
            visited = [False] * n
            tour = [start]
            visited[start] = True
            for _ in range(n - 1):
                last = tour[-1]
                nearest = -1
                nearest_dist = float('inf')
                for j in range(n):
                    if not visited[j] and dist[last][j] < nearest_dist:
                        nearest = j
                        nearest_dist = dist[last][j]
                tour.append(nearest)
                visited[nearest] = True
            
            cost = tour_cost(tour)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour[:]
        
        # 2-opt improvement
        improved = True
        while improved and time.time() - start_time < time_limit - 1:
            improved = False
            for i in range(1, n - 1):
                if time.time() - start_time > time_limit - 1:
                    break
                for j in range(i + 1, n):
                    # Reverse segment [i, j]
                    d1 = dist[best_tour[i-1]][best_tour[i]] + dist[best_tour[j]][best_tour[(j+1) % n]]
                    d2 = dist[best_tour[i-1]][best_tour[j]] + dist[best_tour[i]][best_tour[(j+1) % n]]
                    if d2 < d1 - 1e-10:
                        best_tour[i:j+1] = reversed(best_tour[i:j+1])
                        best_cost = tour_cost(best_tour)
                        improved = True
                        
                        solution_dict = {}
                        for city in range(n):
                            for t in range(n):
                                solution_dict[f"x_{city}_{t}"] = 0
                        for t, city in enumerate(best_tour):
                            solution_dict[f"x_{city}_{t}"] = 1
                        if logger:
                            logger.log_solution(best_cost, solution_dict)
        
        solution = {}
        for city in range(n):
            for t in range(n):
                solution[f"x_{city}_{t}"] = 0
        for t, city in enumerate(best_tour):
            solution[f"x_{city}_{t}"] = 1
        
        return {
            "objective_value": best_cost,
            "solution": solution,
            "tour": best_tour
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    from solution_logger import SolutionLogger
    
    # Determine problem type
    if "n_cities" in instance.get("parameters", {}):
        problem_type = "TSP"
    else:
        problem_type = "MaxCut"
    
    if problem_type == "TSP":
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
        result = solve_tsp(instance, args.time_limit, logger)
    else:
        # MaxCut problem
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
        result = solve_maxcut(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(result, f)


def solve_maxcut(instance, time_limit, logger):
    import random
    random.seed(0)
    
    start_time = time.time()
    
    n_nodes = instance["parameters"]["n_nodes"]
    edges = instance["data"]["edges"]
    
    # Build adjacency list
    adj = [[] for _ in range(n_nodes)]
    for e in edges:
        u, v = e[0], e[1]
        w = e[2] if len(e) > 2 else 1
        adj[u].append((v, w))
        adj[v].append((u, w))
    
    # Initialize with random partition
    partition = [random.randint(0, 1) for _ in range(n_nodes)]
    
    def compute_cut():
        total = 0
        for e in edges:
            u, v = e[0], e[1]
            w = e[2] if len(e) > 2 else 1
            if partition[u] != partition[v]:
                total += w
        return total
    
    # Compute initial cut value and node contributions
    cut_value = compute_cut()
    best_cut = cut_value
    best_partition = partition[:]
    
    # Build solution dict
    def build_solution():
        sol = {}
        for i in range(n_nodes):
            sol[f"x_{i}"] = best_partition[i]
        return sol
    
    if logger:
        logger.log_solution(best_cut, build_solution())
    
    # Greedy local search with node flipping
    improved = True
    while improved and time.time() - start_time < time_limit - 1:
        improved = False
        for node in range(n_nodes):
            if time.time() - start_time > time_limit - 1:
                break
            # Compute gain of flipping this node
            gain = 0
            for neighbor, w in adj[node]:
                if partition[node] == partition[neighbor]:
                    gain += w  # This edge would become cut
                else:
                    gain -= w  # This edge would become uncut
            
            if gain > 0:
                partition[node] = 1 - partition[node]
                cut_value += gain
                improved = True
                
                if cut_value > best_cut:
                    best_cut = cut_value
                    best_partition = partition[:]
                    if logger:
                        logger.log_solution(best_cut, build_solution())
    
    # Simulated annealing / random restarts if time permits
    while time.time() - start_time < time_limit - 2:
        # Random restart
        partition = [random.randint(0, 1) for _ in range(n_nodes)]
        cut_value = compute_cut()
        
        # Local search
        improved = True
        while improved and time.time() - start_time < time_limit - 1:
            improved = False
            for node in range(n_nodes):
                if time.time() - start_time > time_limit - 1:
                    break
                gain = 0
                for neighbor, w in adj[node]:
                    if partition[node] == partition[neighbor]:
                        gain += w
                    else:
                        gain -= w
                if gain > 0:
                    partition[node] = 1 - partition[node]
                    cut_value += gain
                    improved = True
                    
                    if cut_value > best_cut:
                        best_cut = cut_value
                        best_partition = partition[:]
                        if logger:
                            logger.log_solution(best_cut, build_solution())
    
    solution = build_solution()
    
    return {
        "objective_value": best_cut,
        "solution": solution
    }


if __name__ == "__main__":
    main()