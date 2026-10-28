#!/usr/bin/env python3
import argparse
import json
import time
import itertools
import random
import sys
import os

# Suppress any D-Bus or GUI related warnings in containerized environments
os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", "disabled:")


def solve_tsp(instance, time_limit, logger):
    start_time = time.time()
    n = instance["parameters"]["n_cities"]
    dist = instance["data"]["distance_matrix"]
    
    if n <= 10:
        best_cost = float('inf')
        best_tour = None
        
        cities = list(range(n))
        remaining = cities[1:]
        
        for perm in itertools.permutations(remaining):
            if time.time() - start_time > time_limit - 1:
                break
            tour = [0] + list(perm)
            cost = sum(dist[tour[i]][tour[i+1]] for i in range(n-1)) + dist[tour[-1]][tour[0]]
            if cost < best_cost:
                best_cost = cost
                best_tour = tour[:]
                
                solution_dict = {}
                for city in range(n):
                    for t in range(n):
                        solution_dict["x_{}_{}".format(city, t)] = 0
                for t, city in enumerate(best_tour):
                    solution_dict["x_{}_{}".format(city, t)] = 1
                
                if logger:
                    logger.log_solution(best_cost, solution_dict)
    else:
        best_cost = float('inf')
        best_tour = None
        
        def tour_cost(tour):
            return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))
        
        # Nearest neighbor from each start
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
                    d1 = dist[best_tour[i-1]][best_tour[i]] + dist[best_tour[j]][best_tour[(j+1) % n]]
                    d2 = dist[best_tour[i-1]][best_tour[j]] + dist[best_tour[i]][best_tour[(j+1) % n]]
                    if d2 < d1 - 1e-10:
                        best_tour[i:j+1] = reversed(best_tour[i:j+1])
                        best_cost = tour_cost(best_tour)
                        improved = True
                        
                        if logger:
                            solution_dict = {}
                            for city in range(n):
                                for t in range(n):
                                    solution_dict["x_{}_{}".format(city, t)] = 0
                            for t, city in enumerate(best_tour):
                                solution_dict["x_{}_{}".format(city, t)] = 1
                            logger.log_solution(best_cost, solution_dict)
    
    solution = {}
    for city in range(n):
        for t in range(n):
            solution["x_{}_{}".format(city, t)] = 0
    for t, city in enumerate(best_tour):
        solution["x_{}_{}".format(city, t)] = 1
    
    return {
        "objective_value": best_cost,
        "solution": solution,
        "tour": best_tour
    }


def solve_maxcut(instance, time_limit, logger):
    random.seed(0)
    start_time = time.time()
    
    params = instance.get("parameters", {})
    data = instance.get("data", {})
    
    # Determine number of nodes
    n_nodes = 0
    for key in ["n_nodes", "num_nodes", "n", "N", "num_vertices", "n_vertices"]:
        if key in params:
            n_nodes = int(params[key])
            break
    
    # Build edge list from available data
    edge_list = []
    
    if "adjacency_matrix" in data:
        mat = data["adjacency_matrix"]
        if n_nodes == 0:
            n_nodes = len(mat)
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                w = mat[i][j]
                if w != 0:
                    edge_list.append((i, j, float(w)))
    elif "edges" in data:
        edges_raw = data["edges"]
        max_node = 0
        for e in edges_raw:
            u, v = int(e[0]), int(e[1])
            max_node = max(max_node, u, v)
            w = float(e[2]) if len(e) > 2 else 1.0
            edge_list.append((u, v, w))
        if n_nodes == 0:
            n_nodes = max_node + 1
    elif "graph" in data:
        graph = data["graph"]
        if "adjacency_matrix" in graph:
            mat = graph["adjacency_matrix"]
            if n_nodes == 0:
                n_nodes = len(mat)
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    w = mat[i][j]
                    if w != 0:
                        edge_list.append((i, j, float(w)))
        elif "edges" in graph:
            edges_raw = graph["edges"]
            max_node = 0
            for e in edges_raw:
                u, v = int(e[0]), int(e[1])
                max_node = max(max_node, u, v)
                w = float(e[2]) if len(e) > 2 else 1.0
                edge_list.append((u, v, w))
            if n_nodes == 0:
                n_nodes = max_node + 1
    
    # If we still don't have n_nodes, try to infer from parameters
    if n_nodes == 0:
        # Try all params for anything that looks like node count
        for key, val in params.items():
            if isinstance(val, int) and val > 0:
                n_nodes = val
                break
    
    # If no edges found, try to look in other data structures
    if not edge_list and n_nodes == 0:
        # Fallback: check if there's a QUBO or similar
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0:
                if isinstance(val[0], list):
                    n_nodes = len(val)
                    for i in range(n_nodes):
                        for j in range(i+1, n_nodes):
                            if val[i][j] != 0:
                                edge_list.append((i, j, float(val[i][j])))
                    break
    
    if n_nodes == 0:
        n_nodes = 1
    
    # Build adjacency list
    adj = [[] for _ in range(n_nodes)]
    for u, v, w in edge_list:
        if u < n_nodes and v < n_nodes:
            adj[u].append((v, w))
            adj[v].append((u, w))
    
    def compute_cut(part):
        total = 0.0
        for u, v, w in edge_list:
            if part[u] != part[v]:
                total += w
        return total
    
    def build_solution(part):
        sol = {}
        for i in range(n_nodes):
            sol["x_{}".format(i)] = part[i]
        return sol
    
    def compute_gain(partition, node):
        gain = 0.0
        for neighbor, w in adj[node]:
            if partition[node] == partition[neighbor]:
                gain += w
            else:
                gain -= w
        return gain
    
    def local_search(partition, cut_value):
        improved = True
        while improved and time.time() - start_time < time_limit - 1:
            improved = False
            for node in range(n_nodes):
                if time.time() - start_time > time_limit - 1:
                    break
                gain = compute_gain(partition, node)
                if gain > 1e-10:
                    partition[node] = 1 - partition[node]
                    cut_value += gain
                    improved = True
        return partition, cut_value
    
    best_cut = -1.0
    best_partition = [0] * n_nodes
    
    restart = 0
    while time.time() - start_time < time_limit - 2:
        if restart == 0:
            # Greedy construction
            partition = [0] * n_nodes
            for i in range(n_nodes):
                gain0 = sum(w for nb, w in adj[i] if partition[nb] == 1)
                gain1 = sum(w for nb, w in adj[i] if partition[nb] == 0)
                partition[i] = 1 if gain1 > gain0 else 0
        else:
            partition = [random.randint(0, 1) for _ in range(n_nodes)]
        
        cut_value = compute_cut(partition)
        partition, cut_value = local_search(partition, cut_value)
        
        if cut_value > best_cut:
            best_cut = cut_value
            best_partition = partition[:]
            if logger:
                logger.log_solution(best_cut, build_solution(best_partition))
        
        restart += 1
        if time.time() - start_time > time_limit - 2:
            break
    
    return {
        "objective_value": best_cut,
        "solution": build_solution(best_partition)
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
    
    params = instance.get("parameters", {})
    is_tsp = "n_cities" in params
    
    logger = None
    try:
        from solution_logger import SolutionLogger
        if args.log_path:
            if is_tsp:
                logger = SolutionLogger(args.log_path, sense="minimize")
            else:
                logger = SolutionLogger(args.log_path, sense="maximize")
    except Exception:
        logger = None
    
    if is_tsp:
        result = solve_tsp(instance, args.time_limit, logger)
    else:
        result = solve_maxcut(instance, args.time_limit, logger)
    
    with open(args.solution_path, 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()