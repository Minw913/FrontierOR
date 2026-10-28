import time
import argparse
import json
from collections import deque
import sys

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

class MWC_Solver:
    def __init__(self, adj_dict, weight_dict, time_limit, start_time, global_best, logger=None):
        self.adj_dict = adj_dict
        self.weight_dict = weight_dict
        self.best_clique = []
        self.best_weight = 0
        self.time_limit = time_limit
        self.start_time = start_time
        self.global_best = global_best
        self.logger = logger

    def compute_bound(self, candidate_set):
        if not candidate_set:
            return 0
        candidate_list = sorted(candidate_set, key=lambda x: self.weight_dict[x], reverse=True)
        max_weights = []
        color_assignment = {}

        for v in candidate_list:
            neighbor_colors = set()
            for u in self.adj_dict[v]:
                if u in candidate_set and u in color_assignment:
                    neighbor_colors.add(color_assignment[u])
            color_index = 0
            while color_index < len(max_weights):
                if color_index not in neighbor_colors:
                    break
                color_index += 1
            if color_index == len(max_weights):
                max_weights.append(self.weight_dict[v])
            else:
                if self.weight_dict[v] > max_weights[color_index]:
                    max_weights[color_index] = self.weight_dict[v]
            color_assignment[v] = color_index

        return sum(max_weights)

    def solve_iterative(self, initial_candidates):
        stack = deque()
        stack.append(([], 0, initial_candidates))
        best_clique = []
        best_weight = 0

        while stack:
            if time.time() - self.start_time > self.time_limit:
                break

            current_clique, current_weight, candidate_set = stack.pop()

            if current_weight > best_weight:
                best_clique = current_clique
                best_weight = current_weight
                if best_weight > self.global_best['weight']:
                    self.global_best['weight'] = best_weight
                    self.global_best['clique'] = current_clique
                    if self.logger:
                        self.logger.log(best_weight)

            if not candidate_set:
                continue

            bound = current_weight + self.compute_bound(candidate_set)
            if bound <= self.global_best['weight']:
                continue

            v = max(candidate_set, key=lambda x: self.weight_dict[x])
            candidate_set_without_v = candidate_set - {v}

            stack.append((current_clique, current_weight, candidate_set_without_v))

            new_candidate_set = candidate_set & self.adj_dict[v]
            new_clique = current_clique + [v]
            new_weight = current_weight + self.weight_dict[v]
            stack.append((new_clique, new_weight, new_candidate_set))

        self.best_clique = best_clique
        self.best_weight = best_weight

def get_connected_components(vertices, adj_dict):
    visited = set()
    components = []
    for vertex in vertices:
        if vertex not in visited:
            comp = []
            stack = [vertex]
            visited.add(vertex)
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in adj_dict[v]:
                    if u not in visited:
                        visited.add(u)
                        stack.append(u)
            components.append(comp)
    return components

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")

    try:
        with open(args.instance_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading instance file: {e}")
        return

    graph = data['graph']
    n = graph['num_vertices']
    vertices_list = graph['vertices']
    edges_list = graph['edges']
    weights_list = data['vertex_weights']

    weight_dict = {}
    for i, vertex in enumerate(vertices_list):
        weight_dict[vertex] = weights_list[i]

    adj_dict = {vertex: set() for vertex in vertices_list}
    for edge in edges_list:
        u, v = edge
        adj_dict[u].add(v)
        adj_dict[v].add(u)

    components = get_connected_components(vertices_list, adj_dict)
    
    global_best = {'clique': [], 'weight': 0}
    
    for comp in components:
        if time.time() - start_time > args.time_limit:
            break
        solver = MWC_Solver(adj_dict, weight_dict, args.time_limit, start_time, global_best, logger)
        solver.solve_iterative(set(comp))

    solution = {
        "objective_value": float(global_best['weight']),
        "clique_vertices": global_best['clique']
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()