import argparse
import json
import time
import random
import sys

class LocalSearch:
    def __init__(self, n, m, capacity, weights, values, pw, time_limit, logger):
        self.n = n
        self.m = m
        self.capacity = capacity
        self.weights = weights
        self.values = values
        self.pw = pw
        self.time_limit = time_limit
        self.logger = logger
        self.start_time = time.time()
        
        self.assignment = [-1] * self.n
        self.knap_w = [0] * self.m
        self.knap_items = [set() for _ in range(self.m)]
        self.current_obj = 0
        
        self.best_assignment = {}
        self.best_obj = -float('inf')

    def update_best(self):
        if self.current_obj > self.best_obj:
            self.best_obj = self.current_obj
            self.best_assignment = {str(i): int(self.assignment[i]) for i in range(self.n) if self.assignment[i] != -1}
            if self.logger:
                self.logger.log_solution(float(self.best_obj), self.best_assignment)

    def build_greedy(self):
        self.assignment = [-1] * self.n
        self.knap_w = [0] * self.m
        self.knap_items = [set() for _ in range(self.m)]
        self.current_obj = 0
        
        order = sorted(range(self.n), key=lambda i: -(self.values[i] / max(self.weights[i], 1)))
        for i in order:
            best_k = -1
            best_gain = -float('inf')
            for k in range(self.m):
                if self.knap_w[k] + self.weights[i] <= self.capacity:
                    gain = self.values[i]
                    for j in self.knap_items[k]:
                        gain += self.pw[i][j]
                    if gain > best_gain:
                        best_gain = gain
                        best_k = k
            if best_k != -1 and best_gain > 0:
                self.assignment[i] = best_k
                self.knap_w[best_k] += self.weights[i]
                self.knap_items[best_k].add(i)
                self.current_obj += best_gain
        self.update_best()

    def local_search(self):
        improved = True
        while improved and time.time() - self.start_time < self.time_limit:
            improved = False
            
            # Add unassigned items
            for i in range(self.n):
                if time.time() - self.start_time >= self.time_limit: return
                if self.assignment[i] != -1: continue
                best_k = -1
                best_delta = 0.0
                for k in range(self.m):
                    if self.knap_w[k] + self.weights[i] > self.capacity: continue
                    delta = self.values[i]
                    for j in self.knap_items[k]:
                        delta += self.pw[i][j]
                    if delta > best_delta:
                        best_delta = delta
                        best_k = k
                if best_k != -1 and best_delta > 0:
                    self.knap_items[best_k].add(i)
                    self.knap_w[best_k] += self.weights[i]
                    self.assignment[i] = best_k
                    self.current_obj += best_delta
                    improved = True
                    self.update_best()
                    break
            if improved: continue

            # Move assigned items
            for i in range(self.n):
                if time.time() - self.start_time >= self.time_limit: return
                if self.assignment[i] == -1: continue
                k1 = self.assignment[i]
                best_k2 = -1
                best_delta = 1e-6
                for k2 in range(self.m):
                    if k2 == k1: continue
                    if self.knap_w[k2] + self.weights[i] > self.capacity: continue
                    delta = 0.0
                    for j in self.knap_items[k2]:
                        delta += self.pw[i][j]
                    for j in self.knap_items[k1]:
                        delta -= self.pw[i][j]
                    if delta > best_delta:
                        best_delta = delta
                        best_k2 = k2
                if best_k2 != -1:
                    self.knap_items[k1].remove(i)
                    self.knap_items[best_k2].add(i)
                    self.knap_w[k1] -= self.weights[i]
                    self.knap_w[best_k2] += self.weights[i]
                    self.assignment[i] = best_k2
                    self.current_obj += best_delta
                    improved = True
                    self.update_best()
                    break
            if improved: continue

            # Unassign
            for i in range(self.n):
                if time.time() - self.start_time >= self.time_limit: return
                if self.assignment[i] == -1: continue
                k1 = self.assignment[i]
                delta = -self.values[i]
                for j in self.knap_items[k1]:
                    delta -= self.pw[i][j]
                if delta > 1e-6:
                    self.knap_items[k1].remove(i)
                    self.knap_w[k1] -= self.weights[i]
                    self.assignment[i] = -1
                    self.current_obj += delta
                    improved = True
                    self.update_best()
                    break
            if improved: continue

            # Swap (only for N <= 150 to prevent TLE on large instances)
            if self.n <= 150:
                for i1 in range(self.n):
                    if time.time() - self.start_time >= self.time_limit: return
                    if self.assignment[i1] == -1: continue
                    k1 = self.assignment[i1]
                    for i2 in range(self.n):
                        if i1 >= i2: continue
                        if self.assignment[i2] == -1: continue
                        k2 = self.assignment[i2]
                        if k1 == k2: continue
                        
                        if self.knap_w[k1] - self.weights[i1] + self.weights[i2] > self.capacity: continue
                        if self.knap_w[k2] - self.weights[i2] + self.weights[i1] > self.capacity: continue
                        
                        delta = 0.0
                        for j in self.knap_items[k1]:
                            if j != i1: delta -= self.pw[i1][j]
                            if j != i2: delta += self.pw[i2][j]
                        for j in self.knap_items[k2]:
                            if j != i2: delta -= self.pw[i2][j]
                            if j != i1: delta += self.pw[i1][j]
                            
                        if delta > 1e-6:
                            self.knap_items[k1].remove(i1)
                            self.knap_items[k2].remove(i2)
                            self.knap_items[k1].add(i2)
                            self.knap_items[k2].add(i1)
                            self.knap_w[k1] = self.knap_w[k1] - self.weights[i1] + self.weights[i2]
                            self.knap_w[k2] = self.knap_w[k2] - self.weights[i2] + self.weights[i1]
                            self.assignment[i1] = k2
                            self.assignment[i2] = k1
                            self.current_obj += delta
                            improved = True
                            self.update_best()
                            break
                    if improved: break

    def run(self):
        self.build_greedy()
        while time.time() - self.start_time < self.time_limit:
            self.local_search()
            if time.time() - self.start_time >= self.time_limit:
                break
            # Perturbation
            assigned = [i for i in range(self.n) if self.assignment[i] != -1]
            if not assigned:
                self.build_greedy()
            else:
                drop_count = max(1, len(assigned) // 10)
                to_drop = random.sample(assigned, drop_count)
                for i in to_drop:
                    k = self.assignment[i]
                    delta = -self.values[i]
                    for j in self.knap_items[k]:
                        delta -= self.pw[i][j]
                    self.knap_items[k].remove(i)
                    self.knap_w[k] -= self.weights[i]
                    self.assignment[i] = -1
                    self.current_obj += delta
                    
        if self.best_obj == -float('inf'):
            self.best_obj = 0.0
            self.best_assignment = {}
        return self.best_obj, self.best_assignment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except Exception:
            logger = None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
    m = data['m']
    capacity = data['capacity']
    weights = data['weights']
    values = data['values']
    pw = data['pairwise_values']

    solver = LocalSearch(n, m, capacity, weights, values, pw, args.time_limit, logger)
    best_obj, best_assignment = solver.run()

    result = {
        "objective_value": float(best_obj),
        "assignment": best_assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    main()