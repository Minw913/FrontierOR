import argparse
import json
import sys
import time

class SolutionLogger:
    def __init__(self, log_path, sense):
        self.log_path = log_path
        self.sense = sense
        self.best_objective = float('-inf') if sense == 'maximize' else float('inf')

    def log(self, objective_value):
        if (self.sense == 'maximize' and objective_value > self.best_objective) or (self.sense == 'minimize' and objective_value < self.best_objective):
            self.best_objective = objective_value
            with open(self.log_path, 'a') as f:
                f.write(json.dumps({'objective_value': objective_value}) + '\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    num_vertices = instance['num_vertices']
    vertices = instance['vertices']
    edges = instance['edges']
    edge_dict = {(edge['i'], edge['j']): edge['weight'] for edge in edges}

    best_clique = []
    best_objective = 0

    start_time = time.time()

    # Brute force approach to find the maximum weighted clique
    for mask in range(1 << num_vertices):
        clique = [vertices[i] for i in range(num_vertices) if (mask & (1 << i))]
        is_valid_clique = True
        objective = 0

        for v1 in clique:
            for v2 in clique:
                if v1 < v2 and (v1, v2) not in edge_dict and (v2, v1) not in edge_dict:
                    is_valid_clique = False
                    break
            if not is_valid_clique:
                break

        if is_valid_clique:
            for v1 in clique:
                for v2 in clique:
                    if v1 < v2 and (v1, v2) in edge_dict:
                        objective += edge_dict[(v1, v2)]
            if objective > best_objective:
                best_objective = objective
                best_clique = clique
                if logger:
                    logger.log(best_objective)

        if time.time() - start_time > time_limit:
            break

    return best_objective, best_clique

def write_solution(solution_path, objective_value, clique):
    solution = {'objective_value': objective_value, 'clique': clique}
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    args = parse_args()
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense='maximize') if args.log_path else None
    try:
        objective_value, clique = solve(instance, args.time_limit, logger)
        write_solution(args.solution_path, objective_value, clique)
    except Exception as e:
        print(f'An error occurred: {e}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()