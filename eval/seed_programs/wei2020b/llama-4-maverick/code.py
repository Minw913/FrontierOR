import json
import argparse
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def first_fit_decreasing(instance):
    n = instance['n']
    W = instance['W']
    V = instance['V']
    items = sorted(instance['items'], key=lambda x: x['w'] + x['v'], reverse=True)

    bins = []
    for item in items:
        assigned = False
        for bin in bins:
            if bin['total_weight'] + item['w'] <= W and bin['total_volume'] + item['v'] <= V:
                bin['items'].append(item)
                bin['total_weight'] += item['w']
                bin['total_volume'] += item['v']
                assigned = True
                break
        if not assigned:
            bins.append({
                'items': [item],
                'total_weight': item['w'],
                'total_volume': item['v']
            })

    solution = {
        'objective_value': len(bins),
        'bins': [{'items': [i for i, item in enumerate(instance['items']) if item in bin['items']], 
                  'total_weight': bin['total_weight'], 
                  'total_volume': bin['total_volume']} for bin in bins]
    }
    return solution

def write_solution(solution_path, solution):
    with open(solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = first_fit_decreasing(instance)
    if logger:
        logger.log(solution['objective_value'])
    write_solution(args.solution_path, solution)

if __name__ == "__main__":
    main()