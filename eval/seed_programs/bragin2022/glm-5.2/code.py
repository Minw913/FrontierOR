import argparse
import json
import random
import time
import math
from solution_logger import SolutionLogger

def solve(instance, time_limit, log_path):
    num_machines = instance["num_machines"]
    num_jobs = instance["num_jobs"]
    cost_matrix = instance["cost_matrix"]
    resource_matrix = instance["resource_matrix"]
    capacities = instance["capacities"]

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    start_time = time.time()
    
    max_cost = 0
    for m in range(num_machines):
        for j in range(num_jobs):
            if cost_matrix[m][j] > max_cost:
                max_cost = cost_matrix[m][j]
                
    penalty = max_cost * 10 + 1
    
    def calc_cost(assign):
        cost = 0
        for j in range(num_jobs):
            cost += cost_matrix[assign[j]][j]
        return cost
        
    def calc_violation(assign):
        usage = [0] * num_machines
        for j in range(num_jobs):
            usage[assign[j]] += resource_matrix[assign[j]][j]
        viol = 0
        for m in range(num_machines):
            if usage[m] > capacities[m]:
                viol += usage[m] - capacities[m]
        return viol, usage

    # Greedy assignment for initial solution
    assign = [0] * num_jobs
    usage = [0] * num_machines
    for j in range(num_jobs):
        best_m = 0
        min_cost = float('inf')
        for m in range(num_machines):
            if cost_matrix[m][j] < min_cost:
                min_cost = cost_matrix[m][j]
                best_m = m
        assign[j] = best_m
        usage[best_m] += resource_matrix[best_m][j]

    cost = calc_cost(assign)
    viol, usage = calc_violation(assign)
    energy = cost + penalty * viol
    
    best_assign = list(assign)
    best_cost = cost
    best_viol = viol
    best_energy = energy
    
    if viol == 0:
        if logger:
            logger.log_solution(float(best_cost), {
                "objective_value": float(best_cost),
                "assignments": {str(j): best_assign[j] for j in range(num_jobs)}
            })

    T = max_cost * 10 + 1
    T_min = 1e-3
    alpha = 0.999
    
    while time.time() - start_time < time_limit:
        if random.random() < 0.8:
            # Move operator
            j = random.randint(0, num_jobs - 1)
            old_m = assign[j]
            new_m = random.randint(0, num_machines - 1)
            
            if new_m == old_m:
                continue
                
            delta_cost = cost_matrix[new_m][j] - cost_matrix[old_m][j]
            
            delta_viol = 0
            if usage[old_m] > capacities[old_m]:
                delta_viol -= min(usage[old_m] - capacities[old_m], resource_matrix[old_m][j])
            if usage[new_m] + resource_matrix[new_m][j] > capacities[new_m]:
                delta_viol += (usage[new_m] + resource_matrix[new_m][j] - capacities[new_m])
                
            delta_energy = delta_cost + penalty * delta_viol
            
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                assign[j] = new_m
                cost += delta_cost
                viol += delta_viol
                usage[old_m] -= resource_matrix[old_m][j]
                usage[new_m] += resource_matrix[new_m][j]
                energy = cost + penalty * viol
                
                if viol == 0 and (best_viol > 0 or cost < best_cost):
                    best_assign = list(assign)
                    best_cost = cost
                    best_viol = 0
                    best_energy = energy
                    if logger:
                        logger.log_solution(float(best_cost), {
                            "objective_value": float(best_cost),
                            "assignments": {str(j): best_assign[j] for j in range(num_jobs)}
                        })
                elif viol < best_viol:
                    best_assign = list(assign)
                    best_cost = cost
                    best_viol = viol
                    best_energy = energy
        else:
            # Swap operator
            j1 = random.randint(0, num_jobs - 1)
            j2 = random.randint(0, num_jobs - 1)
            if j1 == j2:
                continue
            m1 = assign[j1]
            m2 = assign[j2]
            if m1 == m2:
                continue
                
            delta_cost = (cost_matrix[m2][j1] - cost_matrix[m1][j1]) + (cost_matrix[m1][j2] - cost_matrix[m2][j2])
            
            new_usage = list(usage)
            new_usage[m1] -= resource_matrix[m1][j1]
            new_usage[m1] += resource_matrix[m1][j2]
            new_usage[m2] -= resource_matrix[m2][j2]
            new_usage[m2] += resource_matrix[m2][j1]
            
            new_viol = 0
            for m in range(num_machines):
                if new_usage[m] > capacities[m]:
                    new_viol += new_usage[m] - capacities[m]
            delta_viol = new_viol - viol
            
            delta_energy = delta_cost + penalty * delta_viol
            
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                assign[j1], assign[j2] = assign[j2], assign[j1]
                cost += delta_cost
                viol = new_viol
                usage = new_usage
                energy = cost + penalty * viol
                
                if viol == 0 and (best_viol > 0 or cost < best_cost):
                    best_assign = list(assign)
                    best_cost = cost
                    best_viol = 0
                    best_energy = energy
                    if logger:
                        logger.log_solution(float(best_cost), {
                            "objective_value": float(best_cost),
                            "assignments": {str(j): best_assign[j] for j in range(num_jobs)}
                        })
                elif viol < best_viol:
                    best_assign = list(assign)
                    best_cost = cost
                    best_viol = viol
                    best_energy = energy

        T = max(T * alpha, T_min)
        if T == T_min:
            T = max_cost * 10 + 1  # Reheat

    return {
        "objective_value": float(best_cost),
        "assignments": {str(j): best_assign[j] for j in range(num_jobs)}
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    solution = solve(instance, args.time_limit, args.log_path)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)