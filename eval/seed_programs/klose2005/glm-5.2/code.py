import argparse
import json
import time
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    num_customers = instance["num_customers"]
    num_plants = instance["num_plants"]
    demands = instance["customers"]["demands"]
    fixed_costs = instance["plants"]["fixed_costs"]
    capacities = instance["plants"]["capacities"]
    shipping_costs = instance["shipping_costs"]

    total_demand = sum(demands)
    total_capacity = sum(capacities)

    # Precompute unit costs to avoid redundant divisions
    unit_costs = [[0.0] * num_plants for _ in range(num_customers)]
    for i in range(num_customers):
        if demands[i] > 0:
            for j in range(num_plants):
                unit_costs[i][j] = shipping_costs[i][j] / demands[i]

    def evaluate(y):
        open_plants = [j for j in range(num_plants) if y[j]]
        if not open_plants:
            return float('inf'), [], []
        
        total_cap = sum(capacities[j] for j in open_plants)
        if total_cap < total_demand - 1e-6:
            return float('inf'), [], []
        
        pairs = []
        for i in range(num_customers):
            if demands[i] == 0:
                continue
            for j in open_plants:
                pairs.append((unit_costs[i][j], i, j))
        pairs.sort(key=lambda x: x[0])
        
        assigned = [0.0] * num_customers
        rem_cap = [capacities[j] for j in range(num_plants)]
        assignments = [{} for _ in range(num_customers)]
        
        for cost_per_unit, i, j in pairs:
            if assigned[i] < demands[i] - 1e-9 and rem_cap[j] > 1e-9:
                amount = min(demands[i] - assigned[i], rem_cap[j])
                assigned[i] += amount
                rem_cap[j] -= amount
                assignments[i][j] = amount / demands[i]
                
        # Handle any remaining unassigned demand (feasibility repair)
        for i in range(num_customers):
            if demands[i] > 0 and abs(assigned[i] - demands[i]) > 1e-6:
                unassigned_amount = demands[i] - assigned[i]
                for j in open_plants:
                    if unassigned_amount <= 1e-9:
                        break
                    if rem_cap[j] > 1e-9:
                        amount = min(unassigned_amount, rem_cap[j])
                        assigned[i] += amount
                        rem_cap[j] -= amount
                        assignments[i][j] = assignments[i].get(j, 0.0) + amount / demands[i]
                        unassigned_amount -= amount
                if unassigned_amount > 1e-6:
                    return float('inf'), [], []
                
        total_cost = sum(fixed_costs[j] for j in open_plants)
        for i in range(num_customers):
            for j, frac in assignments[i].items():
                total_cost += frac * shipping_costs[i][j]
                
        return total_cost, open_plants, assignments

    random.seed(0)
    best_cost = float('inf')
    best_open = []
    best_assignments = []

    def log_solution():
        if logger:
            sol_assignments = {}
            for i in range(num_customers):
                inner = {}
                if i < len(best_assignments):
                    for j, frac in best_assignments[i].items():
                        if frac > 1e-9:
                            inner[str(j)] = frac
                if inner:
                    sol_assignments[str(i)] = inner
            sol = {
                "objective_value": best_cost,
                "open_plants": best_open,
                "assignments": sol_assignments
            }
            logger.log_solution(best_cost, sol)

    # Start with all plants open
    y = [True] * num_plants
    curr_cost, curr_open, curr_assignments = evaluate(y)
    if curr_cost < best_cost:
        best_cost = curr_cost
        best_open = curr_open
        best_assignments = curr_assignments
        log_solution()

    start_time = time.time()
    indices = list(range(num_plants))

    # Initial descent: try closing plants one by one
    improved = True
    while improved and time.time() - start_time < args.time_limit:
        improved = False
        random.shuffle(indices)
        for j in indices:
            if time.time() - start_time >= args.time_limit:
                break
            if y[j]:
                y[j] = False
                new_cost, new_open, new_assign = evaluate(y)
                if new_cost < curr_cost:
                    curr_cost = new_cost
                    curr_open = new_open
                    curr_assignments = new_assign
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_open = new_open
                        best_assignments = new_assign
                        log_solution()
                    improved = True
                else:
                    y[j] = True

    # Random restarts and 1-flip local search
    while time.time() - start_time < args.time_limit:
        y = [True] * num_plants
        num_close = random.randint(1, max(1, num_plants // 2))
        for _ in range(num_close):
            idx = random.choice(indices)
            y[idx] = False
        
        cost, open_plants, assignments = evaluate(y)
        if cost == float('inf'):
            continue
            
        improved = True
        while improved and time.time() - start_time < args.time_limit:
            improved = False
            random.shuffle(indices)
            for j in indices:
                if time.time() - start_time >= args.time_limit:
                    break
                y[j] = not y[j]
                new_cost, new_open, new_assign = evaluate(y)
                if new_cost < cost:
                    cost = new_cost
                    open_plants = new_open
                    assignments = new_assign
                    improved = True
                else:
                    y[j] = not y[j]
        
        if cost < best_cost:
            best_cost = cost
            best_open = open_plants
            best_assignments = assignments
            log_solution()

    sol_assignments = {}
    for i in range(num_customers):
        inner = {}
        if i < len(best_assignments):
            for j, frac in best_assignments[i].items():
                if frac > 1e-9:
                    inner[str(j)] = frac
        if inner:
            sol_assignments[str(i)] = inner

    solution = {
        "objective_value": best_cost,
        "open_plants": best_open,
        "assignments": sol_assignments
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()