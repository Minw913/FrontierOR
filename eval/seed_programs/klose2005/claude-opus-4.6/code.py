import argparse
import json
import time
import numpy as np
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    start_time = time.time()

    num_customers = data['num_customers']
    num_plants = data['num_plants']
    demands = np.array(data['customers']['demands'], dtype=np.float64)
    capacities = np.array(data['plants']['capacities'], dtype=np.float64)
    fixed_costs = np.array(data['plants']['fixed_costs'], dtype=np.float64)
    shipping_costs = np.array(data['shipping_costs'], dtype=np.float64)  # [customer][plant]

    best_solution = None
    best_obj = float('inf')

    def evaluate_and_assign(open_set):
        """Given a set of open plants, optimally assign customers (LP) using greedy capacity-aware assignment."""
        open_list = sorted(open_set)
        if not open_list:
            return None, float('inf')

        n_open = len(open_list)
        # Fixed cost
        fc = sum(fixed_costs[j] for j in open_list)

        # Remaining capacity
        rem_cap = {j: capacities[j] for j in open_list}

        # Assignment: fraction of customer i served by plant j
        assignments = {}
        total_ship = 0.0

        # Sort customers by demand (largest first for better packing)
        cust_order = np.argsort(-demands)

        for i in cust_order:
            i = int(i)
            d_i = demands[i]
            if d_i == 0:
                # Zero demand, assign to cheapest
                j_best = open_list[np.argmin([shipping_costs[i][j] for j in open_list])]
                assignments[i] = {j_best: 1.0}
                continue

            # Sort open plants by shipping cost for this customer
            plant_costs = [(shipping_costs[i][j], j) for j in open_list]
            plant_costs.sort()

            remaining_demand = d_i
            cust_assign = {}
            for cost_j, j in plant_costs:
                if remaining_demand <= 1e-10:
                    break
                available = rem_cap[j]
                if available <= 1e-10:
                    continue
                serve = min(remaining_demand, available)
                frac = serve / d_i
                cust_assign[j] = frac
                rem_cap[j] -= serve
                remaining_demand -= serve
                total_ship += cost_j * frac

            if remaining_demand > 1e-6:
                return None, float('inf')  # Infeasible

            assignments[i] = cust_assign

        obj = fc + total_ship
        return assignments, obj

    def log_if_better(obj, assignments, open_set):
        nonlocal best_obj, best_solution
        if obj < best_obj - 1e-6:
            best_obj = obj
            best_solution = (obj, list(open_set), assignments)
            if logger:
                logger.log(obj)

    # Greedy construction: start by opening cheapest effective plants
    # For each plant, compute "value" = how much shipping cost it could save
    # Simple heuristic: for each customer, find the cheapest plant

    # Strategy 1: Open all plants, then close expensive ones
    all_plants = set(range(num_plants))
    assignments, obj = evaluate_and_assign(all_plants)
    if assignments is not None:
        log_if_better(obj, assignments, all_plants)

    # Strategy 2: For each customer, find cheapest plant -> open those
    essential_plants = set()
    for i in range(num_customers):
        j_best = int(np.argmin(shipping_costs[i]))
        essential_plants.add(j_best)
    assignments, obj = evaluate_and_assign(essential_plants)
    if assignments is not None:
        log_if_better(obj, assignments, essential_plants)

    # Strategy 3: Greedy add - start empty, add plant that reduces cost most
    def greedy_add():
        nonlocal best_obj, best_solution
        current_open = set()
        current_obj = float('inf')

        # Rank plants by a score: ratio of fixed cost to capacity
        plant_scores = []
        for j in range(num_plants):
            # Average shipping cost to all customers weighted by demand
            avg_ship = np.sum(shipping_costs[:, j] * demands) / max(np.sum(demands), 1)
            score = fixed_costs[j] / max(capacities[j], 1) + avg_ship / max(np.sum(demands), 1)
            plant_scores.append((score, j))
        plant_scores.sort()

        for _, j in plant_scores:
            if time.time() - start_time > time_limit - 2:
                break
            trial = current_open | {j}
            assignments, obj = evaluate_and_assign(trial)
            if assignments is not None and obj < current_obj:
                current_obj = obj
                current_open = trial
                log_if_better(obj, assignments, current_open)

        return current_open, current_obj

    open_set, obj = greedy_add()

    # Strategy 4: Start from best solution, try removing each plant
    def local_search_remove(open_set):
        nonlocal best_obj, best_solution
        improved = True
        current_open = set(open_set)
        while improved and time.time() - start_time < time_limit - 2:
            improved = False
            for j in list(current_open):
                if time.time() - start_time > time_limit - 2:
                    break
                trial = current_open - {j}
                if not trial:
                    continue
                assignments, obj = evaluate_and_assign(trial)
                if assignments is not None and obj < best_obj - 1e-6:
                    current_open = trial
                    log_if_better(obj, assignments, current_open)
                    improved = True
                    break
        return current_open

    if best_solution is not None:
        open_set = set(best_solution[1])
        open_set = local_search_remove(open_set)

    # Strategy 5: Local search - swap (remove one, add another)
    def local_search_swap(open_set):
        nonlocal best_obj, best_solution
        current_open = set(open_set)
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            closed_plants = [j for j in range(num_plants) if j not in current_open]
            for j_remove in list(current_open):
                if time.time() - start_time > time_limit - 3:
                    return current_open
                for j_add in closed_plants:
                    if time.time() - start_time > time_limit - 3:
                        return current_open
                    trial = (current_open - {j_remove}) | {j_add}
                    assignments, obj = evaluate_and_assign(trial)
                    if assignments is not None and obj < best_obj - 1e-6:
                        current_open = trial
                        log_if_better(obj, assignments, current_open)
                        improved = True
                        break
                if improved:
                    break
        return current_open

    if best_solution is not None:
        open_set = set(best_solution[1])
        open_set = local_search_swap(open_set)

    # Try add again after swaps
    if best_solution is not None and time.time() - start_time < time_limit - 3:
        current_open = set(best_solution[1])
        for j in range(num_plants):
            if time.time() - start_time > time_limit - 2:
                break
            if j not in current_open:
                trial = current_open | {j}
                assignments, obj = evaluate_and_assign(trial)
                if assignments is not None and obj < best_obj - 1e-6:
                    current_open = trial
                    log_if_better(obj, assignments, current_open)

        local_search_remove(current_open)

    # Format output
    obj_val, open_plants, assignments = best_solution

    assignments_out = {}
    for i in range(num_customers):
        cust_assign = assignments.get(i, {})
        inner = {}
        for j, frac in cust_assign.items():
            if frac > 1e-10:
                inner[str(j)] = frac
        assignments_out[str(i)] = inner

    solution = {
        "objective_value": obj_val,
        "open_plants": sorted(open_plants),
        "assignments": assignments_out
    }

    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)