import json
import argparse
import time
import sys
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    time_limit = args.time_limit

    # Load instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['num_facilities']
    m = data['num_customers']
    opening_costs = data['opening_costs']
    alloc_costs = data['allocation_costs']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Simple but effective heuristic: greedy facility selection + optimal allocation
    best_obj = float('inf')
    best_y = None
    best_x = None

    # Rank facilities by opening cost + average allocation cost
    avg_alloc = [sum(row) / m for row in alloc_costs]
    scores = [opening_costs[j] + 0.05 * avg_alloc[j] for j in range(n)]
    sorted_idx = sorted(range(n), key=lambda j: scores[j])

    max_fac_to_try = min(n, 30)
    
    # Try different numbers of facilities from the ranked list
    for k in range(1, max_fac_to_try + 1):
        if time.time() - start_time > time_limit * 0.9:
            break

        current_y = [0] * n
        for j_idx in range(k):
            current_y[sorted_idx[j_idx]] = 1

        current_x = [[0.0 for _ in range(m)] for _ in range(n)]

        # Iterative local search with optimal allocation for quadratic objective
        for iteration in range(12):
            if time.time() - start_time > time_limit * 0.85:
                break

            open_facilities = [j for j in range(n) if current_y[j] == 1]
            if not open_facilities:
                break

            # Optimal fractional assignment for quadratic costs (closed form)
            for i in range(m):
                coeffs = [alloc_costs[j][i] for j in open_facilities]
                inv_coeffs = [1.0 / max(c, 1e-12) for c in coeffs]
                total_inv = sum(inv_coeffs)
                if total_inv < 1e-9:
                    frac = 1.0 / len(open_facilities)
                    for idx, j in enumerate(open_facilities):
                        current_x[j][i] = frac
                else:
                    for idx, j in enumerate(open_facilities):
                        current_x[j][i] = inv_coeffs[idx] / total_inv

            # Compute current objective
            current_obj = sum(opening_costs[j] for j in range(n) if current_y[j] == 1)
            for j in range(n):
                for i in range(m):
                    current_obj += alloc_costs[j][i] * (current_x[j][i] ** 2)

            improved = False
            # Try flipping each facility status (focus on borderline ones)
            candidates = list(range(n))
            # Prioritize flipping low-cost or high-cost ones
            candidates.sort(key=lambda j: opening_costs[j])

            for flip_j in candidates:
                if time.time() - start_time > time_limit * 0.8:
                    break

                temp_y = current_y[:]
                temp_y[flip_j] = 1 - temp_y[flip_j]
                open_set = [jj for jj in range(n) if temp_y[jj] == 1]
                if not open_set:
                    continue

                temp_x = [[0.0] * m for _ in range(n)]
                new_obj = sum(opening_costs[jj] for jj in range(n) if temp_y[jj] == 1)

                for i in range(m):
                    coeffs = [alloc_costs[jj][i] for jj in open_set]
                    invs = [1.0 / max(c, 1e-12) for c in coeffs]
                    total_inv = sum(invs)
                    for idx, jj in enumerate(open_set):
                        temp_x[jj][i] = invs[idx] / total_inv
                        new_obj += alloc_costs[jj][i] * (temp_x[jj][i] ** 2)

                if new_obj < current_obj - 1e-6:
                    current_y = temp_y
                    current_x = [row[:] for row in temp_x]
                    current_obj = new_obj
                    improved = True
                    break

            if not improved:
                break

        # Compute final objective for this candidate
        final_obj = sum(opening_costs[j] for j in range(n) if current_y[j] == 1)
        for j in range(n):
            for i in range(m):
                final_obj += alloc_costs[j][i] * current_x[j][i] ** 2

        if final_obj < best_obj - 1e-6:
            best_obj = final_obj
            best_y = current_y[:]
            best_x = [row[:] for row in current_x]

            solution_dict = {
                "objective_value": float(best_obj),
                "variant": "quadratic",
                "y": best_y,
                "x": best_x
            }
            if logger:
                logger.log_solution(best_obj, solution_dict)

    # Fallback: open a reasonable number of cheapest facilities
    if best_y is None or best_obj == float('inf'):
        best_y = [0] * n
        best_x = [[0.0] * m for _ in range(n)]
        num_to_open = max(3, min(8, n // 4))
        cheapest = sorted(range(n), key=lambda j: opening_costs[j])[:num_to_open]
        for j in cheapest:
            best_y[j] = 1

        open_set = [j for j in range(n) if best_y[j] == 1]
        best_obj = sum(opening_costs[j] for j in open_set)

        for i in range(m):
            coeffs = [alloc_costs[j][i] for j in open_set]
            invs = [1.0 / max(c, 1e-12) for c in coeffs]
            total = sum(invs)
            for idx, j in enumerate(open_set):
                best_x[j][i] = invs[idx] / total
                best_obj += alloc_costs[j][i] * (best_x[j][i] ** 2)

        solution_dict = {
            "objective_value": float(best_obj),
            "variant": "quadratic",
            "y": best_y,
            "x": best_x
        }
        if logger:
            logger.log_solution(best_obj, solution_dict)

    # Final output
    final_solution = {
        "objective_value": float(best_obj),
        "variant": "quadratic",
        "y": best_y,
        "x": best_x
    }

    if logger:
        logger.log_solution(best_obj, final_solution)

    with open(args.solution_path, 'w') as f:
        json.dump(final_solution, f, indent=2)


if __name__ == "__main__":
    main()