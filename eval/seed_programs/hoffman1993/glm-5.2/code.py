import time
import json
import argparse
import sys

from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_rows = instance['dimensions']['num_rows']
    num_cols = instance['dimensions']['num_cols']
    costs = instance['cost_vector']
    columns = instance['constraint_matrix_A']['columns']

    # Precompute mappings and filter invalid rows
    col_to_rows = []
    for col in columns:
        valid_rows = [r for r in set(col) if 0 <= r < num_rows]
        col_to_rows.append(valid_rows)
        
    row_to_cols = [[] for _ in range(num_rows)]
    for j, rows in enumerate(col_to_rows):
        for r in rows:
            row_to_cols[r].append(j)

    # Check for rows with no covering columns
    for r in range(num_rows):
        if not row_to_cols[r]:
            solution = {
                "objective_value": 0.0,
                "selected_rotations": [],
                "variable_values": {str(j): 0.0 for j in range(num_cols)}
            }
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f)
            return

    has_base = instance.get('has_base_constraints', False)
    if has_base:
        bc = instance['base_constraints']
        num_bases = bc['num_bases']
        D_rows = bc['D_matrix']['rows']
        lower = bc['lower_bounds_d1']
        upper = bc['upper_bounds_d2']
    else:
        num_bases = 0
        D_rows = []
        lower = []
        upper = []

    # Sort columns by cost for each row to help branch and bound
    for r in range(num_rows):
        row_to_cols[r].sort(key=lambda c: costs[c])

    col_active = [True] * num_cols
    row_uncovered = [True] * num_rows
    row_degrees = [len(cols) for cols in row_to_cols]
    uncovered_count = num_rows

    base_sums = [0.0] * num_bases
    remaining_max = [0.0] * num_bases
    if has_base:
        for k in range(num_bases):
            remaining_max[k] = sum(max(0.0, D_rows[k][c]) for c in range(num_cols))

    best_cost = float('inf')
    best_selected = []
    
    start_time = time.time()
    current_selected = []

    # --- Greedy Initial Solution ---
    greedy_selected = []
    greedy_uncovered = [True] * num_rows
    greedy_active = [True] * num_cols
    possible = True
    for r in range(num_rows):
        if greedy_uncovered[r]:
            best_c = -1
            for c in row_to_cols[r]:
                if greedy_active[c]:
                    best_c = c
                    break
            if best_c == -1:
                possible = False
                break
            greedy_selected.append(best_c)
            greedy_active[best_c] = False
            for r2 in col_to_rows[best_c]:
                if greedy_uncovered[r2]:
                    greedy_uncovered[r2] = False
                    for other_c in row_to_cols[r2]:
                        if other_c != best_c:
                            greedy_active[other_c] = False
    if possible:
        valid = True
        if has_base:
            for k in range(num_bases):
                s = sum(D_rows[k][c] for c in greedy_selected)
                if s < lower[k] - 1e-6 or s > upper[k] + 1e-6:
                    valid = False
                    break
        if valid:
            best_cost = sum(costs[c] for c in greedy_selected)
            best_selected = greedy_selected
            if logger:
                solution = {
                    "objective_value": float(best_cost),
                    "selected_rotations": best_selected,
                    "variable_values": {str(j): (1.0 if j in best_selected else 0.0) for j in range(num_cols)}
                }
                logger.log_solution(float(best_cost), solution)

    # --- Branch and Bound Exact Search ---
    def dfs(current_cost):
        nonlocal best_cost, best_selected, uncovered_count
        
        if time.time() - start_time > args.time_limit:
            return True

        if has_base:
            for k in range(num_bases):
                if base_sums[k] > upper[k] + 1e-6:
                    return False
                if base_sums[k] + remaining_max[k] < lower[k] - 1e-6:
                    return False

        if uncovered_count == 0:
            if has_base:
                for k in range(num_bases):
                    if base_sums[k] < lower[k] - 1e-6 or base_sums[k] > upper[k] + 1e-6:
                        return False
            if current_cost < best_cost:
                best_cost = current_cost
                best_selected = list(current_selected)
                if logger:
                    solution = {
                        "objective_value": float(best_cost),
                        "selected_rotations": best_selected,
                        "variable_values": {str(j): (1.0 if j in best_selected else 0.0) for j in range(num_cols)}
                    }
                    logger.log_solution(float(best_cost), solution)
            return False

        # Find the uncovered row with the minimum degree
        min_degree = float('inf')
        chosen_row = -1
        for r in range(num_rows):
            if row_uncovered[r]:
                if row_degrees[r] == 0:
                    return False
                if row_degrees[r] < min_degree:
                    min_degree = row_degrees[r]
                    chosen_row = r
                    if min_degree == 1:
                        break

        for c in row_to_cols[chosen_row]:
            if not col_active[c]:
                continue
            
            if current_cost + costs[c] >= best_cost:
                break

            current_selected.append(c)
            col_active[c] = False
            
            for r_deg in col_to_rows[c]:
                if row_uncovered[r_deg]:
                    row_degrees[r_deg] -= 1
                    
            if has_base:
                for k in range(num_bases):
                    val = max(0.0, D_rows[k][c])
                    if val > 0:
                        remaining_max[k] -= val
                    base_sums[k] += D_rows[k][c]

            covered_rows = []
            deactivated_cols = []
            
            for r in col_to_rows[c]:
                if row_uncovered[r]:
                    row_uncovered[r] = False
                    uncovered_count -= 1
                    covered_rows.append(r)
                    for other_c in row_to_cols[r]:
                        if other_c != c and col_active[other_c]:
                            col_active[other_c] = False
                            deactivated_cols.append(other_c)
                            for r2 in col_to_rows[other_c]:
                                if row_uncovered[r2]:
                                    row_degrees[r2] -= 1

            time_up = dfs(current_cost + costs[c])
            
            for other_c in deactivated_cols:
                col_active[other_c] = True
                for r2 in col_to_rows[other_c]:
                    if row_uncovered[r2]:
                        row_degrees[r2] += 1
            
            for r in covered_rows:
                row_uncovered[r] = True
                uncovered_count += 1
                
            for r_deg in col_to_rows[c]:
                if row_uncovered[r_deg]:
                    row_degrees[r_deg] += 1
                    
            if has_base:
                for k in range(num_bases):
                    val = max(0.0, D_rows[k][c])
                    if val > 0:
                        remaining_max[k] += val
                    base_sums[k] -= D_rows[k][c]
                    
            col_active[c] = True
            current_selected.pop()
            
            if time_up:
                return True
                
        return False

    dfs(0)

    if best_selected:
        solution = {
            "objective_value": float(best_cost),
            "selected_rotations": best_selected,
            "variable_values": {str(j): (1.0 if j in best_selected else 0.0) for j in range(num_cols)}
        }
    else:
        solution = {
            "objective_value": 0.0,
            "selected_rotations": [],
            "variable_values": {str(j): 0.0 for j in range(num_cols)}
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()