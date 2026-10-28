import argparse
import json
import time
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    num_rows = instance['dimensions']['num_rows']
    num_cols = instance['dimensions']['num_cols']
    cost_vector = instance['cost_vector']
    columns = instance['constraint_matrix_A']['columns']
    has_base_constraints = instance.get('has_base_constraints', False)
    
    # Build row-to-columns mapping
    row_to_cols = [[] for _ in range(num_rows)]
    for j in range(num_cols):
        for row_idx in columns[j]:
            row_to_cols[row_idx].append(j)
    
    # Base constraints data
    base_data = None
    if has_base_constraints and instance.get('base_constraints'):
        base_data = instance['base_constraints']
    
    def check_base_constraints(selected_set):
        if not base_data:
            return True
        num_bases = base_data['num_bases']
        D_matrix = base_data['D_matrix']['rows']
        lower_bounds = base_data['lower_bounds_d1']
        upper_bounds = base_data['upper_bounds_d2']
        for b in range(num_bases):
            d_row = D_matrix[b]
            val = sum(d_row[j] for j in selected_set)
            if val < lower_bounds[b] - 1e-9 or val > upper_bounds[b] + 1e-9:
                return False
        return True
    
    def compute_cost(selected_set):
        return sum(cost_vector[j] for j in selected_set)
    
    # Try to solve LP relaxation for guidance
    def solve_lp_relaxation():
        try:
            # Build sparse constraint matrix
            row_indices = []
            col_indices = []
            data = []
            for j in range(num_cols):
                for row_idx in columns[j]:
                    row_indices.append(row_idx)
                    col_indices.append(j)
                    data.append(1.0)
            
            A_eq = csr_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols))
            b_eq = np.ones(num_rows)
            c = np.array(cost_vector, dtype=float)
            bounds = [(0, 1) for _ in range(num_cols)]
            
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs',
                           options={'time_limit': max(1, time_limit // 4)})
            if result.success:
                return result.x
            return None
        except Exception:
            return None
    
    # Greedy set partitioning
    def greedy_solve(cost_weights=None):
        """Greedy algorithm using cost/coverage ratio"""
        if cost_weights is None:
            cost_weights = cost_vector
        
        uncovered = set(range(num_rows))
        selected = []
        used_rows = set()
        
        # Precompute column sets
        col_sets = [set(columns[j]) for j in range(num_cols)]
        
        while uncovered:
            if time.time() - start_time > time_limit - 2:
                return None
            
            best_j = -1
            best_score = float('inf')
            best_new_cover = set()
            
            for j in range(num_cols):
                col_set = col_sets[j]
                # Check if this rotation conflicts with already covered rows
                if col_set & used_rows:
                    continue
                new_cover = col_set & uncovered
                if not new_cover:
                    continue
                score = cost_weights[j] / len(new_cover)
                if score < best_score:
                    best_score = score
                    best_j = j
                    best_new_cover = new_cover
            
            if best_j == -1:
                # Try to find any column that covers at least one uncovered row
                # even if it overlaps (shouldn't happen in exact cover, but fallback)
                return None
            
            selected.append(best_j)
            uncovered -= best_new_cover
            used_rows |= col_sets[best_j]
        
        # Check if it's valid (exact cover - no overlaps already ensured by construction)
        return selected
    
    # LP-guided greedy
    def lp_guided_greedy(lp_sol):
        """Use LP solution to guide greedy selection"""
        # Sort columns by LP value (descending) then by cost
        col_priority = sorted(range(num_cols), key=lambda j: (-lp_sol[j], cost_vector[j]))
        
        uncovered = set(range(num_rows))
        selected = []
        used_rows = set()
        col_sets = [set(columns[j]) for j in range(num_cols)]
        
        # First pass: try columns with high LP values
        for j in col_priority:
            if not uncovered:
                break
            if time.time() - start_time > time_limit - 2:
                break
            col_set = col_sets[j]
            if col_set & used_rows:
                continue
            new_cover = col_set & uncovered
            if not new_cover:
                continue
            # Only select if it covers new rows and LP value is relatively high
            if lp_sol[j] > 0.01 or len(new_cover) > 0:
                selected.append(j)
                uncovered -= new_cover
                used_rows |= col_set
        
        if uncovered:
            # Fill remaining with greedy
            for j in range(num_cols):
                if not uncovered:
                    break
                if j in set(selected):
                    continue
                col_set = col_sets[j]
                if col_set & used_rows:
                    continue
                new_cover = col_set & uncovered
                if not new_cover:
                    continue
                selected.append(j)
                uncovered -= new_cover
                used_rows |= col_set
        
        if uncovered:
            return None
        return selected
    
    # Local search improvement
    def local_search(selected, time_budget):
        """Try to improve solution by swapping rotations"""
        ls_start = time.time()
        col_sets = [set(columns[j]) for j in range(num_cols)]
        
        best_selected = list(selected)
        best_cost = compute_cost(best_selected)
        improved = True
        
        while improved and time.time() - ls_start < time_budget and time.time() - start_time < time_limit - 2:
            improved = False
            selected_set = set(best_selected)
            
            # Try removing one rotation and replacing with cheaper alternatives
            for idx in range(len(best_selected)):
                if time.time() - start_time > time_limit - 2:
                    break
                
                j_remove = best_selected[idx]
                rows_to_cover = col_sets[j_remove]
                other_used = set()
                for k_idx, k in enumerate(best_selected):
                    if k_idx != idx:
                        other_used |= col_sets[k]
                
                # Find a single replacement that covers exactly the same rows
                for j_new in range(num_cols):
                    if j_new in selected_set:
                        continue
                    if col_sets[j_new] == rows_to_cover:
                        if cost_vector[j_new] < cost_vector[j_remove]:
                            # Check no overlap with other selected
                            if not (col_sets[j_new] & other_used):
                                new_selected = list(best_selected)
                                new_selected[idx] = j_new
                                if not has_base_constraints or check_base_constraints(new_selected):
                                    best_selected = new_selected
                                    best_cost = compute_cost(best_selected)
                                    improved = True
                                    if logger:
                                        logger.log(best_cost)
                                    break
                
                # Try 1-to-many replacement: remove one, add multiple cheaper ones
                # (skip for now to keep time manageable)
        
        return best_selected
    
    best_solution = None
    best_obj = float('inf')
    
    def update_best(selected):
        nonlocal best_solution, best_obj
        if selected is None:
            return
        if has_base_constraints and not check_base_constraints(selected):
            return
        cost = compute_cost(selected)
        if cost < best_obj:
            best_obj = cost
            best_solution = list(selected)
            if logger:
                logger.log(best_obj)
    
    # Strategy 1: Pure greedy with original costs
    sol = greedy_solve()
    update_best(sol)
    
    # Strategy 2: LP relaxation guided greedy
    if time.time() - start_time < time_limit - 5:
        lp_sol = solve_lp_relaxation()
        if lp_sol is not None:
            sol = lp_guided_greedy(lp_sol)
            update_best(sol)
            
            # Strategy 3: Greedy with LP-adjusted costs
            if time.time() - start_time < time_limit - 5:
                adjusted_costs = [cost_vector[j] * (1.0 - 0.5 * lp_sol[j]) for j in range(num_cols)]
                sol = greedy_solve(adjusted_costs)
                update_best(sol)
            
            # Strategy 4: Rounding LP solution
            if time.time() - start_time < time_limit - 5:
                # Try threshold rounding
                for threshold in [0.9, 0.7, 0.5, 0.3]:
                    if time.time() - start_time > time_limit - 3:
                        break
                    selected = []
                    used_rows = set()
                    col_sets_local = [set(columns[j]) for j in range(num_cols)]
                    
                    # Select columns above threshold
                    candidates = [(j, lp_sol[j]) for j in range(num_cols) if lp_sol[j] >= threshold]
                    candidates.sort(key=lambda x: -x[1])
                    
                    uncovered = set(range(num_rows))
                    for j, _ in candidates:
                        if col_sets_local[j] & used_rows:
                            continue
                        selected.append(j)
                        used_rows |= col_sets_local[j]
                        uncovered -= col_sets_local[j]
                    
                    # Fill remaining greedily
                    if uncovered:
                        remaining_cols = sorted(range(num_cols), key=lambda j: cost_vector[j] / max(1, len(col_sets_local[j] & uncovered)))
                        for j in remaining_cols:
                            if not uncovered:
                                break
                            if col_sets_local[j] & used_rows:
                                continue
                            new_cover = col_sets_local[j] & uncovered
                            if not new_cover:
                                continue
                            selected.append(j)
                            used_rows |= col_sets_local[j]
                            uncovered -= new_cover
                    
                    if not uncovered:
                        update_best(selected)
    
    # Local search on best solution
    if best_solution and time.time() - start_time < time_limit - 3:
        remaining_time = max(1, time_limit - (time.time() - start_time) - 2)
        improved = local_search(best_solution, remaining_time)
        update_best(improved)
    
    # Build output
    if best_solution is not None:
        selected_set = set(best_solution)
        var_values = {}
        for j in range(num_cols):
            var_values[str(j)] = 1.0 if j in selected_set else 0.0
        
        solution = {
            "objective_value": best_obj,
            "selected_rotations": sorted(best_solution),
            "variable_values": var_values
        }
    else:
        solution = {
            "objective_value": float('inf'),
            "selected_rotations": [],
            "variable_values": {str(j): 0.0 for j in range(num_cols)}
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


def main():
    parser = argparse.ArgumentParser(description="Crew Scheduling Optimization")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()