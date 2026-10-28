import json
import argparse
import time
import numpy as np
import random


def greedy_construct(n, p, cost_matrix):
    """Greedy construction: add medians one by one maximizing cost reduction."""
    selected = []
    selected_set = set()
    current_cost = np.full(n, np.inf)

    for _ in range(p):
        best_gain = -np.inf
        best_j = -1
        for j in range(n):
            if j in selected_set:
                continue
            new_cost = np.minimum(current_cost, cost_matrix[:, j])
            gain = np.sum(current_cost - new_cost)
            if gain > best_gain:
                best_gain = gain
                best_j = j
        selected.append(best_j)
        selected_set.add(best_j)
        current_cost = np.minimum(current_cost, cost_matrix[:, best_j])

    return selected


def compute_assignments(n, cost_matrix, selected):
    """Compute the best assignment for each node."""
    best_cost = np.full(n, np.inf)
    best_med = np.full(n, -1, dtype=int)

    for j in selected:
        col = cost_matrix[:, j]
        mask = col < best_cost
        best_cost[mask] = col[mask]
        best_med[mask] = j

    return best_cost, best_med


def compute_second_cost(n, cost_matrix, selected, best_med):
    """Compute second-best cost for each node."""
    second_cost = np.full(n, np.inf)
    for j in selected:
        col = cost_matrix[:, j]
        mask = (best_med != j) & (col < second_cost)
        second_cost[mask] = col[mask]
    return second_cost


def compute_objective(n, cost_matrix, selected):
    best_cost, _ = compute_assignments(n, cost_matrix, selected)
    return int(round(np.sum(best_cost)))


def swap_local_search(n, p, cost_matrix, selected, time_limit):
    """Teitz-Bart swap local search with best improvement."""
    start_time = time.time()
    selected = list(selected)
    selected_set = set(selected)

    max_iterations = 2000
    for iteration in range(max_iterations):
        elapsed = time.time() - start_time
        if elapsed > time_limit * 0.95:
            break

        best_cost, best_med = compute_assignments(n, cost_matrix, selected)
        second_cost = compute_second_cost(n, cost_matrix, selected, best_med)

        best_delta = -1e-10
        best_swap = None

        for idx_a in range(p):
            a = selected[idx_a]
            mask_a = (best_med == a)

            for b in range(n):
                if b in selected_set:
                    continue

                # Cost change for nodes currently assigned to a
                new_cost_a = np.minimum(cost_matrix[mask_a, b], second_cost[mask_a])
                delta_a = np.sum(new_cost_a) - np.sum(best_cost[mask_a])

                # Cost change for other nodes (potential savings)
                mask_other = ~mask_a
                if np.any(mask_other):
                    savings = np.maximum(best_cost[mask_other] - cost_matrix[mask_other, b], 0)
                    delta_other = -np.sum(savings)
                else:
                    delta_other = 0.0

                delta = delta_a + delta_other

                if delta < best_delta:
                    best_delta = delta
                    best_swap = (idx_a, a, b)

        if best_swap is not None:
            idx_a, a, b = best_swap
            selected[idx_a] = b
            selected_set.remove(a)
            selected_set.add(b)
        else:
            break

    return selected


def build_solution(n, p, cost_matrix, selected):
    """Build the solution dictionary."""
    best_cost, best_med = compute_assignments(n, cost_matrix, selected)

    objective_value = int(round(np.sum(best_cost)))

    assignments = []
    for i in range(n):
        assignments.append({
            "customer": i,
            "facility": int(best_med[i]),
            "cost": int(cost_matrix[i, int(best_med[i])])
        })

    return {
        "objective_value": objective_value,
        "opened_facilities": sorted([int(x) for x in selected]),
        "assignments": assignments
    }


def solve_gurobi(n, p, cost_matrix, time_limit, initial_selected=None):
    """Solve using Gurobi MIP."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None

    try:
        model = gp.Model("p_median")
        model.setParam('OutputFlag', 0)
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1.0, time_limit))

        # Decision variables
        y = model.addVars(n, vtype=GRB.BINARY, name="y")
        x = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")

        # Objective: minimize total assignment cost
        obj_expr = gp.quicksum(
            int(cost_matrix[i, j]) * x[i, j]
            for i in range(n) for j in range(n)
        )
        model.setObjective(obj_expr, GRB.MINIMIZE)

        # Each node assigned to exactly one median
        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i, j] for j in range(n)) == 1,
                name=f"assign_{i}"
            )

        # Exactly p medians opened
        model.addConstr(
            gp.quicksum(y[j] for j in range(n)) == p,
            name="num_medians"
        )

        # Can only assign to open medians
        for i in range(n):
            for j in range(n):
                model.addConstr(x[i, j] <= y[j], name=f"link_{i}_{j}")

        # Warm start from heuristic solution
        if initial_selected is not None:
            init_set = set(initial_selected)
            for j in range(n):
                y[j].Start = 1.0 if j in init_set else 0.0
            best_cost, best_med = compute_assignments(n, cost_matrix, initial_selected)
            for i in range(n):
                x[i, int(best_med[i])].Start = 1.0

        model.optimize()

        if model.SolCount > 0:
            selected = [j for j in range(n) if y[j].X > 0.5]
            if len(selected) == p:
                obj = int(round(model.ObjVal))
                return obj, selected

        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="p-Median Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    p = instance['p']
    cost_matrix = np.array(instance['cost_matrix'], dtype=np.float64)

    best_obj = float('inf')
    best_selected = None
    best_solution = None

    # Phase 1: Greedy construction
    selected = greedy_construct(n, p, cost_matrix)
    obj = compute_objective(n, cost_matrix, selected)
    best_obj = obj
    best_selected = selected[:]
    best_solution = build_solution(n, p, cost_matrix, selected)
    if logger:
        logger.log_solution(best_obj, best_solution)

    # Phase 2: Swap local search (Teitz-Bart)
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time > 0.5:
        ls_time = min(remaining_time * 0.4, remaining_time - 1.0) if args.time_limit > 5 else remaining_time * 0.8
        selected_ls = swap_local_search(n, p, cost_matrix, selected, max(0.5, ls_time))
        obj_ls = compute_objective(n, cost_matrix, selected_ls)
        if obj_ls < best_obj:
            best_obj = obj_ls
            best_selected = selected_ls[:]
            best_solution = build_solution(n, p, cost_matrix, selected_ls)
            if logger:
                logger.log_solution(best_obj, best_solution)

    # Phase 3: Gurobi MIP (for smaller instances)
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time > 3 and n <= 250:
        gurobi_result = solve_gurobi(n, p, cost_matrix, remaining_time - 1.0, best_selected)
        if gurobi_result is not None:
            g_obj, g_selected = gurobi_result
            if g_obj < best_obj:
                best_obj = g_obj
                best_selected = g_selected[:]
                best_solution = build_solution(n, p, cost_matrix, g_selected)
                if logger:
                    logger.log_solution(best_obj, best_solution)

    # Phase 4: Iterated Local Search for further improvement
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time > 1.0:
        rng = random.Random(42)
        end_time = start_time + args.time_limit - 0.5

        while time.time() < end_time:
            # Perturbation: randomly swap k medians
            k = max(1, p // 4)
            perturbed = best_selected[:]
            perturbed_set = set(perturbed)
            closed = [j for j in range(n) if j not in perturbed_set]

            if not closed:
                break

            for _ in range(min(k, len(closed))):
                remove_idx = rng.randint(0, len(perturbed) - 1)
                add_idx = rng.randint(0, len(closed) - 1)
                removed = perturbed[remove_idx]
                added = closed[add_idx]
                perturbed[remove_idx] = added
                perturbed_set.discard(removed)
                perturbed_set.add(added)
                closed[add_idx] = removed

            # Local search on perturbed solution
            remaining = end_time - time.time()
            if remaining < 0.3:
                break

            improved = swap_local_search(n, p, cost_matrix, perturbed, min(remaining, 10.0))
            obj_new = compute_objective(n, cost_matrix, improved)

            if obj_new < best_obj:
                best_obj = obj_new
                best_selected = improved[:]
                best_solution = build_solution(n, p, cost_matrix, improved)
                if logger:
                    logger.log_solution(best_obj, best_solution)

    # Write final solution
    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f)


if __name__ == "__main__":
    main()