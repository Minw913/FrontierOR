import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def compute_objective(assignment, values, pairwise):
    """Compute total objective value for a given assignment."""
    knapsacks = {}
    for item_str, k in assignment.items():
        i = int(item_str)
        if k not in knapsacks:
            knapsacks[k] = []
        knapsacks[k].append(i)

    total = 0.0
    for k, items in knapsacks.items():
        for i in items:
            total += values[i]
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                total += pairwise[items[a]][items[b]]
    return total


def greedy_heuristic(n, m, capacity, weights, values, pairwise):
    """Greedy heuristic with multiple item orderings."""
    orderings = [
        sorted(range(n), key=lambda i: (-values[i], weights[i])),
        sorted(range(n), key=lambda i: (-values[i] / max(weights[i], 1), weights[i])),
        sorted(range(n), key=lambda i: (weights[i], -values[i])),
        sorted(range(n), key=lambda i: -sum(pairwise[i])),
    ]

    best_assignment = {}
    best_obj = 0.0

    for order in orderings:
        assignment = {}
        kw = [0] * m
        ki = [[] for _ in range(m)]

        for i in order:
            best_k = -1
            best_marginal = 0.0

            for k in range(m):
                if kw[k] + weights[i] <= capacity:
                    marginal = float(values[i])
                    for j in ki[k]:
                        marginal += pairwise[i][j]
                    if marginal > best_marginal:
                        best_marginal = marginal
                        best_k = k

            if best_k >= 0:
                assignment[str(i)] = best_k
                kw[best_k] += weights[i]
                ki[best_k].append(i)

        obj = compute_objective(assignment, values, pairwise)
        if obj > best_obj:
            best_obj = obj
            best_assignment = assignment

    return best_assignment


def local_search(assignment, n, m, capacity, weights, values, pairwise, end_time):
    """First-improvement local search with add/remove/move moves."""
    best_assignment = dict(assignment)
    best_obj = compute_objective(best_assignment, values, pairwise)

    improved = True
    while improved and time.time() < end_time:
        improved = False

        # Build current state
        knapsack_items = [[] for _ in range(m)]
        item_knapsack = [-1] * n
        knapsack_weight = [0] * m

        for item_str, k in best_assignment.items():
            i = int(item_str)
            knapsack_items[k].append(i)
            item_knapsack[i] = k
            knapsack_weight[k] += weights[i]

        for i in range(n):
            if time.time() >= end_time:
                break

            current_k = item_knapsack[i]

            for target_k in range(-1, m):
                if target_k == current_k:
                    continue
                if target_k >= 0 and knapsack_weight[target_k] + weights[i] > capacity:
                    continue

                delta = 0.0
                if current_k >= 0:
                    delta -= values[i]
                    for j in knapsack_items[current_k]:
                        if j != i:
                            delta -= pairwise[i][j]
                if target_k >= 0:
                    delta += values[i]
                    for j in knapsack_items[target_k]:
                        delta += pairwise[i][j]

                if delta > 1e-10:
                    new_assignment = dict(best_assignment)
                    if current_k >= 0:
                        del new_assignment[str(i)]
                    if target_k >= 0:
                        new_assignment[str(i)] = target_k

                    best_assignment = new_assignment
                    best_obj += delta
                    improved = True
                    break

            if improved:
                break

    return best_assignment, best_obj


def solve_mip(data, time_limit, logger, start_time, warm_start=None):
    """Solve using Gurobi MIQP formulation."""
    n = data['n']
    m = data['m']
    capacity = data['capacity']
    weights = data['weights']
    values = data['values']
    pairwise = data['pairwise_values']

    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time <= 1.0:
        return -float('inf'), {}

    try:
        model = gp.Model("multi_knapsack_pairwise")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(remaining_time - 0.5, 1.0))
        model.setParam('OutputFlag', 0)

        # Decision variables: x[i,k] = 1 if item i is in knapsack k
        x = {}
        for i in range(n):
            for k in range(m):
                x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")

        model.update()

        # Quadratic objective
        obj = gp.QuadExpr()
        for i in range(n):
            for k in range(m):
                obj += values[i] * x[i, k]

        for i in range(n):
            for j in range(i + 1, n):
                pv = pairwise[i][j]
                if pv != 0:
                    for k in range(m):
                        obj += pv * x[i, k] * x[j, k]

        model.setObjective(obj, GRB.MAXIMIZE)

        # Each item in at most one knapsack
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1, name=f"assign_{i}")

        # Capacity constraints
        for k in range(m):
            model.addConstr(
                gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacity,
                name=f"cap_{k}"
            )

        model.update()

        # Warm start from heuristic
        if warm_start:
            for i in range(n):
                for k in range(m):
                    x[i, k].Start = 0.0
            for item_str, k in warm_start.items():
                x[int(item_str), k].Start = 1.0

        # Callback to log incumbent solutions
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    assignment = {}
                    for i in range(n):
                        for k in range(m):
                            val = model.cbGetSolution(x[i, k])
                            if val is not None and val > 0.5:
                                assignment[str(i)] = k
                                break
                    obj_val = compute_objective(assignment, values, pairwise)
                    if logger:
                        logger.log_solution(obj_val, {
                            "objective_value": float(obj_val),
                            "assignment": assignment
                        })
                except Exception:
                    pass

        model.optimize(callback)

        # Extract best solution
        if model.SolCount > 0:
            assignment = {}
            for i in range(n):
                for k in range(m):
                    if x[i, k].X > 0.5:
                        assignment[str(i)] = k
                        break

            obj_val = compute_objective(assignment, values, pairwise)
            model.dispose()
            return obj_val, assignment

        model.dispose()
        return -float('inf'), {}

    except Exception:
        return -float('inf'), {}


def main():
    parser = argparse.ArgumentParser(description="Multi-knapsack with pairwise values")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
    m = data['m']
    capacity = data['capacity']
    weights = data['weights']
    values = data['values']
    pairwise = data['pairwise_values']

    start_time = time.time()

    # Handle trivial cases
    if n == 0 or m == 0:
        solution = {"objective_value": 0.0, "assignment": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log_solution(0.0, solution)
        return

    # Step 1: Greedy heuristic
    best_assignment = greedy_heuristic(n, m, capacity, weights, values, pairwise)
    best_obj = compute_objective(best_assignment, values, pairwise)

    if logger:
        logger.log_solution(best_obj, {
            "objective_value": float(best_obj),
            "assignment": best_assignment
        })

    # Step 2: Local search on greedy solution (use ~15% of time)
    ls_end = start_time + args.time_limit * 0.15
    if time.time() < ls_end:
        ls_assignment, ls_obj = local_search(
            best_assignment, n, m, capacity, weights, values, pairwise, ls_end
        )
        if ls_obj > best_obj:
            best_assignment = ls_assignment
            best_obj = ls_obj
            if logger:
                logger.log_solution(best_obj, {
                    "objective_value": float(best_obj),
                    "assignment": best_assignment
                })

    # Step 3: MIP (use ~75% of remaining time)
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 2:
        mip_end = start_time + args.time_limit * 0.90
        mip_time = mip_end - time.time()
        if mip_time > 1:
            mip_obj, mip_assignment = solve_mip(
                data, mip_end, logger, start_time, best_assignment
            )
            if mip_assignment and mip_obj > best_obj:
                best_assignment = mip_assignment
                best_obj = mip_obj
                if logger:
                    logger.log_solution(best_obj, {
                        "objective_value": float(best_obj),
                        "assignment": best_assignment
                    })

    # Step 4: Final local search on best solution
    final_end = start_time + args.time_limit - 0.2
    if time.time() < final_end:
        ls_assignment, ls_obj = local_search(
            best_assignment, n, m, capacity, weights, values, pairwise, final_end
        )
        if ls_obj > best_obj:
            best_assignment = ls_assignment
            best_obj = ls_obj
            if logger:
                logger.log_solution(best_obj, {
                    "objective_value": float(best_obj),
                    "assignment": best_assignment
                })

    # Write final solution
    solution = {
        "objective_value": float(best_obj),
        "assignment": best_assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()