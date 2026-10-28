import json
import argparse
import time
import sys

def compute_objective(selected, profit_matrix):
    """Compute total profit for a given selection."""
    n = len(selected)
    total = 0
    for i in range(n):
        if selected[i]:
            total += profit_matrix[i][i]
            for j in range(i + 1, n):
                if selected[j]:
                    total += 2 * profit_matrix[i][j]
    return total


def greedy_heuristic(instance):
    """Greedy heuristic: sort by profit density and add items."""
    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    P = instance['profit_matrix']

    # Score each item by (individual profit + average pair potential) / weight
    scores = []
    for i in range(n):
        individual = P[i][i]
        pair_sum = sum(P[i][j] for j in range(n) if j != i)
        score = (individual + pair_sum) / max(weights[i], 1)
        scores.append((score, i))

    scores.sort(reverse=True)

    selected = [0] * n
    remaining_capacity = capacity

    for _, i in scores:
        if weights[i] <= remaining_capacity:
            selected[i] = 1
            remaining_capacity -= weights[i]

    return selected


def local_search(selected, instance, time_budget):
    """Local search: try add, remove, and swap moves."""
    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    P = instance['profit_matrix']

    start_time = time.time()
    best_selected = selected[:]
    best_obj = compute_objective(selected, P)
    current_weight = sum(weights[i] * selected[i] for i in range(n))

    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False

        # Try adding items
        for i in range(n):
            if time.time() - start_time >= time_budget:
                break
            if selected[i] == 0 and current_weight + weights[i] <= capacity:
                selected[i] = 1
                new_obj = compute_objective(selected, P)
                if new_obj > best_obj:
                    best_obj = new_obj
                    best_selected = selected[:]
                    current_weight += weights[i]
                    improved = True
                else:
                    selected[i] = 0

        # Try swaps
        for i in range(n):
            if time.time() - start_time >= time_budget:
                break
            if selected[i] == 1:
                for j in range(n):
                    if time.time() - start_time >= time_budget:
                        break
                    if selected[j] == 0 and current_weight - weights[i] + weights[j] <= capacity:
                        selected[i] = 0
                        selected[j] = 1
                        new_obj = compute_objective(selected, P)
                        if new_obj > best_obj:
                            best_obj = new_obj
                            best_selected = selected[:]
                            current_weight = current_weight - weights[i] + weights[j]
                            improved = True
                            break
                        else:
                            selected[i] = 1
                            selected[j] = 0

        selected = best_selected[:]
        current_weight = sum(weights[i] * selected[i] for i in range(n))

    return best_selected, best_obj


def solve_with_gurobi(instance, time_limit, logger, initial_solution=None):
    """Solve using Gurobi MIP formulation."""
    import gurobipy as gp
    from gurobipy import GRB

    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    P = instance['profit_matrix']

    model = gp.Model("QKP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0

    # Decision variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Create y variables only for pairs with positive off-diagonal profit
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            if P[i][j] > 0:
                y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    model.update()

    # Objective: maximize sum of P[i][i]*x[i] + 2*P[i][j]*y[i][j]
    obj = gp.quicksum(P[i][i] * x[i] for i in range(n))
    obj += gp.quicksum(2 * P[i][j] * y[i, j] for (i, j) in y)
    model.setObjective(obj, GRB.MAXIMIZE)

    # Weight constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, name="capacity")

    # Linearization constraints
    for (i, j) in y:
        model.addConstr(y[i, j] <= x[i], name=f"lin1_{i}_{j}")
        model.addConstr(y[i, j] <= x[j], name=f"lin2_{i}_{j}")
        model.addConstr(y[i, j] >= x[i] + x[j] - 1, name=f"lin3_{i}_{j}")

    # Set initial solution as MIP start if provided
    if initial_solution is not None:
        for i in range(n):
            x[i].Start = initial_solution[i]
        for (i, j) in y:
            y[i, j].Start = initial_solution[i] * initial_solution[j]

    # Callback to log incumbent solutions
    best_logged = [0]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            sol = [int(round(model.cbGetSolution(x[i]).X)) for i in range(n)]
            obj_val = compute_objective(sol, P)
            if obj_val > best_logged[0]:
                best_logged[0] = obj_val
                if logger:
                    sol_dict = {
                        "objective_value": float(obj_val),
                        "selected_items": sol
                    }
                    logger.log_solution(obj_val, sol_dict)

    model.optimize(callback)

    # Extract best solution
    if model.SolCount > 0:
        selected = [int(round(x[i].X)) for i in range(n)]
        obj_val = compute_objective(selected, P)
        return selected, obj_val
    else:
        return None, -1


def main():
    parser = argparse.ArgumentParser(description="Quadratic Knapsack Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    start_time = time.time()
    n = instance['n']
    capacity = instance['capacity']
    weights = instance['weights']
    P = instance['profit_matrix']

    # Phase 1: Quick greedy heuristic
    greedy_selected = greedy_heuristic(instance)
    greedy_obj = compute_objective(greedy_selected, P)

    best_selected = greedy_selected[:]
    best_obj = greedy_obj

    if logger:
        logger.log_solution(best_obj, {
            "objective_value": float(best_obj),
            "selected_items": best_selected
        })

    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed

    # Phase 2: Local search improvement (use up to 20% of remaining time)
    if remaining > 2:
        ls_budget = min(remaining * 0.2, remaining - 1)
        if ls_budget > 0.5:
            ls_selected, ls_obj = local_search(best_selected[:], instance, ls_budget)
            if ls_obj > best_obj:
                best_selected = ls_selected[:]
                best_obj = ls_obj
                if logger:
                    logger.log_solution(best_obj, {
                        "objective_value": float(best_obj),
                        "selected_items": best_selected
                    })

    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed

    # Phase 3: Gurobi MIP (use remaining time)
    if remaining > 1:
        try:
            gurobi_selected, gurobi_obj = solve_with_gurobi(
                instance, remaining, logger, initial_solution=best_selected
            )
            if gurobi_selected is not None and gurobi_obj > best_obj:
                best_selected = gurobi_selected[:]
                best_obj = gurobi_obj
        except Exception as e:
            # If Gurobi fails, keep heuristic solution
            pass

    # Ensure final solution is logged
    if logger:
        logger.log_solution(best_obj, {
            "objective_value": float(best_obj),
            "selected_items": best_selected
        })

    # Write final solution
    solution = {
        "objective_value": float(best_obj),
        "selected_items": best_selected
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


from solution_logger import SolutionLogger

if __name__ == '__main__':
    main()