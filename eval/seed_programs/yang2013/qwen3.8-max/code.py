import json
import argparse
import time
import sys
from solution_logger import SolutionLogger


def compute_objective(solution, profit_matrix):
    """Compute total profit for a given solution."""
    n = len(solution)
    selected = [i for i in range(n) if solution[i] == 1]
    total = 0
    for i in selected:
        total += profit_matrix[i][i]
    for idx1 in range(len(selected)):
        for idx2 in range(idx1 + 1, len(selected)):
            total += profit_matrix[selected[idx1]][selected[idx2]]
    return total


def greedy_solution(n, capacity, weights, profit_matrix):
    """Greedy heuristic: iteratively add the item with best marginal profit."""
    solution = [0] * n
    current_weight = 0
    selected = set()

    while True:
        best_gain = 0
        best_item = -1
        for i in range(n):
            if i in selected:
                continue
            if current_weight + weights[i] > capacity:
                continue
            gain = profit_matrix[i][i]
            for j in selected:
                gain += profit_matrix[i][j]
            if gain > best_gain:
                best_gain = gain
                best_item = i

        if best_item == -1:
            break

        solution[best_item] = 1
        selected.add(best_item)
        current_weight += weights[best_item]

    return solution


def local_search(solution, n, capacity, weights, profit_matrix, time_limit, logger, best_obj):
    """Local search with add/remove/swap moves."""
    start_time = time.time()
    selected = set(i for i in range(n) if solution[i] == 1)
    current_weight = sum(weights[i] for i in selected)
    improved = True

    while improved and (time.time() - start_time) < time_limit:
        improved = False

        # Try removing items
        for i in list(selected):
            if time.time() - start_time > time_limit:
                break
            # Compute loss of removing i
            loss = profit_matrix[i][i]
            for j in selected:
                if j != i:
                    loss += profit_matrix[i][j]
            # If removing improves (shouldn't with non-negative profits, but just in case)
            # Actually with non-negative profits, removing never helps, skip

        # Try adding items
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            if i in selected:
                continue
            if current_weight + weights[i] > capacity:
                continue
            gain = profit_matrix[i][i]
            for j in selected:
                gain += profit_matrix[i][j]
            if gain > 0:
                selected.add(i)
                solution[i] = 1
                current_weight += weights[i]
                improved = True

        # Try swap moves
        for i in list(selected):
            if time.time() - start_time > time_limit:
                break
            for j in range(n):
                if time.time() - start_time > time_limit:
                    break
                if j in selected:
                    continue
                if current_weight - weights[i] + weights[j] > capacity:
                    continue
                # Gain of swapping i out and j in
                # Loss from removing i
                loss = profit_matrix[i][i]
                for k in selected:
                    if k != i:
                        loss += profit_matrix[i][k]
                # Gain from adding j (without i)
                gain = profit_matrix[j][j]
                for k in selected:
                    if k != i:
                        gain += profit_matrix[j][k]
                if gain > loss:
                    selected.remove(i)
                    selected.add(j)
                    solution[i] = 0
                    solution[j] = 1
                    current_weight = current_weight - weights[i] + weights[j]
                    improved = True
                    break

    obj = compute_objective(solution, profit_matrix)
    return solution, obj


def solve_with_gurobi(n, capacity, weights, profit_matrix, time_limit, logger, initial_solution=None):
    """Solve using Gurobi MIQP formulation."""
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("QKP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0

    # Create binary variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Objective: maximize sum of diagonal profits + pairwise interaction profits
    obj = gp.QuadExpr()
    for i in range(n):
        obj += profit_matrix[i][i] * x[i]
    for i in range(n):
        for j in range(i + 1, n):
            if profit_matrix[i][j] != 0:
                obj += profit_matrix[i][j] * x[i] * x[j]

    model.setObjective(obj, GRB.MAXIMIZE)

    # Capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "capacity")

    # Set initial solution if available
    if initial_solution is not None:
        for i in range(n):
            x[i].Start = initial_solution[i]

    # Callback to log incumbent solutions
    best_logged = [0.0]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            sol = [int(round(model.cbGetSolution(x[i]))) for i in range(n)]
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val > best_logged[0]:
                best_logged[0] = obj_val
                if logger:
                    solution_dict = {
                        "objective_value": float(obj_val),
                        "solution": sol
                    }
                    logger.log_solution(float(obj_val), solution_dict)

    model.optimize(callback)

    if model.SolCount > 0:
        solution = [int(round(x[i].X)) for i in range(n)]
        obj_val = compute_objective(solution, profit_matrix)
        return solution, obj_val

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance["n"]
    capacity = instance["capacity"]
    weights = instance["weights"]
    profit_matrix = instance["profit_matrix"]

    # Greedy initial solution
    greedy_sol = greedy_solution(n, capacity, weights, profit_matrix)
    greedy_obj = compute_objective(greedy_sol, profit_matrix)

    best_solution = greedy_sol[:]
    best_obj = greedy_obj

    if logger:
        logger.log_solution(float(best_obj), {"objective_value": float(best_obj), "solution": best_solution})

    # Local search improvement
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 1:
        ls_sol, ls_obj = local_search(best_solution[:], n, capacity, weights, profit_matrix,
                                       min(remaining * 0.3, 10), logger, best_obj)
        if ls_obj > best_obj:
            best_solution = ls_sol
            best_obj = ls_obj
            if logger:
                logger.log_solution(float(best_obj), {"objective_value": float(best_obj), "solution": best_solution})

    # Gurobi MIQP
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    if remaining > 0.5:
        try:
            gurobi_sol, gurobi_obj = solve_with_gurobi(
                n, capacity, weights, profit_matrix, remaining, logger, best_solution
            )
            if gurobi_sol is not None and gurobi_obj > best_obj:
                best_solution = gurobi_sol
                best_obj = gurobi_obj
                if logger:
                    logger.log_solution(float(best_obj), {"objective_value": float(best_obj), "solution": best_solution})
        except Exception as e:
            pass

    # Write final solution
    output = {
        "objective_value": float(best_obj),
        "solution": best_solution
    }

    with open(args.solution_path, 'w') as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()