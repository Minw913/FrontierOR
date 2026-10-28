import json
import argparse
import time
from gurobipy import *
from solution_logger import SolutionLogger


def calculate_objective(assignment, profits, pairwise_profits):
    """Calculate the total objective value for a given assignment."""
    total = 0
    for item_idx, knapsack_idx in assignment:
        total += profits[item_idx]

    knapsack_items = {}
    for item_idx, knapsack_idx in assignment:
        if knapsack_idx not in knapsack_items:
            knapsack_items[knapsack_idx] = []
        knapsack_items[knapsack_idx].append(item_idx)

    for k, items in knapsack_items.items():
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                total += pairwise_profits[items[a]][items[b]]

    return total


def solve_gurobi(instance, time_limit, logger):
    """Solve using Gurobi MIQP formulation."""
    n = instance["n"]
    m = instance["m"]
    profits = instance["profits"]
    pairwise_profits = instance["pairwise_profits"]
    weights = instance["weights"]
    capacities = instance["capacities"]

    model = Model("quadratic_knapsack")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 0)

    # Decision variables: x[i,k] = 1 if item i is assigned to knapsack k
    x = {}
    for i in range(n):
        for k in range(m):
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")

    model.update()

    # Objective: individual profits + pairwise profits for co-assigned items
    obj = QuadExpr()
    for i in range(n):
        for k in range(m):
            if profits[i] != 0:
                obj += profits[i] * x[i, k]

    for i in range(n):
        for j in range(i + 1, n):
            pw = pairwise_profits[i][j]
            if pw != 0:
                for k in range(m):
                    obj += pw * x[i, k] * x[j, k]

    model.setObjective(obj, GRB.MAXIMIZE)

    # Each item assigned to at most one knapsack
    for i in range(n):
        model.addConstr(quicksum(x[i, k] for k in range(m)) <= 1, name=f"assign_{i}")

    # Capacity constraints
    for k in range(m):
        model.addConstr(quicksum(weights[i] * x[i, k] for i in range(n)) <= capacities[k], name=f"cap_{k}")

    model.update()

    # Callback to log incumbent solutions
    best_obj = float('-inf')
    best_assignment = []

    def callback(model, where):
        nonlocal best_obj, best_assignment
        if where == GRB.Callback.MIPSOL:
            sol_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if sol_obj > best_obj:
                best_obj = sol_obj
                assignment = []
                for i in range(n):
                    for k in range(m):
                        if model.cbGetSolution(x[i, k]) > 0.5:
                            assignment.append([i, k])
                best_assignment = assignment
                if logger:
                    sol_dict = {"objective_value": float(sol_obj), "assignment": assignment}
                    logger.log_solution(sol_obj, sol_dict)

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        assignment = []
        for i in range(n):
            for k in range(m):
                if x[i, k].X > 0.5:
                    assignment.append([i, k])
        obj_val = calculate_objective(assignment, profits, pairwise_profits)
        return obj_val, assignment
    elif best_assignment:
        obj_val = calculate_objective(best_assignment, profits, pairwise_profits)
        return obj_val, best_assignment
    else:
        return 0, []


def solve_heuristic(instance, time_limit, logger):
    """Solve using greedy construction + best-improvement local search."""
    n = instance["n"]
    m = instance["m"]
    profits = instance["profits"]
    pairwise_profits = instance["pairwise_profits"]
    weights = instance["weights"]
    capacities = instance["capacities"]

    start_time = time.time()

    # State
    knapsack_items = [[] for _ in range(m)]
    knapsack_weight = [0] * m
    item_assignment = [-1] * n
    # pw_sum[i][k] = sum of pairwise_profits[i][j] for all j currently in knapsack k
    pw_sum = [[0] * m for _ in range(n)]

    def add_item(i, k):
        knapsack_items[k].append(i)
        knapsack_weight[k] += weights[i]
        item_assignment[i] = k
        for j in range(n):
            if j != i:
                pw_sum[j][k] += pairwise_profits[i][j]

    def remove_item(i, k):
        knapsack_items[k].remove(i)
        knapsack_weight[k] -= weights[i]
        item_assignment[i] = -1
        for j in range(n):
            if j != i:
                pw_sum[j][k] -= pairwise_profits[i][j]

    def get_assignment():
        return [[i, item_assignment[i]] for i in range(n) if item_assignment[i] >= 0]

    # Greedy construction: sort by profit descending, add to best knapsack
    item_order = sorted(range(n), key=lambda i: (-profits[i], weights[i]))

    for i in item_order:
        best_gain = 0
        best_k = -1
        for k in range(m):
            if knapsack_weight[k] + weights[i] <= capacities[k]:
                gain = profits[i] + pw_sum[i][k]
                if gain > best_gain:
                    best_gain = gain
                    best_k = k
        if best_k >= 0:
            add_item(i, best_k)

    current_obj = calculate_objective(get_assignment(), profits, pairwise_profits)

    if logger:
        logger.log_solution(current_obj, {"objective_value": float(current_obj), "assignment": get_assignment()})

    # Best-improvement local search
    while time.time() - start_time < time_limit - 1:
        best_delta = 0
        best_move = None

        for i in range(n):
            current_k = item_assignment[i]

            if current_k >= 0:
                # Try removing item i from its current knapsack
                delta = -(profits[i] + pw_sum[i][current_k])
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', i, current_k, -1)

                # Try moving item i to another knapsack
                for k in range(m):
                    if k == current_k:
                        continue
                    if knapsack_weight[k] + weights[i] <= capacities[k]:
                        delta = pw_sum[i][k] - pw_sum[i][current_k]
                        if delta > best_delta:
                            best_delta = delta
                            best_move = ('move', i, current_k, k)
            else:
                # Try adding unassigned item i to a knapsack
                for k in range(m):
                    if knapsack_weight[k] + weights[i] <= capacities[k]:
                        delta = profits[i] + pw_sum[i][k]
                        if delta > best_delta:
                            best_delta = delta
                            best_move = ('add', i, -1, k)

        if best_move is None or best_delta <= 1e-9:
            break

        move_type, item, from_k, to_k = best_move

        if move_type == 'remove':
            remove_item(item, from_k)
        elif move_type == 'move':
            remove_item(item, from_k)
            add_item(item, to_k)
        elif move_type == 'add':
            add_item(item, to_k)

        current_obj += best_delta

        if logger:
            logger.log_solution(current_obj, {"objective_value": float(current_obj), "assignment": get_assignment()})

    return current_obj, get_assignment()


def main():
    parser = argparse.ArgumentParser(description="Quadratic Multiple Knapsack Problem Solver")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to JSONL log file")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    n = instance["n"]
    m = instance["m"]

    # Estimate number of quadratic terms to decide approach
    num_quad_terms = n * (n - 1) // 2 * m

    try:
        if num_quad_terms <= 500000:
            obj_val, assignment = solve_gurobi(instance, args.time_limit, logger)
        else:
            obj_val, assignment = solve_heuristic(instance, args.time_limit, logger)
    except Exception as e:
        # Fall back to heuristic if Gurobi fails
        obj_val, assignment = solve_heuristic(instance, args.time_limit, logger)

    solution = {
        "objective_value": float(obj_val),
        "assignment": assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()