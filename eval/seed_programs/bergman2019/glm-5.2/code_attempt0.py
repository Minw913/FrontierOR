import argparse
import json
import time
import random
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        inst = json.load(f)
    n = inst['n']
    m = inst['m']
    profits = inst['profits']
    pairwise = inst['pairwise_profits']
    weights = inst['weights']
    capacities = inst['capacities']

    start_time = time.time()
    time_limit = args.time_limit

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Edge case: no items
    if n == 0:
        solution_dict = {
            "objective_value": 0.0,
            "assignment": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f, indent=2)
        return

    def compute_objective(assigned):
        obj = 0.0
        items = list(assigned.keys())
        for i in items:
            obj += profits[i]
        for idx, i in enumerate(items):
            k_i = assigned[i]
            for j in items[idx+1:]:
                if assigned[j] == k_i:
                    obj += pairwise[i][j]
        return obj

    def local_search(assigned, remaining_cap):
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            items = list(assigned.keys())
            for i in items:
                if time.time() - start_time >= time_limit:
                    break
                k_old = assigned[i]
                # try move to another knapsack
                for k_new in range(m):
                    if k_new == k_old:
                        continue
                    if remaining_cap[k_new] >= weights[i]:
                        delta = 0.0
                        for j in assigned:
                            if j == i:
                                continue
                            if assigned[j] == k_old:
                                delta -= pairwise[i][j]
                            if assigned[j] == k_new:
                                delta += pairwise[i][j]
                        if delta > 1e-6:
                            assigned[i] = k_new
                            remaining_cap[k_old] += weights[i]
                            remaining_cap[k_new] -= weights[i]
                            improved = True
                            break
                if improved:
                    break
                # try swap with another item
                for j in items:
                    if j == i:
                        continue
                    k_j = assigned[j]
                    if k_j == k_old:
                        continue
                    if (remaining_cap[k_old] + weights[i] - weights[j] >= 0 and
                        remaining_cap[k_j] + weights[j] - weights[i] >= 0):
                        delta = 0.0
                        for x in assigned:
                            if x == i or x == j:
                                continue
                            if assigned[x] == k_old:
                                delta -= pairwise[i][x]
                            if assigned[x] == k_j:
                                delta -= pairwise[j][x]
                        for x in assigned:
                            if x == i or x == j:
                                continue
                            if assigned[x] == k_j:
                                delta += pairwise[i][x]
                            if assigned[x] == k_old:
                                delta += pairwise[j][x]
                        if delta > 1e-6:
                            assigned[i] = k_j
                            assigned[j] = k_old
                            remaining_cap[k_old] += weights[i] - weights[j]
                            remaining_cap[k_j] += weights[j] - weights[i]
                            improved = True
                            break
                if improved:
                    break
                # try unassign
                loss = profits[i] + sum(pairwise[i][x] for x in assigned if x != i and assigned[x] == k_old)
                if loss < 0:
                    del assigned[i]
                    remaining_cap[k_old] += weights[i]
                    improved = True
                    break
            # try assign unassigned items
            for i in range(n):
                if i in assigned:
                    continue
                if time.time() - start_time >= time_limit:
                    break
                w_i = weights[i]
                for k in range(m):
                    if remaining_cap[k] >= w_i:
                        gain = profits[i] + sum(pairwise[i][j] for j in assigned if assigned[j] == k)
                        if gain > 1e-6:
                            assigned[i] = k
                            remaining_cap[k] -= w_i
                            improved = True
                            break
                if improved:
                    break
        return assigned, remaining_cap

    def greedy_potential():
        assigned = {}
        remaining_cap = list(capacities)
        items = list(range(n))
        items.sort(key=lambda i: profits[i] + sum(max(0, pairwise[i][j]) for j in range(n) if i != j), reverse=True)
        changed = True
        while changed and time.time() - start_time < time_limit:
            changed = False
            best_gain = 1e-6
            best_i = -1
            best_k = -1
            for i in items:
                if time.time() - start_time >= time_limit:
                    break
                if i in assigned:
                    continue
                w_i = weights[i]
                for k in range(m):
                    if remaining_cap[k] >= w_i:
                        gain = profits[i]
                        for j in assigned:
                            if assigned[j] == k:
                                gain += pairwise[i][j]
                        if gain > best_gain:
                            best_gain = gain
                            best_i = i
                            best_k = k
            if best_i != -1:
                assigned[best_i] = best_k
                remaining_cap[best_k] -= weights[best_i]
                changed = True
        return assigned, remaining_cap

    def best_pair_init():
        assigned = {}
        remaining_cap = list(capacities)
        items = list(range(n))
        random.shuffle(items)
        # seed each knapsack with the best pair
        for k in range(m):
            best_val = -float('inf')
            best_i = -1
            best_j = -1
            for idx, i in enumerate(items):
                if i in assigned:
                    continue
                for j in items[idx+1:]:
                    if j in assigned:
                        continue
                    if weights[i] + weights[j] <= remaining_cap[k]:
                        val = profits[i] + profits[j] + pairwise[i][j]
                        if val > best_val:
                            best_val = val
                            best_i = i
                            best_j = j
            if best_i != -1 and best_val > 0:
                assigned[best_i] = k
                assigned[best_j] = k
                remaining_cap[k] -= weights[best_i] + weights[best_j]
        # greedy fill remaining
        random.shuffle(items)
        changed = True
        while changed and time.time() - start_time < time_limit:
            changed = False
            best_gain = 1e-6
            best_i = -1
            best_k = -1
            for i in items:
                if i in assigned:
                    continue
                w_i = weights[i]
                for k in range(m):
                    if remaining_cap[k] >= w_i:
                        gain = profits[i]
                        for j in assigned:
                            if assigned[j] == k:
                                gain += pairwise[i][j]
                        if gain > best_gain:
                            best_gain = gain
                            best_i = i
                            best_k = k
            if best_i != -1:
                assigned[best_i] = best_k
                remaining_cap[best_k] -= weights[best_i]
                changed = True
        return assigned, remaining_cap

    def random_assignment():
        assigned = {}
        remaining_cap = list(capacities)
        items = list(range(n))
        random.shuffle(items)
        for i in items:
            feasible = [k for k in range(m) if remaining_cap[k] >= weights[i]]
            if feasible:
                k = random.choice(feasible)
                assigned[i] = k
                remaining_cap[k] -= weights[i]
        return assigned, remaining_cap

    best_obj = float('-inf')
    best_assigned = {}

    def update_best(assigned, obj):
        nonlocal best_obj, best_assigned
        if obj > best_obj + 1e-9:
            best_obj = obj
            best_assigned = dict(assigned)
            if logger:
                sol = {
                    "objective_value": float(best_obj),
                    "assignment": [[int(i), int(k)] for i, k in best_assigned.items()]
                }
                logger.log_solution(best_obj, sol)

    # Strategy 1: greedy potential
    if time.time() - start_time < time_limit:
        assigned, rem = greedy_potential()
        assigned, rem = local_search(assigned, rem)
        update_best(assigned, compute_objective(assigned))

    # Strategy 2: best pair initializations
    for _ in range(5):
        if time.time() - start_time >= time_limit:
            break
        assigned, rem = best_pair_init()
        assigned, rem = local_search(assigned, rem)
        update_best(assigned, compute_objective(assigned))

    # Strategy 3: random restarts until time limit
    while time.time() - start_time < time_limit:
        assigned, rem = random_assignment()
        assigned, rem = local_search(assigned, rem)
        update_best(assigned, compute_objective(assigned))

    # Strategy 4: Gurobi exact MIQP if instance size is moderate
    time_left = time_limit - (time.time() - start_time)
    num_vars = n * m
    num_quad = (n * (n - 1) // 2) * m
    if time_left > 2 and num_vars <= 5000 and num_quad <= 200000:
        try:
            model = Model('MQKP')
            model.params.Seed = 0
            model.params.MIPGap = 1e-4
            model.params.NumericFocus = 0
            model.params.Threads = 1
            model.params.TimeLimit = time_left
            model.params.NonConvex = 2
            model.params.OutputFlag = 0

            x = {}
            for i in range(n):
                for k in range(m):
                    x[i, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{k}')
            model.update()

            # each item at most one knapsack
            for i in range(n):
                model.addConstr(quicksum(x[i, k] for k in range(m)) <= 1)

            # capacity constraints
            for k in range(m):
                model.addConstr(quicksum(weights[i] * x[i, k] for i in range(n)) <= capacities[k])

            # objective
            obj_expr = quicksum(profits[i] * x[i, k] for i in range(n) for k in range(m))
            for i in range(n):
                for j in range(i + 1, n):
                    pij = pairwise[i][j]
                    if pij == 0:
                        continue
                    for k in range(m):
                        obj_expr += pij * x[i, k] * x[j, k]
            model.setObjective(obj_expr, GRB.MAXIMIZE)

            # complete MIP start from best heuristic solution
            for i in range(n):
                for k in range(m):
                    x[i, k].Start = 0.0
            if best_assigned:
                for i, k in best_assigned.items():
                    x[i, k].Start = 1.0

            # callback to log incumbents
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    sol = model.cbGetSolution(x)
                    assignment = []
                    for i in range(n):
                        for k in range(m):
                            if sol[i, k] > 0.5:
                                assignment.append([i, k])
                    sol_dict = {
                        "objective_value": float(obj),
                        "assignment": [[int(i), int(k)] for i, k in assignment]
                    }
                    if logger:
                        logger.log_solution(obj, sol_dict)

            model.optimize(callback)

            if model.SolCount > 0:
                obj_val = model.ObjVal
                if obj_val > best_obj + 1e-9:
                    best_obj = obj_val
                    best_assigned = {}
                    for i in range(n):
                        for k in range(m):
                            if x[i, k].X > 0.5:
                                best_assigned[i] = k
                    if logger:
                        sol_dict = {
                            "objective_value": float(best_obj),
                            "assignment": [[int(i), int(k)] for i, k in best_assigned.items()]
                        }
                        logger.log_solution(best_obj, sol_dict)
        except Exception:
            # If Gurobi fails for any reason, keep the heuristic solution
            pass

    # Write final solution
    solution_dict = {
        "objective_value": float(best_obj) if best_obj != float('-inf') else 0.0,
        "assignment": [[int(i), int(k)] for i, k in best_assigned.items()] if best_assigned else []
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)

if __name__ == '__main__':
    solve()