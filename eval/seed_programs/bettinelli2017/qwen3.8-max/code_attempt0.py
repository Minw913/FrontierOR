import json
import argparse
import time
import sys
from solution_logger import SolutionLogger

def solve():
    parser = argparse.ArgumentParser(description="Knapsack with Conflicts Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_items = instance['num_items']
    capacity = instance['capacity']
    profits = instance['items']['profits']
    weights = instance['items']['weights']
    edges = instance['conflict_graph']['edges']

    # Build adjacency sets for quick conflict checking
    adj = [set() for _ in range(num_items)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    start_time = time.time()

    best_objective = 0
    best_solution = []

    def make_solution_dict(obj, items):
        return {
            "objective_value": float(obj),
            "selected_items": sorted(items)
        }

    def update_best(obj, items):
        nonlocal best_objective, best_solution
        if obj > best_objective:
            best_objective = obj
            best_solution = sorted(items)
            if logger:
                logger.log_solution(best_objective, make_solution_dict(best_objective, best_solution))

    # Greedy heuristic: try multiple orderings
    def greedy(order):
        selected = set()
        total_weight = 0
        total_profit = 0
        for i in order:
            if total_weight + weights[i] <= capacity:
                conflict = False
                for j in selected:
                    if j in adj[i]:
                        conflict = True
                        break
                if not conflict:
                    selected.add(i)
                    total_weight += weights[i]
                    total_profit += profits[i]
        return total_profit, selected

    # Ordering 1: by profit/weight ratio (descending)
    order1 = sorted(range(num_items), key=lambda i: profits[i] / max(weights[i], 1), reverse=True)
    p, s = greedy(order1)
    update_best(p, s)

    # Ordering 2: by profit (descending)
    order2 = sorted(range(num_items), key=lambda i: profits[i], reverse=True)
    p, s = greedy(order2)
    update_best(p, s)

    # Ordering 3: by weight (ascending) to fit more items
    order3 = sorted(range(num_items), key=lambda i: weights[i])
    p, s = greedy(order3)
    update_best(p, s)

    # Ordering 4: by profit/weight ratio but break ties by fewer conflicts
    order4 = sorted(range(num_items), key=lambda i: (-profits[i] / max(weights[i], 1), len(adj[i])))
    p, s = greedy(order4)
    update_best(p, s)

    # Local search improvement on best greedy solution
    elapsed = time.time() - start_time
    if elapsed < args.time_limit * 0.1 and best_solution:
        improved = True
        current_set = set(best_solution)
        current_profit = best_objective
        current_weight = sum(weights[i] for i in current_set)
        
        while improved and (time.time() - start_time) < args.time_limit * 0.2:
            improved = False
            # Try adding items not in the set
            candidates = sorted(set(range(num_items)) - current_set, 
                              key=lambda i: profits[i], reverse=True)
            for i in candidates:
                if current_weight + weights[i] > capacity:
                    continue
                if any(j in adj[i] for j in current_set):
                    continue
                current_set.add(i)
                current_weight += weights[i]
                current_profit += profits[i]
                improved = True
                break
            
            # Try swaps: remove one item, add another
            if not improved:
                for i in list(current_set):
                    removed_weight = weights[i]
                    removed_profit = profits[i]
                    for j in candidates:
                        if j in current_set:
                            continue
                        if current_weight - removed_weight + weights[j] > capacity:
                            continue
                        # Check conflicts with remaining items
                        remaining = current_set - {i}
                        if any(k in adj[j] for k in remaining):
                            continue
                        # Check if swap improves profit
                        if profits[j] > removed_profit:
                            current_set.remove(i)
                            current_set.add(j)
                            current_weight = current_weight - removed_weight + weights[j]
                            current_profit = current_profit - removed_profit + profits[j]
                            improved = True
                            break
                    if improved:
                        break
        
        update_best(current_profit, current_set)

    # Gurobi MIP solver
    elapsed = time.time() - start_time
    remaining_time = args.time_limit - elapsed

    if remaining_time > 2:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("knapsack_conflict")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(remaining_time - 1, 1)
            model.Params.OutputFlag = 0

            # Decision variables
            x = model.addVars(num_items, vtype=GRB.BINARY, name="x")

            # Objective: maximize profit
            model.setObjective(
                gp.quicksum(profits[i] * x[i] for i in range(num_items)),
                GRB.MAXIMIZE
            )

            # Capacity constraint
            model.addConstr(
                gp.quicksum(weights[i] * x[i] for i in range(num_items)) <= capacity,
                name="capacity"
            )

            # Conflict constraints
            for u, v in edges:
                model.addConstr(x[u] + x[v] <= 1, name=f"conflict_{u}_{v}")

            # Provide warm start from best heuristic solution
            for i in range(num_items):
                if i in set(best_solution):
                    x[i].Start = 1.0
                else:
                    x[i].Start = 0.0

            # Callback to log incumbent solutions
            def callback(model_cb, where):
                if where == GRB.Callback.MIPSOL:
                    sol = model_cb.cbGetSolution(x)
                    cb_items = [i for i in range(num_items) if sol[i] > 0.5]
                    cb_profit = sum(profits[i] for i in cb_items)
                    if cb_profit > best_objective:
                        update_best(cb_profit, cb_items)

            model.optimize(callback)

            # Extract final solution from Gurobi
            if model.SolCount > 0:
                gurobi_items = [i for i in range(num_items) if x[i].X > 0.5]
                gurobi_profit = sum(profits[i] for i in gurobi_items)
                update_best(gurobi_profit, gurobi_items)

        except ImportError:
            pass
        except Exception as e:
            # If Gurobi fails, fall back to heuristic solution
            pass

    # Write final solution
    solution = make_solution_dict(best_objective, best_solution)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

    # Final log if not already logged
    if logger and best_objective > 0:
        pass  # Already logged in update_best

if __name__ == "__main__":
    solve()