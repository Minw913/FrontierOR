import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_items = instance['num_items']
    bin_capacity = instance['bin_capacity']
    items = instance['items']
    conflict_edges = instance['conflict_edges']

    item_ids = [item['id'] for item in items]
    weights = {item['id']: item['weight'] for item in items}
    max_id = max(item_ids)
    
    # Build conflict set for faster lookup
    conflicts = {i: set() for i in item_ids}
    for a, b in conflict_edges:
        if a in conflicts and b in conflicts:
            conflicts[a].add(b)
            conflicts[b].add(a)

    # Heuristic upper bound: greedy coloring like approach with bin packing
    # Use a simple First-Fit Decreasing with conflicts
    def heuristic_upper_bound():
        sorted_items = sorted(item_ids, key=lambda x: weights[x], reverse=True)
        bins = []
        for item in sorted_items:
            placed = False
            for b in bins:
                total_w = sum(weights[i] for i in b)
                if total_w + weights[item] <= bin_capacity:
                    conflict_free = all(item not in conflicts[other] for other in b)
                    if conflict_free:
                        b.append(item)
                        placed = True
                        break
            if not placed:
                bins.append([item])
        return len(bins), bins

    ub, heuristic_bins = heuristic_upper_bound()
    best_objective = ub
    best_bins = heuristic_bins

    if logger:
        logger.log(best_objective)

    # If only one bin needed, we can return immediately
    if best_objective == 1:
        write_solution(args.solution_path, best_objective, best_bins, weights)
        return

    # Try to solve with MIP using Gurobi with time limit
    try:
        m = gp.Model("BinPackingWithConflicts")
        m.setParam('TimeLimit', args.time_limit)
        m.setParam('Threads', 1)
        m.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions
        m.setParam('Heuristics', 0.8)
        m.setParam('Cuts', 2)

        # We will use a formulation with potential bins up to the heuristic UB
        K = list(range(ub))  # 0 to ub-1
        x = m.addVars(item_ids, K, vtype=GRB.BINARY, name="x")
        y = m.addVars(K, vtype=GRB.BINARY, name="y")

        # Objective
        m.setObjective(gp.quicksum(y[k] for k in K), GRB.MINIMIZE)

        # Each item in exactly one bin
        for i in item_ids:
            m.addConstr(gp.quicksum(x[i, k] for k in K) == 1, name=f"assign_{i}")

        # Capacity constraint
        for k in K:
            m.addConstr(gp.quicksum(weights[i] * x[i, k] for i in item_ids) <= bin_capacity * y[k],
                        name=f"capacity_{k}")

        # Conflict constraints
        for k in K:
            for a, b in conflict_edges:
                if a in item_ids and b in item_ids:
                    m.addConstr(x[a, k] + x[b, k] <= 1, name=f"conflict_{a}_{b}_{k}")

        # Symmetry breaking: use bins in order
        for k in range(1, len(K)):
            m.addConstr(y[k] <= y[k-1], name=f"symmetry_{k}")

        # Warm start with heuristic solution
        for kidx, bin_items in enumerate(heuristic_bins):
            if kidx < len(K):
                y[kidx].Start = 1.0
                for i in bin_items:
                    x[i, kidx].Start = 1.0
                # Set other bins to 0 for these items
                for other_k in range(len(K)):
                    if other_k != kidx:
                        for i in bin_items:
                            x[i, other_k].Start = 0.0

        # Callback to log improved solutions
        def callback(model, where):
            nonlocal best_objective, best_bins
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                if obj < best_objective - 0.5:  # improved
                    best_objective = int(round(obj))
                    # Extract solution
                    sol_bins = []
                    x_sol = model.cbGetSolution(x)
                    for k in K:
                        bin_items = [i for i in item_ids if x_sol[i, k] > 0.5]
                        if bin_items:
                            sol_bins.append(bin_items)
                    best_bins = sol_bins
                    if logger:
                        logger.log(best_objective)

        m.optimize(callback)

        if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
            if m.SolCount > 0:
                current_obj = int(m.ObjVal + 0.5)
                if current_obj < best_objective:
                    best_objective = current_obj
                    # Extract best solution found
                    sol_bins = []
                    for k in K:
                        bin_items = [i for i in item_ids if x[i, k].X > 0.5]
                        if bin_items:
                            sol_bins.append(bin_items)
                    best_bins = sol_bins
                    if logger:
                        logger.log(best_objective)

    except Exception as e:
        # If MIP fails for any reason, fall back to heuristic
        pass

    write_solution(args.solution_path, best_objective, best_bins, weights)


def write_solution(solution_path, objective_value, bins_list, weights):
    solution = {
        "objective_value": objective_value,
        "bins": {}
    }
    
    for bin_id, item_list in enumerate(bins_list):
        total_weight = sum(weights[i] for i in item_list)
        solution["bins"][str(bin_id)] = {
            "items": item_list,
            "total_weight": total_weight
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()