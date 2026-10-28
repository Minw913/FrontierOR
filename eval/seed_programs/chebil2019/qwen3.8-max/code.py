import json
import argparse
import time
import sys

def main():
    parser = argparse.ArgumentParser(description="Knapsack with Setup Costs Solver")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    N = instance['N']
    capacity = instance['knapsack_capacity']
    families = instance['families']

    start_time = time.time()

    # Preprocessing: filter families that can never be activated
    valid_families = []
    for fam in families:
        if fam['setup_capacity'] <= capacity:
            valid_families.append(fam)

    # If capacity is small enough, use DP approach
    if capacity <= 50000 and len(valid_families) <= 200:
        solution = solve_dp(valid_families, capacity, logger)
    else:
        solution = None

    # If DP didn't solve or capacity too large, use Gurobi
    if solution is None:
        remaining_time = args.time_limit - (time.time() - start_time)
        if remaining_time > 1:
            solution = solve_gurobi(valid_families, capacity, remaining_time, logger)

    # Fallback: empty solution
    if solution is None:
        solution = {
            "objective_value": 0.0,
            "families_selected": [],
            "items_selected": []
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def solve_dp(families, capacity, logger):
    """Solve using dynamic programming when capacity is manageable."""
    N = len(families)
    if N == 0:
        return {"objective_value": 0.0, "families_selected": [], "items_selected": []}

    # For each family, compute the Pareto-optimal (capacity_used, net_profit) options
    # Option: activate family and select a subset of items
    # capacity_used = setup_capacity + sum of item weights
    # net_profit = sum of item profits - setup_cost

    family_options = []
    for fam in families:
        fid = fam['family_id']
        setup_cap = fam['setup_capacity']
        setup_cost = fam['setup_cost']
        items = fam['items']
        n_items = len(items)

        # Compute knapsack DP for items in this family
        max_item_cap = capacity - setup_cap
        if max_item_cap < 0:
            continue

        # dp[c] = max profit using items with total weight <= c
        dp_items = [0] * (max_item_cap + 1)
        # Track which items are selected for reconstruction
        # For memory efficiency, we'll reconstruct later
        item_dp = [[0] * (max_item_cap + 1) for _ in range(n_items + 1)]

        for j in range(n_items):
            w = items[j]['weight']
            p = items[j]['profit']
            for c in range(max_item_cap + 1):
                item_dp[j + 1][c] = item_dp[j][c]
                if c >= w and item_dp[j][c - w] + p > item_dp[j + 1][c]:
                    item_dp[j + 1][c] = item_dp[j][c - w] + p

        # Generate options: for each possible item capacity c_item,
        # total capacity = setup_cap + c_item, net profit = item_dp[n_items][c_item] - setup_cost
        # Only include if net_profit > 0 or c_item == 0 with net_profit >= 0
        options = []
        for c_item in range(max_item_cap + 1):
            total_cap = setup_cap + c_item
            net_profit = item_dp[n_items][c_item] - setup_cost
            if net_profit > 0 or (net_profit == 0 and c_item == 0):
                options.append((total_cap, net_profit, c_item))

        # Filter dominated options: keep only Pareto-optimal
        # Sort by capacity, then filter
        options.sort(key=lambda x: x[0])
        pareto = []
        best_profit = -1
        for cap, prof, c_item in options:
            if prof > best_profit:
                pareto.append((cap, prof, c_item))
                best_profit = prof

        if pareto:
            family_options.append({
                'family': fam,
                'options': pareto,
                'item_dp': item_dp,
                'items': items
            })

    # Outer DP over families
    # dp[c] = max net profit using capacity exactly c (or at most c)
    dp = [0] * (capacity + 1)
    # Track decisions for reconstruction
    # choice[i][c] = option index chosen for family i at capacity c, or -1 if skipped
    choices = []

    for fi, fo in enumerate(family_options):
        new_dp = dp[:]
        choice = [-1] * (capacity + 1)

        for opt_idx, (opt_cap, opt_profit, c_item) in enumerate(fo['options']):
            if opt_cap > capacity:
                continue
            for c in range(opt_cap, capacity + 1):
                val = dp[c - opt_cap] + opt_profit
                if val > new_dp[c]:
                    new_dp[c] = val
                    choice[c] = opt_idx

        dp = new_dp
        choices.append(choice)

    # Find best capacity
    best_cap = 0
    best_val = 0
    for c in range(capacity + 1):
        if dp[c] > best_val:
            best_val = dp[c]
            best_cap = c

    # Reconstruct solution
    families_selected = []
    items_selected = []
    remaining_cap = best_cap

    for fi in range(len(family_options) - 1, -1, -1):
        opt_idx = choices[fi][remaining_cap]
        if opt_idx >= 0:
            fo = family_options[fi]
            fam = fo['family']
            fid = fam['family_id']
            opt_cap, opt_profit, c_item = fo['options'][opt_idx]
            families_selected.append(fid)
            remaining_cap -= opt_cap

            # Reconstruct items selected using item_dp
            if c_item > 0:
                items = fo['items']
                item_dp = fo['item_dp']
                n_items = len(items)
                c = c_item
                for j in range(n_items - 1, -1, -1):
                    if item_dp[j + 1][c] != item_dp[j][c]:
                        items_selected.append({"family": fid, "item": items[j]['item_id']})
                        c -= items[j]['weight']

    solution = {
        "objective_value": float(best_val),
        "families_selected": families_selected,
        "items_selected": items_selected
    }

    if logger:
        logger.log_solution(best_val, solution)

    return solution


def solve_gurobi(families, capacity, time_limit, logger):
    """Solve using Gurobi MIP."""
    import gurobipy as gp
    from gurobipy import GRB

    N = len(families)
    if N == 0:
        return {"objective_value": 0.0, "families_selected": [], "items_selected": []}

    model = gp.Model("setup_knapsack")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', max(1, time_limit))
    model.setParam('OutputFlag', 0)

    y = {}
    x = {}

    for i in range(N):
        fam = families[i]
        fid = fam['family_id']
        y[fid] = model.addVar(vtype=GRB.BINARY, name=f"y_{fid}")
        for j in range(fam['n_i']):
            item = fam['items'][j]
            iid = item['item_id']
            x[(fid, iid)] = model.addVar(vtype=GRB.BINARY, name=f"x_{fid}_{iid}")

    model.update()

    # Objective
    obj = gp.LinExpr()
    for i in range(N):
        fam = families[i]
        fid = fam['family_id']
        obj -= fam['setup_cost'] * y[fid]
        for j in range(fam['n_i']):
            item = fam['items'][j]
            iid = item['item_id']
            obj += item['profit'] * x[(fid, iid)]

    model.setObjective(obj, GRB.MAXIMIZE)

    # Capacity constraint
    cap_expr = gp.LinExpr()
    for i in range(N):
        fam = families[i]
        fid = fam['family_id']
        cap_expr += fam['setup_capacity'] * y[fid]
        for j in range(fam['n_i']):
            item = fam['items'][j]
            iid = item['item_id']
            cap_expr += item['weight'] * x[(fid, iid)]

    model.addConstr(cap_expr <= capacity, name="capacity")

    # Linking constraints
    for i in range(N):
        fam = families[i]
        fid = fam['family_id']
        for j in range(fam['n_i']):
            item = fam['items'][j]
            iid = item['item_id']
            model.addConstr(x[(fid, iid)] <= y[fid], name=f"link_{fid}_{iid}")

    model.update()

    # Greedy MIP start
    try:
        greedy_sol = greedy_heuristic(families, capacity)
        if greedy_sol:
            for i in range(N):
                fam = families[i]
                fid = fam['family_id']
                y[fid].Start = 1.0 if fid in greedy_sol['families_selected'] else 0.0
                for j in range(fam['n_i']):
                    item = fam['items'][j]
                    iid = item['item_id']
                    is_selected = any(
                        s['family'] == fid and s['item'] == iid
                        for s in greedy_sol['items_selected']
                    )
                    x[(fid, iid)].Start = 1.0 if is_selected else 0.0
    except Exception:
        pass

    # Callback for logging incumbents
    best_logged = [float('-inf')]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val > best_logged[0]:
                best_logged[0] = obj_val
                fam_sel = []
                item_sel = []
                for i in range(N):
                    fam = families[i]
                    fid = fam['family_id']
                    if model.cbGetSolution(y[fid]) > 0.5:
                        fam_sel.append(fid)
                    for j in range(fam['n_i']):
                        item = fam['items'][j]
                        iid = item['item_id']
                        if model.cbGetSolution(x[(fid, iid)]) > 0.5:
                            item_sel.append({"family": fid, "item": iid})

                sol = {
                    "objective_value": obj_val,
                    "families_selected": fam_sel,
                    "items_selected": item_sel
                }
                if logger:
                    logger.log_solution(obj_val, sol)

    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        best_obj = model.ObjVal
        families_selected = []
        items_selected = []

        for i in range(N):
            fam = families[i]
            fid = fam['family_id']
            if y[fid].X > 0.5:
                families_selected.append(fid)
            for j in range(fam['n_i']):
                item = fam['items'][j]
                iid = item['item_id']
                if x[(fid, iid)].X > 0.5:
                    items_selected.append({"family": fid, "item": iid})

        return {
            "objective_value": float(best_obj),
            "families_selected": families_selected,
            "items_selected": items_selected
        }

    return None


def greedy_heuristic(families, capacity):
    """Compute a greedy initial solution based on profit/weight ratio."""
    # For each family, compute best greedy subset and efficiency
    family_scores = []
    for fam in families:
        fid = fam['family_id']
        setup_cap = fam['setup_capacity']
        setup_cost = fam['setup_cost']
        items = fam['items']

        if setup_cap > capacity:
            continue

        # Greedy: sort items by profit/weight ratio, add until capacity exhausted
        remaining_cap = capacity - setup_cap
        sorted_items = sorted(items, key=lambda it: it['profit'] / max(it['weight'], 1), reverse=True)

        selected = []
        total_profit = 0
        total_weight = 0
        for item in sorted_items:
            if total_weight + item['weight'] <= remaining_cap:
                selected.append(item)
                total_profit += item['profit']
                total_weight += item['weight']

        net_profit = total_profit - setup_cost
        total_cap_used = setup_cap + total_weight

        if net_profit > 0 and total_cap_used > 0:
            efficiency = net_profit / total_cap_used
            family_scores.append((efficiency, fid, selected, net_profit, total_cap_used))

    # Sort families by efficiency and greedily add
    family_scores.sort(key=lambda x: -x[0])

    remaining = capacity
    families_selected = []
    items_selected = []
    total_obj = 0

    for eff, fid, selected, net_profit, cap_used in family_scores:
        if cap_used <= remaining:
            families_selected.append(fid)
            for item in selected:
                items_selected.append({"family": fid, "item": item['item_id']})
            remaining -= cap_used
            total_obj += net_profit

    if total_obj > 0:
        return {
            "objective_value": total_obj,
            "families_selected": families_selected,
            "items_selected": items_selected
        }
    return None


if __name__ == '__main__':
    main()