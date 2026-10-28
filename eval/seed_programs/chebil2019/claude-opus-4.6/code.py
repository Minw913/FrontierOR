import argparse
import json
import time
import sys
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)

    N = data['N']
    capacity = data['knapsack_capacity']
    families = data['families']

    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    start_time = time.time()

    # Try to use Gurobi MIP solver
    try:
        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("knapsack_family")
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, time_limit - 2))

        # Decision variables
        # y[f] = 1 if family f is activated
        y = {}
        # x[f][i] = 1 if item i of family f is selected
        x = {}

        for fam in families:
            fid = fam['family_id']
            y[fid] = model.addVar(vtype=GRB.BINARY, name=f"y_{fid}")
            x[fid] = {}
            for item in fam['items']:
                iid = item['item_id']
                x[fid][iid] = model.addVar(vtype=GRB.BINARY, name=f"x_{fid}_{iid}")

        model.update()

        # Objective: maximize sum of item profits - sum of family setup costs
        obj = gp.LinExpr()
        for fam in families:
            fid = fam['family_id']
            obj -= fam['setup_cost'] * y[fid]
            for item in fam['items']:
                iid = item['item_id']
                obj += item['profit'] * x[fid][iid]

        model.setObjective(obj, GRB.MAXIMIZE)

        # Capacity constraint
        cap_expr = gp.LinExpr()
        for fam in families:
            fid = fam['family_id']
            cap_expr += fam['setup_capacity'] * y[fid]
            for item in fam['items']:
                iid = item['item_id']
                cap_expr += item['weight'] * x[fid][iid]

        model.addConstr(cap_expr <= capacity, "capacity")

        # Linking constraints: item can only be selected if family is activated
        for fam in families:
            fid = fam['family_id']
            for item in fam['items']:
                iid = item['item_id']
                model.addConstr(x[fid][iid] <= y[fid], f"link_{fid}_{iid}")

        # Callback for logging
        best_obj = [float('-inf')]

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val > best_obj[0] + 1e-6:
                    best_obj[0] = obj_val
                    if logger:
                        logger.log(obj_val)

        model.optimize(callback)

        # Extract solution
        if model.SolCount > 0:
            objective_value = model.ObjVal
            families_selected = []
            items_selected = []

            for fam in families:
                fid = fam['family_id']
                if y[fid].X > 0.5:
                    families_selected.append(fid)
                    for item in fam['items']:
                        iid = item['item_id']
                        if x[fid][iid].X > 0.5:
                            items_selected.append({"family": fid, "item": iid})

            solution = {
                "objective_value": objective_value,
                "families_selected": families_selected,
                "items_selected": items_selected
            }
        else:
            # No feasible solution found, return empty
            solution = {
                "objective_value": 0.0,
                "families_selected": [],
                "items_selected": []
            }
            if logger:
                logger.log(0.0)

    except Exception as e:
        # Fallback: greedy approach
        print(f"Gurobi failed: {e}, using greedy fallback", file=sys.stderr)
        solution = greedy_solve(families, capacity, logger)

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def greedy_solve(families, capacity, logger):
    # For each family, compute the best items by profit/weight ratio
    # Then greedily select families and items
    
    best_obj = 0.0
    best_solution = {
        "objective_value": 0.0,
        "families_selected": [],
        "items_selected": []
    }

    # Evaluate each family independently first
    family_values = []
    for fam in families:
        fid = fam['family_id']
        setup_cost = fam['setup_cost']
        setup_cap = fam['setup_capacity']
        
        # Sort items by profit/weight ratio (descending)
        items_sorted = sorted(fam['items'], 
                            key=lambda it: it['profit'] / max(it['weight'], 1e-9), 
                            reverse=True)
        
        # Greedy selection within family given remaining capacity
        family_values.append((fid, setup_cost, setup_cap, items_sorted))

    # Try all subsets? Too expensive. Use greedy by family value density.
    # For each family, estimate net value
    family_estimates = []
    for fid, setup_cost, setup_cap, items_sorted in family_values:
        total_profit = 0
        total_weight = setup_cap
        for item in items_sorted:
            if total_weight + item['weight'] <= capacity:
                total_profit += item['profit']
                total_weight += item['weight']
        net = total_profit - setup_cost
        if net > 0 and setup_cap <= capacity:
            density = net / max(total_weight, 1)
            family_estimates.append((density, fid, setup_cost, setup_cap, items_sorted))

    family_estimates.sort(reverse=True)

    remaining_cap = capacity
    families_selected = []
    items_selected = []
    total_obj = 0.0

    for _, fid, setup_cost, setup_cap, items_sorted in family_estimates:
        if setup_cap > remaining_cap:
            continue
        
        # Try adding this family
        test_cap = remaining_cap - setup_cap
        selected = []
        for item in items_sorted:
            if item['weight'] <= test_cap:
                selected.append(item)
                test_cap -= item['weight']
        
        net_gain = sum(it['profit'] for it in selected) - setup_cost
        if net_gain > 0:
            remaining_cap -= setup_cap
            families_selected.append(fid)
            total_obj += -setup_cost
            for item in selected:
                remaining_cap -= item['weight']
                total_obj += item['profit']
                items_selected.append({"family": fid, "item": item['item_id']})

    if total_obj > best_obj:
        best_obj = total_obj
        best_solution = {
            "objective_value": best_obj,
            "families_selected": families_selected,
            "items_selected": items_selected
        }

    if logger:
        logger.log(best_obj)

    return best_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)