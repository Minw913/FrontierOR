import json
import argparse
import time
import math
import sys


def main():
    parser = argparse.ArgumentParser(description='Cutting Stock Problem Solver')
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

    W = instance['bin_capacity']
    items = instance['items']

    widths = {}
    demands = {}
    for item in items:
        widths[item['type_id']] = item['width']
        demands[item['type_id']] = item['demand']

    type_ids = sorted(widths.keys())
    total_demand = sum(demands.values())

    # Edge case: no items needed
    if total_demand == 0:
        solution = {"objective_value": 0, "model_variables": {}}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log_solution(0, solution)
        return

    # Lower bound
    total_width = sum(widths[t] * demands[t] for t in type_ids)
    lower_bound = math.ceil(total_width / W)

    def bins_to_arc_flow(bins):
        """Convert list of bins (each a list of (width, type_id)) to arc-flow variables."""
        model_vars = {}
        for bin_items in bins:
            pos = 0
            for w, t in bin_items:
                var_name = f"x_{t}_{pos}"
                model_vars[var_name] = model_vars.get(var_name, 0) + 1
                pos += w
            # Loss arcs from pos to W
            for j in range(pos, W):
                var_name = f"loss_{j}"
                model_vars[var_name] = model_vars.get(var_name, 0) + 1
        return len(bins), model_vars

    def first_fit_decreasing():
        item_list = []
        for t in type_ids:
            for _ in range(demands[t]):
                item_list.append((widths[t], t))
        item_list.sort(key=lambda x: (-x[0], x[1]))

        bins = []
        bin_remaining = []

        for w, t in item_list:
            placed = False
            for i in range(len(bins)):
                if bin_remaining[i] >= w:
                    bins[i].append((w, t))
                    bin_remaining[i] -= w
                    placed = True
                    break
            if not placed:
                bins.append([(w, t)])
                bin_remaining.append(W - w)
        return bins

    def best_fit_decreasing():
        item_list = []
        for t in type_ids:
            for _ in range(demands[t]):
                item_list.append((widths[t], t))
        item_list.sort(key=lambda x: (-x[0], x[1]))

        bins = []
        bin_remaining = []

        for w, t in item_list:
            best_idx = -1
            best_rem = W + 1
            for i in range(len(bins)):
                rem = bin_remaining[i] - w
                if rem >= 0 and rem < best_rem:
                    best_rem = rem
                    best_idx = i
            if best_idx >= 0:
                bins[best_idx].append((w, t))
                bin_remaining[best_idx] -= w
            else:
                bins.append([(w, t)])
                bin_remaining.append(W - w)
        return bins

    def worst_fit_decreasing():
        item_list = []
        for t in type_ids:
            for _ in range(demands[t]):
                item_list.append((widths[t], t))
        item_list.sort(key=lambda x: (-x[0], x[1]))

        bins = []
        bin_remaining = []

        for w, t in item_list:
            worst_idx = -1
            worst_rem = -1
            for i in range(len(bins)):
                rem = bin_remaining[i] - w
                if rem >= 0 and rem > worst_rem:
                    worst_rem = rem
                    worst_idx = i
            if worst_idx >= 0:
                bins[worst_idx].append((w, t))
                bin_remaining[worst_idx] -= w
            else:
                bins.append([(w, t)])
                bin_remaining.append(W - w)
        return bins

    # Run heuristics
    best_obj = float('inf')
    best_vars = {}

    for heuristic_fn in [first_fit_decreasing, best_fit_decreasing, worst_fit_decreasing]:
        bins = heuristic_fn()
        obj, mvars = bins_to_arc_flow(bins)
        if obj < best_obj:
            best_obj = obj
            best_vars = mvars
            if logger:
                logger.log_solution(best_obj, {
                    "objective_value": best_obj,
                    "model_variables": best_vars
                })

    # If heuristic achieves lower bound, we're optimal
    if best_obj == lower_bound:
        solution = {"objective_value": best_obj, "model_variables": best_vars}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    # Try arc-flow MIP with Gurobi
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed

    num_item_arcs = sum(W - widths[t] + 1 for t in type_ids)
    total_vars_est = num_item_arcs + W

    if total_vars_est <= 300000 and remaining > 3:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            model = gp.Model("cutting_stock_arc_flow")
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(1.0, remaining - 2.0)
            model.Params.OutputFlag = 0

            # Create variables
            x = {}
            for t in type_ids:
                w = widths[t]
                for j in range(W - w + 1):
                    x[t, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{t}_{j}")

            loss = {}
            for j in range(W):
                loss[j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"loss_{j}")

            model.update()

            # Objective: minimize total flow leaving node 0
            obj_expr = gp.quicksum(x[t, 0] for t in type_ids) + loss[0]
            model.setObjective(obj_expr, GRB.MINIMIZE)

            # Flow conservation at nodes 1 to W-1
            for j in range(1, W):
                inflow = gp.LinExpr()
                for t in type_ids:
                    w = widths[t]
                    if j - w >= 0:
                        inflow += x[t, j - w]
                inflow += loss[j - 1]

                outflow = gp.LinExpr()
                for t in type_ids:
                    w = widths[t]
                    if j + w <= W:
                        outflow += x[t, j]
                if j < W:
                    outflow += loss[j]

                model.addConstr(inflow == outflow, name=f"fc_{j}")

            # Demand constraints
            for t in type_ids:
                w = widths[t]
                model.addConstr(
                    gp.quicksum(x[t, j] for j in range(W - w + 1)) >= demands[t],
                    name=f"dem_{t}"
                )

            # Warm start from best heuristic
            for key, val in best_vars.items():
                try:
                    if key.startswith("x_"):
                        parts = key.split("_")
                        t_id = int(parts[1])
                        j_pos = int(parts[2])
                        if (t_id, j_pos) in x:
                            x[t_id, j_pos].Start = val
                    elif key.startswith("loss_"):
                        j_pos = int(key.split("_")[1])
                        if j_pos in loss:
                            loss[j_pos].Start = val
                except (ValueError, IndexError):
                    pass

            model.optimize()

            if model.SolCount > 0:
                mip_obj = int(round(model.ObjVal))
                if mip_obj < best_obj:
                    model_vars = {}
                    for t in type_ids:
                        w = widths[t]
                        for j in range(W - w + 1):
                            val = x[t, j].X
                            if val > 0.5:
                                model_vars[f"x_{t}_{j}"] = int(round(val))
                    for j in range(W):
                        val = loss[j].X
                        if val > 0.5:
                            model_vars[f"loss_{j}"] = int(round(val))

                    best_obj = mip_obj
                    best_vars = model_vars

                    if logger:
                        logger.log_solution(best_obj, {
                            "objective_value": best_obj,
                            "model_variables": best_vars
                        })

        except Exception as e:
            # Fall back to heuristic solution
            pass

    # Write final solution
    solution = {
        "objective_value": best_obj,
        "model_variables": best_vars
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()