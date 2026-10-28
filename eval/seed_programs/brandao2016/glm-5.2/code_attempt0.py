import argparse
import json
import time
from solution_logger import SolutionLogger


def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    W = instance['parameters']['W']
    items_data = instance['items']

    weights = {}
    demands = {}
    item_types = []
    for item in items_data:
        it = item['item_type']
        weights[it] = item['weight']
        demands[it] = item['demand']
        item_types.append(it)

    def build_solution_dict(bins_list):
        return {
            "objective_value": float(len(bins_list)),
            "num_bins": len(bins_list),
            "bins": bins_list
        }

    total_weight = sum(weights[it] * demands[it] for it in item_types)
    lb = max(1, -(-total_weight // W))

    # Efficient batch first-fit decreasing
    def ffd_batch(sorted_types):
        bins_cap = []
        bin_items = []
        for it in sorted_types:
            w = weights[it]
            d = demands[it]
            remaining = d
            for i in range(len(bins_cap)):
                if remaining == 0:
                    break
                fit = min(remaining, bins_cap[i] // w)
                if fit > 0:
                    bins_cap[i] -= fit * w
                    bin_items[i].extend([it] * fit)
                    remaining -= fit
            while remaining > 0:
                fit = min(remaining, W // w)
                bins_cap.append(W - fit * w)
                bin_items.append([it] * fit)
                remaining -= fit
        return bin_items

    # Item-wise best-fit decreasing (for smaller instances)
    def bfd_itemwise(sorted_types):
        all_items = []
        for it in sorted_types:
            for _ in range(demands[it]):
                all_items.append(it)
        bins = []
        for it in all_items:
            best_idx = -1
            best_remaining = W + 1
            for i in range(len(bins)):
                remaining = W - bins[i][0]
                if weights[it] <= remaining < best_remaining:
                    best_idx = i
                    best_remaining = remaining
            if best_idx >= 0:
                bins[best_idx][0] += weights[it]
                bins[best_idx][1].append(it)
            else:
                bins.append([weights[it], [it]])
        return [b[1] for b in bins]

    sorted_by_weight = sorted(item_types, key=lambda t: weights[t], reverse=True)

    best_bins = ffd_batch(sorted_by_weight)
    best_obj = len(best_bins)

    if logger:
        logger.log_solution(best_obj, build_solution_dict(best_bins))

    # Try BFD if instance is not too large
    total_n = sum(demands[it] for it in item_types)
    if total_n <= 100000:
        bfd_bins = bfd_itemwise(sorted_by_weight)
        if len(bfd_bins) < best_obj:
            best_bins = bfd_bins
            best_obj = len(bfd_bins)
            if logger:
                logger.log_solution(best_obj, build_solution_dict(best_bins))

    # Try MIP if not already optimal and time permits
    if best_obj > lb:
        remaining_time = args.time_limit - (time.time() - start_time)
        if remaining_time > 3 and len(item_types) * best_obj <= 80000 and best_obj <= 3000:
            try:
                import gurobipy as gp
                from gurobipy import GRB

                B_max = best_obj

                env = gp.Env(empty=True)
                env.setParam('Seed', 0)
                env.setParam('Threads', 1)
                env.start()

                model = gp.Model(env=env)
                model.setParam('MIPGap', 1e-4)
                model.setParam('NumericFocus', 0)
                model.setParam('TimeLimit', max(1, remaining_time - 2))

                y = [model.addVar(vtype=GRB.BINARY, name=f'y_{b}') for b in range(B_max)]
                x = {}
                for it in item_types:
                    for b in range(B_max):
                        x[(it, b)] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=demands[it], name=f'x_{it}_{b}')

                for b in range(B_max):
                    model.addConstr(
                        gp.quicksum(weights[it] * x[(it, b)] for it in item_types) <= W * y[b]
                    )

                for it in item_types:
                    model.addConstr(
                        gp.quicksum(x[(it, b)] for b in range(B_max)) >= demands[it]
                    )

                model.setObjective(gp.quicksum(y[b] for b in range(B_max)), GRB.MINIMIZE)

                # Symmetry breaking: bins used in order
                for b in range(B_max - 1):
                    model.addConstr(y[b] >= y[b + 1])

                # Warm start from heuristic
                for b in range(B_max):
                    y[b].Start = 1.0 if b < len(best_bins) else 0.0
                bin_counts = {}
                for b in range(len(best_bins)):
                    for it in best_bins[b]:
                        bin_counts[(it, b)] = bin_counts.get((it, b), 0) + 1
                for it in item_types:
                    for b in range(B_max):
                        x[(it, b)].Start = bin_counts.get((it, b), 0)

                model.optimize()

                if model.SolCount > 0:
                    obj_val = int(round(model.ObjVal))
                    if obj_val < best_obj:
                        sol_bins = []
                        for b in range(B_max):
                            if y[b].X > 0.5:
                                bin_items = []
                                for it in item_types:
                                    count = int(round(x[(it, b)].X))
                                    for _ in range(count):
                                        bin_items.append(it)
                                if bin_items:
                                    sol_bins.append(bin_items)
                        best_bins = sol_bins
                        best_obj = len(sol_bins)
                        if logger:
                            logger.log_solution(best_obj, build_solution_dict(best_bins))
            except Exception:
                pass

    sol_dict = build_solution_dict(best_bins)
    with open(args.solution_path, 'w') as f:
        json.dump(sol_dict, f)


if __name__ == '__main__':
    solve()