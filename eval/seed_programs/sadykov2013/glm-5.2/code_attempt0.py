import argparse
import json
import time
import math
import sys
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True, type=str)
    parser.add_argument('--solution_path', required=True, type=str)
    parser.add_argument('--time_limit', required=True, type=int)
    parser.add_argument('--log_path', default=None, type=str)
    args = parser.parse_args()

    start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path) as f:
        data = json.load(f)

    num_items = data['num_items']
    C = data['bin_capacity']
    items = data['items']
    conflict_edges = data['conflict_edges']

    id_to_idx = {item['id']: i for i, item in enumerate(items)}
    weights = [item['weight'] for item in items]
    n = num_items

    # Build conflict adjacency
    conflicts = [set() for _ in range(n)]
    for u, v in conflict_edges:
        i = id_to_idx[u]
        j = id_to_idx[v]
        conflicts[i].add(j)
        conflicts[j].add(i)

    total_weight = sum(weights)
    if C == 0:
        LB = 1
    else:
        LB = max(1, math.ceil(total_weight / C))

    def time_left():
        return args.time_limit - (time.time() - start)

    # Heuristic: Best Fit Decreasing with conflict check
    # Sort by weight descending, then degree descending
    order = list(range(n))
    order.sort(key=lambda i: (weights[i], len(conflicts[i])), reverse=True)

    bins = []  # list of dicts: {'items': set, 'weight': int}
    for i in order:
        w = weights[i]
        best_bin = -1
        best_rem = C + 1
        for b_idx, b in enumerate(bins):
            if b['weight'] + w <= C:
                if not (conflicts[i] & b['items']):
                    rem = C - (b['weight'] + w)
                    if rem < best_rem:
                        best_rem = rem
                        best_bin = b_idx
        if best_bin != -1:
            bins[best_bin]['items'].add(i)
            bins[best_bin]['weight'] += w
        else:
            bins.append({'items': {i}, 'weight': w})

    def build_solution(bins_list):
        sol_bins = {}
        for idx, b in enumerate(bins_list):
            sol_bins[str(idx)] = {
                'items': [items[i]['id'] for i in b['items']],
                'total_weight': b['weight']
            }
        return {'objective_value': len(bins_list), 'bins': sol_bins}

    if logger:
        logger.log_solution(len(bins), build_solution(bins))

    # Lower bound check
    if len(bins) == LB:
        sol = build_solution(bins)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        if logger:
            logger.log_solution(sol['objective_value'], sol)
        return

    # Improvement: merging and bin elimination
    improved = True
    while improved and time_left() > 2:
        improved = False

        # Merge bins
        i = 0
        while i < len(bins) and time_left() > 2:
            j = i + 1
            merged = False
            while j < len(bins) and time_left() > 2:
                if bins[i]['weight'] + bins[j]['weight'] <= C:
                    conflict = False
                    for item_i in bins[i]['items']:
                        if conflicts[item_i] & bins[j]['items']:
                            conflict = True
                            break
                    if not conflict:
                        bins[i]['items'].update(bins[j]['items'])
                        bins[i]['weight'] += bins[j]['weight']
                        bins.pop(j)
                        improved = True
                        merged = True
                        break
                j += 1
            if not merged:
                i += 1

        if improved:
            if logger:
                logger.log_solution(len(bins), build_solution(bins))
            if len(bins) == LB:
                sol = build_solution(bins)
                with open(args.solution_path, 'w') as f:
                    json.dump(sol, f, indent=2)
                return
            continue

        # Eliminate bins by moving their items
        bins.sort(key=lambda b: len(b['items']))
        for b_idx in range(len(bins)):
            if len(bins[b_idx]) == 0:
                continue
            if time_left() < 2:
                break

            bin_items = list(bins[b_idx]['items'])
            bin_items.sort(key=lambda i: weights[i], reverse=True)
            other_bins = [k for k in range(len(bins)) if k != b_idx]
            other_bins.sort(key=lambda k: C - bins[k]['weight'])  # tightest first

            # Greedy placement
            temp_w = {k: bins[k]['weight'] for k in other_bins}
            temp_it = {k: set(bins[k]['items']) for k in other_bins}
            assignment = []
            feasible = True
            for item in bin_items:
                w = weights[item]
                placed = False
                for k in other_bins:
                    if temp_w[k] + w <= C and not (conflicts[item] & temp_it[k]):
                        temp_w[k] += w
                        temp_it[k].add(item)
                        assignment.append(k)
                        placed = True
                        break
                if not placed:
                    feasible = False
                    break

            if feasible:
                for item, k in zip(bin_items, assignment):
                    bins[k]['items'].add(item)
                    bins[k]['weight'] += weights[item]
                bins.pop(b_idx)
                improved = True
                break

            # Backtracking if few items and time allows
            if len(bin_items) <= 10 and time_left() > 2:
                temp_w = {k: bins[k]['weight'] for k in other_bins}
                temp_it = {k: set(bins[k]['items']) for k in other_bins}

                def backtrack(idx):
                    if idx == len(bin_items):
                        return True
                    item = bin_items[idx]
                    w = weights[item]
                    for k in other_bins:
                        if temp_w[k] + w <= C and not (conflicts[item] & temp_it[k]):
                            temp_w[k] += w
                            temp_it[k].add(item)
                            if backtrack(idx + 1):
                                return True
                            temp_w[k] -= w
                            temp_it[k].remove(item)
                    return False

                if backtrack(0):
                    for item in bin_items:
                        for k in other_bins:
                            if item in temp_it[k]:
                                bins[k]['items'].add(item)
                                bins[k]['weight'] += weights[item]
                                break
                    bins.pop(b_idx)
                    improved = True
                    break

        if improved:
            if logger:
                logger.log_solution(len(bins), build_solution(bins))
            if len(bins) == LB:
                sol = build_solution(bins)
                with open(args.solution_path, 'w') as f:
                    json.dump(sol, f, indent=2)
                return
            continue

    # Gurobi phase
    remaining = time_left()
    UB = len(bins)
    best_obj = UB

    if remaining > 5 and n * UB <= 50000:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            env = gp.Env(empty=True)
            env.setParam('Seed', 0)
            env.setParam('MIPGap', 1e-4)
            env.setParam('NumericFocus', 0)
            env.setParam('Threads', 1)
            env.start()

            model = gp.Model(env=env)
            model.setParam('TimeLimit', max(1.0, remaining - 1.0))
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)

            x = {}
            for i in range(n):
                for b in range(UB):
                    x[i, b] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{b}')
            y = {}
            for b in range(UB):
                y[b] = model.addVar(vtype=GRB.BINARY, name=f'y_{b}')

            model.setObjective(gp.quicksum(y[b] for b in range(UB)), GRB.MINIMIZE)

            for i in range(n):
                model.addConstr(gp.quicksum(x[i, b] for b in range(UB)) == 1)

            for b in range(UB):
                model.addConstr(gp.quicksum(weights[i] * x[i, b] for i in range(n)) <= C * y[b])
                for i in range(n):
                    model.addConstr(x[i, b] <= y[b])

            for i in range(n):
                for j in conflicts[i]:
                    if i < j:
                        for b in range(UB):
                            model.addConstr(x[i, b] + x[j, b] <= 1)

            # MIP start
            for b_idx, b in enumerate(bins):
                for i in b['items']:
                    x[i, b_idx].start = 1
                y[b_idx].start = 1
            for b in range(UB):
                if y[b].start == GRB.UNDEFINED:
                    y[b].start = 0
            for i in range(n):
                for b in range(UB):
                    if x[i, b].start == GRB.UNDEFINED:
                        x[i, b].start = 0

            model.optimize()

            if model.SolCount > 0:
                sol_bins = {}
                obj = 0
                for b in range(UB):
                    if y[b].X > 0.5:
                        bin_items = []
                        total_w = 0
                        for i in range(n):
                            if x[i, b].X > 0.5:
                                bin_items.append(items[i]['id'])
                                total_w += weights[i]
                        if bin_items:
                            sol_bins[str(obj)] = {
                                'items': bin_items,
                                'total_weight': total_w
                            }
                            obj += 1
                if obj < best_obj:
                    if logger:
                        logger.log_solution(obj, {'objective_value': obj, 'bins': sol_bins})
                    with open(args.solution_path, 'w') as f:
                        json.dump({'objective_value': obj, 'bins': sol_bins}, f, indent=2)
                    return
        except Exception:
            pass

    # Fallback to heuristic solution
    sol = build_solution(bins)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)
    if logger:
        logger.log_solution(sol['objective_value'], sol)

if __name__ == '__main__':
    main()