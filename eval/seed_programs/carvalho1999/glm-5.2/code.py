import json
import argparse
import time
import sys
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    with open(args.instance_path) as f:
        instance = json.load(f)

    W = instance['bin_capacity']
    items = instance['items']
    n = len(items)

    start_time = time.time()

    def make_solution(bins):
        bin_assignments = []
        for b in bins:
            bin_assignments.append({'items': list(b), 'total_size': sum(b)})
        return {
            'objective_value': len(bin_assignments),
            'bin_assignments': bin_assignments,
            'num_bins': len(bin_assignments)
        }

    def write_solution(path, bins):
        sol = make_solution(bins)
        with open(path, 'w') as f:
            json.dump(sol, f, indent=2)

    if n == 0:
        sol = {'objective_value': 0, 'bin_assignments': [], 'num_bins': 0}
        if logger:
            logger.log_solution(0, sol)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f, indent=2)
        return

    sorted_items = sorted(items, reverse=True)

    # FFD
    ffd_bins = []
    ffd_loads = []
    for item in sorted_items:
        placed = False
        for i in range(len(ffd_bins)):
            if ffd_loads[i] + item <= W:
                ffd_bins[i].append(item)
                ffd_loads[i] += item
                placed = True
                break
        if not placed:
            ffd_bins.append([item])
            ffd_loads.append(item)

    best_bins = ffd_bins
    best_obj = len(best_bins)
    if logger:
        logger.log_solution(best_obj, make_solution(best_bins))

    # BFD
    bfd_bins = []
    bfd_loads = []
    for item in sorted_items:
        best_bin = -1
        best_space = W + 1
        for i in range(len(bfd_bins)):
            space = W - bfd_loads[i]
            if item <= space and space - item < best_space:
                best_space = space - item
                best_bin = i
        if best_bin >= 0:
            bfd_bins[best_bin].append(item)
            bfd_loads[best_bin] += item
        else:
            bfd_bins.append([item])
            bfd_loads.append(item)
    if len(bfd_bins) < best_obj:
        best_obj = len(bfd_bins)
        best_bins = bfd_bins
        if logger:
            logger.log_solution(best_obj, make_solution(best_bins))

    # Local search: merge bins
    def merge_bins(bins, loads):
        bins = [list(b) for b in bins]
        loads = list(loads)
        merged = True
        while merged:
            merged = False
            for i in range(len(bins)):
                for j in range(i + 1, len(bins)):
                    if loads[i] + loads[j] <= W:
                        bins[i].extend(bins[j])
                        loads[i] += loads[j]
                        bins.pop(j)
                        loads.pop(j)
                        merged = True
                        break
                if merged:
                    break
        return bins, loads

    best_bins, best_loads = merge_bins(best_bins, [sum(b) for b in best_bins])
    if len(best_bins) < best_obj:
        best_obj = len(best_bins)
        if logger:
            logger.log_solution(best_obj, make_solution(best_bins))

    # Local search: move items to empty bins
    def move_items(bins, loads):
        bins = [list(b) for b in bins]
        loads = list(loads)
        improved = True
        iterations = 0
        while improved and iterations < 100:
            improved = False
            iterations += 1
            order = sorted(range(len(bins)), key=lambda i: loads[i])
            for i in order:
                if loads[i] == 0:
                    continue
                for item in list(bins[i]):
                    for j in range(len(bins)):
                        if i == j:
                            continue
                        if loads[j] + item <= W:
                            bins[i].remove(item)
                            loads[i] -= item
                            bins[j].append(item)
                            loads[j] += item
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    new_bins = []
                    new_loads = []
                    for b, l in zip(bins, loads):
                        if l > 0:
                            new_bins.append(b)
                            new_loads.append(l)
                    bins = new_bins
                    loads = new_loads
                    break
        return bins, loads

    best_bins, best_loads = move_items(best_bins, best_loads)
    if len(best_bins) < best_obj:
        best_obj = len(best_bins)
        if logger:
            logger.log_solution(best_obj, make_solution(best_bins))

    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time < 2:
        write_solution(args.solution_path, best_bins)
        return

    # Try MIP approach
    try:
        import gurobipy as gp
        from gurobipy import GRB

        item_counts = Counter(items)
        sizes = list(item_counts.keys())
        demands = [item_counts[s] for s in sizes]
        m = len(sizes)
        size_to_idx = {s: i for i, s in enumerate(sizes)}

        # Compact MIP for small n
        if n <= 200:
            K = best_obj
            model = gp.Model("bin_packing")
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = remaining_time
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Seed = 0

            x = model.addVars(n, K, vtype=GRB.BINARY, name="x")
            y = model.addVars(K, vtype=GRB.BINARY, name="y")

            model.addConstrs(x[i, k] <= y[k] for i in range(n) for k in range(K))
            for k in range(K):
                model.addConstr(gp.quicksum(items[i] * x[i, k] for i in range(n)) <= W)
            model.addConstrs(gp.quicksum(x[i, k] for k in range(K)) == 1 for i in range(n))
            for k in range(K - 1):
                model.addConstr(y[k] >= y[k + 1])
            model.setObjective(gp.quicksum(y[k] for k in range(K)), GRB.MINIMIZE)

            model.optimize()
            if model.SolCount > 0:
                mip_obj = round(model.ObjVal)
                if mip_obj < best_obj:
                    new_bins = [[] for _ in range(mip_obj)]
                    for i in range(n):
                        for k in range(K):
                            if x[i, k].X > 0.5:
                                if k < len(new_bins):
                                    new_bins[k].append(items[i])
                                break
                    new_bins = [b for b in new_bins if b]
                    if len(new_bins) == mip_obj and sum(len(b) for b in new_bins) == n:
                        best_obj = mip_obj
                        best_bins = new_bins
                        if logger:
                            logger.log_solution(best_obj, make_solution(best_bins))
        elif m <= 1000:
            # Column generation
            patterns = []
            seen = set()

            # Initial single-type patterns
            for i in range(m):
                max_fit = min(demands[i], W // sizes[i])
                if max_fit > 0:
                    pat = [0] * m
                    pat[i] = max_fit
                    t = tuple(pat)
                    if t not in seen:
                        seen.add(t)
                        patterns.append(pat)

            # Heuristic patterns
            for b in best_bins:
                pat = [0] * m
                for item in b:
                    pat[size_to_idx[item]] += 1
                t = tuple(pat)
                if t not in seen:
                    seen.add(t)
                    patterns.append(pat)

            cg_time_limit = remaining_time * 0.7
            cg_start = time.time()

            while True:
                elapsed = time.time() - cg_start
                if elapsed > cg_time_limit or (args.time_limit - (time.time() - start_time)) < 1:
                    break

                # Master LP
                lp_model = gp.Model("master_lp")
                lp_model.Params.OutputFlag = 0
                lp_model.Params.Threads = 1
                lp_model.Params.TimeLimit = max(1, (args.time_limit - (time.time() - start_time)) * 0.5)
                lam = lp_model.addVars(len(patterns), name="lam", lb=0)
                cons = lp_model.addConstrs(
                    gp.quicksum(patterns[j][i] * lam[j] for j in range(len(patterns))) == demands[i]
                    for i in range(m)
                )
                lp_model.setObjective(gp.quicksum(lam[j] for j in range(len(patterns))), GRB.MINIMIZE)
                lp_model.optimize()
                if lp_model.status != GRB.OPTIMAL:
                    break
                duals = [cons[i].Pi for i in range(m)]

                # Subproblem (knapsack)
                sub_model = gp.Model("subproblem")
                sub_model.Params.OutputFlag = 0
                sub_model.Params.Threads = 1
                sub_model.Params.TimeLimit = max(1, (args.time_limit - (time.time() - start_time)) * 0.5)
                z = sub_model.addVars(m, vtype=GRB.INTEGER, name="z", lb=0)
                for i in range(m):
                    z[i].ub = min(demands[i], W // sizes[i]) if sizes[i] > 0 else 0
                sub_model.addConstr(gp.quicksum(sizes[i] * z[i] for i in range(m)) <= W)
                sub_model.setObjective(gp.quicksum(duals[i] * z[i] for i in range(m)), GRB.MAXIMIZE)
                sub_model.optimize()
                if sub_model.status != GRB.OPTIMAL:
                    break
                pattern = [int(z[i].X) for i in range(m)]
                val = sum(duals[i] * pattern[i] for i in range(m))
                if val <= 1 + 1e-5:
                    break
                t = tuple(pattern)
                if t not in seen:
                    seen.add(t)
                    patterns.append(pattern)
                else:
                    break

            # Restricted master IP
            ip_time_limit = max(1, args.time_limit - (time.time() - start_time))
            ip_model = gp.Model("master_ip")
            ip_model.Params.OutputFlag = 0
            ip_model.Params.Threads = 1
            ip_model.Params.TimeLimit = ip_time_limit
            ip_model.Params.MIPGap = 1e-4
            ip_model.Params.NumericFocus = 0
            ip_model.Params.Seed = 0
            lam = ip_model.addVars(len(patterns), vtype=GRB.INTEGER, name="lam", lb=0)
            ip_model.addConstrs(
                gp.quicksum(patterns[j][i] * lam[j] for j in range(len(patterns))) == demands[i]
                for i in range(m)
            )
            ip_model.setObjective(gp.quicksum(lam[j] for j in range(len(patterns))), GRB.MINIMIZE)
            ip_model.optimize()

            if ip_model.SolCount > 0:
                ip_obj = round(ip_model.ObjVal)
                if ip_obj < best_obj:
                    new_bins = []
                    for j in range(len(patterns)):
                        if lam[j].X > 0.5:
                            count = round(lam[j].X)
                            pat = patterns[j]
                            for _ in range(count):
                                bin_items = []
                                for i in range(m):
                                    bin_items.extend([sizes[i]] * pat[i])
                                new_bins.append(bin_items)
                    if sum(len(b) for b in new_bins) == n:
                        best_obj = ip_obj
                        best_bins = new_bins
                        if logger:
                            logger.log_solution(best_obj, make_solution(best_bins))
    except Exception:
        pass

    write_solution(args.solution_path, best_bins)


if __name__ == "__main__":
    main()