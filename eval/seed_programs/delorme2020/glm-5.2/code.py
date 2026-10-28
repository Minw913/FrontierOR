import argparse
import json
import time
import bisect
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def best_fit_decreasing(weights, demands, capacity):
    """Best Fit Decreasing heuristic. Returns list of bins (each a list of widths)."""
    items = []
    for w, d in zip(weights, demands):
        items.extend([w] * d)
    items.sort(reverse=True)
    
    bins = []
    sorted_rem = []  # sorted list of (remaining_capacity, bin_index)
    
    for w in items:
        lo, hi = 0, len(sorted_rem)
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_rem[mid][0] >= w:
                hi = mid
            else:
                lo = mid + 1
        
        if lo < len(sorted_rem):
            rem_cap, bidx = sorted_rem.pop(lo)
            bins[bidx].append(w)
            new_rem = rem_cap - w
            ins_pos = bisect.bisect_left(sorted_rem, (new_rem, 0))
            sorted_rem.insert(ins_pos, (new_rem, bidx))
        else:
            bidx = len(bins)
            bins.append([w])
            new_rem = capacity - w
            ins_pos = bisect.bisect_left(sorted_rem, (new_rem, 0))
            sorted_rem.insert(ins_pos, (new_rem, bidx))
    
    return bins


def make_solution_dict(obj, bins):
    return {
        "objective_value": float(obj),
        "num_bins": int(round(obj)),
        "bin_assignments": bins
    }


def verify_solution(bins, weights, demands, capacity):
    """Verify solution feasibility."""
    for b in bins:
        if sum(b) > capacity:
            return False
    width_demand = {}
    for j in range(len(weights)):
        w = weights[j]
        width_demand[w] = width_demand.get(w, 0) + demands[j]
    width_count = {}
    for b in bins:
        for w in b:
            width_count[w] = width_count.get(w, 0) + 1
    for w, d in width_demand.items():
        if width_count.get(w, 0) < d:
            return False
    return True


def fix_overproduction(bins, weights, demands):
    """Remove over-produced items from bins."""
    width_demand = {}
    for j in range(len(weights)):
        w = weights[j]
        width_demand[w] = width_demand.get(w, 0) + demands[j]
    
    width_count = {}
    for b in bins:
        for w in b:
            width_count[w] = width_count.get(w, 0) + 1
    
    excess = {}
    for w, d in width_demand.items():
        cnt = width_count.get(w, 0)
        if cnt > d:
            excess[w] = cnt - d
    
    if not excess:
        return bins
    
    new_bins = []
    for b in bins:
        new_b = []
        for w in b:
            if excess.get(w, 0) > 0:
                excess[w] -= 1
            else:
                new_b.append(w)
        if new_b:
            new_bins.append(new_b)
    
    return new_bins


def solve(instance, time_limit, logger):
    n = instance['n']
    m = instance['m']
    c = instance['c']
    weights = instance['weights']
    demands = instance['demands']
    
    start_time = time.time()
    
    if n == 0 or m == 0 or all(d == 0 for d in demands):
        return 0, []
    
    # Step 1: BFD heuristic
    heuristic_bins = best_fit_decreasing(weights, demands, c)
    best_obj = len(heuristic_bins)
    best_bins = heuristic_bins
    
    if logger:
        logger.log_solution(float(best_obj), make_solution_dict(best_obj, best_bins))
    
    try:
        return _solve_mip(weights, demands, c, m, heuristic_bins, best_obj, best_bins,
                         start_time, time_limit, logger)
    except Exception:
        return best_obj, best_bins


def _solve_mip(weights, demands, c, m, heuristic_bins, best_obj, best_bins,
               start_time, time_limit, logger):
    
    # Step 2: Generate initial patterns
    patterns = []
    pattern_set = set()
    
    def add_pattern(pattern):
        key = tuple(pattern)
        if key not in pattern_set and sum(p * w for p, w in zip(pattern, weights)) <= c and sum(pattern) > 0:
            pattern_set.add(key)
            patterns.append(list(pattern))
    
    for j in range(m):
        if demands[j] > 0:
            pattern = [0] * m
            pattern[j] = min(demands[j], c // weights[j])
            add_pattern(pattern)
    
    for b in heuristic_bins:
        pattern = [0] * m
        for w in b:
            for j in range(m):
                if weights[j] == w and pattern[j] < demands[j]:
                    pattern[j] += 1
                    break
        add_pattern(pattern)
    
    if not patterns:
        return best_obj, best_bins
    
    # Step 3: Column generation (Gilmore-Gomory)
    kp_model = gp.Model("kp")
    kp_model.Params.OutputFlag = 0
    kp_model.Params.Threads = 1
    kp_model.Params.Seed = 0
    
    a_vars = [kp_model.addVar(vtype=GRB.INTEGER, lb=0, ub=demands[j], name=f"a_{j}") for j in range(m)]
    kp_model.addConstr(gp.quicksum(weights[j] * a_vars[j] for j in range(m)) <= c)
    
    cg_time_limit = start_time + time_limit * 0.5
    max_patterns = 10000
    max_iterations = 300
    iteration = 0
    
    while time.time() < cg_time_limit and len(patterns) < max_patterns and iteration < max_iterations:
        iteration += 1
        n_patterns = len(patterns)
        
        lp_model = gp.Model("master_lp")
        lp_model.Params.OutputFlag = 0
        lp_model.Params.Threads = 1
        lp_model.Params.Seed = 0
        
        x_lp = lp_model.addVars(n_patterns, vtype=GRB.CONTINUOUS, lb=0, name="x")
        
        constrs = []
        for j in range(m):
            if demands[j] > 0:
                constrs.append(lp_model.addConstr(
                    gp.quicksum(patterns[p][j] * x_lp[p] for p in range(n_patterns)) >= demands[j]
                ))
            else:
                constrs.append(None)
        
        lp_model.setObjective(gp.quicksum(x_lp[p] for p in range(n_patterns)), GRB.MINIMIZE)
        lp_model.optimize()
        
        if lp_model.status != GRB.OPTIMAL:
            break
        
        duals = [constrs[j].Pi if constrs[j] is not None else 0.0 for j in range(m)]
        
        kp_model.setObjective(gp.quicksum(duals[j] * a_vars[j] for j in range(m)), GRB.MAXIMIZE)
        kp_model.optimize()
        
        if kp_model.status != GRB.OPTIMAL:
            break
        
        if kp_model.objVal <= 1.0 + 1e-6:
            break
        
        new_pattern = [int(round(a_vars[j].X)) for j in range(m)]
        key = tuple(new_pattern)
        if key in pattern_set:
            break
        
        add_pattern(new_pattern)
    
    # Step 4: Solve MIP
    remaining_time = time_limit - (time.time() - start_time) - 1.0
    if remaining_time < 0.5:
        return best_obj, best_bins
    
    n_patterns = len(patterns)
    
    mip_model = gp.Model("master_mip")
    mip_model.Params.OutputFlag = 0
    mip_model.Params.Threads = 1
    mip_model.Params.Seed = 0
    mip_model.Params.MIPGap = 1e-4
    mip_model.Params.NumericFocus = 0
    mip_model.Params.TimeLimit = remaining_time
    
    x = mip_model.addVars(n_patterns, vtype=GRB.INTEGER, lb=0, name="x")
    
    for j in range(m):
        if demands[j] > 0:
            mip_model.addConstr(gp.quicksum(patterns[p][j] * x[p] for p in range(n_patterns)) >= demands[j])
    
    mip_model.setObjective(gp.quicksum(x[p] for p in range(n_patterns)), GRB.MINIMIZE)
    
    state = {'best_obj': best_obj, 'best_bins': best_bins}
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                sol_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if sol_obj < state['best_obj'] - 1e-6:
                    x_vals = model.cbGet(GRB.Callback.MIPSOL_X, [x[p] for p in range(n_patterns)])
                    bins = []
                    for p in range(n_patterns):
                        count = int(round(x_vals[p]))
                        for _ in range(count):
                            bin_items = []
                            for j in range(m):
                                bin_items.extend([weights[j]] * patterns[p][j])
                            if bin_items:
                                bins.append(bin_items)
                    bins = fix_overproduction(bins, weights, demands)
                    state['best_obj'] = len(bins)
                    state['best_bins'] = bins
                    if logger:
                        logger.log_solution(float(state['best_obj']), make_solution_dict(state['best_obj'], bins))
            except Exception:
                pass
    
    mip_model.optimize(callback)
    
    # Get final solution from MIP
    if mip_model.SolCount > 0:
        bins = []
        for p in range(n_patterns):
            count = int(round(x[p].X))
            if count > 0:
                for _ in range(count):
                    bin_items = []
                    for j in range(m):
                        bin_items.extend([weights[j]] * patterns[p][j])
                    if bin_items:
                        bins.append(bin_items)
        bins = fix_overproduction(bins, weights, demands)
        obj = len(bins)
        if obj < best_obj:
            best_obj = obj
            best_bins = bins
            if logger:
                logger.log_solution(float(best_obj), make_solution_dict(best_obj, best_bins))
    
    # Verify solution
    if not verify_solution(best_bins, weights, demands, c):
        best_obj = len(heuristic_bins)
        best_bins = heuristic_bins
    
    return best_obj, best_bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    logger = None
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    best_obj, best_bins = solve(instance, args.time_limit, logger)
    
    solution = make_solution_dict(best_obj, best_bins)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()