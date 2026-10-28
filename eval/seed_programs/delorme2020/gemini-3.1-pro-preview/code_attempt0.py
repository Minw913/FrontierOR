import sys
import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def compute_true_assignments(pats, demands, m, weights):
    """
    Expands the pattern dictionary into exact bin assignments matching strictly the
    required demands without overproduction. Ensures we form as few bins as possible
    by sorting bins with larger filled capacities to the front. 
    """
    all_bins = []
    for pt, count in pats.items():
        for _ in range(count):
            all_bins.append(pt)
            
    # Sort bins to prioritize those that pack more items/weight 
    # so partial bins are discarded at the end when demand is fulfilled.
    all_bins.sort(key=lambda pt: sum(pt[t] * weights[t] for t in range(m)), reverse=True)
    
    assignments = []
    rem_d = list(demands)
    for pt in all_bins:
        bin_items = []
        for t in range(m):
            take = min(pt[t], rem_d[t])
            if take > 0:
                rem_d[t] -= take
                bin_items.extend([weights[t]] * take)
        if bin_items:
            assignments.append(bin_items)
            
    return assignments

def bfd_bulk(c, m, weights, demands):
    """
    An extremely fast Grouped Best Fit Decreasing (BFD) heuristic.
    Process the identical items in batches instead of one by one.
    This guarantees mathematically identical results to BFD 
    but works instantaneously for millions of items.
    """
    types = sorted(range(m), key=lambda i: weights[i], reverse=True)
    
    # bins: list of [rem_cap, pattern_list, count]
    bins = []
    
    for t in types:
        req = demands[t]
        w = weights[t]
        
        bins.sort(key=lambda b: b[0]) # BFD sorts by remaining capacity Ascending
        
        new_bins = []
        for i in range(len(bins)):
            if req == 0: 
                break
            rem_cap, p, count = bins[i]
            
            if count == 0:
                continue
                
            if rem_cap >= w:
                max_fit_per_bin = rem_cap // w
                full_bins_to_fill = min(req // max_fit_per_bin, count)
                
                if full_bins_to_fill > 0:
                    new_p = list(p)
                    new_p[t] += max_fit_per_bin
                    new_bins.append([rem_cap - max_fit_per_bin * w, new_p, full_bins_to_fill])
                    req -= full_bins_to_fill * max_fit_per_bin
                    bins[i][2] -= full_bins_to_fill
                    count -= full_bins_to_fill
                    
                if req > 0 and count > 0:
                    take = req
                    if take > max_fit_per_bin: 
                        take = max_fit_per_bin 
                    new_p = list(p)
                    new_p[t] += take
                    new_bins.append([rem_cap - take * w, new_p, 1])
                    req -= take
                    bins[i][2] -= 1
                    count -= 1
                    
        bins = [b for b in bins if b[2] > 0]
        bins.extend(new_bins)
        
        if req > 0:
            items_per_bin = c // w
            full_bins = req // items_per_bin
            if full_bins > 0:
                p = [0] * m
                p[t] = items_per_bin
                bins.append([c - items_per_bin * w, p, full_bins])
                req -= full_bins * items_per_bin
                
            if req > 0:
                p = [0] * m
                p[t] = req
                bins.append([c - req * w, p, 1])
                req = 0
                
    patterns = {}
    for rem, p, count in bins:
        pt = tuple(p)
        patterns[pt] = patterns.get(pt, 0) + count
        
    return patterns

def main():
    parser = argparse.ArgumentParser(description='Solve Bin Packing / Cutting Stock Problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file (optional)')
    args = parser.parse_args()

    start_time = time.time()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    # 1. Read input data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    n = data['n']
    m = data['m']
    c = data['c']
    weights = data['weights']
    demands = data['demands']
    
    # Handle empty requests
    if m == 0 or sum(demands) == 0:
        out_data = {
            "objective_value": 0.0,
            "num_bins": 0,
            "bin_assignments": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(out_data, f, indent=2)
        return

    # 2. Get high-quality initial bound by using Grouped Best Fit Decreasing
    init_patterns = bfd_bulk(c, m, weights, demands)
    init_assign = compute_true_assignments(init_patterns, demands, m, weights)
    
    best_obj = len(init_assign)
    best_pats = [init_patterns.copy()]
    best_obj_val = [best_obj]

    if logger:
        logger.log(best_obj)

    # Convert initial patterns list for Column Generation basis
    P = list(init_patterns.keys())
    P_set = set(P)

    # 3. Setup Cutting Stock Restricted Master Problem (RMP)
    rmp = gp.Model("RMP")
    rmp.setParam('OutputFlag', 0)
    rmp.setParam('Threads', 1)

    lambdas = []
    for p in P:
        lambdas.append(rmp.addVar(obj=1.0, vtype=GRB.CONTINUOUS))

    constrs = []
    for i in range(m):
        constrs.append(rmp.addConstr(
            gp.quicksum(P[j][i] * lambdas[j] for j in range(len(P))) >= demands[i],
            name=f"dem_{i}"
        ))

    # Setup Pricing Subproblem (Bounded Knapsack)
    price_prob = gp.Model("Pricing")
    price_prob.setParam('OutputFlag', 0)
    price_prob.setParam('Threads', 1)
    
    z = []
    for i in range(m):
        z.append(price_prob.addVar(vtype=GRB.INTEGER, lb=0, ub=c // weights[i], name=f"z_{i}"))

    price_prob.addConstr(gp.quicksum(weights[i] * z[i] for i in range(m)) <= c, "cap")

    # 4. Column Generation phase
    max_cg_iterations = 2000
    ip_reserve = min(args.time_limit * 0.2, 10.0)
    cg_time_limit = args.time_limit - ip_reserve

    for it in range(max_cg_iterations):
        elapsed = time.time() - start_time
        if elapsed > cg_time_limit:
            break
            
        rmp.optimize()
        if rmp.status != GRB.OPTIMAL:
            break
            
        duals = [max(0.0, cons.Pi) for cons in constrs]
        
        price_prob.setObjective(gp.quicksum(duals[i] * z[i] for i in range(m)), GRB.MAXIMIZE)
        rem_cg = cg_time_limit - elapsed
        if rem_cg < 0:
            break
        price_prob.setParam('TimeLimit', rem_cg)
        price_prob.optimize()
        
        if price_prob.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            if price_prob.status == GRB.TIME_LIMIT and price_prob.SolCount == 0:
                break
            
            max_red_cost = price_prob.ObjVal - 1.0
            if max_red_cost > 1e-6:
                new_p = tuple(int(round(z[i].X)) for i in range(m))
                if new_p not in P_set:
                    P.append(new_p)
                    P_set.add(new_p)
                    
                    col = gp.Column()
                    for i in range(m):
                        if new_p[i] > 0:
                            col.addTerms(new_p[i], constrs[i])
                    new_var = rmp.addVar(obj=1.0, vtype=GRB.CONTINUOUS, column=col)
                    lambdas.append(new_var)
                else: 
                    break # Repeated column limits progress 
            else: 
                break # Optimal LG Relaxation found
        else:
            break

    # 5. Integer Program Phase on the extended columns
    for var in lambdas:
        var.vtype = GRB.INTEGER
        
    for j, p in enumerate(P):
        if p in best_pats[0]:
            lambdas[j].Start = best_pats[0][p]
        else:
            lambdas[j].Start = 0

    rmp.update()
    
    def mip_cb(model, where):
        if where == GRB.Callback.MIPSOL:
            sol = model.cbGetSolution(lambdas)
            curr_p = {}
            for j, val in enumerate(sol):
                c_val = int(round(val))
                if c_val > 0:
                    curr_p[P[j]] = c_val
                    
            # Accurately translate incumbent variables mapping exactly to dropped requirements 
            assignments = compute_true_assignments(curr_p, demands, m, weights)
            true_num = len(assignments)
            
            if true_num < best_obj_val[0]:
                best_obj_val[0] = true_num
                best_pats[0] = curr_p
                if logger:
                    logger.log(true_num)

    rem_time = args.time_limit - (time.time() - start_time)
    if rem_time > 0:
        rmp.setParam('TimeLimit', rem_time)
        rmp.optimize(mip_cb)

    # 6. Final Outputs Setup
    final_assign = compute_true_assignments(best_pats[0], demands, m, weights)
    final_obj = len(final_assign)
    
    out_data = {
        "objective_value": float(final_obj),
        "num_bins": final_obj,
        "bin_assignments": final_assign
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(out_data, f, indent=2)


if __name__ == '__main__':
    main()