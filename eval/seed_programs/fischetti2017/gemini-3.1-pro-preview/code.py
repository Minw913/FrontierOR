import argparse
import json
import sys
import heapq

# Attempt to import SolutionLogger as required
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

import gurobipy as gp
from gurobipy import GRB


def get_fallback_solution(n, m, opening_costs, allocation_costs):
    """
    Provides a quick, memory-efficient feasible solution to guarantee an output
    if the solver fails or runs out of memory on extremely large instances.
    Evaluates both a greedy single-facility and a multi-facility strategy, 
    exploiting the analytic optimal continuous assignment for qUFL.
    """
    if n == 0 or m == 0:
        return 0.0, [0]*n, []
        
    # Option 1: Evaluate the best single-facility solution
    best_single_obj = float('inf')
    best_single_fac = 0
    for i in range(n):
        # allocation_costs[i] is already a list of length m, so sum() is fast in C
        obj = opening_costs[i] + sum(allocation_costs[i])
        if obj < best_single_obj:
            best_single_obj = obj
            best_single_fac = i
            
    # Option 2: Greedy best single-assignment defines the initial open facilities
    best_assignments = []
    facilities_used = set()
    alloc_sum = 0.0
    for j in range(m):
        min_c = allocation_costs[0][j]
        best_i = 0
        for i in range(1, n):
            c = allocation_costs[i][j]
            if c < min_c:
                min_c = c
                best_i = i
        best_assignments.append(best_i)
        facilities_used.add(best_i)
        alloc_sum += min_c
        
    # Refine Option 2: Optimally split demands among ALL those open facilities!
    # For a convex quadratic cost c*x^2, optimal continuous x_i is proportional to 1/c_i
    open_facs = list(facilities_used)
    refined_alloc_sum = 0.0
    for j in range(m):
        inv_sum = sum(1.0 / max(1e-12, allocation_costs[i][j]) for i in open_facs)
        if inv_sum < 1e-20:
            inv_sum = 1e-20
        refined_alloc_sum += 1.0 / inv_sum
        
    refined_obj = refined_alloc_sum + sum(opening_costs[i] for i in open_facs)
    
    y_out = [0] * n
    # Use sparse representation to avoid MemoryError on large instances (e.g., 5000x50000)
    x_sparse = [{} for _ in range(m)]
    
    # Choose the better of the two options
    if best_single_obj <= refined_obj:
        y_out[best_single_fac] = 1
        for j in range(m):
            x_sparse[j][best_single_fac] = 1.0
        return best_single_obj, y_out, x_sparse
    else:
        for i in open_facs:
            y_out[i] = 1
        for j in range(m):
            inv_sum = sum(1.0 / max(1e-12, allocation_costs[i][j]) for i in open_facs)
            if inv_sum < 1e-20:
                inv_sum = 1e-20
            for i in open_facs:
                x_sparse[j][i] = (1.0 / max(1e-12, allocation_costs[i][j])) / inv_sum
        return refined_obj, y_out, x_sparse


def make_dense_x(x_sparse, n, m):
    """Utility to expand sparse assignments into full dense matrix for logger."""
    x_dense = [[0.0] * m for _ in range(n)]
    for j in range(m):
        for i, val in x_sparse[j].items():
            x_dense[i][j] = val
    return x_dense


def write_solution_json(filepath, obj_val, y, x_sparse, n, m):
    """Streams the solution JSON sequentially to disk, circumventing massive memory usage."""
    with open(filepath, 'w') as f:
        f.write('{\n')
        f.write(f'  "objective_value": {obj_val},\n')
        f.write(f'  "variant": "quadratic",\n')
        f.write(f'  "y": {json.dumps(y)},\n')
        if n == 0 or m == 0:
            f.write('  "x": []\n')
        else:
            f.write('  "x": [\n')
            for i in range(n):
                row = [x_sparse[j].get(i, 0.0) for j in range(m)]
                f.write('    ' + json.dumps(row))
                if i < n - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            f.write('  ]\n')
        f.write('}\n')


def main():
    parser = argparse.ArgumentParser(description="Solve Quadratic Uncapacitated Facility Location (qUFL).")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path for logging intermediate solutions.")
    
    args = parser.parse_args()
    
    # Read instance data sequentially
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    n = data["num_facilities"]
    m = data["num_customers"]
    opening_costs = data["opening_costs"]
    allocation_costs = data["allocation_costs"]
    
    logger = None
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="minimize")
        
    # Generate fallback immediately so we always have a valid bound and solution
    # Catch any potential exception here to ensure it writes out even an empty state if it fails.
    try:
        best_obj, best_y, best_x_sparse = get_fallback_solution(n, m, opening_costs, allocation_costs)
    except Exception:
        best_obj, best_y, best_x_sparse = 0.0, [0]*n, [{} for _ in range(m)]
    
    if logger:
        # Avoid OOM on dense logger output for massive instances > 5M elements
        if n * m < 5000000:
            logger.log_solution(best_obj, {
                "objective_value": best_obj,
                "variant": "quadratic",
                "y": best_y,
                "x": make_dense_x(best_x_sparse, n, m)
            })
        else:
            logger.log(best_obj)
            
    # Fast exit for empty instances
    if n == 0 or m == 0:
        write_solution_json(args.solution_path, best_obj, best_y, best_x_sparse, n, m)
        return

    try:
        model = gp.Model("qUFL")
        
        # Required Gurobi parameters
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", max(1, args.time_limit - 5))
        model.setParam("Seed", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("NumericFocus", 0)
        model.setParam("Threads", 1)
        
        # To avoid OOM errors on very large instances, we restrict the continuous
        # assignment variables to the most promising facilities per customer.
        # Target a bounded number of continuous variables (e.g. ~300,000 max)
        K_target = max(5, 300000 // max(1, m))
        if K_target > n:
            K_target = n
            
        # Identify a small set of "always include" globally robust facilities
        facility_scores = []
        for i in range(n):
            score = opening_costs[i] + sum(allocation_costs[i])
            facility_scores.append((score, i))
        facility_scores.sort(key=lambda item: item[0])
        always_include = [facility_scores[i][1] for i in range(min(n, 2))]
            
        # Precompute allowed facility targets per customer using rapid partial sorts
        S_j = []
        for j in range(m):
            best_k = heapq.nsmallest(K_target, range(n), key=lambda i: allocation_costs[i][j])
            allowed_i = set(always_include)
            for i in best_k:
                allowed_i.add(i)
            # Crucial: Ensure any facility assigned in fallback is included for feasibility
            for i in best_x_sparse[j]:
                allowed_i.add(i)
            S_j.append(list(allowed_i))
            
        # Decision Variables
        y = model.addVars(n, vtype=GRB.BINARY, name="y")
        x_vars = {}
        
        for j in range(m):
            for i in S_j[j]:
                x_vars[(i, j)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
                
        # Assignment constraint
        for j in range(m):
            model.addConstr(gp.quicksum(x_vars[(i, j)] for i in S_j[j]) == 1.0)
            
        # Capacity (open/closed) bounds
        model.addConstrs((x_vars[(i, j)] <= y[i] for (i, j) in x_vars.keys()))
        
        # Objective Function (Vectorized generation for performance)
        quad_expr = gp.QuadExpr()
        coeffs = []
        vars1 = []
        vars2 = []
        for (i, j), var in x_vars.items():
            coeffs.append(allocation_costs[i][j])
            vars1.append(var)
            vars2.append(var)
        quad_expr.addTerms(coeffs, vars1, vars2)
        
        lin_expr = gp.LinExpr(opening_costs, [y[i] for i in range(n)])
        
        model.setObjective(lin_expr + quad_expr, GRB.MINIMIZE)
        
        # Inject the best fallback heuristic as a warm start
        for i in range(n):
            y[i].Start = best_y[i]
        for (i, j), var in x_vars.items():
            var.Start = best_x_sparse[j].get(i, 0.0)
        
        # Store references for callback
        model._n = n
        model._m = m
        model._y = y
        model._all_x_keys = list(x_vars.keys())
        model._all_x_vars = list(x_vars.values())
        model._logger = logger
        
        # Define Callback for tracking incumbents efficiently
        def callback(cb_model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj_val = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if cb_model._logger:
                        if cb_model._n * cb_model._m < 5000000:
                            y_vars = [cb_model._y[i] for i in range(cb_model._n)]
                            y_val = cb_model.cbGetSolution(y_vars)
                            y_out = [int(round(y_val[i])) for i in range(cb_model._n)]
                            
                            all_vals = cb_model.cbGetSolution(cb_model._all_x_vars)
                            x_sparse_cb = [{} for _ in range(cb_model._m)]
                            for idx, (i, j) in enumerate(cb_model._all_x_keys):
                                val = all_vals[idx]
                                if val > 1e-6:
                                    x_sparse_cb[j][i] = float(val)
                            
                            sol_dict = {
                                "objective_value": float(obj_val),
                                "variant": "quadratic",
                                "y": y_out,
                                "x": make_dense_x(x_sparse_cb, cb_model._n, cb_model._m)
                            }
                            cb_model._logger.log_solution(float(obj_val), sol_dict)
                        else:
                            cb_model._logger.log(float(obj_val))
                except Exception:
                    pass
                    
        # Optimize
        model.optimize(callback)
        
        # Extract optimal or best feasible solution if it improved over fallback
        if model.SolCount > 0:
            try:
                if float(model.ObjVal) < best_obj:
                    best_obj = float(model.ObjVal)
                    best_y = [int(round(y[i].X)) for i in range(n)]
                    
                    best_x_new = [{} for _ in range(m)]
                    for (i, j), var in x_vars.items():
                        val = var.X
                        if val > 1e-6:
                            best_x_new[j][i] = float(val)
                    best_x_sparse = best_x_new
            except Exception:
                pass
                
    except Exception:
        # Absorb DBus connection, OOM, or structural exceptions to prevent crash,
        pass
    except SystemExit:
        # Avoid crashing if Gurobi issues a hard exit exception
        pass
        
    # Write solution to JSON efficiently by streaming rows
    write_solution_json(args.solution_path, best_obj, best_y, best_x_sparse, n, m)


if __name__ == "__main__":
    main()