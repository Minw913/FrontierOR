import json
import argparse
import time
import sys
from gurobipy import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    start_time = time.time()
    time_limit = args.time_limit
    
    # Parse instance data
    n1 = instance['n1']
    n2 = instance['n2']
    m1 = instance['m1']
    m2 = instance['m2']
    c = instance['c']
    d2 = instance['d2']
    A1 = instance['A1']
    B1 = instance['B1']
    b1 = instance['b1']
    A2 = instance['A2']
    B2 = instance['B2']
    b2 = instance['b2']
    integer_follower_vars = instance['integer_follower_vars']
    
    int_follower_set = set(v - 1 for v in integer_follower_vars)
    
    # Compute upper bounds on x variables from constraints
    x_ub = [float('inf')] * n1
    for i in range(m1):
        for j in range(n1):
            if A1[i][j] > 0:
                x_ub[j] = min(x_ub[j], b1[i] / A1[i][j])
    for i in range(m2):
        for j in range(n1):
            if A2[i][j] > 0:
                x_ub[j] = min(x_ub[j], b2[i] / A2[i][j])
    
    MAX_UB = 1000
    x_ub_int = []
    for u in x_ub:
        if u == float('inf'):
            x_ub_int.append(MAX_UB)
        else:
            x_ub_int.append(min(int(u), MAX_UB))
    
    def remaining_time():
        return max(0.1, time_limit - (time.time() - start_time))
    
    best_obj = -float('inf')
    best_x = None
    best_y = None
    best_follower_obj = None
    
    def solve_follower(x_vals, t_limit):
        """Solve follower's MIP for fixed x. Returns (opt_val, opt_y) or (None, None)."""
        model = Model("follower")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = max(0.1, t_limit)
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-6
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        
        y_vars = []
        for j in range(n2):
            if j in int_follower_set:
                y_vars.append(model.addVar(vtype=GRB.INTEGER, lb=0))
            else:
                y_vars.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0))
        model.update()
        
        for i in range(m2):
            rhs = b2[i] - sum(A2[i][j] * x_vals[j] for j in range(n1))
            if rhs < -1e-8:
                return None, None
            model.addConstr(quicksum(B2[i][j] * y_vars[j] for j in range(n2)) <= rhs + 1e-8)
        
        model.setObjective(quicksum(d2[j] * y_vars[j] for j in range(n2)), GRB.MAXIMIZE)
        model.optimize()
        
        if model.Status == GRB.INFEASIBLE:
            return None, None
        if model.Status == GRB.UNBOUNDED:
            return float('inf'), None
        
        if model.SolCount > 0:
            return model.ObjVal, [y_vars[j].X for j in range(n2)]
        return None, None
    
    def check_bilevel_feasible(x_vals, V_star, t_limit):
        """Find y optimal for follower satisfying upper-level constraints."""
        model = Model("bilevel_check")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = max(0.1, t_limit)
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-6
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        
        y_vars = []
        for j in range(n2):
            if j in int_follower_set:
                y_vars.append(model.addVar(vtype=GRB.INTEGER, lb=0))
            else:
                y_vars.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0))
        model.update()
        
        # Lower-level constraints
        for i in range(m2):
            rhs = b2[i] - sum(A2[i][j] * x_vals[j] for j in range(n1))
            model.addConstr(quicksum(B2[i][j] * y_vars[j] for j in range(n2)) <= rhs + 1e-8)
        
        # Follower optimality: d2*y >= V_star
        model.addConstr(quicksum(d2[j] * y_vars[j] for j in range(n2)) >= V_star - 1e-4)
        
        # Upper-level constraints: A1*x + B1*y <= b1
        for i in range(m1):
            lhs_x = sum(A1[i][j] * x_vals[j] for j in range(n1))
            model.addConstr(lhs_x + quicksum(B1[i][j] * y_vars[j] for j in range(n2)) <= b1[i] + 1e-8)
        
        model.setObjective(0, GRB.MAXIMIZE)
        model.optimize()
        
        if model.SolCount > 0:
            return [y_vars[j].X for j in range(n2)]
        return None
    
    def try_x(x_vals):
        """Try a candidate x. Returns True if bilevel feasible and improves best."""
        nonlocal best_obj, best_x, best_y, best_follower_obj
        
        if remaining_time() < 1:
            return False
        
        obj_val = sum(c[j] * x_vals[j] for j in range(n1))
        if obj_val <= best_obj + 1e-8:
            return False
        
        t = min(remaining_time() / 3, 30)
        V_star, _ = solve_follower(x_vals, t)
        
        if V_star is None or V_star == float('inf'):
            return False
        
        t = min(remaining_time() / 3, 30)
        y_feas = check_bilevel_feasible(x_vals, V_star, t)
        
        if y_feas is not None:
            best_obj = obj_val
            best_x = x_vals[:]
            best_y = y_feas[:]
            best_follower_obj = V_star
            
            solution_dict = {
                "objective_value": float(best_obj),
                "x": [float(v) for v in best_x],
                "y": [float(v) for v in best_y],
                "follower_objective_value": float(best_follower_obj)
            }
            if logger:
                logger.log_solution(best_obj, solution_dict)
            return True
        return False
    
    # === Strategy 1: Solve relaxed problem with solution pool ===
    if remaining_time() > 2:
        pool_time = min(remaining_time() * 0.5, 180)
        
        model = Model("relaxed")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = pool_time
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions = 500
        model.Params.PoolGap = 0.99
        
        x_vars = [model.addVar(vtype=GRB.INTEGER, lb=0, ub=x_ub_int[j]) for j in range(n1)]
        y_vars = []
        for j in range(n2):
            if j in int_follower_set:
                y_vars.append(model.addVar(vtype=GRB.INTEGER, lb=0))
            else:
                y_vars.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0))
        model.update()
        
        for i in range(m1):
            model.addConstr(quicksum(A1[i][j] * x_vars[j] for j in range(n1)) + 
                           quicksum(B1[i][j] * y_vars[j] for j in range(n2)) <= b1[i])
        
        for i in range(m2):
            model.addConstr(quicksum(A2[i][j] * x_vars[j] for j in range(n1)) + 
                           quicksum(B2[i][j] * y_vars[j] for j in range(n2)) <= b2[i])
        
        model.setObjective(quicksum(c[j] * x_vars[j] for j in range(n1)), GRB.MAXIMIZE)
        model.optimize()
        
        if model.SolCount > 0:
            num_sols = min(model.SolCount, 500)
            checked_x = set()
            
            for i in range(num_sols):
                if remaining_time() < 2:
                    break
                model.Params.SolutionNumber = i
                x_vals = tuple(int(round(x_vars[j].Xn)) for j in range(n1))
                
                if x_vals in checked_x:
                    continue
                checked_x.add(x_vals)
                
                if try_x(list(x_vals)):
                    # Found a solution; since pool is ordered, this is likely optimal
                    # Continue checking a few more to be safe
                    if i > 20:
                        break
    
    # === Strategy 2: Iterative approach with no-good cuts ===
    if remaining_time() > 5:
        no_good_list = []
        max_iter = 100
        
        for iteration in range(max_iter):
            if remaining_time() < 3:
                break
            
            model = Model("iter")
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = min(remaining_time() * 0.4, 60)
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            
            x_vars = [model.addVar(vtype=GRB.INTEGER, lb=0, ub=x_ub_int[j]) for j in range(n1)]
            y_vars = []
            for j in range(n2):
                if j in int_follower_set:
                    y_vars.append(model.addVar(vtype=GRB.INTEGER, lb=0))
                else:
                    y_vars.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0))
            model.update()
            
            for i in range(m1):
                model.addConstr(quicksum(A1[i][j] * x_vars[j] for j in range(n1)) + 
                               quicksum(B1[i][j] * y_vars[j] for j in range(n2)) <= b1[i])
            
            for i in range(m2):
                model.addConstr(quicksum(A2[i][j] * x_vars[j] for j in range(n1)) + 
                               quicksum(B2[i][j] * y_vars[j] for j in range(n2)) <= b2[i])
            
            # No-good cuts using binary indicators
            for cut_x in no_good_list:
                z_vars = []
                for j in range(n1):
                    z = model.addVar(vtype=GRB.BINARY)
                    z_vars.append(z)
                model.update()
                for j in range(n1):
                    ub_j = max(x_ub_int[j], 1)
                    model.addConstr(x_vars[j] - cut_x[j] <= ub_j * z_vars[j])
                    model.addConstr(cut_x[j] - x_vars[j] <= ub_j * z_vars[j])
                model.addConstr(quicksum(z_vars) >= 1)
            
            model.setObjective(quicksum(c[j] * x_vars[j] for j in range(n1)), GRB.MAXIMIZE)
            model.optimize()
            
            if model.SolCount == 0:
                break
            
            x_vals = [int(round(x_vars[j].X)) for j in range(n1)]
            
            # Check if this x was already tried
            x_tuple = tuple(x_vals)
            already_tried = any(tuple(ng) == x_tuple for ng in no_good_list)
            if already_tried:
                break
            
            if try_x(x_vals):
                no_good_list.append(x_vals[:])
                # Check optimality
                if hasattr(model, 'ObjBound') and model.ObjBound <= best_obj + 1e-6:
                    break
            else:
                no_good_list.append(x_vals[:])
    
    # === Strategy 3: Heuristic fallback ===
    if best_obj == -float('inf') and remaining_time() > 2:
        # Try x = 0
        try_x([0] * n1)
        
        # Try greedy: set variables with positive c to their upper bounds
        if remaining_time() > 2:
            x_greedy = [0] * n1
            for j in range(n1):
                if c[j] > 0:
                    x_greedy[j] = x_ub_int[j]
            try_x(x_greedy)
        
        # Try individual variables
        for j in range(n1):
            if remaining_time() < 2:
                break
            if c[j] > 0:
                x_test = [0] * n1
                x_test[j] = x_ub_int[j]
                try_x(x_test)
    
    # Write final solution
    if best_x is not None:
        solution = {
            "objective_value": float(best_obj),
            "x": [float(v) for v in best_x],
            "y": [float(v) for v in best_y],
            "follower_objective_value": float(best_follower_obj)
        }
    else:
        # Fallback: output zero solution
        solution = {
            "objective_value": 0.0,
            "x": [0.0] * n1,
            "y": [0.0] * n2,
            "follower_objective_value": 0.0
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()