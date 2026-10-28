import argparse
import json
import numpy as np
import time
from itertools import combinations

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    k = data['k']
    b = data['b']
    r = np.array(data['r'], dtype=np.float64)
    Q = np.array(data['Q'], dtype=np.float64)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    best_obj = float('inf')
    best_x = np.zeros(n)
    
    def time_remaining():
        return time_limit - (time.time() - start_time)
    
    def solve_continuous_subset(indices):
        """Solve the continuous QP for a given subset of asset indices.
        
        min 0.5 * x^T Q_sub x
        s.t. sum(x) = 1
             r_sub^T x >= b
        
        Using KKT conditions with two equality constraints (treating inequality as active first).
        """
        idx = np.array(indices)
        m = len(idx)
        if m == 0:
            return None, float('inf')
        
        Q_sub = Q[np.ix_(idx, idx)]
        r_sub = r[idx]
        
        # Try solving with return constraint active: r_sub^T x = b
        # KKT: Q_sub x + lambda1 * 1 + lambda2 * (-r_sub) = 0
        #       1^T x = 1
        #       r_sub^T x = b
        # 
        # [Q_sub  | 1  | -r_sub ] [x      ]   [0]
        # [1^T    | 0  |  0     ] [lambda1] = [1]
        # [-r_sub^T| 0 |  0     ] [lambda2]   [-b]
        
        # Build the KKT system
        M = np.zeros((m + 2, m + 2))
        M[:m, :m] = Q_sub
        M[:m, m] = 1.0
        M[:m, m+1] = -r_sub
        M[m, :m] = 1.0
        M[m+1, :m] = -r_sub
        
        rhs = np.zeros(m + 2)
        rhs[m] = 1.0
        rhs[m+1] = -b
        
        try:
            sol = np.linalg.solve(M, rhs)
            x_sub = sol[:m]
            lambda2 = sol[m+1]
            
            # Check if lambda2 >= 0 (dual variable for >= constraint, with -r formulation means lambda2 >= 0)
            # Actually, the constraint is r^T x >= b, written as -r^T x <= -b
            # dual lambda2 >= 0 for this constraint being active
            if lambda2 >= -1e-10:
                # Valid solution with constraint active
                port_ret = r_sub @ x_sub
                if port_ret >= b - 1e-8:
                    x_full = np.zeros(n)
                    x_full[idx] = x_sub
                    obj = 0.5 * x_full @ Q @ x_full
                    return x_full, obj
        except np.linalg.LinAlgError:
            pass
        
        # Try without return constraint active (just budget constraint)
        # KKT: Q_sub x + lambda * 1 = 0, 1^T x = 1
        M2 = np.zeros((m + 1, m + 1))
        M2[:m, :m] = Q_sub
        M2[:m, m] = 1.0
        M2[m, :m] = 1.0
        
        rhs2 = np.zeros(m + 1)
        rhs2[m] = 1.0
        
        try:
            sol2 = np.linalg.solve(M2, rhs2)
            x_sub = sol2[:m]
            port_ret = r_sub @ x_sub
            if port_ret >= b - 1e-8:
                x_full = np.zeros(n)
                x_full[idx] = x_sub
                obj = 0.5 * x_full @ Q @ x_full
                return x_full, obj
        except np.linalg.LinAlgError:
            pass
        
        return None, float('inf')
    
    # If k >= n, solve the full problem
    if k >= n:
        x_full, obj = solve_continuous_subset(list(range(n)))
        if x_full is not None and obj < best_obj:
            best_obj = obj
            best_x = x_full.copy()
            if logger:
                logger.log(best_obj)
    else:
        # For small n, enumerate all C(n, k') for k' = 1..k
        # For larger n, use Gurobi MIP
        
        total_combos = 0
        for kk in range(1, k + 1):
            from math import comb
            total_combos += comb(n, kk)
        
        if total_combos <= 500000 and n <= 100:
            # Enumerate all subsets
            for kk in range(1, k + 1):
                for subset in combinations(range(n), kk):
                    if time_remaining() < 0.5:
                        break
                    x_sub, obj = solve_continuous_subset(list(subset))
                    if x_sub is not None and obj < best_obj:
                        best_obj = obj
                        best_x = x_sub.copy()
                        if logger:
                            logger.log(best_obj)
                if time_remaining() < 0.5:
                    break
        else:
            # Use Gurobi for the MIQP
            import gurobipy as gp
            from gurobipy import GRB
            
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            
            model = gp.Model(env=env)
            model.setParam('TimeLimit', max(1, time_remaining() - 1))
            model.setParam('Threads', 1)
            
            # Decision variables
            x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
            z = model.addVars(n, vtype=GRB.BINARY, name="z")
            
            # Big-M for linking constraints
            # We need to determine reasonable bounds on x_i
            # Use a large M
            M_val = 100.0
            
            # Constraints
            # Budget: sum(x) = 1
            model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1.0)
            
            # Return constraint: r^T x >= b
            model.addConstr(gp.quicksum(r[i] * x[i] for i in range(n)) >= b)
            
            # Cardinality: sum(z) <= k
            model.addConstr(gp.quicksum(z[i] for i in range(n)) <= k)
            
            # Linking: x[i] = 0 if z[i] = 0
            for i in range(n):
                model.addConstr(x[i] <= M_val * z[i])
                model.addConstr(x[i] >= -M_val * z[i])
            
            # Objective: min 0.5 * x^T Q x
            obj_expr = gp.QuadExpr()
            for i in range(n):
                for j in range(n):
                    if Q[i][j] != 0:
                        obj_expr += 0.5 * Q[i][j] * x[i] * x[j]
            
            model.setObjective(obj_expr, GRB.MINIMIZE)
            
            # Callback for logging
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    nonlocal best_obj, best_x
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_val < best_obj:
                        best_obj = obj_val
                        x_vals = model.cbGetSolution([x[i] for i in range(n)])
                        best_x = np.array(x_vals)
                        if logger:
                            logger.log(best_obj)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                obj_val = model.ObjVal
                x_vals = np.array([x[i].X for i in range(n)])
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_x = x_vals.copy()
                    if logger:
                        logger.log(best_obj)
    
    # If no solution found, try a simple feasible solution
    if best_obj == float('inf'):
        # Try single assets
        for i in range(n):
            if r[i] >= b - 1e-8:
                x_try = np.zeros(n)
                x_try[i] = 1.0
                obj = 0.5 * Q[i][i]
                if obj < best_obj:
                    best_obj = obj
                    best_x = x_try.copy()
                    if logger:
                        logger.log(best_obj)
    
    # Clean up near-zero values
    for i in range(n):
        if abs(best_x[i]) < 1e-10:
            best_x[i] = 0.0
    
    # Recompute objective
    final_obj = 0.5 * best_x @ Q @ best_x
    
    solution = {
        "objective_value": float(final_obj),
        "x": best_x.tolist()
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)