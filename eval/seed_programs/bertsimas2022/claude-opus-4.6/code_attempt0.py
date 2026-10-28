import argparse
import json
import numpy as np
import time
from solution_logger import SolutionLogger

def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    k = data['k']
    r = data['r']
    gamma = data['gamma']
    kappa = data['kappa']
    mu = np.array(data['mu'], dtype=np.float64)
    F = np.array(data['factor_loadings'], dtype=np.float64)  # shape [n, r]
    d = np.array(data['idiosyncratic_variance'], dtype=np.float64)  # shape [n]
    
    constraints = data['constraints']
    has_min_return = constraints['has_min_return_constraint']
    r_bar = constraints['r_bar']
    has_min_invest = constraints['has_min_investment_constraint']
    l_min = np.array(constraints['l_min_investment'], dtype=np.float64) if constraints['l_min_investment'] is not None else None
    u_max = np.array(constraints['u_max_investment'], dtype=np.float64) if constraints['u_max_investment'] is not None else None
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Covariance: Sigma = F @ F^T + diag(d)
    # Objective: 0.5 * x^T Sigma x + (1/(2*gamma)) * x^T x - kappa * mu^T x
    # = 0.5 * x^T (Sigma + (1/gamma) I) x - kappa * mu^T x
    # = 0.5 * x^T Q x - kappa * mu^T x
    # where Q = Sigma + (1/gamma) I = F F^T + diag(d + 1/gamma)
    
    ridge = 1.0 / gamma
    d_eff = d + ridge  # diagonal part of Q
    
    # For scoring subsets, we need to solve QP for each candidate subset S (|S| <= k):
    # min 0.5 x_S^T Q_SS x_S - kappa * mu_S^T x_S
    # s.t. sum(x_S) = 1, x_S >= l_min_S (if applicable), x_S <= u_max_S, etc.
    
    def compute_objective(x):
        """Compute full objective value."""
        Fx = F.T @ x  # shape [r]
        variance = x @ (d * x) + np.dot(Fx, Fx)
        ridge_term = ridge * np.dot(x, x)
        obj = 0.5 * (variance + ridge_term) - kappa * np.dot(mu, x)
        return obj
    
    def solve_qp_for_subset(indices):
        """Solve the QP for a given subset of indices using an active-set-like approach."""
        S = np.array(indices)
        ns = len(S)
        if ns == 0:
            return None, float('inf')
        
        mu_s = mu[S]
        F_s = F[S]  # shape [ns, r]
        d_s = d_eff[S]  # diagonal of Q for subset
        
        # Q_ss = F_s @ F_s^T + diag(d_s)
        # We solve: min 0.5 x^T Q x - kappa * mu^T x
        # s.t. 1^T x = 1, lb <= x <= ub
        
        if has_min_invest and l_min is not None:
            lb = l_min[S].copy()
        else:
            lb = np.zeros(ns)
        
        if u_max is not None:
            ub = u_max[S].copy()
        else:
            ub = np.ones(ns)
        
        # Check feasibility of bounds
        if np.sum(lb) > 1.0 + 1e-10:
            return None, float('inf')
        if np.sum(ub) < 1.0 - 1e-10:
            return None, float('inf')
        
        # Use iterative clipping / projected gradient or direct KKT solve
        # For small subsets (ns <= k, typically small), form Q_ss explicitly
        if ns <= 500:
            Q_ss = F_s @ F_s.T + np.diag(d_s)
            return solve_qp_dense(Q_ss, mu_s, lb, ub, ns, has_min_return, r_bar, mu, S)
        else:
            # Shouldn't happen since k is small
            Q_ss = F_s @ F_s.T + np.diag(d_s)
            return solve_qp_dense(Q_ss, mu_s, lb, ub, ns, has_min_return, r_bar, mu, S)
    
    def solve_qp_dense(Q, mu_s, lb, ub, ns, has_min_ret, r_bar_val, mu_full, S):
        """Solve dense QP with equality constraint sum=1 and box constraints using projected gradient."""
        kap = kappa
        # min 0.5 x^T Q x - kap * mu_s^T x
        # gradient: Q x - kap * mu_s
        
        # Initialize feasible point
        x = project_simplex_box(np.ones(ns) / ns, lb, ub)
        
        if has_min_ret and r_bar_val is not None:
            # Additional constraint: mu_s^T x >= r_bar
            return solve_qp_with_min_return(Q, mu_s, lb, ub, ns, r_bar_val, S, mu_full)
        
        # Projected gradient descent with line search
        best_obj = float('inf')
        best_x = x.copy()
        
        max_iter = 2000
        # Estimate Lipschitz constant
        L = np.linalg.norm(Q, ord=2) + 1.0
        step = 1.0 / L
        
        for it in range(max_iter):
            grad = Q @ x - kap * mu_s
            x_new = project_simplex_box(x - step * grad, lb, ub)
            
            obj = 0.5 * x_new @ Q @ x_new - kap * mu_s @ x_new
            if obj < best_obj:
                best_obj = obj
                best_x = x_new.copy()
            
            if np.max(np.abs(x_new - x)) < 1e-12:
                break
            x = x_new
        
        # Map back to full vector
        x_full = np.zeros(n)
        x_full[S] = best_x
        full_obj = compute_objective(x_full)
        return x_full, full_obj
    
    def solve_qp_with_min_return(Q, mu_s, lb, ub, ns, r_bar_val, S, mu_full):
        """Solve QP with additional minimum return constraint using projected gradient with dual on return constraint."""
        kap = kappa
        
        # Use augmented Lagrangian or simply project onto the feasible set
        # Feasible set: sum(x) = 1, lb <= x <= ub, mu_s^T x >= r_bar
        
        x = project_simplex_box(np.ones(ns) / ns, lb, ub)
        
        L = np.linalg.norm(Q, ord=2) + 1.0
        step = 1.0 / L
        
        # Dual variable for return constraint
        lam = 0.0
        
        best_obj = float('inf')
        best_x = None
        
        for outer in range(100):
            for it in range(200):
                grad = Q @ x - kap * mu_s - lam * mu_s
                x_new = project_simplex_box(x - step * grad, lb, ub)
                if np.max(np.abs(x_new - x)) < 1e-12:
                    break
                x = x_new
            
            ret_val = mu_s @ x
            violation = r_bar_val - ret_val
            
            if violation <= 1e-10:
                obj = 0.5 * x @ Q @ x - kap * mu_s @ x
                if obj < best_obj:
                    best_obj = obj
                    best_x = x.copy()
            
            if abs(violation) < 1e-8:
                break
            
            lam = max(0.0, lam + 0.1 * violation)
        
        if best_x is None:
            # Try harder - might be infeasible
            best_x = x
            best_obj = 0.5 * x @ Q @ x - kap * mu_s @ x
        
        x_full = np.zeros(n)
        x_full[S] = best_x
        full_obj = compute_objective(x_full)
        return x_full, full_obj
    
    def project_simplex_box(x, lb, ub):
        """Project x onto {x : sum(x)=1, lb <= x <= ub}."""
        ns = len(x)
        # Clip to bounds first
        x = np.clip(x, lb, ub)
        
        # Now project onto sum=1 while maintaining box constraints
        # Use iterative bisection on the dual variable for sum constraint
        
        def compute_sum(lam):
            return np.sum(np.clip(x - lam, lb, ub))
        
        s = np.sum(x)
        if abs(s - 1.0) < 1e-14:
            return x
        
        # Binary search for lambda such that sum(clip(x - lam, lb, ub)) = 1
        lo = -2.0
        hi = 2.0
        
        # Expand bounds if needed
        while compute_sum(lo) < 1.0:
            lo *= 2
        while compute_sum(hi) > 1.0:
            hi *= 2
        
        for _ in range(100):
            mid = (lo + hi) / 2.0
            s = compute_sum(mid)
            if s > 1.0:
                lo = mid
            else:
                hi = mid
            if abs(s - 1.0) < 1e-14:
                break
        
        lam = (lo + hi) / 2.0
        return np.clip(x - lam, lb, ub)
    
    def try_gurobi_solve(indices, time_remaining):
        """Try to solve the QP using Gurobi for the given subset."""
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return None, float('inf')
        
        S = np.array(indices)
        ns = len(S)
        
        mu_s = mu[S]
        F_s = F[S]  # [ns, r]
        d_s = d_eff[S]
        
        if has_min_invest and l_min is not None:
            lb = l_min[S]
        else:
            lb = np.zeros(ns)
        
        if u_max is not None:
            ub = u_max[S]
        else:
            ub = np.ones(ns)
        
        try:
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.setParam('TimeLimit', max(1, time_remaining))
            env.setParam('Threads', 1)
            env.start()
            
            m = gp.Model(env=env)
            m.setParam('OutputFlag', 0)
            
            x_vars = m.addVars(ns, lb=[lb[i] for i in range(ns)], 
                              ub=[ub[i] for i in range(ns)], name='x')
            
            # sum = 1
            m.addConstr(gp.quicksum(x_vars[i] for i in range(ns)) == 1.0)
            
            # min return constraint
            if has_min_return and r_bar is not None:
                m.addConstr(gp.quicksum(mu_s[i] * x_vars[i] for i in range(ns)) >= r_bar)
            
            # Objective: 0.5 x^T Q x - kappa * mu^T x
            # Q = F_s F_s^T + diag(d_s)
            # Use factor form: introduce y = F_s^T x, then x^T F_s F_s^T x = y^T y
            
            y_vars = m.addVars(r, lb=-GRB.INFINITY, name='y')
            for j in range(r):
                m.addConstr(y_vars[j] == gp.quicksum(F_s[i, j] * x_vars[i] for i in range(ns)))
            
            obj = gp.QuadExpr()
            # 0.5 * y^T y
            for j in range(r):
                obj += 0.5 * y_vars[j] * y_vars[j]
            # 0.5 * x^T diag(d_s) x
            for i in range(ns):
                obj += 0.5 * d_s[i] * x_vars[i] * x_vars[i]
            # -kappa * mu^T x
            for i in range(ns):
                obj -= kappa * mu_s[i] * x_vars[i]
            
            m.setObjective(obj, GRB.MINIMIZE)
            m.optimize()
            
            if m.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                x_sol = np.zeros(n)
                for i in range(ns):
                    x_sol[S[i]] = x_vars[i].X
                full_obj = compute_objective(x_sol)
                return x_sol, full_obj
            else:
                return None, float('inf')
        except Exception:
            return None, float('inf')
    
    # Strategy: 
    # 1. Score securities and select top-k candidates
    # 2. For small problems, try Gurobi MIQP
    # 3. For larger problems, use greedy + local search
    
    # Score each security: approximate marginal benefit
    # If we invest everything in security i: obj = 0.5*(Sigma_ii + ridge) - kappa*mu_i
    scores = 0.5 * (d + np.sum(F**2, axis=1) + ridge) - kappa * mu
    
    # Lower score = better
    sorted_indices = np.argsort(scores)
    
    best_obj = float('inf')
    best_x = np.zeros(n)
    # Simple uniform allocation on best security
    best_x[sorted_indices[0]] = 1.0
    best_obj = compute_objective(best_x)
    if logger:
        logger.log(best_obj)
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    # Try Gurobi full MIQP if n is manageable
    use_full_gurobi = True
    
    if use_full_gurobi and remaining > 2:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.setParam('Threads', 1)
            env.start()
            
            m = gp.Model(env=env)
            m.setParam('OutputFlag', 0)
            m.setParam('TimeLimit', max(1, remaining - 1))
            
            if has_min_invest and l_min is not None:
                lb_arr = l_min
            else:
                lb_arr = np.zeros(n)
            
            if u_max is not None:
                ub_arr = u_max
            else:
                ub_arr = np.ones(n)
            
            # Binary variables for selection
            z = m.addVars(n, vtype=GRB.BINARY, name='z')
            x_vars = m.addVars(n, lb=0.0, name='x')
            
            # Link x and z
            for i in range(n):
                m.addConstr(x_vars[i] <= ub_arr[i] * z[i])
                if has_min_invest and l_min is not None:
                    m.addConstr(x_vars[i] >= lb_arr[i] * z[i])
            
            # Cardinality constraint
            m.addConstr(gp.quicksum(z[i] for i in range(n)) <= k)
            
            # Sum = 1
            m.addConstr(gp.quicksum(x_vars[i] for i in range(n)) == 1.0)
            
            # Min return
            if has_min_return and r_bar is not None:
                m.addConstr(gp.quicksum(mu[i] * x_vars[i] for i in range(n)) >= r_bar)
            
            # Factor variables
            y_vars = m.addVars(r, lb=-GRB.INFINITY, name='y')
            for j in range(r):
                m.addConstr(y_vars[j] == gp.quicksum(F[i, j] * x_vars[i] for i in range(n)))
            
            obj = gp.QuadExpr()
            for j in range(r):
                obj += 0.5 * y_vars[j] * y_vars[j]
            for i in range(n):
                obj += 0.5 * d_eff[i] * x_vars[i] * x_vars[i]
            for i in range(n):
                obj -= kappa * mu[i] * x_vars[i]
            
            m.setObjective(obj, GRB.MINIMIZE)
            
            # Warm start with best solution found so far
            for i in range(n):
                x_vars[i].Start = best_x[i]
                z[i].Start = 1.0 if best_x[i] > 1e-10 else 0.0
            
            # Callback to log intermediate solutions
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    nonlocal best_obj, best_x
                    if obj_val < best_obj:
                        x_sol = np.array([model.cbGetSolution(x_vars[i]) for i in range(n)])
                        actual_obj = compute_objective(x_sol)
                        if actual_obj < best_obj:
                            best_obj = actual_obj
                            best_x = x_sol.copy()
                            if logger:
                                logger.log(best_obj)
            
            m.optimize(callback)
            
            if m.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
                if m.SolCount > 0:
                    x_sol = np.array([x_vars[i].X for i in range(n)])
                    actual_obj = compute_objective(x_sol)
                    if actual_obj < best_obj:
                        best_obj = actual_obj
                        best_x = x_sol.copy()
                        if logger:
                            logger.log(best_obj)
        except Exception as e:
            pass
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    # Also try heuristic approaches: solve continuous QP for top candidate subsets
    if remaining > 1:
        # Try several candidate subsets
        candidate_sets = []
        
        # Top-k by score
        candidate_sets.append(sorted_indices[:k].tolist())
        
        # Top-k by expected return
        ret_sorted = np.argsort(-mu)
        candidate_sets.append(ret_sorted[:k].tolist())
        
        # Top-k by return/risk ratio (Sharpe-like)
        vol = np.sqrt(d + np.sum(F**2, axis=1))
        sharpe = mu / (vol + 1e-10)
        sharpe_sorted = np.argsort(-sharpe)
        candidate_sets.append(sharpe_sorted[:k].tolist())
        
        for cset in candidate_sets:
            if time.time() - start_time > time_limit - 0.5:
                break
            remaining = time_limit - (time.time() - start_time)
            x_sol, obj_val = try_gurobi_solve(cset, min(remaining * 0.3, 10))
            if x_sol is not None and obj_val < best_obj:
                best_obj = obj_val
                best_x = x_sol.copy()
                if logger:
                    logger.log(best_obj)
    
    # Write solution
    solution = {
        "objective_value": float(best_obj),
        "x": best_x.tolist()
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)

if __name__ == '__main__':
    main()