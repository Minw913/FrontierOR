import json
import argparse
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def compute_objective(x, F, ivar, mu, gamma, kappa):
    x = np.array(x, dtype=float)
    z = F.T @ x
    var_term = 0.5 * np.dot(z, z)
    diag_term = 0.5 * np.sum(ivar * x**2)
    ridge_term = 0.5 / gamma * np.sum(x**2) if gamma > 1e-9 else 0.0
    ret_term = kappa * np.dot(mu, x)
    return float(var_term + diag_term + ridge_term - ret_term)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
    k = data['k']
    r = data['r']
    gamma = data.get('gamma', 0.0)
    kappa = data.get('kappa', 0)
    mu = np.array(data.get('mu', [0.0]*n), dtype=float)
    factor_loadings = np.array(data.get('factor_loadings', [[0.0]*r for _ in range(n)]), dtype=float)
    ivar = np.array(data.get('idiosyncratic_variance', [0.0]*n), dtype=float)

    constrs = data.get('constraints', {})
    has_min_ret = constrs.get('has_min_return_constraint', False)
    r_bar = constrs.get('r_bar', None)
    has_min_inv = constrs.get('has_min_investment_constraint', False)
    l_min = constrs.get('l_min_investment', None)
    u_max = constrs.get('u_max_investment', None)

    l_min_arr = np.array(l_min, dtype=float) if l_min is not None else np.zeros(n)
    u_max_arr = np.array(u_max, dtype=float) if u_max is not None else np.ones(n)

    model = gp.Model('portfolio')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', max(1, args.time_limit - 2))

    x = model.addVars(n, lb=0.0, ub=1.0, name='x')
    y = model.addVars(n, vtype=GRB.BINARY, name='y')
    z = model.addVars(r, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z')

    ridge = 1.0 / gamma if gamma > 1e-9 else 0.0
    
    obj = gp.QuadExpr()
    for j in range(r):
        obj.add(0.5 * z[j] * z[j])
    
    for i in range(n):
        obj.add(0.5 * (ivar[i] + ridge) * x[i] * x[i])
        obj.add(-kappa * mu[i] * x[i])
        
    model.setObjective(obj, GRB.MINIMIZE)

    for j in range(r):
        expr = gp.LinExpr()
        for i in range(n):
            if abs(factor_loadings[i][j]) > 1e-12:
                expr += factor_loadings[i][j] * x[i]
        model.addConstr(z[j] == expr, name=f'factor_{j}')

    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1.0, name='budget')
    model.addConstr(gp.quicksum(y[i] for i in range(n)) <= k, name='cardinality')

    for i in range(n):
        ub = u_max_arr[i]
        lb = l_min_arr[i]
        if lb > 1e-12:
            model.addConstr(x[i] >= lb * y[i], name=f'lb_{i}')
        model.addConstr(x[i] <= ub * y[i], name=f'ub_{i}')

    if has_min_ret and r_bar is not None:
        model.addConstr(gp.quicksum(mu[i] * x[i] for i in range(n)) >= r_bar, name='min_ret')

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals = model.cbGetSolution([x[i] for i in range(n)])
            sol = {'x': list(x_vals)}
            if logger:
                logger.log_solution(obj_val, sol)

    model.optimize(callback=callback)

    if model.SolCount > 0:
        x_vals = [x[i].X for i in range(n)]
        obj_val = model.ObjVal
    else:
        # Fallback heuristic
        x_vals = np.zeros(n)
        remaining = 1.0
        selected = 0
        if has_min_ret:
            idx = np.argsort(-mu)
        else:
            idx = np.argsort(ivar)
        for i in idx:
            if selected >= k or remaining <= 1e-6:
                break
            ub = u_max_arr[i]
            lb = l_min_arr[i]
            if lb > remaining:
                continue
            val = min(ub, remaining)
            if val < lb:
                continue
            x_vals[i] = val
            remaining -= val
            selected += 1
        if remaining > 1e-6:
            for i in idx:
                if x_vals[i] > 0:
                    continue
                ub = u_max_arr[i]
                lb = l_min_arr[i]
                if lb > remaining:
                    continue
                val = min(ub, remaining)
                if val < lb:
                    continue
                x_vals[i] = val
                remaining -= val
                selected += 1
                if remaining <= 1e-6:
                    break
        if abs(sum(x_vals) - 1.0) > 1e-6 and sum(x_vals) > 0:
            x_vals = x_vals / sum(x_vals)
        obj_val = compute_objective(x_vals, factor_loadings, ivar, mu, gamma, kappa)

    # Clean up solution
    cleaned_x = []
    for v in x_vals:
        if abs(v) < 1e-9:
            cleaned_x.append(0.0)
        else:
            cleaned_x.append(v)
    s = sum(cleaned_x)
    if s > 0:
        cleaned_x = [v/s for v in cleaned_x]

    if model.SolCount > 0:
        final_obj = model.ObjVal
    else:
        final_obj = compute_objective(cleaned_x, factor_loadings, ivar, mu, gamma, kappa)

    solution = {
        'objective_value': final_obj,
        'x': cleaned_x
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()