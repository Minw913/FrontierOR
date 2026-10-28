import json
import argparse
import time
import numpy as np
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    k = instance['k']
    r = instance['r']
    gamma = instance['gamma']
    mu = np.array(instance['mu'])
    factor_loadings = np.array(instance['factor_loadings'])
    idiosyncratic_variance = np.array(instance['idiosyncratic_variance'])
    cons = instance['constraints']
    has_min_return = cons['has_min_return_constraint']
    r_bar = cons.get('r_bar')
    has_min_invest = cons['has_min_investment_constraint']
    l_min = np.array(cons['l_min_investment']) if cons.get('l_min_investment') is not None else None
    u_max = np.array(cons.get('u_max_investment')) if cons.get('u_max_investment') is not None else None

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    m = gp.Model("portfolio")
    m.setParam(GRB.Param.Threads, 1)
    m.setParam(GRB.Param.TimeLimit, args.time_limit)
    m.setParam(GRB.Param.MIPGap, 1e-4)
    m.setParam(GRB.Param.Heuristics, 0.5)
    m.setParam(GRB.Param.MIPFocus, 1)

    x = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="x")
    z = m.addVars(n, vtype=GRB.BINARY, name="z")
    f = m.addVars(r, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="f")

    # Exactly k selections
    m.addConstr(gp.quicksum(z[i] for i in range(n)) == k, "cardinality")

    # Linking constraints with big-M (using provided bounds)
    for i in range(n):
        if u_max is not None and u_max[i] > 0:
            ub_i = u_max[i]
        else:
            ub_i = 1.0
        m.addConstr(x[i] <= ub_i * z[i], f"link_ub_{i}")

        if has_min_invest and l_min is not None and l_min[i] > 0:
            m.addConstr(x[i] >= l_min[i] * z[i], f"link_lb_{i}")
        else:
            m.addConstr(x[i] >= 0, f"link_lb0_{i}")

    # Factor model auxiliary
    Fl = factor_loadings.T  # r x n
    for j in range(r):
        m.addConstr(f[j] == gp.quicksum(Fl[j, i] * x[i] for i in range(n)), f"factor_{j}")

    # Objective: (1/2) x^T Sigma x + (1/(2*gamma)) ||x||^2 - kappa * mu^T x
    # Sigma = F @ F.T + D, so x^T Sigma x = ||F.T x||^2 + x^T D x
    quad_terms = []
    for j in range(r):
        quad_terms.append(f[j] * f[j])
    for i in range(n):
        if idiosyncratic_variance[i] > 1e-12:
            quad_terms.append(idiosyncratic_variance[i] * x[i] * x[i])

    variance_term = 0.5 * gp.quicksum(quad_terms)
    ridge_term = (1.0 / (2.0 * gamma)) * gp.quicksum(x[i] * x[i] for i in range(n))
    return_term = gp.quicksum(mu[i] * x[i] for i in range(n))

    kappa = instance.get('kappa', 1)
    objective = variance_term + ridge_term - (kappa * return_term)
    m.setObjective(objective, GRB.MINIMIZE)

    # Minimum return constraint
    if has_min_return and r_bar is not None and kappa == 0:
        m.addConstr(return_term >= r_bar, "min_return")

    # Sum to 1
    m.addConstr(gp.quicksum(x[i] for i in range(n)) == 1.0, "budget")

    best_obj = None
    best_x = None

    def callback(model, where):
        nonlocal best_obj, best_x
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if best_obj is None or obj_val < best_obj:
                best_obj = obj_val
                x_sol = model.cbGetSolution(x)
                best_x = [float(x_sol[i]) for i in range(n)]
                if logger:
                    logger.log(float(obj_val))
        elif where == GRB.Callback.MIP:
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            if runtime > args.time_limit * 0.98:
                model.terminate()

    m.optimize(callback)

    # Extract solution
    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT or m.status == GRB.INTERRUPTED:
        if best_x is None:
            x_sol = m.getAttr('X', x)
            best_x = [float(x_sol[i]) for i in range(n)]
            best_obj = float(m.ObjVal)
    else:
        # If no solution found, return uniform as fallback (very unlikely)
        best_x = [1.0 / n] * n
        # Compute objective for fallback
        x_arr = np.array(best_x)
        f_arr = factor_loadings.T @ x_arr
        var = 0.5 * (np.dot(f_arr, f_arr) + np.dot(idiosyncratic_variance, x_arr**2))
        ridge = (1.0 / (2.0 * gamma)) * np.dot(x_arr, x_arr)
        ret = np.dot(mu, x_arr)
        best_obj = var + ridge - kappa * ret

    # Enforce sum-to-1 numerically
    total = sum(best_x)
    if abs(total - 1.0) > 1e-8:
        best_x = [v / total for v in best_x]

    result = {
        "objective_value": float(best_obj),
        "x": best_x
    }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()