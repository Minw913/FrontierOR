import argparse
import json
import gurobipy as gp
from solution_logger import SolutionLogger

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
    kappa = instance['kappa']
    mu = instance['mu']
    factor_loadings = instance['factor_loadings']
    idiosyncratic_variance = instance['idiosyncratic_variance']
    constraints = instance['constraints']
    
    model = gp.Model()
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)
    
    x = model.addVars(n, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="x")
    z = model.addVars(n, vtype=gp.GRB.BINARY, name="z")
    
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1, name="sum_one")
    model.addConstr(gp.quicksum(z[i] for i in range(n)) <= k, name="cardinality")
    model.addConstrs((x[i] <= z[i] for i in range(n)), name="link")
    
    if constraints['has_min_investment_constraint']:
        l_min = constraints['l_min_investment']
        u_max = constraints['u_max_investment']
        for i in range(n):
            model.addConstr(x[i] >= l_min[i] * z[i], name=f"min_inv_{i}")
            model.addConstr(x[i] <= u_max[i] * z[i], name=f"max_inv_{i}")
    
    if constraints['has_min_return_constraint']:
        r_bar = constraints['r_bar']
        model.addConstr(gp.quicksum(mu[i] * x[i] for i in range(n)) >= r_bar, name="min_return")
    
    quad_term = 0.0
    for j in range(r):
        expr = gp.LinExpr(0.0)
        for i in range(n):
            fl_ij = factor_loadings[i][j]
            expr.addTerms(fl_ij, x[i])
        quad_term += expr * expr
    
    term1 = 0.5 * quad_term
    term2 = 0.5 * gp.quicksum((idiosyncratic_variance[i] + 1.0 / gamma) * x[i] * x[i] for i in range(n))
    linear_term = gp.LinExpr(0.0)
    if kappa != 0:
        for i in range(n):
            linear_term += -kappa * mu[i] * x[i]
    
    model.setObjective(term1 + term2 + linear_term, gp.GRB.MINIMIZE)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    if logger:
        def callback(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                obj_val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
        model.optimize(callback)
    else:
        model.optimize()
    
    if model.SolCount > 0:
        x_sol = [x[i].X for i in range(n)]
        obj_value = model.ObjVal
    else:
        if constraints['has_min_investment_constraint']:
            l_min_vec = constraints['l_min_investment']
            u_max_vec = constraints['u_max_investment']
            candidates = [i for i in range(n) if 0.0 < l_min_vec[i] <= 1.0 <= u_max_vec[i]]
        else:
            candidates = list(range(n))
        if not candidates:
            candidates = list(range(n))
        best_i = -1
        best_mu = -1e18
        for i in candidates:
            current_mu = mu[i]
            if current_mu > best_mu:
                best_mu = current_mu
                best_i = i
        x_sol = [0.0] * n
        x_sol[best_i] = 1.0
        total_var = 0.0
        for j in range(r):
            s = 0.0
            for i in range(n):
                s += factor_loadings[i][j] * x_sol[i]
            total_var += s * s
        total_var *= 0.5
        diag_sum = 0.0
        for i in range(n):
            diag_sum += idiosyncratic_variance[i] * (x_sol[i]**2) + (1.0 / gamma) * (x_sol[i]**2)
        total_var += 0.5 * diag_sum
        return_term = 0.0
        if kappa != 0:
            for i in range(n):
                return_term += kappa * mu[i] * x_sol[i]
        obj_value = total_var - return_term
    
    out_data = {
        "objective_value": obj_value,
        "x": x_sol
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(out_data, f, indent=2)

if __name__ == "__main__":
    main()