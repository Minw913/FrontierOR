import argparse
import json
import time
import sys

# Try importing Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False

# Try importing Scipy and Numpy for fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

try:
    from scipy.optimize import linprog, minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

def calculate_objective(x, c, q, cost_type):
    obj = 0.0
    n = len(x)
    m = len(x[0]) if n > 0 else 0
    for i in range(n):
        for j in range(m):
            val = x[i][j]
            if cost_type == 'quadratic':
                obj += c[i][j] * val + q[i][j] * val * val
            else:
                obj += c[i][j] * val
    return obj

def solve_lp_greedy(c, d, s, U):
    n = len(s)
    m = len(d)
    x = [[0.0]*m for _ in range(n)]
    s_left = list(s)
    d_left = list(d)
    arcs = []
    for i in range(n):
        for j in range(m):
            arcs.append((c[i][j], i, j))
    arcs.sort()
    for cost, i, j in arcs:
        flow = min(s_left[i], d_left[j], U)
        if flow > 0:
            x[i][j] = flow
            s_left[i] -= flow
            d_left[j] -= flow
    return x

def solve_qp_gd(c, q, d, s, U, time_limit):
    n = len(s)
    m = len(d)
    lam = [0.0] * m
    mu = [0.0] * n
    q_eff = [[q[i][j] + 1e-6 for j in range(m)] for i in range(n)]
    min_q = min(min(row) for row in q_eff)
    if min_q <= 0: min_q = 1e-6
    step = 1.0 / (2 * min_q) * 0.5
    
    start = time.time()
    while time.time() - start < time_limit - 1:
        x = [[0.0]*m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                val = (lam[j] + mu[i] - c[i][j]) / (2 * q_eff[i][j])
                if val < 0: val = 0.0
                if val > U: val = U
                x[i][j] = val
        g_lam = [0.0] * m
        g_mu = [0.0] * n
        for j in range(m):
            g_lam[j] = d[j] - sum(x[i][j] for i in range(n))
        for i in range(n):
            g_mu[i] = s[i] - sum(x[i][j] for j in range(m))
        for j in range(m):
            lam[j] += step * g_lam[j]
        for i in range(n):
            mu[i] = max(0.0, mu[i] + step * g_mu[i])
        if max(abs(v) for v in g_lam) < 1e-4 and max(abs(v) for v in g_mu) < 1e-4:
            break
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    m = instance['m']
    cost_type = instance['cost_type']
    supplies = instance['supplies']
    demands = instance['demands']
    arc_capacity = instance['arc_capacity']
    linear_costs = instance.get('linear_costs', [[0.0]*m for _ in range(n)])
    quadratic_costs = instance.get('quadratic_costs', [[0.0]*m for _ in range(n)])

    start_time = time.time()
    solution_dict = {"objective_value": 0.0, "flows": {}}
    
    # Initial zero solution
    for i in range(n):
        for j in range(m):
            solution_dict["flows"][f"x_{i}_{j}"] = 0.0
            
    # Try Gurobi
    use_gurobi = GUROBI_AVAILABLE
    if use_gurobi:
        try:
            env = gp.Env()
            env.setParam('OutputFlag', 0)
            env.setParam('Seed', 0)
            env.setParam('MIPGap', 1e-4)
            env.setParam('NumericFocus', 0)
            env.setParam('Threads', 1)
            env.start()
            
            model = gp.Model(env)
            
            x = {}
            for i in range(n):
                for j in range(m):
                    x[i, j] = model.addVar(0.0, arc_capacity, 0.0, GRB.CONTINUOUS, f"x_{i}_{j}")
                    
            for j in range(m):
                model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == demands[j], f"demand_{j}")
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, j] for j in range(m)) <= supplies[i], f"supply_{i}")
                
            if cost_type == 'quadratic':
                obj = gp.QuadExpr()
                for i in range(n):
                    for j in range(m):
                        lc = linear_costs[i][j]
                        qc = quadratic_costs[i][j]
                        if lc != 0:
                            obj.add(lc * x[i, j])
                        if qc != 0:
                            obj.add(qc * x[i, j] * x[i, j])
                model.setObjective(obj, GRB.MINIMIZE)
                try:
                    model.Params.NonConvex = 2
                except:
                    pass
            else:
                obj = gp.LinExpr()
                for i in range(n):
                    for j in range(m):
                        lc = linear_costs[i][j]
                        if lc != 0:
                            obj.add(lc * x[i, j])
                model.setObjective(obj, GRB.MINIMIZE)
                
            time_left = max(1, int(args.time_limit - (time.time() - start_time) - 1))
            model.Params.TimeLimit = time_left
            
            model.optimize()
            
            if model.SolCount > 0:
                obj_val = float(model.ObjVal)
                flows_dict = {}
                for i in range(n):
                    for j in range(m):
                        flows_dict[f"x_{i}_{j}"] = float(x[i, j].X)
                solution_dict = {"objective_value": obj_val, "flows": flows_dict}
                if logger:
                    logger.log_solution(obj_val, solution_dict)
            else:
                use_gurobi = False
        except Exception as e:
            use_gurobi = False

    if not use_gurobi:
        if SCIPY_AVAILABLE and NUMPY_AVAILABLE:
            c_np = np.array(linear_costs, dtype=float)
            q_np = np.array(quadratic_costs, dtype=float)
            d_np = np.array(demands, dtype=float)
            s_np = np.array(supplies, dtype=float)
            
            time_left = max(1.0, args.time_limit - (time.time() - start_time) - 1.0)
            
            if cost_type != 'quadratic':
                # LP using scipy
                c_flat = c_np.flatten()
                A_eq = np.zeros((m, n*m))
                for j in range(m):
                    for i in range(n):
                        A_eq[j, i*m + j] = 1.0
                A_ub = np.zeros((n, n*m))
                for i in range(n):
                    for j in range(m):
                        A_ub[i, i*m + j] = 1.0
                bounds = [(0, arc_capacity) for _ in range(n*m)]
                
                try:
                    res = linprog(c_flat, A_ub=A_ub, b_ub=s_np, A_eq=A_eq, b_eq=d_np, bounds=bounds, method='highs', options={'time_limit': time_left})
                    if res.success:
                        x_np = res.x.reshape(n, m)
                        obj_val = float(np.sum(c_np * x_np))
                        flows_dict = {f"x_{i}_{j}": float(x_np[i, j]) for i in range(n) for j in range(m)}
                        solution_dict = {"objective_value": obj_val, "flows": flows_dict}
                        if logger:
                            logger.log_solution(obj_val, solution_dict)
                except Exception:
                    pass
            else:
                # QP using Dual Ascent + L-BFGS-B
                q_reg = q_np + 1e-6
                def neg_dual(y):
                    lam = y[:m]
                    mu = y[m:]
                    x = np.clip((lam[None, :] + mu[:, None] - c_np) / (2 * q_reg), 0, arc_capacity)
                    val = np.sum(c_np * x + q_reg * x**2 - (lam[None, :] + mu[:, None]) * x) + np.sum(lam * d_np) + np.sum(mu * s_np)
                    return -val

                def neg_dual_grad(y):
                    lam = y[:m]
                    mu = y[m:]
                    x = np.clip((lam[None, :] + mu[:, None] - c_np) / (2 * q_reg), 0, arc_capacity)
                    grad_lam = d_np - np.sum(x, axis=0)
                    grad_mu = s_np - np.sum(x, axis=1)
                    return np.concatenate([-grad_lam, -grad_mu])

                bounds_dual = [(None, None)] * m + [(0, None)] * n
                y0 = np.zeros(m + n)
                
                try:
                    res = minimize(neg_dual, y0, jac=neg_dual_grad, bounds=bounds_dual, method='L-BFGS-B', options={'maxiter': 10000, 'ftol': 1e-10, 'gtol': 1e-8})
                    lam = res.x[:m]
                    mu = res.x[m:]
                    x_np = np.clip((lam[None, :] + mu[:, None] - c_np) / (2 * q_reg), 0, arc_capacity)
                    
                    # Correction LP to restore exact feasibility
                    b_ub_corr = s_np - np.sum(x_np, axis=1)
                    b_ub_corr = np.maximum(0, b_ub_corr)
                    b_eq_corr = d_np - np.sum(x_np, axis=0)
                    
                    obj_corr = (c_np + 2 * q_np * x_np).flatten()
                    x_flat = np.clip(x_np.flatten(), 0, arc_capacity)
                    bounds_corr = [(-x_flat[k], arc_capacity - x_flat[k]) for k in range(n*m)]
                    
                    A_eq = np.zeros((m, n*m))
                    for j in range(m):
                        for i in range(n):
                            A_eq[j, i*m + j] = 1.0
                    A_ub = np.zeros((n, n*m))
                    for i in range(n):
                        for j in range(m):
                            A_ub[i, i*m + j] = 1.0
                            
                    res_corr = linprog(obj_corr, A_ub=A_ub, b_ub=b_ub_corr, A_eq=A_eq, b_eq=b_eq_corr, bounds=bounds_corr, method='highs', options={'time_limit': max(1.0, time_left - 5.0)})
                    if res_corr.success:
                        dx = res_corr.x.reshape(n, m)
                        x_np = x_np + dx
                        
                    obj_val = float(np.sum(c_np * x_np + q_np * x_np**2))
                    flows_dict = {f"x_{i}_{j}": float(x_np[i, j]) for i in range(n) for j in range(m)}
                    solution_dict = {"objective_value": obj_val, "flows": flows_dict}
                    if logger:
                        logger.log_solution(obj_val, solution_dict)
                except Exception:
                    pass
        else:
            # Pure Python fallback if no Gurobi or Scipy
            time_left = max(1.0, args.time_limit - (time.time() - start_time) - 1.0)
            if cost_type != 'quadratic':
                x = solve_lp_greedy(linear_costs, demands, supplies, arc_capacity)
            else:
                x = solve_qp_gd(linear_costs, quadratic_costs, demands, supplies, arc_capacity, time_left)
            obj_val = calculate_objective(x, linear_costs, quadratic_costs, cost_type)
            flows_dict = {f"x_{i}_{j}": float(x[i][j]) for i in range(n) for j in range(m)}
            solution_dict = {"objective_value": float(obj_val), "flows": flows_dict}
            if logger:
                logger.log_solution(float(obj_val), solution_dict)

    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)

if __name__ == '__main__':
    main()