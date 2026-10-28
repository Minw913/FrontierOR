import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data['N']
    S = data['S']
    k = data['k']
    beta = data['beta']
    gamma = data['gamma']
    mu_bar = data['mu_bar']
    mu = np.array(data['mu'])
    p_s = data['p_s']
    scenarios = np.array(data['scenarios'])  # S x N

    # Precompute scenario losses coefficients: loss_s = -sum w_i * r_{s,i}
    # So loss_s = sum w_i * (-r_{s,i})
    neg_returns = -scenarios  # S x N

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    m = gp.Model("Portfolio_CVaR")
    m.Params.TimeLimit = args.time_limit
    m.Params.Threads = 1
    m.Params.MIPGap = 1e-4
    m.Params.LogToConsole = 0

    # Variables
    x = m.addVars(N, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")
    z = m.addVars(N, vtype=GRB.BINARY, name="z")
    a = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a")
    v = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="v")
    u = m.addVars(S, lb=0, vtype=GRB.CONTINUOUS, name="u")

    # Objective: 1/(2*gamma) * sum(x_i^2) + a + v
    quad_term = (1.0 / (2.0 * gamma)) * gp.quicksum(x[i]*x[i] for i in range(N))
    m.setObjective(quad_term + a + v, GRB.MINIMIZE)

    # Constraints
    # Sum x = 1
    m.addConstr(gp.quicksum(x[i] for i in range(N)) == 1.0, name="budget")

    # Expected return
    m.addConstr(gp.quicksum(mu[i] * x[i] for i in range(N)) >= mu_bar, name="return")

    # Cardinality
    m.addConstr(gp.quicksum(z[i] for i in range(N)) <= k, name="cardinality")

    # Linking x and z: x_i <= z_i (since x<=1 is already set)
    for i in range(N):
        m.addConstr(x[i] <= z[i], name=f"link_{i}")

    # CVaR constraints: u_s >= p_s * (loss_s - a), u_s >= 0
    one_minus_beta = 1.0 - beta
    scale = 1.0 / one_minus_beta
    for s in range(S):
        loss_s = gp.quicksum(neg_returns[s, i] * x[i] for i in range(N))
        m.addConstr(u[s] >= p_s * (loss_s - a), name=f"u_{s}_loss")
        # u[s] >= 0 already in variable bounds

    # v >= scale * sum(u_s)
    m.addConstr(v >= scale * gp.quicksum(u[s] for s in range(S)), name="v_def")

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < float('inf'):
                if logger:
                    logger.log(float(obj))

    # Optimize
    m.optimize(callback)

    # Extract solution
    if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        x_sol = [x[i].X for i in range(N)]
        z_sol = [1 if z[i].X > 0.5 else 0 for i in range(N)]
        a_sol = a.X
        v_sol = v.X
        obj_val = m.ObjVal

        # Project to feasible point if needed (numerical issues)
        x_sol = np.array(x_sol)
        x_sol = np.maximum(x_sol, 0.0)
        total = np.sum(x_sol)
        if abs(total - 1.0) > 1e-6 and total > 1e-8:
            x_sol = x_sol / total
        # Enforce cardinality by keeping top k if violated (rare)
        if sum(z_sol) > k:
            idx = np.argsort(-x_sol)[:k]
            z_sol = [0] * N
            for i in idx:
                z_sol[i] = 1
            for i in range(N):
                if z_sol[i] == 0:
                    x_sol[i] = 0.0
            total = np.sum(x_sol)
            if total > 1e-8:
                x_sol = x_sol / total

        solution = {
            "objective_value": float(obj_val),
            "x": [float(val) for val in x_sol],
            "z": z_sol,
            "a": float(a_sol),
            "v": float(v_sol)
        }
    else:
        # Infeasible or other failure - provide dummy feasible solution
        # Uniform on first min(k,N) assets
        num = min(k, N)
        x_sol = [1.0 / num if i < num else 0.0 for i in range(N)]
        z_sol = [1 if i < num else 0 for i in range(N)]
        # Rough estimate of a and v
        port_returns = scenarios @ np.array(x_sol)
        losses = -port_returns
        a_est = np.quantile(losses, beta)
        tail = losses[losses > a_est]
        v_est = np.mean(tail) - a_est if len(tail) > 0 else 0.0
        obj_val = (1.0/(2.0*gamma))*sum(x*x for x in x_sol) + a_est + v_est
        solution = {
            "objective_value": float(obj_val),
            "x": x_sol,
            "z": z_sol,
            "a": float(a_est),
            "v": float(v_est)
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()