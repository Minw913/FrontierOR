import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        print("SolutionLogger not found. Proceeding without logging.")
        logger = None

    with open(args.instance_path) as f:
        data = json.load(f)
    
    N = data['N']
    S = data['S']
    k = data['k']
    beta = data['beta']
    gamma = data['gamma']
    mu_bar = data['mu_bar']
    mu = data['mu']
    p_s_val = data['p_s']
    scenarios = data['scenarios']

    try:
        m = gp.Model()
        m.setParam('Threads', 1)
        m.setParam('TimeLimit', args.time_limit)

        x = m.addMVar(N, lb=0.0, ub=1.0, name="x")
        z = m.addVars(N, vtype=GRB.BINARY, name="z")
        a_var = m.addVar(lb=-GRB.INFINITY, name="a")
        v_var = m.addVar(lb=0.0, name="v")
        y = m.addMVar(S, lb=0.0, name="y")

        m.addConstr(x.sum() == 1, name="sum_weights")
        for i in range(N):
            m.addConstr(x[i] <= z[i], name=f"link_{i}")
        m.addConstr(gp.quicksum(z[i] for i in range(N)) <= k, name="cardinality")
        m.addConstr(gp.quicksum(mu[i] * x[i] for i in range(N)) >= mu_bar, name="exp_return")

        for s in range(S):
            scen_expr = gp.LinExpr()
            for i in range(N):
                scen_expr += scenarios[s][i] * x[i]
            m.addConstr(y[s] >= -scen_expr - a_var, name=f"cvar_scen_{s}")

        cvar_factor = p_s_val / (1.0 - beta)
        m.addConstr(v_var >= cvar_factor * y.sum(), name="cvar_def")

        reg_term = (1.0 / (2.0 * gamma)) * gp.quicksum(x[i] * x[i] for i in range(N))
        m.setObjective(reg_term + a_var + v_var)

        if logger:
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj_val)
            m.optimize(callback)
        else:
            m.optimize()

        if m.SolCount > 0:
            obj_val = m.ObjVal
            x_val = [x[i].X for i in range(N)]
            z_val = [int(round(z[i].X)) for i in range(N)]
            a_val = a_var.X
            v_val = v_var.X
        else:
            x_val, z_val, obj_val, a_val, v_val = fallback_solution(N, k, mu_bar, mu, gamma, S, scenarios, beta, p_s_val)
            
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        x_val, z_val, obj_val, a_val, v_val = fallback_solution(N, k, mu_bar, mu, gamma, S, scenarios, beta, p_s_val)

    solution = {
        "objective_value": obj_val,
        "x": x_val,
        "z": z_val,
        "a": a_val,
        "v": v_val
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def fallback_solution(N, k, mu_bar, mu, gamma, S, scenarios, beta, p_s_val):
    sorted_indices = sorted(range(N), key=lambda i: -mu[i])
    selected_indices = sorted_indices[:k]
    x_val = [0.0] * N
    z_val = [0] * N
    exp_ret = 0.0
    for idx in selected_indices:
        x_val[idx] = 1.0 / k
        z_val[idx] = 1
        exp_ret += mu[idx] / k
    if exp_ret < mu_bar:
        max_idx = sorted_indices[0]
        x_val = [0.0] * N
        z_val = [0] * N
        x_val[max_idx] = 1.0
        z_val[max_idx] = 1
        exp_ret = mu[max_idx]
    
    losses = []
    for s in range(S):
        ret = 0.0
        for i in range(N):
            ret += scenarios[s][i] * x_val[i]
        losses.append(-ret)
    
    losses_sorted = sorted(losses)
    best_f = float('inf')
    best_a = losses_sorted[0]
    for a_candidate in losses_sorted:
        sum_excess = 0.0
        for loss in losses:
            if loss > a_candidate:
                sum_excess += (loss - a_candidate)
        v_candidate = (p_s_val * sum_excess) / (1 - beta)
        f_val = a_candidate + v_candidate
        if f_val < best_f:
            best_f = f_val
            best_a = a_candidate
    a_val = best_a
    sum_excess = 0.0
    for loss in losses:
        if loss > a_val:
            sum_excess += (loss - a_val)
    v_val = (p_s_val * sum_excess) / (1 - beta)
    reg_term = (1.0 / (2.0 * gamma)) * sum(x_val[i] ** 2 for i in range(N))
    obj_val = reg_term + a_val + v_val
    return x_val, z_val, obj_val, a_val, v_val

if __name__ == "__main__":
    main()