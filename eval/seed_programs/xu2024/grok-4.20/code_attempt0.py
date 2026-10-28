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

    n = data['n']
    k = data['k']
    b = data['b']
    r = np.array(data['r'])
    Q = np.array(data['Q'])

    # Precompute diagonal and some statistics for big-M and initial solution
    diag = np.diag(Q)
    max_var = float(np.max(diag))
    min_var = float(np.min(diag))
    bigM = 10.0 * (1.0 + max_var)  # Conservative bound on |x_i|

    # Initial heuristic: top-k by Sharpe-like ratio (r_i / sqrt(Q_ii))
    scores = r / np.sqrt(np.maximum(diag, 1e-8))
    idx = np.argsort(-scores)[:k]
    x_init = np.zeros(n)
    selected = idx[:min(k, n)]
    x_init[selected] = 1.0 / len(selected)
    init_ret = float(np.dot(r, x_init))
    if init_ret < b - 1e-6:
        # Scale up if needed (risky but often helps)
        scale = (b + 0.01) / (init_ret + 1e-8)
        x_init *= scale
        x_init = np.clip(x_init, -5.0, 5.0)
        total = np.sum(x_init)
        if abs(total) > 1e-8:
            x_init /= total
    init_obj = 0.5 * float(x_init @ Q @ x_init)
    best_obj = init_obj
    best_x = x_init.copy()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    if logger:
        logger.log(float(best_obj))

    start_time = time.time()
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time < 0.5:
        # write initial solution
        write_solution(args.solution_path, best_obj, best_x.tolist())
        return

    model = gp.Model("Portfolio")
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, max(1.0, args.time_limit - 5.0))
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.Heuristics, 0.8)
    model.setParam(GRB.Param.MIPFocus, 1)
    model.setParam(GRB.Param.Cutoff, best_obj * 1.1)

    x = model.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    y = model.addMVar(n, lb=-bigM, ub=bigM, name="y")  # y = Qx

    model.addConstr(x.sum() == 1.0, "budget")
    model.addConstr(r @ x >= b, "return")
    model.addConstrs((z[i] == 1) >> (x[i] * x[i] <= bigM * bigM) for i in range(n))  # dummy to link
    for i in range(n):
        model.addConstr(x[i] <= bigM * z[i])
        model.addConstr(x[i] >= -bigM * z[i])
    model.addConstr(z.sum() <= k, "cardinality")

    # Linearized quadratic: y = Q @ x
    for i in range(n):
        model.addConstr(y[i] == gp.quicksum(Q[i, j] * x[j] for j in range(n)))

    obj = 0.5 * (x @ y)
    model.setObjective(obj, GRB.MINIMIZE)

    # Provide initial solution
    init_z = [1.0 if abs(best_x[i]) > 1e-8 else 0.0 for i in range(n)]
    model.cbSetSolution([x[i] for i in range(n)] + [z[i] for i in range(n)],
                        list(best_x) + init_z)
    model.update()

    def cb(model, where):
        nonlocal best_obj, best_x
        if where == GRB.Callback.MIPSOL:
            objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if objval < best_obj - 1e-8:
                xsol = model.cbGetSolution(x)
                best_obj = objval
                best_x = np.array(xsol)
                if logger:
                    logger.log(float(best_obj))
                model.cbSetCutoff(best_obj - 1e-6)

    model.optimize(cb)

    runtime = time.time() - start_time
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        if model.SolCount > 0:
            xval = model.getAttr('X', x)
            objval = model.ObjVal
            if objval < best_obj:
                best_obj = objval
                best_x = np.array([xval[i] for i in range(n)])
                if logger:
                    logger.log(float(best_obj))

    # Final polishing: solve continuous relaxation with fixed support from best solution
    support = [i for i in range(n) if abs(best_x[i]) > 1e-7]
    if len(support) > 0 and (time.time() - start_time) < args.time_limit - 0.5:
        try:
            m2 = gp.Model("Polish")
            m2.setParam(GRB.Param.Threads, 1)
            m2.setParam(GRB.Param.OutputFlag, 0)
            m2.setParam(GRB.Param.TimeLimit, max(0.5, args.time_limit - (time.time() - start_time)))
            x2 = m2.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            m2.addConstr(x2.sum() == 1.0)
            m2.addConstr(r @ x2 >= b)
            for i in range(n):
                if i not in support:
                    m2.addConstr(x2[i] == 0.0)
            y2 = m2.addMVar(n)
            for i in range(n):
                m2.addConstr(y2[i] == gp.quicksum(Q[i, j] * x2[j] for j in range(n)))
            m2.setObjective(0.5 * (x2 @ y2), GRB.MINIMIZE)
            m2.optimize()
            if m2.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m2.SolCount > 0:
                new_obj = m2.ObjVal
                if new_obj < best_obj - 1e-8:
                    best_obj = new_obj
                    best_x = np.array([x2.X[i] for i in range(n)])
                    if logger:
                        logger.log(float(best_obj))
        except:
            pass

    write_solution(args.solution_path, best_obj, best_x.tolist())

def write_solution(solution_path, objective_value, x):
    solution = {
        "objective_value": float(objective_value),
        "x": [float(xi) for xi in x]
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()