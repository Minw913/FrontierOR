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

    diag = np.diag(Q)
    max_var = float(np.max(diag))
    bigM = 10.0 * (1.0 + max_var)

    # Heuristic warm-start
    scores = r / np.sqrt(np.maximum(diag, 1e-8))
    idx = np.argsort(-scores)[:k]
    x_init = np.zeros(n)
    selected = idx[:min(k, n)]
    x_init[selected] = 1.0 / len(selected)
    init_ret = float(np.dot(r, x_init))
    if init_ret < b - 1e-6:
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
    remaining = args.time_limit - (time.time() - start_time)
    if remaining < 0.5:
        write_solution(args.solution_path, best_obj, best_x.tolist())
        return

    model = gp.Model("Portfolio")
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, max(1.0, args.time_limit - 3.0))
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.Heuristics, 0.8)
    model.setParam(GRB.Param.MIPFocus, 1)

    x = model.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    y = model.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")

    model.addConstr(x.sum() == 1.0, "budget")
    model.addConstr(r @ x >= b, "return")
    for i in range(n):
        model.addConstr(x[i] <= bigM * z[i], f"ub_{i}")
        model.addConstr(x[i] >= -bigM * z[i], f"lb_{i}")
    model.addConstr(gp.quicksum(z[i] for i in range(n)) <= k, "cardinality")

    for i in range(n):
        model.addConstr(y[i] == gp.quicksum(Q[i, j] * x[j] for j in range(n)), f"quad_{i}")

    model.setObjective(0.5 * (x @ y), GRB.MINIMIZE)

    # Set initial solution (without setPWLObj)
    init_z_val = [1.0 if abs(best_x[i]) > 1e-8 else 0.0 for i in range(n)]
    for i in range(n):
        x[i].Start = float(best_x[i])
        z[i].Start = float(init_z_val[i])
    model.update()

    def cb(model, where):
        nonlocal best_obj, best_x
        if where == GRB.Callback.MIPSOL:
            objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if objval < best_obj - 1e-8:
                xsol = model.cbGetSolution(x)
                best_obj = float(objval)
                best_x = np.array(xsol, dtype=float)
                if logger:
                    logger.log(best_obj)
                model.cbSetCutoff(best_obj - 1e-6)

    try:
        model.optimize(cb)
    except Exception:
        pass

    # Extract best solution from MIP
    if model.SolCount > 0:
        try:
            cur_obj = model.ObjVal
            if cur_obj < best_obj - 1e-8:
                best_obj = cur_obj
                best_x = np.array([x[i].X for i in range(n)])
                if logger:
                    logger.log(float(best_obj))
        except:
            pass

    # Local polishing on support
    support = [i for i in range(n) if abs(best_x[i]) > 1e-7]
    if len(support) > 0 and (time.time() - start_time) < args.time_limit - 0.5:
        try:
            m2 = gp.Model("Polish")
            m2.setParam(GRB.Param.Threads, 1)
            m2.setParam(GRB.Param.OutputFlag, 0)
            m2.setParam(GRB.Param.TimeLimit, max(0.5, args.time_limit - (time.time() - start_time) - 0.1))
            x2 = m2.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            m2.addConstr(x2.sum() == 1.0, "budget2")
            m2.addConstr(r @ x2 >= b, "return2")
            for i in range(n):
                if i not in support:
                    m2.addConstr(x2[i] == 0.0, f"fix_{i}")
            y2 = m2.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            for i in range(n):
                m2.addConstr(y2[i] == gp.quicksum(Q[i, j] * x2[j] for j in support), f"quad2_{i}")
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