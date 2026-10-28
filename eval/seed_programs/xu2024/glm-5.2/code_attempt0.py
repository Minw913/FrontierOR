import argparse
import json
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve_qp(Q_sub, r_sub, b, time_limit=5):
    """Solve a continuous QP on a subset of assets."""
    n_sub = len(r_sub)
    Q_sub_np = np.array(Q_sub, dtype=float)
    r_sub_np = np.array(r_sub, dtype=float)
    model = gp.Model("qp_sub")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Seed', 0)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    x = model.addMVar(n_sub, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    model.addConstr(x.sum() == 1)
    model.addConstr(r_sub_np @ x >= b)
    obj = 0.5 * (x @ Q_sub_np @ x)
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    if model.status == GRB.OPTIMAL or model.SolCount > 0:
        return model.ObjVal, x.X.tolist()
    return None, None

def heuristic_solve(n, k, b, r, Q, logger=None, time_limit=5):
    """Generate a good feasible solution using simple heuristics."""
    best_obj = float('inf')
    best_x = None

    # Full unconstrained QP
    full_obj, full_x = solve_qp(Q, r, b, time_limit)
    if full_x is not None:
        nonzeros = [i for i in range(n) if abs(full_x[i]) > 1e-6]
        if len(nonzeros) <= k:
            x = [0.0] * n
            for i in nonzeros:
                x[i] = full_x[i]
            obj = 0.5 * sum(Q[i][j] * x[i] * x[j] for i in range(n) for j in range(n))
            return obj, x

        # Candidate 1: top k by absolute weight
        order = sorted(range(n), key=lambda i: abs(full_x[i]), reverse=True)
        S = order[:k]
        Q_sub = [[Q[i][j] for j in S] for i in S]
        r_sub = [r[i] for i in S]
        obj, x_sub = solve_qp(Q_sub, r_sub, b, time_limit)
        if obj is not None:
            x = [0.0] * n
            for idx, i in enumerate(S):
                x[i] = x_sub[idx]
            if obj < best_obj:
                best_obj, best_x = obj, x
                if logger:
                    logger.log_solution(obj, {"objective_value": obj, "x": x})

    # Candidate 2: top k by expected return
    order_r = sorted(range(n), key=lambda i: r[i], reverse=True)
    S = order_r[:k]
    Q_sub = [[Q[i][j] for j in S] for i in S]
    r_sub = [r[i] for i in S]
    obj, x_sub = solve_qp(Q_sub, r_sub, b, time_limit)
    if obj is not None:
        x = [0.0] * n
        for idx, i in enumerate(S):
            x[i] = x_sub[idx]
        if obj < best_obj:
            best_obj, best_x = obj, x
            if logger:
                logger.log_solution(obj, {"objective_value": obj, "x": x})

    # Candidate 3: top k-1 by abs weight + highest return
    if full_x is not None and k >= 2:
        order = sorted(range(n), key=lambda i: abs(full_x[i]), reverse=True)
        S = order[:k-1]
        remaining = [i for i in range(n) if i not in S]
        remaining.sort(key=lambda i: r[i], reverse=True)
        if remaining:
            S.append(remaining[0])
        Q_sub = [[Q[i][j] for j in S] for i in S]
        r_sub = [r[i] for i in S]
        obj, x_sub = solve_qp(Q_sub, r_sub, b, time_limit)
        if obj is not None:
            x = [0.0] * n
            for idx, i in enumerate(S):
                x[i] = x_sub[idx]
            if obj < best_obj:
                best_obj, best_x = obj, x
                if logger:
                    logger.log_solution(obj, {"objective_value": obj, "x": x})

    if best_x is not None:
        best_obj = 0.5 * sum(Q[i][j] * best_x[i] * best_x[j] for i in range(n) for j in range(n))
        return best_obj, best_x
    return None, None

def find_feasible_fallback(n, k, b, r, Q):
    """Return a simple feasible solution if possible."""
    if k >= 1:
        for i in range(n):
            if r[i] >= b - 1e-9:
                x = [0.0] * n
                x[i] = 1.0
                obj = 0.5 * Q[i][i]
                return obj, x
    if k >= 2:
        for i in range(n):
            for j in range(i+1, n):
                if abs(r[i] - r[j]) > 1e-9:
                    S = [i, j]
                    Q_sub = [[Q[i][j] for j in S] for i in S]
                    r_sub = [r[i] for i in S]
                    obj, x_sub = solve_qp(Q_sub, r_sub, b, 2)
                    if obj is not None:
                        x = [0.0] * n
                        for idx, idx2 in enumerate(S):
                            x[idx2] = x_sub[idx]
                        return obj, x
    # Last resort (may violate constraints if problem is infeasible)
    x = [0.0] * n
    x[0] = 1.0
    obj = 0.5 * Q[0][0]
    return obj, x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)
    n = data['n']
    k = data['k']
    b = float(data['b'])
    r = [float(x) for x in data['r']]
    Q = [[float(x) for x in row] for row in data['Q']]

    start_time = time.time()

    # Heuristic search for a good starting point
    heur_obj, heur_x = None, None
    if args.time_limit > 5:
        heur_time_limit = min(5, args.time_limit * 0.2)
        try:
            heur_obj, heur_x = heuristic_solve(n, k, b, r, Q, logger, heur_time_limit)
        except Exception:
            heur_obj, heur_x = None, None

    # Fallback if heuristic failed
    if heur_x is None:
        heur_obj, heur_x = find_feasible_fallback(n, k, b, r, Q)
        if heur_x is not None and logger:
            logger.log_solution(heur_obj, {"objective_value": heur_obj, "x": heur_x})

    # Build MIQP model
    model = gp.Model("portfolio")
    model.setParam('OutputFlag', 0)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)

    Q_np = np.array(Q, dtype=float)
    r_np = np.array(r, dtype=float)

    x = model.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    y = model.addMVars(n, vtype=GRB.BINARY, name="y")

    model.addConstr(x.sum() == 1, name="budget")
    model.addConstr(r_np @ x >= b, name="return")
    model.addConstr(y.sum() <= k, name="cardinality")

    for i in range(n):
        model.addGenConstrIndicator(y[i], False, x[i], GRB.EQUAL, 0.0)

    obj = 0.5 * (x @ Q_np @ x)
    model.setObjective(obj, GRB.MINIMIZE)

    # Provide MIP start
    if heur_x is not None:
        for i in range(n):
            x[i].Start = heur_x[i]
            y[i].Start = 1 if abs(heur_x[i]) > 1e-6 else 0

    # Callback for logging incumbents and enforcing time limit
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals = model.cbGet(GRB.Callback.MIPSOL_SOL)
            x_list = x_vals[:n]
            if logger:
                logger.log_solution(obj_val, {"objective_value": obj_val, "x": x_list})
        if where in (GRB.Callback.MIP, GRB.Callback.MIPSOL, GRB.Callback.MIPNODE):
            if time.time() - start_time > args.time_limit:
                model.terminate()

    model.setCallback(callback)

    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed - 1.0
    if remaining < 0.1:
        remaining = 0.1
    model.setParam('TimeLimit', remaining)

    model.optimize()

    # Extract best solution
    if model.SolCount > 0:
        x_val = x.X.tolist()
        obj_val = model.ObjVal
    else:
        if heur_x is not None:
            x_val = heur_x
            obj_val = heur_obj
        else:
            obj_val, x_val = find_feasible_fallback(n, k, b, r, Q)

    # Recompute objective to ensure accuracy
    obj_val = 0.5 * float(np.array(x_val) @ Q_np @ np.array(x_val))

    solution = {
        "objective_value": obj_val,
        "x": x_val
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()