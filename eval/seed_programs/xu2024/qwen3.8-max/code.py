import argparse
import json
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize as scipy_minimize

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def solve_qp_for_subset(Q, r, b, subset):
    """
    Solve the QP for a given subset of assets (indices).
    Minimize 0.5 * x'Qx subject to sum(x)=1, r'x >= b.
    Uses analytical solution with KKT conditions.
    """
    m = len(subset)
    if m == 0:
        return None, None

    Q_sub = Q[np.ix_(subset, subset)]
    r_sub = r[subset]

    # Try to solve: min 0.5 x'Qx s.t. 1'x = 1, r'x >= b
    # KKT: Qx = lambda * 1 + mu * r, 1'x = 1, r'x >= b, mu >= 0
    # If mu = 0: Qx = lambda * 1 => x = lambda * Q^{-1} * 1
    #   1'x = 1 => lambda = 1 / (1'Q^{-1}1)
    # Check if r'x >= b

    try:
        Q_inv = np.linalg.inv(Q_sub)
    except np.linalg.LinAlgError:
        try:
            Q_inv = np.linalg.pinv(Q_sub)
        except:
            return None, None

    ones = np.ones(m)

    # Case 1: mu = 0 (return constraint not binding)
    denom = ones @ Q_inv @ ones
    if abs(denom) < 1e-15:
        return None, None
    x1 = Q_inv @ ones / denom
    ret1 = r_sub @ x1

    if ret1 >= b - 1e-10:
        obj = 0.5 * x1 @ Q_sub @ x1
        return obj, x1

    # Case 2: mu > 0 (return constraint binding: r'x = b)
    # Qx = lambda*1 + mu*r
    # x = Q^{-1}(lambda*1 + mu*r)
    # Constraints: 1'x = 1, r'x = b
    # [1'Q^{-1}1  1'Q^{-1}r] [lambda]   [1]
    # [r'Q^{-1}1  r'Q^{-1}r] [mu    ] = [b]
    a = ones @ Q_inv @ ones
    c = ones @ Q_inv @ r_sub
    d = r_sub @ Q_inv @ r_sub

    det = a * d - c * c
    if abs(det) < 1e-15:
        return None, None

    lam = (d - c * b) / det
    mu = (a * b - c) / det

    if mu < -1e-10:
        return None, None

    x2 = Q_inv @ (lam * ones + mu * r_sub)
    obj = 0.5 * x2 @ Q_sub @ x2
    return obj, x2


def solve_subset_scipy(Q, r, b, subset):
    """Fallback: solve QP for subset using scipy."""
    m = len(subset)
    if m == 0:
        return None, None

    Q_sub = Q[np.ix_(subset, subset)]
    r_sub = r[subset]

    def objective(x):
        return 0.5 * x @ Q_sub @ x

    def grad(x):
        return Q_sub @ x

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(m)},
        {'type': 'ineq', 'fun': lambda x: r_sub @ x - b, 'jac': lambda x: r_sub}
    ]

    # Try multiple starting points
    best_obj = None
    best_x = None

    starts = [np.ones(m) / m]
    if m > 1:
        starts.append(np.array([1.0] + [0.0] * (m - 1)))
        starts.append(np.array([0.0] * (m - 1) + [1.0]))

    for x0 in starts:
        try:
            res = scipy_minimize(objective, x0, jac=grad, method='SLSQP',
                                 constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                x_sol = res.x
                ret = r_sub @ x_sol
                if ret >= b - 1e-8:
                    obj = 0.5 * x_sol @ Q_sub @ x_sol
                    if best_obj is None or obj < best_obj:
                        best_obj = obj
                        best_x = x_sol.copy()
        except:
            continue

    return best_obj, best_x


def solve_gurobi_mip(Q, r, b, n, k, time_limit, logger, start_time):
    """Use Gurobi to solve the cardinality-constrained QP via MIQP."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return None, None

    try:
        model = gp.Model("portfolio")
        model.setParam('Seed', 0)
        model.setParam('MIPGap', 1e-4)
        model.setParam('NumericFocus', 0)
        model.setParam('Threads', 1)
        model.setParam('TimeLimit', max(1, time_limit - (time.time() - start_time)))
        model.setParam('OutputFlag', 0)

        # Variables
        x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
        z = model.addVars(n, vtype=GRB.BINARY, name="z")

        # Objective: minimize 0.5 * x'Qx
        obj = gp.QuadExpr()
        for i in range(n):
            for j in range(n):
                if abs(Q[i][j]) > 1e-15:
                    if i == j:
                        obj.add(0.5 * Q[i][j] * x[i] * x[j])
                    elif i < j:
                        obj.add(Q[i][j] * x[i] * x[j])
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints
        model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1.0, "budget")
        model.addConstr(gp.quicksum(r[i] * x[i] for i in range(n)) >= b, "return")
        model.addConstr(gp.quicksum(z[i] for i in range(n)) <= k, "cardinality")

        # Linking constraints: x[i] != 0 => z[i] = 1
        # We need big-M constraints: -M*z[i] <= x[i] <= M*z[i]
        # Choose M based on problem structure
        M = 10.0
        for i in range(n):
            model.addConstr(x[i] <= M * z[i], f"link_upper_{i}")
            model.addConstr(x[i] >= -M * z[i], f"link_lower_{i}")

        model.optimize()

        if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
            if model.SolCount > 0:
                x_sol = np.array([x[i].X for i in range(n)])
                # Clean up tiny weights
                for i in range(n):
                    if abs(x_sol[i]) < 1e-8:
                        x_sol[i] = 0.0
                obj_val = 0.5 * x_sol @ np.array(Q) @ x_sol
                return obj_val, x_sol
    except Exception as e:
        pass

    return None, None


def main():
    args = parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    k = instance['k']
    b = instance['b']
    r = np.array(instance['r'], dtype=float)
    Q = np.array(instance['Q'], dtype=float)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    best_obj = None
    best_x = None

    def update_best(obj_val, x_sol):
        nonlocal best_obj, best_x
        if obj_val is not None and (best_obj is None or obj_val < best_obj - 1e-12):
            best_obj = obj_val
            best_x = x_sol.copy()
            if logger:
                solution_dict = {
                    "objective_value": float(best_obj),
                    "x": [float(v) for v in best_x]
                }
                logger.log_solution(float(best_obj), solution_dict)

    # Strategy 1: If n is small enough, enumerate all subsets of size <= k
    # For each subset, solve the QP analytically
    if n <= 20 and k <= n:
        # Enumerate subsets of size 1 to k
        # But if k is large relative to n, this can be expensive
        # Number of subsets: sum_{i=1}^{k} C(n, i)
        total_subsets = sum(int(np.math.comb(n, i)) for i in range(1, k + 1))

        if total_subsets <= 500000:
            for size in range(1, k + 1):
                if time.time() - start_time > args.time_limit * 0.8:
                    break
                for subset in combinations(range(n), size):
                    if time.time() - start_time > args.time_limit * 0.8:
                        break
                    obj_val, x_sub = solve_qp_for_subset(Q, r, b, list(subset))
                    if obj_val is not None:
                        x_full = np.zeros(n)
                        for idx, s in enumerate(subset):
                            x_full[s] = x_sub[idx]
                        update_best(obj_val, x_full)

            if best_obj is not None:
                solution = {
                    "objective_value": float(best_obj),
                    "x": [float(v) for v in best_x]
                }
                with open(args.solution_path, 'w') as f:
                    json.dump(solution, f)
                return

    # Strategy 2: Use Gurobi MIQP
    remaining_time = args.time_limit - (time.time() - start_time)
    if remaining_time > 1:
        obj_val, x_sol = solve_gurobi_mip(Q, r, b, n, k, remaining_time, logger, start_time)
        if obj_val is not None:
            update_best(obj_val, x_sol)

    # Strategy 3: Greedy/heuristic approach if Gurobi didn't find solution
    if best_obj is None or (time.time() - start_time < args.time_limit * 0.5):
        # Try greedy forward selection
        selected = []
        remaining = list(range(n))

        # Sort by return/risk ratio as initial heuristic
        diag = np.diag(Q)
        scores = np.array([r[i] / max(diag[i], 1e-10) for i in range(n)])
        sorted_assets = np.argsort(-scores)

        for asset in sorted_assets:
            if len(selected) >= k:
                break
            if time.time() - start_time > args.time_limit * 0.9:
                break

            candidate = selected + [int(asset)]
            obj_val, x_sub = solve_qp_for_subset(Q, r, b, candidate)
            if obj_val is not None:
                x_full = np.zeros(n)
                for idx, s in enumerate(candidate):
                    x_full[s] = x_sub[idx]
                update_best(obj_val, x_full)
                selected = candidate
                remaining.remove(int(asset))

        # Try swaps to improve
        if best_obj is not None and time.time() - start_time < args.time_limit * 0.7:
            improved = True
            while improved and time.time() - start_time < args.time_limit * 0.7:
                improved = False
                for i, s in enumerate(selected):
                    for a in remaining:
                        if time.time() - start_time > args.time_limit * 0.7:
                            break
                        candidate = [x for x in selected if x != s] + [a]
                        obj_val, x_sub = solve_qp_for_subset(Q, r, b, candidate)
                        if obj_val is not None and (best_obj is None or obj_val < best_obj - 1e-10):
                            x_full = np.zeros(n)
                            for idx, s2 in enumerate(candidate):
                                x_full[s2] = x_sub[idx]
                            update_best(obj_val, x_full)
                            remaining.append(s)
                            remaining.remove(a)
                            selected = candidate
                            improved = True
                            break
                    if improved:
                        break

        # Also try subsets of size < k if we haven't tried
        if time.time() - start_time < args.time_limit * 0.5:
            for size in range(1, min(k + 1, len(selected) + 1)):
                if time.time() - start_time > args.time_limit * 0.5:
                    break
                # Try all combinations of current best selected set
                if len(selected) > size:
                    for subset in combinations(selected, size):
                        obj_val, x_sub = solve_qp_for_subset(Q, r, b, list(subset))
                        if obj_val is not None:
                            x_full = np.zeros(n)
                            for idx, s in enumerate(subset):
                                x_full[s] = x_sub[idx]
                            update_best(obj_val, x_full)

    # If still no solution, try scipy on full problem with cardinality handled by rounding
    if best_obj is None:
        # Solve unconstrained (no cardinality) QP
        obj_val, x_sub = solve_qp_for_subset(Q, r, b, list(range(n)))
        if obj_val is not None:
            # Keep top-k by absolute weight
            indices = np.argsort(-np.abs(x_sub))[:k]
            subset = sorted(indices.tolist())
            obj_val2, x_sub2 = solve_qp_for_subset(Q, r, b, subset)
            if obj_val2 is not None:
                x_full = np.zeros(n)
                for idx, s in enumerate(subset):
                    x_full[s] = x_sub2[idx]
                update_best(obj_val2, x_full)
            else:
                # Just use the rounded solution
                x_full = np.zeros(n)
                for idx in indices:
                    x_full[idx] = x_sub[idx]
                # Renormalize
                s = np.sum(x_full)
                if abs(s) > 1e-10:
                    x_full = x_full / s
                obj_val3 = 0.5 * x_full @ Q @ x_full
                if r @ x_full >= b - 1e-8:
                    update_best(obj_val3, x_full)

    # Final fallback: equal weight on top-k return assets
    if best_obj is None:
        top_k = np.argsort(-r)[:k]
        x_full = np.zeros(n)
        x_full[top_k] = 1.0 / k
        obj_val = 0.5 * x_full @ Q @ x_full
        update_best(obj_val, x_full)

    # Write solution
    if best_obj is not None:
        solution = {
            "objective_value": float(best_obj),
            "x": [float(v) for v in best_x]
        }
    else:
        # Absolute fallback
        x_full = np.zeros(n)
        x_full[0] = 1.0
        solution = {
            "objective_value": float(0.5 * Q[0, 0]),
            "x": [float(v) for v in x_full]
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()