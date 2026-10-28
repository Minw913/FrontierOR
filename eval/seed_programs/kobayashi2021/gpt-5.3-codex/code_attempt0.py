import argparse
import json
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def compute_cvar_components(losses, probs, beta):
    """Return (a, v, t) with t_s = max(loss_s - a, 0), v = 1/(1-beta) * sum p_s t_s."""
    # Quantile-based VaR candidate (works well for warm starts / fallback)
    try:
        a = float(np.quantile(losses, beta, method="higher"))
    except TypeError:
        a = float(np.quantile(losses, beta, interpolation="higher"))
    t = np.maximum(losses - a, 0.0)
    v = float(max(0.0, (probs @ t) / (1.0 - beta)))
    return a, v, t


def build_fallback_solution(mu, R, probs, beta, gamma, mu_bar, k):
    """Simple feasible-ish fallback (best effort)."""
    N = len(mu)
    x = np.zeros(N, dtype=float)

    if N > 0:
        j = int(np.argmax(mu))
        x[j] = 1.0

    z = np.zeros(N, dtype=int)
    if N > 0:
        z[int(np.argmax(x))] = 1

    losses = -R.dot(x)
    a, v, _ = compute_cvar_components(losses, probs, beta)
    obj = float((x @ x) / (2.0 * gamma) + a + v)

    return {
        "objective_value": obj,
        "x": x.tolist(),
        "z": z.tolist(),
        "a": float(a),
        "v": float(v),
    }


def solve_instance(data, time_limit, logger=None):
    N = int(data["N"])
    S = int(data["S"])
    k = int(data["k"])
    beta = float(data["beta"])
    gamma = float(data["gamma"])
    mu_bar = float(data["mu_bar"])

    mu = np.asarray(data["mu"], dtype=float)
    R = np.asarray(data["scenarios"], dtype=float)

    p_raw = data["p_s"]
    if isinstance(p_raw, list):
        probs = np.asarray(p_raw, dtype=float)
    else:
        probs = np.full(S, float(p_raw), dtype=float)

    # Safety normalization if needed
    psum = probs.sum()
    if psum > 0 and abs(psum - 1.0) > 1e-10:
        probs = probs / psum

    tail_coeff = 1.0 / (1.0 - beta)

    model = gp.Model("cardinality_cvar_portfolio")
    model.Params.TimeLimit = int(time_limit)
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    # Variables
    x = model.addMVar(N, lb=0.0, ub=1.0, name="x")
    use_binaries = k < N
    z = model.addMVar(N, vtype=GRB.BINARY, name="z") if use_binaries else None
    a = model.addVar(lb=-GRB.INFINITY, name="a")
    v = model.addVar(lb=0.0, name="v")
    t = model.addMVar(S, lb=0.0, name="t")

    # Constraints
    model.addConstr(x.sum() == 1.0, name="budget")
    model.addConstr(mu @ x >= mu_bar, name="required_return")

    if use_binaries:
        model.addConstr(x <= z, name="link_x_z")
        model.addConstr(z.sum() <= k, name="cardinality")

    # t_s >= loss_s - a, where loss_s = -R_s x
    model.addConstr(t >= (-R @ x) - a, name="tail_excess")
    model.addConstr(v >= tail_coeff * (probs @ t), name="cvar_def")

    # Objective: (1/(2*gamma)) * ||x||^2 + a + v
    model.setObjective((1.0 / (2.0 * gamma)) * (x @ x) + a + v, GRB.MINIMIZE)

    # Warm start: invest all in highest expected-return asset if feasible for return
    if N > 0 and k >= 1:
        x0 = np.zeros(N, dtype=float)
        j = int(np.argmax(mu))
        x0[j] = 1.0
        losses0 = -R.dot(x0)
        a0, v0, t0 = compute_cvar_components(losses0, probs, beta)

        x.Start = x0
        a.Start = a0
        v.Start = v0
        t.Start = t0
        if use_binaries:
            z0 = (x0 > 1e-9).astype(float)
            z.Start = z0

    # Incumbent logging callback for MIP
    best_logged = {"obj": math.inf}

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged["obj"]:
                best_logged["obj"] = obj
                logger.log(float(obj))

    if use_binaries:
        model.optimize(cb)
    else:
        model.optimize()

    if model.SolCount > 0:
        x_sol = np.array(x.X, dtype=float)
        if use_binaries:
            z_sol = [int(val > 0.5) for val in np.array(z.X, dtype=float)]
        else:
            z_sol = [int(val > 1e-8) for val in x_sol]

        result = {
            "objective_value": float(model.ObjVal),
            "x": x_sol.tolist(),
            "z": z_sol,
            "a": float(a.X),
            "v": float(v.X),
        }

        if logger is not None:
            obj = float(model.ObjVal)
            if obj + 1e-9 < best_logged["obj"]:
                logger.log(obj)

        return result

    # No feasible solution found by solver within time / infeasible -> fallback
    fallback = build_fallback_solution(mu, R, probs, beta, gamma, mu_bar, k)
    if logger is not None:
        logger.log(float(fallback["objective_value"]))
    return fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    solution = solve_instance(data, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()