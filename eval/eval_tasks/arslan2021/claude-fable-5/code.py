import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=600)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    C = float(inst["problem_parameters"]["knapsack_capacity"])
    Gamma = float(inst["problem_parameters"]["uncertainty_budget"])
    items = inst["items"]
    n = len(items)
    w = [float(it["weight"]) for it in items]
    p = [float(it["nominal_profit"]) for it in items]
    d = [float(it["max_degradation"]) for it in items]
    c = [float(it["outsource_penalty"]) for it in items]
    rc = [float(it["repair_capacity"]) for it in items]

    def make_sol(obj, xs):
        return {"objective_value": obj, "x": {str(i): int(xs[i]) for i in range(n)}}

    # Trivial feasible solution: select nothing -> cost 0
    best_obj = 0.0
    best_x = [0] * n
    if logger:
        logger.log_solution(best_obj, make_sol(best_obj, best_x))

    # ------------------------------------------------------------------
    # Reformulation:
    #   min_x  sum_i x_i (c_i - p_i) + phi(x)
    # where phi(x) = max_{delta in U} min_{0<=r<=y<=x, cap}  sum_i y_i(d_i delta_i - c_i) - r_i d_i delta_i
    # Dualizing the inner LP (delta appears only in its objective), the
    # max-min becomes a single joint LP over (delta, lam, mu, nu):
    #   max -C*lam - sum_i x_i mu_i
    #   s.t. mu_i >= nu_i + c_i - d_i delta_i - lam w_i
    #        nu_i >= d_i delta_i - lam rc_i
    #        0 <= delta_i <= 1, sum delta_i <= Gamma, lam,mu,nu >= 0
    # phi(x) is convex piecewise-linear in x  ->  exact Benders cuts:
    #   theta >= -C*lam^k - sum_i mu_i^k x_i
    # ------------------------------------------------------------------

    # ----- Master problem -----
    mm = gp.Model("master")
    mm.Params.OutputFlag = 0
    mm.Params.Seed = 0
    mm.Params.MIPGap = 1e-4
    mm.Params.NumericFocus = 0
    mm.Params.Threads = 1

    xv = mm.addVars(n, vtype=GRB.BINARY, name="x")
    theta = mm.addVar(lb=-(sum(c) + 1.0), ub=GRB.INFINITY, name="theta")
    mm.setObjective(
        gp.quicksum((c[i] - p[i]) * xv[i] for i in range(n)) + theta, GRB.MINIMIZE
    )

    # ----- Subproblem (joint adversary + dual LP) -----
    sm = gp.Model("sub")
    sm.Params.OutputFlag = 0
    sm.Params.Seed = 0
    sm.Params.NumericFocus = 0
    sm.Params.Threads = 1

    lam = sm.addVar(lb=0.0, name="lam")
    delta = sm.addVars(n, lb=0.0, ub=1.0, name="delta")
    mu = sm.addVars(n, lb=0.0, ub=[c[i] + d[i] + 1.0 for i in range(n)], name="mu")
    nu = sm.addVars(n, lb=0.0, ub=[d[i] + 1.0 for i in range(n)], name="nu")
    for i in range(n):
        sm.addConstr(mu[i] - nu[i] + d[i] * delta[i] + w[i] * lam >= c[i])
        sm.addConstr(nu[i] - d[i] * delta[i] + rc[i] * lam >= 0.0)
    sm.addConstr(gp.quicksum(delta[i] for i in range(n)) <= Gamma)

    max_iters = 100000
    it = 0
    while it < max_iters:
        it += 1
        rem = args.time_limit - (time.time() - t0) - 0.5
        if rem <= 0:
            break

        mm.Params.TimeLimit = max(rem, 0.1)
        mm.optimize()
        if mm.SolCount == 0:
            break
        xs = [1 if xv[i].X > 0.5 else 0 for i in range(n)]
        th = theta.X
        try:
            LB = mm.ObjBound
        except Exception:
            LB = -float("inf")

        rem = args.time_limit - (time.time() - t0) - 0.5
        if rem <= 0:
            break

        # Solve subproblem: exact worst-case second-stage value phi(xs)
        sm.setObjective(
            -C * lam - gp.quicksum(mu[i] for i in range(n) if xs[i] == 1),
            GRB.MAXIMIZE,
        )
        sm.Params.TimeLimit = max(rem, 0.1)
        sm.optimize()
        if sm.Status != GRB.OPTIMAL:
            break

        phi = sm.ObjVal
        lv = max(0.0, lam.X)
        dv = [min(1.0, max(0.0, delta[i].X)) for i in range(n)]
        # Strengthened (minimal) feasible dual multipliers for the cut
        nuv = [max(0.0, d[i] * dv[i] - lv * rc[i]) for i in range(n)]
        muv = [
            max(0.0, nuv[i] + c[i] - d[i] * dv[i] - lv * w[i]) for i in range(n)
        ]

        fs = sum((c[i] - p[i]) for i in range(n) if xs[i] == 1)
        ub = fs + phi
        if ub < best_obj - 1e-9:
            best_obj = ub
            best_x = xs
            if logger:
                logger.log_solution(best_obj, make_sol(best_obj, best_x))

        # Convergence: theta already dominates true recourse value at xs
        if phi <= th + max(1e-7, 1e-6 * abs(phi)):
            break
        # Global gap closed
        if best_obj - LB <= max(1e-7, 1e-6 * abs(best_obj)):
            break

        # Add Benders optimality cut
        mm.addConstr(
            theta
            + gp.quicksum(muv[i] * xv[i] for i in range(n) if muv[i] > 1e-12)
            >= -C * lv
        )

    out = make_sol(best_obj, best_x)
    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()