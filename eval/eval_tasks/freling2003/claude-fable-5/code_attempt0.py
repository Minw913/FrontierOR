import argparse
import json
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def compute_min_inventory(dem, cap):
    """Given per-period demand and capacity for one facility, compute the
    pointwise-minimal feasible end-of-period inventory profile.
    Constraint: dem[t] + I[t] <= cap[t] + I[t-1], I[-1] = 0, I >= 0.
    Backward recursion: I[t-1] >= max(0, dem[t] + I[t] - cap[t]), I[T-1] >= 0.
    """
    T = len(dem)
    I = [0.0] * T
    for t in range(T - 1, 0, -1):
        I[t - 1] = max(0.0, dem[t] + I[t] - cap[t])
    return I


def build_solution(assignment, data, Ccoef):
    """Given assignment (list: customer -> facility), compute optimal inventory
    profiles and total objective. Returns (objective, solution_dict)."""
    F = data["parameters"]["num_facilities"]
    C = data["parameters"]["num_customers"]
    T = data["parameters"]["num_periods"]
    sf = data["seasonal_factors"]
    demands = data["demands"]
    caps = data["capacities"]
    hold = data["holding_costs"]

    # per-facility per-period demand
    fac_dem = [[0.0] * T for _ in range(F)]
    for c in range(C):
        f = assignment[c]
        for t in range(T):
            fac_dem[f][t] += demands[c] * sf[t]

    inventory = {}
    hold_cost = 0.0
    for f in range(F):
        I = compute_min_inventory(fac_dem[f], caps[f])
        inventory[str(f)] = [float(v) for v in I]
        for t in range(T):
            hold_cost += hold[f][t] * I[t]

    assign_cost = 0.0
    for c in range(C):
        assign_cost += Ccoef[assignment[c]][c]

    obj = float(assign_cost + hold_cost)
    sol = {
        "objective_value": obj,
        "assignment": {str(c): int(assignment[c]) for c in range(C)},
        "inventory": inventory,
    }
    return obj, sol


def greedy_assignment(data, Ccoef):
    """Fallback greedy: assign customers (largest demand first) to the cheapest
    facility whose cumulative capacity remains feasible."""
    F = data["parameters"]["num_facilities"]
    C = data["parameters"]["num_customers"]
    T = data["parameters"]["num_periods"]
    sf = np.array(data["seasonal_factors"], dtype=float)
    demands = np.array(data["demands"], dtype=float)
    caps = np.array(data["capacities"], dtype=float)

    cum_cap = np.cumsum(caps, axis=1)  # [F][T]
    cum_load = np.zeros((F, T))
    cum_sf = np.cumsum(sf)

    assignment = [-1] * C
    order = sorted(range(C), key=lambda c: -demands[c])
    for c in order:
        add = demands[c] * cum_sf  # cumulative demand of this customer
        best_f, best_cost = -1, None
        for f in np.argsort(np.array([Ccoef[f][c] for f in range(F)])):
            f = int(f)
            if np.all(cum_load[f] + add <= cum_cap[f] + 1e-9):
                best_f = f
                break
        if best_f < 0:
            # force onto cheapest facility (infeasible instance safeguard)
            best_f = int(np.argmin([Ccoef[f][c] for f in range(F)]))
        assignment[c] = best_f
        cum_load[best_f] += add
    return assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as fh:
        data = json.load(fh)

    F = data["parameters"]["num_facilities"]
    C = data["parameters"]["num_customers"]
    T = data["parameters"]["num_periods"]
    sf = data["seasonal_factors"]
    demands = data["demands"]
    caps = data["capacities"]
    hold = data["holding_costs"]
    tc = np.array(data["transportation_costs"], dtype=float)  # [F][C][T]

    # Assignment cost coefficient: total supply cost over horizon for pair (f,c)
    Ccoef = tc.sum(axis=2)  # [F][C]

    best = {"obj": float("inf"), "sol": None}

    # ---------------- Build MIP ----------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("prod_dist", env=env)
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x = model.addVars(F, C, vtype=GRB.BINARY, name="x")
    I = model.addVars(F, T, lb=0.0, name="I")

    # objective
    obj_expr = gp.LinExpr()
    for f in range(F):
        for c in range(C):
            obj_expr.addTerms(float(Ccoef[f][c]), x[f, c])
    for f in range(F):
        for t in range(T):
            obj_expr.addTerms(float(hold[f][t]), I[f, t])
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # each customer assigned to exactly one facility
    for c in range(C):
        model.addConstr(gp.quicksum(x[f, c] for f in range(F)) == 1)

    # capacity/inventory balance per facility & period
    for f in range(F):
        for t in range(T):
            expr = gp.LinExpr()
            coef_t = sf[t]
            for c in range(C):
                d = demands[c] * coef_t
                if d != 0.0:
                    expr.addTerms(d, x[f, c])
            expr.addTerms(1.0, I[f, t])
            if t > 0:
                expr.addTerms(-1.0, I[f, t - 1])
            model.addConstr(expr <= float(caps[f][t]))

    elapsed = time.time() - start
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    model.Params.TimeLimit = remaining

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xvals = m.cbGetSolution(x)
                assignment = [0] * C
                for c in range(C):
                    bf, bv = 0, -1.0
                    for f in range(F):
                        v = xvals[f, c]
                        if v > bv:
                            bv = v
                            bf = f
                    assignment[c] = bf
                obj, sol = build_solution(assignment, data, Ccoef)
                if obj < best["obj"] - 1e-9:
                    best["obj"] = obj
                    best["sol"] = sol
                    if logger:
                        logger.log_solution(obj, sol)
            except Exception:
                pass

    try:
        model.optimize(callback)
    except gp.GurobiError:
        pass

    # Extract final MIP solution if available
    if model.SolCount > 0:
        assignment = [0] * C
        for c in range(C):
            bf, bv = 0, -1.0
            for f in range(F):
                v = x[f, c].X
                if v > bv:
                    bv = v
                    bf = f
            assignment[c] = bf
        obj, sol = build_solution(assignment, data, Ccoef)
        if obj < best["obj"] - 1e-9:
            best["obj"] = obj
            best["sol"] = sol
            if logger:
                logger.log_solution(obj, sol)

    # Fallback if no solution was found at all
    if best["sol"] is None:
        assignment = greedy_assignment(data, Ccoef)
        obj, sol = build_solution(assignment, data, Ccoef)
        best["obj"] = obj
        best["sol"] = sol
        if logger:
            logger.log_solution(obj, sol)

    with open(args.solution_path, "w") as fh:
        json.dump(best["sol"], fh, indent=2)


if __name__ == "__main__":
    main()