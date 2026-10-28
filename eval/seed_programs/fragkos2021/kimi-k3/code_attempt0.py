#!/usr/bin/env python
"""
Reliable facility location under random facility disruptions.

For a fixed set S of opened facilities, the expected transportation / penalty
cost is computed exactly as

    E(S) = sum_i d_i * [ sum_r c_i[r] * p_i[r] * x_{pi(r)} * prod_{s<r} u_i[s]
                         + P_i * prod_r u_i[r] ]

where, for customer i, facilities are sorted by increasing shipment cost,
p_i[r] = survival probability of the facility of rank r, u_i[s] = q_i[s]
(disruption probability) if that facility is open and 1 otherwise.

Two complementary solvers are used:
  * an exact linear MIP obtained by linearizing the probability chain with
    variables z (chain probabilities) and w = z * x (McCormick, exact since x
    is binary), solved with Gurobi;
  * a greedy / local-search heuristic with fast vectorized evaluation, used
    as warm start and as fallback for very large instances.
"""

import argparse
import json
import time

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB

    HAVE_GUROBI = True
except Exception:  # pragma: no cover
    HAVE_GUROBI = False

from solution_logger import SolutionLogger


# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------
class Problem:
    def __init__(self, inst):
        nodes = {nd["id"]: nd for nd in inst["nodes"]}
        self.C = int(inst["num_customers"])
        self.F = int(inst["num_facilities"])
        C, F = self.C, self.F

        self.demand = np.array([float(nodes[i]["demand"]) for i in range(C)])
        self.fixed = np.array(
            [float(nodes[j]["fixed_setup_cost"]) for j in range(F)]
        )
        if "survival_probability" in next(iter(nodes.values()), {}):
            psurv = np.array(
                [float(nodes[j]["survival_probability"]) for j in range(F)]
            )
        else:
            psurv = np.array(
                [1.0 - float(nodes[j]["disruption_probability"]) for j in range(F)]
            )
        self.psurv = np.clip(psurv, 0.0, 1.0)
        self.qfail = 1.0 - self.psurv  # mutually consistent pair

        cost = np.asarray(inst["transportation_cost_matrix"], dtype=float)  # C x F
        self.penalty = np.asarray(inst["penalty_cost"], dtype=float)        # C

        if F > 0 and C > 0:
            # per-customer facility order by ascending shipment cost
            order = np.argsort(cost, axis=1, kind="stable")          # C x F
            pos = np.argsort(order, axis=1, kind="stable")           # C x F
            rows = np.arange(C)[:, None]
            self.order = order
            self.pos = pos
            self.csort = cost[rows, order]                           # C x F
            self.psort = self.psurv[order]                           # C x F
        else:
            self.order = np.zeros((C, F), dtype=int)
            self.pos = np.zeros((C, F), dtype=int)
            self.csort = np.zeros((C, F))
            self.psort = np.zeros((C, F))

    # exact expected unit cost per customer for a given binary vector x
    def expected_unit_cost(self, x):
        if self.F == 0:
            return self.penalty.copy()
        xs = x[self.order]                                  # C x F (0/1)
        u = 1.0 - self.psort * xs                           # u = q if open else 1
        cum = np.cumprod(u, axis=1)
        prefix = np.ones_like(cum)
        if self.F > 1:
            prefix[:, 1:] = cum[:, :-1]
        served = self.csort * self.psort * xs * prefix      # C x F
        unit = served.sum(axis=1) + self.penalty * cum[:, -1]
        return unit

    def objective(self, x):
        x = np.asarray(x, dtype=float)
        if self.C == 0:
            return float(self.fixed @ x)
        return float(self.fixed @ x + self.demand @ self.expected_unit_cost(x))

    def solution_dict(self, x, obj):
        return {
            "objective_value": float(obj),
            "facility_locations": [int(v) for v in np.asarray(x).tolist()],
        }

    # ------------------------------------------------------------------
    # Greedy add: efficiently evaluates the marginal value of opening any
    # currently closed facility in O(C*F) per iteration (fully vectorized).
    # ------------------------------------------------------------------
    def greedy_add(self, deadline):
        F = self.F
        x = np.zeros(F, dtype=float)
        obj = self.objective(x)
        if F == 0:
            return x, obj
        tol = 1e-9 * max(1.0, abs(obj))
        while time.time() < deadline:
            xs = x[self.order]
            u = 1.0 - self.psort * xs
            cum = np.cumprod(u, axis=1)
            prefix = np.empty_like(cum)
            prefix[:, 0] = 1.0
            if F > 1:
                prefix[:, 1:] = cum[:, :-1]
            term = self.csort * self.psort * xs * prefix
            csum = np.cumsum(term, axis=1)
            a_excl = np.zeros_like(csum)                    # sum of terms before rank r
            if F > 1:
                a_excl[:, 1:] = csum[:, :-1]
            rsum = np.cumsum(term[:, ::-1], axis=1)[:, ::-1]  # suffix incl. rank r
            suf_after = np.zeros_like(rsum)                 # sum of terms after rank r
            if F > 1:
                suf_after[:, :-1] = rsum[:, 1:]
            full = cum[:, F - 1]
            e_cur = term.sum(axis=1) + self.penalty * full

            pos = self.pos                                  # rank of facility j per customer
            a_pos = np.take_along_axis(a_excl, pos, axis=1)
            c_pos = np.take_along_axis(self.csort, pos, axis=1)
            p_pos = np.take_along_axis(self.psort, pos, axis=1)
            prefix_pos = np.take_along_axis(prefix, pos, axis=1)
            suf_pos = np.take_along_axis(suf_after, pos, axis=1)
            q_j = self.qfail[None, :]
            pen_i = self.penalty[:, None]

            new_e = (
                a_pos
                + c_pos * p_pos * prefix_pos
                + q_j * suf_pos
                + pen_i * q_j * full[:, None]
            )
            delta = self.fixed[None, :] + self.demand[:, None] * (
                new_e - e_cur[:, None]
            )
            gain = delta.sum(axis=0)   # total cost change of opening facility j
            gain[x > 0.5] = np.inf
            j = int(np.argmin(gain))
            if gain[j] < -tol:
                x[j] = 1.0
                obj = self.objective(x)   # recompute to avoid FP drift
                tol = 1e-9 * max(1.0, abs(obj))
            else:
                break
        return x, float(obj)

    # ------------------------------------------------------------------
    # Local search: single add/drop flips first; when no flip helps, try a
    # swap (close one open facility, open one closed facility).
    # ------------------------------------------------------------------
    def local_search(self, x, obj, deadline, logger=None):
        F = self.F
        x = np.array(x, dtype=float)
        obj = float(obj)
        if F == 0:
            return x, obj
        tol = 1e-9 * max(1.0, abs(obj))
        improved = True
        while improved and time.time() < deadline:
            improved = False
            tol = 1e-9 * max(1.0, abs(obj))
            # ---- best single flip ----
            best_j, best_val = -1, obj
            for j in range(F):
                x[j] = 1.0 - x[j]
                val = self.objective(x)
                x[j] = 1.0 - x[j]
                if val < best_val - tol:
                    best_val, best_j = val, j
                if (j & 15) == 15 and time.time() > deadline:
                    break
            if best_j >= 0:
                x[best_j] = 1.0 - x[best_j]
                obj = self.objective(x)
                improved = True
                if logger:
                    logger.log_solution(obj, self.solution_dict(x, obj))
                continue
            # ---- otherwise try swaps (bounded work) ----
            open_idx = np.where(x > 0.5)[0]
            closed_idx = np.where(x <= 0.5)[0]
            if len(open_idx) == 0 or len(closed_idx) == 0:
                break
            if len(open_idx) * len(closed_idx) > 40000:
                break  # too expensive; keep time for the MIP
            best_pair, best_val = None, obj
            stop = False
            for a in open_idx:
                for b in closed_idx:
                    x[a] = 0.0
                    x[b] = 1.0
                    val = self.objective(x)
                    x[a] = 1.0
                    x[b] = 0.0
                    if val < best_val - tol:
                        best_val, best_pair = val, (a, b)
                    if time.time() > deadline:
                        stop = True
                        break
                if stop:
                    break
            if best_pair is not None:
                a, b = best_pair
                x[a] = 0.0
                x[b] = 1.0
                obj = self.objective(x)
                improved = True
                if logger:
                    logger.log_solution(obj, self.solution_dict(x, obj))
        return x, obj


# ---------------------------------------------------------------------------
# Exact MIP via probability-chain linearization
# ---------------------------------------------------------------------------
def mip_solve(prob, warm_x, warm_obj, tlimit, logger):
    if not HAVE_GUROBI:
        return warm_x, warm_obj
    C, F = prob.C, prob.F
    order = prob.order
    psort = prob.psort
    csort = prob.csort
    dem = prob.demand
    pen = prob.penalty
    fix = prob.fixed

    m = gp.Model("ReliableFacilityLocation")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1.0, tlimit)
    m.Params.OutputFlag = 0

    x = m.addVars(F, vtype=GRB.BINARY, name="x")
    z = m.addVars(C, F + 1, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z")
    w = m.addVars(C, F, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w")

    # z[i,0] = 1
    m.addConstrs((z[i, 0] == 1.0 for i in range(C)), name="zinit")

    pairs = [(i, r) for i in range(C) for r in range(F)]
    # chain: z[i,r+1] = z[i,r] - p_i[r] * w[i,r]   (u = 1 - p*x when open)
    m.addConstrs(
        (z[i, r + 1] == z[i, r] - float(psort[i, r]) * w[i, r] for (i, r) in pairs),
        name="chain",
    )
    # McCormick for w[i,r] = z[i,r] * x[order[i,r]]  (exact, x binary)
    m.addConstrs((w[i, r] <= z[i, r] for (i, r) in pairs), name="mc_z")
    m.addConstrs(
        (w[i, r] <= x[int(order[i, r])] for (i, r) in pairs), name="mc_x"
    )
    m.addConstrs(
        (
            w[i, r] >= z[i, r] + x[int(order[i, r])] - 1.0
            for (i, r) in pairs
        ),
        name="mc_zx",
    )

    expr = gp.LinExpr()
    expr.addTerms(fix.tolist(), [x[j] for j in range(F)])
    for i in range(C):
        co = (dem[i] * csort[i] * psort[i]).tolist()
        expr.addTerms(co, [w[i, r] for r in range(F)])
        expr.add(float(dem[i] * pen[i]), z[i, F])
    m.setObjective(expr, GRB.MINIMIZE)

    # ---- warm start ----
    wx = np.asarray(warm_x, dtype=float)
    for j in range(F):
        x[j].Start = 1.0 if wx[j] >= 0.5 else 0.0
    if F > 0:
        xsm = wx[order]
        u = 1.0 - psort * xsm
        cum = np.cumprod(u, axis=1)
        pref = np.ones_like(cum)
        if F > 1:
            pref[:, 1:] = cum[:, :-1]
        for i in range(C):
            for r in range(F):
                z[i, r].Start = float(pref[i, r])
                w[i, r].Start = float(pref[i, r] * (1.0 if wx[int(order[i, r])] >= 0.5 else 0.0))
            z[i, F].Start = float(cum[i, F - 1])

    xvars = [x[j] for j in range(F)]
    state = {"best": float(warm_obj)}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                vals = model.cbGetSolution(xvars)
                xv = np.array(
                    [1.0 if v >= 0.5 else 0.0 for v in vals], dtype=float
                )
                objv = prob.objective(xv)
                if objv < state["best"] - 1e-9:
                    state["best"] = objv
                    logger.log_solution(objv, prob.solution_dict(xv, objv))
            except Exception:
                pass

    m.optimize(callback)

    best_x, best_obj = np.array(warm_x, dtype=float), float(warm_obj)
    if m.SolCount > 0:
        vals = m.getAttr("X", xvars)
        xv = np.array([1.0 if v >= 0.5 else 0.0 for v in vals], dtype=float)
        objv = prob.objective(xv)
        if objv < best_obj - 1e-9:
            best_x, best_obj = xv, objv
    m.dispose()
    return best_x, best_obj


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def solve(prob, time_limit, logger):
    t0 = time.time()
    deadline = t0 + max(time_limit, 0)

    # ---- heuristic warm start (bounded share of the budget) ----
    heur_deadline = t0 + 0.35 * max(time_limit, 0)
    x, obj = prob.greedy_add(heur_deadline)
    x, obj = prob.local_search(x, obj, heur_deadline, logger=logger)
    obj = prob.objective(x)
    best_x = x
    best_obj = obj
    if logger:
        logger.log_solution(best_obj, prob.solution_dict(best_x, best_obj))

    use_mip = (
        HAVE_GUROBI
        and prob.C * prob.F <= 150_000
        and (deadline - time.time() - 0.3) > 1.0
    )
    if use_mip:
        remaining = deadline - time.time() - 0.3
        gx, gobj = mip_solve(prob, best_x, best_obj, remaining, logger)
        if gobj < best_obj - 1e-9:
            best_x, best_obj = gx, gobj
    else:
        # randomized restart fallback for instances too large for the MIP
        rng = np.random.RandomState(0)
        F = prob.F
        while F > 0 and time.time() < deadline - 0.1:
            kicked = np.array(best_x, dtype=float)
            k = max(1, F // 8)
            idx = rng.choice(F, size=min(k, F), replace=False)
            for j in idx:
                kicked[j] = 1.0 - kicked[j]
            tx, tobj = prob.local_search(
                kicked, prob.objective(kicked), deadline - 0.05, logger=None
            )
            if tobj < best_obj - 1e-9:
                best_x, best_obj = tx, tobj
                if logger:
                    logger.log_solution(
                        best_obj, prob.solution_dict(best_x, best_obj)
                    )

    return best_x, prob.objective(best_x)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance_path", required=True)
    p.add_argument("--solution_path", required=True)
    p.add_argument("--time_limit", type=int, required=True)
    p.add_argument("--log_path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    logger = (
        SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    )

    prob = Problem(inst)
    best_x, best_obj = solve(prob, args.time_limit, logger)

    out = prob.solution_dict(best_x, best_obj)
    if logger:
        logger.log_solution(best_obj, out)
    with open(args.solution_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()