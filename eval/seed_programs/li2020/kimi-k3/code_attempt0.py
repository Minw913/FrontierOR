import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", type=str, required=True)
    ap.add_argument("--solution_path", type=str, required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", type=str, default=None)
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Exact evaluation of the defender's minimum cost for a given attack pattern
# --------------------------------------------------------------------------- #
def evaluate_defender(I, J1, J2, theta, sigma, alpha, beta, gamma,
                      alpha_p, beta_p, gamma_p, D, C1, Q1, Q2,
                      dc1, dc2, d12, d1, d2, pat1, pat2):
    m = gp.Model("defender_lp")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    x1 = m.addVars(I, J1, lb=0.0, name="x1")   # type I: customer -> level1
    x2 = m.addVars(I, J2, lb=0.0, name="x2")   # type I: customer -> level2
    u1 = m.addVars(I, lb=0.0, name="u1")       # type I outsourced
    y = m.addVars(I, J2, lb=0.0, name="y")     # type II: customer -> level2
    u2 = m.addVars(I, lb=0.0, name="u2")       # type II outsourced
    r = m.addVars(J1, J2, lb=0.0, name="r")    # referral: level1 -> level2
    o1 = m.addVars(J1, lb=0.0, name="o1")      # referral outsourced at level1
    s = m.addVars(J2, lb=0.0, name="s")        # referral self-served at level2
    o2 = m.addVars(J2, lb=0.0, name="o2")      # referral outsourced at level2

    obj = gp.LinExpr()
    for i in range(I):
        for j in range(J1):
            obj += alpha * dc1[i][j] * x1[i, j]
        for l in range(J2):
            obj += beta * dc2[i][l] * (x2[i, l] + y[i, l])
        obj += (alpha_p + sigma * gamma_p) * u1[i]
        obj += beta_p * u2[i]
    for j in range(J1):
        for l in range(J2):
            obj += gamma * d12[j][l] * r[j, l]
        obj += gamma_p * o1[j]
    for l in range(J2):
        obj += gamma_p * o2[l]
    m.setObjective(obj, GRB.MINIMIZE)

    for i in range(I):
        m.addConstr(gp.quicksum(x1[i, j] for j in range(J1))
                    + gp.quicksum(x2[i, l] for l in range(J2))
                    + u1[i] == theta * D[i])
        m.addConstr(gp.quicksum(y[i, l] for l in range(J2)) + u2[i]
                    == (1.0 - theta) * D[i])
    for j in range(J1):
        m.addConstr(gp.quicksum(r[j, l] for l in range(J2)) + o1[j]
                    == sigma * gp.quicksum(x1[i, j] for i in range(I)))
        m.addConstr(gp.quicksum(x1[i, j] for i in range(I))
                    <= C1[j] * (1.0 - d1[pat1[j]]))
    for l in range(J2):
        m.addConstr(s[l] + o2[l]
                    == sigma * gp.quicksum(x2[i, l] for i in range(I)))
        m.addConstr(gp.quicksum(x2[i, l] for i in range(I))
                    <= Q1[l] * (1.0 - d2[pat2[l]]))
        m.addConstr(gp.quicksum(y[i, l] for i in range(I))
                    + gp.quicksum(r[j, l] for j in range(J1)) + s[l]
                    <= Q2[l] * (1.0 - d2[pat2[l]]))

    m.optimize()
    return float(m.ObjVal)


# --------------------------------------------------------------------------- #
# Main solution procedure (single-level reformulation via LP duality)
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        raw = json.load(f)

    gparm = raw["global_parameters"]
    iparm = raw["instance_parameters"]
    ipar = raw["interdiction_parameters"]

    theta = float(gparm["theta"])
    sigma = float(gparm["sigma"])
    alpha = float(gparm["alpha"])
    beta = float(gparm["beta"])
    gamma = float(gparm["gamma"])
    alpha_p = float(gparm["alpha_prime"])
    beta_p = float(gparm["beta_prime"])
    gamma_p = float(gparm["gamma_prime"])

    I = int(iparm["num_customers"])
    J1 = int(iparm["num_level1_facilities"])
    J2 = int(iparm["num_level2_facilities"])
    K = int(iparm["num_interdiction_levels_K"])
    B = float(iparm["total_interdiction_budget"])
    levels = list(range(K))
    pos_levels = levels[1:]

    h1 = [0.0] * K
    d1 = [0.0] * K
    h2 = [0.0] * K
    d2 = [0.0] * K
    for k, v in ipar["interdiction_cost_level1_h1"].items():
        h1[int(k)] = float(v)
    for k, v in ipar["capacity_reduction_level1_d1"].items():
        d1[int(k)] = float(v)
    for k, v in ipar["interdiction_cost_level2_h2"].items():
        h2[int(k)] = float(v)
    for k, v in ipar["capacity_reduction_level2_d2"].items():
        d2[int(k)] = float(v)

    D = [float(v) for v in raw["customers"]["demands"]]
    C1 = [float(v) for v in raw["level1_facilities"]["capacity_type1"]]
    Q1 = [float(v) for v in raw["level2_facilities"]["capacity_type1"]]
    Q2 = [float(v) for v in raw["level2_facilities"]["capacity_type2"]]
    dc1 = raw["distances"]["customer_to_level1_facility"]
    dc2 = raw["distances"]["customer_to_level2_facility"]
    d12 = raw["distances"]["level1_to_level2_facility"]

    def make_solution(obj_val, pat1, pat2):
        pattern = {}
        for j in range(J1):
            pattern["S1_{}".format(j)] = int(pat1[j])
        for l in range(J2):
            pattern["S2_{}".format(l)] = int(pat2[l])
        return {"objective_value": float(obj_val), "interdiction_pattern": pattern}

    # ---- Baseline: no attack (always feasible) --------------------------- #
    pat1 = [0] * J1
    pat2 = [0] * J2
    best_val = evaluate_defender(I, J1, J2, theta, sigma, alpha, beta, gamma,
                                 alpha_p, beta_p, gamma_p, D, C1, Q1, Q2,
                                 dc1, dc2, d12, d1, d2, pat1, pat2)
    if logger:
        logger.log_solution(best_val, make_solution(best_val, pat1, pat2))

    remaining = args.time_limit - (time.time() - t0)

    # ---- Bilevel master: attack decisions + dual of defender LP ---------- #
    if remaining > 3.0 and (J1 + J2) > 0:
        # Big-M bounds on the capacity-related dual multipliers.
        Mlam = alpha_p + sigma * gamma_p
        Mlam = Mlam + 1e-6 * (1.0 + abs(Mlam))
        Mnu = max(beta_p, gamma_p)
        Mnu = Mnu + 1e-6 * (1.0 + abs(Mnu))

        m = gp.Model("interdiction_master")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = max(1.0, remaining - 2.0)

        x1b = m.addVars(J1, K, vtype=GRB.BINARY, name="x1b")
        x2b = m.addVars(J2, K, vtype=GRB.BINARY, name="x2b")

        p = m.addVars(I, lb=0.0, name="p")
        q = m.addVars(I, lb=0.0, name="q")
        a = m.addVars(J1, lb=0.0, name="a")
        b = m.addVars(J2, lb=0.0, name="b")
        lam = m.addVars(J1, lb=0.0, name="lam")
        mu = m.addVars(J2, lb=0.0, name="mu")
        nu = m.addVars(J2, lb=0.0, name="nu")

        w1 = m.addVars(J1, pos_levels, lb=0.0, name="w1")
        w2 = m.addVars(J2, pos_levels, lb=0.0, name="w2")
        w3 = m.addVars(J2, pos_levels, lb=0.0, name="w3")

        # Exactly one intensity level per facility.
        for j in range(J1):
            m.addConstr(x1b.sum(j, "*") == 1)
        for l in range(J2):
            m.addConstr(x2b.sum(l, "*") == 1)

        # Interdiction budget.
        m.addConstr(gp.quicksum(h1[k] * x1b[j, k]
                                for j in range(J1) for k in levels)
                    + gp.quicksum(h2[k] * x2b[l, k]
                                  for l in range(J2) for k in levels) <= B)

        # Dual constraints of the defender LP.
        for i in range(I):
            for j in range(J1):
                m.addConstr(p[i] - sigma * a[j] - lam[j]
                            <= alpha * dc1[i][j])
            for l in range(J2):
                m.addConstr(p[i] - sigma * b[l] - mu[l]
                            <= beta * dc2[i][l])
                m.addConstr(q[i] - nu[l] <= beta * dc2[i][l])
            m.addConstr(p[i] <= alpha_p + sigma * gamma_p)
            m.addConstr(q[i] <= beta_p)
        for j in range(J1):
            m.addConstr(a[j] <= gamma_p)
            for l in range(J2):
                m.addConstr(a[j] - nu[l] <= gamma * d12[j][l])
        for l in range(J2):
            m.addConstr(b[l] <= gamma_p)
            m.addConstr(b[l] - nu[l] <= 0.0)

        # Linearization of lambda*x, mu*x, nu*x products.
        for j in range(J1):
            for k in pos_levels:
                m.addConstr(w1[j, k] <= Mlam * x1b[j, k])
                m.addConstr(w1[j, k] <= lam[j])
                m.addConstr(w1[j, k] >= lam[j] - Mlam * (1 - x1b[j, k]))
        for l in range(J2):
            for k in pos_levels:
                m.addConstr(w2[l, k] <= Mlam * x2b[l, k])
                m.addConstr(w2[l, k] <= mu[l])
                m.addConstr(w2[l, k] >= mu[l] - Mlam * (1 - x2b[l, k]))
                m.addConstr(w3[l, k] <= Mnu * x2b[l, k])
                m.addConstr(w3[l, k] <= nu[l])
                m.addConstr(w3[l, k] >= nu[l] - Mnu * (1 - x2b[l, k]))

        # Objective: dual of defender LP with attack-dependent capacities.
        obj = gp.LinExpr()
        for i in range(I):
            obj += theta * D[i] * p[i] + (1.0 - theta) * D[i] * q[i]
        for j in range(J1):
            obj += -C1[j] * lam[j]
            for k in pos_levels:
                obj += C1[j] * d1[k] * w1[j, k]
        for l in range(J2):
            obj += -Q1[l] * mu[l] - Q2[l] * nu[l]
            for k in pos_levels:
                obj += Q1[l] * d2[k] * w2[l, k]
                obj += Q2[l] * d2[k] * w3[l, k]
        m.setObjective(obj, GRB.MAXIMIZE)

        # MIP start: no attack.
        for j in range(J1):
            x1b[j, 0].Start = 1.0
        for l in range(J2):
            x2b[l, 0].Start = 1.0

        # Incumbent logging callback.
        state = {"best": best_val}
        x1b_flat = [x1b[j, k] for j in range(J1) for k in levels]
        x2b_flat = [x2b[l, k] for l in range(J2) for k in levels]

        def callback(model, where):
            if where != GRB.Callback.MIPSOL or logger is None:
                return
            try:
                obj_bst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            except gp.GurobiError:
                return
            if obj_bst > state["best"] + 1e-6:
                state["best"] = obj_bst
                v1 = model.cbGetSolution(x1b_flat)
                v2 = model.cbGetSolution(x2b_flat)
                cp1 = []
                for j in range(J1):
                    row = v1[j * K:(j + 1) * K]
                    cp1.append(max(range(K), key=lambda k: row[k]))
                cp2 = []
                for l in range(J2):
                    row = v2[l * K:(l + 1) * K]
                    cp2.append(max(range(K), key=lambda k: row[k]))
                logger.log_solution(float(obj_bst),
                                    make_solution(obj_bst, cp1, cp2))

        m.optimize(callback)

        if m.SolCount > 0:
            v1 = m.getAttr("X", x1b_flat)
            v2 = m.getAttr("X", x2b_flat)
            cand1 = []
            for j in range(J1):
                row = v1[j * K:(j + 1) * K]
                cand1.append(max(range(K), key=lambda k: row[k]))
            cand2 = []
            for l in range(J2):
                row = v2[l * K:(l + 1) * K]
                cand2.append(max(range(K), key=lambda k: row[k]))
            cost = sum(h1[k] for k in cand1) + sum(h2[k] for k in cand2)
            if cost <= B + 1e-6:
                pat1, pat2 = cand1, cand2

    # ---- Exact re-evaluation of the chosen attack pattern ---------------- #
    final_val = evaluate_defender(I, J1, J2, theta, sigma, alpha, beta, gamma,
                                  alpha_p, beta_p, gamma_p, D, C1, Q1, Q2,
                                  dc1, dc2, d12, d1, d2, pat1, pat2)
    solution = make_solution(final_val, pat1, pat2)
    if logger and final_val > best_val + 1e-9:
        logger.log_solution(final_val, solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()