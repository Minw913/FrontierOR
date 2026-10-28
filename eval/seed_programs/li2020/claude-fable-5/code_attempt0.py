import argparse
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def euclid(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.time()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    g = data["global_parameters"]
    theta = float(g["theta"])
    sigma = float(g["sigma"])
    alpha = float(g["alpha"])
    beta = float(g["beta"])
    gamma = float(g["gamma"])
    ap = float(g["alpha_prime"])
    bp = float(g["beta_prime"])
    gpr = float(g["gamma_prime"])

    ip = data["instance_parameters"]
    n1 = int(ip["num_level1_facilities"])
    n2 = int(ip["num_level2_facilities"])
    nc = int(ip["num_customers"])
    K = int(ip["num_interdiction_levels_K"])
    kmax = min(int(ip["k_max"]), K - 1)
    B = float(ip["total_interdiction_budget"])

    idp = data["interdiction_parameters"]

    def getlvl(dct, k):
        if dct is None:
            return None
        if str(k) in dct:
            return float(dct[str(k)])
        if k in dct:
            return float(dct[k])
        return None

    h1 = {0: 0.0}
    d1 = {0: 0.0}
    h2 = {0: 0.0}
    d2 = {0: 0.0}
    lv1 = [0]
    lv2 = [0]
    for k in range(1, kmax + 1):
        c = getlvl(idp.get("interdiction_cost_level1_h1"), k)
        r = getlvl(idp.get("capacity_reduction_level1_d1"), k)
        if c is not None and r is not None:
            h1[k] = c
            d1[k] = min(max(r, 0.0), 1.0)
            lv1.append(k)
        c = getlvl(idp.get("interdiction_cost_level2_h2"), k)
        r = getlvl(idp.get("capacity_reduction_level2_d2"), k)
        if c is not None and r is not None:
            h2[k] = c
            d2[k] = min(max(r, 0.0), 1.0)
            lv2.append(k)

    D = [float(x) for x in data["customers"]["demands"]]
    C1 = [float(x) for x in data["level1_facilities"]["capacity_type1"]]
    C21 = [float(x) for x in data["level2_facilities"]["capacity_type1"]]
    C22 = [float(x) for x in data["level2_facilities"]["capacity_type2"]]

    dist = data.get("distances", {})
    cx = data["customers"]["x_coordinates"]
    cy = data["customers"]["y_coordinates"]
    f1x = data["level1_facilities"]["x_coordinates"]
    f1y = data["level1_facilities"]["y_coordinates"]
    f2x = data["level2_facilities"]["x_coordinates"]
    f2y = data["level2_facilities"]["y_coordinates"]

    if "customer_to_level1_facility" in dist:
        dc1 = dist["customer_to_level1_facility"]
    else:
        dc1 = [[euclid(cx[i], cy[i], f1x[j], f1y[j]) for j in range(n1)] for i in range(nc)]
    if "customer_to_level2_facility" in dist:
        dc2 = dist["customer_to_level2_facility"]
    else:
        dc2 = [[euclid(cx[i], cy[i], f2x[m], f2y[m]) for m in range(n2)] for i in range(nc)]
    if "level1_to_level2_facility" in dist:
        df = dist["level1_to_level2_facility"]
    else:
        df = [[euclid(f1x[j], f1y[j], f2x[m], f2y[m]) for m in range(n2)] for j in range(n1)]
    # level2 -> level2 distances (for referral flows among level II facilities)
    dff = [[euclid(f2x[m1], f2y[m1], f2x[m2], f2y[m2]) for m2 in range(n2)] for m1 in range(n2)]

    Dmax = 0.0
    for row in dc1:
        for x in row:
            Dmax = max(Dmax, x)
    for row in dc2:
        for x in row:
            Dmax = max(Dmax, x)
    for row in df:
        for x in row:
            Dmax = max(Dmax, x)
    for row in dff:
        for x in row:
            Dmax = max(Dmax, x)

    # Valid bounds on capacity duals (marginal value of one unit of capacity)
    Mu = max(ap + sigma * gpr,
             max(alpha, beta) * Dmax + sigma * max(gamma * Dmax, gpr)) * 1.0 + 1e-6
    Mw = max(bp, gpr, beta * Dmax, gamma * Dmax) * 1.0 + 1e-6

    # ---------------- primal defender LP (for exact evaluation) ----------------
    def defender_cost(lev1, lev2):
        cap1 = [C1[j] * (1.0 - d1[lev1[j]]) for j in range(n1)]
        cap21 = [C21[m] * (1.0 - d2[lev2[m]]) for m in range(n2)]
        cap22 = [C22[m] * (1.0 - d2[lev2[m]]) for m in range(n2)]
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        lp = gp.Model(env=env)
        lp.Params.Threads = 1
        lp.Params.Seed = 0
        lp.Params.NumericFocus = 0
        x1 = lp.addVars(nc, n1, lb=0.0)
        x2 = lp.addVars(nc, n2, lb=0.0)
        y = lp.addVars(nc, n2, lb=0.0)
        o1 = lp.addVars(nc, lb=0.0)
        o2 = lp.addVars(nc, lb=0.0)
        r1 = lp.addVars(n1, n2, lb=0.0)
        r2 = lp.addVars(n2, n2, lb=0.0)
        ro1 = lp.addVars(n1, lb=0.0)
        ro2 = lp.addVars(n2, lb=0.0)
        for i in range(nc):
            lp.addConstr(gp.quicksum(x1[i, j] for j in range(n1)) +
                         gp.quicksum(x2[i, m] for m in range(n2)) + o1[i] == theta * D[i])
            lp.addConstr(gp.quicksum(y[i, m] for m in range(n2)) + o2[i] == (1.0 - theta) * D[i])
        for j in range(n1):
            lp.addConstr(gp.quicksum(r1[j, m] for m in range(n2)) + ro1[j]
                         == sigma * gp.quicksum(x1[i, j] for i in range(nc)))
            lp.addConstr(gp.quicksum(x1[i, j] for i in range(nc)) <= cap1[j])
        for m in range(n2):
            lp.addConstr(gp.quicksum(r2[m, mm] for mm in range(n2)) + ro2[m]
                         == sigma * gp.quicksum(x2[i, m] for i in range(nc)))
            lp.addConstr(gp.quicksum(x2[i, m] for i in range(nc)) <= cap21[m])
            lp.addConstr(gp.quicksum(y[i, m] for i in range(nc)) +
                         gp.quicksum(r1[j, m] for j in range(n1)) +
                         gp.quicksum(r2[mm, m] for mm in range(n2)) <= cap22[m])
        obj = gp.LinExpr()
        for i in range(nc):
            for j in range(n1):
                obj += alpha * dc1[i][j] * x1[i, j]
            for m in range(n2):
                obj += beta * dc2[i][m] * x2[i, m]
                obj += beta * dc2[i][m] * y[i, m]
            obj += (ap + sigma * gpr) * o1[i]
            obj += bp * o2[i]
        for j in range(n1):
            for m in range(n2):
                obj += gamma * df[j][m] * r1[j, m]
            obj += gpr * ro1[j]
        for m in range(n2):
            for mm in range(n2):
                obj += gamma * dff[m][mm] * r2[m, mm]
            obj += gpr * ro2[m]
        lp.setObjective(obj, GRB.MINIMIZE)
        lp.optimize()
        val = lp.ObjVal if lp.Status == GRB.OPTIMAL else None
        lp.dispose()
        env.dispose()
        return val

    # ---------------- single-level MIP via LP duality ----------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    a = m.addVars(nc, lb=-GRB.INFINITY, ub=ap + sigma * gpr, name="a")
    b = m.addVars(nc, lb=-GRB.INFINITY, ub=bp, name="b")
    pv = m.addVars(n1, lb=-GRB.INFINITY, ub=gpr, name="p")
    qv = m.addVars(n2, lb=-GRB.INFINITY, ub=gpr, name="q")
    u = m.addVars(n1, lb=-Mu, ub=0.0, name="u")
    v = m.addVars(n2, lb=-Mu, ub=0.0, name="v")
    w = m.addVars(n2, lb=-Mw, ub=0.0, name="w")

    z1 = {}
    for j in range(n1):
        for k in lv1:
            z1[j, k] = m.addVar(vtype=GRB.BINARY, name=f"z1_{j}_{k}")
    z2 = {}
    for mm in range(n2):
        for k in lv2:
            z2[mm, k] = m.addVar(vtype=GRB.BINARY, name=f"z2_{mm}_{k}")

    wu = {}
    for j in range(n1):
        for k in lv1[1:]:
            wu[j, k] = m.addVar(lb=-Mu, ub=0.0)
    wv = {}
    ww = {}
    for mm in range(n2):
        for k in lv2[1:]:
            wv[mm, k] = m.addVar(lb=-Mu, ub=0.0)
            ww[mm, k] = m.addVar(lb=-Mw, ub=0.0)

    # dual feasibility
    for i in range(nc):
        di1 = dc1[i]
        di2 = dc2[i]
        for j in range(n1):
            m.addConstr(a[i] - sigma * pv[j] + u[j] <= alpha * di1[j])
        for mm in range(n2):
            m.addConstr(a[i] - sigma * qv[mm] + v[mm] <= beta * di2[mm])
            m.addConstr(b[i] + w[mm] <= beta * di2[mm])
    for j in range(n1):
        for mm in range(n2):
            m.addConstr(pv[j] + w[mm] <= gamma * df[j][mm])
    for m1 in range(n2):
        for mm in range(n2):
            m.addConstr(qv[m1] + w[mm] <= gamma * dff[m1][mm])

    # interdiction selection
    for j in range(n1):
        m.addConstr(gp.quicksum(z1[j, k] for k in lv1) == 1)
    for mm in range(n2):
        m.addConstr(gp.quicksum(z2[mm, k] for k in lv2) == 1)
    m.addConstr(gp.quicksum(h1[k] * z1[j, k] for j in range(n1) for k in lv1[1:]) +
                gp.quicksum(h2[k] * z2[mm, k] for mm in range(n2) for k in lv2[1:]) <= B)

    # McCormick for products z*dual (exact since z binary and coefficients push to lower bound)
    for j in range(n1):
        for k in lv1[1:]:
            m.addConstr(wu[j, k] >= u[j] - Mu * (1 - z1[j, k]))
            m.addConstr(wu[j, k] >= -Mu * z1[j, k])
    for mm in range(n2):
        for k in lv2[1:]:
            m.addConstr(wv[mm, k] >= v[mm] - Mu * (1 - z2[mm, k]))
            m.addConstr(wv[mm, k] >= -Mu * z2[mm, k])
            m.addConstr(ww[mm, k] >= w[mm] - Mw * (1 - z2[mm, k]))
            m.addConstr(ww[mm, k] >= -Mw * z2[mm, k])

    obj = gp.LinExpr()
    for i in range(nc):
        obj += theta * D[i] * a[i]
        obj += (1.0 - theta) * D[i] * b[i]
    for j in range(n1):
        obj += C1[j] * u[j]
        for k in lv1[1:]:
            obj += -C1[j] * d1[k] * wu[j, k]
    for mm in range(n2):
        obj += C21[mm] * v[mm]
        obj += C22[mm] * w[mm]
        for k in lv2[1:]:
            obj += -C21[mm] * d2[k] * wv[mm, k]
            obj += -C22[mm] * d2[k] * ww[mm, k]
    m.setObjective(obj, GRB.MAXIMIZE)

    z1_order = [(j, k) for j in range(n1) for k in lv1]
    z2_order = [(mm, k) for mm in range(n2) for k in lv2]
    z1_vars = [z1[key] for key in z1_order]
    z2_vars = [z2[key] for key in z2_order]

    def extract_pattern(vals1, vals2):
        lev1 = [0] * n1
        lev2 = [0] * n2
        for idx, (j, k) in enumerate(z1_order):
            if vals1[idx] > 0.5:
                lev1[j] = k
        for idx, (mm, k) in enumerate(z2_order):
            if vals2[idx] > 0.5:
                lev2[mm] = k
        return lev1, lev2

    def pattern_dict(lev1, lev2):
        pat = {}
        for j in range(n1):
            pat[f"S1_{j}"] = int(lev1[j])
        for mm in range(n2):
            pat[f"S2_{mm}"] = int(lev2[mm])
        return pat

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                vals1 = model.cbGetSolution(z1_vars)
                vals2 = model.cbGetSolution(z2_vars)
                lev1, lev2 = extract_pattern(vals1, vals2)
                if logger:
                    logger.log_solution(objval, {
                        "objective_value": objval,
                        "interdiction_pattern": pattern_dict(lev1, lev2),
                    })
            except Exception:
                pass

    elapsed = time.time() - t0
    reserve = 8.0
    tl = max(2.0, args.time_limit - elapsed - reserve)
    m.Params.TimeLimit = tl

    try:
        m.optimize(callback)
    except Exception:
        pass

    if m.SolCount > 0:
        vals1 = [zv.X for zv in z1_vars]
        vals2 = [zv.X for zv in z2_vars]
        lev1, lev2 = extract_pattern(vals1, vals2)
        mip_obj = m.ObjVal
    else:
        lev1 = [0] * n1
        lev2 = [0] * n2
        mip_obj = None

    # Exact defender cost for the chosen pattern
    final_obj = None
    try:
        final_obj = defender_cost(lev1, lev2)
    except Exception:
        final_obj = None
    if final_obj is None:
        final_obj = mip_obj if mip_obj is not None else 0.0

    solution = {
        "objective_value": float(final_obj),
        "interdiction_pattern": pattern_dict(lev1, lev2),
    }

    if logger:
        logger.log_solution(float(final_obj), solution)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    m.dispose()
    env.dispose()


if __name__ == "__main__":
    main()