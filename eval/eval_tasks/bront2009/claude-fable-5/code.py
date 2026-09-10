import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(5, args.time_limit) - 2.0

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")

    with open(args.instance_path) as f:
        inst = json.load(f)

    lam = float(inst["lambda"])
    T = float(inst["booking_horizon"]["T"])
    legs = inst["network"]["legs"]
    m_legs = len(legs)
    leg_idx = {leg["leg_id"]: i for i, leg in enumerate(legs)}
    cap = [float(leg["capacity"]) for leg in legs]

    prods = inst["products"]
    n = len(prods)
    pid_idx = {p["product_id"]: j for j, p in enumerate(prods)}
    fare = [float(p["fare"]) for p in prods]
    legs_of = [[leg_idx[l] for l in p["legs_used"]] for p in prods]

    segs = inst["segments"]
    L = len(segs)
    p_l = []
    cons = []
    vw = []
    v0 = []
    for s in segs:
        p_l.append(float(s["lambda_l"]) / lam if lam > 0 else 0.0)
        cs = [pid_idx[pid] for pid in s["consideration_set"]]
        cons.append(cs)
        vw.append({j: float(w) for j, w in zip(cs, s["preference_vector"])})
        v0.append(float(s["no_purchase_preference"]))

    relevant = sorted(set(j for cs in cons for j in cs))
    seg_of = {j: [] for j in relevant}
    for l in range(L):
        for j in cons[l]:
            seg_of[j].append((l, vw[l][j]))

    def write_solution(sol):
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)

    if lam <= 0 or not relevant:
        sol = {"objective_value": 0.0, "active_columns": []}
        if logger:
            logger.log_solution(0.0, sol)
        write_solution(sol)
        return

    # ----- column data computation -----
    def col_data(S):
        P = {}
        for l in range(L):
            inter = [j for j in cons[l] if j in S]
            if not inter:
                continue
            den = v0[l] + sum(vw[l][j] for j in inter)
            pl = p_l[l]
            for j in inter:
                P[j] = P.get(j, 0.0) + pl * vw[l][j] / den
        R = sum(fare[j] * pj for j, pj in P.items())
        Q = [0.0] * m_legs
        for j, pj in P.items():
            for i in legs_of[j]:
                Q[i] += pj
        return R, Q

    # ----- master LP -----
    mm = gp.Model("master")
    mm.Params.OutputFlag = 0
    mm.Params.Seed = 0
    mm.Params.MIPGap = 1e-4
    mm.Params.NumericFocus = 0
    mm.Params.Threads = 1
    mm.ModelSense = GRB.MAXIMIZE
    cap_con = [mm.addLConstr(gp.LinExpr() <= cap[i]) for i in range(m_legs)]
    time_con = mm.addLConstr(gp.LinExpr() <= T)

    cols = {}

    def add_col(S):
        S = frozenset(S)
        if not S or S in cols:
            return False
        R, Q = col_data(S)
        col = gp.Column()
        for i in range(m_legs):
            if Q[i] > 1e-15:
                col.addTerms(lam * Q[i], cap_con[i])
        col.addTerms(1.0, time_con)
        var = mm.addVar(lb=0.0, obj=lam * R, column=col)
        cols[S] = var
        return True

    # initial columns: full set, segment consideration sets, fare-nested prefixes
    add_col(relevant)
    for l in range(L):
        add_col(cons[l])
    order = sorted(relevant, key=lambda j: -fare[j])
    pref = []
    step = max(1, len(order) // 50)
    for k in range(0, len(order)):
        pref.append(order[k])
        if (k + 1) % step == 0 or k == len(order) - 1:
            add_col(pref)

    # ----- heuristic pricing (local search) -----
    def local_search(S0, r):
        S = set(S0)
        num = [0.0] * L
        den = list(v0)
        for l in range(L):
            for j in cons[l]:
                if j in S:
                    num[l] += r[j] * vw[l][j]
                    den[l] += vw[l][j]
        val = sum(p_l[l] * num[l] / den[l] for l in range(L))
        while time.time() < deadline:
            best_d = 1e-9
            best_j = None
            for j in relevant:
                d = 0.0
                if j in S:
                    for l, v in seg_of[j]:
                        d += p_l[l] * ((num[l] - r[j] * v) / (den[l] - v) - num[l] / den[l])
                else:
                    for l, v in seg_of[j]:
                        d += p_l[l] * ((num[l] + r[j] * v) / (den[l] + v) - num[l] / den[l])
                if d > best_d:
                    best_d = d
                    best_j = j
            if best_j is None:
                break
            j = best_j
            if j in S:
                S.discard(j)
                for l, v in seg_of[j]:
                    num[l] -= r[j] * v
                    den[l] -= v
            else:
                S.add(j)
                for l, v in seg_of[j]:
                    num[l] += r[j] * v
                    den[l] += v
            val += best_d
        return frozenset(S), val

    # ----- exact pricing MIP (built once, objective updated) -----
    pm = gp.Model("pricing")
    pm.Params.OutputFlag = 0
    pm.Params.Seed = 0
    pm.Params.MIPGap = 1e-4
    pm.Params.NumericFocus = 0
    pm.Params.Threads = 1
    pm.ModelSense = GRB.MAXIMIZE
    xv = {j: pm.addVar(vtype=GRB.BINARY, name=f"x{j}") for j in relevant}
    yv = []
    for l in range(L):
        ub0 = 1.0 / v0[l]
        y0 = pm.addVar(lb=0.0, ub=ub0)
        yd = {}
        expr = v0[l] * y0
        for j in cons[l]:
            yj = pm.addVar(lb=0.0, ub=ub0)
            yd[j] = yj
            expr += vw[l][j] * yj
            pm.addLConstr(yj <= y0)
            pm.addLConstr(yj <= ub0 * xv[j])
            pm.addLConstr(yj >= y0 - ub0 * (1.0 - xv[j]))
        pm.addLConstr(expr == 1.0)
        yv.append(yd)
    pm.update()

    def exact_pricing(r, warm, time_lim):
        for l in range(L):
            for j in cons[l]:
                yv[l][j].Obj = p_l[l] * r[j] * vw[l][j]
        for j in relevant:
            xv[j].Start = 1.0 if j in warm else 0.0
        pm.Params.TimeLimit = max(1.0, time_lim)
        pm.optimize()
        if pm.SolCount > 0:
            S = frozenset(j for j in relevant if xv[j].X > 0.5)
            proven = (pm.Status == GRB.OPTIMAL)
            return S, pm.ObjVal, proven
        return None, -float("inf"), False

    # ----- column generation loop -----
    best_obj = -1.0
    best_sol = {"objective_value": 0.0, "active_columns": []}
    tol = 1e-6
    max_iter = 20000
    it = 0

    def snapshot(objval):
        active = []
        for S, var in cols.items():
            t = var.X
            if t > 1e-9:
                active.append({
                    "offer_set": sorted(prods[j]["product_id"] for j in S),
                    "time_allocated": float(t),
                })
        return {"objective_value": float(objval), "active_columns": active}

    while it < max_iter and time.time() < deadline:
        it += 1
        mm.optimize()
        if mm.Status != GRB.OPTIMAL:
            break
        obj = mm.ObjVal
        if obj > best_obj + 1e-9:
            best_obj = obj
            best_sol = snapshot(obj)
            if logger:
                logger.log_solution(best_obj, best_sol)

        pi = [max(0.0, c.Pi) for c in cap_con]
        sigma = time_con.Pi
        r = [fare[j] - sum(pi[i] for i in legs_of[j]) for j in range(n)]

        # heuristic pricing from two starts
        cand = []
        S1, v1 = local_search([j for j in relevant if r[j] > 0], r)
        cand.append((S1, v1))
        S2, v2 = local_search(relevant, r)
        if S2 != S1:
            cand.append((S2, v2))
        added = False
        best_heur = frozenset()
        best_heur_val = -float("inf")
        for S, val in cand:
            if val > best_heur_val:
                best_heur_val = val
                best_heur = S
            if S and lam * val > sigma + tol and S not in cols:
                add_col(S)
                added = True
        if added:
            continue

        # exact pricing
        rem = deadline - time.time()
        if rem < 3.0:
            break
        S, val, proven = exact_pricing(r, best_heur, min(rem - 1.0, 120.0))
        if S is not None and S and lam * val > sigma + tol and S not in cols:
            add_col(S)
            continue
        # no improving column found (proven optimal, or timed out) -> stop
        break

    # final solve / snapshot
    mm.optimize()
    if mm.Status == GRB.OPTIMAL:
        obj = mm.ObjVal
        if obj > best_obj + 1e-9:
            best_obj = obj
            best_sol = snapshot(obj)
            if logger:
                logger.log_solution(best_obj, best_sol)
        elif obj >= best_obj - 1e-6:
            best_sol = snapshot(obj)

    if best_obj < 0:
        best_sol = {"objective_value": 0.0, "active_columns": []}
        if logger:
            logger.log_solution(0.0, best_sol)

    write_solution(best_sol)


if __name__ == "__main__":
    main()