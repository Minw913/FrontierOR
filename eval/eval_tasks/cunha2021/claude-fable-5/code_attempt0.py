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

    t0 = time.time()

    with open(args.instance_path) as f:
        data = json.load(f)

    R = int(data["num_retailers"])
    W = int(data["num_warehouses"])
    T = int(data["num_periods"])
    assign = [int(v) for v in data["retailer_warehouse_assignment"]]
    d = [[int(v) for v in row] for row in data["demands"]]
    fP = [float(v) for v in data["fixed_costs"]["plant"]]
    fW = [[float(v) for v in row] for row in data["fixed_costs"]["warehouses"]]
    fR = [[float(v) for v in row] for row in data["fixed_costs"]["retailers"]]
    hP = float(data["holding_costs"]["plant"])
    hW = float(data["holding_costs"]["warehouses"])
    hR = [float(v) for v in data["holding_costs"]["retailers"]]

    wsets = [[] for _ in range(W)]
    for r, w in enumerate(assign):
        wsets[w].append(r)

    logger = SolutionLogger(args.log_path, sense="minimize") if (args.log_path and SolutionLogger) else None

    # remaining demands (from period t through horizon end)
    remR = [[0] * (T + 1) for _ in range(R)]
    for r in range(R):
        for t in range(T - 1, -1, -1):
            remR[r][t] = remR[r][t + 1] + d[r][t]
    remW = [[0] * (T + 1) for _ in range(W)]
    for w in range(W):
        for t in range(T - 1, -1, -1):
            remW[w][t] = remW[w][t + 1] + sum(d[r][t] for r in wsets[w])
    remP = [sum(remW[w][t] for w in range(W)) for t in range(T + 1)]

    def make_solution(xP, xWv, xRv):
        eps = 1e-6

        def cl(v):
            return 0.0 if v < eps else float(v)

        xP = [cl(v) for v in xP]
        xWv = [[cl(v) for v in row] for row in xWv]
        xRv = [[cl(v) for v in row] for row in xRv]
        yP = [1 if xP[t] > 0 else 0 for t in range(T)]
        yW = [[1 if xWv[w][t] > 0 else 0 for t in range(T)] for w in range(W)]
        yR = [[1 if xRv[r][t] > 0 else 0 for t in range(T)] for r in range(R)]

        sR = [[0.0] * T for _ in range(R)]
        for r in range(R):
            cur = 0.0
            for t in range(T):
                cur = cur + xRv[r][t] - d[r][t]
                if abs(cur) < 1e-6:
                    cur = 0.0
                cur = max(cur, 0.0)
                sR[r][t] = cur
        sW = [[0.0] * T for _ in range(W)]
        for w in range(W):
            cur = 0.0
            for t in range(T):
                out = sum(xRv[r][t] for r in wsets[w])
                cur = cur + xWv[w][t] - out
                if abs(cur) < 1e-6:
                    cur = 0.0
                cur = max(cur, 0.0)
                sW[w][t] = cur
        sP = [0.0] * T
        cur = 0.0
        for t in range(T):
            out = sum(xWv[w][t] for w in range(W))
            cur = cur + xP[t] - out
            if abs(cur) < 1e-6:
                cur = 0.0
            cur = max(cur, 0.0)
            sP[t] = cur

        obj = 0.0
        obj += sum(fP[t] for t in range(T) if yP[t])
        obj += sum(fW[w][t] for w in range(W) for t in range(T) if yW[w][t])
        obj += sum(fR[r][t] for r in range(R) for t in range(T) if yR[r][t])
        obj += hP * sum(sP)
        obj += hW * sum(sW[w][t] for w in range(W) for t in range(T))
        obj += sum(hR[r] * sum(sR[r]) for r in range(R))

        sol = {
            "objective_value": float(obj),
            "x_plant": xP,
            "s_plant": sP,
            "y_plant": yP,
            "x_warehouses": xWv,
            "s_warehouses": sW,
            "y_warehouses": yW,
            "x_retailers": xRv,
            "s_retailers": sR,
            "y_retailers": yR,
        }
        return obj, sol

    # ---------- trivial (just-in-time) feasible solution ----------
    xR0 = [[float(d[r][t]) for t in range(T)] for r in range(R)]
    xW0 = [[float(sum(d[r][t] for r in wsets[w])) for t in range(T)] for w in range(W)]
    xP0 = [float(sum(d[r][t] for r in range(R))) for t in range(T)]
    obj0, sol0 = make_solution(xP0, xW0, xR0)
    best = {"obj": obj0, "sol": sol0}
    if logger:
        logger.log_solution(obj0, sol0)

    def write_out():
        with open(args.solution_path, "w") as f:
            json.dump(best["sol"], f)

    remaining = args.time_limit - (time.time() - t0)
    if remaining < 5:
        write_out()
        return

    # decide formulation
    nz = sum(u + 1 for r in range(R) for u in range(T) if d[r][u] > 0)
    use_fl = nz <= 120000 and (args.time_limit >= 60 or nz <= 30000)

    m = gp.Model("mlulsp")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    if use_fl:
        # facility-location (demand-disaggregated) reformulation
        yP = m.addVars(T, vtype=GRB.BINARY)
        yW = m.addVars(W, T, vtype=GRB.BINARY)
        yR = m.addVars(R, T, vtype=GRB.BINARY)
        m.setAttr("Obj", [yP[t] for t in range(T)], fP)
        m.setAttr("Obj", [yW[w, t] for w in range(W) for t in range(T)],
                  [fW[w][t] for w in range(W) for t in range(T)])
        m.setAttr("Obj", [yR[r, t] for r in range(R) for t in range(T)],
                  [fR[r][t] for r in range(R) for t in range(T)])

        zP, zW, zR = {}, {}, {}
        for r in range(R):
            w = assign[r]
            for u in range(T):
                du = d[r][u]
                if du <= 0:
                    continue
                for t in range(u + 1):
                    k = du * (u - t)
                    zP[r, t, u] = m.addVar(ub=1.0, obj=k * hP)
                    zW[r, t, u] = m.addVar(ub=1.0, obj=k * (hW - hP))
                    zR[r, t, u] = m.addVar(ub=1.0, obj=k * (hR[r] - hW))
        m.update()

        for r in range(R):
            w = assign[r]
            for u in range(T):
                if d[r][u] <= 0:
                    continue
                m.addConstr(gp.quicksum(zP[r, t, u] for t in range(u + 1)) == 1)
                m.addConstr(gp.quicksum(zW[r, t, u] for t in range(u + 1)) == 1)
                m.addConstr(gp.quicksum(zR[r, t, u] for t in range(u + 1)) == 1)
                eR = gp.LinExpr()
                eW = gp.LinExpr()
                eP = gp.LinExpr()
                for s in range(u):
                    eR += zR[r, s, u]
                    eW += zW[r, s, u]
                    eP += zP[r, s, u]
                    m.addConstr(eR <= eW)
                    m.addConstr(eW <= eP)
                for t in range(u + 1):
                    m.addConstr(zP[r, t, u] <= yP[t])
                    m.addConstr(zW[r, t, u] <= yW[w, t])
                    m.addConstr(zR[r, t, u] <= yR[r, t])

        # MIP start: just-in-time
        for (r, t, u), v in zP.items():
            v.Start = 1.0 if t == u else 0.0
        for (r, t, u), v in zW.items():
            v.Start = 1.0 if t == u else 0.0
        for (r, t, u), v in zR.items():
            v.Start = 1.0 if t == u else 0.0
        for t in range(T):
            yP[t].Start = 1 if sum(d[r][t] for r in range(R)) > 0 else 0
        for w in range(W):
            for t in range(T):
                yW[w, t].Start = 1 if sum(d[r][t] for r in wsets[w]) > 0 else 0
        for r in range(R):
            for t in range(T):
                yR[r, t].Start = 1 if d[r][t] > 0 else 0

        keysP = list(zP.keys()); varsP = [zP[k] for k in keysP]
        keysW = list(zW.keys()); varsW = [zW[k] for k in keysW]
        keysR = list(zR.keys()); varsR = [zR[k] for k in keysR]

        def build_x(vP, vW, vR):
            xP = [0.0] * T
            xWv = [[0.0] * T for _ in range(W)]
            xRv = [[0.0] * T for _ in range(R)]
            for (r, t, u), v in zip(keysP, vP):
                xP[t] += d[r][u] * v
            for (r, t, u), v in zip(keysW, vW):
                xWv[assign[r]][t] += d[r][u] * v
            for (r, t, u), v in zip(keysR, vR):
                xRv[r][t] += d[r][u] * v
            return xP, xWv, xRv

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj >= best["obj"] - 1e-6:
                    return
                try:
                    vP = model.cbGetSolution(varsP)
                    vW = model.cbGetSolution(varsW)
                    vR = model.cbGetSolution(varsR)
                except Exception:
                    return
                o, sol = make_solution(*build_x(vP, vW, vR))
                if o < best["obj"] - 1e-9:
                    best["obj"] = o
                    best["sol"] = sol
                    if logger:
                        logger.log_solution(o, sol)

        def final_extract():
            vP = m.getAttr("X", varsP)
            vW = m.getAttr("X", varsW)
            vR = m.getAttr("X", varsR)
            return build_x(vP, vW, vR)

    else:
        # standard flow formulation with tight big-M
        xPv = m.addVars(T, lb=0.0)
        sPv = m.addVars(T, lb=0.0, obj=hP)
        yP = m.addVars(T, vtype=GRB.BINARY)
        xWm = m.addVars(W, T, lb=0.0)
        sWm = m.addVars(W, T, lb=0.0, obj=hW)
        yW = m.addVars(W, T, vtype=GRB.BINARY)
        xRm = m.addVars(R, T, lb=0.0)
        sRm = m.addVars(R, T, lb=0.0)
        yR = m.addVars(R, T, vtype=GRB.BINARY)

        m.setAttr("Obj", [yP[t] for t in range(T)], fP)
        m.setAttr("Obj", [yW[w, t] for w in range(W) for t in range(T)],
                  [fW[w][t] for w in range(W) for t in range(T)])
        m.setAttr("Obj", [yR[r, t] for r in range(R) for t in range(T)],
                  [fR[r][t] for r in range(R) for t in range(T)])
        m.setAttr("Obj", [sRm[r, t] for r in range(R) for t in range(T)],
                  [hR[r] for r in range(R) for t in range(T)])

        for t in range(T):
            prev = sPv[t - 1] if t > 0 else 0.0
            m.addConstr(prev + xPv[t] == gp.quicksum(xWm[w, t] for w in range(W)) + sPv[t])
            m.addConstr(xPv[t] <= remP[t] * yP[t])
        for w in range(W):
            for t in range(T):
                prev = sWm[w, t - 1] if t > 0 else 0.0
                m.addConstr(prev + xWm[w, t] == gp.quicksum(xRm[r, t] for r in wsets[w]) + sWm[w, t])
                m.addConstr(xWm[w, t] <= remW[w][t] * yW[w, t])
        for r in range(R):
            for t in range(T):
                prev = sRm[r, t - 1] if t > 0 else 0.0
                m.addConstr(prev + xRm[r, t] == d[r][t] + sRm[r, t])
                m.addConstr(xRm[r, t] <= remR[r][t] * yR[r, t])

        # MIP start: just-in-time
        for t in range(T):
            xPv[t].Start = xP0[t]
            sPv[t].Start = 0.0
            yP[t].Start = 1 if xP0[t] > 0 else 0
        for w in range(W):
            for t in range(T):
                xWm[w, t].Start = xW0[w][t]
                sWm[w, t].Start = 0.0
                yW[w, t].Start = 1 if xW0[w][t] > 0 else 0
        for r in range(R):
            for t in range(T):
                xRm[r, t].Start = xR0[r][t]
                sRm[r, t].Start = 0.0
                yR[r, t].Start = 1 if xR0[r][t] > 0 else 0

        varsXP = [xPv[t] for t in range(T)]
        varsXW = [xWm[w, t] for w in range(W) for t in range(T)]
        varsXR = [xRm[r, t] for r in range(R) for t in range(T)]

        def build_x(vP, vW, vR):
            xP = list(vP)
            xWv = [[vW[w * T + t] for t in range(T)] for w in range(W)]
            xRv = [[vR[r * T + t] for t in range(T)] for r in range(R)]
            return xP, xWv, xRv

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj >= best["obj"] - 1e-6:
                    return
                try:
                    vP = model.cbGetSolution(varsXP)
                    vW = model.cbGetSolution(varsXW)
                    vR = model.cbGetSolution(varsXR)
                except Exception:
                    return
                o, sol = make_solution(*build_x(vP, vW, vR))
                if o < best["obj"] - 1e-9:
                    best["obj"] = o
                    best["sol"] = sol
                    if logger:
                        logger.log_solution(o, sol)

        def final_extract():
            vP = m.getAttr("X", varsXP)
            vW = m.getAttr("X", varsXW)
            vR = m.getAttr("X", varsXR)
            return build_x(vP, vW, vR)

    remaining = args.time_limit - (time.time() - t0) - 3.0
    m.Params.TimeLimit = max(1.0, remaining)

    try:
        m.optimize(cb)
    except Exception:
        pass

    try:
        if m.SolCount > 0:
            o, sol = make_solution(*final_extract())
            if o < best["obj"] - 1e-9:
                best["obj"] = o
                best["sol"] = sol
                if logger:
                    logger.log_solution(o, sol)
    except Exception:
        pass

    write_out()


if __name__ == "__main__":
    main()