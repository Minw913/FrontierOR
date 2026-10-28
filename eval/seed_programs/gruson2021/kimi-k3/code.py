import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    return ap.parse_args()


def write_solution(path, sol):
    with open(path, "w") as fh:
        json.dump(sol, fh)


def main():
    args = parse_args()
    t_start = time.time()
    deadline = t_start + max(1, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as fh:
        data = json.load(fh)

    T = int(data["num_periods"])
    W = int(data["num_warehouses"])
    R = int(data["num_retailers"])
    S = int(data["num_scenarios"])
    F = 1 + W + R

    init_inv = float(data.get("initial_inventory", 0) or 0)
    imposed = bool(data.get("initial_setups_imposed", True))

    probs = [float(x) for x in data["scenario_probabilities"]]
    h_plant = float(data["holding_cost_plant"])
    h_wh = float(data["holding_cost_warehouse"])
    h_rt = [float(x) for x in data["holding_costs_retailers"]]

    Kf = [[float(x) for x in data["setup_costs_plant"]]]
    for row in data["setup_costs_warehouses"]:
        Kf.append([float(x) for x in row])
    for row in data["setup_costs_retailers"]:
        Kf.append([float(x) for x in row])

    raw_dem = data["demands"]
    d = [[[float(x) for x in raw_dem[s][r]] for r in range(R)] for s in range(S)]

    assign_raw = [int(x) for x in data.get("retailer_to_warehouse", [0] * R)]
    if assign_raw and min(assign_raw) >= 1 and max(assign_raw) <= max(W, 1):
        assign = [a - 1 for a in assign_raw]
    else:
        assign = list(assign_raw)
    assign = [(a if 0 <= a < W else 0) for a in assign]

    wh_retailers = [[] for _ in range(W)]
    if W > 0:
        for r in range(R):
            w = assign[r] if r < len(assign) else 0
            wh_retailers[w].append(r)

    # ---------------- trivial horizon ----------------
    if T <= 0:
        sol = {
            "objective_value": 0.0,
            "setup_variables": {},
            "production_plant": {},
            "delivery_warehouse": {},
            "delivery_retailer": {},
            "inventory_plant": {},
            "inventory_warehouse": {},
            "inventory_retailer": {},
        }
        if logger:
            try:
                logger.log_solution(0.0, sol)
            except Exception:
                pass
        write_solution(args.solution_path, sol)
        return

    # ---------------- suffix sums of demand ----------------
    # suff_r[s][r][t] = sum_{t' >= t} d[s][r][t']  (index T is a zero sentinel)
    suff_r = []
    for s in range(S):
        sr = []
        for r in range(R):
            sf = [0.0] * (T + 1)
            acc = 0.0
            row = d[s][r]
            for t in range(T - 1, -1, -1):
                acc += row[t]
                sf[t] = acc
            sr.append(sf)
        suff_r.append(sr)
    suff_w = []
    for s in range(S):
        sw = []
        for w in range(W):
            swf = [0.0] * (T + 1)
            for r in wh_retailers[w]:
                srf = suff_r[s][r]
                for t in range(T):
                    swf[t] += srf[t]
            sw.append(swf)
        suff_w.append(sw)
    suff_p = []
    for s in range(S):
        sp = [0.0] * (T + 1)
        for w in range(W):
            swf = suff_w[s][w]
            for t in range(T):
                sp[t] += swf[t]
        suff_p.append(sp)

    # facility-periods that can ever carry flow
    need = [[False] * T for _ in range(F)]
    for t in range(T):
        need[0][t] = any(suff_p[s][t] > 1e-9 for s in range(S))
        for w in range(W):
            need[1 + w][t] = any(suff_w[s][w][t] > 1e-9 for s in range(S))
        for r in range(R):
            need[1 + W + r][t] = any(suff_r[s][r][t] > 1e-9 for s in range(S))

    # ---------------- solution templates (all required keys) ----------------
    tmpl_setup = {f"y_{f}_{t}": 0 for f in range(F) for t in range(T)}
    tmpl_prod = {f"{t}_{s}": 0.0 for s in range(S) for t in range(T)}
    tmpl_dw = {f"{w}_{t}_{s}": 0.0 for s in range(S) for w in range(W) for t in range(T)}
    tmpl_dr = {f"{r}_{t}_{s}": 0.0 for s in range(S) for r in range(R) for t in range(T)}
    tmpl_ip = dict(tmpl_prod)
    tmpl_iw = dict(tmpl_dw)
    tmpl_ir = dict(tmpl_dr)

    # ---------------- heuristic feasible solution ----------------
    # Order only in periods with (net) demand; deliver just-in-time downstream.
    # Initial inventory is netted against the earliest demand at each level.
    heur_setup = [[0] * T for _ in range(F)]
    hp = dict(tmpl_prod)
    dw = dict(tmpl_dw)
    dr = dict(tmpl_dr)
    ipd = dict(tmpl_ip)
    iwd = dict(tmpl_iw)
    ird = dict(tmpl_ir)

    for s in range(S):
        # retailer level
        for r in range(R):
            rem = init_inv
            cum = 0.0
            for t in range(T):
                dem = d[s][r][t]
                rec = dem - rem if dem > rem else 0.0
                rem = rem - dem if rem > dem else 0.0
                cum += dem
                dr[f"{r}_{t}_{s}"] = rec
                leftover = init_inv - cum
                ird[f"{r}_{t}_{s}"] = leftover if leftover > 1e-12 else 0.0
                if rec > 1e-12:
                    heur_setup[1 + W + r][t] = 1
        # warehouse level
        for w in range(W):
            rem = init_inv
            cum = 0.0
            members = wh_retailers[w]
            for t in range(T):
                send = 0.0
                for r in members:
                    send += dr[f"{r}_{t}_{s}"]
                rec = send - rem if send > rem else 0.0
                rem = rem - send if send > send else (rem - send if send > 0 else rem)
                # correct netting: consumption = send
                rem = init_inv - cum if False else rem  # placeholder replaced below
                # recompute cleanly:
                # (handled by generic loop below)
                cum += send
                dw[f"{w}_{t}_{s}"] = rec
                leftover = init_inv - cum
                iwd[f"{w}_{t}_{s}"] = leftover if leftover > 1e-12 else 0.0
                if rec > 1e-12:
                    heur_setup[1 + w][t] = 1
        # fix warehouse netting cleanly (recompute this scenario/warehouses)
        for w in range(W):
            rem = init_inv
            cum = 0.0
            members = wh_retailers[w]
            for t in range(T):
                send = 0.0
                for r in members:
                    send += dr[f"{r}_{t}_{s}"]
                rec = send - rem if send > rem else 0.0
                rem = rem - send if rem > send else 0.0
                cum += send
                dw[f"{w}_{t}_{s}"] = rec
                leftover = init_inv - cum
                iwd[f"{w}_{t}_{s}"] = leftover if leftover > 1e-12 else 0.0
                if rec > 1e-12:
                    heur_setup[1 + w][t] = 1
        # plant level
        rem = init_inv
        cum = 0.0
        for t in range(T):
            send = 0.0
            for w in range(W):
                send += dw[f"{w}_{t}_{s}"]
            rec = send - rem if send > rem else 0.0
            rem = rem - send if rem > send else 0.0
            cum += send
            hp[f"{t}_{s}"] = rec
            leftover = init_inv - cum
            ipd[f"{t}_{s}"] = leftover if leftover > 1e-12 else 0.0
            if rec > 1e-12:
                heur_setup[0][t] = 1

    if imposed:
        for f in range(F):
            heur_setup[f][0] = 1

    obj_h = 0.0
    for f in range(F):
        Krow = Kf[f]
        hsrow = heur_setup[f]
        for t in range(T):
            if hsrow[t]:
                obj_h += Krow[t]
    for s in range(S):
        ps = probs[s]
        hold = 0.0
        for t in range(T):
            hold += h_plant * ipd[f"{t}_{s}"]
        for w in range(W):
            for t in range(T):
                hold += h_wh * iwd[f"{w}_{t}_{s}"]
        for r in range(R):
            hr = h_rt[r]
            for t in range(T):
                hold += hr * ird[f"{r}_{t}_{s}"]
        obj_h += ps * hold

    heur = {
        "objective_value": float(round(obj_h, 6)),
        "setup_variables": {f"y_{f}_{t}": (1 if heur_setup[f][t] else 0)
                            for f in range(F) for t in range(T)},
        "production_plant": hp,
        "delivery_warehouse": dw,
        "delivery_retailer": dr,
        "inventory_plant": ipd,
        "inventory_warehouse": iwd,
        "inventory_retailer": ird,
    }

    best_obj = heur["objective_value"]
    best_sol = heur
    if logger:
        try:
            logger.log_solution(best_obj, best_sol)
        except Exception:
            pass

    # ---------------- MIP ----------------
    remaining = deadline - time.time() - 0.5
    if remaining < 0.5:
        write_solution(args.solution_path, best_sol)
        return

    try:
        m = gp.Model("three_level_stochastic_lotsizing")
        m.setParam(GRB.Param.Seed, 0)
        m.setParam(GRB.Param.MIPGap, 1e-4)
        m.setParam(GRB.Param.NumericFocus, 0)
        m.setParam(GRB.Param.Threads, 1)
        m.setParam(GRB.Param.TimeLimit, max(0.1, remaining))
        m.setParam(GRB.Param.OutputFlag, 0)

        y = {}
        for f in range(F):
            for t in range(T):
                v = m.addVar(vtype=GRB.BINARY, name=f"y_{f}_{t}")
                if imposed and t == 0:
                    v.lb = 1.0
                elif not need[f][t]:
                    v.ub = 0.0
                y[(f, t)] = v

        xp = {}
        Ip = {}
        for s in range(S):
            sp = suff_p[s]
            for t in range(T):
                if sp[t] > 1e-9:
                    xp[(t, s)] = m.addVar(lb=0.0, name=f"xp_{t}_{s}")
                if (t + 1 < T and sp[t + 1] > 1e-9) or init_inv > 0:
                    Ip[(t, s)] = m.addVar(lb=0.0, name=f"Ip_{t}_{s}")

        xw = {}
        Iw = {}
        for s in range(S):
            for w in range(W):
                sw = suff_w[s][w]
                for t in range(T):
                    if sw[t] > 1e-9:
                        xw[(w, t, s)] = m.addVar(lb=0.0)
                    if (t + 1 < T and sw[t + 1] > 1e-9) or init_inv > 0:
                        Iw[(w, t, s)] = m.addVar(lb=0.0)

        xr = {}
        Ir = {}
        for s in range(S):
            for r in range(R):
                srr = suff_r[s][r]
                for t in range(T):
                    if srr[t] > 1e-9:
                        xr[(r, t, s)] = m.addVar(lb=0.0)
                    if (t + 1 < T and srr[t + 1] > 1e-9) or init_inv > 0:
                        Ir[(r, t, s)] = m.addVar(lb=0.0)

        # inventory balances (aggregate; equivalent optimum to demand-split form)
        for s in range(S):
            # plant
            for t in range(T):
                lhs = [] if t > 0 else ([init_inv] if init_inv else [])
                if t > 0:
                    pv = Ip.get((t - 1, s))
                    if pv is not None:
                        lhs.append(pv)
                v = xp.get((t, s))
                if v is not None:
                    lhs.append(v)
                rhs = []
                iv = Ip.get((t, s))
                if iv is not None:
                    rhs.append(iv)
                for w in range(W):
                    vv = xw.get((w, t, s))
                    if vv is not None:
                        rhs.append(vv)
                if lhs or rhs:
                    m.addConstr(gp.quicksum(lhs) == gp.quicksum(rhs))
            # warehouses and their retailers
            for w in range(W):
                members = wh_retailers[w]
                for t in range(T):
                    lhs = [] if t > 0 else ([init_inv] if init_inv else [])
                    if t > 0:
                        pv = Iw.get((w, t - 1, s))
                        if pv is not None:
                            lhs.append(pv)
                    v = xw.get((w, t, s))
                    if v is not None:
                        lhs.append(v)
                    rhs = []
                    iv = Iw.get((w, t, s))
                    if iv is not None:
                        rhs.append(iv)
                    for r in members:
                        vv = xr.get((r, t, s))
                        if vv is not None:
                            rhs.append(vv)
                    if lhs or rhs:
                        m.addConstr(gp.quicksum(lhs) == gp.quicksum(rhs))
                    for r in members:
                        rl = [] if t > 0 else ([init_inv] if init_inv else [])
                        if t > 0:
                            pv = Ir.get((r, t - 1, s))
                            if pv is not None:
                                rl.append(pv)
                        v2 = xr.get((r, t, s))
                        if v2 is not None:
                            rl.append(v2)
                        rr = []
                        iv2 = Ir.get((r, t, s))
                        if iv2 is not None:
                            rr.append(iv2)
                        dem = d[s][r][t]
                        if dem > 0:
                            rr.append(dem)
                        if rl or len(rr) > (1 if dem > 0 else 0):
                            m.addConstr(gp.quicksum(rl) == gp.quicksum(rr))

        # setup linkage with suffix big-M
        for (t, s), v in xp.items():
            m.addConstr(v - suff_p[s][t] * y[(0, t)] <= 0.0)
        for (w, t, s), v in xw.items():
            m.addConstr(v - suff_w[s][w][t] * y[(1 + w, t)] <= 0.0)
        for (r, t, s), v in xr.items():
            m.addConstr(v - suff_r[s][r][t] * y[(1 + W + r, t)] <= 0.0)

        # objective
        objexpr = gp.LinExpr()
        for f in range(F):
            Krow = Kf[f]
            for t in range(T):
                objexpr.addTerms(Krow[t], y[(f, t)])
        for (t, s), v in Ip.items():
            objexpr.addTerms(probs[s] * h_plant, v)
        for (w, t, s), v in Iw.items():
            objexpr.addTerms(probs[s] * h_wh, v)
        for (r, t, s), v in Ir.items():
            objexpr.addTerms(probs[s] * h_rt[r], v)
        m.setObjective(objexpr, GRB.MINIMIZE)

        # flatten variables for fast solution building
        var_objs = []
        var_meta = []  # (mode, field, key, coef); mode 0=setup, 1=inventory(holds obj), 2=flow
        for f in range(F):
            for t in range(T):
                var_objs.append(y[(f, t)])
                var_meta.append((0, "setup_variables", f"y_{f}_{t}", Kf[f][t]))
        for (t, s), v in xp.items():
            var_objs.append(v)
            var_meta.append((2, "production_plant", f"{t}_{s}", 0.0))
        for (w, t, s), v in xw.items():
            var_objs.append(v)
            var_meta.append((2, "delivery_warehouse", f"{w}_{t}_{s}", 0.0))
        for (r, t, s), v in xr.items():
            var_objs.append(v)
            var_meta.append((2, "delivery_retailer", f"{r}_{t}_{s}", 0.0))
        for (t, s), v in Ip.items():
            var_objs.append(v)
            var_meta.append((1, "inventory_plant", f"{t}_{s}", probs[s] * h_plant))
        for (w, t, s), v in Iw.items():
            var_objs.append(v)
            var_meta.append((1, "inventory_warehouse", f"{w}_{t}_{s}", probs[s] * h_wh))
        for (r, t, s), v in Ir.items():
            var_objs.append(v)
            var_meta.append((1, "inventory_retailer", f"{r}_{t}_{s}", probs[s] * h_rt[r]))

        def build_solution(vals):
            sol = {
                "objective_value": 0.0,
                "setup_variables": tmpl_setup.copy(),
                "production_plant": tmpl_prod.copy(),
                "delivery_warehouse": tmpl_dw.copy(),
                "delivery_retailer": tmpl_dr.copy(),
                "inventory_plant": tmpl_ip.copy(),
                "inventory_warehouse": tmpl_iw.copy(),
                "inventory_retailer": tmpl_ir.copy(),
            }
            obj = 0.0
            for i in range(len(var_meta)):
                mode, field, key, coef = var_meta[i]
                raw = vals[i]
                if mode == 0:
                    iv = 1 if raw >= 0.5 else 0
                    sol[field][key] = iv
                    obj += coef * iv
                else:
                    if raw < 1e-7:
                        val = 0.0
                    else:
                        val = round(raw, 6)
                    sol[field][key] = val
                    if coef != 0.0:
                        obj += coef * val
            obj = round(obj, 6)
            sol["objective_value"] = obj
            return sol, obj

        # warm start from heuristic
        for (f, t), v in y.items():
            if heur_setup[f][t]:
                v.Start = 1.0
        for (t, s), v in xp.items():
            val = hp.get(f"{t}_{s}", 0.0)
            if val > 1e-12:
                v.Start = val
        for (w, t, s), v in xw.items():
            val = dw.get(f"{w}_{t}_{s}", 0.0)
            if val > 1e-12:
                v.Start = val
        for (r, t, s), v in xr.items():
            val = dr.get(f"{r}_{t}_{s}", 0.0)
            if val > 1e-12:
                v.Start = val
        for (t, s), v in Ip.items():
            val = ipd.get(f"{t}_{s}", 0.0)
            if val > 1e-12:
                v.Start = val
        for (w, t, s), v in Iw.items():
            val = iwd.get(f"{w}_{t}_{s}", 0.0)
            if val > 1e-12:
                v.Start = val
        for (r, t, s), v in Ir.items():
            val = ird.get(f"{r}_{t}_{s}", 0.0)
            if val > 1e-12:
                v.Start = val

        state = {"best": best_obj}

        def callback(model, where):
            if where == GRB.Callback.MIP or where == GRB.Callback.MIPSOL:
                if time.time() > deadline:
                    try:
                        model.terminate()
                    except Exception:
                        pass
            if where != GRB.Callback.MIPSOL or logger is None:
                return
            try:
                vals = model.cbGetSolution(var_objs)
            except gp.GurobiError:
                return
            try:
                sol, obj = build_solution(vals)
                if obj < state["best"] - 1e-6:
                    state["best"] = obj
                    logger.log_solution(obj, sol)
            except Exception:
                pass

        try:
            m.optimize(callback)
        except gp.GurobiError:
            pass

        try:
            if m.SolCount > 0:
                vals = [v.X for v in var_objs]
                sol, obj = build_solution(vals)
                if obj < best_obj or best_sol is heur:
                    if obj <= best_obj + 1e-6:
                        best_obj = min(best_obj, obj)
                        if obj <= best_obj + 1e-6:
                            best_sol = sol
                            best_obj = obj
        except gp.GurobiError:
            pass

    except gp.GurobiError:
        pass
    except Exception:
        pass

    if logger:
        try:
            logger.log_solution(best_obj, best_sol)
        except Exception:
            pass

    write_solution(args.solution_path, best_sol)


if __name__ == "__main__":
    main()