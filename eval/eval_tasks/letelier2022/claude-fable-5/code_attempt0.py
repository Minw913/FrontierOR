import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_L(raw):
    s = str(raw).strip().lower()
    if s in ("infinity", "inf", "none", ""):
        return None
    return int(float(s))


def ffd(item_ids, wt, W):
    """First-fit decreasing packing. Returns list of bins (lists of item ids)."""
    order = sorted(item_ids, key=lambda i: (-wt[i], i))
    bins, loads = [], []
    for i in order:
        placed = False
        for b in range(len(bins)):
            if loads[b] + wt[i] <= W:
                bins[b].append(i)
                loads[b] += wt[i]
                placed = True
                break
        if not placed:
            bins.append([i])
            loads.append(wt[i])
    return bins


def build_solution(assign, wt):
    """assign: item_id -> (period, bin). Returns full solution dict."""
    groups = defaultdict(list)
    for i, (t, b) in assign.items():
        groups[(t, b)].append(i)
    assignments = [
        {"item_id": int(i), "bin": int(b), "period": int(t), "weight": int(wt[i])}
        for i, (t, b) in sorted(assign.items())
    ]
    abp = []
    for (t, b), ids in sorted(groups.items()):
        abp.append(
            {
                "bin": int(b),
                "period": int(t),
                "items": sorted(int(x) for x in ids),
                "total_weight": int(sum(wt[i] for i in ids)),
            }
        )
    return {
        "objective_value": float(len(groups)),
        "assignments": assignments,
        "active_bin_periods": abp,
        "num_active_bin_periods": int(len(groups)),
    }


def main():
    start_time = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", type=str, required=True)
    ap.add_argument("--solution_path", type=str, required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", type=str, default=None)
    args = ap.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    params = data["parameters"]
    T = int(params["T"])
    W = int(params["W"])
    L = parse_L(params.get("L", "infinity"))

    source = data["source_item_id"]
    sink = data["sink_item_id"]
    items = data["items"]
    real_items = [it for it in items if it.get("type") == "treatment"]
    wt = {it["item_id"]: int(it["weight"]) for it in real_items}
    real_ids = sorted(wt.keys())
    arcs = [(a["from"], a["to"], int(a["lag"])) for a in data["arcs"]]

    NEG, POS = -(10 ** 9), 10 ** 9
    nnodes = len(items)

    # ---------- Earliest times (longest path from source, Bellman-Ford) ----------
    E = {it["item_id"]: NEG for it in items}
    E[source] = 0
    E[sink] = T + 1
    for _ in range(nnodes + 3):
        changed = False
        for u, v, lag in arcs:
            if v == source or v == sink:
                continue
            eu = E.get(u, NEG)
            if eu > NEG // 2 and eu + lag > E[v]:
                E[v] = eu + lag
                changed = True
        if not changed:
            break

    # ---------- Latest times (backward from sink) ----------
    Ls = {it["item_id"]: POS for it in items}
    Ls[source] = 0
    Ls[sink] = T + 1
    for _ in range(nnodes + 3):
        changed = False
        for u, v, lag in arcs:
            if u == source or u == sink:
                continue
            lv = Ls.get(v, POS)
            if lv < POS // 2 and lv - lag < Ls[u]:
                Ls[u] = lv - lag
                changed = True
        if not changed:
            break

    lo, hi = {}, {}
    for i in real_ids:
        e = E[i] if E[i] > NEG // 2 else 1
        l = Ls[i] if Ls[i] < POS // 2 else T
        lo[i] = max(1, e)
        hi[i] = min(T, l)
        if hi[i] < lo[i]:
            hi[i] = lo[i]  # degenerate safeguard

    real_arcs = [(u, v, lag) for (u, v, lag) in arcs
                 if u in wt and v in wt]

    # ---------- Candidate items per period and per-period bin caps ----------
    Ct = {t: [] for t in range(1, T + 1)}
    for i in real_ids:
        for t in range(lo[i], hi[i] + 1):
            Ct[t].append(i)
    for t in Ct:
        Ct[t].sort()

    Bmax = {}
    for t in range(1, T + 1):
        n = len(Ct[t])
        if n == 0:
            Bmax[t] = 0
            continue
        wsum = sum(wt[i] for i in Ct[t])
        cap = (2 * wsum) // W + 1  # valid ub on bins needed for any subset of Ct
        b = min(n, cap)
        if L is not None:
            b = min(b, L)
        Bmax[t] = max(b, 1)

    # ---------- Greedy warm start: earliest schedule + FFD packing ----------
    greedy_assign = None
    greedy_period_bins = None  # t -> list of bins (lists of item ids)
    per_t = defaultdict(list)
    for i in real_ids:
        per_t[lo[i]].append(i)
    ok = True
    period_bins = {}
    for t, ids in per_t.items():
        bins = ffd(ids, wt, W)
        if L is not None and len(bins) > L:
            ok = False
        period_bins[t] = bins
    if ok:
        greedy_assign = {}
        greedy_period_bins = period_bins
        for t, bins in period_bins.items():
            for bi, blist in enumerate(bins, 1):
                for i in blist:
                    greedy_assign[i] = (t, bi)

    best_sol = None
    best_obj = float("inf")
    if greedy_assign is not None:
        best_sol = build_solution(greedy_assign, wt)
        best_obj = best_sol["objective_value"]
        if logger:
            logger.log_solution(best_obj, best_sol)

    total_x = sum(len(Ct[t]) * Bmax[t] for t in range(1, T + 1))
    use_explicit = total_x <= 400_000

    time_left = args.time_limit - (time.time() - start_time) - 2.0
    if time_left < 1.0:
        time_left = 1.0

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    try:
        if use_explicit:
            # =============== Explicit bin-assignment MIP ===============
            m = gp.Model("bpp_sched", env=env)
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = time_left

            x = {}
            y = {}
            rank = {}  # (i,t) -> rank of i within Ct[t]
            for t in range(1, T + 1):
                B = Bmax[t]
                if B == 0:
                    continue
                for b in range(1, B + 1):
                    y[t, b] = m.addVar(vtype=GRB.BINARY, name=f"y_{t}_{b}")
                for r, i in enumerate(Ct[t], 1):
                    rank[i, t] = r
                    for b in range(1, min(r, B) + 1):
                        x[i, t, b] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}_{b}")

            # each item assigned exactly once
            item_vars = defaultdict(list)
            for (i, t, b), v in x.items():
                item_vars[i].append((t, b, v))
            for i in real_ids:
                m.addConstr(gp.quicksum(v for _, _, v in item_vars[i]) == 1)

            # capacity per bin-period
            bp_vars = defaultdict(list)
            for (i, t, b), v in x.items():
                bp_vars[t, b].append((i, v))
            for (t, b), lst in bp_vars.items():
                m.addConstr(gp.quicksum(wt[i] * v for i, v in lst) <= W * y[t, b])
            # symmetry: bin usage ordering
            for t in range(1, T + 1):
                for b in range(1, Bmax[t]):
                    m.addConstr(y[t, b] >= y[t, b + 1])

            # precedence constraints between real items
            p_expr = {}
            for i in real_ids:
                p_expr[i] = gp.quicksum(t * v for t, _, v in item_vars[i])
            for u, v_, lag in real_arcs:
                m.addConstr(p_expr[v_] - p_expr[u] >= lag)

            # global lower-bound cut on number of bins
            totw = sum(wt.values())
            m.addConstr(gp.quicksum(y.values()) >= -(-totw // W))

            m.setObjective(gp.quicksum(y.values()), GRB.MINIMIZE)

            # warm start
            if greedy_assign is not None:
                for v in x.values():
                    v.Start = 0.0
                for v in y.values():
                    v.Start = 0.0
                start_ok = True
                starts = []
                for t, bins in greedy_period_bins.items():
                    if len(bins) > Bmax.get(t, 0):
                        start_ok = False
                        break
                    # order bins by min rank so item rank >= bin index
                    ordered = sorted(bins, key=lambda bl: min(rank[i, t] for i in bl))
                    for bi, blist in enumerate(ordered, 1):
                        for i in blist:
                            if (i, t, bi) not in x:
                                start_ok = False
                                break
                            starts.append(((i, t, bi), (t, bi)))
                        if not start_ok:
                            break
                    if not start_ok:
                        break
                if start_ok:
                    for key, (t, bi) in starts:
                        x[key].Start = 1.0
                        y[t, bi].Start = 1.0

            xkeys = list(x.keys())
            xvars = [x[k] for k in xkeys]
            state = {"best": best_obj}

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj < state["best"] - 1e-6:
                        vals = model.cbGetSolution(xvars)
                        assign = {}
                        for k, val in zip(xkeys, vals):
                            if val > 0.5:
                                i, t, b = k
                                assign[i] = (t, b)
                        if len(assign) == len(real_ids):
                            sol = build_solution(assign, wt)
                            state["best"] = sol["objective_value"]
                            state["sol"] = sol
                            if logger:
                                logger.log_solution(sol["objective_value"], sol)

            m.optimize(cb)

            if m.SolCount > 0:
                assign = {}
                for k in xkeys:
                    if x[k].X > 0.5:
                        i, t, b = k
                        assign[i] = (t, b)
                if len(assign) == len(real_ids):
                    sol = build_solution(assign, wt)
                    if sol["objective_value"] < best_obj - 1e-6:
                        best_sol = sol
                        best_obj = sol["objective_value"]
                        if logger:
                            logger.log_solution(best_obj, best_sol)
                    elif best_sol is None:
                        best_sol = sol
                        best_obj = sol["objective_value"]
                        if logger:
                            logger.log_solution(best_obj, best_sol)
            if "sol" in state and state["sol"]["objective_value"] < best_obj - 1e-6:
                best_sol = state["sol"]
                best_obj = state["sol"]["objective_value"]

        else:
            # =============== Aggregate MIP (period assignment + bin count) ===============
            m = gp.Model("bpp_agg", env=env)
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = time_left

            z = {}
            for i in real_ids:
                for t in range(lo[i], hi[i] + 1):
                    z[i, t] = m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{t}")
            n = {}
            for t in range(1, T + 1):
                n[t] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=Bmax[t], name=f"n_{t}")

            for i in real_ids:
                m.addConstr(gp.quicksum(z[i, t] for t in range(lo[i], hi[i] + 1)) == 1)
            for t in range(1, T + 1):
                lst = [(i, z[i, t]) for i in Ct[t]]
                if lst:
                    m.addConstr(gp.quicksum(wt[i] * v for i, v in lst) <= W * n[t])
                    big = [v for i, v in lst if 2 * wt[i] > W]
                    if big:
                        m.addConstr(gp.quicksum(big) <= n[t])

            p_expr = {i: gp.quicksum(t * z[i, t] for t in range(lo[i], hi[i] + 1))
                      for i in real_ids}
            for u, v_, lag in real_arcs:
                m.addConstr(p_expr[v_] - p_expr[u] >= lag)

            totw = sum(wt.values())
            m.addConstr(gp.quicksum(n.values()) >= -(-totw // W))
            m.setObjective(gp.quicksum(n.values()), GRB.MINIMIZE)

            if greedy_assign is not None:
                for v in z.values():
                    v.Start = 0.0
                for i in real_ids:
                    t = greedy_assign[i][0]
                    if (i, t) in z:
                        z[i, t].Start = 1.0

            zkeys = list(z.keys())
            zvars = [z[k] for k in zkeys]
            state = {"best": best_obj}

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    vals = model.cbGetSolution(zvars)
                    periods = {}
                    for k, val in zip(zkeys, vals):
                        if val > 0.5:
                            periods[k[0]] = k[1]
                    if len(periods) != len(real_ids):
                        return
                    per = defaultdict(list)
                    for i, t in periods.items():
                        per[t].append(i)
                    assign = {}
                    feas = True
                    for t, ids in per.items():
                        bins = ffd(ids, wt, W)
                        if L is not None and len(bins) > L:
                            feas = False
                            break
                        for bi, blist in enumerate(bins, 1):
                            for i in blist:
                                assign[i] = (t, bi)
                    if feas:
                        sol = build_solution(assign, wt)
                        if sol["objective_value"] < state["best"] - 1e-6:
                            state["best"] = sol["objective_value"]
                            state["sol"] = sol
                            if logger:
                                logger.log_solution(sol["objective_value"], sol)

            m.optimize(cb)

            if "sol" in state and state["sol"]["objective_value"] < best_obj - 1e-6:
                best_sol = state["sol"]
                best_obj = state["sol"]["objective_value"]
            elif m.SolCount > 0 and best_sol is None:
                periods = {}
                for k in zkeys:
                    if z[k].X > 0.5:
                        periods[k[0]] = k[1]
                per = defaultdict(list)
                for i, t in periods.items():
                    per[t].append(i)
                assign = {}
                for t, ids in per.items():
                    bins = ffd(ids, wt, W)
                    for bi, blist in enumerate(bins, 1):
                        for i in blist:
                            assign[i] = (t, bi)
                best_sol = build_solution(assign, wt)
                best_obj = best_sol["objective_value"]
                if logger:
                    logger.log_solution(best_obj, best_sol)
    except gp.GurobiError:
        pass

    # ---------- Fallback if nothing feasible found ----------
    if best_sol is None:
        # best-effort output: earliest schedule + FFD (may violate finite L)
        assign = {}
        for t, ids in per_t.items():
            bins = ffd(ids, wt, W)
            for bi, blist in enumerate(bins, 1):
                for i in blist:
                    assign[i] = (t, bi)
        best_sol = build_solution(assign, wt)
        if logger:
            logger.log_solution(best_sol["objective_value"], best_sol)

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f, indent=2)


if __name__ == "__main__":
    main()