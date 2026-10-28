#!/usr/bin/env python3
import argparse
import bisect
import json
import time

NEG = -(10 ** 18)


# ---------------------------------------------------------------------------
# Instance / context
# ---------------------------------------------------------------------------

def longest_paths(N, arcs, src):
    """Longest-path distances from src (Bellman-Ford style, max-relaxation)."""
    dist = [NEG] * N
    dist[src] = 0
    for _ in range(max(0, N - 1)):
        updated = False
        for a, b, w in arcs:
            da = dist[a]
            if da != NEG:
                v = da + w
                if dist[b] < v:
                    dist[b] = v
                    updated = True
        if not updated:
            break
    return dist


def build_context(data):
    n = int(data["num_real_activities"])
    N = int(data["num_nodes"])
    e = N - 1
    p = [int(v) for v in data["processing_times"]]
    D = int(data["deadline"])
    arcs = [
        (int(a["from"]), int(a["to"]), int(a["delta"]))
        for a in data.get("temporal_constraints", [])
    ]
    resources = data.get("resources", [])
    R = len(resources)
    P = [sorted(int(t) for t in res["time_periods"]) for res in resources]
    dem = [[int(v) for v in res["demands"]] for res in resources]
    cap = [int(res["capacity"]) for res in resources]

    dist0 = longest_paths(N, arcs, 0)
    ES = [dist0[i] if dist0[i] != NEG else 0 for i in range(N)]

    rearcs = [(b, a, w) for (a, b, w) in arcs]
    diste = longest_paths(N, rearcs, e)
    dto = [diste[i] if diste[i] != NEG else 0 for i in range(N)]
    LS = [D - dto[i] for i in range(N)]

    incoming = [[] for _ in range(N)]
    for a, b, w in arcs:
        incoming[b].append((a, w))

    return {
        "n": n, "N": N, "e": e, "p": p, "D": D, "arcs": arcs,
        "R": R, "P": P, "dem": dem, "cap": cap,
        "ES": ES, "LS": LS, "dto": dto, "incoming": incoming,
    }


def span_count(Pk, start, dur):
    # periods t with start < t <= start + dur
    return bisect.bisect_right(Pk, start + dur) - bisect.bisect_right(Pk, start)


def compute_loads(S, ctx):
    N, R, p = ctx["N"], ctx["R"], ctx["p"]
    loads = [0] * R
    for k in range(R):
        Pk, demk = ctx["P"][k], ctx["dem"][k]
        tot = 0
        for i in range(N):
            di, pi = demk[i], p[i]
            if di and pi:
                tot += di * span_count(Pk, S[i], pi)
        loads[k] = tot
    return loads


def check_feasible(S, ctx):
    if S is None:
        return False
    if S[0] != 0 or S[ctx["e"]] > ctx["D"]:
        return False
    for a, b, w in ctx["arcs"]:
        if S[b] < S[a] + w:
            return False
    loads = compute_loads(S, ctx)
    return all(loads[k] <= ctx["cap"][k] for k in range(ctx["R"]))


def propagate(S, ctx):
    """Enforce all lower-bound temporal constraints by only increasing times."""
    S = list(S)
    N, arcs = ctx["N"], ctx["arcs"]
    for _ in range(N):
        updated = False
        for a, b, w in arcs:
            v = S[a] + w
            if S[b] < v:
                S[b] = v
                updated = True
        if not updated:
            break
    if S[ctx["e"]] > ctx["D"]:
        return None
    return S


# ---------------------------------------------------------------------------
# Heuristic: right-shift repair + left-shift improvement
# ---------------------------------------------------------------------------

def total_violation(loads, ctx):
    return sum(max(0, loads[k] - ctx["cap"][k]) for k in range(ctx["R"]))


def repair_resources(S, ctx, end_ts):
    N, R, p, e = ctx["N"], ctx["R"], ctx["p"], ctx["e"]
    S = propagate(S, ctx)
    if S is None:
        return None
    for _ in range(400):
        if time.time() > end_ts:
            break
        loads = compute_loads(S, ctx)
        viol = [loads[k] - ctx["cap"][k] for k in range(R)]
        cur_tv = sum(max(0, v) for v in viol)
        if cur_tv <= 0:
            return S
        cands = []
        for k in range(R):
            if viol[k] <= 0:
                continue
            Pk, demk = ctx["P"][k], ctx["dem"][k]
            for i in range(N):
                ri, pi = demk[i], p[i]
                if ri == 0 or pi == 0:
                    continue
                cur = span_count(Pk, S[i], pi)
                if cur == 0:
                    continue
                cand_ts = set()
                lo, hi = S[i], S[i] + pi
                i0 = bisect.bisect_right(Pk, lo)
                i1 = bisect.bisect_right(Pk, hi)
                for idx in range(i0, i1):
                    cand_ts.add(Pk[idx])          # left edge passes a period
                for tau in Pk:
                    t = tau - pi                  # right edge passes a period
                    if S[i] < t <= ctx["LS"][i]:
                        cand_ts.add(t)
                for t in cand_ts:
                    if t <= S[i] or t > ctx["LS"][i]:
                        continue
                    red = ri * (cur - span_count(Pk, t, pi))
                    if red > 0:
                        cands.append((red, i, t))
        if not cands:
            break
        cands.sort(reverse=True)
        best = None
        for _, i, t in cands[:60]:
            S2 = list(S)
            S2[i] = t
            S2 = propagate(S2, ctx)
            if S2 is None:
                continue
            tv = total_violation(compute_loads(S2, ctx), ctx)
            key = (tv, S2[e])
            if best is None or key < best[0]:
                best = (key, S2)
        if best is None or best[0][0] >= cur_tv:
            break
        S = best[1]
    return S


def left_shift(S, ctx, end_ts):
    N, e = ctx["N"], ctx["e"]
    incoming, ES = ctx["incoming"], ctx["ES"]
    order = sorted(range(N), key=lambda i: ctx["dto"][i])
    for _ in range(6):
        if time.time() > end_ts:
            break
        changed = False
        for i in order:
            if i == 0 or i == e:
                continue
            lb = ES[i]
            for h, w in incoming[i]:
                lb = max(lb, S[h] + w)
            if lb < S[i]:
                old = S[i]
                S[i] = lb
                loads = compute_loads(S, ctx)
                if any(loads[k] > ctx["cap"][k] for k in range(ctx["R"])):
                    S[i] = old
                else:
                    changed = True
        lb_e = ES[e]
        for h, w in incoming[e]:
            lb_e = max(lb_e, S[h] + w)
        if lb_e < S[e]:
            S[e] = lb_e
            changed = True
        if not changed:
            break
    return S


def heuristic(ctx, end_ts):
    S = propagate(list(ctx["ES"]), ctx)
    if S is None:
        S = list(ctx["ES"])
    if not check_feasible(S, ctx):
        S2 = repair_resources(S, ctx, end_ts)
        if S2 is not None:
            S = S2
    if check_feasible(S, ctx):
        S = left_shift(S, ctx, end_ts)
        propagate_inplace_ok = check_feasible(S, ctx)
        if not propagate_inplace_ok:
            S = repair_resources(S, ctx, end_ts) or S
    return S


# ---------------------------------------------------------------------------
# Gurobi MIP
# ---------------------------------------------------------------------------

def solve_mip(ctx, warm, time_limit, logger, incumbent):
    import gurobipy as gp
    from gurobipy import GRB

    N, e, D = ctx["N"], ctx["e"], ctx["D"]
    ES, LS, p, arcs = ctx["ES"], ctx["LS"], ctx["p"], ctx["arcs"]

    dom = []
    keys = []
    for i in range(N):
        lo, hi = ES[i], LS[i]
        ts_i = list(range(lo, hi + 1)) if hi >= lo else []
        dom.append(ts_i)
        for t in ts_i:
            keys.append((i, t))

    model = gp.Model("prcpsp")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(0.0, float(time_limit))

    x = model.addVars(keys, vtype=GRB.BINARY, name="x")
    S = [model.addVar(lb=ES[i], ub=max(LS[i], ES[i]), name="s_%d" % i)
         for i in range(N)]

    model.setObjective(S[e], GRB.MINIMIZE)

    for i in range(N):
        model.addConstr(gp.quicksum(x[i, t] for t in dom[i]) == 1)
        model.addConstr(S[i] == gp.quicksum(t * x[i, t] for t in dom[i]))
    model.addConstr(S[0] == 0)
    for a, b, w in arcs:
        model.addConstr(S[b] >= S[a] + w)

    for k in range(ctx["R"]):
        Pk, demk, capk = ctx["P"][k], ctx["dem"][k], ctx["cap"][k]
        expr = gp.LinExpr()
        for i in range(N):
            ri, pi = demk[i], p[i]
            if ri == 0 or pi == 0:
                continue
            for t in dom[i]:
                c = span_count(Pk, t, pi)
                if c:
                    expr += ri * c * x[i, t]
        model.addConstr(expr <= capk)

    if warm is not None:
        for i in range(N):
            t = warm[i]
            if ES[i] <= t <= max(LS[i], ES[i]) and (i, t) in x:
                x[i, t].Start = 1.0
            S[i].Start = t

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            except Exception:
                return
            if incumbent["obj"] is None or obj < incumbent["obj"] - 1e-6:
                try:
                    starts = [int(round(m.cbGetSolution(S[i]))) for i in range(N)]
                except Exception:
                    return
                sol = {"objective_value": int(round(obj)), "start_times": starts}
                incumbent["obj"] = obj
                incumbent["sol"] = sol
                if logger:
                    logger.log_solution(sol["objective_value"], sol)

    model.optimize(callback)

    if model.SolCount > 0:
        obj = model.ObjVal
        if incumbent["obj"] is None or obj < incumbent["obj"] - 1e-6:
            starts = [int(round(S[i].X)) for i in range(N)]
            sol = {"objective_value": int(round(obj)), "start_times": starts}
            incumbent["obj"] = obj
            incumbent["sol"] = sol
            if logger:
                logger.log_solution(sol["objective_value"], sol)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_ts = time.time()
    end_ts = start_ts + max(0, args.time_limit)

    with open(args.instance_path, "r") as f:
        data = json.load(f)
    ctx = build_context(data)

    incumbent = {"obj": None, "sol": None}

    # --- heuristic ---
    S_heur = heuristic(ctx, end_ts)
    if S_heur is not None and check_feasible(S_heur, ctx):
        obj = S_heur[ctx["e"]]
        sol = {"objective_value": int(obj),
               "start_times": [int(v) for v in S_heur]}
        incumbent["obj"] = obj
        incumbent["sol"] = sol
        if logger:
            logger.log_solution(sol["objective_value"], sol)

    # --- MIP ---
    remaining = end_ts - time.time()
    if remaining > 0.2:
        warm = S_heur if (S_heur is not None and S_heur[ctx["e"]] <= ctx["D"]) else None
        try:
            solve_mip(ctx, warm, remaining, logger, incumbent)
        except Exception:
            pass

    # --- final selection ---
    if incumbent["sol"] is not None:
        final = incumbent["sol"]
    else:
        S = S_heur if S_heur is not None else list(ctx["ES"])
        final = {"objective_value": int(S[ctx["e"]]),
                 "start_times": [int(v) for v in S]}
        if logger:
            logger.log_solution(final["objective_value"], final)

    with open(args.solution_path, "w") as f:
        json.dump(final, f)


if __name__ == "__main__":
    main()