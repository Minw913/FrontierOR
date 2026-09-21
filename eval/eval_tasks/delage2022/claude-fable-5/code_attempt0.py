import argparse
import json
import math
import time
import sys


def build_sol_dict(ids, P, a, r, y, e, cs, cr):
    """Build solution dict from raw decision values.
    a: dict (product_id, pattern_idx) -> pieces
    r: list of repetitions per pattern
    y: list of bool used per pattern
    e: dict product_id -> extra pieces
    """
    used = sum(1 for j in range(P) if y[j])
    obj = cs * used + cr * sum(int(r[j]) for j in range(P))
    patterns = {}
    for j in range(P):
        patterns[str(j)] = {
            "used": bool(y[j]),
            "repetitions": int(r[j]),
            "pieces": {str(i): int(a.get((i, j), 0)) for i in ids},
        }
    extra = {str(i): int(e.get(i, 0)) for i in ids}
    pq = {}
    for i in ids:
        for j in range(P):
            pq["product_{}_pattern_{}".format(i, j)] = int(a.get((i, j), 0)) * int(r[j])
    sol = {
        "objective_value": float(obj),
        "patterns": patterns,
        "extra_pieces": extra,
        "production_quantities": pq,
    }
    return float(obj), sol


def greedy(ids, length, demand, N, L, P, M, mode):
    """Greedy heuristic. Returns (pats, extras) where pats is a list of
    ({product_id: pieces}, repetitions) and extras is {product_id: count}."""
    rem = dict(demand)
    pats = []
    while len(pats) < P and any(rem[i] > 0 for i in ids):
        cand = [i for i in ids if rem[i] > 0 and N[i] >= 1]
        if not cand:
            break
        if mode == "single":
            i = max(cand, key=lambda x: (math.ceil(rem[x] / max(1, N[x])), rem[x]))
            k = min(N[i], rem[i])
            pat = {i: k}
        else:
            cand.sort(key=lambda x: -length[x])
            cap = L
            pat = {}
            for i in cand:
                if length[i] <= cap:
                    pat[i] = 1
                    cap -= length[i]
            if not pat:
                break
            # Expand: repeatedly add a piece for the current bottleneck product
            while True:
                best = None
                bv = -1
                for i in pat:
                    if length[i] <= cap and pat[i] < rem[i]:
                        v = math.ceil(rem[i] / pat[i])
                        if v > bv:
                            bv = v
                            best = i
                if best is None:
                    break
                pat[best] += 1
                cap -= length[best]
        if not pat:
            break
        rj = max(math.ceil(rem[i] / pat[i]) for i in pat)
        rj = min(rj, M)
        if rj <= 0:
            rj = 1
        for i in pat:
            rem[i] = max(0, rem[i] - pat[i] * rj)
        pats.append((pat, rj))
    extras = {i: rem[i] for i in ids if rem[i] > 0}
    return pats, extras


def finalize(pats, extras, ids, P, M, cs, cr):
    """Turn (pats, extras) into full arrays; ensure sum extras <= sum reps.
    Returns (a, rlist, ylist, edict, obj) or None if cannot fit within P slots."""
    pats = [(dict(p), int(rr)) for p, rr in pats]
    E = sum(extras.values())
    totR = sum(rr for _, rr in pats)
    deficit = E - totR
    idx = 0
    while deficit > 0:
        if idx < len(pats):
            add = min(deficit, M - pats[idx][1])
            if add > 0:
                pats[idx] = (pats[idx][0], pats[idx][1] + add)
                deficit -= add
            idx += 1
        elif len(pats) < P:
            add = min(deficit, M)
            if add <= 0:
                return None
            pats.append(({}, add))
            deficit -= add
        else:
            return None
    pats.sort(key=lambda t: -t[1])
    if len(pats) > P:
        return None
    a = {}
    rlist = [0] * P
    ylist = [False] * P
    for j, (pat, rr) in enumerate(pats):
        rlist[j] = rr
        ylist[j] = True
        for i, cnt in pat.items():
            a[(i, j)] = cnt
    edict = dict(extras)
    used = sum(1 for v in ylist if v)
    obj = cs * used + cr * sum(rlist)
    return a, rlist, ylist, edict, obj


def repair(ids, length, demand, N, L, P, M, a, rlist, ylist, edict):
    """Fix small inconsistencies in an extracted solution (rounding etc.)."""
    for j in range(P):
        if rlist[j] > 0 and not ylist[j]:
            ylist[j] = True
        if ylist[j] and rlist[j] < 1:
            # unused effectively; drop it if no pieces would matter
            rlist[j] = max(rlist[j], 1)
    # coverage
    for i in ids:
        prod = sum(a.get((i, j), 0) * rlist[j] for j in range(P))
        short = demand[i] - prod - edict.get(i, 0)
        if short > 0:
            edict[i] = edict.get(i, 0) + short
    E = sum(edict.values())
    R = sum(rlist)
    deficit = E - R
    j = 0
    while deficit > 0 and j < P:
        cap = M - rlist[j]
        add = min(deficit, cap)
        if add > 0:
            rlist[j] += add
            ylist[j] = True
            deficit -= add
        j += 1
    # keep ordering
    order = sorted(range(P), key=lambda j: -rlist[j])
    new_a = {}
    new_r = [0] * P
    new_y = [False] * P
    for pos, j in enumerate(order):
        new_r[pos] = rlist[j]
        new_y[pos] = ylist[j]
        for i in ids:
            v = a.get((i, j), 0)
            if v:
                new_a[(i, pos)] = v
    return new_a, new_r, new_y, edict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    L = int(inst["machine_length"])
    P = int(inst["num_patterns"])
    cs = int(inst["cost_setup_pattern"])
    cr = int(inst["cost_repetition"])
    M = int(inst["M"])
    prods = inst["products"]
    ids = [p["id"] for p in prods]
    length = {p["id"]: int(p["length"]) for p in prods}
    demand = {p["id"]: int(p["demand"]) for p in prods}
    N = {p["id"]: int(p["N_i"]) for p in prods}

    # Trivial case: no demand
    if sum(demand.values()) == 0 or P == 0:
        obj, sol = build_sol_dict(ids, P, {}, [0] * P, [False] * P, {}, cs, cr)
        if logger:
            logger.log_solution(obj, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f, indent=2)
        return

    # ------- Heuristic starts -------
    best_start = None
    for mode in ("pack", "single"):
        pats, extras = greedy(ids, length, demand, N, L, P, M, mode)
        res = finalize(pats, extras, ids, P, M, cs, cr)
        if res is not None:
            if best_start is None or res[4] < best_start[4]:
                best_start = res

    state = {"best": float("inf"), "sol": None}
    if best_start is not None:
        a0, r0, y0, e0, obj0 = best_start
        objv, sol = build_sol_dict(ids, P, a0, r0, y0, e0, cs, cr)
        state["best"] = objv
        state["sol"] = sol
        if logger:
            logger.log_solution(objv, sol)

    # ------- MIP model -------
    try:
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model("open_end_cutting")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        remaining = args.time_limit - (time.time() - t0) - 1.5
        m.Params.TimeLimit = max(1.0, remaining)

        K = max(1, M.bit_length())

        y = {j: m.addVar(vtype=GRB.BINARY, name="y_%d" % j) for j in range(P)}
        r = {j: m.addVar(vtype=GRB.INTEGER, lb=0, ub=M, name="r_%d" % j) for j in range(P)}
        b = {(j, k): m.addVar(vtype=GRB.BINARY) for j in range(P) for k in range(K)}
        a = {(i, j): m.addVar(vtype=GRB.INTEGER, lb=0, ub=N[i]) for i in ids for j in range(P)}
        z = {(i, j, k): m.addVar(lb=0, ub=N[i]) for i in ids for j in range(P) for k in range(K)}
        e = {i: m.addVar(vtype=GRB.INTEGER, lb=0, ub=demand[i]) for i in ids}

        for j in range(P):
            m.addConstr(gp.quicksum(length[i] * a[i, j] for i in ids) <= L * y[j])
            m.addConstr(r[j] == gp.quicksum((1 << k) * b[j, k] for k in range(K)))
            m.addConstr(r[j] <= M * y[j])
            m.addConstr(r[j] >= y[j])
            if j + 1 < P:
                m.addConstr(r[j] >= r[j + 1])
            for i in ids:
                Ni = N[i]
                if Ni == 0:
                    continue
                for k in range(K):
                    m.addConstr(z[i, j, k] <= Ni * b[j, k])
                    m.addConstr(z[i, j, k] <= a[i, j])
                    m.addConstr(z[i, j, k] >= a[i, j] - Ni * (1 - b[j, k]))

        for i in ids:
            m.addConstr(
                gp.quicksum((1 << k) * z[i, j, k] for j in range(P) for k in range(K))
                + e[i] >= demand[i]
            )
        m.addConstr(gp.quicksum(e[i] for i in ids) <= gp.quicksum(r[j] for j in range(P)))

        # Valid lower-bound cuts on total repetitions
        totlen = sum(length[i] * demand[i] for i in ids)
        maxlen = max(length.values())
        Rlb = int(math.ceil(totlen / float(L + maxlen)))
        for i in ids:
            Rlb = max(Rlb, int(math.ceil(demand[i] / float(N[i] + 1))))
        m.addConstr(gp.quicksum(r[j] for j in range(P)) >= Rlb)

        m.setObjective(
            cs * gp.quicksum(y[j] for j in range(P))
            + cr * gp.quicksum(r[j] for j in range(P)),
            GRB.MINIMIZE,
        )

        # MIP start
        if best_start is not None:
            a0, r0, y0, e0, obj0 = best_start
            for j in range(P):
                y[j].Start = 1 if y0[j] else 0
                r[j].Start = r0[j]
                for k in range(K):
                    b[j, k].Start = (r0[j] >> k) & 1
            for i in ids:
                e[i].Start = e0.get(i, 0)
                for j in range(P):
                    a[i, j].Start = a0.get((i, j), 0)

        a_keys = [(i, j) for i in ids for j in range(P)]
        a_list = [a[key] for key in a_keys]
        r_list = [r[j] for j in range(P)]
        y_list = [y[j] for j in range(P)]
        e_list = [e[i] for i in ids]

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj < state["best"] - 1e-4:
                    try:
                        av = model.cbGetSolution(a_list)
                        rv = model.cbGetSolution(r_list)
                        yv = model.cbGetSolution(y_list)
                        ev = model.cbGetSolution(e_list)
                    except Exception:
                        return
                    adict = {a_keys[t]: int(round(av[t])) for t in range(len(a_keys))}
                    rl = [int(round(v)) for v in rv]
                    yl = [v > 0.5 for v in yv]
                    edict = {ids[t]: int(round(ev[t])) for t in range(len(ids))}
                    objv, sol = build_sol_dict(ids, P, adict, rl, yl, edict, cs, cr)
                    if objv < state["best"] - 1e-9:
                        state["best"] = objv
                        state["sol"] = sol
                        if logger:
                            logger.log_solution(objv, sol)

        m.optimize(cb)

        if m.SolCount > 0:
            adict = {key: int(round(a[key].X)) for key in a_keys}
            rl = [int(round(r[j].X)) for j in range(P)]
            yl = [y[j].X > 0.5 for j in range(P)]
            edict = {i: int(round(e[i].X)) for i in ids}
            # verify / repair rounding issues
            ok = True
            for i in ids:
                prod = sum(adict.get((i, j), 0) * rl[j] for j in range(P))
                if prod + edict.get(i, 0) < demand[i]:
                    ok = False
                    break
            if sum(edict.values()) > sum(rl):
                ok = False
            if not ok:
                adict, rl, yl, edict = repair(
                    ids, length, demand, N, L, P, M, adict, rl, yl, edict
                )
            objv, sol = build_sol_dict(ids, P, adict, rl, yl, edict, cs, cr)
            if objv < state["best"] - 1e-9 or state["sol"] is None:
                state["best"] = objv
                state["sol"] = sol
                if logger:
                    logger.log_solution(objv, sol)
    except Exception as ex:
        sys.stderr.write("MIP stage failed: %s\n" % str(ex))

    if state["sol"] is None:
        # Last resort fallback: empty solution (should not normally happen)
        _, sol = build_sol_dict(ids, P, {}, [0] * P, [False] * P, dict(demand), cs, cr)
        state["sol"] = sol

    with open(args.solution_path, "w") as f:
        json.dump(state["sol"], f, indent=2)


if __name__ == "__main__":
    main()