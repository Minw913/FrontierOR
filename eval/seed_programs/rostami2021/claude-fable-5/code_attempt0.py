import argparse
import json
import math
import random
import time

import numpy as np

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + max(5.0, float(args.time_limit)) - 1.5

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = int(inst["n"])
    S = int(inst["num_scenarios"])
    cparams = inst["cost_parameters"]
    chi = float(cparams["chi"])
    alpha = float(cparams["alpha"])
    delta = float(cparams["delta"])
    p_hubs = inst.get("p_hubs", None)
    is_p = p_hubs is not None
    if is_p:
        p_hubs = max(1, min(int(p_hubs), n))
    D = np.asarray(inst["distances"], dtype=float)
    fixed = np.asarray(inst["fixed_costs"], dtype=float)
    caps_raw = inst.get("hub_capacities", None)
    is_cap = caps_raw is not None
    cap = np.asarray(caps_raw, dtype=float) if is_cap else None

    probs = []
    Ws = []
    Os = []
    cs = []
    for sc in inst["scenarios"]:
        probs.append(float(sc["probability"]))
        W = np.asarray(sc["demands"], dtype=float)
        Ws.append(W)
        O = W.sum(axis=1)
        Din = W.sum(axis=0)
        Os.append(O)
        cs.append(chi * O + delta * Din)
    maxTot = max((float(O.sum()) for O in Os), default=0.0)

    rng = random.Random(0)
    cache = {}
    best = {"obj": math.inf, "H": None, "allocs": None}

    # ------------------------------------------------------------------
    def make_sol(obj, H, allocs):
        Hset = set(int(k) for k in H)
        zv = [1.0 if k in Hset else 0.0 for k in range(n)]
        allocations = {}
        x_values = {}
        for s in range(S):
            a = allocs[s]
            allocations[str(s)] = {str(i): int(a[i]) for i in range(n)}
            xs = {}
            for i in range(n):
                ai = int(a[i])
                dd = {}
                for k in range(n):
                    if k == i:
                        continue
                    dd[str(k)] = 1.0 if (k == ai and ai != i) else 0.0
                xs[str(i)] = dd
            x_values[str(s)] = xs
        return {
            "objective_value": float(obj),
            "hubs": sorted(Hset),
            "allocations": allocations,
            "z_values": zv,
            "x_values": x_values,
        }

    def true_cost(H, allocs):
        tot = 0.0 if is_p else float(fixed[sorted(set(int(k) for k in H))].sum())
        idx = np.arange(n)
        for s in range(S):
            a = np.asarray(allocs[s], dtype=int)
            cd = float(np.dot(cs[s], D[idx, a]))
            tr = alpha * float(np.sum(Ws[s] * D[np.ix_(a, a)]))
            tot += probs[s] * (cd + tr)
        return tot

    def maybe_register(H, allocs):
        obj = true_cost(H, allocs)
        if obj < best["obj"] - 1e-9:
            best["obj"] = obj
            best["H"] = sorted(set(int(k) for k in H))
            best["allocs"] = [np.asarray(a, dtype=int).copy() for a in allocs]
            if logger:
                try:
                    logger.log_solution(obj, make_sol(obj, best["H"], best["allocs"]))
                except Exception:
                    pass
        return obj

    # ------------------------------------------------------------------
    def solve_assignment(H, W, O, c, passes=4, pos_init=None):
        h = len(H)
        Harr = np.asarray(H, dtype=int)
        Dih = D[:, Harr]
        Dhh = D[np.ix_(Harr, Harr)]
        hub_pos = {int(k): idx for idx, k in enumerate(H)}
        hubmask = np.zeros(n, dtype=bool)
        hubmask[Harr] = True
        loads = None
        if is_cap:
            capH = cap[Harr]
            if np.any(O[Harr] > capH + 1e-9):
                return None
            if pos_init is not None:
                pos = np.asarray(pos_init, dtype=np.int64).copy()
                for k, idx in hub_pos.items():
                    pos[k] = idx
                loads = np.zeros(h)
                np.add.at(loads, pos, O)
                if np.any(loads > capH + 1e-9):
                    pos_init = None
            if pos_init is None:
                pos = np.full(n, -1, dtype=np.int64)
                loads = np.zeros(h)
                for k, idx in hub_pos.items():
                    pos[k] = idx
                    loads[idx] += O[k]
                order = sorted((i for i in range(n) if not hubmask[i]), key=lambda i: -O[i])
                for i in order:
                    placed = False
                    for idx in np.argsort(Dih[i]):
                        if loads[idx] + O[i] <= capH[idx] + 1e-9:
                            pos[i] = idx
                            loads[idx] += O[i]
                            placed = True
                            break
                    if not placed:
                        return None
        else:
            if pos_init is not None:
                pos = np.asarray(pos_init, dtype=np.int64).copy()
            else:
                pos = np.argmin(Dih, axis=1).astype(np.int64)
            for k, idx in hub_pos.items():
                pos[k] = idx

        Aind = np.zeros((n, h))
        Aind[np.arange(n), pos] = 1.0
        AggOut = W @ Aind
        AggIn = W.T @ Aind
        T = alpha * (AggOut @ Dhh.T + AggIn @ Dhh)
        nonhubs = [i for i in range(n) if not hubmask[i]]
        for _ in range(passes):
            improved = False
            for i in nonhubs:
                k = int(pos[i])
                row = c[i] * Dih[i] + T[i]
                if is_cap:
                    feas = loads + O[i] <= cap[Harr] + 1e-9
                    feas[k] = True
                    r = np.where(feas, row, np.inf)
                else:
                    r = row
                k2 = int(np.argmin(r))
                if k2 != k and r[k2] < row[k] - 1e-9:
                    if is_cap:
                        loads[k] -= O[i]
                        loads[k2] += O[i]
                    pos[i] = k2
                    wc = W[:, i]
                    wr = W[i, :]
                    T += alpha * (np.outer(wc, Dhh[:, k2] - Dhh[:, k]) +
                                  np.outer(wr, Dhh[k2, :] - Dhh[k, :]))
                    improved = True
            if not improved:
                break
        cd = float(np.dot(c, Dih[np.arange(n), pos]))
        tr = 0.5 * float(np.sum(T[np.arange(n), pos]))
        return cd + tr, pos

    def evaluate(Ht, passes=4):
        Ht = tuple(sorted(int(k) for k in Ht))
        res = cache.get(Ht)
        if res is not None:
            return res
        if len(Ht) == 0:
            res = (math.inf, None)
            cache[Ht] = res
            return res
        if is_cap and float(cap[list(Ht)].sum()) < maxTot - 1e-9:
            res = (math.inf, None)
            cache[Ht] = res
            return res
        H = list(Ht)
        tot = 0.0 if is_p else float(fixed[list(Ht)].sum())
        allocs = []
        ok = True
        Hnp = np.asarray(H, dtype=int)
        for s in range(S):
            r = solve_assignment(H, Ws[s], Os[s], cs[s], passes=passes)
            if r is None:
                ok = False
                break
            cost, pos = r
            tot += probs[s] * cost
            allocs.append(Hnp[pos])
        res = (tot, allocs) if ok else (math.inf, None)
        cache[Ht] = res
        return res

    # ------------------------------------------------------------------
    def greedy(dl):
        H = []
        cur = math.inf
        while True:
            if H and time.time() > dl:
                break
            bestk = None
            bestc = math.inf if is_p else cur
            for k in range(n):
                if k in H:
                    continue
                c, _ = evaluate(tuple(sorted(H + [k])), passes=3)
                if c < bestc - 1e-9:
                    bestc = c
                    bestk = k
                if time.time() > dl and bestk is not None:
                    break
            if bestk is None:
                cands = [k for k in range(n) if k not in H]
                if not cands:
                    break
                if is_p or math.isinf(cur):
                    if is_cap:
                        forced = max(cands, key=lambda k: cap[k])
                    else:
                        forced = min(cands, key=lambda k: fixed[k])
                    H.append(forced)
                    if is_p and len(H) >= p_hubs:
                        break
                    if is_p or math.isinf(cur):
                        continue
                break
            H.append(bestk)
            cur = bestc
            if is_p and len(H) >= p_hubs:
                break
            if len(H) >= n:
                break
        if is_p:
            while len(H) < p_hubs:
                cands = [k for k in range(n) if k not in H]
                H.append(cands[0])
        if not H:
            H = [0]
        return sorted(H)

    def local_search(H, cur, dl):
        H = sorted(H)
        improved = True
        while improved and time.time() < dl:
            improved = False
            Hset = set(H)
            outs = [k for k in range(n) if k not in Hset]
            moves = []
            if is_p:
                for a in H:
                    for b in outs:
                        moves.append(("swap", a, b))
            else:
                for b in outs:
                    moves.append(("add", -1, b))
                if len(H) > 1:
                    for a in H:
                        moves.append(("drop", a, -1))
                for a in H:
                    for b in outs:
                        moves.append(("swap", a, b))
            rng.shuffle(moves)
            for (typ, a, b) in moves:
                if time.time() > dl:
                    break
                if typ == "add":
                    newH = sorted(H + [b])
                elif typ == "drop":
                    newH = sorted([k for k in H if k != a])
                else:
                    newH = sorted([k for k in H if k != a] + [b])
                c, al = evaluate(tuple(newH))
                if c < cur - 1e-6:
                    H = newH
                    cur = c
                    improved = True
                    if al is not None:
                        maybe_register(H, al)
                    break
        return H, cur

    def perturb(baseH):
        H = sorted(set(int(k) for k in baseH))
        Hset = set(H)
        outs = [k for k in range(n) if k not in Hset]
        if is_p:
            # keep size p using random swaps
            while len(H) < p_hubs and outs:
                b = rng.choice(outs)
                outs.remove(b)
                H.append(b)
            while len(H) > p_hubs:
                H.remove(rng.choice(H))
            outs = [k for k in range(n) if k not in set(H)]
            nswap = min(2, len(H), len(outs))
            for _ in range(nswap):
                a = rng.choice(H)
                b = rng.choice(outs)
                H.remove(a)
                outs.remove(b)
                H.append(b)
                outs.append(a)
        else:
            for _ in range(2):
                Hset = set(H)
                outs = [k for k in range(n) if k not in Hset]
                r = rng.random()
                if (r < 0.4 or len(H) <= 1) and outs:
                    H.append(rng.choice(outs))
                elif r < 0.7 and len(H) > 1:
                    H.remove(rng.choice(H))
                elif outs and H:
                    a = rng.choice(H)
                    b = rng.choice(outs)
                    H.remove(a)
                    H.append(b)
        H = sorted(set(H))
        if not H:
            H = [0]
        return H

    def heuristic_phase(dl, do_greedy=True):
        Hcur = None
        if do_greedy:
            H0 = greedy(dl)
            c0, al0 = evaluate(tuple(H0))
            if al0 is not None:
                maybe_register(H0, al0)
            H0, c0 = local_search(H0, c0, dl)
            Hcur = H0
        while time.time() < dl - 0.05:
            if best["H"] is not None:
                baseH = best["H"]
            elif Hcur:
                baseH = Hcur
            else:
                baseH = [0]
            Hp = perturb(baseH)
            c1, al1 = evaluate(tuple(Hp))
            if al1 is not None:
                maybe_register(sorted(Hp), al1)
            Hp, c1 = local_search(Hp, c1, dl)
            Hcur = Hp

    def polish():
        if best["H"] is None:
            return
        H = list(best["H"])
        posmap = {int(k): idx for idx, k in enumerate(H)}
        Hnp = np.asarray(H, dtype=int)
        allocs = []
        ok = True
        for s in range(S):
            try:
                pinit = np.array([posmap[int(best["allocs"][s][i])] for i in range(n)],
                                 dtype=np.int64)
            except KeyError:
                ok = False
                break
            r = solve_assignment(H, Ws[s], Os[s], cs[s], passes=30, pos_init=pinit)
            if r is None:
                ok = False
                break
            allocs.append(Hnp[r[1]])
        if ok:
            maybe_register(H, allocs)

    # ------------------------------------------------------------------
    def run_mip():
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            return False
        try:
            m = gp.Model("shlp")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1

            z = m.addVars(n, vtype=GRB.BINARY, name="z")
            xidx = [(s, i, k) for s in range(S) for i in range(n)
                    for k in range(n) if k != i]
            x = m.addVars(xidx, vtype=GRB.BINARY, name="x")
            xvars = [x[t] for t in xidx]
            m.setAttr("Obj", xvars,
                      [probs[s] * cs[s][i] * D[i, k] for (s, i, k) in xidx])
            if not is_p:
                m.setAttr("Obj", [z[k] for k in range(n)],
                          [float(fixed[k]) for k in range(n)])

            for s in range(S):
                for i in range(n):
                    m.addLConstr(
                        gp.quicksum(x[s, i, k] for k in range(n) if k != i) + z[i] == 1)
            for (s, i, k) in xidx:
                m.addLConstr(x[s, i, k] - z[k] <= 0)
            if is_p:
                m.addLConstr(gp.quicksum(z[k] for k in range(n)) == p_hubs)
            if is_cap:
                for s in range(S):
                    O = Os[s]
                    for k in range(n):
                        m.addLConstr(
                            gp.quicksum(float(O[i]) * x[s, i, k]
                                        for i in range(n) if i != k and O[i] > 0)
                            - (float(cap[k]) - float(O[k])) * z[k] <= 0)

            blocks = {}
            tot = 0
            for s in range(S):
                O = Os[s]
                for i in range(n):
                    if O[i] > 1e-12:
                        blocks[(s, i)] = tot
                        tot += n * (n - 1)
            yv = m.addVars(tot, lb=0.0, name="y")
            base_rows = [np.delete(alpha * D[k], k) for k in range(n)]
            basevec = np.concatenate(base_rows) if n > 1 else np.zeros(0)
            yobj = np.zeros(tot)
            for (s, i), off in blocks.items():
                yobj[off:off + n * (n - 1)] = probs[s] * basevec
            if tot > 0:
                m.setAttr("Obj", [yv[t] for t in range(tot)], yobj.tolist())

            out_off = []
            in_off = []
            for k in range(n):
                oo = []
                io = []
                for l in range(n):
                    if l == k:
                        continue
                    oo.append(k * (n - 1) + (l if l < k else l - 1))
                    io.append(l * (n - 1) + (k if k < l else k - 1))
                out_off.append(oo)
                in_off.append(io)

            for (s, i), off in blocks.items():
                W = Ws[s]
                Oi = float(Os[s][i])
                Jpos = np.nonzero(W[i])[0]
                for k in range(n):
                    vars_ = [yv[off + t] for t in out_off[k]]
                    coefs = [1.0] * (n - 1)
                    vars_ += [yv[off + t] for t in in_off[k]]
                    coefs += [-1.0] * (n - 1)
                    vars_.append(z[k] if i == k else x[s, i, k])
                    coefs.append(-Oi)
                    for j in Jpos:
                        j = int(j)
                        vars_.append(z[k] if j == k else x[s, j, k])
                        coefs.append(float(W[i, j]))
                    m.addLConstr(gp.LinExpr(coefs, vars_), GRB.EQUAL, 0.0)

            if best["H"] is not None:
                Hset = set(best["H"])
                al = best["allocs"]
                m.setAttr("Start", [z[k] for k in range(n)],
                          [1.0 if k in Hset else 0.0 for k in range(n)])
                m.setAttr("Start", xvars,
                          [1.0 if int(al[s][i]) == k else 0.0 for (s, i, k) in xidx])

            tl = deadline - time.time() - 2.0
            if tl < 3:
                return False
            m.Params.TimeLimit = tl

            zvlist = [z[k] for k in range(n)]

            def extract_and_register(zv, xv):
                Hs = [k for k in range(n) if zv[k] > 0.5]
                if not Hs:
                    return
                allocs = []
                for s in range(S):
                    a = np.array([i if zv[i] > 0.5 else -1 for i in range(n)], dtype=int)
                    allocs.append(a)
                for t, (s, i, k) in enumerate(xidx):
                    if xv[t] > 0.5 and allocs[s][i] < 0:
                        allocs[s][i] = k
                Ha = np.asarray(Hs, dtype=int)
                for s in range(S):
                    miss = np.where(allocs[s] < 0)[0]
                    for i in miss:
                        allocs[s][i] = int(Ha[np.argmin(D[i, Ha])])
                maybe_register(Hs, allocs)

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        zv = model.cbGetSolution(zvlist)
                        xv = model.cbGetSolution(xvars)
                        extract_and_register(zv, xv)
                    except Exception:
                        pass

            m.optimize(cb)
            if m.SolCount > 0:
                try:
                    zv = m.getAttr("X", zvlist)
                    xv = m.getAttr("X", xvars)
                    extract_and_register(zv, xv)
                except Exception:
                    pass
            return m.Status == GRB.OPTIMAL
        except Exception:
            return False

    # ------------------------------------------------------------------
    if n == 1:
        allocs = [np.zeros(1, dtype=int) for _ in range(S)]
        maybe_register([0], allocs)
        if best["H"] is None:
            best["H"] = [0]
            best["allocs"] = allocs
            best["obj"] = true_cost([0], allocs)
        sol = make_sol(best["obj"], best["H"], best["allocs"])
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    tot_blocks = sum(int(np.count_nonzero(Os[s] > 1e-12)) for s in range(S))
    tot_y = tot_blocks * n * (n - 1)
    nx = S * n * (n - 1)
    mip_ok = (n >= 2) and tot_y <= 900_000 and nx <= 250_000
    build_est = tot_y / 25000.0 + nx / 8000.0 + 3.0
    plan_mip = mip_ok and (deadline - t0) > build_est + 15.0

    if plan_mip:
        h_deadline = min(deadline, time.time() + max(3.0, 0.12 * float(args.time_limit)))
    else:
        h_deadline = deadline

    heuristic_phase(h_deadline, do_greedy=True)

    mip_optimal = False
    if plan_mip and (deadline - time.time()) > 0.5 * build_est + 5.0:
        mip_optimal = run_mip()

    if not mip_optimal and time.time() < deadline - 1.0:
        heuristic_phase(deadline, do_greedy=False)

    polish()

    if best["H"] is None:
        H = list(range(n))
        allocs = [np.arange(n, dtype=int) for _ in range(S)]
        obj = true_cost(H, allocs)
        best["H"] = H
        best["allocs"] = allocs
        best["obj"] = obj
        if logger:
            try:
                logger.log_solution(obj, make_sol(obj, H, allocs))
            except Exception:
                pass

    sol = make_sol(best["obj"], best["H"], best["allocs"])
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()