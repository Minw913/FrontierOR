import argparse
import json
import math
import random
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(2, args.time_limit) - 1.5

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    with open(args.instance_path) as f:
        inst = json.load(f)

    n = int(inst["num_nodes"])
    prm = inst["parameters"]
    tau = float(prm["tau"])
    chi = float(prm["chi"])
    delta = float(prm["delta"])
    D = np.asarray(inst["distances"], dtype=np.float64)
    fc = np.zeros(n, dtype=np.float64)
    for nd in inst["nodes"]:
        fc[int(nd["id"])] = float(nd["fixed_cost"])

    comms = inst.get("commodities", [])
    K = len(comms)

    if K == 0:
        sol = {"objective_value": 0.0, "open_hubs": [], "assignments": []}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    O_all = np.array([int(c["origin"]) for c in comms], dtype=np.int64)
    T_all = np.array([int(c["destination"]) for c in comms], dtype=np.int64)
    F_all = np.array([float(c["flow"]) for c in comms], dtype=np.float64)

    loop_mask = (O_all == T_all)
    idx_l = np.where(loop_mask)[0]
    idx_n = np.where(~loop_mask)[0]
    On, Dn, Fn = O_all[idx_n], T_all[idx_n], F_all[idx_n]
    Ol, Fl = O_all[idx_l], F_all[idx_l]
    Kn = int(idx_n.size)
    Kl = int(idx_l.size)

    rng = random.Random(0)

    # ------------------------------------------------------------------
    # Fast evaluation of a hub set S:
    # cost(o,d) = min_{i,j in S} chi*d(o,i) + tau*d(i,j) + delta*d(j,d)
    # decomposed via g(i,d) = min_j tau*d(i,j) + delta*d(j,d)
    # ------------------------------------------------------------------
    def evaluate(S, need_assign=False):
        S = np.asarray(S, dtype=np.int64)
        h = int(S.size)
        if h == 0:
            return math.inf, None
        total = float(fc[S].sum())
        det = {} if need_assign else None
        if Kn > 0:
            DS = D[S]                       # h x n
            Msub = tau * DS[:, S]           # h x h
            dDS = delta * DS                # h x n
            G_min = np.empty((h, n))
            G_arg = np.empty((h, n), dtype=np.int64) if need_assign else None
            cch = max(1, int(2.5e7 // max(1, h * h)))
            for s in range(0, n, cch):
                e = min(n, s + cch)
                block = Msub[:, :, None] + dDS[None, :, s:e]
                G_min[:, s:e] = block.min(axis=1)
                if need_assign:
                    G_arg[:, s:e] = block.argmin(axis=1)
            if need_assign:
                h1 = np.empty(Kn, dtype=np.int64)
                h2 = np.empty(Kn, dtype=np.int64)
                pu = np.empty(Kn)
            rows = max(1, int(6e6 // h))
            for s in range(0, Kn, rows):
                e = min(Kn, s + rows)
                Tm = chi * D[On[s:e, None], S] + G_min[:, Dn[s:e]].T
                pm = Tm.min(axis=1)
                total += float((pm * Fn[s:e]).sum())
                if need_assign:
                    ii = Tm.argmin(axis=1)
                    jj = G_arg[ii, Dn[s:e]]
                    h1[s:e] = S[ii]
                    h2[s:e] = S[jj]
                    pu[s:e] = pm
            if need_assign:
                det["h1"] = h1
                det["h2"] = h2
                det["cn"] = pu * Fn
        if Kl > 0:
            Dl = D[Ol[:, None], S]
            pm = (chi + delta) * Dl.min(axis=1)
            total += float((pm * Fl).sum())
            if need_assign:
                il = Dl.argmin(axis=1)
                det["hl"] = S[il]
                det["cl"] = pm * Fl
        return total, det

    def build_solution(S):
        S = sorted(int(v) for v in S)
        total, det = evaluate(S, need_assign=True)
        hub1 = np.empty(K, dtype=np.int64)
        hub2 = np.empty(K, dtype=np.int64)
        ck = np.empty(K)
        if Kn > 0:
            hub1[idx_n] = det["h1"]
            hub2[idx_n] = det["h2"]
            ck[idx_n] = det["cn"]
        if Kl > 0:
            hub1[idx_l] = det["hl"]
            hub2[idx_l] = det["hl"]
            ck[idx_l] = det["cl"]
        assignments = []
        for k in range(K):
            a, b = int(hub1[k]), int(hub2[k])
            edge = [a] if a == b else sorted((a, b))
            assignments.append({
                "commodity": k,
                "origin": int(O_all[k]),
                "destination": int(T_all[k]),
                "hub_edge": edge,
                "cost": float(ck[k]),
            })
        return {
            "objective_value": float(total),
            "open_hubs": [int(v) for v in S],
            "assignments": assignments,
        }

    best = {"cost": math.inf, "S": None}

    def try_update(S, cost=None):
        S = sorted(int(v) for v in S)
        if not S:
            return False
        if cost is None:
            cost, _ = evaluate(S)
        if cost < best["cost"] - 1e-9:
            best["cost"] = cost
            best["S"] = list(S)
            if logger:
                try:
                    sol = build_solution(S)
                    logger.log_solution(sol["objective_value"], sol)
                except Exception:
                    try:
                        logger.log(cost)
                    except Exception:
                        pass
            return True
        return False

    # ------------------------------------------------------------------
    # Greedy construction
    # ------------------------------------------------------------------
    def greedy(dl):
        S = []
        cur = math.inf
        while len(S) < n:
            bi, bc = None, cur
            in_set = set(S)
            for i in range(n):
                if i in in_set:
                    continue
                c, _ = evaluate(S + [i])
                if c < bc - 1e-9:
                    bc, bi = c, i
                if time.time() >= dl:
                    break
            if bi is None:
                break
            S.append(bi)
            cur = bc
            if time.time() >= dl:
                break
        if not S:
            S = [int(np.argmin(fc))]
            cur, _ = evaluate(S)
        return S, cur

    # ------------------------------------------------------------------
    # Local search: best-improvement add/drop, first-improvement swap
    # ------------------------------------------------------------------
    def local_search(S0, dl):
        S = set(int(v) for v in S0)
        cur, _ = evaluate(sorted(S))
        try_update(S, cur)
        while time.time() < dl:
            base = sorted(S)
            best_c, best_S = cur, None
            for i in range(n):
                if i in S:
                    continue
                c, _ = evaluate(base + [i])
                if c < best_c - 1e-9:
                    best_c, best_S = c, base + [i]
                if time.time() >= dl:
                    break
            if len(S) > 1 and time.time() < dl:
                for i in base:
                    c, _ = evaluate([x for x in base if x != i])
                    if c < best_c - 1e-9:
                        best_c, best_S = c, [x for x in base if x != i]
                    if time.time() >= dl:
                        break
            if best_S is not None:
                S = set(best_S)
                cur = best_c
                try_update(S, cur)
                continue
            # swap phase (first improvement, randomized order)
            found = False
            timeout = False
            outs = list(S)
            rng.shuffle(outs)
            ins = [i for i in range(n) if i not in S]
            rng.shuffle(ins)
            for o in outs:
                rest = [x for x in S if x != o]
                for i in ins:
                    c, _ = evaluate(rest + [i])
                    if c < cur - 1e-9:
                        S = set(rest + [i])
                        cur = c
                        try_update(S, cur)
                        found = True
                        break
                    if time.time() >= dl:
                        timeout = True
                        break
                if found or timeout:
                    break
            if not found:
                break
        return sorted(S), cur

    # ------------------------------------------------------------------
    # Decide whether an exact MIP phase is tractable
    # ------------------------------------------------------------------
    P = n * (n + 1) // 2
    nvars = Kn * P + Kl * n
    use_mip = (nvars <= 1_000_000) and (deadline - time.time() > 10)

    if use_mip:
        heur_dl = min(deadline, time.time() + max(3.0, 0.15 * args.time_limit))
    else:
        heur_dl = deadline

    # Heuristic phase
    S0, c0 = greedy(heur_dl)
    try_update(S0, c0)
    S_cur, c_cur = local_search(S0, heur_dl)

    while time.time() < heur_dl:
        base = list(best["S"]) if best["S"] else S_cur
        S = set(base)
        if len(S) > 1:
            for r in rng.sample(sorted(S), min(len(S), rng.randint(1, 2))):
                S.discard(r)
        cand = [i for i in range(n) if i not in S]
        if cand:
            for a in rng.sample(cand, min(len(cand), rng.randint(1, 2))):
                S.add(a)
        if not S:
            S = {rng.randrange(n)}
        local_search(sorted(S), heur_dl)

    # ------------------------------------------------------------------
    # Exact MIP phase (strong path-based formulation, unordered hub pairs)
    # ------------------------------------------------------------------
    if use_mip and time.time() < deadline - 5:
        try:
            import gurobipy as gp
            from gurobipy import GRB

            m = gp.Model("uhlp")
            m.Params.OutputFlag = 0
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.TimeLimit = max(1.0, deadline - time.time())

            z = m.addMVar(n, vtype=GRB.BINARY, obj=fc)

            I, J = np.triu_indices(n)
            dIJ = D[I, J]

            if Kn > 0:
                A_or = D[On]
                B_de = D[Dn]
                c1 = chi * A_or[:, I] + delta * B_de[:, J]
                c2 = chi * A_or[:, J] + delta * B_de[:, I]
                Cn = Fn[:, None] * (np.minimum(c1, c2) + tau * dIJ[None, :])
                x = m.addMVar((Kn, P), lb=0.0, ub=1.0, obj=Cn)
                m.addConstr(x.sum(axis=1) == 1)
                for i in range(n):
                    cols = np.where((I == i) | (J == i))[0]
                    m.addConstr(x[:, cols].sum(axis=1) <= z[i] + np.zeros(Kn))
            if Kl > 0:
                Cl = (chi + delta) * Fl[:, None] * D[Ol]
                xl = m.addMVar((Kl, n), lb=0.0, ub=1.0, obj=Cl)
                m.addConstr(xl.sum(axis=1) == 1)
                for i in range(n):
                    m.addConstr(xl[:, i] <= z[i] + np.zeros(Kl))

            m.ModelSense = GRB.MINIMIZE

            if best["S"]:
                zs = np.zeros(n)
                for i in best["S"]:
                    zs[i] = 1.0
                try:
                    z.Start = zs
                except Exception:
                    pass

            def cb(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        zv = np.asarray(model.cbGetSolution(z)).reshape(-1)
                        Ssol = np.where(zv > 0.5)[0]
                        if Ssol.size > 0:
                            try_update(list(Ssol))
                    except Exception:
                        pass

            m.optimize(cb)

            if m.SolCount > 0:
                try:
                    zv = np.asarray(z.X).reshape(-1)
                    Ssol = np.where(zv > 0.5)[0]
                    if Ssol.size > 0:
                        try_update(list(Ssol))
                except Exception:
                    pass
        except Exception:
            pass

    # Continue heuristic with any remaining time
    if time.time() < deadline - 1 and best["S"]:
        local_search(best["S"], deadline)

    if best["S"] is None:
        try_update([int(np.argmin(fc))])

    # Remove unused open hubs if that helps
    sol = build_solution(best["S"])
    used = set()
    for a in sol["assignments"]:
        for hh in a["hub_edge"]:
            used.add(int(hh))
    if used and set(sol["open_hubs"]) != used:
        try_update(sorted(used))
        sol = build_solution(best["S"])

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()