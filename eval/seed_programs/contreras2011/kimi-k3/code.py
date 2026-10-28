import argparse
import json
import math
import time

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB

    HAVE_GUROBI = True
except Exception:
    gp = None
    GRB = None
    HAVE_GUROBI = False

try:
    from scipy import sparse

    HAVE_SCIPY = True
except Exception:
    sparse = None
    HAVE_SCIPY = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger

        logger = SolutionLogger(args.log_path, sense="minimize")

    t0 = time.time()
    total = max(1, int(args.time_limit))
    hard_end = t0 + total - min(2.0, 0.05 * total + 0.2)

    def tleft():
        return hard_end - time.time()

    # -------------------- read instance --------------------
    with open(args.instance_path) as fh:
        inst = json.load(fh)

    n = int(inst["n"])
    cpars = inst["cost_parameters"]
    CHI = float(cpars["collection_cost_chi"])
    ALPHA = float(cpars["transfer_cost_alpha"])
    DELTA = float(cpars["distribution_cost_delta"])
    W = np.asarray(inst["flow_matrix"], dtype=float)
    d = np.asarray(inst["distance_matrix"], dtype=float)
    f = np.asarray(inst["setup_costs"], dtype=float)
    Cap = np.asarray(inst["capacities"], dtype=float)
    O = W.sum(axis=1).astype(float)      # total outgoing flow per node (incl. self)
    Dm = W.sum(axis=0).astype(float)     # total incoming flow per node
    Tot = float(O.sum())
    w_weight = CHI * O + DELTA * Dm

    # a node can be opened as a hub only if it can hold its own outgoing flow
    valid_open = O <= Cap + 1e-9
    if not bool(valid_open.any()):
        valid_open = np.ones(n, dtype=bool)

    arange_n = np.arange(n)

    def objective(assign):
        fixed = float(f[np.unique(assign)].sum())
        coll = CHI * float((O * d[arange_n, assign]).sum())
        distr = DELTA * float((Dm * d[assign, arange_n]).sum())
        trans = ALPHA * float((W * d[np.ix_(assign, assign)]).sum())
        return fixed + coll + distr + trans

    best = {"obj": np.inf, "assign": None}

    def record(assign):
        obj = objective(assign)
        if obj < best["obj"] - 1e-6:
            best["obj"] = obj
            best["assign"] = assign.copy()
            if logger is not None:
                sol = {
                    "objective_value": float(obj),
                    "hubs": [int(h) for h in np.unique(assign)],
                    "assignment": [int(a) for a in assign],
                }
                logger.log_solution(float(obj), sol)
            return True
        return False

    # -------------------- constructive heuristic --------------------
    def greedy_facility(min_capacity):
        selected = np.zeros(n, dtype=bool)
        sel = []
        cur_min = None
        cap = 0.0
        base = 0.0
        while True:
            need_cap = cap < min_capacity
            if cur_min is None:
                est = (w_weight[:, None] * d).sum(axis=0)
                gains = -est - f
            else:
                est = (w_weight[:, None] * np.minimum(cur_min[:, None], d)).sum(axis=0)
                gains = base - est - f
            gains[selected] = -np.inf
            gains[~valid_open] = -np.inf
            k = int(np.argmax(gains))
            if not np.isfinite(gains[k]):
                rem = np.where(~selected)[0]
                if rem.size == 0:
                    break
                k = int(rem[int(np.argmax(Cap[rem]))])
            elif (not need_cap) and gains[k] <= 0:
                break
            selected[k] = True
            sel.append(k)
            cap += float(Cap[k])
            cur_min = d[:, k] if cur_min is None else np.minimum(cur_min, d[:, k])
            base = float((w_weight * cur_min).sum())
        return sel

    def select_candidates(H, must, min_capacity):
        selected = np.zeros(n, dtype=bool)
        sel = []
        for k in must:
            k = int(k)
            if not selected[k] and valid_open[k]:
                selected[k] = True
                sel.append(k)
        cur_min = d[:, sel].min(axis=1) if sel else None
        cap = float(Cap[sel].sum()) if sel else 0.0
        base = float((w_weight * cur_min).sum()) if sel else 0.0
        while len(sel) < H or cap < min_capacity:
            if cur_min is None:
                est = (w_weight[:, None] * d).sum(axis=0)
                gains = -est - f
            else:
                est = (w_weight[:, None] * np.minimum(cur_min[:, None], d)).sum(axis=0)
                gains = base - est - f
            gains[selected] = -np.inf
            gains[~valid_open] = -np.inf
            k = int(np.argmax(gains))
            if not np.isfinite(gains[k]):
                rem = np.where(~selected)[0]
                if rem.size == 0:
                    break
                k = int(rem[int(np.argmax(Cap[rem]))])
            selected[k] = True
            sel.append(k)
            cap += float(Cap[k])
            cur_min = d[:, k] if cur_min is None else np.minimum(cur_min, d[:, k])
            base = float((w_weight * cur_min).sum())
        return sel

    def greedy_assignment(hubs):
        hubs = [int(k) for k in hubs if valid_open[k]]
        if not hubs:
            return None
        assign = np.full(n, -1, dtype=int)
        load = np.zeros(n)
        assigned = np.zeros(n, dtype=bool)
        for k in hubs:
            assign[k] = k
            load[k] = O[k]
            assigned[k] = True
        for i in np.argsort(-O):
            if assigned[i]:
                continue
            while True:
                harr = np.asarray(hubs, dtype=int)
                feas = load[harr] + O[i] <= Cap[harr] + 1e-9
                if not bool(feas.any()):
                    inset = np.zeros(n, dtype=bool)
                    inset[hubs] = True
                    cand = np.where(~inset & valid_open)[0]
                    if cand.size == 0:
                        return None
                    best_j = int(cand[int(np.argmin(f[cand]))])
                    hubs.append(best_j)
                    assign[best_j] = best_j
                    load[best_j] = O[best_j]
                    assigned[best_j] = True
                    if best_j == i:
                        break
                    continue
                col = CHI * O[i] * d[i, harr] + DELTA * Dm[i] * d[harr, i]
                aj = assign[assigned]
                if aj.size:
                    t = (d[np.ix_(harr, aj)] @ W[i, assigned]
                         + d[np.ix_(aj, harr)].T @ W[assigned, i])
                    col = col + ALPHA * t
                col = np.where(feas, col, np.inf)
                k = int(harr[int(np.argmin(col))])
                assign[i] = k
                load[k] += O[i]
                assigned[i] = True
                break
        return assign

    # -------------------- local search --------------------
    def local_search(assign, budget):
        start = time.time()
        load = np.bincount(assign, weights=O, minlength=n).astype(float)
        improved_any = False
        while time.time() - start <= budget:
            open_arr = np.unique(assign)
            dK = d[open_arr]  # rows: hub -> all nodes
            improved = False
            for i in range(n):
                a = int(assign[i])
                if a == i:
                    continue  # hubs stay self-assigned
                outW = np.bincount(assign, weights=W[i], minlength=n)
                inW = np.bincount(assign, weights=W[:, i], minlength=n)
                outW[a] -= W[i, i]   # self-flow transfer cost is always zero
                inW[a] -= W[i, i]
                t_b = dK @ outW + inW @ dK.T
                base = d[a] @ outW + inW @ d[:, a]
                dl = (CHI * O[i] * (d[i, open_arr] - d[i, a])
                      + DELTA * Dm[i] * (d[open_arr, i] - d[a, i])
                      + ALPHA * (t_b - base))
                feas = (load[open_arr] + O[i] <= Cap[open_arr] + 1e-9) & (open_arr != a)
                if not bool(feas.any()):
                    continue
                masked = np.where(feas, dl, np.inf)
                bi = int(np.argmin(masked))
                if masked[bi] < -1e-7:
                    b = int(open_arr[bi])
                    assign[i] = b
                    load[a] -= O[i]
                    load[b] += O[i]
                    improved = True
                    improved_any = True
            if not improved:
                break
        return improved_any

    def close_moves(assign, budget):
        start = time.time()
        improved_any = False
        for a in [int(x) for x in np.unique(assign)]:
            if time.time() - start > budget:
                break
            nodes = np.where(assign == a)[0]
            others = np.setdiff1d(np.unique(assign), [a])
            if others.size == 0:
                continue
            saved = assign.copy()
            load = np.bincount(assign, weights=O, minlength=n).astype(float)
            total_delta = -f[a]
            ok = True
            dK = d[others]
            for i in nodes[np.argsort(-O[nodes])]:
                outW = np.bincount(assign, weights=W[i], minlength=n)
                inW = np.bincount(assign, weights=W[:, i], minlength=n)
                outW[a] -= W[i, i]
                inW[a] -= W[i, i]
                t_b = dK @ outW + inW @ dK.T
                base = d[a] @ outW + inW @ d[:, a]
                dl = (CHI * O[i] * (d[i, others] - d[i, a])
                      + DELTA * Dm[i] * (d[others, i] - d[a, i])
                      + ALPHA * (t_b - base))
                feas = load[others] + O[i] <= Cap[others] + 1e-9
                if not bool(feas.any()):
                    ok = False
                    break
                masked = np.where(feas, dl, np.inf)
                bi = int(np.argmin(masked))
                b = int(others[bi])
                total_delta += float(masked[bi])
                load[a] -= O[i]
                load[b] += O[i]
                assign[i] = b
            if ok and total_delta < -1e-7:
                improved_any = True
            else:
                assign[:] = saved
        return improved_any

    def open_moves(assign, budget, T=6):
        start = time.time()
        improved_any = False
        hubs_open = set(int(h) for h in np.unique(assign))
        closed = np.array([b for b in range(n) if b not in hubs_open and valid_open[b]],
                          dtype=int)
        if closed.size == 0:
            return False
        prom = (w_weight * d[arange_n, assign] - f)[closed]
        order = closed[np.argsort(-prom)]
        for b in order[:T]:
            if time.time() - start > budget:
                break
            b = int(b)
            before = objective(assign)
            saved = assign.copy()
            assign[b] = b
            local_search(assign, min(1.0, max(0.1, budget - (time.time() - start))))
            after = objective(assign)
            if after < before - 1e-6:
                improved_any = True
            else:
                assign[:] = saved
        return improved_any

    # -------------------- run heuristic --------------------
    hubs0 = greedy_facility(Tot * 1.05)
    assign0 = greedy_assignment(hubs0)
    if assign0 is None:
        assign0 = greedy_assignment([int(k) for k in np.where(valid_open)[0]])
    if assign0 is None:
        k = int(np.argmax(Cap))
        assign0 = np.full(n, k, dtype=int)
        assign0[k] = k
    record(assign0)

    hb_total = min(max(1.0, 0.10 * total), 20.0)
    hstart = time.time()

    def hbudget(frac):
        return max(0.05, min(frac * hb_total, hb_total - (time.time() - hstart),
                             tleft() - 0.2))

    local_search(assign0, hbudget(0.6))
    record(assign0)
    improved = True
    while time.time() - hstart < hb_total and tleft() > 0.5 and improved:
        improved = False
        if close_moves(assign0, hbudget(0.25)):
            improved = True
        if local_search(assign0, hbudget(0.35)):
            improved = True
        if open_moves(assign0, hbudget(0.4)):
            improved = True
        if local_search(assign0, hbudget(0.35)):
            improved = True
        record(assign0)

    # -------------------- MIP (Ernst-Krishnamoorthy linearization) --------------------
    def mip_solve(C, warm_assign, time_budget):
        H = len(C)
        C_arr = np.asarray(C, dtype=int)
        pos = {int(c): idx for idx, c in enumerate(C)}
        Nz = n * H * H
        Nx = n * H
        Ntot = Nz + Nx + H

        dCC = d[np.ix_(C_arr, C_arr)]
        cobj = np.empty(Ntot)
        cobj[:Nz] = np.tile((ALPHA * dCC).ravel(), n)
        cx = CHI * O[:, None] * d[:, C_arr] + DELTA * Dm[:, None] * d[C_arr, :].T
        cobj[Nz:Nz + Nx] = cx.ravel()
        cobj[Nz + Nx:] = f[C_arr]

        rows, cols, vals = [], [], []
        senses, rhs = [], []
        row0 = 0

        # R1: sum_k x[i,k] = 1
        rows.append(np.repeat(np.arange(n), H))
        cols.append(Nz + np.arange(Nx))
        vals.append(np.ones(Nx))
        senses += ['='] * n
        rhs += [1.0] * n
        row0 += n

        # R4: sum_i O_i x[i,k] - Cap_k y_k <= 0
        rows.append(row0 + np.tile(np.arange(H), n))
        cols.append(Nz + np.arange(Nx))
        vals.append(np.repeat(O, H))
        rows.append(row0 + np.arange(H))
        cols.append(Nz + Nx + np.arange(H))
        vals.append(-Cap[C_arr])
        senses += ['<'] * H
        rhs += [0.0] * H
        row0 += H

        # R2: x[i,k] - y[k] <= 0
        rows.append(row0 + np.arange(Nx))
        cols.append(Nz + np.arange(Nx))
        vals.append(np.ones(Nx))
        rows.append(row0 + np.arange(Nx))
        cols.append(Nz + Nx + np.tile(np.arange(H), n))
        vals.append(-np.ones(Nx))
        senses += ['<'] * Nx
        rhs += [0.0] * Nx
        row0 += Nx

        # R3: x[c_k, k] - y[k] = 0  (open hub assigned to itself)
        rows.append(row0 + np.arange(H))
        cols.append(Nz + C_arr * H + np.arange(H))
        vals.append(np.ones(H))
        rows.append(row0 + np.arange(H))
        cols.append(Nz + Nx + np.arange(H))
        vals.append(-np.ones(H))
        senses += ['='] * H
        rhs += [0.0] * H
        row0 += H

        # R5: sum_m z[i,k,m] - O_i x[i,k] = 0
        base_rows = np.arange(Nx)  # i*H+k
        rows.append(row0 + np.repeat(base_rows, H))
        cols.append((base_rows[:, None] * H + np.arange(H)[None, :]).ravel())
        vals.append(np.ones(Nx * H))
        rows.append(row0 + base_rows)
        cols.append(Nz + base_rows)
        vals.append(-np.repeat(O, H))
        senses += ['='] * Nx
        rhs += [0.0] * Nx
        row0 += Nx

        # R6: sum_k z[i,k,m] - sum_j W[i,j] x[j,m] = 0
        idx = np.arange(Nz)
        ii6 = idx // (H * H)
        mm6 = idx % H
        rows.append(row0 + ii6 * H + mm6)
        cols.append(idx)
        vals.append(np.ones(Nz))
        I3 = np.arange(n)[:, None, None]
        M3 = np.arange(H)[None, :, None]
        J3 = np.arange(n)[None, None, :]
        rows.append((row0 + I3 * H + M3).ravel())
        cols.append((Nz + J3 * H + M3).ravel())
        vals.append((-np.broadcast_to(W[:, None, :], (n, H, n))).ravel())
        senses += ['='] * Nx
        rhs += [0.0] * Nx
        row0 += Nx

        A = sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(row0, Ntot)).tocsr()
        sense_arr = np.array(senses)
        rhs_arr = np.array(rhs, dtype=float)

        model = gp.Model("csahlp")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = max(0.5, float(time_budget))

        var = model.addMVar(Ntot, lb=0.0, vtype=['C'] * Nz + ['B'] * (Nx + H))
        model.setObjective(cobj @ var, GRB.MINIMIZE)
        model.addMConstr(A, var, sense_arr, rhs_arr)

        # warm start from incumbent heuristic solution
        if warm_assign is not None and all(int(a) in pos for a in warm_assign):
            try:
                start_vec = np.zeros(Ntot)
                xs = np.zeros((n, H))
                ai = np.array([pos[int(a)] for a in warm_assign], dtype=int)
                xs[np.arange(n), ai] = 1.0
                ys = np.zeros(H)
                for c in C:
                    if int(warm_assign[int(c)]) == int(c):
                        ys[pos[int(c)]] = 1.0
                zs = np.zeros((n, H, H))
                G = np.zeros((n, H))
                for i in range(n):
                    bc = np.bincount(warm_assign, weights=W[i], minlength=n)
                    G[i] = bc[C_arr]
                zs[np.arange(n), ai, :] = G
                start_vec[:Nz] = zs.ravel()
                start_vec[Nz:Nz + Nx] = xs.ravel()
                start_vec[Nz + Nx:] = ys
                var.setAttr("Start", start_vec)
            except Exception:
                pass

        def cb(m, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    xv = m.cbGetSolution(var)
                    xmat = xv[Nz:Nz + Nx].reshape(n, H)
                    p = xmat.argmax(axis=1)
                    if xmat[np.arange(n), p].min() > 0.5:
                        record(C_arr[p])
                except Exception:
                    pass

        model.optimize(cb)

        status = model.Status
        if model.SolCount > 0:
            try:
                xv = var.X
                xmat = xv[Nz:Nz + Nx].reshape(n, H)
                p = xmat.argmax(axis=1)
                if xmat[np.arange(n), p].min() > 0.5:
                    assign_new = C_arr[p]
                    hubs_u = np.unique(assign_new)
                    ok = all(int(assign_new[h]) == int(h) for h in hubs_u)
                    if ok:
                        loads = np.bincount(assign_new, weights=O, minlength=n)
                        ok = all(loads[h] <= Cap[h] + 1e-6 for h in hubs_u)
                    if ok:
                        record(assign_new)
            except Exception:
                pass
        model.dispose()
        return status

    if HAVE_GUROBI and HAVE_SCIPY and tleft() > 2.0 and best["assign"] is not None:
        if n <= 70:
            C = list(range(n))
        else:
            ZCAP = min(550000.0, 300000.0 + 1500.0 * total)
            H = int(math.sqrt(ZCAP / n))
            H = max(12, min(H, 70, n))
            nb = len(np.unique(best["assign"]))
            H = max(H, nb + 3)
            C = select_candidates(H,
                                  must=[int(h) for h in np.unique(best["assign"])],
                                  min_capacity=Tot * 1.02)
        while tleft() > 2.0:
            try:
                status = mip_solve(C, best["assign"], tleft() - 0.5)
            except Exception:
                break
            if tleft() > 0.5 and best["assign"] is not None:
                a = best["assign"].copy()
                local_search(a, min(1.0, tleft() - 0.4))
                record(a)
            if len(C) >= n:
                break
            if status != GRB.Status.OPTIMAL:
                break
            # optimal on the restricted candidate set: enlarge it (LNS) and retry
            Hmax = min(n, int(math.sqrt(550000.0 / n)))
            baseC = list(dict.fromkeys(
                C + [int(h) for h in np.unique(best["assign"])]))
            if len(baseC) >= Hmax:
                break
            unserved = w_weight * d[arange_n, best["assign"]]
            inC = np.zeros(n, dtype=bool)
            inC[baseC] = True
            cand = np.where(~inC)[0]
            if cand.size == 0:
                break
            r = min(15, Hmax - len(baseC))
            add = cand[np.argsort(-unserved[cand])[:r]]
            C = baseC + [int(x) for x in add]

    # -------------------- final polish with remaining time --------------------
    if best["assign"] is not None:
        while tleft() > 0.6:
            a = best["assign"].copy()
            changed = local_search(a, min(1.0, tleft() - 0.4))
            ch2, ch3 = False, False
            if tleft() > 1.0:
                ch2 = close_moves(a, min(0.5, tleft() - 0.5))
            if tleft() > 1.2:
                ch3 = open_moves(a, min(0.8, tleft() - 0.5))
            if local_search(a, min(0.5, max(0.1, tleft() - 0.4))):
                changed = True
            record(a)
            if not (changed or ch2 or ch3):
                break

    # -------------------- write solution --------------------
    final_assign = best["assign"]
    if final_assign is None:
        final_assign = np.arange(n, dtype=int)
    obj = best["obj"] if np.isfinite(best["obj"]) else objective(final_assign)
    sol = {
        "objective_value": float(obj),
        "hubs": [int(h) for h in np.unique(final_assign)],
        "assignment": [int(a) for a in final_assign],
    }
    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh)


if __name__ == "__main__":
    main()