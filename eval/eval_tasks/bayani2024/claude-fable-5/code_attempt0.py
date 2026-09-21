import argparse
import json
import time
import random

import numpy as np

from solution_logger import SolutionLogger


def build_quad_arrays(quad):
    if not quad:
        return (np.zeros(0, dtype=np.int64),) * 4 + (np.zeros(0, dtype=float),)
    q = np.array(quad, dtype=float)
    qi = q[:, 0].astype(np.int64)
    qs = q[:, 1].astype(np.int64)
    qj = q[:, 2].astype(np.int64)
    qr = q[:, 3].astype(np.int64)
    qc = q[:, 4]
    return qi, qs, qj, qr, qc


def exact_obj(a, L0, qi, qs, qj, qr, qc):
    n = len(a)
    obj = float(L0[np.arange(n), a].sum())
    if len(qc):
        active = (a[qi] == qs) & (a[qj] == qr)
        obj += float(qc[active].sum())
    return obj


def make_sol_dict(a, obj):
    return {
        "objective_value": obj,
        "assignment": {str(i): int(a[i]) for i in range(len(a))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    start = time.time()
    deadline = start + max(1, args.time_limit) - 1.0

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    n = int(data['n_clients'])
    m = int(data['n_servers'])
    L0 = np.array(data['linear_costs'], dtype=float).reshape(n, m)
    quad = data.get('quadratic_costs') or []
    qi, qs, qj, qr, qc = build_quad_arrays(quad)

    rng = random.Random(0)
    np.random.seed(0)

    # ---------- Trivial cases ----------
    if m == 1 or n == 0:
        a = np.zeros(n, dtype=np.int64)
        obj = exact_obj(a, L0, qi, qs, qj, qr, qc)
        sol = make_sol_dict(a, obj)
        if logger:
            logger.log_solution(obj, sol)
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        return

    # ---------- Build adjacency structures ----------
    # Lm: linear costs with self-pair quadratic entries folded in
    Lm = L0.copy()
    adj_j = [[] for _ in range(n * m)]
    adj_r = [[] for _ in range(n * m)]
    adj_c = [[] for _ in range(n * m)]
    for e in quad:
        i = int(e[0]); s = int(e[1]); j = int(e[2]); r = int(e[3]); c = float(e[4])
        if i == j:
            if s == r:
                Lm[i, s] += c
            continue
        idx1 = i * m + s
        idx2 = j * m + r
        adj_j[idx1].append(j); adj_r[idx1].append(r); adj_c[idx1].append(c)
        adj_j[idx2].append(i); adj_r[idx2].append(s); adj_c[idx2].append(c)

    AJ = [np.array(x, dtype=np.int64) for x in adj_j]
    AR = [np.array(x, dtype=np.int64) for x in adj_r]
    AC = [np.array(x, dtype=float) for x in adj_c]
    del adj_j, adj_r, adj_c

    rows = np.arange(n)

    def rebuild_contrib(a):
        contrib = np.zeros((n, m), dtype=float)
        for i in range(n):
            idx = i * m + int(a[i])
            if len(AJ[idx]):
                np.add.at(contrib, (AJ[idx], AR[idx]), AC[idx])
        return contrib

    def internal_obj(a, contrib):
        return float(Lm[rows, a].sum() + 0.5 * contrib[rows, a].sum())

    # ---------- Greedy construction ----------
    contrib = np.zeros((n, m), dtype=float)
    a = np.zeros(n, dtype=np.int64)
    order = list(range(n))
    rng.shuffle(order)
    for i in order:
        t = int(np.argmin(Lm[i] + contrib[i]))
        a[i] = t
        idx = i * m + t
        if len(AJ[idx]):
            np.add.at(contrib, (AJ[idx], AR[idx]), AC[idx])

    cur = internal_obj(a, contrib)
    best = cur
    best_a = a.copy()
    best_exact = exact_obj(best_a, L0, qi, qs, qj, qr, qc)
    if logger:
        logger.log_solution(best_exact, make_sol_dict(best_a, best_exact))

    # ---------- Tabu search ----------
    tabu = np.zeros((n, m), dtype=np.int64)
    tenure_base = max(5, n // 10)
    tenure_var = max(2, n // 5)
    stall_limit = max(1000, 20 * n)
    it = 0
    last_improve = 0
    resync_every = 5000

    while time.time() < deadline:
        it += 1

        vals = Lm + contrib
        curvals = vals[rows, a]
        delta = vals - curvals[:, None]
        delta[rows, a] = np.inf
        # aspiration: allow tabu move if it beats the best known
        aspir = (cur + delta) < (best - 1e-9)
        blocked = (tabu > it) & (~aspir)
        delta[blocked] = np.inf

        flat = int(np.argmin(delta))
        i, t = divmod(flat, m)
        d = delta[i, t]
        if not np.isfinite(d):
            tabu[:] = 0
            continue

        s = int(a[i])
        idx_old = i * m + s
        idx_new = i * m + t
        if len(AJ[idx_old]):
            np.add.at(contrib, (AJ[idx_old], AR[idx_old]), -AC[idx_old])
        if len(AJ[idx_new]):
            np.add.at(contrib, (AJ[idx_new], AR[idx_new]), AC[idx_new])
        a[i] = t
        cur += float(d)
        tabu[i, s] = it + tenure_base + rng.randrange(tenure_var)

        if it % resync_every == 0:
            cur = internal_obj(a, contrib)

        if cur < best - 1e-9:
            best = cur
            best_a = a.copy()
            last_improve = it
            ex = exact_obj(best_a, L0, qi, qs, qj, qr, qc)
            if ex < best_exact - 1e-12:
                best_exact = ex
                if logger:
                    logger.log_solution(ex, make_sol_dict(best_a, ex))

        if it - last_improve > stall_limit:
            # Perturb from best solution and restart local phase
            a = best_a.copy()
            k = max(1, n // 5)
            for i2 in rng.sample(range(n), min(k, n)):
                a[i2] = rng.randrange(m)
            contrib = rebuild_contrib(a)
            cur = internal_obj(a, contrib)
            tabu[:] = 0
            last_improve = it

    # ---------- Final polish: steepest descent from best ----------
    a = best_a.copy()
    contrib = rebuild_contrib(a)
    cur = internal_obj(a, contrib)
    improved = True
    safety = 0
    while improved and safety < 10 * n and time.time() < deadline + 0.5:
        improved = False
        safety += 1
        vals = Lm + contrib
        curvals = vals[rows, a]
        delta = vals - curvals[:, None]
        delta[rows, a] = np.inf
        flat = int(np.argmin(delta))
        i, t = divmod(flat, m)
        if delta[i, t] < -1e-9:
            s = int(a[i])
            idx_old = i * m + s
            idx_new = i * m + t
            if len(AJ[idx_old]):
                np.add.at(contrib, (AJ[idx_old], AR[idx_old]), -AC[idx_old])
            if len(AJ[idx_new]):
                np.add.at(contrib, (AJ[idx_new], AR[idx_new]), AC[idx_new])
            a[i] = t
            cur += float(delta[i, t])
            improved = True

    if cur < best - 1e-9:
        ex = exact_obj(a, L0, qi, qs, qj, qr, qc)
        if ex < best_exact - 1e-12:
            best_a = a.copy()
            best_exact = ex
            if logger:
                logger.log_solution(ex, make_sol_dict(best_a, ex))

    sol = make_sol_dict(best_a, best_exact)
    with open(args.solution_path, 'w') as f:
        json.dump(sol, f)


if __name__ == '__main__':
    main()