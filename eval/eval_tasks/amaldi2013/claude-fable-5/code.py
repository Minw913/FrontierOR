import argparse
import json
import time
import numpy as np

from solution_logger import SolutionLogger


# ----------------------------- geometry helpers -----------------------------

def fit_plane(X):
    """Total least squares hyperplane fit. Returns unit normal w and offset w0
    with plane {x : w.x = w0}."""
    d = X.shape[1]
    if X.shape[0] == 0:
        w = np.zeros(d)
        w[0] = 1.0
        return w, 0.0
    m = X.mean(axis=0)
    if d == 1:
        return np.array([1.0]), float(m[0])
    Xc = X - m
    C = Xc.T @ Xc
    C = (C + C.T) * 0.5
    try:
        _, vecs = np.linalg.eigh(C)
        w = vecs[:, 0].copy()
    except np.linalg.LinAlgError:
        w = np.zeros(d)
        w[0] = 1.0
    nw = np.linalg.norm(w)
    if not np.isfinite(nw) or nw < 1e-12:
        w = np.zeros(d)
        w[0] = 1.0
    else:
        w = w / nw
    return w, float(w @ m)


def distances(P, w, b):
    return np.abs(P @ w - b)


def seed_local(P, anchor_idx, rng, subset=None):
    """Fit a plane through the anchor point and randomly chosen nearby points."""
    d = P.shape[1]
    if subset is None:
        Q = P
    else:
        Q = P[subset]
    dd = np.linalg.norm(Q - P[anchor_idx], axis=1)
    m = min(len(Q), max(d, 3 * d))
    if len(Q) > m:
        near = np.argpartition(dd, m - 1)[:m]
    else:
        near = np.arange(len(Q))
    k = min(d, len(near))
    if k <= 0:
        w = np.zeros(d)
        w[0] = 1.0
        return w, float(P[anchor_idx, 0])
    idx = rng.choice(near, size=k, replace=False)
    return fit_plane(Q[idx])


# ----------------------------- solution handling -----------------------------

def patch_coverage(P, planes, eps):
    """Ensure every point is covered; add trivial planes for uncovered points."""
    n, d = P.shape
    covered = np.zeros(n, dtype=bool)
    for (w, b) in planes:
        covered |= distances(P, w, b) <= eps
    missing = np.where(~covered)[0]
    for i in missing:
        w = np.zeros(d)
        w[0] = 1.0
        b = float(P[i, 0])
        cov = distances(P, w, b) <= eps
        planes.append((w, b))
        covered |= cov
    return planes


def prune(P, planes, eps):
    """Remove redundant planes greedily (smallest coverage first)."""
    if not planes:
        return planes
    covs = [distances(P, w, b) <= eps for (w, b) in planes]
    cnt = np.zeros(P.shape[0], dtype=np.int64)
    for c in covs:
        cnt += c
    if np.any(cnt == 0):
        return planes  # coverage incomplete; caller must patch first
    keep = [True] * len(planes)
    order = sorted(range(len(planes)), key=lambda j: int(covs[j].sum()))
    for j in order:
        c = covs[j]
        if np.all(cnt[c] >= 2):
            keep[j] = False
            cnt[c] -= 1
    return [planes[j] for j in range(len(planes)) if keep[j]]


def finalize_planes(P, planes, eps):
    planes = list(planes)
    planes = patch_coverage(P, planes, eps)
    planes = prune(P, planes, eps)
    return planes


def build_solution(P, planes, eps):
    n, d = P.shape
    point_assign = [[] for _ in range(n)]
    hps = []
    for j, (w, b) in enumerate(planes):
        cov = np.where(distances(P, w, b) <= eps)[0]
        hps.append({
            "w": [float(x) for x in w],
            "w0": float(b),
            "assigned_points": [int(i) for i in cov],
        })
        for i in cov:
            point_assign[int(i)].append(j)
    return {
        "objective_value": float(len(planes)),
        "hyperplanes": hps,
        "point_assignments": point_assign,
    }


# ----------------------------- greedy RANSAC cover -----------------------------

def greedy_cover(P, eps, deadline, rng, pool, pool_cap=6000):
    n, d = P.shape
    uncovered = np.ones(n, dtype=bool)
    planes = []
    base_trials = int(np.clip(3.0e6 / (n * d + 1), 8, 48))
    while uncovered.any():
        U = np.where(uncovered)[0]
        nu = len(U)
        best = None
        bestc = -1
        trials = base_trials if nu > d else 1
        for t in range(trials):
            if time.time() > deadline and best is not None:
                break
            if nu <= d:
                idx = U
            else:
                if t % 3 == 2:
                    idx = rng.choice(U, size=d, replace=False)
                else:
                    a = U[int(rng.integers(nu))]
                    dd = np.linalg.norm(P[U] - P[a], axis=1)
                    m = min(nu, 3 * d + 1)
                    near = U[np.argpartition(dd, m - 1)[:m]] if nu > m else U
                    idx = rng.choice(near, size=min(d, len(near)), replace=False)
            w, b = fit_plane(P[idx])
            cw, cb, cc, ccov = w, b, -1, None
            for r in range(4):
                dist = distances(P, w, b)
                cov = dist <= eps
                cnt = int(np.count_nonzero(cov & uncovered))
                if cnt > cc:
                    cc, cw, cb, ccov = cnt, w, b, cov
                if r < 3:
                    S = np.where(cov)[0]
                    if len(S) == 0:
                        break
                    w, b = fit_plane(P[S])
            if len(pool) < pool_cap:
                pool.append((cw, cb))
            if cc > bestc:
                bestc = cc
                best = (cw, cb, ccov)
        if best is None or bestc <= 0:
            # guaranteed progress: exact axis plane through one uncovered point
            i = U[0]
            w = np.zeros(d)
            w[0] = 1.0
            b = float(P[i, 0])
            cov = distances(P, w, b) <= eps
            cov[i] = True
            best = (w, b, cov)
            pool.append((w, b))
        w, b, cov = best
        planes.append((w, b))
        uncovered &= ~cov
    return planes


# ----------------------------- k-plane clustering -----------------------------

def kplane_search(P, eps, W, b, deadline, rng, max_iters=60):
    n, d = P.shape
    K = W.shape[0]
    best_viol = np.inf
    stall = 0
    ar = np.arange(n)
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        D = np.abs(P @ W.T - b[None, :])
        assign = np.argmin(D, axis=1)
        mind = D[ar, assign]
        viol = float(mind.max())
        if viol <= eps:
            return True, W, b
        if viol < best_viol - 1e-12:
            best_viol = viol
            stall = 0
        else:
            stall += 1
        if stall >= 8:
            break
        for k in range(K):
            idx = np.where(assign == k)[0]
            if len(idx) > 0:
                W[k], b[k] = fit_plane(P[idx])
        if stall in (3, 6):
            bad = np.where(mind > eps)[0]
            if len(bad) > 0:
                i = int(bad[int(np.argmax(mind[bad]))])
                sizes = np.bincount(assign, minlength=K)
                k = int(np.argmin(sizes))
                W[k], b[k] = seed_local(P, i, rng, subset=bad)
    return False, W, b


def random_init(P, K, rng):
    n, d = P.shape
    W = np.zeros((K, d))
    b = np.zeros(K)
    for k in range(K):
        i = int(rng.integers(n))
        W[k], b[k] = seed_local(P, i, rng)
    return W, b


# ----------------------------- set-cover MIP -----------------------------

def mip_improve(P, pool, base_planes, eps, tlim):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return base_planes
    n = P.shape[0]
    cands = []
    seen = {}
    forced = set()
    for src, lst in (("base", base_planes), ("pool", pool)):
        for (w, b) in lst:
            cov = distances(P, w, b) <= eps
            s = int(cov.sum())
            if s == 0:
                continue
            key = cov.tobytes()
            if key in seen:
                if src == "base":
                    forced.add(seen[key])
                continue
            seen[key] = len(cands)
            if src == "base":
                forced.add(len(cands))
            cands.append((w, b, cov, s))
    if len(cands) <= len(base_planes):
        return base_planes
    MAXC = 3000
    if len(cands) > MAXC:
        order = sorted(range(len(cands)), key=lambda j: -cands[j][3])
        keepset = set(order[:MAXC]) | forced
        newc = []
        newforced = set()
        for j in sorted(keepset):
            if j in forced:
                newforced.add(len(newc))
            newc.append(cands[j])
        cands = newc
        forced = newforced
    try:
        m = gp.Model("setcover")
        m.Params.OutputFlag = 0
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.TimeLimit = max(1.0, float(tlim))
        nc = len(cands)
        y = m.addVars(nc, vtype=GRB.BINARY)
        covmat = np.stack([c[2] for c in cands], axis=0)
        if not covmat.any(axis=0).all():
            return base_planes
        for i in range(n):
            js = np.flatnonzero(covmat[:, i])
            m.addConstr(gp.quicksum(y[int(j)] for j in js) >= 1)
        m.setObjective(gp.quicksum(y[j] for j in range(nc)), GRB.MINIMIZE)
        for j in range(nc):
            y[j].Start = 1.0 if j in forced else 0.0
        m.optimize()
        if m.SolCount > 0:
            chosen = [(cands[j][0], cands[j][1]) for j in range(nc) if y[j].X > 0.5]
            if 0 < len(chosen) < len(base_planes):
                return chosen
    except Exception:
        pass
    return base_planes


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", type=str, required=True)
    ap.add_argument("--solution_path", type=str, required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    total = max(1, int(args.time_limit))
    safety = max(0.5, min(2.0, 0.05 * total))
    deadline = t0 + total - safety

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)
    n = int(data["n"])
    d = int(data["d"])
    eps = float(data["epsilon"])

    if n == 0:
        sol = {"objective_value": 0.0, "hyperplanes": [], "point_assignments": []}
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)
        return

    P = np.asarray(data["points"], dtype=float).reshape(n, d)
    eps_cov = eps + 1e-9  # tiny numeric slack used consistently everywhere
    rng = np.random.default_rng(0)

    best_planes = None
    best_sol = None

    def adopt(planes):
        nonlocal best_planes, best_sol
        if best_planes is None or len(planes) < len(best_planes):
            best_planes = planes
            best_sol = build_solution(P, best_planes, eps_cov)
            if logger:
                logger.log_solution(best_sol["objective_value"], best_sol)
            return True
        return False

    try:
        # quick single-plane check
        w, b = fit_plane(P)
        if distances(P, w, b).max() <= eps_cov:
            adopt([(w, b)])
        elif d == 1:
            # exact 1-D interval covering (optimal)
            xs = np.sort(P[:, 0])
            planes = []
            i = 0
            while i < n:
                c = xs[i] + eps
                planes.append((np.array([1.0]), float(c)))
                while i < n and xs[i] <= c + eps_cov - eps + eps:  # xs[i] <= c + eps
                    if xs[i] <= c + eps:
                        i += 1
                    else:
                        break
            planes = finalize_planes(P, planes, eps_cov)
            adopt(planes)
        else:
            # Phase 1: greedy RANSAC covering
            pool = []
            greedy_deadline = min(deadline, t0 + 0.40 * total)
            planes = greedy_cover(P, eps_cov, greedy_deadline, rng, pool)
            planes = finalize_planes(P, planes, eps_cov)
            adopt(planes)

            # Phase 2: set-cover MIP over the candidate pool
            if best_planes is not None and len(best_planes) > 2 and n <= 30000:
                rem = deadline - time.time()
                tlim = min(0.25 * total, 0.5 * rem)
                if tlim > 2.0:
                    chosen = mip_improve(P, pool, best_planes, eps_cov, tlim)
                    if len(chosen) < len(best_planes):
                        chosen = finalize_planes(P, chosen, eps_cov)
                        adopt(chosen)

            # Phase 3: k-plane clustering to reduce the number of planes
            attempt = 0
            fails = 0
            while (best_planes is not None and len(best_planes) > 1
                   and time.time() < deadline - 0.05 and fails < 400):
                K = len(best_planes) - 1
                mode = attempt % 5
                if mode == 4 or K < 1:
                    W, bb = random_init(P, max(1, K), rng)
                else:
                    if mode == 0:
                        counts = [int((distances(P, ww, wb) <= eps_cov).sum())
                                  for (ww, wb) in best_planes]
                        j = int(np.argmin(counts))
                    else:
                        j = int(rng.integers(len(best_planes)))
                    W = np.array([ww for t, (ww, wb) in enumerate(best_planes) if t != j])
                    bb = np.array([wb for t, (ww, wb) in enumerate(best_planes) if t != j])
                ok, W, bb = kplane_search(P, eps_cov, W, bb, deadline, rng)
                attempt += 1
                if ok:
                    cand = [(W[k].copy(), float(bb[k])) for k in range(W.shape[0])]
                    cand = finalize_planes(P, cand, eps_cov)
                    if adopt(cand):
                        fails = 0
                    else:
                        fails += 1
                else:
                    fails += 1
    except Exception:
        pass

    # absolute fallback: one exact plane per point group
    if best_planes is None:
        planes = []
        i = 0
        while i < n:
            grp = P[i:i + max(1, d)]
            planes.append(fit_plane(grp))
            i += max(1, d)
        planes = finalize_planes(P, planes, eps_cov)
        best_planes = planes
        best_sol = build_solution(P, best_planes, eps_cov)
        if logger:
            logger.log_solution(best_sol["objective_value"], best_sol)

    if best_sol is None:
        best_sol = build_solution(P, best_planes, eps_cov)

    with open(args.solution_path, "w") as f:
        json.dump(best_sol, f)


if __name__ == "__main__":
    main()