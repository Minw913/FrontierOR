import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except Exception:
        logger = None

    start = time.time()
    deadline = start + max(2, args.time_limit) - 1.5

    with open(args.instance_path) as fh:
        data = json.load(fh)

    n = int(data["num_facilities"])
    m = int(data["num_customers"])
    f = np.asarray(data["opening_costs"], dtype=float)
    C = np.asarray(data["allocation_costs"], dtype=float).reshape(n, m)
    # For a fixed open set S, optimal fractional assignment for customer j is
    # x_ij = (1/c_ij) / sum_{k in S} (1/c_kj), with allocation cost 1 / sum_{k in S}(1/c_kj).
    with np.errstate(divide="ignore"):
        A = 1.0 / C
    A = np.where(np.isfinite(A), A, 0.0)
    A = np.maximum(A, 1e-300)  # keep strictly positive to avoid degenerate divisions

    def alloc(D):
        return float(np.sum(1.0 / D))

    def total_cost(mask, D):
        return float(f[mask].sum()) + alloc(D)

    def build_solution(mask, D):
        X = np.zeros((n, m))
        idx = np.flatnonzero(mask)
        X[idx] = A[idx] / D[None, :]
        # normalize columns exactly to 1 for numerical cleanliness
        colsum = X.sum(axis=0)
        colsum[colsum <= 0] = 1.0
        X = X / colsum[None, :]
        obj = float(f[mask].sum() + np.sum(C * X * X * (X > 0)))
        sol = {
            "objective_value": obj,
            "variant": "quadratic",
            "y": [int(v) for v in mask],
            "x": X.tolist(),
        }
        return obj, sol

    best = {"obj": float("inf"), "mask": None}

    def report(mask, D):
        obj = total_cost(mask, D)
        if obj < best["obj"] - 1e-9:
            best["obj"] = obj
            best["mask"] = mask.copy()
            if logger:
                o, sol = build_solution(mask, D)
                logger.log_solution(o, sol)
        return obj

    # ---------------- Greedy construction ----------------
    mask = np.zeros(n, dtype=bool)
    with np.errstate(over="ignore"):
        scores = f + C.sum(axis=1)
    k0 = int(np.nanargmin(np.where(np.isfinite(scores), scores, np.inf)))
    mask[k0] = True
    D = A[mask].sum(axis=0)
    report(mask, D)

    while time.time() < deadline:
        closed = np.flatnonzero(~mask)
        if closed.size == 0:
            break
        ca = alloc(D)
        na = np.sum(1.0 / (D[None, :] + A[closed]), axis=1)
        delta = f[closed] + na - ca
        bi = int(np.argmin(delta))
        if delta[bi] < -1e-9:
            k = int(closed[bi])
            mask[k] = True
            D = A[mask].sum(axis=0)
            report(mask, D)
        else:
            break

    # ---------------- Local search (add / drop / swap) ----------------
    def local_search(mask, D, ls_deadline):
        cur = total_cost(mask, D)
        while time.time() < ls_deadline:
            ca = alloc(D)
            open_idx = np.flatnonzero(mask)
            closed_idx = np.flatnonzero(~mask)
            best_delta = -1e-7
            best_move = None
            # add moves
            if closed_idx.size:
                na = np.sum(1.0 / (D[None, :] + A[closed_idx]), axis=1)
                d = f[closed_idx] + na - ca
                bi = int(np.argmin(d))
                if d[bi] < best_delta:
                    best_delta = float(d[bi])
                    best_move = ("add", int(closed_idx[bi]))
            # drop moves
            if open_idx.size >= 2:
                Dm = D[None, :] - A[open_idx]
                bad = Dm <= 1e-300
                invalid = bad.any(axis=1)
                Dm = np.where(bad, 1.0, Dm)
                na = np.sum(1.0 / Dm, axis=1)
                d = -f[open_idx] + na - ca
                d[invalid] = np.inf
                bi = int(np.argmin(d))
                if d[bi] < best_delta:
                    best_delta = float(d[bi])
                    best_move = ("drop", int(open_idx[bi]))
            # swap moves
            if closed_idx.size and open_idx.size:
                for k in open_idx:
                    if time.time() > ls_deadline:
                        break
                    Dk = np.maximum(D - A[k], 0.0)
                    na = np.sum(1.0 / (Dk[None, :] + A[closed_idx]), axis=1)
                    d = (f[closed_idx] - f[k]) + na - ca
                    bi = int(np.argmin(d))
                    if d[bi] < best_delta:
                        best_delta = float(d[bi])
                        best_move = ("swap", int(k), int(closed_idx[bi]))
            if best_move is None:
                break
            if best_move[0] == "add":
                mask[best_move[1]] = True
            elif best_move[0] == "drop":
                mask[best_move[1]] = False
            else:
                mask[best_move[1]] = False
                mask[best_move[2]] = True
            D = A[mask].sum(axis=0)
            cur = total_cost(mask, D)
            report(mask, D)
        return mask, D, cur

    mask, D, _ = local_search(mask, D, deadline)
    report(mask, D)

    # ---------------- Exact refinement with Gurobi (convex MIQP) ----------------
    used_gurobi = False
    remaining = deadline - time.time()
    if n * m <= 400000 and remaining > 5.0:
        try:
            import gurobipy as gp
            from gurobipy import GRB
            import scipy.sparse as sp

            N = n * m
            model = gp.Model("qufl")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(1.0, deadline - time.time())

            yv = model.addMVar(n, vtype=GRB.BINARY, name="y")
            xv = model.addMVar(N, lb=0.0, ub=1.0, name="x")

            idx = np.arange(N)
            S = sp.csr_matrix((np.ones(N), (idx % m, idx)), shape=(m, N))
            model.addConstr(S @ xv == np.ones(m))
            R = sp.csr_matrix((np.ones(N), (idx, idx // m)), shape=(N, n))
            model.addConstr(xv - R @ yv <= 0)

            Cflat = np.where(np.isfinite(C), C, 1e12).reshape(-1)
            Q = sp.diags(Cflat)
            model.setObjective(f @ yv + xv @ Q @ xv, GRB.MINIMIZE)

            # warm start from heuristic
            if best["mask"] is not None:
                mk = best["mask"]
                Dw = A[mk].sum(axis=0)
                Xw = np.zeros((n, m))
                ii = np.flatnonzero(mk)
                Xw[ii] = A[ii] / Dw[None, :]
                yv.Start = mk.astype(float)
                xv.Start = Xw.reshape(-1)

            def cb(mdl, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        yy = np.asarray(mdl.cbGetSolution(yv))
                    except Exception:
                        return
                    mk2 = yy > 0.5
                    if not mk2.any():
                        return
                    D2 = A[mk2].sum(axis=0)
                    if np.any(D2 <= 0):
                        return
                    obj2 = float(f[mk2].sum() + np.sum(1.0 / D2))
                    if obj2 < best["obj"] - 1e-9:
                        best["obj"] = obj2
                        best["mask"] = mk2.copy()
                        if logger:
                            o, sol = build_solution(mk2, D2)
                            logger.log_solution(o, sol)

            model.optimize(cb)
            used_gurobi = True

            if model.SolCount > 0:
                yy = np.asarray(yv.X)
                mk2 = yy > 0.5
                if mk2.any():
                    D2 = A[mk2].sum(axis=0)
                    report(mk2, D2)
        except Exception:
            used_gurobi = False

    # ---------------- Perturbation restarts if Gurobi not used ----------------
    if not used_gurobi:
        rng = np.random.default_rng(0)
        while time.time() < deadline - 0.2:
            mk = best["mask"].copy()
            nflip = max(1, min(n, int(np.ceil(n * 0.1))))
            idxs = rng.choice(n, size=nflip, replace=False)
            mk[idxs] = ~mk[idxs]
            if not mk.any():
                mk[int(rng.integers(n))] = True
            D2 = A[mk].sum(axis=0)
            local_search(mk, D2, deadline)

    # ---------------- Write final solution ----------------
    if best["mask"] is None:
        mk = np.zeros(n, dtype=bool)
        mk[0] = True
        best["mask"] = mk
    Df = A[best["mask"]].sum(axis=0)
    obj, sol = build_solution(best["mask"], Df)
    if logger:
        logger.log_solution(obj, sol)
    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh)


if __name__ == "__main__":
    main()