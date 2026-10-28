import argparse
import json
import time
import math
import random
from itertools import combinations

import numpy as np

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + max(1, args.time_limit) - 1.0

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    n = int(inst["n_nodes"])
    p = min(int(inst["p"]), n)
    nodes = inst["nodes"]
    w = np.zeros(n, dtype=float)
    for nd in nodes:
        w[int(nd["id"])] = float(nd["waiting_time"])
    D = np.array(inst["demand_matrix"], dtype=float)
    dist = np.array(inst["distance_matrix"], dtype=float)
    comp_hubs = sorted(set(int(h) for h in inst["competitor_hub_locations"]))
    prm = inst["parameters"]
    alpha = float(prm["alpha"])
    gamma = float(prm["gamma"])
    beta = float(prm["beta"])
    lam = float(prm["lambda"])
    cs = float(prm["cost_scalar"])
    A2 = float(prm["attractiveness_2hub"])
    A1 = float(prm["attractiveness_1hub"])

    rng = random.Random(0)

    # ---------- utility matrices ----------
    def u_dir(k, l):
        # route i -> k -> l -> j for all (i,j)
        const_t = dist[k, l] + w[k] + w[l]
        t = (dist[:, k] + w)[:, None] + dist[l, :][None, :] + const_t
        c = cs * (dist[:, k][:, None] + dist[l, :][None, :] + alpha * dist[k, l])
        with np.errstate(divide="ignore", invalid="ignore"):
            den = gamma * np.power(t, beta) + (1.0 - gamma) * np.power(c, lam)
        A = A1 if k == l else A2
        out = np.zeros((n, n), dtype=float)
        np.divide(A, den, out=out, where=den > 0)
        return out

    cache = {}
    cache_cap = max(4 * n + 64, int(2.5e8 / (8.0 * n * n)))

    def get_pair(k, l):
        key = (k, l) if k <= l else (l, k)
        u = cache.get(key)
        if u is not None:
            return u
        if key[0] == key[1]:
            u = u_dir(key[0], key[0])
        else:
            u = np.maximum(u_dir(key[0], key[1]), u_dir(key[1], key[0]))
        if len(cache) >= cache_cap:
            cache.clear()
        cache[key] = u
        return u

    # ---------- competitor utility sum ----------
    C = np.zeros((n, n), dtype=float)
    for idx, k in enumerate(comp_hubs):
        for l in comp_hubs[idx:]:
            C += get_pair(k, l)

    def objective(S):
        tot = S + C
        frac = np.zeros_like(S)
        np.divide(S, tot, out=frac, where=tot > 0)
        return float(np.sum(D * frac))

    def build_S(H):
        S = np.zeros((n, n), dtype=float)
        Hl = list(H)
        for idx, a in enumerate(Hl):
            for b in Hl[idx:]:
                S += get_pair(a, b)
        return S

    # ---------- global best tracking ----------
    best_obj = -1.0
    best_H = []

    def report(o, H):
        nonlocal best_obj, best_H
        if o > best_obj + 1e-12:
            best_obj = o
            best_H = sorted(H)
            if logger:
                logger.log_solution(best_obj, {
                    "objective_value": best_obj,
                    "hub_locations": list(best_H),
                })

    if p == 0:
        report(0.0, [])
        with open(args.solution_path, "w") as f:
            json.dump({"objective_value": 0.0, "hub_locations": []}, f)
        return

    # ---------- brute force for small instances ----------
    try:
        ncomb = math.comb(n, p)
    except Exception:
        ncomb = float("inf")
    total_pairs = n * (n + 1) // 2
    bf_ok = (ncomb * (p * (p + 1) // 2 + 1) * n * n <= 3e8) and \
            (total_pairs * 8.0 * n * n <= 2.0e8)
    if bf_ok:
        done = True
        for combo in combinations(range(n), p):
            S = np.zeros((n, n), dtype=float)
            for idx in range(p):
                for jdx in range(idx, p):
                    S += get_pair(combo[idx], combo[jdx])
            o = objective(S)
            report(o, list(combo))
            if time.time() > deadline:
                done = False
                break
        if done:
            with open(args.solution_path, "w") as f:
                json.dump({"objective_value": best_obj,
                           "hub_locations": list(best_H)}, f)
            return
        # otherwise fall through to heuristic (best so far kept)

    # ---------- greedy construction ----------
    def greedy():
        H = []
        S = np.zeros((n, n), dtype=float)
        while len(H) < p:
            best = None
            for b in range(n):
                if b in H:
                    continue
                add = get_pair(b, b).copy()
                for c in H:
                    add += get_pair(b, c)
                o = objective(S + add)
                if best is None or o > best[0]:
                    best = (o, b, add)
            H.append(best[1])
            S = S + best[2]
            if time.time() > deadline:
                break
        return H, S, objective(S)

    # ---------- swap local search ----------
    def local_search(H, S, obj_val):
        H = list(H)
        while time.time() < deadline:
            Hset = set(H)
            contribs = {}
            for a in H:
                ca = get_pair(a, a).copy()
                for c in H:
                    if c != a:
                        ca += get_pair(a, c)
                contribs[a] = ca
            best_move = None
            best_o = obj_val
            outs = [b for b in range(n) if b not in Hset]
            timed_out = False
            for b in outs:
                addb = get_pair(b, b).copy()
                for c in H:
                    addb += get_pair(b, c)
                for a in H:
                    Snew = S - contribs[a] + addb - get_pair(a, b)
                    o = objective(Snew)
                    if o > best_o + 1e-10:
                        best_o = o
                        best_move = (a, b)
                if time.time() > deadline:
                    timed_out = True
                    break
            if best_move is None:
                break
            a, b = best_move
            addb = get_pair(b, b).copy()
            for c in H:
                addb += get_pair(b, c)
            S = S - contribs[a] + addb - get_pair(a, b)
            H[H.index(a)] = b
            obj_val = objective(S)  # refresh (numerical drift control)
            report(obj_val, H)
            if timed_out:
                break
        return H, S, obj_val

    # ---------- initial: greedy + LS ----------
    H, S, o = greedy()
    if len(H) < p:  # fill if timeout during greedy
        for b in range(n):
            if len(H) >= p:
                break
            if b not in H:
                H.append(b)
        S = build_S(H)
        o = objective(S)
    report(o, H)
    H, S, o = local_search(H, S, o)
    report(o, H)

    cur_H, cur_o = list(best_H), best_obj

    # ---------- iterated local search ----------
    no_improve = 0
    while time.time() < deadline and p < n:
        base = list(best_H)
        k = 1 if (p == 1 or rng.random() < 0.6) else min(2, p)
        newH = list(base)
        outs = [b for b in range(n) if b not in base]
        if not outs:
            break
        if no_improve >= 12:
            # random restart
            newH = rng.sample(range(n), p)
            no_improve = 0
        else:
            remove = rng.sample(range(p), k)
            adds = rng.sample(outs, k)
            for idx, a in zip(remove, adds):
                newH[idx] = a
            newH = list(set(newH))
            while len(newH) < p:
                cand = rng.randrange(n)
                if cand not in newH:
                    newH.append(cand)
        S = build_S(newH)
        o = objective(S)
        report(o, newH)
        prev_best = best_obj
        newH, S, o = local_search(newH, S, o)
        report(o, newH)
        if best_obj > prev_best + 1e-9:
            no_improve = 0
        else:
            no_improve += 1

    if not best_H:
        best_H = sorted(range(p))
        best_obj = objective(build_S(best_H))
        report(best_obj, best_H)

    with open(args.solution_path, "w") as f:
        json.dump({"objective_value": best_obj,
                   "hub_locations": [int(x) for x in sorted(best_H)]}, f)


if __name__ == "__main__":
    main()