import argparse
import json
import time
from typing import Dict, Tuple, List, Set

import numpy as np

from solution_logger import SolutionLogger


class UtilityProvider:
    """
    Lazily computes and caches route utility vectors U_{(a,b)} over all OD pairs,
    for unordered hub pairs (a,b) with a <= b.
    """
    def __init__(self, distance_matrix, waiting_times, params):
        self.D = np.asarray(distance_matrix, dtype=np.float64)
        self.W = np.asarray(waiting_times, dtype=np.float64)
        self.n = self.D.shape[0]

        self.alpha = float(params["alpha"])
        self.gamma = float(params["gamma"])
        self.beta = float(params["beta"])
        self.lam = float(params["lambda"])
        self.cost_scalar = float(params["cost_scalar"])
        self.attr_2hub = float(params["attractiveness_2hub"])
        self.attr_1hub = float(params["attractiveness_1hub"])

        self.cache: Dict[Tuple[int, int], np.ndarray] = {}

        # Precompute origin waiting term shape (n,1) used in all pair computations
        self.W_origin = self.W[:, None]
        self.eps = 1e-12

    def _denom(self, t, c):
        den = self.gamma * np.power(t, self.beta) + (1.0 - self.gamma) * np.power(c, self.lam)
        den = np.maximum(den, self.eps)
        return den

    def get_pair_vector(self, a: int, b: int) -> np.ndarray:
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in self.cache:
            return self.cache[key]

        D = self.D
        W = self.W
        cs = self.cost_scalar

        if a == b:
            # Single-hub route i -> a -> j (modeled as k=l=a)
            orig = D[:, a][:, None]
            dest = D[a, :][None, :]
            t = orig + dest + self.W_origin + W[a] + W[a]
            c = cs * (orig + dest)
            u = self.attr_1hub / self._denom(t, c)
        else:
            dab = D[a, b]

            # Order i -> a -> b -> j
            orig_a = D[:, a][:, None]
            dest_b = D[b, :][None, :]
            t1 = orig_a + dab + dest_b + self.W_origin + W[a] + W[b]
            c1 = cs * (orig_a + self.alpha * dab + dest_b)
            u1 = self.attr_2hub / self._denom(t1, c1)

            # Order i -> b -> a -> j
            orig_b = D[:, b][:, None]
            dest_a = D[a, :][None, :]
            t2 = orig_b + dab + dest_a + self.W_origin + W[b] + W[a]
            c2 = cs * (orig_b + self.alpha * dab + dest_a)
            u2 = self.attr_2hub / self._denom(t2, c2)

            u = np.maximum(u1, u2)

        vec = u.ravel().astype(np.float64, copy=False)
        self.cache[key] = vec
        return vec


def compute_objective(entrant_vec, competitor_vec, demand_vec, pos_idx):
    e = entrant_vec[pos_idx]
    c = competitor_vec[pos_idx]
    den = e + c
    share = np.divide(e, den, out=np.zeros_like(den), where=den > 0.0)
    return float(np.dot(demand_vec[pos_idx], share))


def build_competitor_utility(provider: UtilityProvider, competitor_hubs: List[int], q: int) -> np.ndarray:
    comp = np.zeros(q, dtype=np.float64)
    hubs = sorted(set(int(x) for x in competitor_hubs))
    for h in hubs:
        comp += provider.get_pair_vector(h, h)
    for i in range(len(hubs)):
        for j in range(i + 1, len(hubs)):
            comp += provider.get_pair_vector(hubs[i], hubs[j])
    return comp


def contrib_of_hub_to_set(provider: UtilityProvider, h: int, hub_set: Set[int]) -> np.ndarray:
    # Utility contribution of hub h given current selected hub set hub_set that already includes h:
    # self-route (h,h) + all distinct unordered pairs (h,s), s in hub_set \ {h}
    vec = provider.get_pair_vector(h, h).copy()
    for s in hub_set:
        if s == h:
            continue
        a, b = (h, s) if h < s else (s, h)
        vec += provider.get_pair_vector(a, b)
    return vec


def solve(instance, time_limit: int, logger: SolutionLogger = None):
    start = time.time()

    n = int(instance["n_nodes"])
    p = int(instance["p"])
    distance_matrix = instance["distance_matrix"]
    demand_matrix = np.asarray(instance["demand_matrix"], dtype=np.float64)
    competitor_hubs = instance["competitor_hub_locations"]
    params = instance["parameters"]
    nodes = instance["nodes"]
    waiting_times = [float(nodes[i]["waiting_time"]) for i in range(n)]

    q = n * n
    demand_vec = demand_matrix.ravel()
    pos_idx = np.where(demand_vec > 0.0)[0]

    provider = UtilityProvider(distance_matrix, waiting_times, params)

    competitor_vec = build_competitor_utility(provider, competitor_hubs, q)

    # Greedy constructive phase
    selected: Set[int] = set()
    entrant_vec = np.zeros(q, dtype=np.float64)

    all_nodes = list(range(n))

    best_complete_obj = -1.0
    best_complete_hubs = None

    for _ in range(min(p, n)):
        if time.time() - start >= time_limit:
            break

        best_h = None
        best_h_obj = -1.0
        best_h_delta = None

        for h in all_nodes:
            if h in selected:
                continue

            # delta if adding h
            delta = provider.get_pair_vector(h, h).copy()
            for s in selected:
                a, b = (h, s) if h < s else (s, h)
                delta += provider.get_pair_vector(a, b)

            cand_obj = compute_objective(entrant_vec + delta, competitor_vec, demand_vec, pos_idx)
            if cand_obj > best_h_obj:
                best_h_obj = cand_obj
                best_h = h
                best_h_delta = delta

            if time.time() - start >= time_limit:
                break

        if best_h is None:
            break

        selected.add(best_h)
        entrant_vec += best_h_delta

    # If incomplete due timeout, fill quickly with unused nodes
    if len(selected) < p:
        for h in all_nodes:
            if len(selected) >= p:
                break
            if h not in selected:
                delta = provider.get_pair_vector(h, h).copy()
                for s in selected:
                    a, b = (h, s) if h < s else (s, h)
                    delta += provider.get_pair_vector(a, b)
                selected.add(h)
                entrant_vec += delta

    current_obj = compute_objective(entrant_vec, competitor_vec, demand_vec, pos_idx)
    if len(selected) == p:
        best_complete_obj = current_obj
        best_complete_hubs = sorted(selected)
        if logger:
            logger.log(best_complete_obj)

    # Local swap search (first-improvement), time permitting
    improved = True
    while improved and (time.time() - start < time_limit):
        improved = False
        current_set = set(selected)
        non_selected = [v for v in all_nodes if v not in current_set]

        for u in list(current_set):
            if time.time() - start >= time_limit:
                break

            # remove contribution of u
            rem = provider.get_pair_vector(u, u).copy()
            for s in current_set:
                if s == u:
                    continue
                a, b = (u, s) if u < s else (s, u)
                rem += provider.get_pair_vector(a, b)

            base_vec = entrant_vec - rem
            set_without_u = set(current_set)
            set_without_u.remove(u)

            found_move = False
            for v in non_selected:
                if time.time() - start >= time_limit:
                    break

                add = provider.get_pair_vector(v, v).copy()
                for s in set_without_u:
                    a, b = (v, s) if v < s else (s, v)
                    add += provider.get_pair_vector(a, b)

                cand_vec = base_vec + add
                cand_obj = compute_objective(cand_vec, competitor_vec, demand_vec, pos_idx)

                if cand_obj > current_obj + 1e-9:
                    # apply improving swap
                    selected.remove(u)
                    selected.add(v)
                    entrant_vec = cand_vec
                    current_obj = cand_obj
                    improved = True
                    found_move = True

                    if current_obj > best_complete_obj + 1e-9:
                        best_complete_obj = current_obj
                        best_complete_hubs = sorted(selected)
                        if logger:
                            logger.log(best_complete_obj)
                    break

            if found_move:
                break

    if best_complete_hubs is None:
        best_complete_hubs = sorted(selected)[:p]
        best_complete_obj = compute_objective(entrant_vec, competitor_vec, demand_vec, pos_idx)

    return {
        "objective_value": float(best_complete_obj),
        "hub_locations": [int(x) for x in best_complete_hubs]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    sol = solve(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()