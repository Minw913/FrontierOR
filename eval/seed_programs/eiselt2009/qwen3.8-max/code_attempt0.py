import json
import argparse
import time
import random
import itertools
import numpy as np
from math import comb
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path) as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    n = instance['n_nodes']
    p = instance['p']
    demand = np.array(instance['demand_matrix'], dtype=np.float64)
    dist = np.array(instance['distance_matrix'], dtype=np.float64)
    waiting = np.array([node['waiting_time'] for node in instance['nodes']])
    params = instance['parameters']
    alpha = params['alpha']
    gamma = params['gamma']
    beta = params['beta']
    lam = params['lambda']
    cost_scalar = params['cost_scalar']
    attr_1hub = params['attractiveness_1hub']
    attr_2hub = params['attractiveness_2hub']
    comp_hubs = instance['competitor_hub_locations']
    r = len(comp_hubs)

    def time_remaining():
        return args.time_limit - (time.time() - start_time)

    # Precompute route utility for a single (i, j, k, l) with k <= l
    def route_utility_scalar(i, j, k, l):
        if k == l:
            t = dist[i][k] + dist[k][j] + waiting[i] + 2.0 * waiting[k]
            c = cost_scalar * (dist[i][k] + dist[k][j])
            denom = gamma * (t ** beta) + (1.0 - gamma) * (c ** lam)
            return attr_1hub / denom if denom > 1e-300 else 0.0
        else:
            t1 = dist[i][k] + dist[k][l] + dist[l][j] + waiting[i] + waiting[k] + waiting[l]
            c1 = cost_scalar * (dist[i][k] + alpha * dist[k][l] + dist[l][j])
            denom1 = gamma * (t1 ** beta) + (1.0 - gamma) * (c1 ** lam)
            u1 = attr_2hub / denom1 if denom1 > 1e-300 else 0.0

            t2 = dist[i][l] + dist[l][k] + dist[k][j] + waiting[i] + waiting[l] + waiting[k]
            c2 = cost_scalar * (dist[i][l] + alpha * dist[l][k] + dist[k][j])
            denom2 = gamma * (t2 ** beta) + (1.0 - gamma) * (c2 ** lam)
            u2 = attr_2hub / denom2 if denom2 > 1e-300 else 0.0

            return max(u1, u2)

    # Precompute competitor utility for each OD pair
    comp_util = np.zeros((n, n))
    for k_idx in range(r):
        for l_idx in range(k_idx, r):
            k, l = comp_hubs[k_idx], comp_hubs[l_idx]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        comp_util[i][j] += route_utility_scalar(i, j, k, l)

    # Precompute U[(k,l)] matrices for entrant hub pairs
    use_precomputed = (n <= 80)
    U = {}
    if use_precomputed:
        for k in range(n):
            for l in range(k, n):
                U_kl = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            U_kl[i][j] = route_utility_scalar(i, j, k, l)
                U[(k, l)] = U_kl

    def evaluate_hubs(hub_set):
        """Evaluate total captured demand for a given hub set."""
        hub_list = sorted(hub_set)
        entrant_util = np.zeros((n, n))
        if use_precomputed:
            for k_idx in range(len(hub_list)):
                for l_idx in range(k_idx, len(hub_list)):
                    entrant_util += U[(hub_list[k_idx], hub_list[l_idx])]
        else:
            for k_idx in range(len(hub_list)):
                for l_idx in range(k_idx, len(hub_list)):
                    k, l = hub_list[k_idx], hub_list[l_idx]
                    for i in range(n):
                        for j in range(n):
                            if i != j:
                                entrant_util[i][j] += route_utility_scalar(i, j, k, l)
        total_util = comp_util + entrant_util
        mask = (total_util > 0) & (demand > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(demand[mask] * entrant_util[mask] / total_util[mask]))

    best_hubs = None
    best_val = -1.0

    def update_best(hubs, val):
        nonlocal best_hubs, best_val
        if val > best_val:
            best_hubs = sorted(hubs) if not isinstance(hubs, list) else sorted(hubs)
            best_val = val
            if logger:
                sol = {"objective_value": best_val, "hub_locations": list(best_hubs)}
                logger.log_solution(best_val, sol)

    # Handle edge cases
    if p == 0:
        best_hubs = []
        best_val = 0.0
    elif p >= n:
        best_hubs = list(range(n))
        best_val = evaluate_hubs(set(best_hubs))
        update_best(best_hubs, best_val)
    else:
        total_combs = comb(n, p)

        if total_combs <= 80000 and time_remaining() > 10:
            # Exhaustive enumeration for small instances
            for hubs in itertools.combinations(range(n), p):
                if time_remaining() < 2:
                    break
                val = evaluate_hubs(set(hubs))
                if val > best_val:
                    update_best(set(hubs), val)
        else:
            # Greedy construction
            def greedy_construct(randomize=False):
                selected = set()
                remaining = list(range(n))
                for step in range(p):
                    if time_remaining() < 3:
                        # Fill remaining randomly
                        needed = p - step
                        selected.update(random.sample(remaining, needed))
                        break
                    if randomize and step == 0:
                        # Random first hub for diversification
                        node = random.choice(remaining)
                        selected.add(node)
                        remaining.remove(node)
                        continue
                    best_node = -1
                    best_node_val = -1.0
                    for node in remaining:
                        candidate = selected | {node}
                        val = evaluate_hubs(candidate)
                        if val > best_node_val:
                            best_node_val = val
                            best_node = node
                    selected.add(best_node)
                    remaining.remove(best_node)
                return selected

            # Local search with best-improvement swap
            def local_search(hub_set, time_budget=None):
                current = set(hub_set)
                current_val = evaluate_hubs(current)
                improved = True
                while improved:
                    if time_remaining() < (time_budget if time_budget else 2):
                        break
                    improved = False
                    best_swap_val = current_val
                    best_swap = None
                    current_sorted = sorted(current)
                    for h_out in current_sorted:
                        for h_in in range(n):
                            if h_in in current:
                                continue
                            candidate = (current - {h_out}) | {h_in}
                            val = evaluate_hubs(candidate)
                            if val > best_swap_val:
                                best_swap_val = val
                                best_swap = (h_out, h_in)
                    if best_swap is not None:
                        h_out, h_in = best_swap
                        current = (current - {h_out}) | {h_in}
                        current_val = best_swap_val
                        improved = True
                return current, current_val

            # Run greedy + local search
            greedy_hubs = greedy_construct()
            greedy_val = evaluate_hubs(greedy_hubs)
            update_best(greedy_hubs, greedy_val)

            ls_hubs, ls_val = local_search(greedy_hubs)
            update_best(ls_hubs, ls_val)

            # Iterated local search with random restarts
            iteration = 0
            while time_remaining() > 8:
                iteration += 1
                if iteration % 3 == 0:
                    # Perturbation: swap 2 hubs from best solution
                    perturbed = set(best_hubs)
                    if len(perturbed) >= 2 and time_remaining() > 8:
                        to_remove = random.sample(sorted(perturbed), min(2, len(perturbed)))
                        for h in to_remove:
                            perturbed.remove(h)
                        candidates = [x for x in range(n) if x not in perturbed]
                        to_add = random.sample(candidates, len(to_remove))
                        for h in to_add:
                            perturbed.add(h)
                        ls_hubs, ls_val = local_search(perturbed, time_budget=5)
                        update_best(ls_hubs, ls_val)
                else:
                    # Random restart
                    random_hubs = set(random.sample(range(n), p))
                    ls_hubs, ls_val = local_search(random_hubs, time_budget=5)
                    update_best(ls_hubs, ls_val)

    # Ensure valid solution
    if best_hubs is None:
        best_hubs = sorted(random.sample(range(n), min(p, n)))
        best_val = evaluate_hubs(set(best_hubs))

    solution = {
        "objective_value": float(best_val),
        "hub_locations": sorted(list(best_hubs))
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()