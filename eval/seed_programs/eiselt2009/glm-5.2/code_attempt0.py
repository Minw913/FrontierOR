import json
import time
import argparse
import numpy as np
from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    data = load_instance(args.instance_path)
    n = data['n_nodes']
    p = data['p']
    # competitor hubs
    competitor_hubs = data['competitor_hub_locations']

    demand = np.array(data['demand_matrix'], dtype=np.float64)
    D = np.array(data['distance_matrix'], dtype=np.float64)
    W = np.array([node['waiting_time'] for node in data['nodes']], dtype=np.float64)
    W_col = W[:, None]

    params = data['parameters']
    alpha = params['alpha']
    gamma = params['gamma']
    beta = params['beta']
    lam = params['lambda']
    cost_scalar = params['cost_scalar']
    attr_2hub = params['attractiveness_2hub']
    attr_1hub = params['attractiveness_1hub']

    # Precompute per-hub arrays for fast U2 computation
    Dk_cols = [D[:, k] for k in range(n)]
    Dk_rows = [D[k, :] for k in range(n)]
    Wk_arr = [W[k] for k in range(n)]

    # Precompute U1 for all hubs
    U1 = np.zeros((n, n, n), dtype=np.float64)
    for k in range(n):
        Wk = Wk_arr[k]
        Dk_col = Dk_cols[k]
        Dk_row = Dk_rows[k]
        time = W_col + Dk_col[:, None] + 2.0 * Wk + Dk_row[None, :]
        fare = cost_scalar * (Dk_col[:, None] + Dk_row[None, :])
        denom = gamma * np.power(time, beta) + (1.0 - gamma) * np.power(fare, lam)
        denom = np.maximum(denom, 1e-12)
        U1[k] = attr_1hub / denom

    # Cache for U2 matrices
    u2_cache = {}
    MAX_CACHE = 3000

    def compute_U2(k, l):
        Wk = Wk_arr[k]
        Wl = Wk_arr[l]
        Dkl = D[k, l]
        Dk_col = Dk_cols[k]
        Dl_col = Dk_cols[l]
        Dk_row = Dk_rows[k]
        Dl_row = Dk_rows[l]

        # route i -> k -> l -> j
        time1 = W_col + Wk + Wl + Dkl + Dk_col[:, None] + Dl_row[None, :]
        fare1 = cost_scalar * (Dk_col[:, None] + alpha * Dkl + Dl_row[None, :])
        denom1 = gamma * np.power(time1, beta) + (1.0 - gamma) * np.power(fare1, lam)
        util1 = attr_2hub / np.maximum(denom1, 1e-12)

        # route i -> l -> k -> j
        time2 = W_col + Wk + Wl + Dkl + Dl_col[:, None] + Dk_row[None, :]
        fare2 = cost_scalar * (Dl_col[:, None] + alpha * Dkl + Dk_row[None, :])
        denom2 = gamma * np.power(time2, beta) + (1.0 - gamma) * np.power(fare2, lam)
        util2 = attr_2hub / np.maximum(denom2, 1e-12)

        return np.maximum(util1, util2)

    def get_U2(k, l):
        key = (min(k, l), max(k, l))
        if key in u2_cache:
            return u2_cache[key]
        res = compute_U2(k, l)
        if len(u2_cache) >= MAX_CACHE:
            u2_cache.clear()
        u2_cache[key] = res
        return res

    # Incumbent utilities
    U_inc = np.zeros((n, n), dtype=np.float64)
    for k in competitor_hubs:
        U_inc += U1[k]
    for idx in range(len(competitor_hubs)):
        k = competitor_hubs[idx]
        for jdx in range(idx + 1, len(competitor_hubs)):
            l = competitor_hubs[jdx]
            U_inc += get_U2(k, l)

    def calc_objective(U_ent):
        total = U_ent + U_inc
        prob = U_ent / (total + 1e-12)
        return float(np.sum(demand * prob))

    start_time = time.time()
    time_limit = args.time_limit

    # Handle trivial cases
    if p <= 0:
        obj = 0.0
        solution = {'objective_value': obj, 'hub_locations': []}
        if logger:
            logger.log_solution(obj, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    if p >= n:
        H = list(range(n))
        U_ent = np.zeros((n, n), dtype=np.float64)
        for k in H:
            U_ent += U1[k]
        for idx in range(len(H)):
            k = H[idx]
            for jdx in range(idx + 1, len(H)):
                l = H[jdx]
                U_ent += get_U2(k, l)
        obj = calc_objective(U_ent)
        solution = {'objective_value': obj, 'hub_locations': H}
        if logger:
            logger.log_solution(obj, solution)
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        return

    # ------------------ Greedy initialization ------------------
    H = []
    U_ent = np.zeros((n, n), dtype=np.float64)
    for step in range(p):
        if time.time() - start_time > time_limit * 0.5:
            # If time is running out, fill with random remaining hubs
            remaining = [c for c in range(n) if c not in H]
            if remaining:
                H.extend(remaining[:p - len(H)])
            break

        best_c = -1
        best_obj = -1.0
        for c in range(n):
            if c in H:
                continue
            U_c = U_ent + U1[c]
            for k in H:
                U_c += get_U2(k, c)
            obj = calc_objective(U_c)
            if obj > best_obj:
                best_obj = obj
                best_c = c

        if best_c == -1:
            remaining = [c for c in range(n) if c not in H]
            if remaining:
                H.extend(remaining[:p - len(H)])
            break

        U_ent += U1[best_c]
        for k in H:
            U_ent += get_U2(k, best_c)
        H.append(best_c)
        if logger:
            logger.log_solution(best_obj, {'objective_value': best_obj, 'hub_locations': H.copy()})

    # Ensure exactly p hubs (in case greedy was interrupted)
    if len(H) < p:
        remaining = [c for c in range(n) if c not in H]
        H.extend(remaining[:p - len(H)])
        # Recompute U_ent from scratch
        U_ent = np.zeros((n, n), dtype=np.float64)
        for k in H:
            U_ent += U1[k]
        for idx in range(len(H)):
            k = H[idx]
            for jdx in range(idx + 1, len(H)):
                l = H[jdx]
                U_ent += get_U2(k, l)

    current_obj = calc_objective(U_ent)
    best_H = H.copy()
    best_obj = current_obj
    if logger:
        logger.log_solution(best_obj, {'objective_value': best_obj, 'hub_locations': best_H})

    # ------------------ Local search ------------------
    def local_search(H, U_ent):
        improved = True
        local_best_obj = calc_objective(U_ent)
        local_best_H = H.copy()
        local_best_U = U_ent.copy()
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Precompute T[c] = sum_{k in H, k!=c} U2(c,k) for all c
            T = [None] * n
            # We can compute by iterating over k in H and adding to T[c]
            for c in range(n):
                T[c] = np.zeros((n, n), dtype=np.float64)
            for k in H:
                for c in range(n):
                    if c != k:
                        T[c] += get_U2(c, k)

            current_obj = calc_objective(U_ent)
            best_swap = None
            best_obj_local = current_obj

            for i in H:
                for j in range(n):
                    if j in H:
                        continue
                    if time.time() - start_time > time_limit:
                        break
                    U_new = U_ent - U1[i] - T[i] + U1[j] + T[j] - get_U2(j, i)
                    obj = calc_objective(U_new)
                    if obj > best_obj_local + 1e-9:
                        best_obj_local = obj
                        best_swap = (i, j)
                if time.time() - start_time > time_limit:
                    break

            if best_swap:
                i, j = best_swap
                U_ent = U_ent - U1[i] - T[i] + U1[j] + T[j] - get_U2(j, i)
                H.remove(i)
                H.append(j)
                improved = True
                if best_obj_local > local_best_obj:
                    local_best_obj = best_obj_local
                    local_best_H = H.copy()
                    local_best_U = U_ent.copy()
                    if logger:
                        logger.log_solution(local_best_obj, {'objective_value': local_best_obj, 'hub_locations': local_best_H})
            if time.time() - start_time > time_limit:
                break
        return local_best_H, local_best_U, local_best_obj

    H, U_ent, current_obj = local_search(H, U_ent)
    if current_obj > best_obj:
        best_obj = current_obj
        best_H = H.copy()
        if logger:
            logger.log_solution(best_obj, {'objective_value': best_obj, 'hub_locations': best_H})

    # ------------------ Perturbation / random restart ------------------
    np.random.seed(0)
    while time.time() - start_time < time_limit:
        # Perturb the best solution
        perturb_H = best_H.copy()
        remove_count = max(1, int(0.2 * p))
        remove_count = min(remove_count, len(perturb_H) - 1)
        if remove_count > 0:
            remove_indices = np.random.choice(len(perturb_H), remove_count, replace=False)
            removed = {perturb_H[idx] for idx in remove_indices}
            perturb_H = [h for h in perturb_H if h not in removed]
        # Add random hubs
        available = [c for c in range(n) if c not in perturb_H]
        add_count = p - len(perturb_H)
        if add_count > 0 and available:
            add_indices = np.random.choice(len(available), min(add_count, len(available)), replace=False)
            perturb_H.extend([available[idx] for idx in add_indices])

        # Build U_ent for perturb_H
        U_pert = np.zeros((n, n), dtype=np.float64)
        for k in perturb_H:
            U_pert += U1[k]
        for idx in range(len(perturb_H)):
            k = perturb_H[idx]
            for jdx in range(idx + 1, len(perturb_H)):
                l = perturb_H[jdx]
                U_pert += get_U2(k, l)

        new_H, new_U, new_obj = local_search(perturb_H, U_pert)
        if new_obj > best_obj:
            best_obj = new_obj
            best_H = new_H.copy()
            if logger:
                logger.log_solution(best_obj, {'objective_value': best_obj, 'hub_locations': best_H})

    # Final output
    solution = {
        'objective_value': float(best_obj),
        'hub_locations': [int(h) for h in best_H]
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()