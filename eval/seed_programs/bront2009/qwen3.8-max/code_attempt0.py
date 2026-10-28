import json
import argparse
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description="Network Revenue Management via Column Generation LP")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    start_time = time.time()

    # ---- Load instance ----
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    legs_data = data['network']['legs']
    products_data = data['products']
    segments_data = data['segments']
    T = data['booking_horizon']['T']
    alpha = data.get('alpha', 1.0)
    lam = data['lambda']

    n_prod = len(products_data)
    n_legs = len(legs_data)
    n_seg = len(segments_data)

    if n_prod == 0:
        sol = {"objective_value": 0.0, "active_columns": []}
        with open(args.solution_path, 'w') as f:
            json.dump(sol, f)
        return

    # Legs
    leg_id_to_idx = {}
    capacities = np.zeros(n_legs)
    for i, leg in enumerate(legs_data):
        leg_id_to_idx[leg['leg_id']] = i
        capacities[i] = leg['capacity'] * alpha

    # Products
    prod_ids = [p['product_id'] for p in products_data]
    fares = np.array([p['fare'] for p in products_data], dtype=np.float64)

    # Incidence matrix
    A_inc = np.zeros((n_legs, n_prod), dtype=np.float64)
    for j, p in enumerate(products_data):
        for lid in p['legs_used']:
            if lid in leg_id_to_idx:
                A_inc[leg_id_to_idx[lid], j] = 1.0

    # Segments
    seg_lam = np.array([s['lambda_l'] for s in segments_data], dtype=np.float64)
    seg_cons = []
    seg_pref = []
    seg_v0 = []
    for s in segments_data:
        cons = set(s['consideration_set'])
        seg_cons.append(cons)
        pm = {}
        for i, pid in enumerate(s['consideration_set']):
            pm[pid] = s['preference_vector'][i]
        seg_pref.append(pm)
        seg_v0.append(s['no_purchase_preference'])

    # ---- Helper functions ----
    def choice_probs(offer_set):
        probs = np.zeros(n_prod)
        for k in range(n_seg):
            inter = offer_set & seg_cons[k]
            if not inter:
                continue
            denom = seg_v0[k] + sum(seg_pref[k][j] for j in inter)
            for j in inter:
                probs[j] += (seg_lam[k] / lam) * (seg_pref[k][j] / denom)
        return probs

    def eval_set(offer_set):
        p = choice_probs(offer_set)
        return float(fares @ p), A_inc @ p

    def pricing_val(offer_set, c):
        v = 0.0
        for k in range(n_seg):
            inter = offer_set & seg_cons[k]
            if not inter:
                continue
            num = sum(seg_pref[k][j] * c[j] for j in inter)
            den = seg_v0[k] + sum(seg_pref[k][j] for j in inter)
            v += seg_lam[k] * num / den
        return v

    # ---- Pricing: exact enumeration ----
    EXACT_THRESHOLD = 18

    def pricing_exact(c, pi0):
        # Precompute per-segment lookup tables
        seg_info = []
        for k in range(n_seg):
            cons_list = sorted(seg_cons[k])
            m = len(cons_list)
            size = 1 << m
            num_arr = np.zeros(size)
            den_arr = np.full(size, seg_v0[k])
            for b in range(1, size):
                lsb = b & (-b)
                idx = lsb.bit_length() - 1
                j = cons_list[idx]
                prev = b ^ lsb
                num_arr[b] = num_arr[prev] + seg_pref[k][j] * c[j]
                den_arr[b] = den_arr[prev] + seg_pref[k][j]
            gmask = 0
            for j in cons_list:
                gmask |= (1 << j)
            lmap = {}
            for lb in range(size):
                gb = 0
                for i in range(m):
                    if lb & (1 << i):
                        gb |= (1 << cons_list[i])
                lmap[gb] = lb
            seg_info.append((seg_lam[k], gmask, lmap, num_arr, den_arr))

        best_s, best_rc = None, 1e-9
        for mask in range(1, 1 << n_prod):
            v = 0.0
            for sl, gmask, lmap, num_arr, den_arr in seg_info:
                sub = mask & gmask
                if sub:
                    lb = lmap[sub]
                    v += sl * num_arr[lb] / den_arr[lb]
            rc = v - pi0
            if rc > best_rc:
                best_rc = rc
                best_s = frozenset(j for j in range(n_prod) if mask & (1 << j))
            if time.time() - start_time > args.time_limit - 3:
                break
        return best_s, best_rc

    # ---- Pricing: heuristic ----
    def pricing_heuristic(c, pi0):
        best_s, best_rc = None, 1e-9

        # 1. Revenue-ordered per segment
        for k in range(n_seg):
            prods = sorted(seg_cons[k], key=lambda j: c[j], reverse=True)
            current = set()
            for j in prods:
                if c[j] <= 0:
                    break
                current.add(j)
                fs = frozenset(current)
                v = pricing_val(fs, c)
                rc = v - pi0
                if rc > best_rc:
                    best_rc = rc
                    best_s = fs

        # 2. Greedy construction
        current = set()
        cur_val = 0.0
        for _ in range(n_prod):
            best_j = None
            best_j_val = cur_val
            for j in range(n_prod):
                if j not in current:
                    trial = frozenset(current | {j})
                    v = pricing_val(trial, c)
                    if v > best_j_val:
                        best_j_val = v
                        best_j = j
            if best_j is not None and best_j_val > cur_val:
                current.add(best_j)
                cur_val = best_j_val
                rc = cur_val - pi0
                if rc > best_rc:
                    best_rc = rc
                    best_s = frozenset(current)
            else:
                break

        # 3. Local search on best set found so far
        if best_s is not None:
            for _ in range(3):  # a few passes
                improved = False
                # Try removing
                for j in list(best_s):
                    if len(best_s) > 1:
                        trial = best_s - frozenset([j])
                        v = pricing_val(trial, c)
                        rc = v - pi0
                        if rc > best_rc:
                            best_rc = rc
                            best_s = trial
                            improved = True
                # Try adding
                for j in range(n_prod):
                    if j not in best_s:
                        trial = best_s | frozenset([j])
                        v = pricing_val(trial, c)
                        rc = v - pi0
                        if rc > best_rc:
                            best_rc = rc
                            best_s = trial
                            improved = True
                # Try swapping
                for j_out in list(best_s):
                    for j_in in range(n_prod):
                        if j_in not in best_s:
                            trial = (best_s - frozenset([j_out])) | frozenset([j_in])
                            v = pricing_val(trial, c)
                            rc = v - pi0
                            if rc > best_rc:
                                best_rc = rc
                                best_s = trial
                                improved = True
                if not improved:
                    break

        # 4. Singletons and pairs
        for j in range(n_prod):
            fs = frozenset([j])
            v = pricing_val(fs, c)
            rc = v - pi0
            if rc > best_rc:
                best_rc = rc
                best_s = fs
        if n_prod <= 60:
            for i in range(n_prod):
                for j in range(i + 1, n_prod):
                    fs = frozenset([i, j])
                    v = pricing_val(fs, c)
                    rc = v - pi0
                    if rc > best_rc:
                        best_rc = rc
                        best_s = fs

        return best_s, best_rc

    def solve_pricing(c, pi0):
        if n_prod <= EXACT_THRESHOLD:
            return pricing_exact(c, pi0)
        else:
            return pricing_heuristic(c, pi0)

    # ---- Column pool ----
    column_pool = {}  # frozenset -> (revenue, leg_consumption_array)

    def add_col(offer_set):
        if offer_set not in column_pool:
            rev, lc = eval_set(offer_set)
            column_pool[offer_set] = (rev, lc)

    # Initialize columns
    for j in range(n_prod):
        add_col(frozenset([j]))
    add_col(frozenset(range(n_prod)))

    # Add revenue-ordered assortments per segment (with c = fares, pi0 = 0)
    for k in range(n_seg):
        prods = sorted(seg_cons[k], key=lambda j: fares[j], reverse=True)
        current = set()
        for j in prods:
            if fares[j] <= 0:
                break
            current.add(j)
            add_col(frozenset(current))

    # Add some pairs
    if n_prod <= 40:
        for i in range(n_prod):
            for j in range(i + 1, n_prod):
                add_col(frozenset([i, j]))

    best_solution = None
    best_obj = -float('inf')

    # ---- Column Generation Loop ----
    iteration = 0
    while time.time() - start_time < args.time_limit - 2:
        iteration += 1

        cols = list(column_pool.items())
        n_cols = len(cols)

        # Build and solve master LP
        model = gp.Model("master")
        model.setParam('OutputFlag', 0)
        model.setParam('Seed', 0)
        model.setParam('Threads', 1)
        model.setParam('NumericFocus', 0)
        model.setParam('MIPGap', 1e-4)

        t_vars = model.addVars(n_cols, lb=0.0, name="t")

        obj = gp.quicksum(lam * cols[i][1][0] * t_vars[i] for i in range(n_cols))
        model.setObjective(obj, GRB.MAXIMIZE)

        cap_cons = []
        for l in range(n_legs):
            con = model.addConstr(
                gp.quicksum(lam * cols[i][1][1][l] * t_vars[i] for i in range(n_cols)) <= capacities[l],
                name=f"cap_{l}"
            )
            cap_cons.append(con)

        time_con = model.addConstr(
            gp.quicksum(t_vars[i] for i in range(n_cols)) <= T,
            name="time"
        )

        model.optimize()

        if model.status != GRB.OPTIMAL:
            break

        cur_obj = model.objVal

        # Build solution dict
        active = []
        for i in range(n_cols):
            tv = t_vars[i].x
            if tv > 1e-6:
                active.append({
                    "offer_set": sorted([prod_ids[j] for j in cols[i][0]]),
                    "time_allocated": float(tv)
                })

        sol_dict = {
            "objective_value": float(cur_obj),
            "active_columns": active
        }

        if cur_obj > best_obj + 1e-8:
            best_obj = cur_obj
            best_solution = sol_dict
            if logger:
                logger.log_solution(best_obj, best_solution)

        # Get duals
        duals_cap = np.array([cap_cons[l].pi for l in range(n_legs)])
        pi0 = time_con.pi

        # Net revenue per product
        c = fares - A_inc.T @ duals_cap

        # Pricing
        new_set, rc = solve_pricing(c, pi0)

        if new_set is not None and rc > 1e-6 and new_set not in column_pool:
            add_col(new_set)
        else:
            # Verify with exact pricing if we used heuristic
            if n_prod > EXACT_THRESHOLD and n_prod <= 22:
                new_set2, rc2 = pricing_exact(c, pi0)
                if new_set2 is not None and rc2 > 1e-6 and new_set2 not in column_pool:
                    add_col(new_set2)
                    continue
            break

    # Fallback
    if best_solution is None:
        best_solution = {"objective_value": 0.0, "active_columns": []}
        if logger:
            logger.log_solution(0.0, best_solution)

    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


if __name__ == "__main__":
    main()