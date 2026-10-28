#!/usr/bin/env python3
import argparse
import json
import math
import random
import time

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

from solution_logger import SolutionLogger


class StochasticHubSolver:
    TOL = 1e-9

    def __init__(self, instance, time_limit, logger):
        self.data = instance
        self.logger = logger
        self.start_time = time.time()
        self.time_limit = max(0, int(time_limit))
        reserve = min(5.0, max(0.2, 0.08 * float(self.time_limit))) if self.time_limit > 0 else 0.0
        self.opt_deadline = self.start_time + max(0.0, float(self.time_limit) - reserve)

        self.n = int(instance["n"])
        self.S = int(instance["num_scenarios"])
        cp = instance["cost_parameters"]
        self.chi = float(cp["chi"])
        self.alpha = float(cp["alpha"])
        self.delta = float(cp["delta"])

        self.p_hubs = instance.get("p_hubs", None)
        self.is_p = self.p_hubs is not None
        self.p = int(self.p_hubs) if self.is_p else None

        self.dist = np.array(instance["distances"], dtype=float)
        np.fill_diagonal(self.dist, 0.0)

        self.fixed = np.array(instance.get("fixed_costs") or [0.0] * self.n, dtype=float)
        cap = instance.get("hub_capacities", None)
        self.cap = None if cap is None else np.array(cap, dtype=float)

        scenarios = instance["scenarios"]
        self.prob = np.array([float(sc["probability"]) for sc in scenarios], dtype=float)
        self.W = [np.array(sc["demands"], dtype=float) for sc in scenarios]
        self.O = np.vstack([w.sum(axis=1) for w in self.W]) if self.S else np.zeros((0, self.n))
        self.I = np.vstack([w.sum(axis=0) for w in self.W]) if self.S else np.zeros((0, self.n))
        self.coeff = self.chi * self.O + self.delta * self.I
        self.total_out = self.O.sum(axis=1) if self.S else np.zeros(0)

        self.allowed = np.ones(self.n, dtype=bool)
        if self.cap is not None:
            for k in range(self.n):
                if np.any(self.O[:, k] > self.cap[k] + self.TOL):
                    self.allowed[k] = False
            if not np.any(self.allowed):
                self.allowed[:] = True

        iu = np.triu_indices(self.n, 1)
        self.pair_data = []
        pos_pairs = 0
        for w in self.W:
            b = w + w.T
            bv = b[iu]
            mask = bv > 1e-12
            ii = iu[0][mask].astype(np.int64)
            jj = iu[1][mask].astype(np.int64)
            vv = bv[mask].astype(float)
            pos_pairs += int(vv.size)
            self.pair_data.append((ii, jj, vv))
        self.hp_pairs = [
            (k, l, float(self.dist[k, l]))
            for k in range(self.n)
            for l in range(k + 1, self.n)
            if self.dist[k, l] > 1e-12
        ]

        self.avg_coeff = (self.prob @ self.coeff) if self.S else np.zeros(self.n)
        self.avg_unary = self.avg_coeff[:, None] * self.dist

        y_vars = self.S * self.n * max(0, self.n - 1)
        trans_cons = pos_pairs * len(self.hp_pairs) * 2 if self.alpha != 0.0 else 0
        assign_cons = self.S * self.n + y_vars
        est_cons = trans_cons + assign_cons
        self.use_mip = False
        if gp is not None and self.time_limit > 0:
            if trans_cons == 0:
                self.use_mip = y_vars <= 90000
            else:
                self.use_mip = (est_cons <= 750000) and (y_vars <= 28000)

        random.seed(0)
        self.best_obj = float("inf")
        self.best_H = None
        self.best_sol = None
        self.last_logged = float("inf")

    def remaining(self):
        return max(0.0, self.opt_deadline - time.time())

    def time_up(self):
        return time.time() >= self.opt_deadline

    def fixed_term(self, H):
        if self.is_p:
            return 0.0
        return float(np.sum(self.fixed[list(H)])) if H else 0.0

    def quick_capacity_possible(self, H):
        if self.cap is None:
            return True
        H = list(H)
        if any(not self.allowed[h] for h in H):
            return False
        total_cap = float(np.sum(self.cap[H])) if H else 0.0
        for s in range(self.S):
            if total_cap + self.TOL < self.total_out[s]:
                return False
        return True

    def feasible_H(self, H):
        H = list(H)
        if not H:
            return False
        if self.cap is None:
            return True
        if not self.quick_capacity_possible(H):
            return False
        for s in range(self.S):
            if self.initial_assignment(s, H) is None:
                return False
        return True

    def initial_assignment(self, s, H):
        H = sorted(int(h) for h in H)
        n = self.n
        if not H:
            return None if n else np.zeros(0, dtype=int)
        Ha = np.array(H, dtype=int)
        hub_set = set(H)
        h_arr = np.full(n, -1, dtype=int)
        for h in H:
            h_arr[h] = h
        non = [i for i in range(n) if i not in hub_set]
        if not non:
            return h_arr

        if self.cap is None:
            for i in non:
                costs = self.coeff[s, i] * self.dist[i, Ha]
                h_arr[i] = int(Ha[int(np.argmin(costs))])
            return h_arr

        O = self.O[s]
        capH = self.cap[Ha]
        base_load = O[Ha].astype(float)
        if np.any(base_load > capH + self.TOL):
            return None

        orders = []
        order_o = sorted(non, key=lambda i: -O[i])
        orders.append(order_o)
        orders.append(sorted(non, key=lambda i: -(O[i] + self.I[s, i])))
        if len(H) >= 2:
            cm = self.dist[np.ix_(non, Ha)] * self.coeff[s, non][:, None]
            sc = np.sort(cm, axis=1)
            regret = sc[:, 1] - sc[:, 0]
            orders.append([non[idx] for idx in np.argsort(-regret)])

        for order in orders:
            loads = base_load.copy()
            assign = {}
            ok = True
            for i in order:
                feas = np.where(loads + O[i] <= capH + self.TOL)[0]
                if feas.size == 0:
                    ok = False
                    break
                c = self.coeff[s, i] * self.dist[i, Ha[feas]]
                res_after = capH[feas] - (loads[feas] + O[i])
                idx = np.lexsort((-res_after, c))[0]
                q = int(feas[idx])
                assign[i] = int(Ha[q])
                loads[q] += O[i]
            if ok:
                for i, h in assign.items():
                    h_arr[i] = h
                return h_arr

        loads = base_load.copy()
        assign = {}
        for i in order_o:
            feas = np.where(loads + O[i] <= capH + self.TOL)[0]
            if feas.size == 0:
                return None
            res_after = capH[feas] - (loads[feas] + O[i])
            q = int(feas[int(np.argmax(res_after))])
            assign[i] = int(Ha[q])
            loads[q] += O[i]
        for i, h in assign.items():
            h_arr[i] = h
        return h_arr

    def compute_scenario_cost(self, s, h_arr):
        h_arr = np.asarray(h_arr, dtype=int)
        coll = float(np.sum(self.coeff[s] * self.dist[np.arange(self.n), h_arr]))
        trans = 0.0
        if self.alpha != 0.0 and self.S:
            dh = self.dist[np.ix_(h_arr, h_arr)]
            trans = self.alpha * float(np.sum(self.W[s] * dh))
        return coll + trans

    def local_search_scenario(self, s, h_arr, H, node_order=None, max_passes=30):
        H = sorted(int(h) for h in H)
        if len(H) <= 1:
            return np.asarray(h_arr, dtype=int).copy()
        h_arr = np.asarray(h_arr, dtype=int).copy()
        Ha = np.array(H, dtype=int)
        pos = {h: idx for idx, h in enumerate(H)}
        hub_set = set(H)
        W = self.W[s]
        O = self.O[s]
        coeff_s = self.coeff[s]
        non = [i for i in range(self.n) if i not in hub_set]
        if node_order is None:
            order = sorted(non, key=lambda i: -(O[i] + self.I[s, i]))
        else:
            order = [i for i in node_order if i not in hub_set]

        if self.cap is not None:
            loads = np.zeros(len(H), dtype=float)
            for i in range(self.n):
                loads[pos[int(h_arr[i])]] += O[i]
            capH = self.cap[Ha]
        else:
            loads = None
            capH = None

        arange_n = np.arange(self.n)
        for _ in range(max(1, int(max_passes))):
            changed = False
            for r in order:
                if self.time_up():
                    return h_arr
                a = int(h_arr[r])
                pa = pos[a]
                row = W[r] + W[:, r]
                row[r] = 0.0
                base = self.dist[a, h_arr]
                d_bh = self.dist[np.ix_(Ha, h_arr)]
                trans_from = float(base @ row)
                trans_to = d_bh @ row
                unary_to = coeff_s[r] * self.dist[r, Ha]
                unary_from = coeff_s[r] * self.dist[r, a]
                delta = (unary_to - unary_from) + self.alpha * (trans_to - trans_from)
                delta[pa] = np.inf
                if self.cap is not None:
                    feas = (loads + O[r] <= capH + self.TOL)
                    delta[~feas] = np.inf
                q = int(np.argmin(delta))
                if not np.isfinite(delta[q]) or delta[q] >= -1e-10:
                    continue
                b = int(Ha[q])
                h_arr[r] = b
                if self.cap is not None:
                    loads[pa] -= O[r]
                    loads[q] += O[r]
                changed = True
            if not changed or self.time_up():
                break
        return h_arr

    def evaluate_hubs(self, H, local=True, max_passes=30, polish=False):
        H = sorted(set(int(h) for h in H))
        if not H:
            return float("inf"), None
        allocs = []
        total = self.fixed_term(H)
        for s in range(self.S):
            if self.time_up():
                return float("inf"), None
            h = self.initial_assignment(s, H)
            if h is None:
                return float("inf"), None
            if local and len(H) > 1:
                h = self.local_search_scenario(s, h, H, max_passes=max_passes)
                if polish and not self.time_up():
                    best_h = h
                    best_c = self.compute_scenario_cost(s, best_h)
                    non = [i for i in range(self.n) if i not in set(H)]
                    for _ in range(2):
                        if self.time_up():
                            break
                        order = non[:]
                        random.shuffle(order)
                        h2 = self.local_search_scenario(
                            s, best_h, H, node_order=order, max_passes=max_passes
                        )
                        c2 = self.compute_scenario_cost(s, h2)
                        if c2 < best_c - 1e-9:
                            best_c = c2
                            best_h = h2
                    h = best_h
            allocs.append(h)
            total += float(self.prob[s]) * self.compute_scenario_cost(s, h)
        return float(total), allocs

    def best_candidate_by_proxy(self, H, cur, fixed_sum, require_feasible=False, improvement_over=None):
        hub_set = set(H)
        cands = []
        for r in range(self.n):
            if not self.allowed[r] or r in hub_set:
                continue
            if cur is None:
                ns = float(np.sum(self.avg_unary[:, r]))
            else:
                ns = float(np.sum(np.minimum(cur, self.avg_unary[:, r])))
            proxy = ns if self.is_p else fixed_sum + float(self.fixed[r]) + ns
            cands.append((proxy, r))
        cands.sort(key=lambda x: (x[0], x[1]))
        for proxy, r in cands:
            if improvement_over is not None and proxy >= improvement_over - 1e-10:
                break
            if require_feasible:
                H2 = H + [r]
                if not self.quick_capacity_possible(H2) or not self.feasible_H(H2):
                    continue
            if cur is None:
                newcur = self.avg_unary[:, r].copy()
            else:
                newcur = np.minimum(cur, self.avg_unary[:, r])
            return r, newcur, proxy
        return None

    def repair_capacity(self, H, target):
        if self.cap is None:
            return H
        H = sorted(set(int(h) for h in H if self.allowed[int(h)]))
        allowed = [i for i in range(self.n) if self.allowed[i]]
        if target is None:
            guard = 0
            while not self.feasible_H(H) and len(H) < len(allowed) and guard <= self.n:
                guard += 1
                outside = [r for r in allowed if r not in set(H)]
                if not outside:
                    break
                outside.sort(key=lambda r: (-self.cap[r], self.fixed[r], float(np.sum(self.avg_unary[:, r]))))
                H.append(outside[0])
            return sorted(set(H))

        while len(H) < target:
            outside = [r for r in allowed if r not in set(H)]
            if not outside:
                break
            outside.sort(key=lambda r: (-self.cap[r], float(np.sum(self.avg_unary[:, r]))))
            H.append(outside[0])
        guard = 0
        while not self.feasible_H(H) and guard < 2 * self.n:
            guard += 1
            hub_set = set(H)
            outside = [r for r in allowed if r not in hub_set]
            inside = sorted(H, key=lambda h: (self.cap[h], -self.fixed[h]))
            outside.sort(key=lambda r: (-self.cap[r], self.fixed[r]))
            swapped = False
            cur_cap = float(np.sum(self.cap[H])) if H else 0.0
            for r in outside:
                for h in inside:
                    if self.cap[r] <= self.cap[h] + self.TOL:
                        continue
                    H2 = sorted([x for x in H if x != h] + [r])
                    if not self.quick_capacity_possible(H2):
                        continue
                    new_cap = float(np.sum(self.cap[H2]))
                    if new_cap > cur_cap + self.TOL or self.feasible_H(H2):
                        H = H2
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                break
        return sorted(set(H))

    def greedy_construct(self):
        target = self.p if self.is_p else None
        H = []
        cur = None
        fixed_sum = 0.0
        current_proxy = None
        guard = 0
        while guard <= self.n + 2:
            guard += 1
            if target is not None and len(H) >= target:
                break
            if target is None and cur is not None:
                if self.cap is None:
                    cand = self.best_candidate_by_proxy(
                        H, cur, fixed_sum, require_feasible=False, improvement_over=current_proxy
                    )
                    if cand is None:
                        break
                else:
                    if self.feasible_H(H):
                        cand = self.best_candidate_by_proxy(
                            H, cur, fixed_sum, require_feasible=True, improvement_over=current_proxy
                        )
                        if cand is None:
                            break
                    else:
                        cand = self.best_candidate_by_proxy(
                            H, cur, fixed_sum, require_feasible=False, improvement_over=None
                        )
                        if cand is None:
                            break
            else:
                cand = self.best_candidate_by_proxy(
                    H, cur, fixed_sum, require_feasible=False, improvement_over=None
                )
                if cand is None:
                    break
            r, newcur, _ = cand
            H.append(int(r))
            cur = newcur
            if not self.is_p:
                fixed_sum += float(self.fixed[r])
            current_proxy = float(np.sum(cur)) if self.is_p else fixed_sum + float(np.sum(cur))

        if target is not None and len(H) < target:
            hub_set = set(H)
            rest = [r for r in range(self.n) if self.allowed[r] and r not in hub_set]
            rest.sort(key=lambda r: float(np.sum(self.avg_unary[:, r])))
            while len(H) < target and rest:
                H.append(rest.pop(0))
        if not H:
            first = int(np.argmax(self.allowed)) if np.any(self.allowed) else 0
            H = [first]
        if self.cap is not None:
            H = self.repair_capacity(H, target)
        return sorted(set(int(h) for h in H))

    def total_objective(self, H, allocs):
        total = self.fixed_term(H)
        for s in range(self.S):
            total += float(self.prob[s]) * self.compute_scenario_cost(s, allocs[s])
        return float(total)

    def normalize_allocations(self, H, allocs):
        H = sorted(set(int(h) for h in H))
        if not H:
            H = [0]
        hub_set = set(H)
        Ha = np.array(H, dtype=int)
        out = []
        for s in range(self.S):
            h = np.asarray(allocs[s], dtype=int).copy()
            if h.size != self.n:
                h = np.arange(self.n, dtype=int)
                for i in range(self.n):
                    if i not in hub_set:
                        h[i] = int(Ha[int(np.argmin(self.dist[i, Ha]))])
            for hub in H:
                h[hub] = hub
            for i in range(self.n):
                if i in hub_set:
                    h[i] = i
                elif int(h[i]) not in hub_set:
                    h[i] = int(Ha[int(np.argmin(self.dist[i, Ha]))])
            out.append(h)
        return H, out

    def build_solution_dict(self, H, allocs):
        H, allocs = self.normalize_allocations(H, allocs)
        obj = self.total_objective(H, allocs)
        hub_set = set(H)
        sol = {
            "objective_value": float(obj),
            "hubs": [int(h) for h in H],
            "allocations": {},
            "z_values": [1.0 if i in hub_set else 0.0 for i in range(self.n)],
            "x_values": {},
        }
        n = self.n
        for s in range(self.S):
            alloc_s = {}
            x_s = {}
            h_arr = allocs[s]
            for i in range(n):
                ai = int(h_arr[i])
                alloc_s[str(i)] = ai
                row = {}
                # Schema requests every candidate hub except the origin itself.
                for k in range(n):
                    if k == i:
                        continue
                    row[str(k)] = 1.0 if k == ai else 0.0
                x_s[str(i)] = row
            sol["allocations"][str(s)] = alloc_s
            sol["x_values"][str(s)] = x_s
        return sol

    def set_best(self, H, allocs, log=False):
        sol = self.build_solution_dict(H, allocs)
        val = float(sol["objective_value"])
        improved = val < self.best_obj - max(1e-9, abs(self.best_obj) * 1e-11)
        if self.best_sol is None or improved:
            self.best_obj = val
            self.best_H = [int(h) for h in sol["hubs"]]
            self.best_sol = sol
            if log and self.logger is not None and val < self.last_logged - max(1e-9, abs(self.last_logged) * 1e-11):
                self.logger.log_solution(val, sol)
                self.last_logged = val
        return val

    def nearest_alloc(self, H):
        H = sorted(set(int(h) for h in H)) or [0]
        Ha = np.array(H, dtype=int)
        hub_set = set(H)
        h = np.arange(self.n, dtype=int)
        for i in range(self.n):
            if i in hub_set:
                h[i] = i
            else:
                h[i] = int(Ha[int(np.argmin(self.dist[i, Ha]))])
        return h

    def emergency_solution(self):
        if self.is_p:
            allowed = [i for i in range(self.n) if self.allowed[i]] or list(range(self.n))
            H = allowed[: max(1, min(self.p, len(allowed)))]
            while len(H) < max(1, min(self.p, self.n)):
                for r in range(self.n):
                    if r not in H:
                        H.append(r)
                        break
        else:
            first = int(np.argmax(self.allowed)) if np.any(self.allowed) else 0
            H = [first]
        allocs = []
        for s in range(self.S):
            h = self.initial_assignment(s, H)
            if h is None:
                h = self.nearest_alloc(H)
            allocs.append(h)
        return self.build_solution_dict(H, allocs)

    def improve_heuristic(self):
        if self.best_H is None or self.time_up():
            return
        n = self.n
        if n <= 60:
            K = 40
            max_out = 10 ** 9
        elif n <= 150:
            K = 24
            max_out = 90
        elif n <= 400:
            K = 12
            max_out = 50
        else:
            K = 6
            max_out = 28

        for _ in range(8):
            if self.time_up():
                break
            H = list(self.best_H)
            m = len(H)
            hub_set = set(H)
            costs = self.avg_unary[:, H]
            best_idx = np.argmin(costs, axis=1)
            best_val = costs[np.arange(n), best_idx]
            if m > 1:
                second_val = np.partition(costs, 1, axis=1)[:, 1]
            else:
                second_val = np.full(n, np.inf)
            fixed_sum = self.fixed_term(H)

            rem = []
            for ph, h in enumerate(H):
                mask = best_idx == ph
                inc = float(np.sum(second_val[mask] - best_val[mask])) if np.any(mask) else 0.0
                if not self.is_p:
                    inc -= float(self.fixed[h])
                rem.append((inc, ph, h))
            rem.sort(key=lambda x: (x[0], x[2]))
            max_in = len(rem) if len(rem) <= 80 else 80
            inside = [(ph, h) for _, ph, h in rem[:max_in]]

            outside = [r for r in range(n) if self.allowed[r] and r not in hub_set]
            if len(outside) > max_out:
                scored = [
                    (float(np.sum(self.avg_unary[:, r]) + (0.0 if self.is_p else self.fixed[r])), r)
                    for r in outside
                ]
                scored.sort()
                chosen = [r for _, r in scored[:max_out]]
                rest = [r for _, r in scored[max_out:]]
                if rest:
                    chosen.extend(random.sample(rest, min(8, len(rest))))
                outside = chosen

            cand = []
            for ph, h in inside:
                mask = best_idx == ph
                for r in outside:
                    newval = np.minimum(best_val, self.avg_unary[:, r])
                    if np.any(mask):
                        newval[mask] = np.minimum(second_val[mask], self.avg_unary[mask, r])
                    proxy = float(np.sum(newval))
                    if not self.is_p:
                        proxy += fixed_sum - float(self.fixed[h]) + float(self.fixed[r])
                    H2 = sorted([x for x in H if x != h] + [r])
                    if self.is_p and len(H2) != self.p:
                        continue
                    if self.cap is not None and not self.quick_capacity_possible(H2):
                        continue
                    cand.append((proxy, "swap", h, r, H2))
            if not self.is_p:
                for r in outside:
                    newval = np.minimum(best_val, self.avg_unary[:, r])
                    proxy = fixed_sum + float(self.fixed[r]) + float(np.sum(newval))
                    H2 = sorted(H + [r])
                    if self.cap is not None and not self.quick_capacity_possible(H2):
                        continue
                    cand.append((proxy, "add", None, r, H2))
                if m > 1:
                    for ph, h in inside:
                        mask = best_idx == ph
                        newval = np.where(mask, second_val, best_val)
                        proxy = fixed_sum - float(self.fixed[h]) + float(np.sum(newval))
                        H2 = sorted([x for x in H if x != h])
                        if not H2:
                            continue
                        if self.cap is not None and not self.quick_capacity_possible(H2):
                            continue
                        cand.append((proxy, "remove", h, None, H2))
            if not cand:
                break
            cand.sort(key=lambda x: x[0])
            accepted = False
            for _, _, _, _, H2 in cand[:K]:
                if self.time_up():
                    break
                obj, allocs = self.evaluate_hubs(H2, local=True, max_passes=6, polish=False)
                if not math.isfinite(obj):
                    continue
                if obj < self.best_obj - max(1e-8, abs(self.best_obj) * 1e-10):
                    obj2, allocs2 = self.evaluate_hubs(H2, local=True, max_passes=50, polish=True)
                    if math.isfinite(obj2) and obj2 < obj:
                        obj, allocs = obj2, allocs2
                    if allocs is not None:
                        before = self.best_obj
                        self.set_best(H2, allocs, log=True)
                        accepted = self.best_obj < before - self.TOL
                        if accepted:
                            break
            if not accepted or self.time_up():
                break

        if not self.time_up() and self.best_H is not None:
            obj, allocs = self.evaluate_hubs(self.best_H, local=True, max_passes=80, polish=True)
            if math.isfinite(obj) and allocs is not None:
                self.set_best(self.best_H, allocs, log=True)

    def solve_mip(self, warm_H, warm_allocs):
        if gp is None or self.remaining() <= 1e-6:
            return
        try:
            m = gp.Model("stochastic_single_allocation_hub")
            m.Params.Seed = 0
            m.Params.MIPGap = 1e-4
            m.Params.NumericFocus = 0
            m.Params.Threads = 1
            m.Params.OutputFlag = 0
            m.Params.TimeLimit = max(0.0, self.remaining())

            z = m.addVars(self.n, vtype=GRB.BINARY, name="z")
            y_keys = [(s, i, k) for s in range(self.S) for i in range(self.n) for k in range(self.n) if k != i]
            y = m.addVars(y_keys, vtype=GRB.BINARY, name="y")

            t = None
            if self.alpha != 0.0:
                t_keys = []
                for s, (ii, jj, vv) in enumerate(self.pair_data):
                    for a in range(vv.size):
                        t_keys.append((s, int(ii[a]), int(jj[a])))
                if t_keys:
                    t = m.addVars(t_keys, lb=0.0, name="t")

            obj = gp.LinExpr(0.0)
            if not self.is_p:
                obj += gp.quicksum(float(self.fixed[k]) * z[k] for k in range(self.n))
            for (s, i, k), var in y.items():
                obj += float(self.prob[s] * self.coeff[s, i] * self.dist[i, k]) * var
            if t is not None:
                for s, (ii, jj, vv) in enumerate(self.pair_data):
                    ps = float(self.prob[s] * self.alpha)
                    for a in range(vv.size):
                        obj += ps * float(vv[a]) * t[s, int(ii[a]), int(jj[a])]
            m.setObjective(obj, GRB.MINIMIZE)

            if self.is_p:
                m.addConstr(z.sum() == int(self.p), name="exact_p_hubs")
            else:
                m.addConstr(z.sum() >= 1, name="at_least_one_hub")

            for s in range(self.S):
                for i in range(self.n):
                    m.addConstr(y.sum(s, i, "*") == 1.0 - z[i], name=f"assign_{s}_{i}")
            m.addConstrs((y[s, i, k] <= z[k] for (s, i, k) in y_keys), name="assign_only_open")

            if self.cap is not None:
                for s in range(self.S):
                    for k in range(self.n):
                        lhs = float(self.O[s, k]) * z[k] + gp.quicksum(
                            float(self.O[s, i]) * y[s, i, k] for i in range(self.n) if i != k
                        )
                        m.addConstr(lhs <= float(self.cap[k]) * z[k], name=f"cap_{s}_{k}")

            if t is not None and self.hp_pairs:
                for s, (ii, jj, vv) in enumerate(self.pair_data):
                    for a in range(vv.size):
                        i = int(ii[a])
                        j = int(jj[a])
                        tij = t[s, i, j]
                        for k, l, dkl in self.hp_pairs:
                            aik = z[i] if k == i else y[s, i, k]
                            ajl = z[j] if l == j else y[s, j, l]
                            m.addConstr(tij >= dkl * (aik + ajl - 1.0), name=f"tr1_{s}_{i}_{j}_{k}_{l}")
                            ail = z[i] if l == i else y[s, i, l]
                            ajk = z[j] if k == j else y[s, j, k]
                            m.addConstr(tij >= dkl * (ail + ajk - 1.0), name=f"tr2_{s}_{i}_{j}_{k}_{l}")

            # Warm start from the incumbent heuristic solution.
            for k in range(self.n):
                z[k].Start = 1.0 if k in set(warm_H) else 0.0
            for key in y_keys:
                y[key].Start = 0.0
            for s in range(self.S):
                for i in range(self.n):
                    h = int(warm_allocs[s][i])
                    if h != i and (s, i, h) in y:
                        y[s, i, h].Start = 1.0
            if t is not None:
                for s, (ii, jj, vv) in enumerate(self.pair_data):
                    for a in range(vv.size):
                        i = int(ii[a])
                        j = int(jj[a])
                        hi = int(warm_allocs[s][i])
                        hj = int(warm_allocs[s][j])
                        t[s, i, j].Start = float(self.dist[hi, hj])

            def extract_from_cb(model):
                zv = [float(model.cbGetSolution(z[k])) for k in range(self.n)]
                open_h = [k for k in range(self.n) if zv[k] > 0.5]
                allocs = []
                for s in range(self.S):
                    h_arr = np.zeros(self.n, dtype=int)
                    for i in range(self.n):
                        if zv[i] > 0.5:
                            h_arr[i] = i
                            continue
                        best_k = -1
                        best_v = 0.5
                        for k in range(self.n):
                            if k == i:
                                continue
                            v = float(model.cbGetSolution(y[s, i, k]))
                            if v > best_v:
                                best_v = v
                                best_k = k
                        if best_k < 0:
                            if not open_h:
                                open_h.append(0)
                            best_k = min(open_h, key=lambda kk: self.dist[i, kk])
                        h_arr[i] = best_k
                    allocs.append(h_arr)
                return open_h, allocs

            if self.logger is not None:
                def cb(model, where):
                    if where == GRB.Callback.MIPSOL:
                        try:
                            H_cb, allocs_cb = extract_from_cb(model)
                            sol = self.build_solution_dict(H_cb, allocs_cb)
                            val = float(sol["objective_value"])
                            if val < self.last_logged - max(1e-9, abs(self.last_logged) * 1e-11):
                                self.logger.log_solution(val, sol)
                                self.last_logged = val
                        except Exception:
                            pass
                m.optimize(cb)
            else:
                m.optimize()

            if m.SolCount > 0:
                zv = [float(z[k].X) for k in range(self.n)]
                open_h = [k for k in range(self.n) if zv[k] > 0.5]
                allocs = []
                for s in range(self.S):
                    h_arr = np.zeros(self.n, dtype=int)
                    for i in range(self.n):
                        if zv[i] > 0.5:
                            h_arr[i] = i
                            continue
                        best_k = -1
                        best_v = 0.5
                        for k in range(self.n):
                            if k == i:
                                continue
                            v = float(y[s, i, k].X)
                            if v > best_v:
                                best_v = v
                                best_k = k
                        if best_k < 0:
                            if not open_h:
                                open_h.append(0)
                            best_k = min(open_h, key=lambda kk: self.dist[i, kk])
                        h_arr[i] = best_k
                    allocs.append(h_arr)
                if open_h:
                    self.set_best(open_h, allocs, log=True)
        except Exception:
            return

    def solve(self):
        if self.time_limit <= 0 or self.remaining() <= 0:
            return self.emergency_solution()

        H0 = self.greedy_construct()
        obj0, allocs0 = self.evaluate_hubs(
            H0, local=True, max_passes=25, polish=not self.use_mip
        )
        if allocs0 is None or not math.isfinite(obj0):
            allocs0 = []
            for s in range(self.S):
                h = self.initial_assignment(s, H0)
                if h is None:
                    h = self.nearest_alloc(H0)
                allocs0.append(h)
        self.set_best(H0, allocs0, log=True)

        if self.use_mip:
            self.solve_mip(self.best_H, allocs0)
        else:
            self.improve_heuristic()

        if self.best_sol is None:
            self.best_sol = self.emergency_solution()
            if self.logger is not None:
                val = float(self.best_sol["objective_value"])
                if val < self.last_logged:
                    self.logger.log_solution(val, self.best_sol)
                    self.last_logged = val
        return self.best_sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    solver = StochasticHubSolver(instance, args.time_limit, logger)
    sol = solver.solve()

    # Final consistency pass: ensure the written objective matches the written decisions.
    sol = solver.build_solution_dict(sol["hubs"], [np.array(sol["allocations"][str(s)][str(i)], dtype=int) for s in range(solver.S) for i in [0]][:0] or [[sol["allocations"][str(s)][str(i)] for i in range(solver.n)] for s in range(solver.S)])

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, separators=(",", ":"))


if __name__ == "__main__":
    main()