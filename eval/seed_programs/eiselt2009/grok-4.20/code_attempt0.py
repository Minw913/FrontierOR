import json
import argparse
import time
import numpy as np
from itertools import combinations
from solution_logger import SolutionLogger
import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n_nodes']
    p = instance['p']
    competitor_hubs = instance['competitor_hub_locations']
    demand = np.array(instance['demand_matrix'])
    dist = np.array(instance['distance_matrix'])
    nodes = instance['nodes']
    params = instance['parameters']
    alpha = params['alpha']
    gamma = params['gamma']
    beta = params['beta']
    lam = params['lambda']
    cost_scalar = params['cost_scalar']
    attr_2h = params['attractiveness_2hub']
    attr_1h = params['attractiveness_1hub']
    waiting = np.array([node['waiting_time'] for node in nodes])

    start_time = time.time()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Precompute times and fares
    travel_time = dist.copy()
    fare = cost_scalar * dist
    wait = waiting

    # Competitor utilities (fixed)
    comp_util = np.zeros((n, n))
    r = len(competitor_hubs)
    hubs_c = competitor_hubs

    def compute_util(i, j, k, l, is_entrant):
        if k == l:
            t = (travel_time[i, k] + travel_time[k, j] +
                 wait[i] + wait[k] + wait[j])
            c = (fare[i, k] + fare[k, j])
            attr = attr_1h
        else:
            t = (travel_time[i, k] + travel_time[k, l] + travel_time[l, j] +
                 wait[i] + wait[k] + wait[l] + wait[j])
            c = (fare[i, k] + alpha * fare[k, l] + fare[l, j])
            attr = attr_2h
        denom = gamma * t**beta + (1 - gamma) * c**lam
        u = attr / denom if denom > 1e-12 else 0.0
        return u

    # Competitor single hub
    for k_idx in range(r):
        k = hubs_c[k_idx]
        for i in range(n):
            for j in range(n):
                if demand[i, j] == 0:
                    continue
                u = compute_util(i, j, k, k, False)
                comp_util[i, j] += u

    # Competitor two hubs - only best order
    for a_idx in range(r):
        for b_idx in range(a_idx + 1, r):
            k = hubs_c[a_idx]
            l = hubs_c[b_idx]
            for i in range(n):
                for j in range(n):
                    if demand[i, j] == 0:
                        continue
                    u1 = compute_util(i, j, k, l, False)
                    u2 = compute_util(i, j, l, k, False)
                    comp_util[i, j] += max(u1, u2)

    total_comp = comp_util.copy()

    # Now solve MIP for entrant
    model = gp.Model("AirlineHub")
    model.Params.TimeLimit = args.time_limit - 5  # safety margin
    model.Params.Threads = 1
    model.Params.MIPFocus = 1  # focus on good solutions
    model.Params.Heuristics = 0.8
    model.Params.MIPGap = 0.01

    # Variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")  # y_kl = 1 if both k and l are hubs (k<=l)

    # Objective: sum demand * (entrant_util / (entrant_util + comp_util))
    # We linearize via auxiliary variables using piecewise or big-M but it's nonconvex.
    # Instead we use a greedy + local search + MIP on small set of promising hubs.

    # Since n is typically small (<=30), we can do exact MIP with linearized fractions if we discretize.
    # But better: use a combination of greedy and exact MIP on reduced set.

    # First, compute all possible pair utilities for entrant
    pair_utils = np.zeros((n, n, n, n))  # u[i,j,k,l] for k<=l
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if demand[i,j] == 0: continue
                u = compute_util(i, j, k, k, True)
                pair_utils[i, j, k, k] = u

    for kk in range(n):
        for ll in range(kk+1, n):
            for i in range(n):
                for j in range(n):
                    if demand[i,j] == 0: continue
                    u1 = compute_util(i, j, kk, ll, True)
                    u2 = compute_util(i, j, ll, kk, True)
                    pair_utils[i, j, kk, ll] = max(u1, u2)

    # MIP formulation
    # We maximize sum_{ij} demand_ij * z_ij
    # z_ij * (total_ent_ij + comp_ij) <= total_ent_ij
    # total_ent_ij = sum_{active pairs} u_ijkl * y_kl   (y_kl =1 if both hubs open)

    z = model.addVars(n, n, lb=0.0, ub=1.0, name="z")
    ent_util = model.addVars(n, n, lb=0.0, name="ent_util")
    obj_expr = 0.0

    for i in range(n):
        for j in range(n):
            if demand[i, j] == 0:
                continue
            d = demand[i, j]
            c_ij = total_comp[i, j]
            # ent_util[i,j] = sum valid pair contributions
            model.addConstr(ent_util[i, j] == gp.quicksum(
                pair_utils[i, j, k, l] * y[k, l]
                for k in range(n) for l in range(k, n)
            ))
            # z_ij <= ent_util[i,j] / (ent_util[i,j] + c_ij)
            # equiv to z * (ent + c) <= ent    i.e.   z*ent + z*c <= ent
            #   =>   ent - z*ent >= z * c
            #   =>   ent*(1 - z) >= c * z
            # To linearize we use big-M on bounds of ent_util.
            # But since it's bilinear, we introduce constraints with McCormick or use piecewise.
            # Given time constraints, we use a simple but effective greedy + 2-opt local search.

    model.update()

    # Since full MIP with fractional objective is hard to linearize without many auxiliaries,
    # we implement a fast local search heuristic that performs very well on this structure.

    # Greedy + local search
    best_hubs = None
    best_obj = -1.0
    best_captured = 0.0

    def evaluate(hub_set):
        hub_set = set(hub_set)
        ent_u = np.zeros((n, n))
        for k_idx, k_val in enumerate(hub_set):
            k = k_val
            for i in range(n):
                for j in range(n):
                    if demand[i,j] == 0: continue
                    u = compute_util(i, j, k, k, True)
                    ent_u[i,j] += u
        for ii in range(len(hub_set)):
            for jj in range(ii+1, len(hub_set)):
                k = list(hub_set)[ii]
                l = list(hub_set)[jj]
                for i in range(n):
                    for j in range(n):
                        if demand[i,j] == 0: continue
                        u1 = compute_util(i, j, k, l, True)
                        u2 = compute_util(i, j, l, k, True)
                        ent_u[i,j] += max(u1, u2)
        captured = 0.0
        for i in range(n):
            for j in range(n):
                if demand[i,j] == 0: continue
                total_u = ent_u[i,j] + total_comp[i,j]
                if total_u > 1e-12:
                    captured += demand[i,j] * (ent_u[i,j] / total_u)
        return captured, ent_u

    # Generate promising candidates: high degree or high demand nodes
    demand_out = demand.sum(axis=1)
    demand_in = demand.sum(axis=0)
    score = demand_out + demand_in
    candidates = np.argsort(-score)[:min(25, n)].tolist()

    # Greedy construction
    current_hubs = []
    remaining = set(candidates)
    for _ in range(p):
        best_add = None
        best_val = -1
        for cand in list(remaining):
            trial = current_hubs + [cand]
            val, _ = evaluate(trial)
            if val > best_val:
                best_val = val
                best_add = cand
        if best_add is not None:
            current_hubs.append(best_add)
            remaining.remove(best_add)
            if best_val > best_obj:
                best_obj = best_val
                best_hubs = current_hubs[:]
                best_captured = best_val
                if logger:
                    logger.log(best_captured)
                if time.time() - start_time > args.time_limit - 2:
                    break

    # Local search: swap
    improved = True
    while improved and (time.time() - start_time < args.time_limit - 2):
        improved = False
        current_val, _ = evaluate(current_hubs)
        for i in range(p):
            for new_hub in candidates:
                if new_hub in current_hubs:
                    continue
                trial = current_hubs[:]
                trial[i] = new_hub
                trial_val, _ = evaluate(trial)
                if trial_val > current_val + 1e-6:
                    current_hubs = trial
                    current_val = trial_val
                    improved = True
                    if current_val > best_obj:
                        best_obj = current_val
                        best_hubs = current_hubs[:]
                        best_captured = current_val
                        if logger:
                            logger.log(best_captured)
                    break
            if improved:
                break
        # Also try adding diversity from all nodes if candidates are not enough
        if not improved and len(candidates) < n:
            all_nodes = list(range(n))
            for i in range(p):
                for new_hub in all_nodes:
                    if new_hub in current_hubs:
                        continue
                    trial = current_hubs[:]
                    trial[i] = new_hub
                    trial_val, _ = evaluate(trial)
                    if trial_val > current_val + 1e-4:
                        current_hubs = trial
                        current_val = trial_val
                        improved = True
                        if current_val > best_obj:
                            best_obj = current_val
                            best_hubs = current_hubs[:]
                            best_captured = current_val
                            if logger:
                                logger.log(best_captured)
                        break
                if improved:
                    break

    # Final improvement with MIP on top 15 candidates if time remains
    if time.time() - start_time < args.time_limit - 10 and n >= 8:
        top_candidates = np.argsort(-score)[:15].tolist()
        if len(top_candidates) < p:
            top_candidates = list(range(n))
        model = gp.Model("HubSelect")
        model.Params.TimeLimit = max(5, args.time_limit - int(time.time() - start_time) - 3)
        model.Params.Threads = 1
        x = model.addVars(len(top_candidates), vtype=GRB.BINARY, name="select")
        model.addConstr(gp.quicksum(x[i] for i in range(len(top_candidates))) == p)

        # This is still a nonlinear problem. We instead run multiple 2-hub combinations via enumeration
        # when n small. Since time is limited we skip full MIP and trust local search.

    final_obj, _ = evaluate(best_hubs)
    best_hubs = sorted(best_hubs)

    solution = {
        "objective_value": float(final_obj),
        "hub_locations": best_hubs
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log(final_obj)


if __name__ == "__main__":
    main()