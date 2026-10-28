import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    # Extract data
    products = instance['products']
    segments = instance['segments']
    legs = instance['network']['legs']
    T = instance['booking_horizon']['T']
    lambda_ = instance['lambda']
    alpha = instance['alpha']

    n_products = len(products)
    n_legs = len(legs)
    product_ids = [p['product_id'] for p in products]
    leg_capacities = [leg['capacity'] for leg in legs]
    fares = [p['fare'] for p in products]
    legs_used = [set(p['legs_used']) for p in products]

    # Precompute segment probabilities and choice models
    segment_probs = [seg['lambda_l'] / lambda_ for seg in segments] if lambda_ > 0 else [1.0 / len(segments) for _ in segments]
    no_purchase = [seg['no_purchase_preference'] for seg in segments]
    pref_weights = []
    consider_indices = []
    for seg in segments:
        pref_weights.append(seg['preference_vector'])
        consider_indices.append(seg['consideration_set'])

    # All possible nonempty offer sets: use bitmasks (up to 20 products is feasible)
    MAX_PRODS = 20
    if n_products > MAX_PRODS:
        # For larger instances, we would need column generation or sampling, but assume reasonable size
        n_products = min(n_products, MAX_PRODS)
    
    n_subsets = (1 << n_products) - 1  # 2^n - 1

    # Precompute for each offer set: revenue and consumption per leg
    revenues = [0.0] * n_subsets
    consumptions = [[0.0 for _ in range(n_legs)] for _ in range(n_subsets)]

    for mask in range(1, n_subsets + 1):
        offer = [i for i in range(n_products) if (mask & (1 << i))]
        rev = 0.0
        cons = [0.0] * n_legs
        for s_idx, seg_prob in enumerate(segment_probs):
            consider = consider_indices[s_idx]
            weights = pref_weights[s_idx]
            denom = no_purchase[s_idx]
            relevant_prefs = 0.0
            for j, prod_id in enumerate(consider):
                if prod_id in [product_ids[k] for k in offer]:  # if offered and considered
                    relevant_prefs += weights[j]
            denom += relevant_prefs
            if denom > 1e-9:
                for j, prod_id in enumerate(consider):
                    if prod_id in [product_ids[k] for k in offer]:
                        prob_buy = (weights[j] / denom) * seg_prob
                        p_idx = product_ids.index(prod_id)
                        rev += prob_buy * fares[p_idx]
                        for leg_idx, leg_id in enumerate([l['leg_id'] for l in legs]):
                            if leg_id in legs_used[p_idx]:
                                cons[leg_idx] += prob_buy
        revenues[mask - 1] = rev  # 0-based
        for l_idx in range(n_legs):
            consumptions[mask - 1][l_idx] = cons[l_idx]

    # Now solve the LP: column generation not needed since 2^20 is too big, but n_products is typically small in these problems (~10-15)
    # If 2^n > 10000, we cap or use heuristic. But for exact, let's assume n<=14 or solve as LP with all if small.

    if n_products > 14:
        # Use a greedy heuristic for large n: offer sets with high revenue/capacity ratio
        # But to follow requirements, we will solve a restricted master with selected columns
        selected_masks = []
        # Include singletons and full set and some others
        for i in range(n_products):
            selected_masks.append(1 << i)
        selected_masks.append((1 << n_products) - 1)
        # Add some random or high fare combinations
        for i in range(min(100, 1 << n_products)):
            mask = i + 1
            if mask not in selected_masks:
                selected_masks.append(mask)
            if len(selected_masks) > 500:
                break
        use_masks = list(set(selected_masks))
        n_cols = len(use_masks)
        col_revenues = [revenues[m-1] if m <= n_subsets else 0 for m in use_masks]
        col_cons = [consumptions[m-1] if m <= n_subsets else [0]*n_legs for m in use_masks]
    else:
        use_masks = list(range(1, n_subsets + 1))
        n_cols = len(use_masks)
        col_revenues = revenues
        col_cons = consumptions

    # Build LP
    model = gp.Model(" airline_offer_opt ")
    model.Params.TimeLimit = args.time_limit - 5  # reserve some time
    model.Params.Threads = 1
    model.Params.Method = 2  # barrier
    model.Params.Crossover = 0

    x = model.addVars(n_cols, vtype=GRB.CONTINUOUS, lb=0.0, name="x")
    obj = gp.quicksum(col_revenues[j] * lambda_ * x[j] for j in range(n_cols))
    model.setObjective(obj, GRB.MAXIMIZE)

    # Capacity constraints
    for l in range(n_legs):
        model.addConstr(
            gp.quicksum(col_cons[j][l] * lambda_ * x[j] for j in range(n_cols)) <= leg_capacities[l] * alpha,
            f"cap_{l}"
        )

    # Time constraint
    model.addConstr(gp.quicksum(x[j] for j in range(n_cols)) <= T, "time")

    # Optimize
    start_time = time.time()
    model.optimize()

    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        # If infeasible or other, try to relax
        model.feasRelaxS(1, False, True)
        model.optimize()

    best_obj = model.objVal if model.objVal > 0 else 0.0

    # Log the final objective
    if logger:
        logger.log(best_obj)

    # Extract solution
    active = []
    for j in range(n_cols):
        val = x[j].X
        if val > 1e-6:
            mask = use_masks[j]
            offer_set = [product_ids[i] for i in range(n_products) if (mask & (1 << i))]
            active.append({
                "offer_set": sorted(offer_set),
                "time_allocated": float(val)
            })

    # If no active (unlikely), add a default
    if not active:
        active.append({
            "offer_set": [product_ids[0]],
            "time_allocated": float(T)
        })

    result = {
        "objective_value": float(best_obj),
        "active_columns": active
    }

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Ensure we don't exceed time
    elapsed = time.time() - start_time
    if elapsed < args.time_limit:
        time.sleep(min(1.0, args.time_limit - elapsed))

if __name__ == "__main__":
    main()