import argparse
import json
import time
import numpy as np
from itertools import combinations
from solution_logger import SolutionLogger

def solve(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Parse instance data
    network = instance['network']
    num_legs = network['num_legs']
    legs = network['legs']
    products = instance['products']
    segments = instance['segments']
    T = instance['booking_horizon']['T']
    lam = instance['lambda']  # overall arrival probability per period
    
    num_products = len(products)
    num_segments = len(segments)
    
    # Map product_id to index
    prod_id_to_idx = {p['product_id']: i for i, p in enumerate(products)}
    prod_ids = [p['product_id'] for p in products]
    
    # Leg capacities
    leg_capacities = {leg['leg_id']: leg['capacity'] for leg in legs}
    leg_ids = [leg['leg_id'] for leg in legs]
    leg_id_to_idx = {lid: i for i, lid in enumerate(leg_ids)}
    capacities = np.array([leg_capacities[lid] for lid in leg_ids], dtype=float)
    
    # Product fares
    fares = np.array([p['fare'] for p in products], dtype=float)
    
    # Incidence matrix: legs x products
    A = np.zeros((num_legs, num_products), dtype=float)
    for j, p in enumerate(products):
        for lid in p['legs_used']:
            if lid in leg_id_to_idx:
                A[leg_id_to_idx[lid], j] = 1.0
    
    # Segment data
    seg_lambdas = np.array([s['lambda_l'] for s in segments], dtype=float)
    seg_no_purchase = np.array([s['no_purchase_preference'] for s in segments], dtype=float)
    
    # For each segment, store consideration set indices and preference weights
    seg_consideration = []  # list of arrays of product indices
    seg_preferences = []    # list of arrays of preference weights
    for s in segments:
        cs = s['consideration_set']
        pv = s['preference_vector']
        indices = np.array([prod_id_to_idx[pid] for pid in cs], dtype=int)
        weights = np.array(pv, dtype=float)
        seg_consideration.append(indices)
        seg_preferences.append(weights)
    
    def compute_offer_set_metrics(offer_set_mask):
        """
        Given a boolean mask of size num_products indicating which products are offered,
        compute:
        - expected revenue per arriving customer
        - expected leg consumption per arriving customer (array of size num_legs)
        - aggregate purchase probabilities per product
        """
        purchase_probs = np.zeros(num_products, dtype=float)
        
        for l in range(num_segments):
            cs_indices = seg_consideration[l]
            cs_weights = seg_preferences[l]
            # Which products in consideration set are in the offer set?
            in_offer = offer_set_mask[cs_indices]
            if not np.any(in_offer):
                continue
            
            available_weights = cs_weights * in_offer
            denom = seg_no_purchase[l] + np.sum(available_weights)
            seg_prob = seg_lambdas[l] / lam if lam > 0 else 0
            
            for k in range(len(cs_indices)):
                if in_offer[k]:
                    prob = seg_prob * (cs_weights[k] / denom)
                    purchase_probs[cs_indices[k]] += prob
        
        revenue = np.dot(fares, purchase_probs)
        leg_consumption = A @ purchase_probs
        
        return revenue, leg_consumption, purchase_probs
    
    # For large numbers of products, enumerating all 2^n subsets is infeasible.
    # We use column generation approach with LP.
    
    # The LP is:
    # max sum_S (lambda * R(S) * t_S)
    # s.t. sum_S (lambda * q_i(S) * t_S) <= C_i for each leg i
    #      sum_S t_S <= T
    #      t_S >= 0
    
    # Dual: min sum_i C_i * pi_i + T * sigma
    # s.t. lambda * R(S) <= sum_i pi_i * lambda * q_i(S) + sigma  for all S
    # pi_i >= 0, sigma >= 0
    
    # Rearranged: lambda * (R(S) - sum_i pi_i * q_i(S)) <= sigma
    
    # The pricing problem: find S that maximizes lambda * (R(S) - sum_i pi_i * q_i(S))
    # If this exceeds sigma, we have a violated constraint / improving column.
    
    # R(S) - sum_i pi_i * q_i(S) = sum_j (fare_j - sum_i pi_i * A_ij) * purchase_prob_j(S)
    # = sum_j adjusted_fare_j * purchase_prob_j(S)
    
    # The pricing problem decomposes: for each segment, we want to choose which products
    # in the consideration set to include to maximize the segment's contribution to the
    # adjusted revenue.
    
    # However, the offer set must be consistent across segments (same set S for all).
    # This makes the pricing problem NP-hard in general.
    
    # For small instances, we can enumerate. For larger ones, we use heuristics.
    
    import gurobipy as gp
    from gurobipy import GRB
    
    def solve_pricing(dual_pi, dual_sigma):
        """
        Find the offer set S that maximizes:
        lambda * sum_j adjusted_fare_j * purchase_prob_j(S) - sigma
        where adjusted_fare_j = fare_j - sum_i pi_i * A_ij
        """
        # adjusted fares per product
        adjusted_fares = fares - A.T @ dual_pi  # shape (num_products,)
        
        # We want to maximize: sum over segments of (segment_prob * sum over j in S∩C_l of
        #   adjusted_fare_j * pref_j / (no_purchase + sum of pref_k for k in S∩C_l))
        
        # For small num_products, enumerate all subsets
        if num_products <= 20:
            best_val = 0.0  # empty set gives 0
            best_mask = np.zeros(num_products, dtype=bool)
            
            # Enumerate all non-empty subsets
            for size in range(1, num_products + 1):
                if time.time() - start_time > time_limit - 2:
                    break
                for combo in combinations(range(num_products), size):
                    mask = np.zeros(num_products, dtype=bool)
                    mask[list(combo)] = True
                    
                    val = 0.0
                    for l in range(num_segments):
                        cs_indices = seg_consideration[l]
                        cs_weights = seg_preferences[l]
                        in_offer = mask[cs_indices]
                        if not np.any(in_offer):
                            continue
                        
                        available_weights = cs_weights * in_offer
                        denom = seg_no_purchase[l] + np.sum(available_weights)
                        seg_prob = seg_lambdas[l] / lam if lam > 0 else 0
                        
                        for k in range(len(cs_indices)):
                            if in_offer[k]:
                                prob = seg_prob * (cs_weights[k] / denom)
                                val += prob * adjusted_fares[cs_indices[k]]
                    
                    val *= lam
                    
                    if val > best_val:
                        best_val = val
                        best_mask = mask.copy()
            
            return best_val - dual_sigma, best_mask
        else:
            # Greedy/heuristic approach for larger instances
            # Try multiple strategies
            best_val = 0.0
            best_mask = np.zeros(num_products, dtype=bool)
            
            # Strategy 1: Include only products with positive adjusted fare
            mask = adjusted_fares > 0
            if np.any(mask):
                val = _eval_adjusted(mask, adjusted_fares)
                if val > best_val:
                    best_val = val
                    best_mask = mask.copy()
            
            # Strategy 2: Greedy add
            mask = np.zeros(num_products, dtype=bool)
            current_val = 0.0
            for _ in range(num_products):
                best_add = -1
                best_add_val = current_val
                for j in range(num_products):
                    if not mask[j]:
                        mask[j] = True
                        v = _eval_adjusted(mask, adjusted_fares)
                        if v > best_add_val:
                            best_add_val = v
                            best_add = j
                        mask[j] = False
                if best_add >= 0:
                    mask[best_add] = True
                    current_val = best_add_val
                else:
                    break
            if current_val > best_val:
                best_val = current_val
                best_mask = mask.copy()
            
            # Strategy 3: Greedy remove from all products
            mask = np.ones(num_products, dtype=bool)
            current_val = _eval_adjusted(mask, adjusted_fares)
            improved = True
            while improved:
                improved = False
                for j in range(num_products):
                    if mask[j]:
                        mask[j] = False
                        v = _eval_adjusted(mask, adjusted_fares)
                        if v > current_val:
                            current_val = v
                            improved = True
                        else:
                            mask[j] = True
            if current_val > best_val:
                best_val = current_val
                best_mask = mask.copy()
            
            # Strategy 4: For each segment, find best subset independently, union them
            for l in range(num_segments):
                cs_indices = seg_consideration[l]
                # Include products with positive adjusted fare in this segment's CS
                mask = np.zeros(num_products, dtype=bool)
                for k, j in enumerate(cs_indices):
                    if adjusted_fares[j] > 0:
                        mask[j] = True
                if np.any(mask):
                    v = _eval_adjusted(mask, adjusted_fares)
                    if v > best_val:
                        best_val = v
                        best_mask = mask.copy()
            
            # Strategy 5: Try each individual product
            for j in range(num_products):
                mask = np.zeros(num_products, dtype=bool)
                mask[j] = True
                v = _eval_adjusted(mask, adjusted_fares)
                if v > best_val:
                    best_val = v
                    best_mask = mask.copy()
            
            return best_val - dual_sigma, best_mask
    
    def _eval_adjusted(mask, adjusted_fares):
        val = 0.0
        for l in range(num_segments):
            cs_indices = seg_consideration[l]
            cs_weights = seg_preferences[l]
            in_offer = mask[cs_indices]
            if not np.any(in_offer):
                continue
            available_weights = cs_weights * in_offer
            denom = seg_no_purchase[l] + np.sum(available_weights)
            seg_prob = seg_lambdas[l] / lam if lam > 0 else 0
            for k in range(len(cs_indices)):
                if in_offer[k]:
                    prob = seg_prob * (cs_weights[k] / denom)
                    val += prob * adjusted_fares[cs_indices[k]]
        return val * lam
    
    # Column generation
    # Start with some initial columns
    
    # Initial columns: each individual product, and the full set
    columns = []  # list of (mask, revenue, leg_consumption)
    column_masks_set = set()
    
    def mask_to_tuple(mask):
        return tuple(np.where(mask)[0])
    
    def add_column(mask):
        key = mask_to_tuple(mask)
        if key in column_masks_set or not np.any(mask):
            return False
        column_masks_set.add(key)
        rev, leg_cons, _ = compute_offer_set_metrics(mask)
        columns.append((mask.copy(), rev, leg_cons.copy()))
        return True
    
    # Add individual products
    for j in range(num_products):
        mask = np.zeros(num_products, dtype=bool)
        mask[j] = True
        add_column(mask)
    
    # Add full set
    mask = np.ones(num_products, dtype=bool)
    add_column(mask)
    
    # Add products with positive fare only per segment consideration set
    for s in segments:
        mask = np.zeros(num_products, dtype=bool)
        for pid in s['consideration_set']:
            idx = prod_id_to_idx[pid]
            mask[idx] = True
        add_column(mask)
    
    # For small instances, enumerate all subsets upfront
    if num_products <= 15:
        for size in range(2, num_products + 1):
            if time.time() - start_time > time_limit * 0.3:
                break
            for combo in combinations(range(num_products), size):
                mask = np.zeros(num_products, dtype=bool)
                mask[list(combo)] = True
                add_column(mask)
    
    best_objective = 0.0
    best_solution = []
    
    # Column generation loop
    max_cg_iterations = 1000
    for cg_iter in range(max_cg_iterations):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Solve restricted master problem
        num_cols = len(columns)
        
        model = gp.Model("RMP")
        model.setParam('OutputFlag', 0)
        model.setParam('Method', 2)  # barrier
        
        t_vars = model.addVars(num_cols, lb=0.0, name="t")
        
        # Objective: max sum_S lambda * R(S) * t_S
        obj = gp.LinExpr()
        for k in range(num_cols):
            obj += lam * columns[k][1] * t_vars[k]
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # Capacity constraints
        leg_constrs = []
        for i in range(num_legs):
            lhs = gp.LinExpr()
            for k in range(num_cols):
                coeff = lam * columns[k][2][i]
                if coeff > 1e-12:
                    lhs += coeff * t_vars[k]
            constr = model.addConstr(lhs <= capacities[i], name=f"leg_{i}")
            leg_constrs.append(constr)
        
        # Time constraint
        time_lhs = gp.LinExpr()
        for k in range(num_cols):
            time_lhs += t_vars[k]
        time_constr = model.addConstr(time_lhs <= T, name="time")
        
        model.optimize()
        
        if model.status != GRB.OPTIMAL:
            break
        
        current_obj = model.ObjVal
        if current_obj > best_objective:
            best_objective = current_obj
            best_solution = []
            for k in range(num_cols):
                if t_vars[k].X > 1e-8:
                    offer_set = [prod_ids[j] for j in range(num_products) if columns[k][0][j]]
                    best_solution.append({
                        'offer_set': offer_set,
                        'time_allocated': t_vars[k].X
                    })
            if logger:
                logger.log(best_objective)
        
        # Get dual values
        dual_pi = np.array([leg_constrs[i].Pi for i in range(num_legs)], dtype=float)
        dual_sigma = time_constr.Pi
        
        # Solve pricing problem
        reduced_cost, new_mask = solve_pricing(dual_pi, dual_sigma)
        
        if reduced_cost < 1e-6:
            # No improving column found
            break
        
        added = add_column(new_mask)
        if not added:
            # Column already exists, try perturbation or stop
            # Try a few random perturbations
            found_new = False
            for _ in range(10):
                perturbed = new_mask.copy()
                # Flip a random bit
                j = np.random.randint(num_products)
                perturbed[j] = not perturbed[j]
                if np.any(perturbed):
                    if add_column(perturbed):
                        found_new = True
                        break
            if not found_new:
                break
    
    # Final solve with all columns
    if len(columns) > 0 and time.time() - start_time < time_limit - 2:
        num_cols = len(columns)
        model = gp.Model("Final")
        model.setParam('OutputFlag', 0)
        
        t_vars = model.addVars(num_cols, lb=0.0, name="t")
        
        obj = gp.LinExpr()
        for k in range(num_cols):
            obj += lam * columns[k][1] * t_vars[k]
        model.setObjective(obj, GRB.MAXIMIZE)
        
        for i in range(num_legs):
            lhs = gp.LinExpr()
            for k in range(num_cols):
                coeff = lam * columns[k][2][i]
                if coeff > 1e-12:
                    lhs += coeff * t_vars[k]
            model.addConstr(lhs <= capacities[i])
        
        time_lhs = gp.LinExpr()
        for k in range(num_cols):
            time_lhs += t_vars[k]
        model.addConstr(time_lhs <= T)
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            current_obj = model.ObjVal
            if current_obj > best_objective:
                best_objective = current_obj
                best_solution = []
                for k in range(num_cols):
                    if t_vars[k].X > 1e-8:
                        offer_set = [prod_ids[j] for j in range(num_products) if columns[k][0][j]]
                        best_solution.append({
                            'offer_set': offer_set,
                            'time_allocated': t_vars[k].X
                        })
                if logger:
                    logger.log(best_objective)
    
    # Write solution
    solution = {
        'objective_value': best_objective,
        'active_columns': best_solution
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()