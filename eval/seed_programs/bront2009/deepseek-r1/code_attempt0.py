import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path=None):
    start_time = time.time()
    logger = SolutionLogger(log_path, sense='maximize') if log_path else None
    best_obj_so_far = -1e9

    # Read instance
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    global_lambda = data['lambda']
    T = data['booking_horizon']['T']
    alpha = data['alpha']
    legs = data['network']['legs']
    products_data = data['products']
    segments_data = data['segments']
    num_legs = len(legs)
    
    # Build mapping from leg_id to leg info
    leg_by_id = {leg['leg_id']: leg for leg in legs}
    leg_id_to_index = {}
    for idx, leg in enumerate(legs):
        leg_id_to_index[leg['leg_id']] = idx

    # Precompute forbidden products (using legs with zero effective capacity)
    forbidden_products = set()
    for prod in products_data:
        for leg_id in prod['legs_used']:
            leg = leg_by_id[leg_id]
            if alpha * leg['capacity'] == 0:
                forbidden_products.add(prod['product_id'])
                break
    
    # Filter products and segments
    all_products = []
    product_info = {}
    for prod in products_data:
        if prod['product_id'] in forbidden_products:
            continue
        all_products.append(prod['product_id'])
        product_info[prod['product_id']] = {
            'fare': prod['fare'],
            'legs_used': prod['legs_used']
        }
    
    segments = []
    for seg in segments_data:
        new_consideration_set = [j for j in seg['consideration_set'] if j not in forbidden_products]
        new_preference_vector = []
        for idx, j in enumerate(seg['consideration_set']):
            if j not in forbidden_products:
                new_preference_vector.append(seg['preference_vector'][idx])
        segments.append({
            'segment_id': seg['segment_id'],
            'lambda_l': seg['lambda_l'],
            'consideration_set': new_consideration_set,
            'preference_vector': new_preference_vector,
            'no_purchase_preference': seg['no_purchase_preference'],
            'description': seg['description']
        })
    
    num_segments = len(segments)
    no_purchase_preference_m = [seg['no_purchase_preference'] for seg in segments]
    lambda_m = [seg['lambda_l'] for seg in segments]
    
    # Precompute segments_containing_j and weights
    segments_containing_j = {j: [] for j in all_products}
    weights = [dict() for _ in range(num_segments)]
    for m_idx, seg in enumerate(segments):
        for idx, j in enumerate(seg['consideration_set']):
            segments_containing_j[j].append(m_idx)
            weights[m_idx][j] = seg['preference_vector'][idx]
    
    def compute_revenue_and_consumption(S):
        R_S = 0.0
        C_i = [0.0] * num_legs
        
        for m_idx in range(num_segments):
            common_products = S & set(weights[m_idx].keys())
            if not common_products:
                continue
            denom = no_purchase_preference_m[m_idx] + sum(weights[m_idx][j] for j in common_products)
            if denom == 0:
                continue
            for j in common_products:
                prob = weights[m_idx][j] / denom
                R_S += (lambda_m[m_idx] / global_lambda) * product_info[j]['fare'] * prob
                for leg_id in product_info[j]['legs_used']:
                    leg_idx = leg_id_to_index[leg_id]
                    C_i[leg_idx] += (lambda_m[m_idx] / global_lambda) * prob
        return R_S, C_i

    # Initialize columns
    columns = []
    existing_sets = set()
    
    # Add single product sets
    for j in all_products:
        S = {j}
        R_S, C_i_list = compute_revenue_and_consumption(S)
        col_set = frozenset(S)
        columns.append({
            'set': col_set,
            'revenue': R_S,
            'consumption': C_i_list
        })
        existing_sets.add(col_set)
    
    # Add full set
    if all_products:
        S_full = set(all_products)
        R_full, C_i_full = compute_revenue_and_consumption(S_full)
        col_set_full = frozenset(S_full)
        columns.append({
            'set': col_set_full,
            'revenue': R_full,
            'consumption': C_i_full
        })
        existing_sets.add(col_set_full)
    
    # Local search function for pricing
    def local_search(initial_set, r_j, segments_containing_j, weights, no_purchase_preference_m, lambda_m, dual_time, max_iter=1000):
        current_set = set(initial_set)
        current_denom = no_purchase_preference_m.copy()
        current_numerator = [0.0] * len(no_purchase_preference_m)
        
        for j in current_set:
            for m_idx in segments_containing_j[j]:
                if j in weights[m_idx]:
                    current_denom[m_idx] += weights[m_idx][j]
                    current_numerator[m_idx] += r_j[j] * weights[m_idx][j]
        
        current_rc = 0.0
        for m_idx in range(len(no_purchase_preference_m)):
            if current_denom[m_idx] > 0:
                current_rc += lambda_m[m_idx] * (current_numerator[m_idx] / current_denom[m_idx])
        current_rc -= dual_time
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter:
            iteration += 1
            improved = False
            best_delta = 0.0
            best_move = None
            
            # Try adding products
            for j in set(all_products) - current_set:
                delta_rc = 0.0
                for m_idx in segments_containing_j[j]:
                    if j not in weights[m_idx]:
                        continue
                    old_contrib = current_numerator[m_idx] / current_denom[m_idx] if current_denom[m_idx] > 0 else 0
                    new_denom = current_denom[m_idx] + weights[m_idx][j]
                    new_numerator = current_numerator[m_idx] + r_j[j] * weights[m_idx][j]
                    new_contrib = new_numerator / new_denom
                    delta_rc += lambda_m[m_idx] * (new_contrib - old_contrib)
                new_rc = current_rc + delta_rc
                if new_rc - current_rc > best_delta:
                    best_delta = new_rc - current_rc
                    best_move = ('add', j, new_rc)
            
            # Try removing products
            for j in list(current_set):
                if len(current_set) == 1:
                    continue
                delta_rc = 0.0
                for m_idx in segments_containing_j[j]:
                    if j not in weights[m_idx]:
                        continue
                    old_contrib = current_numerator[m_idx] / current_denom[m_idx]
                    new_denom = current_denom[m_idx] - weights[m_idx][j]
                    new_numerator = current_numerator[m_idx] - r_j[j] * weights[m_idx][j]
                    new_contrib = new_numerator / new_denom
                    delta_rc += lambda_m[m_idx] * (new_contrib - old_contrib)
                new_rc = current_rc + delta_rc
                if new_rc - current_rc > best_delta:
                    best_delta = new_rc - current_rc
                    best_move = ('remove', j, new_rc)
            
            if best_delta > 1e-6:
                if best_move[0] == 'add':
                    j = best_move[1]
                    current_set.add(j)
                    for m_idx in segments_containing_j[j]:
                        if j in weights[m_idx]:
                            current_denom[m_idx] += weights[m_idx][j]
                            current_numerator[m_idx] += r_j[j] * weights[m_idx][j]
                    current_rc = best_move[2]
                    improved = True
                else:
                    j = best_move[1]
                    current_set.remove(j)
                    for m_idx in segments_containing_j[j]:
                        if j in weights[m_idx]:
                            current_denom[m_idx] -= weights[m_idx][j]
                            current_numerator[m_idx] -= r_j[j] * weights[m_idx][j]
                    current_rc = best_move[2]
                    improved = True
            else:
                break
        
        return current_set, current_rc

    # Column generation loop
    while time.time() - start_time < time_limit - 1:  # Leave 1 second for final operations
        try:
            model = gp.Model('master')
            model.setParam('OutputFlag', 0)
            
            # Create variables for each column
            x = model.addVars(len(columns), lb=0.0, name='x')
            
            # Time constraint
            time_constr = model.addConstr(gp.quicksum(x[i] for i in range(len(columns))) <= T, "time")
            
            # Leg constraints
            leg_constrs = []
            for i in range(num_legs):
                cap = alpha * legs[i]['capacity']
                expr = gp.quicksum(global_lambda * columns[idx]['consumption'][i] * x[idx] for idx in range(len(columns)))
                leg_constrs.append(model.addConstr(expr <= cap, f"leg_{i}"))
            
            # Objective
            obj_expr = gp.quicksum(global_lambda * columns[idx]['revenue'] * x[idx] for idx in range(len(columns)))
            model.setObjective(obj_expr, GRB.MAXIMIZE)
            
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                current_obj = model.objVal
                if current_obj > best_obj_so_far:
                    best_obj_so_far = current_obj
                    if logger:
                        logger.log(current_obj)
                
                dual_time_val = time_constr.Pi
                dual_legs = [constr.Pi for constr in leg_constrs]
                
                # Compute net revenue for each product
                r_j = {}
                for j in all_products:
                    net_rev = product_info[j]['fare']
                    for leg_id in product_info[j]['legs_used']:
                        leg_idx = leg_id_to_index[leg_id]
                        net_rev -= dual_legs[leg_idx]
                    r_j[j] = net_rev
                
                # Pricing: find new columns
                candidates = []
                
                # Start from full set
                if all_products:
                    S_full = set(all_products)
                    S_candidate, rc_candidate = local_search(S_full, r_j, segments_containing_j, weights, no_purchase_preference_m, lambda_m, dual_time_val)
                    candidates.append((S_candidate, rc_candidate))
                
                # Start from single product sets
                single_set_rc = []
                for j in all_products:
                    rc_val = 0.0
                    for m_idx in segments_containing_j[j]:
                        if j in weights[m_idx]:
                            denom = no_purchase_preference_m[m_idx] + weights[m_idx][j]
                            prob = weights[m_idx][j] / denom
                            rc_val += lambda_m[m_idx] * r_j[j] * prob
                    rc_val -= dual_time_val
                    single_set_rc.append((rc_val, j))
                
                single_set_rc.sort(key=lambda x: x[0], reverse=True)
                k = min(5, len(single_set_rc))
                for i in range(k):
                    rc_val, j = single_set_rc[i]
                    S0 = {j}
                    S_candidate, rc_candidate = local_search(S0, r_j, segments_containing_j, weights, no_purchase_preference_m, lambda_m, dual_time_val)
                    candidates.append((S_candidate, rc_candidate))
                
                # Find best candidate
                best_rc = -1e9
                best_set_candidate = None
                for S, rc_val in candidates:
                    if rc_val > best_rc:
                        best_rc = rc_val
                        best_set_candidate = S
                
                # Add new column if beneficial
                if best_rc > 1e-6:
                    S_frozen = frozenset(best_set_candidate)
                    if S_frozen not in existing_sets:
                        R_S, C_i_list = compute_revenue_and_consumption(best_set_candidate)
                        columns.append({
                            'set': S_frozen,
                            'revenue': R_S,
                            'consumption': C_i_list
                        })
                        existing_sets.add(S_frozen)
                    else:
                        break
                else:
                    break
            else:
                break
        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
            break
    
    # Solve the final master problem
    model = gp.Model('final_master')
    model.setParam('OutputFlag', 0)
    x = model.addVars(len(columns), lb=0.0, name='x')
    time_constr = model.addConstr(gp.quicksum(x[i] for i in range(len(columns))) <= T, "time")
    leg_constrs = []
    for i in range(num_legs):
        cap = alpha * legs[i]['capacity']
        expr = gp.quicksum(global_lambda * columns[idx]['consumption'][i] * x[idx] for idx in range(len(columns)))
        leg_constrs.append(model.addConstr(expr <= cap, f"leg_{i}"))
    obj_expr = gp.quicksum(global_lambda * columns[idx]['revenue'] * x[idx] for idx in range(len(columns)))
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        final_obj = model.objVal
        if final_obj > best_obj_so_far and logger:
            logger.log(final_obj)
        active_columns = []
        for idx in range(len(columns)):
            if x[idx].X > 1e-6:
                active_columns.append({
                    'offer_set': list(columns[idx]['set']),
                    'time_allocated': x[idx].X
                })
        solution = {
            'objective_value': final_obj,
            'active_columns': active_columns
        }
    else:
        solution = {
            'objective_value': 0.0,
            'active_columns': []
        }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)