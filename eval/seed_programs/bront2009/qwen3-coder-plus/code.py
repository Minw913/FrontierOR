import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from solution_logger import SolutionLogger

def solve_airline_network_optimization(instance_path, solution_path, timelimit, log_path=None):
    # Load the instance
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    # Initialize logger if requested
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Extract data
    num_cities = data['network']['num_cities']
    city_names = data['network']['city_names']
    num_legs = data['network']['num_legs']
    legs = data['network']['legs']
    products = data['products']
    segments = data['segments']
    T = data['booking_horizon']['T']
    alpha = data['alpha']
    lambda_global = data['lambda']

    num_products = len(products)
    num_segments = len(segments)
    
    # Scale leg capacities
    scaled_capacities = [leg['capacity'] * alpha for leg in legs]
    
    # Build a mapping from (leg_id -> index)
    leg_id_to_idx = {leg['leg_id']: i for i, leg in enumerate(legs)}
    product_id_to_idx = {prod['product_id']: i for i, prod in enumerate(products)}
    
    # Define binary matrix product_leg_matrix[product_idx][leg_idx] = 1 if product uses leg
    product_leg_matrix = np.zeros((num_products, num_legs))
    for p_idx, product in enumerate(products):
        for leg_id in product['legs_used']:
            leg_idx = leg_id_to_idx[leg_id]
            product_leg_matrix[p_idx][leg_idx] = 1
    
    # Compute total lambda (sum of segment lambdas should equal global lambda)
    segment_lambdas = []
    for seg in segments:
        segment_lambdas.append(seg['lambda_l'])
    
    # Total Lambda check
    total_segment_lambda = sum(segment_lambdas)
    # Normalize if needed
    if abs(total_segment_lambda - lambda_global) > 1e-8:
        # Adjust segment lambdas proportionally to sum to lambda_global
        adjust_factor = lambda_global / total_segment_lambda
        for i in range(len(segment_lambdas)):
            segment_lambdas[i] *= adjust_factor
    
    # Reduce the size of max_set_size to avoid very large problem sizes
    max_set_size = min(7, num_products)  # Much smaller to prevent extremely large numbers of subsets
    
    # Generate all possible nonempty subsets of products of sizes up to max_set_size
    all_subsets = []
    for s in range(1, max_set_size + 1):
        for subset_ids in combinations(range(num_products), s):
            all_subsets.append(list(subset_ids))
    
    # Dictionary to store subset_id to details mapping
    subset_details = {}  # id => subset_info_dict
    for i, subset in enumerate(all_subsets):
        subset_id = i
        subset_product_indices = subset  # These are indices in the products list
        subset_product_ids = [products[idx]['product_id'] for idx in subset_product_indices]
        
        # Now calculate expected revenue and purchase probabilities for this subset
        segment_choice_probs = {}
        for seg_idx, seg in enumerate(segments):
            consid_set_ids = seg['consideration_set']
            # Find intersection with current offer set
            intersection_ids = [pid for pid in subset_product_ids if pid in consid_set_ids]
            
            if not intersection_ids:
                # No products in this subset match consideration set
                for pid in subset_product_ids:
                    segment_choice_probs[(seg_idx, pid)] = 0.0
            else:
                # Calculate weights in consideration set intersect offer set 
                consid_pid_to_wgt = {pid: wgt for pid, wgt in zip(consid_set_ids, seg['preference_vector'])}
                
                # Find weights of items in intersection
                weights_in_intersection = []
                products_in_intersection = []
                for pid in intersection_ids:
                    wgt = consid_pid_to_wgt[pid]
                    weights_in_intersection.append(wgt)
                    products_in_intersection.append(pid)
                
                # Add no-purchase weight
                no_purchase_weight = seg['no_purchase_preference']
                total_weight = sum(weights_in_intersection) + no_purchase_weight
                
                # Calculate choice probability for each product in intersection
                if total_weight > 1e-9:
                    for j, pid in enumerate(products_in_intersection):
                        prob = weights_in_intersection[j] / total_weight
                        segment_choice_probs[(seg_idx, pid)] = prob
                else:
                    # Avoid division by zero
                    for j, pid in enumerate(products_in_intersection):
                        segment_choice_probs[(seg_idx, pid)] = 0.0
                
                # Products in offer set but not in consideration remain 0
                for pid in subset_product_ids:
                    if pid not in consid_set_ids:
                        segment_choice_probs[(seg_idx, pid)] = 0.0
        
        # Aggregate probabilities
        aggregated_probs = {}
        total_revenue = 0.0
        product_purchases = []
        
        for pid in subset_product_ids:
            agg_prob = 0.0
            for seg_idx in range(num_segments):
                prob = segment_choice_probs.get((seg_idx, pid), 0.0)
                seg_lambda = segment_lambdas[seg_idx]
                agg_prob += seg_lambda * prob
            # Map from product ID back to index for later use
            p_idx = product_id_to_idx[pid]
            aggregated_probs[p_idx] = agg_prob
            fare = products[p_idx]['fare']
            total_revenue += agg_prob * fare
            product_purchases.append(agg_prob)  # This is the probability of purchase (demand)
        
        subset_details[subset_id] = {
            'subset': subset_product_indices,  # Store the internal indices 
            'aggregated_probs': aggregated_probs,  # Maps product_index -> probability
            'total_revenue': total_revenue,
            'product_purchases': product_purchases
        }
    
    num_subsets = len(all_subsets)
    
    # Special case handling: if no subsets possible (very rare edge case),
    # return empty solution
    if num_subsets == 0:
        solution = {
            'objective_value': 0.0,
            'active_columns': []
        }
        
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        
        if logger:
            logger.log(0.0)
        
        return
    
    # Set up Gurobi model
    model = gp.Model("AirlineNetworkOptimization")
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', timelimit)
    model.setParam('OutputFlag', 0)  # Silent
    
    # Decision variables: time allocated to each offer set
    time = []
    var_refs = []  # To keep reference to variables
    
    # Create vars for all subsets 
    for i in range(num_subsets):
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"time_{i}")
        time.append(var)
        var_refs.append(var)
    
    # Add constraint: total time <= T
    model.addConstr(gp.quicksum(time) <= T, name="time_constraint")
    
    # Add capacity constraints for each leg
    for leg_idx in range(num_legs):
        expr = gp.LinExpr()
        for k in range(num_subsets):
            subset_detail = subset_details[k]
            subset_indices = subset_detail['subset']
            aggregated_probs = subset_detail['aggregated_probs']  # product_idx -> probability
            
            consumption_for_leg_k = 0.0
            for p_idx in subset_indices:
                # Check if the product with this index uses this leg
                if product_leg_matrix[p_idx][leg_idx] == 1:
                    # Get the probability for this product
                    consumption_for_leg_k += aggregated_probs.get(p_idx, 0.0)
                    
            expr.addTerms(consumption_for_leg_k, time[k])
        
        model.addConstr(expr <= scaled_capacities[leg_idx], name=f"capacity_leg_{leg_idx}")
    
    # Objective function: maximize expected revenue
    obj_expr = gp.LinExpr()
    for k in range(num_subsets):
        subset_detail = subset_details[k]
        rev = subset_detail['total_revenue']
        obj_expr.addTerms(rev, time[k])
    
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    def mycallback(model, where):
        if where == GRB.Callback.MIP:
            obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            if logger:
                logger.log(obj)
    
    try:
        model.optimize(mycallback if logger else None)
    except Exception as e:
        print(f"Gurobi optimization error: {e}")
        solution = {
            'objective_value': 0.0,
            'active_columns': []
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f)
        if logger:
            logger.log(0.0)
        return
    
    # Extract solution
    try:
        if model.SolCount > 0:
            obj = model.ObjVal
            # Log final objective
            if logger and model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                logger.log(obj)

            active_columns = []
            for i in range(num_subsets):
                time_val = time[i].X
                if time_val > 1e-9:  # Only include if time > 0 (tighter tolerance)
                    subset_detail = subset_details[i]
                    subset_product_indices = subset_detail['subset']
                    subset_product_ids = [products[idx]['product_id'] for idx in subset_product_indices]

                    col = {
                        'offer_set': subset_product_ids,
                        'time_allocated': time_val
                    }
                    active_columns.append(col)
                    
            final_objective = obj
        else:
            final_objective = 0.0
            active_columns = []

        solution = {
            'objective_value': final_objective,
            'active_columns': active_columns
        }
    except AttributeError:
        # Handle cases where solution attributes are not accessible
        solution = {
            'objective_value': 0.0,
            'active_columns': []
        }

    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

def main():
    parser = argparse.ArgumentParser(description='Solve airline network optimization problem.')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the input JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to the output JSON solution file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Optional path to log intermediate solutions')
    
    args = parser.parse_args()
    solve_airline_network_optimization(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        timelimit=args.time_limit,
        log_path=args.log_path
    )

if __name__ == "__main__":
    main()