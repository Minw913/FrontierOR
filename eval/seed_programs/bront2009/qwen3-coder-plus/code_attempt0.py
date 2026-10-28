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
    if abs(total_segment_lambda - lambda_global) > 1e-6:
        # Adjust segment lambdas proportionally to sum to lambda_global
        adjust_factor = lambda_global / total_segment_lambda
        for i in range(len(segment_lambdas)):
            segment_lambdas[i] *= adjust_factor
    
    # Determine maximum number of products that might make sense to consider
    # Based on limited practical number, we consider offer sets up to certain size
    max_set_size = min(8, num_products) # Adjust as needed or based on domain knowledge
    
    # Generate all possible nonempty subsets of products of sizes up to max_set_size
    all_subsets = []
    for s in range(1, max_set_size + 1):
        for subset_ids in combinations(range(num_products), s):
            all_subsets.append(list(subset_ids))
    
    # Dictionary to store subset_id to details mapping
    subset_details = {}  # id => subset_info_dict
    for i, subset in enumerate(all_subsets):
        subset_details[i] = {'subset': subset}
        
        # Now calculate expected revenue and purchase probabilities for this subset
        segment_choice_probs = {}
        subset_fares = []
        for seg_idx, seg in enumerate(segments):
            consid_set_ids = seg['consideration_set']
            # Find intersection with current offer set
            intersection_ids = [pid for pid in subset if pid in consid_set_ids]
            
            if not intersection_ids:
                # No products in this subset match consideration set
                for pid in subset:
                    segment_choice_probs[(seg_idx, pid)] = 0.0
            else:
                # Calculate weights in consideration set intersect offer set 
                consid_pid_to_wgt = {pid: wgt for pid, wgt in zip(consid_set_ids, seg['preference_vector'])}
                
                # Calculate weights for items in intersection
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
                for j, pid in enumerate(products_in_intersection):
                    prob = weights_in_intersection[j] / total_weight if total_weight > 1e-9 else 0.0
                    segment_choice_probs[(seg_idx, pid)] = prob
                
                # Products in offer set but not in consideration remain 0
                for pid in subset:
                    if pid not in consid_set_ids:
                        segment_choice_probs[(seg_idx, pid)] = 0.0
                        
        # Aggregate probabilities
        aggregated_probs = {}
        total_revenue = 0.0
        product_purchases = []
        
        for pid in subset:
            agg_prob = 0.0
            for seg_idx in range(num_segments):
                prob = segment_choice_probs.get((seg_idx, pid), 0.0)
                seg_lambda = segment_lambdas[seg_idx]
                agg_prob += seg_lambda * prob
            aggregated_probs[pid] = agg_prob
            fare = products[product_id_to_idx[pid]]['fare']
            total_revenue += agg_prob * fare
            product_purchases.append(agg_prob)  # This is the probability of purchase (demand)
        
        subset_details[i]['aggregated_probs'] = aggregated_probs
        subset_details[i]['total_revenue'] = total_revenue
        subset_details[i]['product_purchases'] = product_purchases
        
    num_subsets = len(all_subsets)
    
    # Set up Gurobi model
    model = gp.Model("AirlineNetworkOptimization")
    model.setParam('TimeLimit', timelimit)
    model.setParam('OutputFlag', 0)  # Silent
    
    # Decision variables: time allocated to each offer set
    time = []
    var_refs = []  # To keep reference to variables
    
    # Create vars only for subsets in range [0, num_subsets)
    for i in range(num_subsets):
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"time_{i}")
        time.append(var)
        var_refs.append(var)
    
    # Add constraint: total time <= T
    model.addConstr(gp.quicksum(time) <= T, name="time_constraint")
    
    # Add capacity constraints for each leg
    for leg_idx in range(num_legs):
        expr = gp.LinExpr()
        for k, subset_id in enumerate(range(num_subsets)):
            subset_detail = subset_details[subset_id]
            subset = subset_detail['subset']
            aggregated_probs = subset_detail['aggregated_probs']
            
            consumption_for_leg_k = 0.0
            for pid in subset:
                # Check if product uses this leg
                if product_leg_matrix[product_id_to_idx[pid]][leg_idx] == 1:
                    consumption_for_leg_k += aggregated_probs[pid]
                    
            expr.addTerms(consumption_for_leg_k, time[k])
        
        model.addConstr(expr <= scaled_capacities[leg_idx], name=f"capacity_leg_{leg_idx}")
    
    # Objective function: maximize expected revenue
    obj_expr = gp.LinExpr()
    for k, subset_id in enumerate(range(num_subsets)):
        subset_detail = subset_details[subset_id]
        rev = subset_detail['total_revenue']
        obj_expr.addTerms(rev, time[k])
    
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    model.optimize()
    
    # Extract solution
    if model.SolCount > 0:
        active_columns = []
        for i in range(num_subsets):
            time_val = time[i].X
            if time_val > 1e-6:  # Only include if time > 0
                offer_set = [products[pid]['product_id'] for pid in subset_details[i]['subset']]
                col = {
                    'offer_set': offer_set,
                    'time_allocated': time_val
                }
                active_columns.append(col)
                
        final_objective = model.ObjVal if model.Status == GRB.OPTIMAL else model.ObjVal if hasattr(model,'ObjVal') else 0
        
        solution = {
            'objective_value': final_objective,
            'active_columns': active_columns
        }
        
        # Log final objective
        if logger:
            logger.log(final_objective)
    else:
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