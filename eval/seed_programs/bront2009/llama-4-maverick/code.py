import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import numpy as np

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def compute_choice_probabilities(instance):
    segments = instance['segments']
    products = instance['products']
    product_ids = [product['product_id'] for product in products]
    
    segment_data = []
    for segment in segments:
        consideration_set = segment['consideration_set']
        preference_vector = segment['preference_vector']
        no_purchase_preference = segment['no_purchase_preference']
        lambda_l = segment['lambda_l']  # Use lambda_l directly
        
        product_preferences = {product_id: preference for product_id, preference in zip(consideration_set, preference_vector)}
        
        segment_data.append({
            'lambda_l': lambda_l,
            'product_preferences': product_preferences,
            'no_purchase_preference': no_purchase_preference
        })
    
    offer_sets = []
    num_products = len(product_ids)
    for i in range(1, 2**num_products):
        offer_set = [product_ids[j] for j in range(num_products) if (i & (1 << j))]
        offer_sets.append(tuple(offer_set))  # Store offer_sets as tuples
    
    choice_probabilities = {}
    for offer_set in offer_sets:
        choice_probabilities[offer_set] = {}
        for product in product_ids:
            aggregate_prob = 0
            for segment in segment_data:
                lambda_l = segment['lambda_l']
                product_preferences = segment['product_preferences']
                no_purchase_preference = segment['no_purchase_preference']
                
                if product not in product_preferences or product not in offer_set:
                    segment_prob = 0
                else:
                    considered_products = [p for p in offer_set if p in product_preferences]
                    denominator = sum(product_preferences[p] for p in considered_products) + no_purchase_preference
                    segment_prob = product_preferences[product] / denominator
                
                aggregate_prob += lambda_l * segment_prob
            
            choice_probabilities[offer_set][product] = aggregate_prob
    
    return choice_probabilities, offer_sets

def build_model(instance, choice_probabilities, offer_sets):
    products = instance['products']
    legs = instance['network']['legs']
    lambda_ = instance['lambda']
    T = instance['booking_horizon']['T']
    alpha = instance['alpha']
    
    product_ids = [product['product_id'] for product in products]
    product_fares = {product['product_id']: product['fare'] for product in products}
    leg_capacities = {leg['leg_id']: leg['capacity'] * alpha for leg in legs}
    product_legs = {product['product_id']: product['legs_used'] for product in products}
    
    model = gp.Model("airline_network_revenue_management")
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0
    
    time_allocated = model.addVars(offer_sets, lb=0, name="time_allocated")
    
    expected_revenue = gp.quicksum(lambda_ * sum(product_fares[product] * choice_probabilities[offer_set][product] for product in product_ids) * time_allocated[offer_set] for offer_set in offer_sets)
    model.setObjective(expected_revenue, GRB.MAXIMIZE)
    
    capacity_constraints = {}
    for leg in legs:
        leg_id = leg['leg_id']
        products_using_leg = [product for product in product_ids if leg_id in product_legs[product]]
        capacity_constraint = gp.quicksum(lambda_ * sum(choice_probabilities[offer_set][product] for product in products_using_leg if product in offer_set) * time_allocated[offer_set] for offer_set in offer_sets)
        capacity_constraints[leg_id] = model.addConstr(capacity_constraint <= leg_capacities[leg_id], name=f"capacity_{leg_id}")
    
    total_time_constraint = model.addConstr(gp.quicksum(time_allocated[offer_set] for offer_set in offer_sets) <= T, name="total_time")
    
    return model, time_allocated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    instance = read_instance(args.instance_path)
    choice_probabilities, offer_sets = compute_choice_probabilities(instance)
    model, time_allocated = build_model(instance, choice_probabilities, offer_sets)
    
    model.Params.TimeLimit = args.time_limit
    
    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    
    model.optimize(log_callback)
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {
            "objective_value": model.ObjVal,
            "active_columns": []
        }
        
        for offer_set in offer_sets:
            time_alloc = time_allocated[offer_set].X
            if time_alloc > 1e-6:  
                solution["active_columns"].append({
                    "offer_set": list(offer_set),
                    "time_allocated": time_alloc
                })
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print(f"Model optimization failed with status {model.status}")

if __name__ == "__main__":
    main()