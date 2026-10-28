import argparse
import json
import math
from gurobipy import *
import sys

def solve_cutting_stock(instance):
    problem_type = instance["problem_type"]
    machine_length = instance["machine_length"]
    num_products = instance["num_products"]
    num_patterns = instance["num_patterns"]
    cost_setup_pattern = instance["cost_setup_pattern"]
    cost_repetition = instance["cost_repetition"]
    M = instance["M"]
    
    products = {p["id"]: p for p in instance["products"]}
    product_ids = [p["id"] for p in instance["products"]]
    product_lengths = {p["id"]: p["length"] for p in instance["products"]}
    demands = {p["id"]: p["demand"] for p in instance["products"]}
    N_is = {p["id"]: p["N_i"] for p in instance["products"]}
    
    # Determine if it's knife dependent
    knife_dependent = problem_type == "cscp_with_knife_dependent_setups"
    if knife_dependent:
        cost_knife_setup = instance["cost_knife_setup"]
    
    # Determine if extra pieces are forced to zero
    x_i_forced_zero = instance.get("x_i_forced_zero", False)
    open_ended = problem_type != "closed_ended_cscp" and not x_i_forced_zero
    
    model = Model("CuttingStock")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    # Variables
    # y_p: binary variable indicating if pattern p is used
    y = {}
    for p_idx in range(num_patterns):
        y[p_idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{p_idx}")
    
    # r_p: number of repetitions of pattern p
    r = {}
    for p_idx in range(num_patterns):
        r[p_idx] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=M, name=f"r_{p_idx}")
    
    # x_ip: number of pieces of product i in pattern p
    x = {}
    for p_idx in range(num_patterns):
        for prod_id in product_ids:
            x[(prod_id, p_idx)] = model.addVar(vtype=GRB.INTEGER, 
                                               lb=0, 
                                               ub=N_is[prod_id], 
                                               name=f"x_{prod_id}_{p_idx}")
    
    # extra_pieces_i: number of extra pieces of product i (for open ended)
    extra_pieces = {}
    if open_ended:
        for prod_id in product_ids:
            extra_pieces[prod_id] = model.addVar(vtype=GRB.INTEGER, 
                                                 lb=0, 
                                                 name=f"extra_{prod_id}")
    
    # Constraints
    # If pattern p is not used, then no pieces can be assigned to it
    for p_idx in range(num_patterns):
        for prod_id in product_ids:
            model.addConstr(x[(prod_id, p_idx)] <= N_is[prod_id] * y[p_idx], 
                            name=f"no_pieces_if_not_used_{prod_id}_{p_idx}")
    
    # Pattern length constraint: sum of lengths of pieces in pattern p <= machine length
    for p_idx in range(num_patterns):
        model.addConstr(quicksum(x[(prod_id, p_idx)] * product_lengths[prod_id] 
                                 for prod_id in product_ids) <= machine_length,
                        name=f"pattern_length_{p_idx}")
    
    # Demand constraints: total production from all patterns + extra pieces >= demand
    for prod_id in product_ids:
        total_production = quicksum(x[(prod_id, p_idx)] * r[p_idx] 
                                    for p_idx in range(num_patterns))
        if open_ended:
            total_production += extra_pieces[prod_id]
        model.addConstr(total_production >= demands[prod_id],
                        name=f"demand_{prod_id}")
    
    # Open ended constraint: total extra pieces <= total repetitions
    if open_ended:
        total_extra = quicksum(extra_pieces[prod_id] for prod_id in product_ids)
        total_reps = quicksum(r[p_idx] for p_idx in range(num_patterns))
        model.addConstr(total_extra <= total_reps, name="extra_leq_total_reps")
    
    # Order patterns by repetition count (to avoid interchangeable orderings)
    for p_idx in range(num_patterns - 1):
        model.addConstr(r[p_idx] >= r[p_idx + 1], name=f"order_patterns_{p_idx}")
    
    # Objective function
    # Fixed setup times for used patterns
    fixed_setup_cost = cost_setup_pattern * quicksum(y[p_idx] for p_idx in range(num_patterns))
    
    # Cutting times for all pattern repetitions
    cutting_time_cost = cost_repetition * quicksum(r[p_idx] for p_idx in range(num_patterns))
    
    if knife_dependent:
        # Count knives in each pattern (total pieces in the pattern)
        knives = {}
        for p_idx in range(num_patterns):
            knives[p_idx] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"knives_{p_idx}")
            model.addConstr(knives[p_idx] == quicksum(x[(prod_id, p_idx)] for prod_id in product_ids),
                            name=f"knives_def_{p_idx}")
        
        knife_setup_cost = cost_knife_setup * quicksum(knives[p_idx] for p_idx in range(num_patterns))
        total_obj = fixed_setup_cost + cutting_time_cost + knife_setup_cost
    else:
        total_obj = fixed_setup_cost + cutting_time_cost
    
    model.setObjective(total_obj, GRB.MINIMIZE)
    
    # Solve
    model.optimize()
    
    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution = {
            "objective_value": model.objVal,
            "patterns": {},
            "extra_pieces": {},
            "production_quantities": {}
        }
        
        for p_idx in range(num_patterns):
            used = y[p_idx].x > 0.5
            reps = int(round(r[p_idx].x)) if used else 0
            pieces_in_pattern = {}
            
            for prod_id in product_ids:
                pieces_in_pattern[prod_id] = int(round(x[(prod_id, p_idx)].x)) if used else 0
            
            pattern_sol = {
                "used": used,
                "repetitions": reps,
                "pieces": pieces_in_pattern
            }
            
            if knife_dependent:
                pattern_sol["knife_count"] = int(round(knives[p_idx].x)) if used else 0
            
            solution["patterns"][str(p_idx)] = pattern_sol
            
            # Calculate production quantities for this pattern
            for prod_id in product_ids:
                prod_qty = pieces_in_pattern[prod_id] * reps
                solution["production_quantities"][f"product_{prod_id}_pattern_{p_idx}"] = prod_qty
        
        # Handle extra pieces
        for prod_id in product_ids:
            if open_ended:
                extra_val = int(round(extra_pieces[prod_id].x))
                solution["extra_pieces"][str(prod_id)] = extra_val
            else:
                solution["extra_pieces"][str(prod_id)] = 0
    
        return solution
    else:
        print(f"Model status: {model.status}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Solve Cutting Stock Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file')
    
    args = parser.parse_args()
    
    # Initialize logger if provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    
    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    # Solve the problem
    solution = solve_cutting_stock(instance)
    
    if solution and logger:
        logger.log_solution(solution['objective_value'], solution)
    
    # Write solution
    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("No solution found.")
        with open(args.solution_path, 'w') as f:
            json.dump({}, f)

if __name__ == "__main__":
    main()