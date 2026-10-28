import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import time
import math
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True, help='Path to the input JSON instance file')
    parser.add_argument('--solution_path', required=True, help='Path to save the output JSON solution file')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds')
    parser.add_argument('--log_path', default=None, help='Path to log intermediate solutions')
    args = parser.parse_args()

    try:
        with open(args.instance_path, 'r') as fin:
            instance_data = json.load(fin)
    except Exception as e:
        raise ValueError(f"Could not read instance file: {e}")
    
    problem_type = instance_data['problem_type']
    force_y_zero = (problem_type == 'closed_ended_cscp')
    machine_length = instance_data['machine_length']
    num_products = instance_data['num_products']
    num_patterns = instance_data['num_patterns']
    cost_setup_pattern = instance_data['cost_setup_pattern']
    cost_repetition = instance_data['cost_repetition']
    cost_knife_setup = instance_data.get('cost_knife_setup', 0.0)
    M = instance_data['M']
    products_list = instance_data['products']   # list of dicts
    
    product_ids = [p['id'] for p in products_list]
    lengths = [p['length'] for p in products_list]
    demands = [p['demand'] for p in products_list]
    N_i = [p['N_i'] for p in products_list]    # maximum pieces per pattern for product i

    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    m = gp.Model()
    m.setParam('Seed', 0)
    m.setParam('MIPGap', 1e-4)
    m.setParam('NumericFocus', 0)
    m.setParam('Threads', 1)
    m.setParam('TimeLimit', args.time_limit)
    
    z = m.addVars(num_patterns, vtype=GRB.BINARY, name="z")
    r = m.addVars(num_patterns, vtype=GRB.INTEGER, lb=0, ub=M, name="r")
    x = m.addVars(num_products, num_patterns, vtype=GRB.INTEGER, name="x")
    y = m.addVars(num_products, vtype=GRB.INTEGER, lb=0, name="y")
    
    if force_y_zero:
        for i in range(num_products):
            m.addConstr(y[i] == 0, name=f"y_zero_{i}")
    
    for j in range(num_patterns):
        for i in range(num_products):
            m.addConstr(x[i,j] <= N_i[i] * z[j], name=f"link_x_z_{i}_{j}")
    
    for j in range(num_patterns):
        m.addConstr(gp.quicksum(lengths[i] * x[i,j] for i in range(num_products)) <= machine_length, 
                    name=f"cap_{j}")

    for i in range(num_products):
        m.addConstr(gp.quicksum(x[i,j] * r[j] for j in range(num_patterns)) + y[i] >= demands[i], 
                   name=f"demand_{i}")
    
    if not force_y_zero:
        m.addConstr(y.sum() <= r.sum(), name="extra_limit")
    
    for j in range(num_patterns-1):
        m.addConstr(r[j] >= r[j+1], name=f"sym_{j}")

    base_obj = cost_setup_pattern * z.sum() + cost_repetition * r.sum()
    if problem_type == 'cscp_with_knife_dependent_setups':
        total_knife_pieces = gp.quicksum(x[i,j] for i in range(num_products) for j in range(num_patterns))
        knife_cost = cost_knife_setup * total_knife_pieces
        obj_expr = base_obj + knife_cost
    else:
        obj_expr = base_obj
    m.setObjective(obj_expr, GRB.MINIMIZE)
    
    def build_solution_dict(z_vals, r_vals, x_vals, y_vals):
        patterns_out = {}
        for j in range(num_patterns):
            z_val = z_vals[j]
            r_val = r_vals[j]
            pieces_dict = {}
            for i in range(num_products):
                x_val = x_vals[i, j]
                pieces_dict[str(product_ids[i])] = int(round(x_val))
            pattern_dict = {
                "used": bool(int(round(z_val))),
                "repetitions": int(round(r_val)),
                "pieces": pieces_dict
            }
            if problem_type == 'cscp_with_knife_dependent_setups':
                knife_count = int(round(sum(x_vals[i,j] for i in range(num_products))))
                pattern_dict['knife_count'] = knife_count
            patterns_out[str(j)] = pattern_dict
        
        extra_pieces_dict = {}
        for i in range(num_products):
            y_val = y_vals[i]
            extra_pieces_dict[str(product_ids[i])] = int(round(y_val))
        
        production_quantities = {}
        for j in range(num_patterns):
            rep_val = int(round(r_vals[j]))
            for i in range(num_products):
                x_val = int(round(x_vals[i,j]))
                total_pieces = x_val * rep_val
                key = f"product_{product_ids[i]}_pattern_{j}"
                production_quantities[key] = total_pieces
        
        output = {
            "patterns": patterns_out,
            "extra_pieces": extra_pieces_dict,
            "production_quantities": production_quantities
        }
        return output
    
    def build_zero_solution():
        patterns_out = {}
        for j in range(num_patterns):
            pattern_dict = {
                "used": False,
                "repetitions": 0,
                "pieces": {str(pid): 0 for pid in product_ids}
            }
            if problem_type == 'cscp_with_knife_dependent_setups':
                pattern_dict['knife_count'] = 0
            patterns_out[str(j)] = pattern_dict
        extra_pieces_dict = {str(pid): 0 for pid in product_ids}
        production_quantities = {}
        for j in range(num_patterns):
            for pid in product_ids:
                key = f"product_{pid}_pattern_{j}"
                production_quantities[key] = 0
        solution_dict = {
            "objective_value": 0.0,
            "patterns": patterns_out,
            "extra_pieces": extra_pieces_dict,
            "production_quantities": production_quantities
        }
        return solution_dict
    
    if logger:
        def gpcallback(model, where):
            if where == GRB.Callback.MIPSOL:
                if logger:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj_val)
        m.optimize(gpcallback)
    else:
        m.optimize()
        
    solution_dict = None
    if m.SolCount >= 1:
        z_vals = [z[j].Xn for j in range(num_patterns)]
        r_vals = [r[j].Xn for j in range(num_patterns)]
        x_vals = {}
        for i in range(num_products):
            for j in range(num_patterns):
                x_vals[i, j] = x[i,j].Xn
        y_vals = [y[i].Xn for i in range(num_products)]
        solution_dict = build_solution_dict(z_vals, r_vals, x_vals, y_vals)
        solution_dict['objective_value'] = m.ObjVal
    else:
        solution_dict = build_zero_solution()
        
    if solution_dict and logger:
        logger.log_solution(solution_dict['objective_value'], solution_dict)
        
    with open(args.solution_path, 'w') as fout:
        json.dump(solution_dict, fout, indent=4)

if __name__ == '__main__':
    main()