import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Extract data
    problem_type = data['problem_type']
    L = data['machine_length']
    num_patterns = data['num_patterns']
    M = data['M']
    cost_setup = data['cost_setup_pattern']
    cost_rep = data['cost_repetition']
    cost_knife = data.get('cost_knife_setup', 0.0)
    is_knife_dependent = problem_type == 'cscp_with_knife_dependent_setups'
    is_closed = problem_type == 'closed_ended_cscp'
    force_extra_zero = data.get('x_i_forced_zero', False)

    products = data['products']
    num_products = len(products)
    prod_id_to_idx = {p['id']: i for i, p in enumerate(products)}
    lengths = [p['length'] for p in products]
    demands = [p['demand'] for p in products]
    N_max = [p['N_i'] for p in products]

    # Start timing
    start_time = time.time()
    remaining_time = args.time_limit

    model = gp.Model("CSCP")
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, remaining_time)

    # Variables
    use = model.addVars(num_patterns, vtype=GRB.BINARY, name="use")
    rep = model.addVars(num_patterns, vtype=GRB.INTEGER, lb=0, ub=M, name="rep")
    pieces = model.addVars(num_patterns, num_products, vtype=GRB.INTEGER, lb=0, name="pieces")
    extra = model.addVars(num_products, vtype=GRB.INTEGER, lb=0, name="extra")

    # Auxiliary for objective (knife cost)
    total_knives = model.addVar(vtype=GRB.INTEGER, lb=0, name="total_knives")

    # Constraints
    # 1. Pattern length
    for p in range(num_patterns):
        model.addConstr(
            gp.quicksum(pieces[p, i] * lengths[i] for i in range(num_products)) <= L,
            name=f"length_{p}"
        )

    # 2. If not used, no pieces and no repetitions
    for p in range(num_patterns):
        for i in range(num_products):
            model.addConstr(pieces[p, i] <= N_max[i] * use[p], name=f"piece_bound_{p}_{i}")
        model.addConstr(rep[p] <= M * use[p], name=f"rep_bound_{p}")

    # 3. Demand satisfaction
    for i in range(num_products):
        prod_from_patterns = gp.quicksum(pieces[p, i] * rep[p] for p in range(num_patterns))
        model.addConstr(prod_from_patterns + extra[i] >= demands[i], name=f"demand_{i}")

    # 4. Extra pieces limit (open-ended)
    if not is_closed and not force_extra_zero:
        model.addConstr(
            gp.quicksum(extra[i] for i in range(num_products)) <= gp.quicksum(rep[p] for p in range(num_patterns)),
            name="extra_limit"
        )
    else:
        for i in range(num_products):
            model.addConstr(extra[i] == 0, name=f"extra_zero_{i}")

    # 5. Ordering to break symmetry: rep[p] >= rep[p+1]
    for p in range(num_patterns - 1):
        model.addConstr(rep[p] >= rep[p + 1], name=f"order_{p}")

    # 6. Knife count for knife-dependent
    if is_knife_dependent:
        for p in range(num_patterns):
            model.addConstr(
                total_knives >= gp.quicksum(pieces[p, i] for i in range(num_products)) - (1 - use[p]) * 1000,
                name=f"knife_lower_{p}"
            )
            model.addConstr(
                total_knives <= gp.quicksum(pieces[p, i] for i in range(num_products)) + (1 - use[p]) * 1000,
                name=f"knife_upper_{p}"
            )

    # Objective
    obj = cost_setup * gp.quicksum(use[p] for p in range(num_patterns)) + \
          cost_rep * gp.quicksum(rep[p] for p in range(num_patterns))
    if is_knife_dependent:
        obj += cost_knife * total_knives
    model.setObjective(obj, GRB.MINIMIZE)

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                # Build solution dict
                sol = {"objective_value": float(obj_val), "patterns": {}, "extra_pieces": {}, "production_quantities": {}}
                
                # Get values
                use_val = model.cbGetSolution(use)
                rep_val = model.cbGetSolution(rep)
                pieces_val = model.cbGetSolution(pieces)
                extra_val = model.cbGetSolution(extra)
                
                for p_idx in range(num_patterns):
                    pat_key = str(p_idx)
                    u = bool(round(use_val[p_idx]))
                    r = int(round(rep_val[p_idx]))
                    pat_pieces = {}
                    for i_idx, prod in enumerate(products):
                        pid = str(prod['id'])
                        pc = int(round(pieces_val[p_idx, i_idx]))
                        pat_pieces[pid] = pc
                    
                    sol["patterns"][pat_key] = {
                        "used": u,
                        "repetitions": r,
                        "pieces": pat_pieces
                    }
                    if is_knife_dependent:
                        knife_c = sum(int(round(pieces_val[p_idx, i_idx])) for i_idx in range(num_products))
                        sol["patterns"][pat_key]["knife_count"] = knife_c
                
                for prod in products:
                    pid_str = str(prod['id'])
                    sol["extra_pieces"][pid_str] = int(round(extra_val[prod_id_to_idx[prod['id']]]))
                
                # Production quantities
                for i_idx, prod in enumerate(products):
                    pid = prod['id']
                    for p_idx in range(num_patterns):
                        key = f"product_{pid}_pattern_{p_idx}"
                        pc = int(round(pieces_val[p_idx, i_idx]))
                        r = int(round(rep_val[p_idx]))
                        sol["production_quantities"][key] = pc * r
                
                logger.log_solution(obj_val, sol)

    # Optimize
    model.optimize(callback)

    # Build final solution
    solution = {
        "objective_value": 0.0,
        "patterns": {},
        "extra_pieces": {},
        "production_quantities": {}
    }

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL):
        if model.SolCount > 0:
            obj_val = model.objVal
            solution["objective_value"] = float(obj_val)
            
            use_val = model.getAttr('X', use)
            rep_val = model.getAttr('X', rep)
            pieces_val = model.getAttr('X', pieces)
            extra_val = model.getAttr('X', extra)
            
            for p_idx in range(num_patterns):
                pat_key = str(p_idx)
                u = bool(round(use_val[p_idx]))
                r = int(round(rep_val[p_idx]))
                pat_pieces = {}
                for prod in products:
                    pid_str = str(prod['id'])
                    i_idx = prod_id_to_idx[prod['id']]
                    pc = int(round(pieces_val[p_idx, i_idx]))
                    pat_pieces[pid_str] = pc
                
                pattern_dict = {
                    "used": u,
                    "repetitions": r,
                    "pieces": pat_pieces
                }
                if is_knife_dependent:
                    knife_c = sum(int(round(pieces_val[p_idx, prod_id_to_idx[prod['id']]])) for prod in products)
                    pattern_dict["knife_count"] = knife_c
                solution["patterns"][pat_key] = pattern_dict
            
            for prod in products:
                pid_str = str(prod['id'])
                i_idx = prod_id_to_idx[prod['id']]
                solution["extra_pieces"][pid_str] = int(round(extra_val[i_idx]))
            
            for prod in products:
                pid = prod['id']
                for p_idx in range(num_patterns):
                    key = f"product_{pid}_pattern_{p_idx}"
                    i_idx = prod_id_to_idx[pid]
                    pc = int(round(pieces_val[p_idx, i_idx]))
                    r = int(round(rep_val[p_idx]))
                    solution["production_quantities"][key] = pc * r
    else:
        # Infeasible or no solution - return dummy feasible if possible, but should not happen
        solution["objective_value"] = 1e9

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()