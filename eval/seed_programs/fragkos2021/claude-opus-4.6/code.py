import argparse
import json
import time
import itertools
import numpy as np
from solution_logger import SolutionLogger


def compute_expected_cost(open_facilities, nodes, trans_cost_matrix, penalty_cost, num_customers, num_facilities):
    """Compute expected transportation + penalty cost given a set of open facilities."""
    total_cost = 0.0
    for j in range(num_customers):
        # Get facilities sorted by transportation cost for customer j
        fac_costs = []
        for i in open_facilities:
            fac_costs.append((trans_cost_matrix[j][i], nodes[i]['disruption_probability']))
        # Sort by cost
        fac_costs.sort(key=lambda x: x[0])
        
        # Compute expected cost
        # E[cost] = sum over levels: cost_level * prob(all cheaper disrupted) * prob(this survives)
        # + penalty * prob(all disrupted)
        exp_cost = 0.0
        prob_all_prev_disrupted = 1.0
        for cost, q in fac_costs:
            exp_cost += cost * prob_all_prev_disrupted * (1.0 - q)
            prob_all_prev_disrupted *= q
        exp_cost += penalty_cost[j] * prob_all_prev_disrupted
        
        total_cost += nodes[j]['demand'] * exp_cost
    return total_cost


def solve_enumeration(nodes, trans_cost_matrix, penalty_cost, num_facilities, num_customers, time_limit, start_time, logger):
    """Solve by enumerating all 2^n subsets of facilities."""
    best_obj = float('inf')
    best_open = []
    
    # Try all non-empty subsets (and empty set)
    for size in range(num_facilities + 1):
        for subset in itertools.combinations(range(num_facilities), size):
            if time.time() - start_time > time_limit - 0.5:
                return best_obj, best_open
            
            open_facs = list(subset)
            fixed_cost = sum(nodes[i]['fixed_setup_cost'] for i in open_facs)
            
            if fixed_cost >= best_obj:
                continue
            
            exp_cost = compute_expected_cost(open_facs, nodes, trans_cost_matrix, penalty_cost, num_customers, num_facilities)
            obj = fixed_cost + exp_cost
            
            if obj < best_obj:
                best_obj = obj
                best_open = open_facs
                if logger:
                    logger.log(best_obj)
    
    return best_obj, best_open


def solve_greedy(nodes, trans_cost_matrix, penalty_cost, num_facilities, num_customers):
    """Greedy solution: start with no facilities, add the one with best marginal improvement."""
    open_facs = set()
    
    # Start with all-penalty cost
    current_exp_cost = compute_expected_cost([], nodes, trans_cost_matrix, penalty_cost, num_customers, num_facilities)
    current_fixed = 0.0
    current_obj = current_fixed + current_exp_cost
    
    best_obj = current_obj
    best_open = set()
    
    available = set(range(num_facilities))
    
    while available:
        best_add = None
        best_add_obj = current_obj
        
        for i in available:
            trial_open = open_facs | {i}
            fixed = current_fixed + nodes[i]['fixed_setup_cost']
            exp_cost = compute_expected_cost(list(trial_open), nodes, trans_cost_matrix, penalty_cost, num_customers, num_facilities)
            obj = fixed + exp_cost
            if obj < best_add_obj:
                best_add_obj = obj
                best_add = i
        
        if best_add is not None:
            open_facs.add(best_add)
            available.remove(best_add)
            current_fixed += nodes[best_add]['fixed_setup_cost']
            current_obj = best_add_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_open = set(open_facs)
        else:
            break
    
    return best_obj, list(best_open)


def solve_local_search(nodes, trans_cost_matrix, penalty_cost, num_facilities, num_customers, 
                       initial_open, initial_obj, time_limit, start_time, logger):
    """Local search with add/drop/swap moves."""
    current_open = set(initial_open)
    current_obj = initial_obj
    best_obj = current_obj
    best_open = set(current_open)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit - 1.0:
            break
        improved = False
        
        # Try adding a facility
        for i in range(num_facilities):
            if time.time() - start_time > time_limit - 1.0:
                break
            if i in current_open:
                continue
            trial = current_open | {i}
            fixed = sum(nodes[f]['fixed_setup_cost'] for f in trial)
            exp = compute_expected_cost(list(trial), nodes, trans_cost_matrix, penalty_cost, num_customers, num_facilities)
            obj = fixed + exp
            if obj < current_obj - 1e-10:
                current_open = trial
                current_obj = obj
                improved = True
                if obj < best_obj:
                    best_obj = obj
                    best_open = set(current_open)
                    if logger:
                        logger.log(best_obj)
                break
        
        # Try dropping a facility
        for i in list(current_open):
            if time.time() - start_time > time_limit - 1.0:
                break
            trial = current_open - {i}
            fixed = sum(nodes[f]['fixed_setup_cost'] for f in trial)
            exp = compute_expected_cost(list(trial), nodes, trans_cost_matrix, penalty_cost, num_customers, num_facilities)
            obj = fixed + exp
            if obj < current_obj - 1e-10:
                current_open = trial
                current_obj = obj
                improved = True
                if obj < best_obj:
                    best_obj = obj
                    best_open = set(current_open)
                    if logger:
                        logger.log(best_obj)
                break
        
        # Try swapping
        for i in list(current_open):
            if time.time() - start_time > time_limit - 1.0:
                break
            found_swap = False
            for j in range(num_facilities):
                if j in current_open:
                    continue
                trial = (current_open - {i}) | {j}
                fixed = sum(nodes[f]['fixed_setup_cost'] for f in trial)
                exp = compute_expected_cost(list(trial), nodes, trans_cost_matrix, penalty_cost, num_customers, num_facilities)
                obj = fixed + exp
                if obj < current_obj - 1e-10:
                    current_open = trial
                    current_obj = obj
                    improved = True
                    found_swap = True
                    if obj < best_obj:
                        best_obj = obj
                        best_open = set(current_open)
                        if logger:
                            logger.log(best_obj)
                    break
            if found_swap:
                break
    
    return best_obj, list(best_open)


def solve_mip(nodes, trans_cost_matrix, penalty_cost, num_facilities, num_customers, 
              time_limit, start_time, logger, initial_open=None, initial_obj=None):
    """
    MIP formulation for the reliable facility location problem.
    
    We use the formulation from Snyder & Daskin (2005) / similar papers.
    
    For each customer j, sort facilities by cost. Let the sorted order be r_1, r_2, ..., r_n
    with c_{j,r_1} <= c_{j,r_2} <= ... <= c_{j,r_n}.
    
    Define assignment variables:
    y_{j,l} for l = 0, 1, ..., n where:
    - y_{j,0} = probability that customer j uses the penalty option
    - y_{j,l} = probability that facility r_l is the one serving customer j
    
    Actually for the independent case, we can use a level-based formulation.
    
    Let me use a different approach: scenario-based or cutting-plane.
    
    For moderate sizes, let's use a compact formulation.
    """
    import gurobipy as gp
    from gurobipy import GRB
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time < 1:
        return initial_obj, initial_open
    
    # For each customer j, precompute sorted facility order by cost
    # and the "effective costs" at each level
    
    # Variables: x_i = 1 if facility i is opened
    # For the expected cost, we use the following linearization:
    # 
    # For customer j with facilities sorted by cost (indices r_1, ..., r_n):
    # Expected cost = d_j * [ sum_{l=1}^{n} c_{j,r_l} * P(r_l serves j) + p_j * P(none serves j) ]
    # 
    # P(r_l serves j) = x_{r_l} * (1-q_{r_l}) * prod_{k<l} (1 - x_{r_k}*(1-q_{r_k}))
    # This is nonlinear. We need to linearize.
    #
    # A standard trick: define z_{j,l} = prod_{k=1}^{l} (q_{r_k} + (1-q_{r_k})*(1-x_{r_k}))
    #                                   = prod_{k=1}^{l} (1 - (1-q_{r_k})*x_{r_k})
    # z_{j,0} = 1
    # z_{j,l} = z_{j,l-1} * (1 - (1-q_{r_k})*x_{r_k})  -- nonlinear
    #
    # We can linearize: z_{j,l} = z_{j,l-1} * (q_{r_l} + (1-q_{r_l})*(1-x_{r_l}))
    #                            = z_{j,l-1} * (1 - (1-q_{r_l})*x_{r_l})
    #                            = z_{j,l-1} - (1-q_{r_l}) * z_{j,l-1} * x_{r_l}
    # Let w_{j,l} = z_{j,l-1} * x_{r_l}, then:
    # z_{j,l} = z_{j,l-1} - (1-q_{r_l}) * w_{j,l}
    #
    # And expected cost for j = d_j * [ sum_l c_{j,r_l} * (1-q_{r_l}) * w_{j,l} + p_j * z_{j,n} ]
    #                        = d_j * [ sum_l c_{j,r_l} * (1-q_{r_l}) * w_{j,l} + p_j * z_{j,n} ]
    #
    # McCormick relaxation for w_{j,l} = z_{j,l-1} * x_{r_l}:
    # w_{j,l} >= 0
    # w_{j,l} <= z_{j,l-1}
    # w_{j,l} <= x_{r_l}
    # w_{j,l} >= z_{j,l-1} + x_{r_l} - 1
    #
    # Since z_{j,l-1} in [0,1] and x_{r_l} in {0,1}, this is exact when x is binary!
    
    try:
        model = gp.Model("RFLP")
        model.Params.TimeLimit = max(remaining_time - 1, 1)
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        
        # Decision variables
        x = model.addVars(num_facilities, vtype=GRB.BINARY, name="x")
        
        # For each customer j, sorted facility indices
        sorted_facs = []
        for j in range(num_customers):
            order = sorted(range(num_facilities), key=lambda i: trans_cost_matrix[j][i])
            sorted_facs.append(order)
        
        # z[j][l] = probability that none of the first l sorted facilities serve customer j
        # z[j][0] = 1 (no facilities considered yet)
        # w[j][l] = z[j][l-1] * x[sorted_facs[j][l-1]]  (l from 1 to n)
        
        z = {}
        w = {}
        
        obj_expr = gp.LinExpr()
        
        # Add fixed costs
        for i in range(num_facilities):
            obj_expr += nodes[i]['fixed_setup_cost'] * x[i]
        
        for j in range(num_customers):
            order = sorted_facs[j]
            n = num_facilities
            d_j = nodes[j]['demand']
            
            # z[j,0] = 1 (constant)
            z[j, 0] = 1.0  # This is a constant
            
            for l in range(1, n + 1):
                rl = order[l - 1]  # facility index at level l
                q_rl = nodes[rl]['disruption_probability']
                s_rl = 1.0 - q_rl  # survival probability
                c_jrl = trans_cost_matrix[j][rl]
                
                # w[j,l] = z[j,l-1] * x[rl]
                w[j, l] = model.addVar(lb=0.0, ub=1.0, name=f"w_{j}_{l}")
                
                # z[j,l] = z[j,l-1] - s_rl * w[j,l]
                z[j, l] = model.addVar(lb=0.0, ub=1.0, name=f"z_{j}_{l}")
                
                if l == 1:
                    # z[j,0] = 1, so w[j,1] = x[rl]
                    model.addConstr(w[j, l] == x[rl])
                    model.addConstr(z[j, l] == 1.0 - s_rl * x[rl])
                else:
                    # McCormick: w[j,l] = z[j,l-1] * x[rl]
                    z_prev = z[j, l - 1]
                    model.addConstr(w[j, l] <= z_prev)
                    model.addConstr(w[j, l] <= x[rl])
                    model.addConstr(w[j, l] >= z_prev + x[rl] - 1.0)
                    # w[j,l] >= 0 already from lb
                    
                    model.addConstr(z[j, l] == z_prev - s_rl * w[j, l])
                
                # Contribution to objective: d_j * c_jrl * s_rl * w[j,l]
                obj_expr += d_j * c_jrl * s_rl * w[j, l]
            
            # Penalty contribution: d_j * p_j * z[j,n]
            obj_expr += d_j * penalty_cost[j] * z[j, n]
        
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Set initial solution if available
        if initial_open is not None:
            for i in range(num_facilities):
                x[i].Start = 1 if i in initial_open else 0
        
        model.optimize()
        
        if model.SolCount > 0:
            obj_val = model.ObjVal
            open_facs = [i for i in range(num_facilities) if x[i].X > 0.5]
            
            # Verify with exact computation
            fixed = sum(nodes[i]['fixed_setup_cost'] for i in open_facs)
            exp = compute_expected_cost(open_facs, nodes, trans_cost_matrix, penalty_cost, num_customers, num_facilities)
            true_obj = fixed + exp
            
            if logger:
                logger.log(true_obj)
            
            return true_obj, open_facs
        else:
            return initial_obj, initial_open
            
    except Exception as e:
        print(f"MIP solver error: {e}")
        return initial_obj, initial_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    nodes = instance['nodes']
    trans_cost_matrix = instance['transportation_cost_matrix']
    penalty_cost = instance['penalty_cost']
    
    # First, get a greedy solution
    greedy_obj, greedy_open = solve_greedy(nodes, trans_cost_matrix, penalty_cost, num_facilities, num_customers)
    if logger:
        logger.log(greedy_obj)
    
    best_obj = greedy_obj
    best_open = greedy_open
    
    # Local search to improve greedy
    ls_obj, ls_open = solve_local_search(nodes, trans_cost_matrix, penalty_cost, num_facilities, num_customers,
                                          best_open, best_obj, args.time_limit, start_time, logger)
    if ls_obj < best_obj:
        best_obj = ls_obj
        best_open = ls_open
    
    # For small instances, try enumeration
    if num_facilities <= 20:
        remaining = args.time_limit - (time.time() - start_time)
        if remaining > 2:
            enum_obj, enum_open = solve_enumeration(nodes, trans_cost_matrix, penalty_cost, num_facilities, num_customers,
                                                     args.time_limit, start_time, logger)
            if enum_obj < best_obj:
                best_obj = enum_obj
                best_open = enum_open
    else:
        # Use MIP for larger instances
        remaining = args.time_limit - (time.time() - start_time)
        if remaining > 2:
            mip_obj, mip_open = solve_mip(nodes, trans_cost_matrix, penalty_cost, num_facilities, num_customers,
                                           args.time_limit, start_time, logger, 
                                           initial_open=set(best_open), initial_obj=best_obj)
            if mip_obj is not None and mip_obj < best_obj:
                best_obj = mip_obj
                best_open = mip_open
    
    # Build solution
    facility_locations = [0] * num_facilities
    for i in best_open:
        facility_locations[i] = 1
    
    solution = {
        "objective_value": best_obj,
        "facility_locations": facility_locations
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()