import argparse
import json
import time
import itertools
from gurobipy import Model, GRB, quicksum

def solve_instance(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    start_time = time.time()
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_facilities = data['num_facilities']
    num_customers = data['num_customers']
    demands = data['demands']
    capacities = data['capacities']
    fixed_costs_obj1 = data['fixed_costs_obj1']
    fixed_costs_obj2 = data['fixed_costs_obj2']
    assignment_costs_obj1 = data['assignment_costs_obj1']
    assignment_costs_obj2 = data['assignment_costs_obj2']
    
    F = range(num_facilities)
    C = range(num_customers)
    
    # We need to find the complete Pareto front for the bi-objective problem.
    # Approach: Use the epsilon-constraint method.
    # First solve for min obj1, then min obj2, then use epsilon constraints to find all non-dominated points.
    
    pareto_solutions = []  # list of (z1, z2, open_facilities, assignments)
    
    best_combined = float('inf')
    best_combined_solution = None
    
    def solve_weighted(w1, w2, z1_upper=None, z2_upper=None, time_remaining=None):
        """Solve weighted sum problem with optional epsilon constraints."""
        m = Model("facility")
        m.setParam('OutputFlag', 0)
        m.setParam('Threads', 1)
        if time_remaining is not None:
            m.setParam('TimeLimit', max(1, time_remaining))
        
        # Variables
        y = {}
        for i in F:
            y[i] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        
        x = {}
        for i in F:
            for j in C:
                x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        
        m.update()
        
        # Constraints: each customer assigned to exactly one facility
        for j in C:
            m.addConstr(quicksum(x[i, j] for i in F) == 1, name=f"assign_{j}")
        
        # Capacity constraints
        for i in F:
            m.addConstr(quicksum(demands[j] * x[i, j] for j in C) <= capacities[i] * y[i], name=f"cap_{i}")
        
        # Linking constraints
        for i in F:
            for j in C:
                m.addConstr(x[i, j] <= y[i], name=f"link_{i}_{j}")
        
        # Objective expressions
        obj1_expr = quicksum(fixed_costs_obj1[i] * y[i] for i in F) + quicksum(assignment_costs_obj1[i][j] * x[i, j] for i in F for j in C)
        obj2_expr = quicksum(fixed_costs_obj2[i] * y[i] for i in F) + quicksum(assignment_costs_obj2[i][j] * x[i, j] for i in F for j in C)
        
        # Epsilon constraints
        if z1_upper is not None:
            m.addConstr(obj1_expr <= z1_upper, name="eps_z1")
        if z2_upper is not None:
            m.addConstr(obj2_expr <= z2_upper, name="eps_z2")
        
        # Weighted objective
        m.setObjective(w1 * obj1_expr + w2 * obj2_expr, GRB.MINIMIZE)
        
        m.optimize()
        
        if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
            if m.SolCount > 0:
                z1_val = int(round(sum(fixed_costs_obj1[i] * y[i].X for i in F) + sum(assignment_costs_obj1[i][j] * x[i, j].X for i in F for j in C)))
                z2_val = int(round(sum(fixed_costs_obj2[i] * y[i].X for i in F) + sum(assignment_costs_obj2[i][j] * x[i, j].X for i in F for j in C)))
                
                open_fac = [i for i in F if y[i].X > 0.5]
                assignments = {}
                for j in C:
                    for i in F:
                        if x[i, j].X > 0.5:
                            assignments[str(j)] = i
                            break
                
                return z1_val, z2_val, open_fac, assignments
        return None
    
    def add_to_pareto(z1, z2, open_fac, assignments):
        nonlocal pareto_solutions, best_combined, best_combined_solution
        # Check if dominated by existing
        dominated = False
        to_remove = []
        for idx, (pz1, pz2, pof, pa) in enumerate(pareto_solutions):
            if pz1 <= z1 and pz2 <= z2 and (pz1 < z1 or pz2 < z2):
                dominated = True
                break
            if z1 <= pz1 and z2 <= pz2 and (z1 < pz1 or z2 < pz2):
                to_remove.append(idx)
            if z1 == pz1 and z2 == pz2:
                dominated = True
                break
        
        if not dominated:
            for idx in sorted(to_remove, reverse=True):
                pareto_solutions.pop(idx)
            pareto_solutions.append((z1, z2, open_fac, assignments))
            
            combined = 0.5 * z1 + 0.5 * z2
            if combined < best_combined:
                best_combined = combined
                best_combined_solution = (z1, z2, open_fac, assignments)
                if logger:
                    logger.log(combined)
    
    # Step 1: Minimize obj1 alone
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining < 1:
        remaining = 1
    
    result = solve_weighted(1.0, 0.0, time_remaining=remaining)
    if result is None:
        # Fallback: try equal weight
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        result = solve_weighted(0.5, 0.5, time_remaining=max(1, remaining))
        if result:
            z1, z2, of, asgn = result
            add_to_pareto(z1, z2, of, asgn)
    else:
        z1_min, z2_at_z1min, of1, asgn1 = result
        add_to_pareto(z1_min, z2_at_z1min, of1, asgn1)
    
    # Step 2: Minimize obj2 alone
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining > 2:
        result = solve_weighted(0.0, 1.0, time_remaining=remaining)
        if result:
            z1_at_z2min, z2_min, of2, asgn2 = result
            add_to_pareto(z1_at_z2min, z2_min, of2, asgn2)
    
    # Step 3: Minimize equal weighted sum
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining > 2:
        result = solve_weighted(0.5, 0.5, time_remaining=remaining)
        if result:
            z1, z2, of, asgn = result
            add_to_pareto(z1, z2, of, asgn)
    
    # Step 4: Epsilon-constraint method to find all Pareto-optimal points
    # Sort current pareto solutions by z1
    def get_pareto_sorted():
        ps = list(pareto_solutions)
        ps.sort(key=lambda x: (x[0], -x[1]))
        return ps
    
    # Use a recursive approach: for each pair of adjacent Pareto points, try to find intermediate points
    def find_intermediate(z1_low, z2_high, z1_high, z2_low, depth=0):
        """Find non-dominated points between (z1_low, z2_high) and (z1_high, z2_low)."""
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 3:
            return
        if depth > 50:
            return
        if z1_high - z1_low <= 1:
            return
        if z2_high - z2_low <= 1:
            return
        
        # Solve with epsilon constraint: z1 <= z1_high - 1 and z2 <= z2_high - 1
        # Minimize z2 with z1 < z1_high
        result = solve_weighted(0.0, 1.0, z1_upper=z1_high - 1, z2_upper=z2_high - 1, time_remaining=min(remaining - 1, max(remaining * 0.5, 10)))
        if result:
            z1_new, z2_new, of_new, asgn_new = result
            if z1_new < z1_high and z2_new < z2_high:
                add_to_pareto(z1_new, z2_new, of_new, asgn_new)
                # Recurse on both sides
                find_intermediate(z1_low, z2_high, z1_new, z2_new, depth + 1)
                find_intermediate(z1_new, z2_new, z1_high, z2_low, depth + 1)
    
    # Also use: minimize z1 with z2 constraint for the other direction
    def find_intermediate_v2(z1_low, z2_high, z1_high, z2_low, depth=0):
        """Find non-dominated points between two adjacent Pareto points using both directions."""
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 3:
            return
        if depth > 50:
            return
        if z1_high - z1_low <= 1 or z2_high - z2_low <= 1:
            return
        
        # Try to find a point strictly between these two
        # Method: minimize weighted sum to find point in between
        # Or use epsilon constraint: minimize z2 subject to z1 <= z1_high - 1
        result = solve_weighted(0.0, 1.0, z1_upper=z1_high - 1, time_remaining=min(remaining - 1, max(10, remaining * 0.3)))
        if result:
            z1_new, z2_new, of_new, asgn_new = result
            if z1_new >= z1_low and z1_new < z1_high and z2_new > z2_low and z2_new <= z2_high:
                # Check if it's actually between the two points
                if z1_new > z1_low or z2_new < z2_high:
                    add_to_pareto(z1_new, z2_new, of_new, asgn_new)
                    # Recurse
                    if z1_new > z1_low and z2_new < z2_high:
                        find_intermediate_v2(z1_low, z2_high, z1_new, z2_new, depth + 1)
                    if z1_new < z1_high - 1 and z2_new > z2_low + 1:
                        find_intermediate_v2(z1_new, z2_new, z1_high, z2_low, depth + 1)
    
    # Now do the epsilon-constraint search between adjacent pairs
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining > 5:
        # Sort pareto solutions by z1
        ps = get_pareto_sorted()
        # Use a queue-based approach
        pairs_to_explore = []
        for idx in range(len(ps) - 1):
            pairs_to_explore.append((ps[idx][0], ps[idx][1], ps[idx+1][0], ps[idx+1][1]))
        
        while pairs_to_explore:
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            if remaining < 3:
                break
            
            z1_low, z2_high, z1_high, z2_low = pairs_to_explore.pop(0)
            
            if z1_high - z1_low <= 1 or z2_high - z2_low <= 1:
                continue
            
            per_solve_time = min(max(5, remaining * 0.2), remaining - 2)
            
            # Minimize z2 with z1 <= z1_high - 1
            result = solve_weighted(0.0, 1.0, z1_upper=z1_high - 1, time_remaining=per_solve_time)
            if result:
                z1_new, z2_new, of_new, asgn_new = result
                if z1_new < z1_high and z2_new < z2_high:
                    add_to_pareto(z1_new, z2_new, of_new, asgn_new)
                    # Add new pairs
                    pairs_to_explore.append((z1_low, z2_high, z1_new, z2_new))
                    pairs_to_explore.append((z1_new, z2_new, z1_high, z2_low))
    
    # Additional: try various weight combinations to find more Pareto points
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining > 10:
        weights_to_try = [(0.25, 0.75), (0.75, 0.25), (0.1, 0.9), (0.9, 0.1), 
                          (0.33, 0.67), (0.67, 0.33), (0.15, 0.85), (0.85, 0.15),
                          (0.4, 0.6), (0.6, 0.4)]
        for w1, w2 in weights_to_try:
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            if remaining < 5:
                break
            per_solve_time = min(max(5, remaining * 0.15), remaining - 2)
            result = solve_weighted(w1, w2, time_remaining=per_solve_time)
            if result:
                z1, z2, of, asgn = result
                add_to_pareto(z1, z2, of, asgn)
    
    # After finding new points from weights, do another round of epsilon-constraint
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining > 5:
        ps = get_pareto_sorted()
        pairs_to_explore = []
        for idx in range(len(ps) - 1):
            pairs_to_explore.append((ps[idx][0], ps[idx][1], ps[idx+1][0], ps[idx+1][1]))
        
        while pairs_to_explore:
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            if remaining < 3:
                break
            
            z1_low, z2_high, z1_high, z2_low = pairs_to_explore.pop(0)
            
            if z1_high - z1_low <= 1 or z2_high - z2_low <= 1:
                continue
            
            per_solve_time = min(max(5, remaining * 0.15), remaining - 2)
            
            result = solve_weighted(0.0, 1.0, z1_upper=z1_high - 1, time_remaining=per_solve_time)
            if result:
                z1_new, z2_new, of_new, asgn_new = result
                if z1_new < z1_high and z2_new < z2_high:
                    add_to_pareto(z1_new, z2_new, of_new, asgn_new)
                    pairs_to_explore.append((z1_low, z2_high, z1_new, z2_new))
                    pairs_to_explore.append((z1_new, z2_new, z1_high, z2_low))
    
    # Also try reverse epsilon constraint direction
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining > 5:
        ps = get_pareto_sorted()
        pairs_to_explore = []
        for idx in range(len(ps) - 1):
            pairs_to_explore.append((ps[idx][0], ps[idx][1], ps[idx+1][0], ps[idx+1][1]))
        
        while pairs_to_explore:
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            if remaining < 3:
                break
            
            z1_low, z2_high, z1_high, z2_low = pairs_to_explore.pop(0)
            
            if z1_high - z1_low <= 1 or z2_high - z2_low <= 1:
                continue
            
            per_solve_time = min(max(5, remaining * 0.15), remaining - 2)
            
            # Minimize z1 with z2 <= z2_high - 1
            result = solve_weighted(1.0, 0.0, z2_upper=z2_high - 1, time_remaining=per_solve_time)
            if result:
                z1_new, z2_new, of_new, asgn_new = result
                if z1_new < z1_high and z2_new < z2_high:
                    # Check if this is truly new
                    is_new = True
                    for pz1, pz2, _, _ in pareto_solutions:
                        if pz1 == z1_new and pz2 == z2_new:
                            is_new = False
                            break
                    if is_new:
                        add_to_pareto(z1_new, z2_new, of_new, asgn_new)
                        pairs_to_explore.append((z1_low, z2_high, z1_new, z2_new))
                        pairs_to_explore.append((z1_new, z2_new, z1_high, z2_low))
    
    # Prepare output
    # Final cleanup of Pareto front
    final_pareto = []
    pareto_solutions.sort(key=lambda x: (x[0], x[1]))
    for sol in pareto_solutions:
        dominated = False
        for other in pareto_solutions:
            if other is sol:
                continue
            if other[0] <= sol[0] and other[1] <= sol[1] and (other[0] < sol[0] or other[1] < sol[1]):
                dominated = True
                break
        if not dominated:
            # Check for duplicates
            already = False
            for fp in final_pareto:
                if fp[0] == sol[0] and fp[1] == sol[1]:
                    already = True
                    break
            if not already:
                final_pareto.append(sol)
    
    final_pareto.sort(key=lambda x: (x[0], -x[1]))
    
    # Find the one that minimizes 0.5*z1 + 0.5*z2
    best_idx = 0
    best_val = float('inf')
    for idx, (z1, z2, of, asgn) in enumerate(final_pareto):
        val = 0.5 * z1 + 0.5 * z2
        if val < best_val:
            best_val = val
            best_idx = idx
    
    # Build output
    pareto_front = [[sol[0], sol[1]] for sol in final_pareto]
    
    solutions = []
    for z1, z2, of, asgn in final_pareto:
        solutions.append({
            "z1": z1,
            "z2": z2,
            "open_facilities": sorted(of),
            "assignments": asgn
        })
    
    output = {
        "objective_value": best_val,
        "pareto_front": pareto_front,
        "solutions": solutions
    }
    
    with open(solution_path, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve_instance(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()