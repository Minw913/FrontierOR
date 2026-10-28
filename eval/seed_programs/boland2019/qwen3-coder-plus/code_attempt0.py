import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
from collections import defaultdict
import itertools


def solve_multi_objective_facility_location(instance_data, time_limit, logger=None):
    num_facilities = instance_data['num_facilities']
    num_customers = instance_data['num_customers']
    demands = instance_data['demands']
    capacities = instance_data['capacities']
    fixed_costs_obj1 = instance_data['fixed_costs_obj1']
    fixed_costs_obj2 = instance_data['fixed_costs_obj2']
    assignment_costs_obj1 = instance_data['assignment_costs_obj1']
    assignment_costs_obj2 = instance_data['assignment_costs_obj2']

    # Create model
    model = gp.Model("multi_objective_facility_location")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)  # Silent mode
    
    # Variables - binary variables for which facilities to open
    y = model.addVars(num_facilities, vtype=GRB.BINARY, name="open_facility")
    
    # Variables - binary variables for assignment of customers to facilities
    x = model.addVars(num_facilities, num_customers, vtype=GRB.BINARY, name="assign_customer")

    # Constraints
    # Each customer must be assigned to exactly one facility
    for j in range(num_customers):
        model.addConstr(gp.quicksum(x[i, j] for i in range(num_facilities)) == 1, 
                       name=f"assign_customer_{j}")

    # Capacity constraints - the sum of demands assigned to a facility cannot exceed its capacity if it's open
    for i in range(num_facilities):
        lhs_expr = gp.LinExpr()
        for j in range(num_customers):
            lhs_expr.addTerms(demands[j], x[i, j])
        rhs_expr = gp.LinExpr([(capacities[i], y[i])])
        model.addConstr(lhs_expr <= rhs_expr, name=f"capacity_{i}")

    # Link assignment to facility opening - if facility is not open, no customers can be assigned to it
    for i in range(num_facilities):
        for j in range(num_customers):
            model.addConstr(x[i, j] <= y[i], name=f"link_{i}_{j}")
    
    # Define objectives
    # Objective 1: minimize total cost including fixed facility costs and assignment costs
    obj1 = (
        gp.quicksum(fixed_costs_obj1[i] * y[i] for i in range(num_facilities)) +
        gp.quicksum(assignment_costs_obj1[i][j] * x[i, j] 
                   for i in range(num_facilities) for j in range(num_customers))
    )
    
    # Objective 2: minimize total cost for second cost structure
    obj2 = (
        gp.quicksum(fixed_costs_obj2[i] * y[i] for i in range(num_facilities)) +
        gp.quicksum(assignment_costs_obj2[i][j] * x[i, j] 
                   for i in range(num_facilities) for j in range(num_customers))
    )

    # Compute the Pareto front by solving multiple scalarized problems
    pareto_front = []
    solutions_list = []
    
    # Step 1: Find extreme points of the Pareto front (minimizing each objective separately)
    # Store the models for objective-specific problems to reuse them later
    model.reset()
    model.setObjective(obj1, GRB.MINIMIZE)
    model.optimize()
    
    z1_min_obj1 = int(model.objVal) if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.solCount > 0 else float('inf')
    sol1_z2 = float('inf')
    
    if model.solCount > 0:
        sol1_assignments = {}
        for j in range(num_customers):
            for i in range(num_facilities):
                if x[i, j].x > 0.99:  # Binary values
                    sol1_assignments[str(j)] = i
        
        sol1_open = [i for i in range(num_facilities) if y[i].x > 0.99]
        
        # Calculate z2 for the solution minimizing z1
        sol1_z2 = sum(fixed_costs_obj2[i] * y[i].x for i in range(num_facilities)) + \
                  sum(assignment_costs_obj2[i][j] * x[i, j].x for i in range(num_facilities) for j in range(num_customers))
        sol1_z2 = int(sol1_z2)
        solutions_list.append({
            "z1": z1_min_obj1,
            "z2": sol1_z2,
            "open_facilities": sol1_open,
            "assignments": sol1_assignments
        })

    model.reset()
    model.setObjective(obj2, GRB.MINIMIZE)
    model.optimize()

    z2_min_obj2 = int(model.objVal) if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.solCount > 0 else float('inf')
    sol2_z1 = float('inf')
    
    if model.solCount > 0:
        sol2_assignments = {}
        for j in range(num_customers):
            for i in range(num_facilities):
                if x[i, j].x > 0.99:  # Binary values
                    sol2_assignments[str(j)] = i
        
        sol2_open = [i for i in range(num_facilities) if y[i].x > 0.99]

        # Calculate z1 for the solution minimizing z2
        sol2_z1 = sum(fixed_costs_obj1[i] * y[i].x for i in range(num_facilities)) + \
                  sum(assignment_costs_obj1[i][j] * x[i, j].x for i in range(num_facilities) for j in range(num_customers))
        sol2_z1 = int(sol2_z1)
        
        # Check if solution is already in solutions list to avoid duplicates
        if sol2_z1 != z1_min_obj1 or sol2_z1 >= float('inf'):
            solutions_list.append({
                "z1": sol2_z1,
                "z2": z2_min_obj2,
                "open_facilities": sol2_open,
                "assignments": sol2_assignments
            })
    
    # Add extreme points to the Pareto front
    if z1_min_obj1 < float('inf') and sol1_z2 < float('inf'):
        pareto_front.append([z1_min_obj1, sol1_z2])

    if z2_min_obj2 < float('inf'):
        existing = False
        for point in pareto_front:
            if point[0] == sol2_z1 and point[1] == z2_min_obj2:
                existing = True
                break
        if not existing:
            pareto_front.append([sol2_z1, z2_min_obj2])
    
    # Step 2: Generate additional Pareto-optimal solutions by scalarization
    # Add constraints and alternate weights in scalarized objective
    weights = [(i / 10.0, (10 - i) / 10.0) for i in range(8, 0, -1)]
    
    visited_solutions = set()
    
    for weight1, weight2 in weights:
        model.reset()
        scalarized_obj = weight1 * obj1 + weight2 * obj2
        model.setObjective(scalarized_obj, GRB.MINIMIZE)
        model.optimize()
        
        if model.solCount > 0:
            z1_val = sum(fixed_costs_obj1[i] * y[i].x for i in range(num_facilities)) + \
                     sum(assignment_costs_obj1[i][j] * x[i, j].x for i in range(num_facilities) for j in range(num_customers))
            
            z2_val = sum(fixed_costs_obj2[i] * y[i].x for i in range(num_facilities)) + \
                     sum(assignment_costs_obj2[i][j] * x[i, j].x for i in range(num_facilities) for j in range(num_customers))
            
            z1_val, z2_val = int(z1_val), int(z2_val)
            
            # Create signature to avoid duplicate solutions
            assignment_sig = str(sorted([f"{j}-{int(x[i,j].x*100)}-{i}" for i in range(num_facilities) for j in range(num_customers)]))
            sig = f"{z1_val}-{z2_val}-{assignment_sig}"
            
            if sig not in visited_solutions:
                visited_solutions.add(sig)
                
                assignments = {}
                for j in range(num_customers):
                    for i in range(num_facilities):
                        if x[i, j].x > 0.99:
                            assignments[str(j)] = i
                
                open_facilities = [i for i in range(num_facilities) if y[i].x > 0.99]
                
                # Add solution if it's not dominated by existing
                dominated = False
                for pf_point in pareto_front:
                    # Current solution is dominated if there exists a point in PF that is better or equal in both,
                    # and strictly better in at least one
                    if (pf_point[0] <= z1_val and pf_point[1] < z2_val) or \
                       (pf_point[0] < z1_val and pf_point[1] <= z2_val):
                        dominated = True
                        break
                        
                if not dominated:
                    # Remove any points in front that are now dominated by current
                    updated_pareto = []
                    for pf_point in pareto_front:
                        if (z1_val <= pf_point[0] and z2_val < pf_point[1]) or \
                           (z1_val < pf_point[0] and z2_val <= pf_point[1]):
                            continue  # This PF point is dominated by current
                        else:
                            updated_pareto.append(pf_point)
                    
                    pareto_front = updated_pareto
                    pareto_front.append([z1_val, z2_val])
                
                    existing_solution = False
                    for s in solutions_list:
                        if s["z1"] == z1_val and s["z2"] == z2_val:
                            existing_solution = True
                            break
                    if not existing_solution:
                        solutions_list.append({
                            "z1": z1_val,
                            "z2": z2_val,
                            "open_facilities": sorted(open_facilities),
                            "assignments": assignments
                        })
    
    # Final step: Identify non-dominated solutions and update logging after each improvement
    combined_costs = [(point[0] + point[1])/2.0 for point in pareto_front]
    if combined_costs:
        min_combined_cost = min(combined_costs)
        objective_value = min_combined_cost
    else:
        # Default: in case nothing found
        objective_value = float('inf')

    return {
        "objective_value": objective_value,
        "pareto_front": pareto_front,
        "solutions": solutions_list
    }


def main():
    parser = argparse.ArgumentParser(description='Solve Multi-Objective Facility Location Problem')
    parser.add_argument('--instance_path', required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', help='Path to the log JSONL file (optional)')
    
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_multi_objective_facility_location(instance_data, args.time_limit, logger)
    
    # Log objective value if available
    if logger and solution["objective_value"] < float('inf'):
        logger.log(solution["objective_value"])

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()