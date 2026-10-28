import argparse
import json
import math
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB


def calculate_euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def solve_partial_set_covering_location_problem(num_facilities, num_customers, 
                                               facility_cost, customer_demands, 
                                               facility_to_customers_coverage,
                                               covering_demand_D):
    """
    Solve the Partial Set Covering Location Problem.
    
    Minimize total opening cost while ensuring at least covering_demand_D is covered.
    """
    # Create model
    model = gp.Model("PSCLP")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    # Decision variables
    x = model.addVars(num_facilities, vtype=GRB.BINARY, name="x")  # Open facility i
    y = model.addVars(num_customers, vtype=GRB.BINARY, name="y")  # Customer j is covered
    
    # Objective function: minimize total opening cost
    model.setObjective(gp.quicksum(facility_cost[i] * x[i] for i in range(num_facilities)), GRB.MINIMIZE)
    
    # Constraint: minimum demand threshold must be met
    model.addConstr(gp.quicksum(customer_demands[j] * y[j] for j in range(num_customers)) >= covering_demand_D, 
                    name="demand_threshold")
    
    # Constraint: a customer can only be covered if at least one of its covering facilities is open
    for j in range(num_customers):
        covering_facilities = []
        if str(j) in facility_to_customers_coverage:
            for i in range(num_facilities):
                if str(i) in facility_to_customers_coverage and j in facility_to_customers_coverage[str(i)]:
                    covering_facilities.append(i)
        else:
            # Check all facilities for this customer
            for i in range(num_facilities):
                covering_facilities.append(i)
        
        # Rebuild the list properly
        covering_facilities = []
        for i in range(num_facilities):
            if str(i) in facility_to_customers_coverage and j in facility_to_customers_coverage[str(i)]:
                covering_facilities.append(i)
                
        if covering_facilities:
            model.addConstr(gp.quicksum(x[i] for i in covering_facilities) >= y[j], 
                            name=f"coverage_{j}")
        else:
            # If no facility can cover this customer, then y[j] must be 0
            model.addConstr(y[j] == 0, name=f"no_coverage_{j}")
    
    # Optimize the model
    model.optimize()
    
    # Extract solution
    open_facilities = [i for i in range(num_facilities) if x[i].x > 0.5]
    covered_demand = sum(customer_demands[j] for j in range(num_customers) if y[j].x > 0.5)
    objective_value = model.objVal if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT else float('inf')
    
    return {
        "objective_value": objective_value,
        "open_facilities": open_facilities,
        "covered_demand": covered_demand
    }


def solve_maximal_covering_location_problem(num_facilities, num_customers, 
                                           facility_cost, customer_demands, 
                                           facility_to_customers_coverage,
                                           budget_B):
    """
    Solve the Maximal Covering Location Problem.
    
    Maximize total covered demand while staying within budget B.
    """
    # Create model
    model = gp.Model("MCLP")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    # Decision variables
    x = model.addVars(num_facilities, vtype=GRB.BINARY, name="x")  # Open facility i
    y = model.addVars(num_customers, vtype=GRB.BINARY, name="y")  # Customer j is covered
    
    # Objective function: maximize total covered demand
    model.setObjective(gp.quicksum(customer_demands[j] * y[j] for j in range(num_customers)), GRB.MAXIMIZE)
    
    # Constraint: total opening cost must not exceed budget
    model.addConstr(gp.quicksum(facility_cost[i] * x[i] for i in range(num_facilities)) <= budget_B, 
                    name="budget_constraint")
    
    # Constraint: a customer can only be covered if at least one of its covering facilities is open
    for j in range(num_customers):
        covering_facilities = []
        for i in range(num_facilities):
            if str(i) in facility_to_customers_coverage and j in facility_to_customers_coverage[str(i)]:
                covering_facilities.append(i)
                
        if covering_facilities:
            model.addConstr(gp.quicksum(x[i] for i in covering_facilities) >= y[j], 
                            name=f"coverage_{j}")
        else:
            # If no facility can cover this customer, then y[j] must be 0
            model.addConstr(y[j] == 0, name=f"no_coverage_{j}")
    
    # Optimize the model
    model.optimize()
    
    # Extract solution
    open_facilities = [i for i in range(num_facilities) if x[i].x > 0.5]
    covered_demand = sum(customer_demands[j] for j in range(num_customers) if y[j].x > 0.5)
    objective_value = model.objVal if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT else 0.0
    
    return {
        "objective_value": objective_value,
        "open_facilities": open_facilities,
        "covered_demand": covered_demand
    }


def main():
    parser = argparse.ArgumentParser(description='Solve Facility Location Problems')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to save the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file')
    
    args = parser.parse_args()
    
    # Initialize logger
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Extract parameters
    num_facilities = instance_data['num_facilities']
    num_customers = instance_data['num_customers']
    budget_B = instance_data['budget_B']
    covering_demand_D = instance_data['covering_demand_D']
    radius_of_coverage = instance_data['radius_of_coverage']
    facility_cost = instance_data['facility_cost']
    facility_coordinates = instance_data['facility_coordinates']
    customer_coordinates = instance_data['customer_coordinates']
    customer_demands = instance_data['customer_demands']
    facility_to_customers_coverage = {int(k): v for k, v in instance_data['facility_to_customers_coverage'].items()}
    customer_to_facilities_coverage = {int(k): v for k, v in instance_data['customer_to_facilities_coverage'].items()}
    
    start_time = time.time()
    
    # Solve Partial Set Covering Location Problem
    psclp_result = solve_partial_set_covering_location_problem(
        num_facilities, num_customers, facility_cost, customer_demands,
        facility_to_customers_coverage, covering_demand_D
    )
    
    # Update time remaining for next problem
    elapsed_time = time.time() - start_time
    remaining_time = max(0, args.time_limit - elapsed_time)
    
    # Solve Maximal Covering Location Problem
    mclp_result = solve_maximal_covering_location_problem(
        num_facilities, num_customers, facility_cost, customer_demands,
        facility_to_customers_coverage, budget_B
    )
    
    # Prepare final solution
    solution = {
        "objective_value": psclp_result["objective_value"],  # Using PSCLP as primary by default
        "primary_problem_type": "PSCLP",
        "results": {
            "PSCLP": psclp_result,
            "MCLP": mclp_result
        }
    }
    
    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
    

if __name__ == "__main__":
    main()