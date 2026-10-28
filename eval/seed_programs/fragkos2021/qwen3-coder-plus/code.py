import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import product
import random
import sys


def solve_disruption_problem(instance_data, time_limit):
    # Parse input data
    num_nodes = instance_data["num_nodes"]
    num_facilities = instance_data["num_facilities"]
    num_customers = instance_data["num_customers"]
    nodes_data = instance_data["nodes"]
    transportation_cost_matrix = np.array(instance_data["transportation_cost_matrix"]).tolist()
    penalty_cost = instance_data["penalty_cost"]

    # Extract node parameters
    demands = [node["demand"] for node in nodes_data]
    fixed_setup_costs = [node["fixed_setup_cost"] for node in nodes_data[:num_facilities]]
    disruption_probabilities = [node["disruption_probability"] for node in nodes_data[:num_facilities]]
    survival_probabilities = [node["survival_probability"] for node in nodes_data[:num_facilities]]

    # Validate dimensions
    assert len(fixed_setup_costs) == num_facilities
    assert len(disruption_probabilities) == num_facilities
    assert len(survival_probabilities) == num_facilities
    assert len(demands) == num_customers
    assert len(penalty_cost) == num_customers
    assert len(transportation_cost_matrix) == num_customers
	
    for i in range(num_customers):
        assert len(transportation_cost_matrix[i]) == num_facilities

    # Model creation
    model = gp.Model("FacilityLocationWithDisruptions")
    model.setParam("Threads", 1)
    
    # Set time limit
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1)  # Enable logging

    # Decision variables
    # x[j]: whether facility j is opened (binary)
    x = model.addVars(num_facilities, vtype=GRB.BINARY, name='x')
    
    # For stochastic approach, we need to consider scenarios of disruptions
    # To manage complexity, take a limited number of scenarios
    max_scenarios = min(2**num_facilities, 256)  # Use at most 256 scenarios
    scenario_indices = []
    all_operational_combinations = []

    # Enumerate scenarios up to maximum limit
    n_scenarios = min(2**num_facilities, max_scenarios)
    for idx in range(n_scenarios):
        op_state = [(idx >> j) & 1 for j in range(num_facilities)]
        all_operational_combinations.append(op_state)
        scenario_indices.append(idx)

    # Calculate probability for each scenario - approximating independence
    scenario_probs = []
    for op_state in all_operational_combinations:
        prob = 1.0
        for j in range(num_facilities):
            if op_state[j] == 1:
                prob *= survival_probabilities[j]
            else:
                prob *= (1 - survival_probabilities[j])
        scenario_probs.append(prob)

    # Normalize probs just in case
    total_prob = sum(scenario_probs)
    scenario_probs = [p / total_prob for p in scenario_probs]

    # Variables for assignments in each scenario
    z_customer_facility_scenario = {}
    z_customer_penalty_scenario = {}
    
    for s_idx in range(len(all_operational_combinations)):
        op_state = all_operational_combinations[s_idx]  # Operational status of each facility
        sprob = scenario_probs[s_idx]
        
        for i in range(num_customers):
            # Binary variable: y[i,j,s] = 1 if customer i served by facility j in scenario s
            for j in range(num_facilities):
                # Only define for operational facilities in this scenario
                if op_state[j] == 1:
                    key = (i, j, s_idx)
                    z_customer_facility_scenario[key] = model.addVar(vtype=GRB.BINARY, name=f"z_cf_{i}_{j}_{s_idx}")
            
            # For penalty cost option when all opened facilities for customer are disrupted
            z_customer_penalty_scenario[(i, s_idx)] = model.addVar(vtype=GRB.BINARY, name=f"z_cp_{i}_{s_idx}")

    # Constraints
    # 1. Each customer is served by exactly one facility or penalty in each scenario
    for s_idx in range(len(all_operational_combinations)):
        for i in range(num_customers):
            c_assign = model.addConstr(gp.LinExpr(sum(z_customer_facility_scenario.get((i, j, s_idx), 0) for j in range(num_facilities) if all_operational_combinations[s_idx][j] == 1) + z_customer_penalty_scenario[(i, s_idx)]) == 1, 
                                      name=f"customer_serve_constraint_{i}_{s_idx}")

    # 2. Customer can only be served by an opened facility that is operational
    for s_idx in range(len(all_operational_combinations)):
        scenario_state = all_operational_combinations[s_idx]
        for i in range(num_customers):
            for j in range(num_facilities):
                if scenario_state[j] == 1:  # j is operational in this scenario
                    model.addConstr(z_customer_facility_scenario[(i, j, s_idx)] <= x[j], 
                                   name=f"opened_facility_constraint_{i}_{j}_{s_idx}")

    # Objective function
    # Fixed costs + expected variable costs
    fixed_cost = gp.quicksum(fixed_setup_costs[j] * x[j] for j in range(num_facilities))
    
    expected_var_cost = gp.LinExpr(0.0)
    for s_idx in range(len(all_operational_combinations)):
        s_prob = scenario_probs[s_idx]
        for i in range(num_customers):
            for j in range(num_facilities):
                if all_operational_combinations[s_idx][j] == 1 and (i, j, s_idx) in z_customer_facility_scenario:
                    expected_var_cost.addTerms(s_prob * demands[i] * transportation_cost_matrix[i][j],
                                               z_customer_facility_scenario[(i, j, s_idx)])
            
            # Add penalty cost terms
            expected_var_cost.addTerms(s_prob * demands[i] * penalty_cost[i],
                                       z_customer_penalty_scenario[(i, s_idx)])
    
    model.setObjective(fixed_cost + expected_var_cost, GRB.MINIMIZE)
    
    try:
        model.optimize()
    except Exception as e:
        print(f"Gurobi optimization error: {e}", file=sys.stderr)
        return {
            "facility_locations": [0 for _ in range(num_facilities)],
            "objective_value": float('inf')
        }

    # Check solution status
    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        # Extract solution
        result_x = [int(x[j].X) for j in range(num_facilities)]
        obj_val = model.ObjVal

        return {
            "facility_locations": result_x,
            "objective_value": obj_val
        }
    else:
        # No feasible solution found within time limit
        return {
            "facility_locations": [0 for _ in range(num_facilities)],
            "objective_value": float('inf')
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, help="Path to log file (optional)")
    
    args = parser.parse_args()
    
    # Initialize logger if path provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Load problem instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the problem
    solution = solve_disruption_problem(instance_data, args.time_limit)
    
    # Log the result if logger is available
    if logger and solution["objective_value"] != float('inf'):
        logger.log(solution["objective_value"])

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()