import argparse
import json
import time
import numpy as np
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_one_stage_ip_mu(scenarios, K, time_limit, remaining_time):
    """
    Solve the one-stage IP with binary assignment (mu) and continuous lambda weights.
    """
    start_time = time.time()
    
    # Ensure we don't exceed the remaining time
    actual_time_limit = min(remaining_time, time_limit)
    if actual_time_limit <= 0:
        return None
    
    N, n = len(scenarios), len(scenarios[0])
    
    # Create model
    model = gp.Model("one_stage_scenario_reduction")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', actual_time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    # Variables
    # Binary variables for which scenarios are selected as aggregates
    y = model.addVars(N, vtype=GRB.BINARY, name="y")
    
    # Binary assignment variables: mu[i,k] = 1 if scenario i is assigned to aggregate k
    mu = model.addVars(N, K, vtype=GRB.BINARY, name="mu")
    
    # Continuous variables for convex combination weights
    lambda_vars = model.addVars(N, K, vtype=GRB.CONTINUOUS, lb=0.0, name="lambda")
    
    # Scaling factor t (to maximize)
    t = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="t")
    
    # Constraints
    # Exactly K scenarios must be selected as aggregates
    model.addConstr(gp.quicksum(y) == K, name="select_K_aggregates")
    
    # Each original scenario must be assigned to exactly one aggregate
    for i in range(N):
        model.addConstr(gp.quicksum(mu[i, k] for k in range(K)) == 1, name=f"assign_scenario_{i}")
    
    # Lambda weights must sum to 1 for each aggregate
    for k in range(K):
        for j in range(n):
            model.addConstr(
                gp.quicksum(lambda_vars[i, k] * scenarios[i][j] for i in range(N)) >= 
                t * scenarios[i][j] for i in range(N) if mu[i, k].x == 1
            )
            
    # Lambda weights are only non-zero for scenarios assigned to an aggregate
    for i in range(N):
        for k in range(K):
            model.addConstr(lambda_vars[i, k] <= y[i], name=f"lambda_support_{i}_{k}")
            model.addConstr(lambda_vars[i, k] <= mu[i, k], name=f"lambda_assign_{i}_{k}")
            model.addConstr(lambda_vars[i, k] >= mu[i, k] + y[i] - 1, name=f"lambda_link_{i}_{k}")
    
    # For each aggregate k, lambda weights must sum to 1
    for k in range(K):
        model.addConstr(gp.quicksum(lambda_vars[i, k] for i in range(N)) == 1, name=f"lambda_sum_{k}")
    
    # For each aggregate k and each scenario i not assigned to k, lambda[i,k] = 0
    for i in range(N):
        for k in range(K):
            model.addConstr(lambda_vars[i, k] <= mu[i, k], name=f"lambda_zero_{i}_{k}")
    
    # Re-implement the constraints properly
    model.remove(model.getConstrs())
    
    # Exactly K scenarios must be selected as aggregates
    model.addConstr(gp.quicksum(y) == K, name="select_K_aggregates")
    
    # Each original scenario must be assigned to exactly one aggregate
    for i in range(N):
        model.addConstr(gp.quicksum(mu[i, k] for k in range(K)) == 1, name=f"assign_scenario_{i}")
    
    # Lambda weights are only non-zero for scenarios assigned to an aggregate
    for i in range(N):
        for k in range(K):
            model.addConstr(lambda_vars[i, k] <= mu[i, k], name=f"lambda_support_{i}_{k}")
    
    # For each aggregate k, lambda weights for assigned scenarios must sum to 1
    for k in range(K):
        model.addConstr(gp.quicksum(lambda_vars[i, k] for i in range(N)) == 1, name=f"lambda_sum_{k}")
    
    # Scaling constraint: for each original scenario i, each dimension j,
    # the weighted average of costs of scenarios assigned to the same aggregate
    # as i should be at most 1/t times the cost of scenario i
    for i in range(N):  # Original scenario
        for k in range(K):  # Aggregate
            for j in range(n):  # Dimension
                # If scenario i is assigned to aggregate k, then:
                # sum(lambda_vars[i_prime, k] * scenarios[i_prime][j]) <= (1/t) * scenarios[i][j]
                # Equivalently: t * sum(lambda_vars[i_prime, k] * scenarios[i_prime][j]) <= scenarios[i][j]
                model.addConstr(
                    t * gp.quicksum(lambda_vars[i_prime, k] * scenarios[i_prime][j] for i_prime in range(N))
                    <= scenarios[i][j],
                    name=f"scaling_{i}_{k}_{j}"
                )
    
    # Objective: maximize t
    model.setObjective(t, GRB.MAXIMIZE)
    
    model.optimize()
    
    if model.Status == GRB.INFEASIBLE:
        return None
    
    if model.SolCount == 0:
        return None
    
    # Extract solution
    selected_aggregates = [i for i in range(N) if y[i].X > 0.5]
    assignment = {}
    lambda_weights = defaultdict(dict)
    
    for i in range(N):
        for k in range(K):
            if mu[i, k].X > 0.5:
                assignment[i] = k
                break
    
    for k in range(K):
        for i in range(N):
            if lambda_vars[i, k].X > 1e-6:
                if k not in lambda_weights:
                    lambda_weights[k] = {}
                lambda_weights[k][i] = lambda_vars[i, k].X
    
    result = {
        "t": model.ObjVal,
        "approximation_ratio": 1.0 / model.ObjVal if model.ObjVal > 0 else float('inf'),
        "assignment": assignment,
        "lambda_weights": dict(lambda_weights),
        "solver_status": model.Status,
        "best_bound": model.ObjBound,
        "mip_gap": model.MIPGap
    }
    
    return result


def solve_two_stage_ip(scenarios, K, time_limit, remaining_time):
    """
    Solve the two-stage IP with direct scenario selection (without aggregation).
    """
    start_time = time.time()
    
    # Ensure we don't exceed the remaining time
    actual_time_limit = min(remaining_time, time_limit)
    if actual_time_limit <= 0:
        return None
    
    N, n = len(scenarios), len(scenarios[0])
    
    # Create model
    model = gp.Model("two_stage_scenario_reduction")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', actual_time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    
    # Binary variables for which scenarios are selected
    y = model.addVars(N, vtype=GRB.BINARY, name="y")
    
    # Assignment variables: z[i,j] = 1 if original scenario i is represented by selected scenario j
    z = model.addVars(N, N, vtype=GRB.BINARY, name="z")
    
    # Scaling factor t (to maximize)
    t = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="t")
    
    # Constraints
    # Exactly K scenarios must be selected
    model.addConstr(gp.quicksum(y) == K, name="select_K_scenarios")
    
    # Each original scenario must be assigned to exactly one selected scenario
    for i in range(N):
        model.addConstr(gp.quicksum(z[i, j] for j in range(N)) == 1, name=f"assign_original_{i}")
    
    # Assignment can only be made to selected scenarios
    for i in range(N):
        for j in range(N):
            model.addConstr(z[i, j] <= y[j], name=f"assign_to_selected_{i}_{j}")
    
    # Scaling constraint: for each original scenario i and each dimension j,
    # the cost of the representing selected scenario j must be at least t * cost of scenario i
    for i in range(N):  # Original scenario
        for j_idx in range(n):  # Dimension
            model.addConstr(
                gp.quicksum(z[i, k] * scenarios[k][j_idx] for k in range(N))
                >= t * scenarios[i][j_idx],
                name=f"scaling_{i}_{j_idx}"
            )
    
    # Objective: maximize t
    model.setObjective(t, GRB.MAXIMIZE)
    
    model.optimize()
    
    if model.Status == GRB.INFEASIBLE:
        return None
    
    if model.SolCount == 0:
        return None
    
    selected_scenarios = [i for i in range(N) if y[i].X > 0.5]
    
    result = {
        "t": model.ObjVal,
        "approximation_ratio": 1.0 / model.ObjVal if model.ObjVal > 0 else float('inf'),
        "selected_scenarios": selected_scenarios
    }
    
    return result


def solve_instance(instance_data, time_limit, log_path=None):
    """
    Solve the entire instance with all required K values.
    """
    start_time = time.time()
    
    scenarios = instance_data["scenarios"]
    N = instance_data["num_scenarios_N"]
    K_values = instance_data["K_values"]
    
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Initialize solution
    solution = {
        "objective_value": 0.0,
        "one_stage_IP_mu": {},
        "two_stage_IP": {}
    }
    
    # Solve for each K value
    for K in K_values:
        remaining_time = time_limit - (time.time() - start_time)
        
        # Solve one-stage IP with mu assignment
        if remaining_time > 0:
            try:
                one_stage_result = solve_one_stage_ip_mu(scenarios, K, time_limit, remaining_time)
                if one_stage_result is not None:
                    solution["one_stage_IP_mu"][str(K)] = one_stage_result
                    current_obj = one_stage_result["t"]
                    if current_obj > solution["objective_value"]:
                        solution["objective_value"] = current_obj
                        if logger:
                            logger.log_solution(current_obj, solution)
            except Exception as e:
                print(f"Error solving one_stage_IP_mu for K={K}: {e}")
        
        remaining_time = time_limit - (time.time() - start_time)
        
        # Solve two-stage IP
        if remaining_time > 0:
            try:
                two_stage_result = solve_two_stage_ip(scenarios, K, time_limit, remaining_time)
                if two_stage_result is not None:
                    solution["two_stage_IP"][str(K)] = two_stage_result
                    current_obj = two_stage_result["t"]
                    if current_obj > solution["objective_value"]:
                        solution["objective_value"] = current_obj
                        if logger:
                            logger.log_solution(current_obj, solution)
            except Exception as e:
                print(f"Error solving two_stage_IP for K={K}: {e}")
    
    # Final logging
    if logger and solution["objective_value"] > 0:
        logger.log_solution(solution["objective_value"], solution)
    
    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve scenario reduction problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to a JSONL file for logging intermediate solutions (optional)')
    
    args = parser.parse_args()
    
    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve the instance
    solution = solve_instance(instance_data, args.time_limit, args.log_path)
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()