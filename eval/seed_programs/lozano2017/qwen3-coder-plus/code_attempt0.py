import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger


def solve_bilevel_milp(instance_data, time_limit, log_path):
    # Extract instance data
    n = instance_data["n"]
    n1 = instance_data["n1"]
    n2 = instance_data["n2"]
    m1 = instance_data["m1"]
    m2 = instance_data["m2"]
    all_leader_vars_integer = instance_data["all_leader_vars_integer"]
    leader_follower_objective_coefficients_fixed_zero = instance_data["leader_follower_objective_coefficients_fixed_zero"]
    num_integer_follower_vars = instance_data["num_integer_follower_vars"]
    integer_follower_vars = instance_data["integer_follower_vars"]  # 1-indexed
    c = instance_data["c"]
    d2 = instance_data["d2"]
    A1 = instance_data["A1"]
    B1 = instance_data["B1"]
    b1 = instance_data["b1"]
    A2 = instance_data["A2"]
    B2 = instance_data["B2"]
    b2 = instance_data["b2"]

    # Convert 1-indexed to 0-indexed for integer follower vars
    integer_follower_indices = [i - 1 for i in integer_follower_vars]

    # Initialize logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    # Create model
    model = gp.Model("bilevel_milp")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)

    # Define leader variables (non-negative integers)
    x = []
    for i in range(n1):
        var = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}")
        x.append(var)

    # Define follower variables
    y = []
    for j in range(n2):
        if j in integer_follower_indices:
            var = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{j}")
        else:
            var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{j}")
        y.append(var)

    # Add upper-level constraints: A1*x + B1*y <= b1
    for i in range(m1):
        lhs = gp.LinExpr()
        for j in range(n1):
            lhs += A1[i][j] * x[j]
        for j in range(n2):
            lhs += B1[i][j] * y[j]
        model.addConstr(lhs <= b1[i], name=f"upper_level_constr_{i}")

    # Reformulate the bilevel problem using KKT conditions or big-M method
    # Here we'll use the KKT-based reformulation for the optimistic case
    
    # First, create dual variables for the follower's problem
    lambda_vars = []  # Dual variables for lower-level constraints
    for i in range(m2):
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"lambda_{i}")
        lambda_vars.append(var)

    # Slack variables for complementary slackness
    s_lower = []  # Slack for lower-level constraint
    for i in range(m2):
        var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"s_lower_{i}")
        s_lower.append(var)

    # Add lower-level constraints: A2*x + B2*y <= b2
    for i in range(m2):
        lhs = gp.LinExpr()
        for j in range(n1):
            lhs += A2[i][j] * x[j]
        for j in range(n2):
            lhs += B2[i][j] * y[j]
        constr = model.addConstr(lhs <= b2[i], name=f"lower_level_constr_{i}")
        
        # Update slack variable to make it equal to b2[i] - (A2*x + B2*y)
        model.addConstr(s_lower[i] == b2[i] - lhs, name=f"slack_def_{i}")

    # Add follower's stationarity conditions: d2 - sum(B2^T * lambda) = 0
    for j in range(n2):
        lhs = gp.LinExpr()
        lhs += d2[j]  # Coefficient from the objective
        for i in range(m2):
            lhs -= B2[i][j] * lambda_vars[i]  # Subtract dual contribution
        
        # For non-zero variables, add the reduced cost expression
        model.addConstr(lhs == 0, name=f"stationarity_{j}")

    # Complementary slackness constraints: lambda_i * s_lower_i = 0
    # We use big-M formulation: lambda_i <= M1*u_i, s_lower_i <= M2*(1-u_i)
    u_binary = []
    M1_vals = []
    M2_vals = []
    
    # Estimate reasonable big-M values
    for i in range(m2):
        max_lambda = max(100, abs(max(d2)) + 1)  # Rough estimate based on objective coeffs
        max_slack = max(100, max(b2))  # Rough estimate based on RHS
        M1_vals.append(max_lambda)
        M2_vals.append(max_slack)
        
        u_var = model.addVar(vtype=GRB.BINARY, name=f"u_{i}")
        u_binary.append(u_var)
        
        # Big-M constraints for complementary slackness
        model.addConstr(lambda_vars[i] <= M1_vals[i] * u_binary[i])
        model.addConstr(s_lower[i] <= M2_vals[i] * (1 - u_binary[i]))

    # Set leader's objective: maximize c^T * x
    obj_expr = gp.LinExpr()
    for j in range(n1):
        obj_expr += c[j] * x[j]
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Solve the reformulated problem
    model.optimize()

    # Process results
    if model.SolCount > 0:
        x_vals = [x[i].X for i in range(n1)]
        y_vals = [y[j].X for j in range(n2)]
        
        # Calculate objective values
        leader_obj_val = sum(c[j] * x_vals[j] for j in range(n1))
        follower_obj_val = sum(d2[j] * y_vals[j] for j in range(n2))
        
        solution = {
            "objective_value": leader_obj_val,
            "x": x_vals,
            "y": y_vals,
            "follower_objective_value": follower_obj_val
        }
        
        # Log the solution if logger exists
        if logger:
            logger.log_solution(leader_obj_val, solution)
        
        return solution
    else:
        # No feasible solution found
        return {
            "objective_value": float('-inf'),
            "x": [0.0] * n1,
            "y": [0.0] * n2,
            "follower_objective_value": 0.0
        }


def main():
    parser = argparse.ArgumentParser(description='Solve bilevel mixed-integer linear program.')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the problem instance.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path where the final solution JSON file must be written.')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to a JSONL file for logging intermediate solutions.')

    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_bilevel_milp(instance_data, args.time_limit, args.log_path)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()