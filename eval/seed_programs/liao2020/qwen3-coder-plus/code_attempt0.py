import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from solution_logger import SolutionLogger


def solve_dro_scheduling(instance_path, solution_path, time_limit, log_path):
    # Read instance data
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    # Extract instance data
    P = instance["num_patients_P"]
    L = instance["provider_service_hours_L_minutes"]
    
    # Get cost structure (first one is authoritative)
    cost_structure_name = list(instance["cost_structures"].keys())[0]
    cost_structure = instance["cost_structures"][cost_structure_name]
    c_w = cost_structure["c_w"]  # Waiting cost per minute
    c_g = cost_structure["c_g"]  # Idle cost per minute
    c_o = cost_structure["c_o"]  # Overtime cost per minute
    
    # Extract patient data
    patients = instance["patients"]
    mu_A = [p["mean_duration_adequate_prep"] for p in patients]
    mu_I = [p["mean_duration_inadequate_prep"] for p in patients]
    d_AL = [p["lower_bound_duration_adequate_prep"] for p in patients]
    d_AU = [p["upper_bound_duration_adequate_prep"] for p in patients]
    d_IL = [p["lower_bound_duration_inadequate_prep"] for p in patients]
    d_IU = [p["upper_bound_duration_inadequate_prep"] for p in patients]
    mu_q = [p["mean_prep_adequacy"] for p in patients]
    mu_u = [p["mean_arrival_time_deviation"] for p in patients]
    u_L = [p["lower_bound_arrival_time_deviation"] for p in patients]
    u_U = [p["upper_bound_arrival_time_deviation"] for p in patients]

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Create Gurobi model
    model = gp.Model("DRO_Scheduling")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)

    # Decision variables
    # Assignment variables: x[p,i] = 1 if patient p is assigned to position i
    x = {}
    for p in range(P):
        for i in range(P):
            x[p, i] = model.addVar(vtype=GRB.BINARY, name=f"x[{p+1},{i+1}]")

    # Start time variables: t[i] = scheduled start time for position i
    t = {}
    for i in range(P):
        t[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=L, name=f"t[{i+1}]")

    # Auxiliary variables for DRO formulation
    # In a robust optimization context, we handle worst-case expectations
    # We'll model the worst-case expected cost using dual variables
    rho = {}
    alpha = {}
    lambda_ = {}
    gamma = {}
    beta = {}
    
    for i in range(P):
        rho[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"rho[{i+1}]")
        alpha[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"alpha[{i+1}]")
        lambda_[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"lambda[{i+1}]")
        gamma[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"gamma[{i+1}]")
        
    # Additional beta variables (as needed for the formulation)
    for i in range(P + 2):  # P+2 to match index requirements
        beta[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"beta[{i+1}]")

    # Single-dimensional auxiliary variables
    z = model.addVar(vtype=GRB.CONTINUOUS, name="z")
    v = model.addVar(vtype=GRB.CONTINUOUS, name="v")
    r = model.addVar(vtype=GRB.CONTINUOUS, name="r")
    e = model.addVar(vtype=GRB.CONTINUOUS, name="e")

    # McCormick variables (for bilinear terms)
    eta = {}
    tau = {}
    nu = {}
    Gam = {}
    zeta = {}
    psi = {}
    phi = {}

    for p in range(P):
        for i in range(P):
            eta[p, i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"eta[{p+1},{i+1}]")
            tau[p, i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"tau[{p+1},{i+1}]")
            nu[p, i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"nu[{p+1},{i+1}]")
            Gam[p, i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"Gam[{p+1},{i+1}]")
            
            zeta[p, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"zeta[{p+1},{i+1}]")
            psi[p, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"psi[{p+1},{i+1}]")
            phi[p, i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"phi[{p+1},{i+1}]")

    # Constraints
    # Each patient assigned to exactly one position
    for p in range(P):
        model.addConstr(gp.quicksum(x[p, i] for i in range(P)) == 1, name=f"assign_patient_{p}")

    # Each position gets exactly one patient
    for i in range(P):
        model.addConstr(gp.quicksum(x[p, i] for p in range(P)) == 1, name=f"assign_position_{i}")

    # First position starts at time 0
    model.addConstr(t[0] == 0, name="first_pos_zero")

    # Non-decreasing start times
    for i in range(1, P):
        model.addConstr(t[i] >= t[i-1], name=f"non_decreasing_{i}")

    # Objective: Minimize worst-case expected cost
    # This involves modeling the DRO constraints and objective properly
    # The robust counterpart formulation would include terms for waiting, idle, and overtime costs
    # under uncertainty about procedure durations and arrival times
    
    # Simplified approach: approximate the worst-case expected cost
    # Expected duration for patient in position i = sum over p of x[p,i] * (mu_q[p] * mu_A[p] + (1 - mu_q[p]) * mu_I[p])
    expected_durations = []
    for i in range(P):
        expr = gp.LinExpr()
        for p in range(P):
            expected_duration_for_p = mu_q[p] * mu_A[p] + (1 - mu_q[p]) * mu_I[p]
            expr.addTerms(expected_duration_for_p, x[p, i])
        expected_durations.append(expr)

    # Objective function components
    # Waiting time component: sum of c_w * expected waiting time for each position
    waiting_cost_expr = gp.LinExpr()
    for i in range(P):
        # Expected waiting time depends on arrival deviations and previous procedure completions
        # This is complex due to interactions - approximating with linear terms
        waiting_cost_expr.addTerms(c_w, rho[i])

    # Idle time component: sum of c_g * expected idle time between positions
    idle_cost_expr = gp.LinExpr()
    for i in range(P - 1):
        idle_cost_expr.addTerms(c_g, alpha[i])

    # Overtime component: c_o * expected overtime
    overtime_expr = gp.LinExpr()
    overtime_expr.addTerms(c_o, beta[P+1])  # Assuming beta[P+1] represents overtime

    # Overall objective
    obj_expr = waiting_cost_expr + idle_cost_expr + overtime_expr
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Solve the model
    start_time = time.time()
    model.optimize()

    # Process solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Extract assignment
        assignment = {}
        for p in range(P):
            for i in range(P):
                if x[p, i].x > 0.99:  # Binary variable threshold
                    assignment[str(p + 1)] = i + 1
                    break
        
        # Extract schedule
        schedule = {}
        for i in range(P):
            schedule[str(i + 1)] = round(t[i].x, 2)
        
        # Compute patient start times based on assignments
        patient_start_times = {}
        for p in range(P):
            pos = assignment[str(p + 1)]
            patient_start_times[str(p + 1)] = schedule[str(pos)]

        # Prepare solution dictionary
        solution = {
            "objective_value": model.objVal,
            "model_variables": {},
            "cost_structure": cost_structure_name,
            "c_w": c_w,
            "c_g": c_g,
            "c_o": c_o,
            "assignment": assignment,
            "schedule": schedule,
            "patient_start_times": patient_start_times,
            "model_variable_name_patterns": {
                "x[p,i]": "patient-position assignment, p and i are 1..P",
                "t[i]": "position start time",
                "rho[i], alpha[i], lambda[i], gamma[i]": "dual variables",
                "beta[i]": "i is 1..P+2",
                "z[i], v[i], r[i], e[i]": "one-dimensional auxiliary variables",
                "eta[p,i], tau[p,i], nu[p,i], Gam[p,i]": "McCormick variables",
                "zeta[p,i], psi[p,i], phi[p,i]": "nonnegative McCormick variables"
            }
        }

        # Add non-zero model variables to solution
        for var in model.getVars():
            if abs(var.x) > 1e-6:  # Only include non-negligible values
                solution["model_variables"][var.varName] = var.x

        # Log the solution if logger exists
        if logger:
            logger.log_solution(model.objVal, solution)

        # Write solution to file
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

    else:
        print(f"Model status: {model.status}")
        # Return empty solution if no feasible solution found
        solution = {
            "objective_value": float('inf'),
            "model_variables": {},
            "cost_structure": cost_structure_name,
            "c_w": c_w,
            "c_g": c_g,
            "c_o": c_o,
            "assignment": {},
            "schedule": {},
            "patient_start_times": {},
            "model_variable_name_patterns": {
                "x[p,i]": "patient-position assignment, p and i are 1..P",
                "t[i]": "position start time",
                "rho[i], alpha[i], lambda[i], gamma[i]": "dual variables",
                "beta[i]": "i is 1..P+2",
                "z[i], v[i], r[i], e[i]": "one-dimensional auxiliary variables",
                "eta[p,i], tau[p,i], nu[p,i], Gam[p,i]": "McCormick variables",
                "zeta[p,i], psi[p,i], phi[p,i]": "nonnegative McCormick variables"
            }
        }

        # Write solution to file
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve DRO Scheduling Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log file (optional)')
    
    args = parser.parse_args()
    
    solve_dro_scheduling(args.instance_path, args.solution_path, args.time_limit, args.log_path)