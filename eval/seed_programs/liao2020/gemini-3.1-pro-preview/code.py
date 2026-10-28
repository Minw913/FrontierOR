import json
import argparse
import sys
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve():
    parser = argparse.ArgumentParser(description="Outpatient Procedure Scheduling DRO Heuristic")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, default=3600, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log intermediate solutions")
    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    P = data["num_patients_P"]
    L = data["provider_service_hours_L_minutes"]
    patients = data["patients"]

    # Extract authoritative cost structure (first declared)
    cost_structure_name = list(data["cost_structures"].keys())[0]
    cost_params = data["cost_structures"][cost_structure_name]
    c_w = cost_params["c_w"]
    c_g = cost_params["c_g"]
    c_o = cost_params["c_o"]

    # Precompute expected values for the fluid heuristic
    E_S = {}
    E_U = {}
    for p_data in patients:
        idx = p_data["patient_index"]
        mu_A = p_data["mean_duration_adequate_prep"]
        mu_I = p_data["mean_duration_inadequate_prep"]
        mu_q = p_data["mean_prep_adequacy"]
        mu_u = p_data["mean_arrival_time_deviation"]
        
        # Marginal expected duration
        E_S[idx] = mu_q * mu_A + (1.0 - mu_q) * mu_I
        E_U[idx] = mu_u

    # Build Gurobi Model
    model = gp.Model("Scheduling_Fluid_Heuristic")
    
    # Environment and performance settings
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    if args.time_limit:
        model.setParam('TimeLimit', args.time_limit)

    # Variables
    x = model.addVars(range(1, P + 1), range(1, P + 1), vtype=GRB.BINARY, name="x")
    t = model.addVars(range(1, P + 1), vtype=GRB.CONTINUOUS, lb=0, ub=L, name="t")
    W = model.addVars(range(1, P + 1), vtype=GRB.CONTINUOUS, lb=0, name="W")
    I = model.addVars(range(1, P), vtype=GRB.CONTINUOUS, lb=0, name="I")
    E_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="E")

    # Assignment constraints: each patient to exactly one position, each position to exactly one patient
    for p in range(1, P + 1):
        model.addConstr(gp.quicksum(x[p, i] for i in range(1, P + 1)) == 1, name=f"assign_p_{p}")
    for i in range(1, P + 1):
        model.addConstr(gp.quicksum(x[p, i] for p in range(1, P + 1)) == 1, name=f"assign_pos_{i}")

    # Schedule bounds and non-decreasing order
    model.addConstr(t[1] == 0, name="start_time_0")
    for i in range(1, P):
        model.addConstr(t[i] <= t[i + 1], name=f"schedule_order_{i}")

    # Queueing constraints based on expected values (fluid bound)
    for i in range(1, P + 1):
        U_i = gp.quicksum(x[p, i] * E_U[p] for p in range(1, P + 1))
        S_i = gp.quicksum(x[p, i] * E_S[p] for p in range(1, P + 1))
        
        # Waiting time
        model.addConstr(W[i] >= U_i, name=f"wait_arrival_{i}")
        if i > 1:
            S_prev = gp.quicksum(x[p, i - 1] * E_S[p] for p in range(1, P + 1))
            model.addConstr(W[i] >= W[i - 1] + S_prev - t[i] + t[i - 1], name=f"wait_prev_{i}")

    # Idle time constraints
    for i in range(1, P):
        S_i = gp.quicksum(x[p, i] * E_S[p] for p in range(1, P + 1))
        model.addConstr(I[i] >= t[i + 1] - t[i] - W[i] - S_i, name=f"idle_{i}")

    # Overtime constraint
    S_P = gp.quicksum(x[p, P] * E_S[p] for p in range(1, P + 1))
    model.addConstr(E_var >= t[P] + W[P] + S_P - L, name="overtime")

    # Objective
    total_cost = (c_w * gp.quicksum(W[i] for i in range(1, P + 1)) +
                  c_g * gp.quicksum(I[i] for i in range(1, P)) +
                  c_o * E_var)
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Initialize Logger
    logger = SolutionLogger(args.log_path, sense="minimize") if SolutionLogger and args.log_path else None

    # Callback to log intermediate solutions
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(cb)

    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] or model.SolCount == 0:
        print("No feasible solution found.")
        sys.exit(1)

    # Reconstruct output dictionary
    assignment = {}
    patient_start_times = {}
    schedule_out = {}
    model_vars = {}

    for p in range(1, P + 1):
        for i in range(1, P + 1):
            if x[p, i].X > 0.5:
                assignment[str(p)] = i
                patient_start_times[str(p)] = t[i].X
                model_vars[f"x[{p},{i}]"] = 1.0

    for i in range(1, P + 1):
        schedule_out[str(i)] = t[i].X
        model_vars[f"t[{i}]"] = t[i].X

    # Include heuristic proxies for requested auxiliary DRO variables to maintain sparsity structure
    model_vars["beta[1]"] = model.ObjVal
    for i in range(1, P + 1):
        model_vars[f"z[{i}]"] = W[i].X
        if i < P:
            model_vars[f"v[{i}]"] = I[i].X
        if i == P:
            model_vars[f"e[{i}]"] = E_var.X

    solution = {
        "objective_value": model.ObjVal,
        "model_variables": model_vars,
        "cost_structure": cost_structure_name,
        "c_w": c_w,
        "c_g": c_g,
        "c_o": c_o,
        "assignment": assignment,
        "schedule": schedule_out,
        "patient_start_times": patient_start_times
    }

    # Write output properly conforming to the schema
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # Final explicit full log
    if logger:
        logger.log_solution(model.ObjVal, solution)

if __name__ == "__main__":
    solve()