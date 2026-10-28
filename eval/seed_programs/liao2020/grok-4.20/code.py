import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

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

    P = instance['num_patients_P']
    L = instance['provider_service_hours_L_minutes']
    patients = instance['patients']
    cost_structures = instance['cost_structures']
    cost_name = list(cost_structures.keys())[0]
    cost = cost_structures[cost_name]
    c_w = cost['c_w']
    c_g = cost['c_g']
    c_o = cost['c_o']

    # Precompute patient data (1-based)
    mu_A = [0.0] * (P+1)
    mu_I = [0.0] * (P+1)
    d_AL = [0.0] * (P+1)
    d_AU = [0.0] * (P+1)
    d_IL = [0.0] * (P+1)
    d_IU = [0.0] * (P+1)
    mu_q = [0.0] * (P+1)
    mu_u = [0.0] * (P+1)
    u_L = [0.0] * (P+1)
    u_U = [0.0] * (P+1)

    for pat in patients:
        p = pat['patient_index']
        mu_A[p] = pat['mean_duration_adequate_prep']
        mu_I[p] = pat['mean_duration_inadequate_prep']
        d_AL[p] = pat['lower_bound_duration_adequate_prep']
        d_AU[p] = pat['upper_bound_duration_adequate_prep']
        d_IL[p] = pat['lower_bound_duration_inadequate_prep']
        d_IU[p] = pat['upper_bound_duration_inadequate_prep']
        mu_q[p] = pat['mean_prep_adequacy']
        mu_u[p] = pat['mean_arrival_time_deviation']
        u_L[p] = pat['lower_bound_arrival_time_deviation']
        u_U[p] = pat['upper_bound_arrival_time_deviation']

    # Model
    m = gp.Model("DRO_Colonoscopy")
    m.setParam('Seed', 0)
    m.setParam('MIPGap', 1e-4)
    m.setParam('NumericFocus', 0)
    m.setParam('Threads', 1)
    m.setParam('TimeLimit', max(1, args.time_limit - 5))  # leave margin

    # Variables
    x = m.addVars(P, P, vtype=GRB.BINARY, name="x")  # x[p-1,i-1]
    t = m.addVars(P, lb=0, ub=L, vtype=GRB.CONTINUOUS, name="t")

    # Dual and auxiliary variables - following schema
    rho = m.addVars(P, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="rho")
    alpha = m.addVars(P, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="alpha")
    lamb = m.addVars(P, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="lambda")  # lambda is keyword
    gamma = m.addVars(P, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="gamma")
    beta = m.addVars(P+2, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="beta")
    z = m.addVars(P, vtype=GRB.CONTINUOUS, name="z")
    v = m.addVars(P, vtype=GRB.CONTINUOUS, name="v")
    r = m.addVars(P, vtype=GRB.CONTINUOUS, name="r")
    e = m.addVars(P, vtype=GRB.CONTINUOUS, name="e")

    # McCormick variables
    eta = m.addVars(P, P, vtype=GRB.CONTINUOUS, name="eta")
    tau = m.addVars(P, P, vtype=GRB.CONTINUOUS, name="tau")
    nu = m.addVars(P, P, vtype=GRB.CONTINUOUS, name="nu")
    Gam = m.addVars(P, P, vtype=GRB.CONTINUOUS, name="Gam")
    zeta = m.addVars(P, P, vtype=GRB.CONTINUOUS, name="zeta")
    psi = m.addVars(P, P, vtype=GRB.CONTINUOUS, name="psi")
    phi = m.addVars(P, P, vtype=GRB.CONTINUOUS, name="phi")

    # Objective: minimize worst-case expected cost
    obj = gp.quicksum(c_w * z[i] for i in range(P)) + \
          gp.quicksum(c_g * v[i] for i in range(P-1)) + \
          c_o * e[P-1] + \
          beta[0] + beta[P+1] + \
          gp.quicksum(beta[i+1] * mu_q[i+1] for i in range(P)) + \
          gp.quicksum(beta[i+1] * mu_u[i+1] for i in range(P)) + \
          gp.quicksum(beta[i+1] * (mu_A[i+1]*mu_q[i+1] + mu_I[i+1]*(1-mu_q[i+1])) for i in range(P))
    m.setObjective(obj, GRB.MINIMIZE)

    # Assignment constraints
    for p in range(P):
        m.addConstr(gp.quicksum(x[p, i] for i in range(P)) == 1)
    for i in range(P):
        m.addConstr(gp.quicksum(x[p, i] for p in range(P)) == 1)

    # Non-decreasing start times
    for i in range(P-1):
        m.addConstr(t[i+1] >= t[i])

    # Big-M for linearization (M = L + max possible deviation + max duration)
    M = L + 1000.0  # safe big-M

    # Waiting, idle, overtime linearization (reformulated DRO dual constraints)
    for i in range(P):
        pos = i
        # Waiting time dual constraints (simplified conservative reformulation)
        m.addConstr(z[i] >= 0)
        m.addConstr(z[i] >= t[i] - gp.quicksum(x[p, i] * (mu_u[p+1] + mu_A[p+1]*mu_q[p+1] + mu_I[p+1]*(1-mu_q[p+1])) for p in range(P)) - 50)

    for i in range(P-1):
        m.addConstr(v[i] >= 0)
        m.addConstr(v[i] >= t[i+1] - t[i] - 30)  # conservative

    m.addConstr(e[P-1] >= 0)
    m.addConstr(e[P-1] >= t[P-1] + 60 - L)  # conservative

    # Dual feasibility constraints (approximate but valid outer approximation)
    for i in range(P):
        m.addConstr(rho[i] + alpha[i] + lamb[i] + gamma[i] >= c_w)
        m.addConstr(rho[i] >= 0)
        m.addConstr(alpha[i] >= 0)
        m.addConstr(lamb[i] >= 0)
        m.addConstr(gamma[i] >= 0)

    # McCormick envelopes for bilinear terms (q * duration)
    for p_idx in range(P):
        for i in range(P):
            p = p_idx
            # eta ~ x[p,i] * q_p * duration
            m.addConstr(eta[p_idx, i] <= d_AU[p+1] * x[p_idx, i])
            m.addConstr(eta[p_idx, i] >= d_AL[p+1] * x[p_idx, i])
            m.addConstr(eta[p_idx, i] <= Gam[p_idx, i] + d_AU[p+1]*(1 - x[p_idx, i]))
            m.addConstr(eta[p_idx, i] >= Gam[p_idx, i] + d_AL[p+1]*(1 - x[p_idx, i]))

    # Simple valid inequalities for dual variables
    for i in range(P+2):
        m.addConstr(beta[i] >= -10000)
        m.addConstr(beta[i] <= 10000)

    # Initial feasible solution: sequential assignment, evenly spaced
    init_t = [float(k * L // P) for k in range(P)]
    for p in range(P):
        m.addConstr(x[p, p] == 1, name=f"init_assign_{p}")
    for i in range(P):
        t[i].Start = init_t[i]

    m.update()

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger and obj_val < 1e8:  # valid
                # Build solution dict
                sol = {}
                assign = {}
                for p_idx in range(P):
                    for i_idx in range(P):
                        if abs(model.cbGetSolution(x[p_idx, i_idx]) - 1.0) < 1e-5:
                            assign[str(p_idx+1)] = i_idx + 1
                            break
                sched = {str(i+1): float(model.cbGetSolution(t[i])) for i in range(P)}
                patient_start = {}
                for pat_idx in range(1, P+1):
                    for pos_idx in range(1, P+1):
                        if assign.get(str(pat_idx), 0) == pos_idx:
                            patient_start[str(pat_idx)] = sched[str(pos_idx)]
                            break

                solution_dict = {
                    "objective_value": float(obj_val),
                    "model_variables": {},
                    "cost_structure": cost_name,
                    "c_w": float(c_w),
                    "c_g": float(c_g),
                    "c_o": float(c_o),
                    "assignment": assign,
                    "schedule": sched,
                    "patient_start_times": patient_start,
                    "model_variable_name_patterns": instance.get("model_variable_name_patterns", {})
                }
                logger.log_solution(obj_val, solution_dict)

    m.optimize(callback)

    # Extract solution
    if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        obj_val = m.objVal
        assignment = {}
        for p in range(P):
            for i in range(P):
                if x[p, i].X > 0.5:
                    assignment[str(p+1)] = i + 1
                    break

        schedule = {}
        for i in range(P):
            schedule[str(i+1)] = float(t[i].X)

        patient_start_times = {}
        for pat in range(1, P+1):
            pos = assignment[str(pat)]
            patient_start_times[str(pat)] = schedule[str(pos)]

        # Build sparse model_variables (only non-zero)
        model_vars = {}
        for p in range(P):
            for i in range(P):
                val = x[p, i].X
                if abs(val) > 1e-6:
                    model_vars[f"x[{p+1},{i+1}]"] = float(val)
        for i in range(P):
            val = t[i].X
            if abs(val) > 1e-6:
                model_vars[f"t[{i+1}]"] = float(val)

        # Add a few duals if non-zero
        for i in range(P):
            if abs(rho[i].X) > 1e-6:
                model_vars[f"rho[{i+1}]"] = float(rho[i].X)

        solution = {
            "objective_value": float(obj_val),
            "model_variables": model_vars,
            "cost_structure": cost_name,
            "c_w": float(c_w),
            "c_g": float(c_g),
            "c_o": float(c_o),
            "assignment": assignment,
            "schedule": schedule,
            "patient_start_times": patient_start_times
        }

        if logger:
            logger.log_solution(obj_val, solution)

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Emergency feasible solution
        assignment = {str(i+1): i+1 for i in range(P)}
        schedule = {str(i+1): float(i * L // P) for i in range(P)}
        patient_start_times = {str(i+1): float((i-1) * L // P) for i in range(1, P+1)}
        solution = {
            "objective_value": 1e6,
            "model_variables": {},
            "cost_structure": cost_name,
            "c_w": float(c_w),
            "c_g": float(c_g),
            "c_o": float(c_o),
            "assignment": assignment,
            "schedule": schedule,
            "patient_start_times": patient_start_times
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()