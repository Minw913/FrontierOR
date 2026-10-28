import sys
import json
import math
import argparse
import gurobipy as gp
from gurobipy import GRB
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    # Initialize solution logger if requested
    try:
        from solution_logger import SolutionLogger
        # We are maximizing the minimum cost
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    except ImportError:
        logger = None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    glb = data['global_parameters']
    theta = float(glb['theta'])
    sigma = float(glb['sigma'])
    alpha = float(glb['alpha'])
    beta = float(glb['beta'])
    gamma = float(glb['gamma'])
    alpha_prime = float(glb['alpha_prime'])
    beta_prime = float(glb['beta_prime'])
    gamma_prime = float(glb['gamma_prime'])

    inst = data['instance_parameters']
    N1 = int(inst['num_level1_facilities'])
    N2 = int(inst['num_level2_facilities'])
    M = int(inst['num_customers'])
    K_max = int(inst['k_max'])
    B = float(inst['total_interdiction_budget'])

    # Interdiction costs
    h1 = {0: 0.0}
    for k, v in data['interdiction_parameters']['interdiction_cost_level1_h1'].items():
        h1[int(k)] = float(v)

    h2 = {0: 0.0}
    for k, v in data['interdiction_parameters']['interdiction_cost_level2_h2'].items():
        h2[int(k)] = float(v)

    # Interdiction destruction ratios
    d1_ratio = {0: 0.0}
    for k, v in data['interdiction_parameters']['capacity_reduction_level1_d1'].items():
        d1_ratio[int(k)] = float(v)

    d2_ratio = {0: 0.0}
    for k, v in data['interdiction_parameters']['capacity_reduction_level2_d2'].items():
        d2_ratio[int(k)] = float(v)

    # Capacities
    C1 = data['level1_facilities']['capacity_type1'] if N1 > 0 else []
    C2I = data['level2_facilities']['capacity_type1'] if N2 > 0 else []
    C2II = data['level2_facilities']['capacity_type2'] if N2 > 0 else []

    # Demands
    D = data['customers']['demands'] if M > 0 else []
    D1 = [theta * d for d in D]
    D2 = [(1.0 - theta) * d for d in D]

    # Distances
    dist1 = data['distances'].get('customer_to_level1_facility', [])
    dist2 = data['distances'].get('customer_to_level2_facility', [])
    dist12 = data['distances'].get('level1_to_level2_facility', [])

    # Compute Level 2 to Level 2 distances
    if N2 > 0:
        l2_x = data['level2_facilities']['x_coordinates']
        l2_y = data['level2_facilities']['y_coordinates']
        dist22 = [[math.hypot(l2_x[j] - l2_x[k], l2_y[j] - l2_y[k]) for k in range(N2)] for j in range(N2)]
    else:
        dist22 = []

    # Safe Upper Bounds (Big-M) for Dual Capacity Variables
    Mw1 = float(alpha_prime + sigma * gamma_prime)
    Mw2 = float(alpha_prime + sigma * gamma_prime)
    Mw3 = float(max(beta_prime, gamma_prime))

    # Gurobi setup
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()
    model = gp.Model("Bilevel_Interdiction", env=env)
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', args.time_limit)

    # Dual Variables (Defender's Shadow Prices)
    u1 = model.addVars(M, lb=0.0, ub=Mw1, name="u1")
    u2 = model.addVars(M, lb=0.0, ub=beta_prime, name="u2")
    v1 = model.addVars(N1, lb=0.0, ub=gamma_prime, name="v1")
    v2 = model.addVars(N2, lb=0.0, ub=gamma_prime, name="v2")
    
    w1 = model.addVars(N1, lb=0.0, ub=Mw1, name="w1")
    w2 = model.addVars(N2, lb=0.0, ub=Mw2, name="w2")
    w3 = model.addVars(N2, lb=0.0, ub=Mw3, name="w3")

    # Interdictor Decision Variables
    x = model.addVars(N1, K_max + 1, vtype=GRB.BINARY, name="x")
    y = model.addVars(N2, K_max + 1, vtype=GRB.BINARY, name="y")

    # Linearization Variables exactly mapping to (Capacity Usage * Decision)
    z1 = model.addVars(N1, K_max + 1, lb=0.0, name="z1")
    z2I = model.addVars(N2, K_max + 1, lb=0.0, name="z2I")
    z2II = model.addVars(N2, K_max + 1, lb=0.0, name="z2II")

    # Budget constraint
    model.addConstr(
        gp.quicksum(h1[l] * x[i, l] for i in range(N1) for l in range(1, K_max + 1)) +
        gp.quicksum(h2[l] * y[j, l] for j in range(N2) for l in range(1, K_max + 1)) <= B,
        "Budget"
    )

    # Exactly one intensity level assignment constraint
    for i in range(N1):
        model.addConstr(gp.quicksum(x[i, l] for l in range(K_max + 1)) == 1, f"x_assign_{i}")
    for j in range(N2):
        model.addConstr(gp.quicksum(y[j, l] for l in range(K_max + 1)) == 1, f"y_assign_{j}")

    # Linearization Constraints Formulation: pulls z appropriately based on attack intensity
    for i in range(N1):
        for l in range(K_max + 1):
            model.addConstr(z1[i, l] >= w1[i] - Mw1 * (1.0 - x[i, l]), f"lin1_{i}_{l}")

    for j in range(N2):
        for l in range(K_max + 1):
            model.addConstr(z2I[j, l] >= w2[j] - Mw2 * (1.0 - y[j, l]), f"lin2I_{j}_{l}")
            model.addConstr(z2II[j, l] >= w3[j] - Mw3 * (1.0 - y[j, l]), f"lin2II_{j}_{l}")

    # Dual Constraints (Bounding defender's allocation to paths based on path's physical routing cost)
    for c in range(M):
        for i in range(N1):
            model.addConstr(u1[c] - sigma * v1[i] - w1[i] <= alpha * dist1[c][i], f"dc_1_{c}_{i}")
        for j in range(N2):
            model.addConstr(u1[c] - sigma * v2[j] - w2[j] <= beta * dist2[c][j], f"dc_2_{c}_{j}")
            model.addConstr(u2[c] - w3[j] <= beta * dist2[c][j], f"dc_3_{c}_{j}")

    for i in range(N1):
        for j in range(N2):
            model.addConstr(v1[i] - w3[j] <= gamma * dist12[i][j], f"dc_ref1_{i}_{j}")

    for j in range(N2):
        for k in range(N2):
            model.addConstr(v2[j] - w3[k] <= gamma * dist22[j][k], f"dc_ref2_{j}_{k}")

    # Dual Maximization Objective precisely maps to worst-case Min-cost Flow Primal evaluation
    obj = gp.quicksum(D1[c] * u1[c] + D2[c] * u2[c] for c in range(M)) \
        - gp.quicksum(C1[i] * (1.0 - d1_ratio[l]) * z1[i, l] for i in range(N1) for l in range(K_max + 1)) \
        - gp.quicksum(C2I[j] * (1.0 - d2_ratio[l]) * z2I[j, l] for j in range(N2) for l in range(K_max + 1)) \
        - gp.quicksum(C2II[j] * (1.0 - d2_ratio[l]) * z2II[j, l] for j in range(N2) for l in range(K_max + 1))

    model.setObjective(obj, GRB.MAXIMIZE)

    # Callback specifically tracking solver findings to align with external logger
    def callback_logger(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj_val)
            except Exception:
                pass

    model.optimize(callback_logger)

    # Output solution retrieval
    if model.SolCount > 0:
        obj_val = model.ObjVal
        pattern = dict()

        for i in range(N1):
            chosen_l = max(range(K_max + 1), key=lambda l: x[i, l].X)
            pattern[f"S1_{i}"] = chosen_l
            
        for j in range(N2):
            chosen_l = max(range(K_max + 1), key=lambda l: y[j, l].X)
            pattern[f"S2_{j}"] = chosen_l

        solution = {
            "objective_value": float(obj_val),
            "interdiction_pattern": pattern
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        # Fallback to empty if absolutely zero feasible solutions are found
        fallback_sol = {
            "objective_value": 0.0,
            "interdiction_pattern": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(fallback_sol, f, indent=4)

if __name__ == "__main__":
    main()