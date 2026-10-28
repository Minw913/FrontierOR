import sys
import math
import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB

def main():
    # 1. Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    start_time = time.time()

    # 2. Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n1 = data['n1']
    n2 = data['n2']
    m1 = data['m1']
    m2 = data['m2']
    c = data['c']
    d2 = data['d2']
    A1 = data['A1']
    B1 = data['B1']
    b1 = data['b1']
    A2 = data['A2']
    B2 = data['B2']
    b2 = data['b2']
    int_f_vars = set(data['integer_follower_vars'])

    # 3. Setup logger
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except ImportError:
            pass

    # 4. Initialize Gurobi environment
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    # 5. Find global lower bound for the follower's objective (z_min)
    m_zmin = gp.Model(env=env)
    m_zmin.Params.OutputFlag = 0
    m_zmin.Params.Threads = 1
    y_z = m_zmin.addVars(n2, lb=0, vtype=GRB.CONTINUOUS)
    for i in range(m2):
        m_zmin.addConstr(gp.quicksum(B2[i][j]*y_z[j] for j in range(n2)) <= b2[i])
    m_zmin.setObjective(gp.quicksum(d2[j]*y_z[j] for j in range(n2)), GRB.MINIMIZE)
    m_zmin.optimize()
    
    if m_zmin.status == GRB.OPTIMAL:
        z_min = m_zmin.objVal
    elif m_zmin.status == GRB.UNBOUNDED:
        z_min = -1e6
    else:
        z_min = 0

    # 6. Initialize the Master Problem (MP)
    mp = gp.Model(env=env)
    mp.Params.OutputFlag = 0
    mp.Params.Seed = 0
    mp.Params.MIPGap = 1e-4
    mp.Params.NumericFocus = 0
    mp.Params.Threads = 1

    x = mp.addVars(n1, lb=0, vtype=GRB.INTEGER, name="x")
    y = mp.addVars(n2, lb=0, vtype=GRB.CONTINUOUS, name="y")
    for j in range(n2):
        if (j + 1) in int_f_vars:
            y[j].vtype = GRB.INTEGER

    z = mp.addVar(lb=z_min, vtype=GRB.CONTINUOUS, name="z")

    for i in range(m1):
        mp.addConstr(gp.quicksum(A1[i][j]*x[j] for j in range(n1)) + 
                     gp.quicksum(B1[i][j]*y[j] for j in range(n2)) <= b1[i])

    for i in range(m2):
        mp.addConstr(gp.quicksum(A2[i][j]*x[j] for j in range(n1)) + 
                     gp.quicksum(B2[i][j]*y[j] for j in range(n2)) <= b2[i])

    mp.setObjective(gp.quicksum(c[j]*x[j] for j in range(n1)), GRB.MAXIMIZE)
    mp.addConstr(z <= gp.quicksum(d2[j]*y[j] for j in range(n2)))

    best_obj = -float('inf')
    best_sol = None
    iteration = 0

    # 7. Iterative Benders-like algorithm
    while True:
        elapsed = time.time() - start_time
        remaining = args.time_limit - elapsed
        if remaining <= 0:
            break

        mp.Params.TimeLimit = remaining
        mp.optimize()

        if mp.status == GRB.INFEASIBLE:
            break
        if mp.status == GRB.TIME_LIMIT and mp.SolCount == 0:
            break
        if mp.SolCount == 0:
            break

        x_val = [round(x[j].X) for j in range(n1)]
        y_val = [y[j].X for j in range(n2)]
        
        # Follower Problem (FP) for fixed x
        fp = gp.Model(env=env)
        fp.Params.OutputFlag = 0
        fp.Params.Seed = 0
        fp.Params.MIPGap = 1e-4
        fp.Params.NumericFocus = 0
        fp.Params.Threads = 1
        
        y_f = fp.addVars(n2, lb=0, vtype=GRB.CONTINUOUS)
        for j in range(n2):
            if (j + 1) in int_f_vars:
                y_f[j].vtype = GRB.INTEGER
                
        for i in range(m2):
            fp.addConstr(gp.quicksum(B2[i][j]*y_f[j] for j in range(n2)) <= b2[i] - sum(A2[i][j]*x_val[j] for j in range(n1)))
            
        fp.setObjective(gp.quicksum(d2[j]*y_f[j] for j in range(n2)), GRB.MAXIMIZE)
        
        fp_remaining = args.time_limit - (time.time() - start_time)
        if fp_remaining <= 0:
            break
        fp.Params.TimeLimit = fp_remaining
        fp.optimize()

        if fp.status != GRB.OPTIMAL and fp.status != GRB.TIME_LIMIT:
            break
        if fp.SolCount == 0:
            break

        is_fp_optimal = (fp.status == GRB.OPTIMAL)
        y_hat = [y_f[j].X for j in range(n2)]
        v_star = sum(d2[j]*y_hat[j] for j in range(n2))

        leader_obj = sum(c[j]*x_val[j] for j in range(n1))
        follower_obj_mp = sum(d2[j]*y_val[j] for j in range(n2))

        if is_fp_optimal and follower_obj_mp >= v_star - 1e-4:
            # We found an optimal optimistic bilevel solution
            if leader_obj > best_obj:
                best_obj = leader_obj
                best_sol = {
                    "objective_value": float(leader_obj),
                    "x": [float(x_v) for x_v in x_val],
                    "y": [float(y_v) for y_v in y_val],
                    "follower_objective_value": float(follower_obj_mp)
                }
                if logger:
                    logger.log_solution(best_obj, best_sol)
            break
        else:
            # Check if y_hat itself constitutes a valid optimistic bilevel response
            is_ul_feasible = True
            for i in range(m1):
                if sum(A1[i][j]*x_val[j] for j in range(n1)) + sum(B1[i][j]*y_hat[j] for j in range(n2)) > b1[i] + 1e-5:
                    is_ul_feasible = False
                    break
            
            if is_fp_optimal and is_ul_feasible:
                if leader_obj > best_obj:
                    best_obj = leader_obj
                    best_sol = {
                        "objective_value": float(leader_obj),
                        "x": [float(x_v) for x_v in x_val],
                        "y": [float(y_v) for y_v in y_hat],
                        "follower_objective_value": float(v_star)
                    }
                    if logger:
                        logger.log_solution(best_obj, best_sol)
            
            # Add dynamic intersection/optimality cut based on the exact integer property of x
            M_z = max(0.0, v_star - z_min)
            w = []
            for i in range(m2):
                C_i = sum(B2[i][j]*y_hat[j] for j in range(n2))
                # F_i maps exactly the limit. If A2*x <= F_i it is strictly feasible, if A2*x >= F_i + 1 it is infeasible.
                F_i = math.floor(b2[i] - C_i + 1e-6)
                w_i = mp.addVar(vtype=GRB.BINARY, name=f"w_{iteration}_{i}")
                w.append(w_i)
                mp.addGenConstrIndicator(w_i, 1, gp.quicksum(A2[i][j]*x[j] for j in range(n1)) >= F_i + 1)
            
            mp.addConstr(z >= v_star - M_z * gp.quicksum(w[i] for i in range(m2)))

        iteration += 1

    # 8. Output fallback and validation
    if best_sol is None:
        # Fallback dummy zero solution if completely failed
        best_sol = {
            "objective_value": 0.0,
            "x": [0.0]*n1,
            "y": [0.0]*n2,
            "follower_objective_value": 0.0
        }

    with open(args.solution_path, 'w') as f:
        json.dump(best_sol, f, indent=2)

if __name__ == "__main__":
    main()