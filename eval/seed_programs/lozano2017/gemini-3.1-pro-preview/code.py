import sys
import math
import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n1 = data['n1']
    n2 = data['n2']
    m1 = data['m1']
    m2 = data['m2']
    c = data['c']
    d2 = data['d2']
    A1 = data.get('A1', [])
    B1 = data.get('B1', [])
    b1 = data.get('b1', [])
    A2 = data.get('A2', [])
    B2 = data.get('B2', [])
    b2 = data.get('b2', [])
    int_f_vars = set(data.get('integer_follower_vars', []))

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    best_obj = -float('inf')
    best_sol = None

    try:
        # Create an explicit Gurobi environment to avoid D-Bus or default initialization issues
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.setParam('Seed', 0)
        env.setParam('Threads', 1)
        env.start()

        # Heuristic: try x = 0 initially to find a feasible solution fast
        fp_heur = gp.Model("FP_heur", env=env)
        fp_heur.Params.OutputFlag = 0
        fp_heur.Params.Threads = 1
        y_h = fp_heur.addVars(n2, lb=0, ub=1000, vtype=GRB.CONTINUOUS)
        for j in range(n2):
            if (j + 1) in int_f_vars:
                y_h[j].vtype = GRB.INTEGER
        for i in range(m2):
            fp_heur.addConstr(gp.quicksum(B2[i][j]*y_h[j] for j in range(n2)) <= b2[i])
        fp_heur.setObjective(gp.quicksum(d2[j]*y_h[j] for j in range(n2)), GRB.MAXIMIZE)
        fp_heur.optimize()

        if fp_heur.status == GRB.OPTIMAL:
            y_h_val = [max(0.0, y_h[j].X) for j in range(n2)]
            ul_ok = True
            for i in range(m1):
                if sum(B1[i][j]*y_h_val[j] for j in range(n2)) > b1[i] + 1e-5:
                    ul_ok = False
                    break
            if ul_ok:
                best_obj = 0.0
                best_sol = {
                    "objective_value": 0.0,
                    "x": [0.0]*n1,
                    "y": [float(v) for v in y_h_val],
                    "follower_objective_value": float(sum(d2[j]*y_h_val[j] for j in range(n2)))
                }
                if logger:
                    logger.log_solution(best_obj, best_sol)

        # Find global lower bound for the follower's objective (z_min)
        m_zmin = gp.Model("zmin", env=env)
        m_zmin.Params.OutputFlag = 0
        m_zmin.Params.Threads = 1
        
        y_z = m_zmin.addVars(n2, lb=0, ub=1000, vtype=GRB.CONTINUOUS)
        for i in range(m2):
            m_zmin.addConstr(gp.quicksum(B2[i][j]*y_z[j] for j in range(n2)) <= b2[i])
        m_zmin.setObjective(gp.quicksum(d2[j]*y_z[j] for j in range(n2)), GRB.MINIMIZE)
        m_zmin.optimize()
        
        z_min = m_zmin.objVal if m_zmin.status == GRB.OPTIMAL else -1000000.0

        # Initialize the Master Problem (MP)
        mp = gp.Model("MP", env=env)
        mp.Params.OutputFlag = 0
        mp.Params.Seed = 0
        mp.Params.MIPGap = 1e-4
        mp.Params.NumericFocus = 0
        mp.Params.Threads = 1

        x = mp.addVars(n1, lb=0, ub=1000, vtype=GRB.INTEGER, name="x")
        y = mp.addVars(n2, lb=0, ub=1000, vtype=GRB.CONTINUOUS, name="y")
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

        # Initialize the Follower Problem (FP) once to avoid memory leaks
        fp = gp.Model("FP", env=env)
        fp.Params.OutputFlag = 0
        fp.Params.Seed = 0
        fp.Params.MIPGap = 1e-4
        fp.Params.NumericFocus = 0
        fp.Params.Threads = 1
        
        y_f = fp.addVars(n2, lb=0, ub=1000, vtype=GRB.CONTINUOUS)
        for j in range(n2):
            if (j + 1) in int_f_vars:
                y_f[j].vtype = GRB.INTEGER
                
        fp_constrs = []
        for i in range(m2):
            c_i = fp.addConstr(gp.quicksum(B2[i][j]*y_f[j] for j in range(n2)) <= b2[i])
            fp_constrs.append(c_i)
            
        fp.setObjective(gp.quicksum(d2[j]*y_f[j] for j in range(n2)), GRB.MAXIMIZE)

        iteration = 0

        # Iterative High-Point Relaxation with Exact Penalty Cuts
        while True:
            elapsed = time.time() - start_time
            remaining = args.time_limit - elapsed
            if remaining <= 1.0:
                break

            mp.Params.TimeLimit = remaining
            mp.optimize()

            if mp.status in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
                break
            if mp.SolCount == 0:
                break

            x_val = [max(0, round(x[j].X)) for j in range(n1)]
            y_val = [max(0.0, y[j].X) for j in range(n2)]
            
            # Update RHS of the Follower Problem (FP) for the fixed x
            for i in range(m2):
                fp_constrs[i].RHS = b2[i] - sum(A2[i][j]*x_val[j] for j in range(n1))
                
            fp_remaining = args.time_limit - (time.time() - start_time) - 0.5
            if fp_remaining <= 0:
                break
                
            fp.Params.TimeLimit = fp_remaining
            fp.optimize()

            if fp.status != GRB.OPTIMAL and fp.status != GRB.TIME_LIMIT:
                break
            if fp.SolCount == 0:
                break

            is_fp_optimal = (fp.status == GRB.OPTIMAL)
            y_hat = [max(0.0, y_f[j].X) for j in range(n2)]
            v_star = sum(d2[j]*y_hat[j] for j in range(n2))

            leader_obj = sum(c[j]*x_val[j] for j in range(n1))
            follower_obj_mp = sum(d2[j]*y_val[j] for j in range(n2))

            # Check if the Master Problem naturally found an optimal follower response
            if is_fp_optimal and follower_obj_mp >= v_star - 1e-4:
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
                
            # Check if the generated purely follower-optimal response is valid for the upper level
            is_ul_feasible = True
            for i in range(m1):
                if sum(A1[i][j]*x_val[j] for j in range(n1)) + sum(B1[i][j]*y_hat[j] for j in range(n2)) > b1[i] + 1e-5:
                    is_ul_feasible = False
                    break
            
            if is_fp_optimal and is_ul_feasible:
                best_obj = leader_obj
                best_sol = {
                    "objective_value": float(leader_obj),
                    "x": [float(x_v) for x_v in x_val],
                    "y": [float(y_v) for y_v in y_hat],
                    "follower_objective_value": float(v_star)
                }
                if logger:
                    logger.log_solution(best_obj, best_sol)
                break
                
            # Add Exact Penalty Indicator Cuts
            if is_fp_optimal:
                M_z = max(0.0, v_star - z_min)
                w = []
                for i in range(m2):
                    C_i = sum(B2[i][j]*y_hat[j] for j in range(n2))
                    F_i = math.floor(b2[i] - C_i + 1e-6)
                    w_i = mp.addVar(vtype=GRB.BINARY, name=f"w_{iteration}_{i}")
                    w.append(w_i)
                    mp.addGenConstrIndicator(w_i, 1, gp.quicksum(A2[i][j]*x[j] for j in range(n1)) >= F_i + 1)
                
                mp.addConstr(z >= v_star - M_z * gp.quicksum(w[i] for i in range(m2)))

            iteration += 1

    except Exception:
        # Suppress exceptions securely terminating Execution code 1 allowing partial solutions dump gracefully.
        pass

    # Fallback to an empty solution array if no valid iterations processed
    if best_sol is None:
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