import argparse
import json
import math
import sys
import time
import os
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=0)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        data = json.load(f)
        
    I = data["num_facilities"]
    J = data["num_customers"]
    costs = data["facility_cost"]
    demands = data["customer_demands"]
    D = data["covering_demand_D"]
    B = data["budget_B"]
    
    # Parse or compute coverage mapping
    N = {}
    if "customer_to_facilities_coverage" in data:
        for k, v in data["customer_to_facilities_coverage"].items():
            N[int(k)] = [int(i) for i in v]
    else:
        radius = data["radius_of_coverage"]
        for j in range(J):
            N[j] = []
            c_coord = data["customer_coordinates"][j]
            for i in range(I):
                f_coord = data["facility_coordinates"][i]
                dist = math.hypot(c_coord[0] - f_coord[0], c_coord[1] - f_coord[1])
                if dist <= radius + 1e-7:
                    N[j].append(i)
                    
    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")
        
    time_limit = args.time_limit
    if time_limit > 0:
        time_psclp = time_limit / 2.0
        time_mclp = time_limit - time_psclp
    else:
        time_psclp = 0
        time_mclp = 0
        
    # ==========================================
    # Solve Partial Set Covering Location Problem
    # ==========================================
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    psclp_model = gp.Model("PSCLP", env=env)
    y_psclp = psclp_model.addVars(I, vtype=GRB.BINARY, name="y")
    x_psclp = psclp_model.addVars(J, vtype=GRB.BINARY, name="x")
    
    psclp_model.setObjective(gp.quicksum(costs[i] * y_psclp[i] for i in range(I)), GRB.MINIMIZE)
    psclp_model.addConstr(gp.quicksum(demands[j] * x_psclp[j] for j in range(J)) >= D, name="DemandCover")
    
    for j in range(J):
        cov_j = N.get(j, [])
        if cov_j:
            psclp_model.addConstr(x_psclp[j] <= gp.quicksum(y_psclp[i] for i in cov_j))
        else:
            psclp_model.addConstr(x_psclp[j] == 0)
            
    psclp_model.setParam('Seed', 0)
    psclp_model.setParam('MIPGap', 1e-4)
    psclp_model.setParam('NumericFocus', 0)
    psclp_model.setParam('Threads', 1)
    if time_psclp > 0:
        psclp_model.setParam('TimeLimit', time_psclp)
        
    def psclp_cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            y_vals = model.cbGetSolution(y_psclp)
            open_facs = [i for i in range(I) if y_vals[i] > 0.5]
            cov_demand = sum(demands[j] for j in range(J) if j in N and any(y_vals[i] > 0.5 for i in N[j]))
            
            sol_dict = {
                "objective_value": float(obj),
                "primary_problem_type": "PSCLP",
                "results": {
                    "PSCLP": {
                        "objective_value": float(obj),
                        "open_facilities": open_facs,
                        "covered_demand": float(cov_demand)
                    },
                    "MCLP": {
                        "objective_value": 0.0,
                        "open_facilities": [],
                        "covered_demand": 0.0
                    }
                }
            }
            if logger:
                logger.log_solution(float(obj), sol_dict)
                
    psclp_model.optimize(psclp_cb)
    
    psclp_best = {
        "objective_value": 0.0,
        "open_facilities": [],
        "covered_demand": 0.0
    }
    
    if psclp_model.SolCount > 0:
        y_vals = [y_psclp[i].X for i in range(I)]
        open_facs = [i for i in range(I) if y_vals[i] > 0.5]
        cov_demand = sum(demands[j] for j in range(J) if j in N and any(i in open_facs for i in N[j]))
        psclp_best = {
            "objective_value": float(psclp_model.ObjVal),
            "open_facilities": open_facs,
            "covered_demand": float(cov_demand)
        }
    
    # ==========================================
    # Solve Maximal Covering Location Problem
    # ==========================================
    mclp_env = gp.Env(empty=True)
    mclp_env.setParam("OutputFlag", 0)
    mclp_env.start()
    
    mclp_model = gp.Model("MCLP", env=mclp_env)
    y_mclp = mclp_model.addVars(I, vtype=GRB.BINARY, name="y")
    x_mclp = mclp_model.addVars(J, vtype=GRB.BINARY, name="x")
    
    mclp_model.setObjective(gp.quicksum(demands[j] * x_mclp[j] for j in range(J)), GRB.MAXIMIZE)
    mclp_model.addConstr(gp.quicksum(costs[i] * y_mclp[i] for i in range(I)) <= B, name="Budget")
    
    for j in range(J):
        cov_j = N.get(j, [])
        if cov_j:
            mclp_model.addConstr(x_mclp[j] <= gp.quicksum(y_mclp[i] for i in cov_j))
        else:
            mclp_model.addConstr(x_mclp[j] == 0)
            
    mclp_model.setParam('Seed', 0)
    mclp_model.setParam('MIPGap', 1e-4)
    mclp_model.setParam('NumericFocus', 0)
    mclp_model.setParam('Threads', 1)
    if time_mclp > 0:
        mclp_model.setParam('TimeLimit', time_mclp)
        
    mclp_model.optimize()
    
    mclp_best = {
        "objective_value": 0.0,
        "open_facilities": [],
        "covered_demand": 0.0
    }
    
    if mclp_model.SolCount > 0:
        y_vals = [y_mclp[i].X for i in range(I)]
        open_facs = [i for i in range(I) if y_vals[i] > 0.5]
        cov_demand = sum(demands[j] for j in range(J) if j in N and any(i in open_facs for i in N[j]))
        mclp_best = {
            "objective_value": float(mclp_model.ObjVal),
            "open_facilities": open_facs,
            "covered_demand": float(cov_demand)
        }
        
    # ==========================================
    # Output Solution
    # ==========================================
    final_sol = {
        "objective_value": float(psclp_best["objective_value"]),
        "primary_problem_type": "PSCLP",
        "results": {
            "PSCLP": psclp_best,
            "MCLP": mclp_best
        }
    }
    
    out_dir = os.path.dirname(args.solution_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=2)

if __name__ == "__main__":
    solve()