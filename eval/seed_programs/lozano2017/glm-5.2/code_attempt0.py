import gurobipy as gp
from gurobipy import GRB
import json
import argparse
import sys

# solution_logger is assumed to be pre-installed
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve_follower_mip(A2, B2, b2, d2, x, integer_follower_vars, n2, time_limit=10):
    """Solves the follower's MILP for a given leader choice x."""
    m = gp.Model("follower")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("Seed", 0)
    m.setParam("Threads", 1)
    
    y = m.addVars(n2, lb=0, name="y")
    for j in integer_follower_vars:
        y[j].vtype = GRB.INTEGER
    
    for i in range(len(A2)):
        m.addConstr(gp.quicksum(A2[i][j] * x[j] for j in range(len(x))) + 
                    gp.quicksum(B2[i][j] * y[j] for j in range(n2)) <= b2[i])
    
    m.setObjective(gp.quicksum(d2[j] * y[j] for j in range(n2)), GRB.MAXIMIZE)
    m.optimize()
    if m.status == GRB.OPTIMAL or m.SolCount > 0:
        return [y[j].x for j in range(n2)]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path and SolutionLogger else None

    with open(args.instance_path) as f:
        inst = json.load(f)

    n1 = inst['n1']
    n2 = inst['n2']
    m1 = inst['m1']
    m2 = inst['m2']
    c = inst['c']
    d2 = inst['d2']
    A1 = inst['A1']
    B1 = inst['B1']
    b1 = inst['b1']
    A2 = inst['A2']
    B2 = inst['B2']
    b2 = inst['b2']
    int_foll = inst['integer_follower_vars']
    
    int_foll_0 = [i - 1 for i in int_foll]

    best_obj = -float('inf')
    best_x = [0.0] * n1
    best_y = [0.0] * n2

    # Try x=0 as a fallback and baseline
    y0 = solve_follower_mip(A2, B2, b2, d2, [0]*n1, int_foll_0, n2, time_limit=10)
    if y0 is not None:
        feasible = True
        for i in range(m1):
            if sum(A1[i][j] * 0 for j in range(n1)) + sum(B1[i][j] * y0[j] for j in range(n2)) > b1[i] + 1e-6:
                feasible = False
                break
        if feasible:
            best_obj = 0.0
            best_x = [0.0] * n1
            best_y = y0
            if logger:
                logger.log_solution(best_obj, {"objective_value": best_obj, "x": best_x, "y": best_y, "follower_objective_value": sum(d2[j]*y0[j] for j in range(n2))})

    # Continuous KKT Model
    # The follower's continuous relaxation is represented using KKT conditions.
    model = gp.Model("continuous_kkt")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("OutputFlag", 0)
    
    max_b2 = max(b2) if b2 else 110
    max_b1 = max(b1) if b1 else 130
    max_x = max(max_b2, max_b1) * 10
    max_y = max(max_b1, max_b2) * 2
    
    x = model.addVars(n1, vtype=GRB.INTEGER, lb=0, ub=max_x, name="x")
    y = model.addVars(n2, vtype=GRB.CONTINUOUS, lb=0, ub=max_y, name="y")
    
    M_u = 1000
    M_s1 = max_b2
    M_s2 = 100000
    
    u = model.addVars(m2, vtype=GRB.CONTINUOUS, lb=0, ub=M_u, name="u")
    z1 = model.addVars(m2, vtype=GRB.BINARY, name="z1")
    z2 = model.addVars(n2, vtype=GRB.BINARY, name="z2")
    
    # Leader constraints
    for i in range(m1):
        model.addConstr(gp.quicksum(A1[i][j] * x[j] for j in range(n1)) + 
                        gp.quicksum(B1[i][j] * y[j] for j in range(n2)) <= b1[i])
    
    # Follower primal constraints
    slack1 = model.addVars(m2, vtype=GRB.CONTINUOUS, lb=0, name="slack1")
    for i in range(m2):
        model.addConstr(slack1[i] == b2[i] - gp.quicksum(A2[i][j] * x[j] for j in range(n1)) - 
                        gp.quicksum(B2[i][j] * y[j] for j in range(n2)))
        model.addConstr(u[i] <= M_u * (1 - z1[i]))
        model.addConstr(slack1[i] <= M_s1 * z1[i])
        
    # Follower dual constraints
    slack2 = model.addVars(n2, vtype=GRB.CONTINUOUS, lb=0, name="slack2")
    for j in range(n2):
        model.addConstr(slack2[j] == gp.quicksum(B2[i][j] * u[i] for i in range(m2)) - d2[j])
        model.addConstr(y[j] <= max_y * z2[j])
        model.addConstr(slack2[j] <= M_s2 * (1 - z2[j]))
        
    model.setObjective(gp.quicksum(c[j] * x[j] for j in range(n1)), GRB.MAXIMIZE)
    
    # Wrapping nonlocal variables in lists for callback
    nonlocal_best_obj = [best_obj]
    nonlocal_best_x = [best_x]
    nonlocal_best_y = [best_y]
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val > nonlocal_best_obj[0]:
                x_val = model.cbGetSolution(x)
                x_int = [int(round(val)) for val in x_val]
                # Evaluate true MILP follower
                y_true = solve_follower_mip(A2, B2, b2, d2, x_int, int_foll_0, n2, time_limit=5)
                if y_true is not None:
                    feasible = True
                    for i in range(m1):
                        if sum(A1[i][j] * x_int[j] for j in range(n1)) + sum(B1[i][j] * y_true[j] for j in range(n2)) > b1[i] + 1e-6:
                            feasible = False
                            break
                    if feasible:
                        nonlocal_best_obj[0] = obj_val
                        nonlocal_best_x[0] = [float(val) for val in x_int]
                        nonlocal_best_y[0] = y_true
                        if logger:
                            logger.log_solution(obj_val, {"objective_value": obj_val, "x": nonlocal_best_x[0], "y": nonlocal_best_y[0], "follower_objective_value": sum(d2[j]*y_true[j] for j in range(n2))})
    
    model.optimize(callback)
    
    best_obj = nonlocal_best_obj[0]
    best_x = nonlocal_best_x[0]
    best_y = nonlocal_best_y[0]

    # Check the final solutions found by the model
    if model.SolCount > 0:
        for sol in range(model.SolCount):
            model.setParam("SolutionNumber", sol)
            x_val = [int(round(x[j].Xn)) for j in range(n1)]
            obj_val = sum(c[j] * x_val[j] for j in range(n1))
            if obj_val > best_obj:
                y_true = solve_follower_mip(A2, B2, b2, d2, x_val, int_foll_0, n2, time_limit=5)
                if y_true is not None:
                    feasible = True
                    for i in range(m1):
                        if sum(A1[i][j] * x_val[j] for j in range(n1)) + sum(B1[i][j] * y_true[j] for j in range(n2)) > b1[i] + 1e-6:
                            feasible = False
                            break
                    if feasible:
                        best_obj = obj_val
                        best_x = [float(val) for val in x_val]
                        best_y = y_true
                        if logger:
                            logger.log_solution(best_obj, {"objective_value": best_obj, "x": best_x, "y": best_y, "follower_objective_value": sum(d2[j]*y_true[j] for j in range(n2))})
                            
    solution = {
        "objective_value": best_obj,
        "x": best_x,
        "y": best_y,
        "follower_objective_value": sum(d2[j] * best_y[j] for j in range(n2))
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()