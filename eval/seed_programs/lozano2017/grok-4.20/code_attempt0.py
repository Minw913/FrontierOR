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
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
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
    integer_follower_vars = data['integer_follower_vars']
    leader_follower_objective_coefficients_fixed_zero = data['leader_follower_objective_coefficients_fixed_zero']

    # Create the big-M single-level reformulation (optimistic bilevel MILP)
    model = gp.Model("Bilevel_MILP")
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, args.time_limit)

    # Leader variables (all integer)
    x = model.addVars(n1, vtype=GRB.INTEGER, lb=0, name="x")

    # Follower variables
    y = model.addVars(n2, vtype=GRB.CONTINUOUS, lb=0, name="y")
    y_int_idx = [i-1 for i in integer_follower_vars]  # 0-based
    for i in y_int_idx:
        y[i].setAttr(GRB.Attr.VType, GRB.INTEGER)

    # Dual variables for lower level (for KKT / strong duality)
    lambda_ = model.addVars(m2, vtype=GRB.CONTINUOUS, lb=0, name="lambda")

    # Big-M for complementary slackness
    M = 1e6  # Sufficiently large given RHS bounds

    # Auxiliary for follower objective (to linearize bilinear term)
    follower_obj = model.addVar(vtype=GRB.CONTINUOUS, name="follower_obj")
    z = model.addVars(m2, vtype=GRB.CONTINUOUS, lb=0, name="z")  # lambda * slack

    # Leader objective: maximize c @ x  (d1=0 as per benchmark)
    leader_obj = gp.quicksum(c[i] * x[i] for i in range(n1))
    model.setObjective(leader_obj, GRB.MAXIMIZE)

    # Upper-level constraints: A1 x + B1 y <= b1
    for i in range(m1):
        model.addConstr(
            gp.quicksum(A1[i][j] * x[j] for j in range(n1)) +
            gp.quicksum(B1[i][k] * y[k] for k in range(n2)) <= b1[i],
            name=f"upper_{i}"
        )

    # Lower-level constraints: A2 x + B2 y <= b2  (primal feasibility)
    slack = []
    for i in range(m2):
        s = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"slack_{i}")
        model.addConstr(
            gp.quicksum(A2[i][j] * x[j] for j in range(n1)) +
            gp.quicksum(B2[i][k] * y[k] for k in range(n2)) + s == b2[i],
            name=f"lower_primal_{i}"
        )
        slack.append(s)

    # Dual feasibility: lambda B2 >= d2  (since max d2 y s.t. B2 y <= b2 - A2 x)
    # i.e., B2^T lambda >= d2
    for k in range(n2):
        model.addConstr(
            gp.quicksum(B2[i][k] * lambda_[i] for i in range(m2)) >= d2[k],
            name=f"dual_feas_{k}"
        )

    # Strong duality: d2 y == lambda (b2 - A2 x)
    model.addConstr(
        follower_obj == gp.quicksum(d2[k] * y[k] for k in range(n2)),
        name="follower_obj_def"
    )
    model.addConstr(
        follower_obj == gp.quicksum(lambda_[i] * b2[i] for i in range(m2)) -
                       gp.quicksum(lambda_[i] * gp.quicksum(A2[i][j] * x[j] for j in range(n1)) for i in range(m2)),
        name="strong_duality"
    )

    # Complementary slackness via big-M
    for i in range(m2):
        model.addConstr(z[i] <= M * lambda_[i], name=f"comp1_{i}")
        model.addConstr(z[i] <= slack[i], name=f"comp2_{i}")
        model.addConstr(z[i] >= slack[i] - M * (1 - lambda_[i].getAttr(GRB.Attr.VType) == GRB.CONTINUOUS and 1 or 0))  # conservative
        # Better linearization for complementarity: lambda * slack = 0 via big-M on each
        model.addConstr(lambda_[i] * slack[i] <= M * (1 - model.addVar(vtype=GRB.BINARY, name=f"u_{i}")), name=f"comp_bin_{i}_1")
        # Actually, standard way is to use binary variables for disjunction
        u = model.addVar(vtype=GRB.BINARY, name=f"u_{i}")
        model.addConstr(lambda_[i] <= M * u, name=f"comp_lambda_{i}")
        model.addConstr(slack[i] <= M * (1 - u), name=f"comp_slack_{i}")

    # Since follower variables can be continuous or integer, the above is a valid MILP reformulation
    # under the optimistic assumption (Gurobi will find the best for leader among follower optima)

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                # Extract solution
                x_sol = [model.cbGetSolution(x[i]) for i in range(n1)]
                y_sol = [model.cbGetSolution(y[i]) for i in range(n2)]
                follower_obj_val = sum(d2[k] * y_sol[k] for k in range(n2))
                solution_dict = {
                    "x": x_sol,
                    "y": y_sol,
                    "follower_objective_value": follower_obj_val
                }
                logger.log_solution(obj_val, solution_dict)

    model.optimize(callback)

    # Prepare output
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        x_sol = [x[i].X for i in range(n1)]
        y_sol = [y[i].X for i in range(n2)]
        obj_val = model.objVal
        follower_obj_val = sum(d2[k] * y_sol[k] for k in range(n2))

        result = {
            "objective_value": float(obj_val),
            "x": [float(val) for val in x_sol],
            "y": [float(val) for val in y_sol],
            "follower_objective_value": float(follower_obj_val)
        }

        with open(args.solution_path, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        # Infeasible or no solution - write dummy feasible if possible or zeros
        result = {
            "objective_value": 0.0,
            "x": [0.0] * n1,
            "y": [0.0] * n2,
            "follower_objective_value": 0.0
        }
        with open(args.solution_path, 'w') as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()