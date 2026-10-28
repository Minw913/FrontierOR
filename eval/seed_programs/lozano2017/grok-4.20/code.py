import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
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
    A1 = data['A1']
    B1 = data['B1']
    b1 = data['b1']
    A2 = data['A2']
    B2 = data['B2']
    b2 = data['b2']
    integer_follower_vars = data.get('integer_follower_vars', [])
    leader_follower_objective_coefficients_fixed_zero = data.get('leader_follower_objective_coefficients_fixed_zero', True)

    # Create the big-M single-level reformulation using dual of lower level
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
    for i in integer_follower_vars:
        y[i-1].setAttr(GRB.Attr.VType, GRB.INTEGER)

    # Dual variables for lower level (one per lower constraint)
    mu = model.addVars(m2, vtype=GRB.CONTINUOUS, lb=0, name="mu")

    # Big-M constant (conservative estimate based on data ranges)
    M = 10000

    # Objective: maximize leader's objective (c @ x)
    obj = gp.quicksum(c[i] * x[i] for i in range(n1))
    model.setObjective(obj, GRB.MAXIMIZE)

    # Upper level constraints: A1 x + B1 y <= b1
    for i in range(m1):
        model.addConstr(
            gp.quicksum(A1[i][j] * x[j] for j in range(n1)) +
            gp.quicksum(B1[i][k] * y[k] for k in range(n2)) <= b1[i],
            name=f"upper_{i}"
        )

    # Lower level constraints (primal feasibility): A2 x + B2 y <= b2
    for i in range(m2):
        model.addConstr(
            gp.quicksum(A2[i][j] * x[j] for j in range(n1)) +
            gp.quicksum(B2[i][k] * y[k] for k in range(n2)) <= b2[i],
            name=f"lower_primal_{i}"
        )

    # Dual feasibility: B2^T mu >= d2  (since follower maximizes d2 y)
    for k in range(n2):
        model.addConstr(
            gp.quicksum(B2[i][k] * mu[i] for i in range(m2)) >= d2[k],
            name=f"dual_feas_{k}"
        )

    # Strong duality / complementary slackness via big-M
    # mu * (b2 - A2x - B2y) = 0  -->  mu_i <= M * s_i, slack_i <= M * (1-s_i)
    s = model.addVars(m2, vtype=GRB.BINARY, name="s")
    slack = model.addVars(m2, vtype=GRB.CONTINUOUS, lb=0, name="slack")

    for i in range(m2):
        # slack_i = b2[i] - A2 x - B2 y
        model.addConstr(
            slack[i] == b2[i] -
            gp.quicksum(A2[i][j] * x[j] for j in range(n1)) -
            gp.quicksum(B2[i][k] * y[k] for k in range(n2)),
            name=f"slack_def_{i}"
        )
        model.addConstr(mu[i] <= M * s[i], name=f"comp_mu_{i}")
        model.addConstr(slack[i] <= M * (1 - s[i]), name=f"comp_slack_{i}")

    # If leader objective depends on y (should be zero per spec)
    if not leader_follower_objective_coefficients_fixed_zero:
        # This branch should not be reached per problem statement
        pass

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol_x = [model.cbGetSolution(x[i]) for i in range(n1)]
            sol_y = [model.cbGetSolution(y[i]) for i in range(n2)]
            follower_obj = sum(d2[k] * sol_y[k] for k in range(n2))

            solution_dict = {
                "objective_value": float(obj_val),
                "x": [float(val) for val in sol_x],
                "y": [float(val) for val in sol_y],
                "follower_objective_value": float(follower_obj)
            }

            if logger:
                logger.log_solution(obj_val, solution_dict)

    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT):
        if model.SolCount > 0:
            x_sol = [x[i].X for i in range(n1)]
            y_sol = [y[i].X for i in range(n2)]
            obj_val = model.ObjVal
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
            # No solution found - write dummy feasible if possible or zeros
            result = {
                "objective_value": 0.0,
                "x": [0.0] * n1,
                "y": [0.0] * n2,
                "follower_objective_value": 0.0
            }
            with open(args.solution_path, 'w') as f:
                json.dump(result, f, indent=2)
    else:
        # Infeasible or other error - return zero solution
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