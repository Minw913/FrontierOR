import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    n1 = instance['n1']
    n2 = instance['n2']
    m1 = instance['m1']
    m2 = instance['m2']
    c = instance['c']
    d2 = instance['d2']
    A1 = instance['A1']
    B1 = instance['B1']
    b1 = instance['b1']
    A2 = instance['A2']
    B2 = instance['B2']
    b2 = instance['b2']
    num_integer_follower_vars = instance['num_integer_follower_vars']
    integer_follower_vars = instance['integer_follower_vars']

    try:
        model = gp.Model()
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        # Leader variables
        x = model.addVars(n1, vtype=GRB.INTEGER, name='x')

        # Follower variables
        y = model.addVars(n2, name='y')
        for i in range(num_integer_follower_vars):
            y[integer_follower_vars[i] - 1].vtype = GRB.INTEGER

        # Upper-level constraints
        for i in range(m1):
            expr = gp.LinExpr()
            for j in range(n1):
                expr += A1[i][j] * x[j]
            for j in range(n2):
                expr += B1[i][j] * y[j]
            model.addConstr(expr <= b1[i], name=f'upper_level_constr_{i}')

        # Lower-level constraints
        for i in range(m2):
            expr = gp.LinExpr()
            for j in range(n1):
                expr += A2[i][j] * x[j]
            for j in range(n2):
                expr += B2[i][j] * y[j]
            model.addConstr(expr <= b2[i], name=f'lower_level_constr_{i}')

        # Leader objective
        leader_obj = gp.LinExpr()
        for j in range(n1):
            leader_obj += c[j] * x[j]

        # Follower objective
        follower_obj = gp.LinExpr()
        for j in range(n2):
            follower_obj += d2[j] * y[j]

        # Set objective sense
        model.ModelSense = GRB.MAXIMIZE

        # Bilevel optimization
        model.setParam('InfUnbdInfo', 1)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            x_val = [x[i].X for i in range(n1)]
            y_val = [y[i].X for i in range(n2)]
            objective_value = leader_obj.getValue()
            follower_objective_value = follower_obj.getValue()
            return objective_value, x_val, y_val, follower_objective_value
        else:
            return None, None, None, None

    except gp.GurobiError as e:
        print(f'Gurobi error: {e}')
        return None, None, None, None

def main():
    args = parse_args()
    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    model = gp.Model()
    model.Params.TimeLimit = args.time_limit

    n1 = instance['n1']
    n2 = instance['n2']
    m1 = instance['m1']
    m2 = instance['m2']
    c = instance['c']
    d2 = instance['d2']
    A1 = instance['A1']
    B1 = instance['B1']
    b1 = instance['b1']
    A2 = instance['A2']
    B2 = instance['B2']
    b2 = instance['b2']
    num_integer_follower_vars = instance['num_integer_follower_vars']
    integer_follower_vars = instance['integer_follower_vars']

    try:
        # Leader variables
        x = model.addVars(n1, vtype=GRB.INTEGER, name='x')

        # Follower variables
        y = model.addVars(n2, name='y')
        for i in range(num_integer_follower_vars):
            y[integer_follower_vars[i] - 1].vtype = GRB.INTEGER

        # Upper-level constraints
        for i in range(m1):
            expr = gp.LinExpr()
            for j in range(n1):
                expr += A1[i][j] * x[j]
            for j in range(n2):
                expr += B1[i][j] * y[j]
            model.addConstr(expr <= b1[i], name=f'upper_level_constr_{i}')

        # Lower-level constraints
        for i in range(m2):
            expr = gp.LinExpr()
            for j in range(n1):
                expr += A2[i][j] * x[j]
            for j in range(n2):
                expr += B2[i][j] * y[j]
            model.addConstr(expr <= b2[i], name=f'lower_level_constr_{i}')

        # Leader objective
        leader_obj = gp.LinExpr()
        for j in range(n1):
            leader_obj += c[j] * x[j]

        # Follower objective
        follower_obj = gp.LinExpr()
        for j in range(n2):
            follower_obj += d2[j] * y[j]

        # Set objective sense
        model.ModelSense = GRB.MAXIMIZE

        # Add constraint for follower optimality
        model.addConstr(follower_obj == gp.max_(follower_obj, "follower_subproblem"))

        # Bilevel optimization using Gurobi's built-in support
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                x_val = model.cbGetSolution(model.getVars()[:n1])
                y_val = model.cbGetSolution(model.getVars()[n1:])
                objective_value = model.cbGetSolution(leader_obj)
                follower_objective_value = model.cbGetSolution(follower_obj)
                if logger:
                    solution_dict = {
                        'objective_value': objective_value,
                        'x': x_val,
                        'y': y_val,
                        'follower_objective_value': follower_objective_value
                    }
                    logger.log_solution(objective_value, solution_dict)

        model.optimize(callback)

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            x_val = [x[i].X for i in range(n1)]
            y_val = [y[i].X for i in range(n2)]
            objective_value = leader_obj.getValue()
            follower_objective_value = follower_obj.getValue()
            solution = {
                'objective_value': objective_value,
                'x': x_val,
                'y': y_val,
                'follower_objective_value': follower_objective_value
            }
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f)
        else:
            print("No solution found.")

    except gp.GurobiError as e:
        print(f'Gurobi error: {e}')

if __name__ == '__main__':
    main()