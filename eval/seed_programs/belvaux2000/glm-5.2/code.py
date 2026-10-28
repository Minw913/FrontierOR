import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    NI = data['parameters']['NI']
    NK = data['parameters']['NK']
    NT = data['parameters']['NT']

    f_cost = data['costs']['setup_cost_f']
    g_cost = data['costs']['startup_cost_g']
    h = data['costs']['holding_cost_h']
    e_cost = data['costs']['backlogging_cost_e']

    C = data['machine_data']['capacity_C']
    sigma = data['machine_data']['startup_time_sigma']

    demand = data['demand']

    s0 = data['initial_conditions']['initial_stock_s0']
    r0 = data['initial_conditions']['initial_backlog_r0']

    start_time = time.time()

    env = gp.Env()
    model = gp.Model(env=env)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 0)

    elapsed = time.time() - start_time
    remaining = max(1, args.time_limit - elapsed)
    model.setParam('TimeLimit', remaining)

    # Variables
    s = {}
    r = {}
    x = {}
    y = {}
    z = {}

    for i in range(NI):
        for t in range(NT):
            s[i, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f's_{i}_{t}')
            r[i, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'r_{i}_{t}')

    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                x[i, k, t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'x_{i}_{k}_{t}')
                y[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{k}_{t}')
                z[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f'z_{i}_{k}_{t}')

    model.update()

    # Objective
    obj = gp.LinExpr()
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                obj += f_cost * y[i, k, t]
                obj += g_cost * z[i, k, t]
    for i in range(NI):
        for t in range(NT):
            obj += h[i] * s[i, t]
            obj += e_cost[i] * r[i, t]
    model.setObjective(obj, GRB.MINIMIZE)

    # Flow balance: s_t - r_t = s_{t-1} - r_{t-1} + prod_t - d_t
    for i in range(NI):
        for t in range(NT):
            prod_sum = gp.quicksum(x[i, k, t] for k in range(NK))
            if t == 0:
                model.addConstr(
                    s[i, t] - r[i, t] == s0[i] - r0[i] + prod_sum - demand[i][t]
                )
            else:
                model.addConstr(
                    s[i, t] - r[i, t] == s[i, t-1] - r[i, t-1] + prod_sum - demand[i][t]
                )

    # At most one item per machine per period
    for k in range(NK):
        for t in range(NT):
            model.addConstr(gp.quicksum(y[i, k, t] for i in range(NI)) <= 1)

    # Production linking and capacity with startup loss
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                model.addConstr(x[i, k, t] <= C[k] * y[i, k, t])
                model.addConstr(x[i, k, t] + sigma[k] * z[i, k, t] <= C[k])

    # Startup definition
    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                if t == 0:
                    model.addConstr(z[i, k, t] == y[i, k, t])
                else:
                    model.addConstr(z[i, k, t] >= y[i, k, t] - y[i, k, t-1])
                    model.addConstr(z[i, k, t] <= y[i, k, t])
                    model.addConstr(z[i, k, t] <= 1 - y[i, k, t-1])

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                production = {}
                stock = {}
                backlog = {}
                setup = {}
                startup = {}
                for i in range(NI):
                    for t in range(NT):
                        stock[f'{i}_{t}'] = model.cbGetSolution(s[i, t])
                        backlog[f'{i}_{t}'] = model.cbGetSolution(r[i, t])
                        for k in range(NK):
                            production[f'{i}_{k}_{t}'] = model.cbGetSolution(x[i, k, t])
                            setup[f'{i}_{k}_{t}'] = round(model.cbGetSolution(y[i, k, t]))
                            startup[f'{i}_{k}_{t}'] = round(model.cbGetSolution(z[i, k, t]))
                sol = {
                    "objective_value": obj_val,
                    "production": production,
                    "stock": stock,
                    "backlog": backlog,
                    "setup": setup,
                    "startup": startup
                }
                logger.log_solution(obj_val, sol)

    model.optimize(callback)

    # Extract and write solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        production = {}
        stock = {}
        backlog = {}
        setup = {}
        startup = {}

        for i in range(NI):
            for t in range(NT):
                stock[f'{i}_{t}'] = s[i, t].X
                backlog[f'{i}_{t}'] = r[i, t].X
                for k in range(NK):
                    production[f'{i}_{k}_{t}'] = x[i, k, t].X
                    setup[f'{i}_{k}_{t}'] = round(y[i, k, t].X)
                    startup[f'{i}_{k}_{t}'] = round(z[i, k, t].X)

        solution = {
            "objective_value": obj_val,
            "production": production,
            "stock": stock,
            "backlog": backlog,
            "setup": setup,
            "startup": startup
        }
    else:
        # No solution found — output trivially feasible solution (all backlog/stock from initial)
        production = {}
        stock = {}
        backlog = {}
        setup = {}
        startup = {}
        for i in range(NI):
            net = s0[i] - r0[i]
            for t in range(NT):
                net -= demand[i][t]
                if net >= 0:
                    stock[f'{i}_{t}'] = float(net)
                    backlog[f'{i}_{t}'] = 0.0
                else:
                    stock[f'{i}_{t}'] = 0.0
                    backlog[f'{i}_{t}'] = float(-net)
                for k in range(NK):
                    production[f'{i}_{k}_{t}'] = 0.0
                    setup[f'{i}_{k}_{t}'] = 0
                    startup[f'{i}_{k}_{t}'] = 0
        obj_val = sum(h[i] * stock[f'{i}_{t}'] for i in range(NI) for t in range(NT)) + \
                  sum(e_cost[i] * backlog[f'{i}_{t}'] for i in range(NI) for t in range(NT))
        solution = {
            "objective_value": obj_val,
            "production": production,
            "stock": stock,
            "backlog": backlog,
            "setup": setup,
            "startup": startup
        }

    with open(args.solution_path, 'w') as fout:
        json.dump(solution, fout, indent=2)

    env.dispose()


if __name__ == '__main__':
    main()