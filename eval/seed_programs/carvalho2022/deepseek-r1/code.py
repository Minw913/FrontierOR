import json
import argparse
import gurobipy as gp
from gurobipy import GRB
import os
import math
from solution_logger import SolutionLogger

class CallbackState:
    def __init__(self):
        self.best_obj = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['dimensions']['n']
    m = data['dimensions']['m']
    p = data['dimensions']['p']
    
    demands = data['demands']
    inventory_costs = data['inventory_costs']
    processing_time_val = data['processing_time']
    setup_times = data['setup_times']
    setup_costs = data['setup_costs']
    machine_capacities = data['machine_capacities']
    max_production = data['max_production']
    max_setups_per_item = data['max_setups_per_item']
    minimum_lot_sizes = data['minimum_lot_sizes']
    shortcut_items = data['shortcut_items']
    machine_eligibility = data['machine_eligibility']

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model("production_planning", env=env)
    model.setParam("Threads", 1)
    model.setParam('LogToConsole', 0)
    model.setParam('TimeLimit', args.time_limit)

    x = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                if max_setups_per_item[i][k][t] > 0:
                    for s in range(t, p):
                        var_name = f'x_{i}_{k}_{t}_{s}'
                        x[(i, k, t, s)] = model.addVar(vtype=GRB.CONTINUOUS, name=var_name, lb=0.0)

    y = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(m):
                for t in range(p):
                    if max_setups_per_item[j][k][t] > 0:
                        var_name = f'y_{i}_{j}_{k}_{t}'
                        y[(i, j, k, t)] = model.addVar(vtype=GRB.INTEGER, name=var_name, lb=0.0)

    z = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                var_name = f'z_{i}_{k}_{t}'
                z[(i, k, t)] = model.addVar(vtype=GRB.BINARY, name=var_name)

    P = {}
    for i in range(n):
        for k in range(m):
            for t in range(p):
                expr = gp.LinExpr()
                if max_setups_per_item[i][k][t] > 0:
                    for s in range(t, p):
                        key = (i, k, t, s)
                        if key in x:
                            expr += x[key]
                P[(i, k, t)] = expr

    for i in range(n):
        for s in range(p):
            expr = gp.LinExpr()
            for k in range(m):
                for t in range(0, s + 1):
                    if max_setups_per_item[i][k][t] > 0:
                        key = (i, k, t, s)
                        if key in x:
                            expr += x[key]
            model.addConstr(expr == demands[i][s], name=f'demand_{i}_{s}')

    for k in range(m):
        for t in range(p):
            expr_prod = gp.LinExpr()
            for i in range(n):
                if max_setups_per_item[i][k][t] > 0:
                    expr_prod += P[(i, k, t)] * processing_time_val
                else:
                    pass
            expr_setup = gp.LinExpr()
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    key = (i, j, k, t)
                    if key in y:
                        expr_setup += y[key] * setup_times[i][j][k]
            model.addConstr(expr_prod + expr_setup <= machine_capacities[k][t], name=f'cap_{k}_{t}')

    for i in range(n):
        for t in range(p):
            expr = gp.LinExpr()
            for k in range(m):
                if max_setups_per_item[i][k][t] > 0:
                    expr += P[(i, k, t)]
            model.addConstr(expr <= max_production[i][t], name=f'maxprod_{i}_{t}')

    for i in range(n):
        for k in range(m):
            for t in range(p):
                if max_setups_per_item[i][k][t] == 0:
                    model.addConstr(P[(i, k, t)] == 0, name=f'not_eligible_prod_{i}_{k}_{t}')
                    for j in range(n):
                        if j != i:
                            key = (j, i, k, t)
                            if key in y:
                                model.addConstr(y[key] == 0, name=f'not_eligible_setup_in_{j}_{i}_{k}_{t}')

    for i in range(n):
        for k in range(m):
            for t in range(p):
                if max_setups_per_item[i][k][t] > 0:
                    expr_setups = gp.LinExpr()
                    for j in range(n):
                        if j != i:
                            key = (j, i, k, t)
                            if key in y:
                                expr_setups += y[key]
                    model.addConstr(expr_setups <= max_setups_per_item[i][k][t], name=f'max_setups_{i}_{k}_{t}')

    for i in range(n):
        for k in range(m):
            for t in range(p):
                if max_setups_per_item[i][k][t] > 0:
                    expr_setups = gp.LinExpr()
                    for j in range(n):
                        if j != i:
                            key = (j, i, k, t)
                            if key in y:
                                expr_setups += y[key]
                    model.addConstr(P[(i, k, t)] >= minimum_lot_sizes[i] * expr_setups, name=f'min_lot_{i}_{k}_{t}')

    for k in range(m):
        for t in range(p - 1):
            for j in range(n):
                lhs = z[(j, k, t)]
                for i in range(n):
                    if i == j:
                        continue
                    key = (i, j, k, t)
                    if key in y:
                        lhs += y[key]
                rhs = z[(j, k, t + 1)]
                for i in range(n):
                    if i == j:
                        continue
                    key = (j, i, k, t)
                    if key in y:
                        rhs += y[key]
                model.addConstr(lhs == rhs, name=f'flow_{j}_{k}_{t}')

    for k in range(m):
        for t in range(p):
            expr = gp.LinExpr()
            for i in range(n):
                expr += z[(i, k, t)]
            model.addConstr(expr == 1, name=f'one_state_{k}_{t}')

    obj = gp.LinExpr()

    for i in range(n):
        for k in range(m):
            for t in range(p):
                for s in range(t, p):
                    key = (i, k, t, s)
                    if key in x:
                        holding_periods = s - t
                        obj += inventory_costs[i] * holding_periods * x[key]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(m):
                for t in range(p):
                    key = (i, j, k, t)
                    if key in y:
                        obj += setup_costs[i][j][k] * y[key]

    model.setObjective(obj, GRB.MINIMIZE)

    cb_state = CallbackState()

    def callback_function(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if cb_state.best_obj is None or obj_val < cb_state.best_obj:
                cb_state.best_obj = obj_val
                if logger:
                    logger.log(obj_val)

    model.optimize(callback_function if logger else None)

    if model.status == GRB.OPTIMAL:
        objective_value = model.ObjVal
    elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        objective_value = model.ObjVal
    else:
        if model.SolCount > 0:
            objective_value = model.ObjVal
        else:
            raise Exception("No feasible solution found within time limit")

    production_dict = {}
    for key, var in x.items():
        if var.X > 1e-6:
            i, k, t, s = key
            production_dict[f'x_{i}_{k}_{t}_{s}'] = float(var.X)

    setups_dict = {}
    for key, var in y.items():
        if var.X > 0.5:
            i, j, k, t = key
            setups_dict[f'y_{i}_{j}_{k}_{t}'] = float(var.X)

    carryover_dict = {}
    for key, var in z.items():
        if var.X > 0.5:
            i, k, t = key
            carryover_dict[f'z_{i}_{k}_{t}'] = 1

    solution = {
        "objective_value": float(objective_value),
        "production": production_dict,
        "setups": setups_dict,
        "carryover": carryover_dict
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()