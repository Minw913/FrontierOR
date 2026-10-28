import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit):
    n_items = instance['n_items']
    n_periods = instance['n_periods']
    capacity = instance['capacity']
    initial_inventory = instance['initial_inventory']
    demands = instance['demands']
    setup_costs = instance['setup_costs']
    holding_costs = instance['holding_costs']
    variable_costs = instance['variable_costs']
    setup_times = instance['setup_times']
    variable_times = instance['variable_times']
    big_M = instance['big_M']

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0

    x = model.addVars(n_items, n_periods, vtype=GRB.CONTINUOUS, name='x')
    s = model.addVars(n_items, n_periods + 1, vtype=GRB.CONTINUOUS, name='s')
    y = model.addVars(n_items, n_periods, vtype=GRB.BINARY, name='y')

    model.setObjective(
        gp.quicksum(setup_costs[i][t] * y[i, t] for i in range(n_items) for t in range(n_periods)) +
        gp.quicksum(variable_costs[i][t] * x[i, t] for i in range(n_items) for t in range(n_periods)) +
        gp.quicksum(holding_costs[i][t] * s[i, t] for i in range(n_items) for t in range(n_periods)),
        GRB.MINIMIZE
    )

    model.addConstrs((s[i, t] + x[i, t] == demands[i][t] + s[i, t + 1] for i in range(n_items) for t in range(n_periods)), name='inventory_balance')
    model.addConstrs((x[i, t] <= big_M[i][t] * y[i, t] for i in range(n_items) for t in range(n_periods)), name='production_limit')
    model.addConstrs((gp.quicksum(setup_times[i][t] * y[i, t] + variable_times[i][t] * x[i, t] for i in range(n_items)) <= capacity[t] for t in range(n_periods)), name='capacity_constraint')
    model.addConstrs((s[i, 0] == initial_inventory[i] for i in range(n_items)), name='initial_inventory')
    model.addConstrs((s[i, n_periods] == 0 for i in range(n_items)), name='final_inventory')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        x_sol = [[x[i, t].X for t in range(n_periods)] for i in range(n_items)]
        s_sol = [[s[i, t].X for t in range(n_periods + 1)] for i in range(n_items)]
        y_sol = [[int(y[i, t].X) for t in range(n_periods)] for i in range(n_items)]
        return objective_value, x_sol, s_sol, y_sol
    else:
        return None, None, None, None

def write_solution(solution_path, objective_value, x, s, y):
    solution = {
        'objective_value': objective_value,
        'x': x,
        's': s,
        'y': y
    }
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)

    def callback(model, where):
        if where == GRB.Callback.MIP:
            incumbent = model.cbGet(GRB.Callback.MIP_OBJBST)
            if model._incumbent != incumbent:
                model._incumbent = incumbent
                if logger:
                    logger.log(incumbent)

    model = gp.Model()
    model._incumbent = float('inf')

    n_items = instance['n_items']
    n_periods = instance['n_periods']
    capacity = instance['capacity']
    initial_inventory = instance['initial_inventory']
    demands = instance['demands']
    setup_costs = instance['setup_costs']
    holding_costs = instance['holding_costs']
    variable_costs = instance['variable_costs']
    setup_times = instance['setup_times']
    variable_times = instance['variable_times']
    big_M = instance['big_M']

    x = model.addVars(n_items, n_periods, vtype=GRB.CONTINUOUS, name='x')
    s = model.addVars(n_items, n_periods + 1, vtype=GRB.CONTINUOUS, name='s')
    y = model.addVars(n_items, n_periods, vtype=GRB.BINARY, name='y')

    model.setObjective(
        gp.quicksum(setup_costs[i][t] * y[i, t] for i in range(n_items) for t in range(n_periods)) +
        gp.quicksum(variable_costs[i][t] * x[i, t] for i in range(n_items) for t in range(n_periods)) +
        gp.quicksum(holding_costs[i][t] * s[i, t] for i in range(n_items) for t in range(n_periods)),
        GRB.MINIMIZE
    )

    model.addConstrs((s[i, t] + x[i, t] == demands[i][t] + s[i, t + 1] for i in range(n_items) for t in range(n_periods)), name='inventory_balance')
    model.addConstrs((x[i, t] <= big_M[i][t] * y[i, t] for i in range(n_items) for t in range(n_periods)), name='production_limit')
    model.addConstrs((gp.quicksum(setup_times[i][t] * y[i, t] + variable_times[i][t] * x[i, t] for i in range(n_items)) <= capacity[t] for t in range(n_periods)), name='capacity_constraint')
    model.addConstrs((s[i, 0] == initial_inventory[i] for i in range(n_items)), name='initial_inventory')
    model.addConstrs((s[i, n_periods] == 0 for i in range(n_items)), name='final_inventory')

    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0

    model.optimize(callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        x_sol = [[x[i, t].X for t in range(n_periods)] for i in range(n_items)]
        s_sol = [[s[i, t].X for t in range(n_periods + 1)] for i in range(n_items)]
        y_sol = [[int(y[i, t].X) for t in range(n_periods)] for i in range(n_items)]
        write_solution(args.solution_path, objective_value, x_sol, s_sol, y_sol)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()