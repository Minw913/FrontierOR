import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve_instance(instance, time_limit, logger):
    NI = instance['parameters']['NI']
    NK = instance['parameters']['NK']
    NT = instance['parameters']['NT']
    setup_cost_f = instance['costs']['setup_cost_f']
    startup_cost_g = instance['costs']['startup_cost_g']
    holding_cost_h = instance['costs']['holding_cost_h']
    backlogging_cost_e = instance['costs']['backlogging_cost_e']
    capacity_C = instance['machine_data']['capacity_C']
    startup_time_sigma = instance['machine_data']['startup_time_sigma']
    demand = instance['demand']
    initial_stock_s0 = instance['initial_conditions']['initial_stock_s0']
    initial_backlog_r0 = instance['initial_conditions']['initial_backlog_r0']

    model = gp.Model()
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output

    # Variables
    production = model.addVars(NI, NK, NT, vtype=GRB.CONTINUOUS, name='production')
    stock = model.addVars(NI, NT, vtype=GRB.CONTINUOUS, name='stock')
    backlog = model.addVars(NI, NT, vtype=GRB.CONTINUOUS, name='backlog')
    setup = model.addVars(NI, NK, NT, vtype=GRB.BINARY, name='setup')
    startup = model.addVars(NI, NK, NT, vtype=GRB.BINARY, name='startup')
    is_setup = model.addVars(NK, NT, vtype=GRB.BINARY, name='is_setup')

    # Objective
    model.setObjective(
        setup_cost_f * gp.quicksum(is_setup[k, t] for k in range(NK) for t in range(NT)) +
        startup_cost_g * gp.quicksum(startup[i, k, t] for i in range(NI) for k in range(NK) for t in range(NT)) +
        gp.quicksum(holding_cost_h[i] * stock[i, t] for i in range(NI) for t in range(NT)) +
        gp.quicksum(backlogging_cost_e[i] * backlog[i, t] for i in range(NI) for t in range(NT)),
        GRB.MINIMIZE
    )

    # Constraints
    for i in range(NI):
        for t in range(NT):
            model.addConstr(stock[i, t] - backlog[i, t] - (initial_stock_s0[i] - initial_backlog_r0[i] if t == 0 else stock[i, t-1] - backlog[i, t-1]) - gp.quicksum(production[i, k, t] for k in range(NK)) + demand[i][t] == 0, name=f'balance_{i}_{t}')

    for k in range(NK):
        for t in range(NT):
            model.addConstr(gp.quicksum(production[i, k, t] for i in range(NI)) + gp.quicksum(startup[i, k, t] for i in range(NI)) * startup_time_sigma[k] <= capacity_C[k], name=f'capacity_{k}_{t}')
            model.addConstr(gp.quicksum(setup[i, k, t] for i in range(NI)) <= 1, name=f'setup_{k}_{t}')
            model.addConstr(is_setup[k, t] == gp.quicksum(setup[i, k, t] for i in range(NI)), name=f'is_setup_{k}_{t}')

    for i in range(NI):
        for k in range(NK):
            for t in range(NT):
                model.addConstr(production[i, k, t] <= capacity_C[k] * setup[i, k, t], name=f'prod_setup_{i}_{k}_{t}')
                if t > 0:
                    model.addConstr(startup[i, k, t] >= setup[i, k, t] - setup[i, k, t-1], name=f'startup_{i}_{k}_{t}')

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if logger and model.objVal < float('inf'):
            logger.log(model.objVal)

        production_dict = {(f'{i}_{k}_{t}'): production[i, k, t].x for i in range(NI) for k in range(NK) for t in range(NT)}
        stock_dict = {(f'{i}_{t}'): stock[i, t].x for i in range(NI) for t in range(NT)}
        backlog_dict = {(f'{i}_{t}'): backlog[i, t].x for i in range(NI) for t in range(NT)}
        setup_dict = {(f'{i}_{k}_{t}'): setup[i, k, t].x for i in range(NI) for k in range(NK) for t in range(NT)}
        startup_dict = {(f'{i}_{k}_{t}'): startup[i, k, t].x for i in range(NI) for k in range(NK) for t in range(NT)}

        solution = {
            'objective_value': model.objVal,
            'production': production_dict,
            'stock': stock_dict,
            'backlog': backlog_dict,
            'setup': setup_dict,
            'startup': startup_dict
        }
        return solution
    else:
        return None

def write_solution(solution_path, solution):
    with open(solution_path, 'w') as file:
        json.dump(solution, file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve_instance(instance, args.time_limit, logger)
    if solution is not None:
        write_solution(args.solution_path, solution)

if __name__ == "__main__":
    main()