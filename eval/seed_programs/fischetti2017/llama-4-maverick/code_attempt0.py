import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve_qufl(instance, time_limit, logger):
    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    opening_costs = instance['opening_costs']
    allocation_costs = instance['allocation_costs']

    model = gp.Model('qUFL')
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit

    y = model.addVars(num_facilities, vtype=GRB.BINARY, name='y')
    x = model.addVars(num_facilities, num_customers, vtype=GRB.CONTINUOUS, name='x')

    model.setObjective(
        gp.quicksum(opening_costs[i] * y[i] for i in range(num_facilities)) +
        gp.quicksum(allocation_costs[i][j] * x[i, j] * x[i, j] for i in range(num_facilities) for j in range(num_customers)),
        GRB.MINIMIZE
    )

    model.addConstrs((gp.quicksum(x[i, j] for i in range(num_facilities)) == 1 for j in range(num_customers)), name='demand_coverage')
    model.addConstrs((x[i, j] <= y[i] for i in range(num_facilities) for j in range(num_customers)), name='facility_open')

    model.optimize()

    if model.SolCount > 0:
        objective_value = model.ObjVal
        y_sol = [int(y[i].X) for i in range(num_facilities)]
        x_sol = [[x[i, j].X for j in range(num_customers)] for i in range(num_facilities)]

        solution = {
            'objective_value': objective_value,
            'variant': 'quadratic',
            'y': y_sol,
            'x': x_sol
        }

        if logger:
            logger.log_solution(objective_value, solution)

        return solution
    else:
        raise Exception('No solution found')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense='minimize') if args.log_path else None

    instance = read_instance(args.instance_path)
    solution = solve_qufl(instance, args.time_limit, logger)

    with open(args.solution_path, 'w') as file:
        json.dump(solution, file, indent=4)

if __name__ == '__main__':
    main()