import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, logger):
    n_clients = instance['n_clients']
    n_servers = instance['n_servers']
    demands = instance['demands']
    processing_rates = instance['processing_rates']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance['quadratic_costs']

    model = Model()
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = time_limit

    # Variables
    x = model.addVars(n_clients, n_servers, vtype=GRB.BINARY, name='x')

    # Constraints: each client is assigned to exactly one server
    model.addConstrs((quicksum(x[c, s] for s in range(n_servers)) == 1 for c in range(n_clients)), name='assign_client')

    # Objective: minimize total cost (linear + quadratic)
    linear_cost = quicksum(linear_costs[c][s] * x[c, s] for c in range(n_clients) for s in range(n_servers))
    quadratic_cost = quicksum(cost * x[c1, s1] * x[c2, s2] for c1, s1, c2, s2, cost in quadratic_costs)
    model.setObjective(linear_cost + quadratic_cost, GRB.MINIMIZE)

    # Optimize
    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        assignment = {f'{c}': int(x[c, s].X) for c in range(n_clients) for s in range(n_servers) if x[c, s].X > 0.5}
        objective_value = model.ObjVal
        if logger:
            logger.log(objective_value)
        return objective_value, assignment
    else:
        raise Exception(f"Model status: {model.status}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    try:
        objective_value, assignment = solve(instance, args.time_limit, logger)
        solution = {
            'objective_value': objective_value,
            'assignment': assignment
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()