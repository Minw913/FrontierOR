import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit):
    num_facilities = instance['num_facilities']
    num_customers = instance['num_customers']
    demands = instance['demands']
    capacities = instance['capacities']
    fixed_costs_obj1 = instance['fixed_costs_obj1']
    fixed_costs_obj2 = instance['fixed_costs_obj2']
    assignment_costs_obj1 = instance['assignment_costs_obj1']
    assignment_costs_obj2 = instance['assignment_costs_obj2']

    model = Model()
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.PoolSolutions = 1000000
    model.Params.PoolGap = 0.1

    # Variables
    open_facilities = model.addVars(num_facilities, vtype=GRB.BINARY, name='open_facilities')
    assignments = model.addVars(num_facilities, num_customers, vtype=GRB.BINARY, name='assignments')

    # Constraints
    model.addConstrs((quicksum(assignments[i, j] for i in range(num_facilities)) == 1 for j in range(num_customers)), name='assign_customers')
    model.addConstrs((quicksum(demands[j] * assignments[i, j] for j in range(num_customers)) <= capacities[i] * open_facilities[i] for i in range(num_facilities)), name='capacity_constraints')
    model.addConstrs((assignments[i, j] <= open_facilities[i] for i in range(num_facilities) for j in range(num_customers)), name='link_open_assign')

    # Objectives
    obj1 = quicksum(fixed_costs_obj1[i] * open_facilities[i] for i in range(num_facilities)) + quicksum(assignment_costs_obj1[i][j] * assignments[i, j] for i in range(num_facilities) for j in range(num_customers))
    obj2 = quicksum(fixed_costs_obj2[i] * open_facilities[i] for i in range(num_facilities)) + quicksum(assignment_costs_obj2[i][j] * assignments[i, j] for i in range(num_facilities) for j in range(num_customers))

    model.setObjectiveN(obj1, index=0, priority=1)
    model.setObjectiveN(obj2, index=1, priority=1)

    model.optimize()

    pareto_front = []
    solutions = []
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        for sol in range(model.SolCount):
            model.Params.SolutionNumber = sol
            obj1_val = obj1.getValue()
            obj2_val = obj2.getValue()
            pareto_front.append([int(obj1_val), int(obj2_val)])

            open_facilities_sol = [i for i in range(num_facilities) if open_facilities[i].Xn > 0.5]
            assignments_sol = {j: i for i in range(num_facilities) for j in range(num_customers) if assignments[i, j].Xn > 0.5}

            solutions.append({
                'z1': int(obj1_val),
                'z2': int(obj2_val),
                'open_facilities': open_facilities_sol,
                'assignments': {str(j): int(i) for j, i in assignments_sol.items()}
            })

    return pareto_front, solutions

def calculate_objective_value(pareto_front, solutions):
    min_weighted_sum = float('inf')
    best_solution = None
    for solution in solutions:
        weighted_sum = (solution['z1'] + solution['z2']) / 2
        if weighted_sum < min_weighted_sum:
            min_weighted_sum = weighted_sum
            best_solution = solution
    return min_weighted_sum, best_solution

def write_solution(solution_path, objective_value, pareto_front, solutions):
    with open(solution_path, 'w') as f:
        json.dump({
            'objective_value': objective_value,
            'pareto_front': pareto_front,
            'solutions': solutions
        }, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    pareto_front, solutions = solve_instance(instance, args.time_limit)
    objective_value, _ = calculate_objective_value(pareto_front, solutions)

    if logger:
        logger.log(objective_value)

    write_solution(args.solution_path, objective_value, pareto_front, solutions)

if __name__ == '__main__':
    main()