import json
import argparse
from gurobipy import Model, GRB, quicksum
import time
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve(instance, time_limit, log_path=None):
    N = instance['N']
    m = instance['m']
    group_capacities = instance['group_capacities']
    distances_upper_triangular = instance['distances_upper_triangular']

    # Initialize the logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    # Create the model
    model = Model()
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0  # Silence Gurobi output
    model.Params.TimeLimit = time_limit  # Set time limit

    # Variables: x[i, k] = 1 if element i is assigned to group k, 0 otherwise
    x = model.addVars(N, m, vtype=GRB.BINARY, name='x')

    # Objective: Maximize total diversity
    objective = quicksum(
        distances_upper_triangular[i * (N - 1) // 2 + j - i - 1] * x[i, k] * x[j, k]
        for k in range(m)
        for i in range(N)
        for j in range(i + 1, N)
    )
    model.setObjective(objective, GRB.MAXIMIZE)

    # Constraints: Each element is assigned to exactly one group
    for i in range(N):
        model.addConstr(quicksum(x[i, k] for k in range(m)) == 1, f'assign_{i}')

    # Constraints: Group capacity limits
    for k in range(m):
        L_g = group_capacities[k]['L_g']
        U_g = group_capacities[k]['U_g']
        model.addConstr(quicksum(x[i, k] for i in range(N)) >= L_g, f'lower_{k}')
        model.addConstr(quicksum(x[i, k] for i in range(N)) <= U_g, f'upper_{k}')

    # Linearize the objective
    y = model.addVars(N, N, m, vtype=GRB.BINARY, name='y')
    for k in range(m):
        for i in range(N):
            for j in range(i + 1, N):
                model.addConstr(y[i, j, k] <= x[i, k], f'link1_{i}_{j}_{k}')
                model.addConstr(y[i, j, k] <= x[j, k], f'link2_{i}_{j}_{k}')
                model.addConstr(y[i, j, k] >= x[i, k] + x[j, k] - 1, f'link3_{i}_{j}_{k}')
    objective_linearized = quicksum(
        distances_upper_triangular[i * (N - 1) // 2 + j - i - 1] * y[i, j, k]
        for k in range(m)
        for i in range(N)
        for j in range(i + 1, N)
    )
    # model.setObjective(objective_linearized, GRB.MAXIMIZE)  # Uncommenting this line will cause issues because it overwrites the original objective.

    # Optimize
    model.optimize()

    # Extract solution
    assignment = {}
    groups = {k: [] for k in range(m)}
    if model.SolCount > 0:
        for i in range(N):
            for k in range(m):
                if x[i, k].X > 0.5:
                    assignment[i] = k
                    groups[k].append(i)
                    break
        for k in range(m):
            groups[k].sort()
        objective_value = model.ObjVal
        if logger:
            logger.log(objective_value)
    else:
        objective_value = 0

    return {
        'objective_value': objective_value,
        'assignment': assignment,
        'groups': groups
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    try:
        instance = read_instance(args.instance_path)
    except Exception as e:
        print(f"Error reading instance: {e}")
        return

    solution = solve(instance, args.time_limit, args.log_path)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    main()