import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as file:
        instance = json.load(file)
    return instance

def solve_instance(instance, time_limit, log_path):
    n = instance['n']
    A = instance['A']
    B = instance['B']
    C = instance['C']

    model = gp.Model()
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = False  # Optional: disable Gurobi console output

    x = model.addVars(n, n, vtype=GRB.BINARY, name='x')

    # Constraints for one-to-one assignment
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f'entity_{i}')
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1, name=f'position_{j}')

    # Aggregate costs
    aggregate_A = gp.quicksum(A[i][j] * x[i, j] for i in range(n) for j in range(n))
    aggregate_B = gp.quicksum(B[i][j] * x[i, j] for i in range(n) for j in range(n))
    aggregate_C = gp.quicksum(C[i][j] * x[i, j] for i in range(n) for j in range(n))

    # Objective function
    objective = aggregate_A * aggregate_B + aggregate_C
    model.setObjective(objective, GRB.MINIMIZE)

    # Initialize logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Callback to log intermediate solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_value)

    model.optimize(callback=callback)

    # Extract and return solution
    if model.SolCount > 0:
        assignment = [int(gp.quicksum(j * x[i, j] for j in range(n)).getValue()) for i in range(n)]
        objective_value = model.ObjVal
        return objective_value, assignment
    else:
        raise Exception("No solution found")

def write_solution(solution_path, objective_value, assignment):
    solution = {
        'objective_value': objective_value,
        'assignment': assignment
    }
    with open(solution_path, 'w') as file:
        json.dump(solution, file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    objective_value, assignment = solve_instance(instance, args.time_limit, args.log_path)
    write_solution(args.solution_path, objective_value, assignment)

if __name__ == '__main__':
    main()