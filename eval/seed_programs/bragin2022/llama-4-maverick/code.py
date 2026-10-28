import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, logger):
    num_machines = instance['num_machines']
    num_jobs = instance['num_jobs']
    cost_matrix = instance['cost_matrix']
    resource_matrix = instance['resource_matrix']
    capacities = instance['capacities']

    model = gp.Model()
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.params.LogToConsole = 0  # Silence Gurobi output
    model.params.TimeLimit = time_limit  # Set time limit

    # Decision variables: x[m, j] = 1 if job j is assigned to machine m
    x = model.addVars(num_machines, num_jobs, vtype=GRB.BINARY, name='x')

    # Objective function: minimize total assignment cost
    model.setObjective(gp.quicksum(cost_matrix[m][j] * x[m, j] for m in range(num_machines) for j in range(num_jobs)), GRB.MINIMIZE)

    # Constraints: each job is assigned to exactly one machine
    for j in range(num_jobs):
        model.addConstr(gp.quicksum(x[m, j] for m in range(num_machines)) == 1, name=f'job_{j}')

    # Constraints: total resource consumption per machine does not exceed capacity
    for m in range(num_machines):
        model.addConstr(gp.quicksum(resource_matrix[m][j] * x[m, j] for j in range(num_jobs)) <= capacities[m], name=f'capacity_{m}')

    # Optimize the model
    model.optimize()

    # Extract and return the solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        assignments = {}
        for j in range(num_jobs):
            for m in range(num_machines):
                if x[m, j].x > 0.5:  # x[m, j] is 1 if job j is assigned to machine m
                    assignments[str(j)] = m
                    break
        return objective_value, assignments
    else:
        raise Exception(f"Gurobi failed to find a solution. Status code: {model.status}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # We need to define a callback function to log intermediate solutions
    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_value)

    # Solve the instance with the callback
    model = gp.Model()
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    num_machines = instance['num_machines']
    num_jobs = instance['num_jobs']
    cost_matrix = instance['cost_matrix']
    resource_matrix = instance['resource_matrix']
    capacities = instance['capacities']

    model.params.LogToConsole = 0  # Silence Gurobi output
    model.params.TimeLimit = args.time_limit  # Set time limit

    # Decision variables: x[m, j] = 1 if job j is assigned to machine m
    x = model.addVars(num_machines, num_jobs, vtype=GRB.BINARY, name='x')

    # Objective function: minimize total assignment cost
    model.setObjective(gp.quicksum(cost_matrix[m][j] * x[m, j] for m in range(num_machines) for j in range(num_jobs)), GRB.MINIMIZE)

    # Constraints: each job is assigned to exactly one machine
    for j in range(num_jobs):
        model.addConstr(gp.quicksum(x[m, j] for m in range(num_machines)) == 1, name=f'job_{j}')

    # Constraints: total resource consumption per machine does not exceed capacity
    for m in range(num_machines):
        model.addConstr(gp.quicksum(resource_matrix[m][j] * x[m, j] for j in range(num_jobs)) <= capacities[m], name=f'capacity_{m}')

    # Optimize the model with callback
    model.optimize(callback=log_callback)

    # Extract and return the solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        assignments = {}
        for j in range(num_jobs):
            for m in range(num_machines):
                if x[m, j].x > 0.5:  # x[m, j] is 1 if job j is assigned to machine m
                    assignments[str(j)] = m
                    break
        solution = {
            'objective_value': objective_value,
            'assignments': assignments
        }
    else:
        raise Exception(f"Gurobi failed to find a solution. Status code: {model.status}")

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()