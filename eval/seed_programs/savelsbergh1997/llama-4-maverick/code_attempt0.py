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
    num_agents = instance['num_agents']
    num_jobs = instance['num_jobs']
    profits = instance['profits']
    weights = instance['weights']
    capacities = instance['capacities']

    model = gp.Model()
    model.Params.LogToConsole = 0  # Silence Gurobi output
    model.Params.TimeLimit = time_limit  # Set time limit

    # Decision variables: x[a, j] = 1 if job j is assigned to agent a
    x = model.addVars(num_agents, num_jobs, vtype=GRB.BINARY, name='x')

    # Objective: Maximize total profit
    model.setObjective(gp.quicksum(profits[a][j] * x[a, j] for a in range(num_agents) for j in range(num_jobs)), GRB.MAXIMIZE)

    # Constraints: Each job is assigned to exactly one agent
    model.addConstrs((gp.quicksum(x[a, j] for a in range(num_agents)) == 1 for j in range(num_jobs)), name='assign_job')

    # Constraints: Total weight of jobs assigned to an agent does not exceed its capacity
    model.addConstrs((gp.quicksum(weights[a][j] * x[a, j] for j in range(num_jobs)) <= capacities[a] for a in range(num_agents)), name='agent_capacity')

    model.optimize()

    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        assignment = {str(j): None for j in range(num_jobs)}
        objective_value = model.getObjective().getValue()
        for j in range(num_jobs):
            for a in range(num_agents):
                if x[a, j].x > 0.5:  # Considering binary variables, this should be either 0 or 1
                    assignment[str(j)] = a
        return objective_value, assignment
    else:
        raise Exception(f"Gurobi model failed to find a feasible solution. Status: {model.status}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    instance = read_instance(args.instance_path)

    # Create a Gurobi model and set up a callback for logging intermediate solutions
    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            objective_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(objective_value)

    model = gp.Model()
    model.Params.LogToConsole = 0  # Silence Gurobi output
    model.Params.TimeLimit = args.time_limit  # Set time limit

    num_agents = instance['num_agents']
    num_jobs = instance['num_jobs']
    profits = instance['profits']
    weights = instance['weights']
    capacities = instance['capacities']

    # Decision variables: x[a, j] = 1 if job j is assigned to agent a
    x = model.addVars(num_agents, num_jobs, vtype=GRB.BINARY, name='x')

    # Objective: Maximize total profit
    model.setObjective(gp.quicksum(profits[a][j] * x[a, j] for a in range(num_agents) for j in range(num_jobs)), GRB.MAXIMIZE)

    # Constraints: Each job is assigned to exactly one agent
    model.addConstrs((gp.quicksum(x[a, j] for a in range(num_agents)) == 1 for j in range(num_jobs)), name='assign_job')

    # Constraints: Total weight of jobs assigned to an agent does not exceed its capacity
    model.addConstrs((gp.quicksum(weights[a][j] * x[a, j] for j in range(num_jobs)) <= capacities[a] for a in range(num_agents)), name='agent_capacity')

    model.optimize(log_callback)

    # Extract solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        assignment = {str(j): None for j in range(num_jobs)}
        objective_value = model.getObjective().getValue()
        for j in range(num_jobs):
            for a in range(num_agents):
                if x[a, j].x > 0.5:  # Considering binary variables, this should be either 0 or 1
                    assignment[str(j)] = a
        solution = {
            'objective_value': objective_value,
            'assignment': assignment
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        raise Exception(f"Gurobi model failed to find a feasible solution. Status: {model.status}")

if __name__ == "__main__":
    main()