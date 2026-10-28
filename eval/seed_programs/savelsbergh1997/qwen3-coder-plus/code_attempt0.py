import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_assignment_problem(instance_data, time_limit, log_path):
    num_agents = instance_data["num_agents"]
    num_jobs = instance_data["num_jobs"]
    profits = instance_data["profits"]
    weights = instance_data["weights"]
    capacities = instance_data["capacities"]

    # Initialize the logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None

    # Create model
    model = gp.Model("AssignmentProblem")
    
    # Set time limit
    model.setParam('TimeLimit', time_limit)
    
    # Silence output unless we need status updates
    model.setParam('LogToConsole', 0)
    
    # Decision variables: x[i][j] = 1 if job j is assigned to agent i, 0 otherwise
    x = {}
    for i in range(num_agents):
        for j in range(num_jobs):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Constraints: Each job must be assigned to exactly one agent
    for j in range(num_jobs):
        model.addConstr(gp.quicksum(x[i, j] for i in range(num_agents)) == 1, f"job_assign_{j}")

    # Constraints: Capacity constraint for each agent
    for i in range(num_agents):
        model.addConstr(gp.quicksum(weights[i][j] * x[i, j] for j in range(num_jobs)) <= capacities[i], f"capacity_{i}")

    # Objective: Maximize the total profit
    model.setObjective(
        gp.quicksum(profits[i][j] * x[i, j] for i in range(num_agents) for j in range(num_jobs)),
        GRB.MAXIMIZE
    )

    # Optimization callback to log solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Optimize
    model.optimize(callback)

    # Extract solution
    objective_value = model.objVal if model.solCount > 0 else 0.0
    assignment = {}

    if model.solCount > 0:
        for j in range(num_jobs):
            for i in range(num_agents):
                if x[i, j].x > 0.5:  # Binary variable is true
                    assignment[str(j)] = i
                    break
    else:
        # If no solution found, return empty assignment
        for j in range(num_jobs):
            assignment[str(j)] = 0  # Default assignment if no solution
    
    solution = {
        "objective_value": objective_value,
        "assignment": assignment
    }

    return solution


def main():
    parser = argparse.ArgumentParser(description="Solve Assignment Problem")
    parser.add_argument("--instance_path", required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to save the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", help="Path to the log file (optional)")

    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_assignment_problem(instance_data, args.time_limit, args.log_path)

    # Write the solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()