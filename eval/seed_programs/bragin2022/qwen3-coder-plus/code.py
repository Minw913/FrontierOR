import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
def solve_assignment_problem(instance_data, time_limit, log_path):
    num_machines = instance_data["num_machines"]
    num_jobs = instance_data["num_jobs"]
    cost_matrix = instance_data["cost_matrix"]  # [machine][job]
    resource_matrix = instance_data["resource_matrix"]  # [machine][job]
    capacities = instance_data["capacities"]  # [machine]

    # Create model
    model = gp.Model("assignment_problem")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)

    # Suppress console output
    model.setParam(GRB.Param.LogToConsole, 0)

    # Decision variables: x[machine][job] = 1 if job is assigned to machine
    x = {}
    for m in range(num_machines):
        for j in range(num_jobs):
            x[m, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{m}_{j}")

    # Objective function: minimize total cost
    obj_expr = gp.LinExpr()
    for m in range(num_machines):
        for j in range(num_jobs):
            obj_expr += cost_matrix[m][j] * x[m, j]
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Constraint 1: Each job must be assigned to exactly one machine
    for j in range(num_jobs):
        model.addConstr(gp.quicksum(x[m, j] for m in range(num_machines)) == 1, f"job_assigned_{j}")

    # Constraint 2: Resource capacity constraints for each machine
    for m in range(num_machines):
        model.addConstr(
            gp.quicksum(resource_matrix[m][j] * x[m, j] for j in range(num_jobs)) <= capacities[m],
            f"capacity_{m}"
        )

    # Set time limit
    model.setParam(GRB.Param.TimeLimit, time_limit)

    # Create logger if log_path is provided
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    # Capture best solutions during the search
    def log_solution(model, where):
        if where == GRB.Callback.MIPSol:
            obj_val = model.cbGet(GRB.Callback.MIPSolObj)
            if logger:
                logger.log(obj_val)

    # Optimize the model
    model.optimize(log_solution)

    # Get solution
    if model.SolCount > 0:
        assignments = {}
        for j in range(num_jobs):
            for m in range(num_machines):
                if x[m, j].X > 0.5:  # Binary variable assigned to 1
                    assignments[str(j)] = m
                    break
        
        objective_value = model.ObjVal
    else:
        # Return empty solution if no feasible solution was found
        assignments = {}
        for j in range(num_jobs):
            assignments[str(j)] = 0  # Default to first machine if no solution found
        objective_value = float('inf')

    return {
        "objective_value": objective_value,
        "assignments": assignments
    }


def main():
    parser = argparse.ArgumentParser(description='Solve Job Assignment Problem')
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the problem instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the final solution JSON file must be written")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to a JSONL file for logging intermediate solutions")

    args = parser.parse_args()

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve the problem
    solution = solve_assignment_problem(instance_data, args.time_limit, args.log_path)

    # Write solution to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()