import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_machines = instance['num_machines']
    num_jobs = instance['num_jobs']
    cost_matrix = instance['cost_matrix']
    resource_matrix = instance['resource_matrix']
    capacities = instance['capacities']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model("AssignmentWithCapacity")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('LogToConsole', 0)
    model.setParam('Threads', 1)

    x = model.addVars(num_machines, num_jobs, vtype=GRB.BINARY, name="x")

    model.addConstrs((gp.quicksum(x[i, j] for i in range(num_machines)) == 1
                      for j in range(num_jobs)), name="assign_job")

    model.addConstrs((gp.quicksum(resource_matrix[i][j] * x[i, j]
                                  for j in range(num_jobs)) <= capacities[i]
                      for i in range(num_machines)), name="capacity")

    model.setObjective(gp.quicksum(cost_matrix[i][j] * x[i, j]
                                   for i in range(num_machines)
                                   for j in range(num_jobs)), GRB.MINIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    start_time = time.time()
    model.optimize(callback)

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        objective_value = model.objVal
        assignments = {}
        for j in range(num_jobs):
            for i in range(num_machines):
                if abs(x[i, j].X) > 0.5:
                    assignments[str(j)] = i
                    break

        solution = {
            "objective_value": float(objective_value),
            "assignments": assignments
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # If infeasible or other error, write minimal feasible if possible or empty
        solution = {
            "objective_value": 0.0,
            "assignments": {str(j): 0 for j in range(num_jobs)}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()