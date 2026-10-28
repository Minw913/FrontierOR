import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def greedy_assignment(num_machines, num_jobs, cost_matrix, resource_matrix, capacities):
    remaining_capacities = capacities.copy()
    assignments = {}
    total_cost = 0

    max_resources = [max(resource_matrix[i][j] for i in range(num_machines)) for j in range(num_jobs)]
    job_order = sorted(range(num_jobs), key=lambda j: max_resources[j], reverse=True)

    for j in job_order:
        best_machine = None
        best_cost = float('inf')
        for i in range(num_machines):
            if resource_matrix[i][j] <= remaining_capacities[i]:
                if cost_matrix[i][j] < best_cost:
                    best_machine = i
                    best_cost = cost_matrix[i][j]
        if best_machine is None:
            best_machine = min(range(num_machines), key=lambda i: cost_matrix[i][j])
        assignments[str(j)] = best_machine
        total_cost += cost_matrix[best_machine][j]
        remaining_capacities[best_machine] -= resource_matrix[best_machine][j]
    return assignments, total_cost

def main():
    parser = argparse.ArgumentParser(description='Solve Generalized Assignment Problem')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_machines = data['num_machines']
    num_jobs = data['num_jobs']
    cost_matrix = data['cost_matrix']
    resource_matrix = data['resource_matrix']
    capacities = data['capacities']

    try:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        model = gp.Model("gap", env=env)
        x = {}
        for i in range(num_machines):
            for j in range(num_jobs):
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        for j in range(num_jobs):
            model.addConstr(gp.quicksum(x[i, j] for i in range(num_machines)) == 1, f"job_{j}")

        for i in range(num_machines):
            model.addConstr(gp.quicksum(resource_matrix[i][j] * x[i, j] for j in range(num_jobs)) <= capacities[i], f"cap_{i}")

        model.setObjective(
            gp.quicksum(cost_matrix[i][j] * x[i, j] for i in range(num_machines) for j in range(num_jobs)),
            GRB.MINIMIZE
        )
        model.update()
    except gp.GurobiError as e:
        print(f"Gurobi error during setup: {e}")
        assignments_greedy, total_cost_greedy = greedy_assignment(num_machines, num_jobs, cost_matrix, resource_matrix, capacities)
        with open(args.solution_path, 'w') as f:
            json.dump({"objective_value": total_cost_greedy, "assignments": assignments_greedy}, f, indent=2)
        return

    time_used = time.time() - start_time
    time_remaining = args.time_limit - time_used
    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, "minimize")

    if time_remaining > 0:
        model.Params.TimeLimit = time_remaining
        model.Params.Threads = 1

        def callback(model_, where):
            if where == GRB.Callback.MIPSOL:
                if logger is not None:
                    obj_val = model_.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj_val)
        
        model.optimize(callback if args.log_path else None)
    else:
        print("Not solving MIP because time is up.")

    if model.SolCount > 0:
        assignments = {}
        for j in range(num_jobs):
            for i in range(num_machines):
                if abs(x[i, j].X) > 0.5:
                    assignments[str(j)] = i
                    break
        total_cost = model.ObjVal
    else:
        assignments, total_cost = greedy_assignment(num_machines, num_jobs, cost_matrix, resource_matrix, capacities)

    with open(args.solution_path, 'w') as f:
        json.dump({"objective_value": total_cost, "assignments": assignments}, f, indent=2)

if __name__ == '__main__':
    main()