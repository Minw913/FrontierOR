import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(instance, time_limit, log_path):
    num_machines = instance["num_machines"]
    num_jobs = instance["num_jobs"]
    cost_matrix = instance["cost_matrix"]
    resource_matrix = instance["resource_matrix"]
    capacities = instance["capacities"]

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    model = gp.Model("GAP")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)

    x = model.addVars(num_machines, num_jobs, vtype=GRB.BINARY, name="x")
    
    # Objective: Minimize total cost
    model.setObjective(
        gp.quicksum(cost_matrix[m][j] * x[m, j] 
                    for m in range(num_machines) 
                    for j in range(num_jobs)), 
        GRB.MINIMIZE
    )

    # Constraint: Each job must be assigned to exactly one machine
    for j in range(num_jobs):
        model.addConstr(
            gp.quicksum(x[m, j] for m in range(num_machines)) == 1, 
            name=f"assign_job_{j}"
        )

    # Constraint: Machine capacity
    for m in range(num_machines):
        model.addConstr(
            gp.quicksum(resource_matrix[m][j] * x[m, j] for j in range(num_jobs)) <= capacities[m], 
            name=f"cap_machine_{m}"
        )

    model._vars = [x[m, j] for m in range(num_machines) for j in range(num_jobs)]
    model._num_machines = num_machines
    model._num_jobs = num_jobs
    model._logger = logger

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            vals = model.cbGetSolution(model._vars)
            assignments = {}
            for j in range(model._num_jobs):
                for m in range(model._num_machines):
                    if vals[m * model._num_jobs + j] > 0.5:
                        assignments[str(j)] = m
                        break
            if model._logger:
                model._logger.log_solution(float(obj), {
                    "objective_value": float(obj),
                    "assignments": assignments
                })

    model.optimize(callback)

    if model.SolCount > 0:
        obj_val = model.ObjVal
        assignments = {}
        for j in range(num_jobs):
            for m in range(num_machines):
                if x[m, j].X > 0.5:
                    assignments[str(j)] = m
                    break
        return {
            "objective_value": float(obj_val),
            "assignments": assignments
        }
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, type=str)
    parser.add_argument("--solution_path", required=True, type=str)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    solution = solve(instance, args.time_limit, args.log_path)
    
    if solution is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        with open(args.solution_path, 'w') as f:
            json.dump({}, f, indent=2)