import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance['n'], instance['p'], instance['cost_matrix']

def solve_p_median(n, p, cost_matrix, time_limit, log_path=None):
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    model = gp.Model("p_median")
    model.Params.TimeLimit = time_limit
    model.Params.LogToConsole = 0
    
    # Variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    
    # Objective
    model.setObjective(gp.quicksum(cost_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
    
    # Constraints
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n)) == 1 for i in range(n)), name="assign")
    model.addConstrs((x[i, j] <= y[j] for i in range(n) for j in range(n)), name="serve")
    model.addConstr(gp.quicksum(y[j] for j in range(n)) == p, name="p_medians")
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.getObjective().getValue()
        opened_facilities = [j for j in range(n) if y[j].x > 0.5]
        assignments = [{'customer': i, 'facility': next(j for j in range(n) if x[i, j].x > 0.5), 'cost': cost_matrix[i][next(j for j in range(n) if x[i, j].x > 0.5)]} for i in range(n)]
        
        if logger:
            logger.log(objective_value)
        
        return {
            'objective_value': int(objective_value),
            'opened_facilities': opened_facilities,
            'assignments': assignments
        }
    else:
        raise RuntimeError("Failed to solve the problem")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()
    
    n, p, cost_matrix = read_instance(args.instance_path)
    solution = solve_p_median(n, p, cost_matrix, args.time_limit, args.log_path)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()