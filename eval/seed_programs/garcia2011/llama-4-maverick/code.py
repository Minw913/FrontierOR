import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    n = instance['n']
    p = instance['p']
    cost_matrix = instance['cost_matrix']
    return n, p, cost_matrix

def solve_p_median(n, p, cost_matrix, time_limit, log_path=None):
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    model = gp.Model("p_median")
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0
    
    # Variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    
    # Objective function
    model.setObjective(gp.quicksum(cost_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
    
    # Constraints
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n)) == 1 for i in range(n)), name="assign")
    model.addConstrs((x[i, j] <= y[j] for i in range(n) for j in range(n)), name="open")
    model.addConstr(gp.quicksum(y[j] for j in range(n)) == p, name="p_median")
    
    # Optimize
    model.Params.TimeLimit = time_limit
    model.optimize()
    
    # Retrieve solution
    try:
        objective_value = int(model.getObjective().getValue())
        opened_facilities = [j for j in range(n) if y[j].x > 0.5]
        assignments = [{'customer': i, 'facility': j, 'cost': cost_matrix[i][j]} for i in range(n) for j in range(n) if x[i, j].x > 0.5]
        
        if logger:
            logger.log(objective_value)
        
        return {
            'objective_value': objective_value,
            'opened_facilities': opened_facilities,
            'assignments': assignments
        }
    except gp.GurobiError:
        # Handle case where no feasible solution is found
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()
    
    try:
        n, p, cost_matrix = read_instance(args.instance_path)
        solution = solve_p_median(n, p, cost_matrix, args.time_limit, args.log_path)
        
        if solution is not None:
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=4)
        else:
            print("No feasible solution found within the time limit.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()