import argparse
import json
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
        data = json.load(f)
    
    n = data['n']
    m = data['m']
    capacity = data['capacity']
    weights = data['weights']
    values = data['values']
    pairwise_values = data['pairwise_values']
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('NonConvex', 2)
    model.setParam('TimeLimit', args.time_limit)
    
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    
    linear_obj = gp.quicksum(values[i] * x[i, k] for i in range(n) for k in range(m))
    quadratic_obj = gp.quicksum(
        pairwise_values[i][j] * x[i, k] * x[j, k] 
        for k in range(m) 
        for i in range(n) 
        for j in range(i + 1, n)
    )
    model.setObjective(linear_obj + quadratic_obj, GRB.MAXIMIZE)
    
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1, name=f"item_{i}_assignment")
    
    for k in range(m):
        model.addConstr(gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacity, name=f"capacity_{k}")
    
    if logger:
        class CallbackWrapper:
            def __init__(self, logger):
                self.logger = logger
            def callback(self, model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    self.logger.log(obj)
        cb_wrapper = CallbackWrapper(logger)
        model.optimize(cb_wrapper.callback)
    else:
        model.optimize()
    
    assignment = {}
    if model.SolCount > 0:
        objective_value = model.objVal
        for i in range(n):
            for k in range(m):
                if x[i, k].X > 0.5:
                    assignment[str(i)] = k
    else:
        objective_value = 0.0
    
    solution = {
        "objective_value": float(objective_value),
        "assignment": assignment
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()