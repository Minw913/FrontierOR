import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="Solve Knapsack with Pairwise Values")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to input JSON instance")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to output JSON solution")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument('--log_path', type=str, required=False, help="Path for logging intermediate solutions")
    args = parser.parse_args()

    # Read the JSON instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    m = data['m']
    capacity = data['capacity']
    weights = data['weights']
    values = data['values']
    P = data['pairwise_values']

    # Initialize logger if the path is provided
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    except ImportError:
        logger = None

    # Setup Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    model = gp.Model("Knapsack_with_Pairs", env=env)
    
    if args.time_limit > 0:
        model.setParam('TimeLimit', args.time_limit)
    
    # Restrict to a single CPU core
    model.setParam('Threads', 1)

    # Variables: x[i,k] = 1 if item i is placed in knapsack k
    x = {}
    for i in range(n):
        for k in range(m):
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{k}")

    # Variables: y[i,j,k] = 1 if both item i and j are placed in knapsack k
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            pval = P[i][j]
            if pval != 0:
                for k in range(m):
                    y[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}")

    # Objective Function
    obj = gp.quicksum(values[i] * x[i, k] for i in range(n) for k in range(m)) + \
          gp.quicksum(P[i][j] * y[i, j, k] 
                      for i in range(n) 
                      for j in range(i + 1, n) if P[i][j] != 0 
                      for k in range(m))
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraint 1: Each item is placed in at most one knapsack
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1, name=f"Assign_{i}")

    # Constraint 2: Total weight per knapsack cannot exceed capacity
    for k in range(m):
        model.addConstr(gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacity, name=f"Cap_{k}")

    # Constraint 3: Linearize the bilinear mapping of items paired
    for i in range(n):
        for j in range(i + 1, n):
            pval = P[i][j]
            if pval > 0:
                for k in range(m):
                    model.addConstr(y[i, j, k] <= x[i, k])
                    model.addConstr(y[i, j, k] <= x[j, k])
            elif pval < 0:
                for k in range(m):
                    model.addConstr(y[i, j, k] >= x[i, k] + x[j, k] - 1)

    # Symmetry breaking constraints: orders total weights of items down the identical knapsacks
    for k in range(m - 1):
        model.addConstr(
            gp.quicksum(weights[i] * x[i, k] for i in range(n)) >= 
            gp.quicksum(weights[i] * x[i, k + 1] for i in range(n)),
            name=f"SymBreak_{k}"
        )
    # Give Gurobi permission to apply aggressive symmetrical reductions
    model.setParam('Symmetry', 2)

    # Callback logic to invoke the logger directly at incumbent upgrades 
    class CallbackData:
        def __init__(self, solution_logger):
            self.logger = solution_logger
            self.best_obj = -float('inf')

        def __call__(self, model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj > self.best_obj + 1e-5:
                    self.best_obj = obj
                    if self.logger:
                        self.logger.log(obj)

    cb_data = CallbackData(logger)
    
    # Solve the model 
    model.optimize(cb_data)

    # Evaluate results into strictly formatted outputs
    assignment = {}
    best_obj = 0.0
    if model.SolCount > 0:
        best_obj = model.ObjVal
        for i in range(n):
            for k in range(m):
                if x[i, k].X > 0.5:
                    # Key matches schema pattern strictly
                    assignment[str(i)] = k
                    break

    solution = {
        "objective_value": float(best_obj),
        "assignment": assignment
    }

    # Write output sequence to file
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()