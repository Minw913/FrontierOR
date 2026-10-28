import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Quadratic Knapsack Problem Solver")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the problem instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument('--time_limit', type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument('--log_path', type=str, default="", help="Path for logging intermediate solutions.")
    args = parser.parse_args()

    if SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    else:
        logger = None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n = data['n']
    capacity = data['capacity']
    weights = data['weights']
    profit_matrix = data['profit_matrix']

    # -------------------------------------------------------------
    # Initialization: Fast Greedy O(N^2) Approach for an initial solution
    # -------------------------------------------------------------
    greedy_sol = [0] * n
    unselected = set(range(n))
    pi = [profit_matrix[i][i] for i in range(n)]
    rem_cap = capacity
    
    while unselected:
        best_item = -1
        best_score = (-1.0, -1.0)
        
        for i in unselected:
            if weights[i] <= rem_cap:
                if weights[i] == 0:
                    score = (float('inf'), pi[i])
                else:
                    score = (pi[i] / weights[i], pi[i])
                
                if score > best_score:
                    best_score = score
                    best_item = i
                    
        if best_item != -1:
            greedy_sol[best_item] = 1
            rem_cap -= weights[best_item]
            unselected.remove(best_item)
            
            # Update marginal profit (pi) for remaining unselected items
            for k in unselected:
                pi[k] += profit_matrix[k][best_item]
        else:
            break

    # Calculate objective of the greedy solution
    greedy_obj = 0.0
    for i in range(n):
        if greedy_sol[i] == 1:
            greedy_obj += profit_matrix[i][i]
            for j in range(i + 1, n):
                if greedy_sol[j] == 1:
                    greedy_obj += profit_matrix[i][j]

    if logger:
        logger.log(greedy_obj)

    best_obj = greedy_obj

    # -------------------------------------------------------------
    # Optimal Solver: MIP Formulation via Gurobi
    # -------------------------------------------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)  # Silence standard console output
    env.start()
    
    model = gp.Model("QKP", env=env)
    
    # Configure limits
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)

    # Decision variables: x[i] is 1 if item i is selected, 0 otherwise
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Construct Objective Function
    q_coeffs = []
    q_vars1 = []
    q_vars2 = []
    l_coeffs = []
    l_vars = []

    for i in range(n):
        if profit_matrix[i][i] != 0:
            l_coeffs.append(profit_matrix[i][i])
            l_vars.append(x[i])
        for j in range(i + 1, n):
            # Only count upper diagonal off-diagonal elements for pairwise interactions
            if profit_matrix[i][j] != 0:
                q_coeffs.append(profit_matrix[i][j])
                q_vars1.append(x[i])
                q_vars2.append(x[j])
                
    l_expr = gp.LinExpr()
    if l_coeffs:
        l_expr.addTerms(l_coeffs, l_vars)
        
    obj_expr = gp.QuadExpr(l_expr)
    if q_coeffs:
        obj_expr.addTerms(q_coeffs, q_vars1, q_vars2)

    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Constraint: sum of weights of selected items must not exceed the capacity
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "Capacity")

    # Provide initial MIP start with greedy solution
    for i in range(n):
        x[i].Start = greedy_sol[i]

    # Callback to track and log optimal intermediate improvements
    def callback(cb_model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj > best_obj + 1e-5:
                best_obj = obj
                if logger:
                    logger.log(obj)

    # Solve the model properly
    model.optimize(callback)

    # -------------------------------------------------------------
    # Output Solution Delivery
    # -------------------------------------------------------------
    if model.SolCount > 0:
        sol = [int(round(x[i].X)) for i in range(n)]
        obj_val = model.ObjVal
    else:
        # Fallback to greedy if no viable solution is reported from solver iteration
        sol = greedy_sol
        obj_val = greedy_obj

    solution_data = {
        "objective_value": float(obj_val),
        "solution": sol
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution_data, f, indent=2)

if __name__ == '__main__':
    main()