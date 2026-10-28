import argparse
import json
import time
import numpy as np
from itertools import permutations

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    p = data['p']
    cost_matrices = np.array(data['cost_matrices'])  # shape [p][n][n]
    reference_point = np.array(data['reference_point'])
    search_direction = np.array(data['search_direction_lambda'])
    
    start_time = time.time()
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    best_objective = float('inf')
    best_assignment = None
    
    def evaluate_assignment(assignment):
        """Evaluate an assignment (list where assignment[i] = agent for task i)"""
        total_costs = np.zeros(p)
        for i in range(n):
            j = assignment[i]
            for k in range(p):
                total_costs[k] += cost_matrices[k][i][j]
        weighted_devs = search_direction * (total_costs - reference_point)
        return float(np.max(weighted_devs))
    
    def update_best(assignment, obj_val=None):
        nonlocal best_objective, best_assignment
        if obj_val is None:
            obj_val = evaluate_assignment(assignment)
        if obj_val < best_objective:
            best_objective = obj_val
            best_assignment = list(assignment)
            if logger:
                logger.log(best_objective)
        return obj_val
    
    # For small n, try brute force
    if n <= 10:
        # Use Gurobi for exact solution
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 1:
                remaining_time = 1
            
            model = gp.Model("assignment")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', remaining_time - 0.5)
            model.setParam('Threads', 1)
            
            # Decision variables
            x = {}
            for i in range(n):
                for j in range(n):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
            
            # Auxiliary variable for worst-case deviation (unrestricted in sign)
            z = model.addVar(lb=-GRB.INFINITY, name='z')
            
            model.update()
            
            # Assignment constraints
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
            for j in range(n):
                model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1)
            
            # z >= lambda_k * (sum of costs for criterion k - reference_k) for each k
            for k in range(p):
                expr = gp.LinExpr()
                for i in range(n):
                    for j in range(n):
                        expr.addTerms(search_direction[k] * cost_matrices[k][i][j], x[i, j])
                expr.addConstant(-search_direction[k] * reference_point[k])
                model.addConstr(z >= expr)
            
            model.setObjective(z, GRB.MINIMIZE)
            
            # Callback for logging incumbents
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    # Extract assignment
                    vals = model.cbGetSolution([x[i, j] for i in range(n) for j in range(n)])
                    assignment = [0] * n
                    idx = 0
                    for i in range(n):
                        for j in range(n):
                            if vals[idx] > 0.5:
                                assignment[i] = j
                            idx += 1
                    update_best(assignment, obj)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                assignment = [0] * n
                for i in range(n):
                    for j in range(n):
                        if x[i, j].X > 0.5:
                            assignment[i] = j
                update_best(assignment)
        
        except Exception as e:
            # Fallback: brute force for very small n
            if n <= 8:
                from itertools import permutations
                for perm in permutations(range(n)):
                    if time.time() - start_time > time_limit - 0.5:
                        break
                    update_best(list(perm))
            else:
                # Simple greedy
                assignment = list(range(n))
                update_best(assignment)
    else:
        # Use Gurobi for larger instances
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 1:
                remaining_time = 1
            
            model = gp.Model("assignment")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', remaining_time - 1.0)
            model.setParam('Threads', 1)
            
            # Decision variables
            x = {}
            for i in range(n):
                for j in range(n):
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
            
            z = model.addVar(lb=-GRB.INFINITY, name='z')
            
            model.update()
            
            # Assignment constraints
            for i in range(n):
                model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
            for j in range(n):
                model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1)
            
            # Precompute weighted costs
            for k in range(p):
                expr = gp.LinExpr()
                lam_k = search_direction[k]
                ref_k = reference_point[k]
                for i in range(n):
                    for j in range(n):
                        coeff = lam_k * float(cost_matrices[k][i][j])
                        if coeff != 0:
                            expr.addTerms(coeff, x[i, j])
                expr.addConstant(-lam_k * ref_k)
                model.addConstr(z >= expr)
            
            model.setObjective(z, GRB.MINIMIZE)
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    vals = model.cbGetSolution([x[i, j] for i in range(n) for j in range(n)])
                    assignment = [0] * n
                    idx = 0
                    for i in range(n):
                        for j in range(n):
                            if vals[idx] > 0.5:
                                assignment[i] = j
                            idx += 1
                    update_best(assignment, obj)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                assignment = [0] * n
                for i in range(n):
                    for j in range(n):
                        if x[i, j].X > 0.5:
                            assignment[i] = j
                update_best(assignment)
        
        except Exception as e:
            # Fallback: use scipy for a greedy approach per criterion, pick best
            try:
                from scipy.optimize import linear_sum_assignment
                for k in range(p):
                    if time.time() - start_time > time_limit - 0.5:
                        break
                    row_ind, col_ind = linear_sum_assignment(cost_matrices[k])
                    assignment = [0] * n
                    for i, j in zip(row_ind, col_ind):
                        assignment[i] = j
                    update_best(assignment)
            except:
                assignment = list(range(n))
                update_best(assignment)
    
    # If no solution found, use identity
    if best_assignment is None:
        best_assignment = list(range(n))
        best_objective = evaluate_assignment(best_assignment)
        if logger:
            logger.log(best_objective)
    
    # Build output
    assignment_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        assignment_matrix[i][best_assignment[i]] = 1
    
    solution = {
        "objective_value": best_objective,
        "assignment": best_assignment,
        "assignment_matrix": assignment_matrix
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)