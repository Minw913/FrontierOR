import argparse
import json
import numpy as np
from gurobipy import *
from solution_logger import SolutionLogger


def solve_bilinear_assignment(instance):
    """
    Solves the bilinear assignment problem with objective:
    min (sum a_ij*x_ij) * (sum b_ij*x_ij) + (sum c_ij*x_ij)
    subject to x being a permutation matrix.
    """
    n = instance['n']
    A = instance['A']
    B = instance['B']
    C = instance['C']

    # Create model
    m = Model("bilinear_assignment")
    m.setParam("Threads", 1)
    m.setParam('OutputFlag', 0)  # Silent optimization
    
    # Decision variables: x[i,j] = 1 if entity i is assigned to position j
    x = {}
    for i in range(n):
        for j in range(n):
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # Variables for the sums to linearize the quadratic term
    A_sum = m.addVar(name="A_sum")
    B_sum = m.addVar(name="B_sum")
    
    # Constraints: one assignment per entity
    for i in range(n):
        m.addConstr(quicksum(x[i, j] for j in range(n)) == 1, name=f"entity_{i}_assigned_once")
    
    # Constraints: one entity per position
    for j in range(n):
        m.addConstr(quicksum(x[i, j] for i in range(n)) == 1, name=f"position_{j}_filled_once")

    # Linking constraints for A_sum and B_sum
    m.addConstr(A_sum == quicksum(A[i][j] * x[i, j] for i in range(n) for j in range(n)), name="A_linking")
    m.addConstr(B_sum == quicksum(B[i][j] * x[i, j] for i in range(n) for j in range(n)), name="B_linking")

    # To handle the bilinear term A_sum * B_sum, we linearize using McCormick envelopes
    # Need bounds on A_sum and B_sum first
    min_A = min(min(row) for row in A)
    max_A = max(max(row) for row in A)
    min_B = min(min(row) for row in B)
    max_B = max(max(row) for row in B)

    # Min and Max possible sums given assignment constraints
    # Since it's a permutation we can just consider the assignments that would produce min/max
    A_min_bound = sum(sorted([A[i][j] for i in range(n) for j in range(n)])[:n])
    A_max_bound = sum(sorted([A[i][j] for i in range(n) for j in range(n)], reverse=True)[:n])
    B_min_bound = sum(sorted([B[i][j] for i in range(n) for j in range(n)])[:n])
    B_max_bound = sum(sorted([B[i][j] for i in range(n) for j in range(n)], reverse=True)[:n])

    # Introduce a new variable for the bilinear product: Z = A_sum * B_sum
    Z = m.addVar(vtype=GRB.CONTINUOUS, lb=A_min_bound*B_min_bound, ub=A_max_bound*B_max_bound, name="Z_product")
    
    # Add McCormick envelopes:
    # 1. Z >= A_min*B_sum + B_min*A_sum - A_min*B_min
    m.addConstr(Z >= A_min_bound * B_sum + B_min_bound * A_sum - A_min_bound * B_min_bound, "McCormick_1")
    
    # 2. Z >= A_max*B_sum + B_max*A_sum - A_max*B_max
    m.addConstr(Z >= A_max_bound * B_sum + B_max_bound * A_sum - A_max_bound * B_max_bound, "McCormick_2")
    
    # 3. Z <= A_max*B_sum + B_min*A_sum - A_max*B_min
    m.addConstr(Z <= A_max_bound * B_sum + B_min_bound * A_sum - A_max_bound * B_min_bound, "McCormick_3")
    
    # 4. Z <= A_min*B_sum + B_max*A_sum - A_min*B_max
    m.addConstr(Z <= A_min_bound * B_sum + B_max_bound * A_sum - A_min_bound * B_max_bound, "McCormick_4")
    
    # Compute the C-term
    C_sum = quicksum(C[i][j] * x[i, j] for i in range(n) for j in range(n))
    
    # Set the objective function: minimize Z + C_sum
    m.setObjective(Z + C_sum, GRB.MINIMIZE)
    
    m.update()
    
    return m, x, A_sum, B_sum, Z


def extract_solution(model, x, n):
    # Extract the assignment
    assignment = [-1] * n
    for i in range(n):
        for j in range(n):
            if model.getVarByName(f"x_{i}_{j}").X > 0.5:
                assignment[i] = j
                break
    
    return assignment


def main():
    parser = argparse.ArgumentParser(description='Solve generalized quadratic assignment problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the instance file in JSON format')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to save the solution file in JSON format')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to log intermediate solutions')

    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Solve the instance using Gurobi model
    model, x, A_sum, B_sum, Z = solve_bilinear_assignment(instance)
    
    n = instance['n']
    C = instance['C']
    
    # Set time limit
    model.setParam('TimeLimit', args.time_limit)
    
    # Define the callback function to process new incumbent solutions
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
            
    # Optimize
    try:
        model.optimize(mycallback)
    except Exception as e:
        print(f"Optimization error: {e}")
        
        # Get current best solution if available
        if model.SolCount == 0:
            # No feasible solution found, return dummy/first guess assignment
            assignment = list(range(n))
            a_cost = sum(instance['A'][i][assignment[i]] for i in range(n))
            b_cost = sum(instance['B'][i][assignment[i]] for i in range(n))
            c_cost = sum(instance['C'][i][assignment[i]] for i in range(n))
            total_cost = a_cost * b_cost + c_cost
            
            solution = {
                "objective_value": float(total_cost),
                "assignment": assignment
            }
            
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f)
            return
    
    # Extract the final solution
    if model.status == GRB.OPTIMAL or model.SolCount > 0:
        assignment = extract_solution(model, x, n)
        # Calculate the final objective value
        a_cost = sum(instance['A'][i][assignment[i]] for i in range(n))
        b_cost = sum(instance['B'][i][assignment[i]] for i in range(n))
        c_cost = sum(instance['C'][i][assignment[i]] for i in range(n))
        total_cost = a_cost * b_cost + c_cost

        solution = {
            "objective_value": float(total_cost),
            "assignment": assignment
        }
    else:
        # No feasible solution found within time limits
        assignment = list(range(n))
        a_cost = sum(instance['A'][i][assignment[i]] for i in range(n))
        b_cost = sum(instance['B'][i][assignment[i]] for i in range(n))
        c_cost = sum(instance['C'][i][assignment[i]] for i in range(n))
        total_cost = a_cost * b_cost + c_cost
        
        solution = {
            "objective_value": float(total_cost),
            "assignment": assignment
        }

    # Save the solution to the specified path
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == '__main__':
    main()