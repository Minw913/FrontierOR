import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    # Initialize logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    # Read instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    n = instance['n']          # number of items
    m = instance['m']          # number of knapsacks
    profits = instance['profits']
    pairwise_profits = instance['pairwise_profits']
    weights = instance['weights']
    capacities = instance['capacities']

    # Create Gurobi model
    model = gp.Model('GeneralizedKnapSack')
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)  # Single core constraint
    model.setParam(GRB.Param.NonConvex', 0)  # This is a convex quadratic problem after reformulation
    
    # Decision variables:
    # x[i,j]: binary, whether item i is assigned to knapsack j
    x = {}  # x: (item i, knapsack j) -> binary variable
    for i in range(n):
        for j in range(m):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Capacity constraints: sum of weights per knapsack must not exceed capacity
    for j in range(m):
        model.addConstr(
            gp.quicksum(x[i, j] * weights[i] for i in range(n)) <= capacities[j],
            name=f"capacity_{j}"
        )

    # Assignment constraints: each item at most to one knapsack
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(m)) <= 1,
            name=f"one_assignment_per_item_{i}"
        )

    # Linearize the pairwise profit term
    # z[i1, i2, j]: binary, whether both i1 and i2 are assigned to knapsack j (i1 < i2)
    z = {}
    for i1 in range(n):
        for i2 in range(i1+1, n):  # Only consider unique pairs
            for j in range(m):
                z[i1, i2, j] = model.addVar(vtype=GRB.BINARY, name=f"z_{i1}_{i2}_{j}")
                # z[i1, i2, j] <= x[i1, j]
                model.addConstr(z[i1, i2, j] <= x[i1, j])
                # z[i1, i2, j] <= x[i2, j]
                model.addConstr(z[i1, i2, j] <= x[i2, j])
                # z[i1, i2, j] >= x[i1, j] + x[i2, j] - 1
                model.addConstr(z[i1, i2, j] >= x[i1, j] + x[i2, j] - 1)

    # Objective function:
    # max sum over i,j: x[i,j] * profits[i]
    #     + sum over i1<i2, j: z[i1,i2,j] * pairwise_profits[i1][i2]
    
    obj = gp.LinExpr()
    # Individual profits
    obj.addTerms([profits[i]]*m, [x[i, j] for i in range(n) for j in range(m)])
    # Pairwise profits
    for i1 in range(n):
        for i2 in range(i1+1, n):
            obj.addTerms([pairwise_profits[i1][i2]]*m, [z[i1, i2, j] for j in range(m)])

    model.setObjective(obj, GRB.MAXIMIZE)

    # Optimize the model
    model.optimize()

    # Extract the solution
    assignment = []
    total_profit = 0

    # Calculate assignment
    for i in range(n):
        for j in range(m):
            if x[i, j].X > 0.99:  # Considering it assigned if value is near 1
                assignment.append([i, j])

    # Calculate the actual objective value
    # Calculate individual profit contribution
    individual_profit = sum(profits[i] for i, j in assignment)
    
    # Calculate pairwise profit contribution
    pairwise_profit = 0
    for i1 in range(n):
        for i2 in range(i1+1, n):
            for j in range(m):
                if x[i1, j].X > 0.99 and x[i2, j].X > 0.99:
                    pairwise_profit += pairwise_profits[i1][i2]

    total_profit = individual_profit + pairwise_profit

    # Create output dictionary
    solution = {
        "objective_value": total_profit,
        "assignment": assignment
    }

    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution, f)

    # If a solution was found (status is feasible), log for improvement tracking
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.objVal is not None:
        if logger:
            logger.log(total_profit)


def main():
    parser = argparse.ArgumentParser(description='Solve Generalized Knapsack Problem')
    parser.add_argument('--instance_path', type=str, required=True, 
                        help='Path to the input JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to write the output JSON solution file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum allowed runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to write the JSONL log file')

    args = parser.parse_args()
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == "__main__":
    main()