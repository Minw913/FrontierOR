import argparse
import json
import time
import numpy as np

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_agents = instance['num_agents']
    num_jobs = instance['num_jobs']
    profits = instance['profits']
    weights = instance['weights']
    capacities = instance['capacities']

    start_time = time.time()

    # Use Gurobi MIP solver
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("GAP")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', max(1, args.time_limit - 2))
    model.setParam('Threads', 1)

    # Decision variables: x[i][j] = 1 if agent i is assigned job j
    x = {}
    for i in range(num_agents):
        for j in range(num_jobs):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}",
                                    obj=profits[i][j])

    model.setAttr('ModelSense', GRB.MAXIMIZE)

    # Each job assigned to exactly one agent
    for j in range(num_jobs):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(num_agents)) == 1,
            name=f"job_{j}"
        )

    # Capacity constraints
    for i in range(num_agents):
        model.addConstr(
            gp.quicksum(weights[i][j] * x[i, j] for j in range(num_jobs)) <= capacities[i],
            name=f"cap_{i}"
        )

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(callback)

    # Extract solution
    if model.SolCount > 0:
        objective_value = model.ObjVal
        assignment = {}
        for j in range(num_jobs):
            for i in range(num_agents):
                if x[i, j].X > 0.5:
                    assignment[str(j)] = i
                    break
    else:
        # Try to find a feasible solution via greedy heuristic
        # Sort jobs by decreasing max profit
        assignment = {}
        remaining_cap = list(capacities)
        objective_value = 0.0

        # Greedy: for each job, assign to agent with best profit that has capacity
        job_order = list(range(num_jobs))
        # Sort by minimum weight across agents (hardest to place first)
        job_order.sort(key=lambda j: min(weights[i][j] for i in range(num_agents)))

        for j in job_order:
            best_agent = -1
            best_profit = -1
            for i in range(num_agents):
                if weights[i][j] <= remaining_cap[i]:
                    if profits[i][j] > best_profit:
                        best_profit = profits[i][j]
                        best_agent = i
            if best_agent >= 0:
                assignment[str(j)] = best_agent
                remaining_cap[best_agent] -= weights[best_agent][j]
                objective_value += profits[best_agent][j]
            else:
                # Infeasible - assign to agent 0 anyway (violating capacity)
                assignment[str(j)] = 0
                objective_value += profits[0][j]

        if logger:
            logger.log(objective_value)

    solution = {
        "objective_value": float(objective_value),
        "assignment": assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    solve()