import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n_clients = instance['n_clients']
    n_servers = instance['n_servers']
    demands = instance['demands']
    processing_rates = instance['processing_rates']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance['quadratic_costs']
    out_of_star_density = instance['out_of_star_density']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()
    remaining_time = args.time_limit

    model = gp.Model("ClientServerAssignment")
    model.setParam('TimeLimit', remaining_time)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.01)
    model.setParam('Heuristics', 0.8)
    model.setParam(' MIPFocus', 1)

    # Binary variables: x[c][s] = 1 if client c assigned to server s
    x = [[model.addVar(vtype=GRB.BINARY, name=f"x_{c}_{s}") for s in range(n_servers)] for c in range(n_clients)]

    # Each client assigned to exactly one server
    for c in range(n_clients):
        model.addConstr(gp.quicksum(x[c][s] for s in range(n_servers)) == 1, name=f"assign_{c}")

    # Linear cost
    linear_expr = gp.quicksum(linear_costs[c][s] * x[c][s] for c in range(n_clients) for s in range(n_servers))

    # Quadratic terms
    quad_expr = gp.QuadExpr()

    # First, handle all adjacent (same-server) quadratic costs
    # For each server, for every pair of distinct clients, add (linear_c1 + linear_c2) * x_c1s * x_c2s
    for s in range(n_servers):
        for i in range(n_clients):
            for j in range(i+1, n_clients):
                coeff = linear_costs[i][s] + linear_costs[j][s]
                quad_expr.add(x[i][s] * x[j][s], coeff)

    # Now add the out-of-star quadratic costs from the list
    for entry in quadratic_costs:
        c1, s1, c2, s2, cost = entry
        # Only add if they are on different servers (out-of-star)
        if s1 != s2:
            quad_expr.add(x[c1][s1] * x[c2][s2], cost)

    # Objective
    model.setObjective(linear_expr + quad_expr, GRB.MINIMIZE)

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    model.optimize(callback)

    # Extract solution
    assignment = {}
    for c in range(n_clients):
        for s in range(n_servers):
            if x[c][s].X > 0.5:
                assignment[str(c)] = s
                break

    objective_value = model.ObjVal

    solution = {
        "objective_value": float(objective_value),
        "assignment": assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.close()

if __name__ == "__main__":
    main()