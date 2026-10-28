import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    m = instance['parameters']['m']
    n = instance['parameters']['n']
    assignment_costs = instance['cost_matrices']['assignment_costs']['matrix']
    interconnection_costs = instance['cost_matrices']['interconnection_costs']['matrix']
    node_weights = instance['steiner_nodes']['node_weights']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model("Steiner_Tree_with_Assignments")
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('MIPGap', 1e-4)
    model.setParam('LogToConsole', 0)

    # Variables
    y = model.addVars(n, vtype=GRB.BINARY, name="y")  # activate steiner
    x = model.addVars(m, n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")  # assignment
    e = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="e")  # edges

    # Objective
    assign_cost = gp.quicksum(assignment_costs[i][j] * x[i, j] for i in range(m) for j in range(n))
    edge_cost = gp.quicksum(interconnection_costs[i][j] * e[i, j] for i in range(n) for j in range(i+1, n))
    activation_cost = gp.quicksum(node_weights[j] * y[j] for j in range(n))
    model.setObjective(assign_cost + edge_cost + activation_cost, GRB.MINIMIZE)

    # Constraints
    # Each target assigned to exactly one steiner
    for i in range(m):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f"assign_{i}")

    # Assignment only to active steiner
    for i in range(m):
        for j in range(n):
            model.addConstr(x[i, j] <= y[j], name=f"assign_bound_{i}_{j}")

    # Only active steiners can have edges
    for i in range(n):
        for j in range(i+1, n):
            model.addConstr(e[i, j] <= y[i], name=f"edge_y1_{i}_{j}")
            model.addConstr(e[i, j] <= y[j], name=f"edge_y2_{i}_{j}")

    # Spanning tree constraints: for each subset S of Steiners with |S|>=2, sum edges in S <= sum y in S -1
    # But we use MTZ formulation for efficiency on small n
    if n > 1:
        u = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=n, name="u")
        for i in range(n):
            for j in range(n):
                if i != j:
                    model.addConstr(u[i] - u[j] + (n * e[min(i,j), max(i,j)]) <= n - 1, name=f"mtz_{i}_{j}")
        # Link u to y
        for j in range(1, n):  # assume node 0 as root if activated
            model.addConstr(u[j] >= 2 - n * (1 - y[j]), name=f"u_lower_{j}")
            model.addConstr(u[j] <= n * y[j], name=f"u_upper_{j}")
        # At least one y if any assignment
        model.addConstr(gp.quicksum(y[j] for j in range(n)) >= 1, name="at_least_one")

    # Degree constraints to prevent trivial cycles but MTZ should suffice
    for i in range(n):
        model.addConstr(gp.quicksum(e[min(i,k), max(i,k)] for k in range(n) if k != i) <= (n-1) * y[i], name=f"degree_{i}")

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        objective = model.objVal
        active = [j for j in range(n) if y[j].X > 0.5]
        assignments = {}
        for i in range(m):
            for j in range(n):
                if x[i, j].X > 0.5:
                    assignments[str(i)] = j
                    break
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if e[i, j].X > 0.5:
                    edges.append([i, j])

        solution = {
            "objective_value": float(objective),
            "active_steiner_nodes": active,
            "assignments": assignments,
            "steiner_tree_edges": edges
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # If infeasible or other error, write empty with high cost
        solution = {
            "objective_value": 1e9,
            "active_steiner_nodes": [],
            "assignments": {},
            "steiner_tree_edges": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()