import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
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
    linear_costs = instance['linear_costs']
    quadratic_costs = instance['quadratic_costs']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = gp.Model("ClientServerAssignment")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Heuristics', 0.8)
    model.setParam('MIPFocus', 1)
    model.setParam('LogToConsole', 0)

    x = [[model.addVar(vtype=GRB.BINARY, name=f"x_{c}_{s}") for s in range(n_servers)] for c in range(n_clients)]

    for c in range(n_clients):
        model.addConstr(gp.quicksum(x[c][s] for s in range(n_servers)) == 1, name=f"assign_{c}")

    linear_expr = gp.quicksum(linear_costs[c][s] * x[c][s] for c in range(n_clients) for s in range(n_servers))

    quad_expr = gp.QuadExpr()

    for s in range(n_servers):
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                coeff = linear_costs[i][s] + linear_costs[j][s]
                quad_expr.add(x[i][s] * x[j][s], coeff)

    for entry in quadratic_costs:
        c1, s1, c2, s2, cost = entry
        if s1 != s2:
            quad_expr.add(x[c1][s1] * x[c2][s2], cost)

    model.setObjective(linear_expr + quad_expr, GRB.MINIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger and obj < float('inf'):
                logger.log(float(obj))

    model.optimize(callback)

    assignment = {}
    for c in range(n_clients):
        for s in range(n_servers):
            if abs(x[c][s].X) > 0.5:
                assignment[str(c)] = int(s)
                break

    objective_value = model.ObjVal if model.SolCount > 0 else 0.0

    solution = {
        "objective_value": float(objective_value),
        "assignment": assignment
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()