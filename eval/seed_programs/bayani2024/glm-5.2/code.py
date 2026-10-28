import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def compute_total_cost(assignment, n_clients, n_servers, linear_costs, quadratic_costs):
    """
    Compute the exact total cost of a given assignment.
    assignment: dict mapping client index (int) to server index (int).
    """
    total = 0.0

    # Linear processing costs
    for i in range(n_clients):
        s = assignment[i]
        total += linear_costs[i][s]

    # Adjacent (same-server) quadratic costs:
    # sum_{i<j assigned to same s} (linear[i][s] + linear[j][s])
    server_clients = {s: [] for s in range(n_servers)}
    for i in range(n_clients):
        server_clients[assignment[i]].append(i)

    for s in range(n_servers):
        clients = server_clients[s]
        m = len(clients)
        if m > 1:
            sum_linear = sum(linear_costs[i][s] for i in clients)
            total += (m - 1) * sum_linear

    # Out-of-star quadratic costs (explicitly provided pairs)
    for entry in quadratic_costs:
        i, s, j, t, cost = entry
        i = int(i)
        s = int(s)
        j = int(j)
        t = int(t)
        if assignment[i] == s and assignment[j] == t:
            total += cost

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n_clients = instance['n_clients']
    n_servers = instance['n_servers']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance.get('quadratic_costs', [])

    # ------------------------------------------------------------------
    # Initial greedy solution: assign each client to the server with
    # the minimum linear cost.
    # ------------------------------------------------------------------
    assignment = {}
    for i in range(n_clients):
        s = min(range(n_servers), key=lambda s: linear_costs[i][s])
        assignment[i] = s

    init_obj = compute_total_cost(assignment, n_clients, n_servers, linear_costs, quadratic_costs)
    if logger:
        sol_dict = {str(i): assignment[i] for i in range(n_clients)}
        logger.log_solution(init_obj, sol_dict)

    # ------------------------------------------------------------------
    # Build and solve the MIQP model with Gurobi
    # ------------------------------------------------------------------
    env = gp.Env()
    model = gp.Model(env=env)
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    model.Params.NonConvex = 2
    model.Params.OutputFlag = 0

    # Binary assignment variables x[i, s]
    x = model.addVars(n_clients, n_servers, vtype=GRB.BINARY, name='x')

    # Each client assigned to exactly one server
    model.addConstrs((x.sum(i, '*') == 1 for i in range(n_clients)), name='assign')

    # Objective: linear + quadratic
    obj = gp.QuadExpr()

    # Linear part
    for i in range(n_clients):
        for s in range(n_servers):
            obj += linear_costs[i][s] * x[i, s]

    # Adjacent quadratic part:
    # For server s, let y_s = sum_j x[j, s]. Then
    # sum_{i<j} (linear[i][s] + linear[j][s]) * x[i,s] * x[j,s]
    #   = sum_i linear[i][s] * x[i,s] * (y_s - 1)
    for s in range(n_servers):
        y = gp.LinExpr(sum(x[i, s] for i in range(n_clients)))
        for i in range(n_clients):
            obj += linear_costs[i][s] * x[i, s] * y - linear_costs[i][s] * x[i, s]

    # Out-of-star quadratic part (explicitly provided pairs)
    for entry in quadratic_costs:
        i, s, j, t, cost = entry
        i = int(i)
        s = int(s)
        j = int(j)
        t = int(t)
        obj += cost * x[i, s] * x[j, t]

    model.setObjective(obj, GRB.MINIMIZE)

    # Provide the greedy solution as a MIP start
    for i in range(n_clients):
        for s in range(n_servers):
            x[i, s].Start = 1 if assignment[i] == s else 0

    # Callback to log every incumbent found during the branch-and-bound
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            sol = model.cbGetSolution(x)
            assign = {}
            for i in range(n_clients):
                for s in range(n_servers):
                    if sol[i, s] > 0.5:
                        assign[str(i)] = s
                        break
            if logger:
                logger.log_solution(obj_val, assign)

    model.optimize(callback)

    # ------------------------------------------------------------------
    # Extract the best solution found
    # ------------------------------------------------------------------
    if model.SolCount > 0:
        final_assignment = {}
        for i in range(n_clients):
            for s in range(n_servers):
                if x[i, s].X > 0.5:
                    final_assignment[i] = s
                    break
        final_obj = compute_total_cost(final_assignment, n_clients, n_servers, linear_costs, quadratic_costs)
    else:
        # Fallback to the initial greedy solution
        final_assignment = assignment
        final_obj = init_obj

    # Write final solution
    output = {
        "objective_value": final_obj,
        "assignment": {str(i): final_assignment[i] for i in range(n_clients)}
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()