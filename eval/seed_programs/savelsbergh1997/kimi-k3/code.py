import argparse
import json
import sys
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    m = int(data["num_agents"])
    n = int(data["num_jobs"])
    profits = data["profits"]
    weights = data["weights"]
    capacities = data["capacities"]
    return m, n, profits, weights, capacities


def greedy_start(m, n, profits, weights, capacities):
    """Build a feasible greedy assignment (may leave jobs unassigned).

    Jobs are processed in order of decreasing 'regret' (best profit minus
    second-best profit), and each is assigned to the feasible agent with
    the highest profit.
    """
    remaining = list(capacities)
    assign = [-1] * n

    regrets = []
    for j in range(n):
        col = [profits[i][j] for i in range(m)]
        best = max(col)
        second = max((c for c in col if c != best), default=0) if m > 1 else 0
        regrets.append(best - second)
    order = sorted(range(n), key=lambda j: -regrets[j])

    for j in order:
        best_i, best_p = -1, -1
        for i in range(m):
            if weights[i][j] <= remaining[i] and profits[i][j] > best_p:
                best_p = profits[i][j]
                best_i = i
        if best_i >= 0:
            assign[j] = best_i
            remaining[best_i] -= weights[best_i][j]
    return assign


def assignment_to_dict(assignment):
    return {str(j): int(a) for j, a in enumerate(assignment)}


def make_solution_dict(assignment, m, n, profits):
    obj = 0.0
    for j in range(n):
        i = assignment[j]
        if i >= 0:
            obj += float(profits[i][j])
    return {"objective_value": obj, "assignment": assignment_to_dict(assignment)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    t_start = time.time()
    m, n, profits, weights, capacities = load_instance(args.instance_path)

    # Greedy heuristic for a quick feasible incumbent / MIP start.
    greedy_assign = greedy_start(m, n, profits, weights, capacities)
    if all(a >= 0 for a in greedy_assign):
        sol_dict = make_solution_dict(greedy_assign, m, n, profits)
        if logger:
            logger.log_solution(sol_dict["objective_value"], sol_dict)

    model = gp.Model("GAP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    remaining_time = max(1, args.time_limit - int(time.time() - t_start))
    model.Params.TimeLimit = remaining_time

    agents = range(m)
    jobs = range(n)

    x = {}
    for i in agents:
        for j in jobs:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Each job assigned to exactly one agent.
    for j in jobs:
        model.addConstr(gp.quicksum(x[i, j] for i in agents) == 1,
                        name=f"job_{j}")

    # Capacity constraints.
    for i in agents:
        model.addConstr(
            gp.quicksum(weights[i][j] * x[i, j] for j in jobs) <= capacities[i],
            name=f"cap_{i}")

    model.setObjective(
        gp.quicksum(profits[i][j] * x[i, j] for i in agents for j in jobs),
        GRB.MAXIMIZE)

    # MIP start from greedy.
    for j, a in enumerate(greedy_assign):
        if a >= 0:
            x[a, j].Start = 1.0

    xvars = [x[i, j] for i in agents for j in jobs]

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL and logger:
            try:
                obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
                vals = mdl.cbGetSolution(xvars)
                assign = []
                for j in jobs:
                    best_i = 0
                    best_v = -0.5
                    for i in agents:
                        v = vals[i * n + j]
                        if v > best_v:
                            best_v = v
                            best_i = i
                    assign.append(best_i)
                # Recompute true objective from the rounded assignment.
                true_obj = float(sum(profits[assign[j]][j] for j in jobs))
                sol = {"objective_value": true_obj,
                       "assignment": assignment_to_dict(assign)}
                logger.log_solution(true_obj, sol)
            except Exception:
                try:
                    logger.log(float(mdl.cbGet(GRB.Callback.MIPSOL_OBJ)))
                except Exception:
                    pass

    model.optimize(callback)

    # Extract the best solution found.
    final_assignment = None
    if model.SolCount > 0:
        final_assignment = []
        for j in jobs:
            best_i = 0
            best_v = -0.5
            for i in agents:
                v = x[i, j].X
                if v > best_v:
                    best_v = v
                    best_i = i
            final_assignment.append(best_i)
    elif all(a >= 0 for a in greedy_assign):
        final_assignment = greedy_assign
    else:
        # Last-resort fallback (may violate capacities); assign each job to
        # its most profitable agent so the output schema is still satisfied.
        final_assignment = [max(agents, key=lambda i: profits[i][j])
                            for j in jobs]

    sol = make_solution_dict(final_assignment, m, n, profits)
    sol["objective_value"] = float(sol["objective_value"])

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)

    if logger:
        # Ensure the final incumbent is recorded.
        try:
            logger.log_solution(sol["objective_value"], sol)
        except Exception:
            pass


if __name__ == "__main__":
    main()