import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solver for the Generalized Assignment Problem (GAP)."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path where the solution JSON must be written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Optional path to a JSONL convergence log file.")
    return parser.parse_args()


def greedy_assignment(M, N, cost, resource, capacities):
    """Fast constructive heuristic used as a warm start / fallback."""
    def attempt(order):
        remaining = list(capacities)
        assign = [-1] * N
        for j in order:
            best_i, best_c = -1, None
            for i in range(M):
                if resource[i][j] <= remaining[i]:
                    c = cost[i][j]
                    if best_c is None or c < best_c:
                        best_c, best_i = c, i
            if best_i < 0:
                return None
            assign[j] = best_i
            remaining[best_i] -= resource[best_i][j]
        return assign

    jobs = list(range(N))
    orderings = [
        sorted(jobs, key=lambda j: -max(resource[i][j] for i in range(M))),
        sorted(jobs, key=lambda j: -(max(cost[i][j] for i in range(M))
                                     - min(cost[i][j] for i in range(M)))),
        jobs,
    ]
    best, best_obj = None, float("inf")
    for order in orderings:
        a = attempt(order)
        if a is not None:
            obj = sum(cost[a[j]][j] for j in range(N))
            if obj < best_obj:
                best, best_obj = a, obj
    return best


def assignment_from_values(vals, M, N):
    """Decode a flat list of x-values (ordered (i, j) lexicographically)."""
    assign = []
    for j in range(N):
        chosen, best_v = 0, -1.0
        for i in range(M):
            v = vals[i * N + j]
            if v > best_v:
                best_v, chosen = v, i
        assign.append(chosen)
    return assign


def evaluate(assign, M, N, cost, resource, capacities):
    """Return objective if capacity-feasible, else None."""
    usage = [0.0] * M
    obj = 0.0
    for j in range(N):
        i = assign[j]
        usage[i] += resource[i][j]
        obj += cost[i][j]
    if all(usage[i] <= capacities[i] + 1e-6 for i in range(M)):
        return float(obj)
    return None


def make_solution_dict(assign, obj, N):
    return {
        "objective_value": float(obj),
        "assignments": {str(j): int(assign[j]) for j in range(N)},
    }


def main():
    args = parse_args()
    start_time = time.time()

    logger = (SolutionLogger(args.log_path, sense="minimize")
              if args.log_path else None)

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    M = int(data["num_machines"])
    N = int(data["num_jobs"])
    cost = data["cost_matrix"]
    resource = data["resource_matrix"]
    capacities = data["capacities"]

    # --- Fast heuristic fallback / warm start -----------------------------
    heuristic_assign = greedy_assignment(M, N, cost, resource, capacities)

    best_logged = [float("inf")]
    if heuristic_assign is not None and logger:
        obj = evaluate(heuristic_assign, M, N, cost, resource, capacities)
        if obj is not None:
            best_logged[0] = obj
            logger.log_solution(obj, make_solution_dict(heuristic_assign, obj, N))

    # --- MIP model ---------------------------------------------------------
    model = gp.Model("generalized_assignment")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    remaining = args.time_limit - (time.time() - start_time)
    model.Params.TimeLimit = max(remaining, 0.0)

    x = model.addVars(M, N, vtype=GRB.BINARY, name="x")

    model.addConstrs(
        (x.sum("*", j) == 1 for j in range(N)), name="assign_job"
    )
    model.addConstrs(
        (gp.quicksum(resource[i][j] * x[i, j] for j in range(N))
         <= capacities[i] for i in range(M)),
        name="capacity",
    )
    model.setObjective(
        gp.quicksum(cost[i][j] * x[i, j] for i in range(M) for j in range(N)),
        GRB.MINIMIZE,
    )

    all_vars = [x[i, j] for i in range(M) for j in range(N)]

    # Warm start with the heuristic solution when available.
    if heuristic_assign is not None:
        for j in range(N):
            hj = heuristic_assign[j]
            for i in range(M):
                x[i, j].Start = 1.0 if i == hj else 0.0

    # --- Incumbent logging callback ---------------------------------------
    def incumbent_callback(mdl, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            vals = mdl.cbGetSolution(all_vars)
        except gp.GurobiError:
            return
        assign = assignment_from_values(vals, M, N)
        obj = evaluate(assign, M, N, cost, resource, capacities)
        if obj is not None and obj < best_logged[0] - 1e-9:
            best_logged[0] = obj
            logger.log_solution(obj, make_solution_dict(assign, obj, N))

    # --- Solve --------------------------------------------------------------
    if model.Params.TimeLimit > 0.0:
        try:
            model.optimize(incumbent_callback)
        except gp.GurobiError:
            pass

    # --- Extract best available solution ------------------------------------
    final_assign = None
    final_obj = None
    if model.SolCount > 0:
        try:
            vals = model.getAttr("X", all_vars)
            cand = assignment_from_values(vals, M, N)
            cand_obj = evaluate(cand, M, N, cost, resource, capacities)
            if cand_obj is not None:
                final_assign, final_obj = cand, cand_obj
        except gp.GurobiError:
            pass

    if final_assign is None and heuristic_assign is not None:
        final_assign = heuristic_assign
        final_obj = evaluate(final_assign, M, N, cost, resource, capacities)

    if final_assign is None:
        # Last-resort schema-valid output (instance may be infeasible).
        final_assign = [0] * N
        final_obj = float(sum(cost[0][j] for j in range(N)))

    if logger and final_obj is not None and final_obj < best_logged[0] - 1e-9:
        best_logged[0] = final_obj
        logger.log_solution(final_obj, make_solution_dict(final_assign, final_obj, N))

    solution = make_solution_dict(final_assign, final_obj, N)
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()