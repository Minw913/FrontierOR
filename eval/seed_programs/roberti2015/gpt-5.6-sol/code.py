import argparse
import json
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_solution(flow, fixed_cost, variable_cost):
    """Normalize a flow matrix and construct the required solution dictionary."""
    m = len(flow)
    n = len(flow[0]) if m else 0

    normalized = [[0.0] * n for _ in range(m)]
    arcs_used = [[0] * n for _ in range(m)]
    objective = 0.0

    for i in range(m):
        for j in range(n):
            value = float(flow[i][j])

            if abs(value) <= 1e-8:
                value = 0.0
            else:
                nearest = round(value)
                if abs(value - nearest) <= 1e-6:
                    value = float(nearest)
                elif value < 0.0 and value >= -1e-6:
                    value = 0.0

            normalized[i][j] = value
            if value > 1e-8:
                arcs_used[i][j] = 1
                objective += float(fixed_cost[i][j])
                objective += float(variable_cost[i][j]) * value

    return {
        "objective_value": float(objective),
        "flow": normalized,
        "arcs_used": arcs_used,
    }


def northwest_corner(supply, demand):
    """Construct a guaranteed feasible transportation solution."""
    m = len(supply)
    n = len(demand)
    flow = [[0.0] * n for _ in range(m)]

    remaining_supply = [int(v) for v in supply]
    remaining_demand = [int(v) for v in demand]

    i = 0
    j = 0
    while i < m and j < n:
        quantity = min(remaining_supply[i], remaining_demand[j])
        if quantity > 0:
            flow[i][j] = float(quantity)
            remaining_supply[i] -= quantity
            remaining_demand[j] -= quantity

        if remaining_supply[i] == 0:
            i += 1
        if j < n and remaining_demand[j] == 0:
            j += 1

    return flow


def greedy_transport(supply, demand, capacity, ordered_arcs):
    """
    Construct a feasible transportation solution by considering routes in a
    supplied preference order. For the stated route capacities
    min(supply[i], demand[j]), this procedure is always feasible.
    """
    m = len(supply)
    n = len(demand)
    flow = [[0.0] * n for _ in range(m)]
    remaining_supply = [int(v) for v in supply]
    remaining_demand = [int(v) for v in demand]

    for index in ordered_arcs:
        i, j = divmod(index, n)
        if remaining_supply[i] <= 0 or remaining_demand[j] <= 0:
            continue

        quantity = min(
            remaining_supply[i],
            remaining_demand[j],
            int(capacity[i][j]),
        )
        if quantity <= 0:
            continue

        flow[i][j] += float(quantity)
        remaining_supply[i] -= quantity
        remaining_demand[j] -= quantity

    if any(v != 0 for v in remaining_supply) or any(v != 0 for v in remaining_demand):
        return None
    return flow


def ensure_parent_directory(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Solve the fixed-charge transportation problem."
    )
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.monotonic()
    deadline = start_time + max(0, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    m = int(instance["num_sources"])
    n = int(instance["num_sinks"])
    supply = [int(v) for v in instance["supply"]]
    demand = [int(v) for v in instance["demand"]]
    fixed_cost = instance["fixed_cost"]
    variable_cost = instance["variable_cost"]
    capacity = instance["capacity"]

    if m <= 0 or n <= 0:
        raise ValueError("The instance must contain at least one source and one sink.")
    if sum(supply) != sum(demand):
        raise ValueError("Total supply must equal total demand.")

    best_solution = None
    best_objective = math.inf

    def register_solution(flow):
        nonlocal best_solution, best_objective
        if flow is None:
            return
        solution = build_solution(flow, fixed_cost, variable_cost)
        objective = solution["objective_value"]
        tolerance = 1e-8 * max(1.0, abs(best_objective)) if math.isfinite(best_objective) else 0.0

        if objective < best_objective - tolerance:
            best_objective = objective
            best_solution = solution
            if logger:
                logger.log_solution(objective, solution)

    # A fast, guaranteed feasible incumbent.
    register_solution(northwest_corner(supply, demand))

    # Generate several inexpensive deterministic heuristic incumbents when time
    # and instance size permit.
    arc_count = m * n
    heuristic_budget_end = min(deadline, start_time + min(3.0, 0.12 * max(1, args.time_limit)))

    if arc_count <= 250000:
        alphas = (0.0, 0.25, 1.0, 4.0, 16.0)
        for alpha in alphas:
            if time.monotonic() >= heuristic_budget_end:
                break

            def score(index, a=alpha):
                i, j = divmod(index, n)
                cap = max(1.0, float(capacity[i][j]))
                return (
                    float(variable_cost[i][j])
                    + a * float(fixed_cost[i][j]) / cap,
                    float(fixed_cost[i][j]),
                    index,
                )

            order = sorted(range(arc_count), key=score)
            register_solution(
                greedy_transport(supply, demand, capacity, order)
            )

    remaining_time = deadline - time.monotonic()

    if remaining_time > 0.01:
        try:
            model = gp.Model("fixed_charge_transportation")
            model.Params.OutputFlag = 0
            model.Params.Seed = 0
            model.Params.MIPGap = 1e-4
            model.Params.NumericFocus = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = max(0.01, remaining_time)

            xvars = []
            yvars = []

            for i in range(m):
                for j in range(n):
                    cap = float(capacity[i][j])
                    x = model.addVar(
                        lb=0.0,
                        ub=cap,
                        vtype=GRB.CONTINUOUS,
                        obj=float(variable_cost[i][j]),
                        name=f"x_{i}_{j}",
                    )
                    y = model.addVar(
                        lb=0.0,
                        ub=1.0,
                        vtype=GRB.BINARY,
                        obj=float(fixed_cost[i][j]),
                        name=f"y_{i}_{j}",
                    )
                    xvars.append(x)
                    yvars.append(y)

            model.ModelSense = GRB.MINIMIZE
            model.update()

            for i in range(m):
                model.addConstr(
                    gp.quicksum(xvars[i * n + j] for j in range(n)) == supply[i],
                    name=f"supply_{i}",
                )

            for j in range(n):
                model.addConstr(
                    gp.quicksum(xvars[i * n + j] for i in range(m)) == demand[j],
                    name=f"demand_{j}",
                )

            for i in range(m):
                for j in range(n):
                    k = i * n + j
                    cap = float(capacity[i][j])
                    model.addConstr(
                        xvars[k] <= cap * yvars[k],
                        name=f"link_upper_{i}_{j}",
                    )
                    # Integer margins and integral capacities guarantee that an
                    # optimal fixed-support transportation flow can be integral.
                    # Thus this valid strengthening does not remove an optimum.
                    if cap >= 1.0:
                        model.addConstr(
                            xvars[k] >= yvars[k],
                            name=f"link_lower_{i}_{j}",
                        )

            # Supply the best heuristic solution as a MIP start.
            if best_solution is not None:
                for i in range(m):
                    for j in range(n):
                        k = i * n + j
                        value = float(best_solution["flow"][i][j])
                        xvars[k].Start = value
                        yvars[k].Start = 1.0 if value > 1e-8 else 0.0

            model.update()

            def incumbent_callback(cb_model, where):
                if time.monotonic() >= deadline:
                    cb_model.terminate()
                    return

                if where != GRB.Callback.MIPSOL:
                    return

                try:
                    values = cb_model.cbGetSolution(xvars)
                    candidate_flow = [
                        [float(values[i * n + j]) for j in range(n)]
                        for i in range(m)
                    ]
                    register_solution(candidate_flow)
                except Exception:
                    # Logging or extracting an incumbent must not invalidate the
                    # optimization run.
                    return

            model.optimize(incumbent_callback)

            # Process the final incumbent as well, including cases where a solver
            # version did not issue a callback for the final solution.
            if model.SolCount > 0:
                final_flow = [
                    [float(xvars[i * n + j].X) for j in range(n)]
                    for i in range(m)
                ]
                register_solution(final_flow)

        except gp.GurobiError:
            # The heuristic incumbent remains a valid fallback if Gurobi cannot
            # start, for example due to an unavailable license.
            pass

    if best_solution is None:
        raise RuntimeError("Unable to construct a feasible solution.")

    ensure_parent_directory(args.solution_path)
    with open(args.solution_path, "w", encoding="utf-8") as file:
        json.dump(best_solution, file, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()