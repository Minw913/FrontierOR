"""Capacitated facility location with splittable customer demand.

Formulation (solved with Gurobi):
    minimize   sum_{i,j} ship[i][j] * x[i][j] + sum_j fixed[j] * y[j]
    subject to sum_j x[i][j] == 1                          for each customer i
               sum_i demand[i] * x[i][j] <= cap[j] * y[j]  for each plant j
               x[i][j] <= y[j]                              (linking / strengthening)
               0 <= x <= 1, y binary

A greedy "open everything" solution is constructed first as a guaranteed
feasible fallback and as a MIP warm start.  Every improved incumbent found
by Gurobi is reported to the SolutionLogger.
"""

import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger

# Add the disaggregated linking constraints x[i,j] <= y[j] only when the
# number of customer-plant pairs is moderate (they greatly strengthen the
# LP relaxation but cost memory/time for very large instances).
LINK_CONSTRAINT_LIMIT = 1_000_000
EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capacitated facility location with splittable demand."
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON problem instance.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the solution JSON will be written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional path to a JSONL convergence log.")
    return parser.parse_args()


def greedy_feasible(n, m, demands, capacities, fixed_costs, shipping):
    """Feasible solution with every plant opened.

    Because total capacity strictly exceeds total demand and demand may be
    split arbitrarily, assigning each customer to the cheapest plants with
    remaining capacity (all plants open) is always feasible.
    """
    y = [1] * m
    remaining = [float(c) for c in capacities]
    x = [[0.0] * m for _ in range(n)]
    for i in range(n):
        d = demands[i]
        order = sorted(range(m), key=lambda j: shipping[i][j])
        if d <= 0.0:
            x[i][order[0]] = 1.0
            continue
        need = 1.0
        for j in order:
            if need <= 1e-12:
                break
            if remaining[j] <= 1e-12:
                continue
            f = remaining[j] / d
            if f > need:
                f = need
            x[i][j] = f
            remaining[j] -= d * f
            need -= f
        if need > 1e-9:  # numerical safety net; should not trigger
            j = max(range(m), key=lambda jj: remaining[jj])
            x[i][j] += need
            remaining[j] -= d * need
    return x, y


def build_solution(x, y, n, m, demands, fixed_costs, shipping):
    """Project raw variable values onto the solution schema.

    Drops numerical dust, removes assignments to closed plants, normalizes
    each customer's fractions to sum to exactly one, and recomputes the
    objective consistently from the cleaned assignment.
    """
    open_plants = sorted(j for j in range(m) if y[j] > 0.5)
    if not open_plants:  # only possible if all demands are zero
        open_plants = [min(range(m), key=lambda j: fixed_costs[j])]
    open_set = set(open_plants)

    assignments = {}
    ship_cost = 0.0
    for i in range(n):
        row = {}
        total = 0.0
        xi = x[i]
        for j in range(m):
            v = xi[j]
            if v > EPS and j in open_set:
                row[j] = v
                total += v
        if total <= EPS:
            j0 = min(open_plants, key=lambda j: shipping[i][j])
            row = {j0: 1.0}
        elif abs(total - 1.0) > EPS:
            inv = 1.0 / total
            row = {j: v * inv for j, v in row.items()}
        assignments[str(i)] = {str(j): v for j, v in row.items()}
        ci = shipping[i]
        for j, v in row.items():
            ship_cost += ci[j] * v

    obj = ship_cost + sum(fixed_costs[j] for j in open_plants)
    solution = {
        "objective_value": float(obj),
        "open_plants": [int(j) for j in open_plants],
        "assignments": assignments,
    }
    return float(obj), solution


def run_mip(n, m, demands, fixed_costs, capacities, shipping,
            start_x, start_y, time_limit, t0, logger, best):
    """Build and solve the MIP; update ``best`` whenever an improved
    feasible solution is found (and log it)."""
    model = gp.Model("capacitated_facility_location")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x = model.addVars(n, m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVars(m, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="y")

    # Every customer's demand must be met in full.
    model.addConstrs((x.sum(i, "*") == 1.0 for i in range(n)), name="demand")

    # Capacity of open plants; closed plants are forced to zero throughput.
    model.addConstrs(
        (gp.LinExpr(demands, x.select("*", j)) <= capacities[j] * y[j]
         for j in range(m)),
        name="capacity",
    )

    # Disaggregated linking constraints strengthen the LP relaxation.
    if n * m <= LINK_CONSTRAINT_LIMIT:
        model.addConstrs(
            (x[i, j] <= y[j] for i in range(n) for j in range(m)),
            name="link",
        )

    obj = gp.LinExpr()
    for i in range(n):
        obj.add(gp.LinExpr(shipping[i], x.select(i, "*")))
    obj.add(gp.LinExpr(fixed_costs, y.select("*")))
    model.setObjective(obj, GRB.MINIMIZE)

    y_list = [y[j] for j in range(m)]
    flat_x = [x[i, j] for i in range(n) for j in range(m)]

    model.update()
    try:
        model.setAttr("Start", y_list, [float(v) for v in start_y])
        model.setAttr("Start", flat_x,
                      [start_x[i][j] for i in range(n) for j in range(m)])
    except Exception:
        pass

    def _callback(model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            try:
                yv = model.cbGetSolution(y_list)
                xv = model.cbGetSolution(flat_x)
            except Exception:
                return
            xr = [xv[i * m:(i + 1) * m] for i in range(n)]
            obj_val, sol = build_solution(
                xr, yv, n, m, demands, fixed_costs, shipping)
            if obj_val < best["obj"] - 1e-9:
                best["obj"] = obj_val
                best["sol"] = sol
                logger.log_solution(obj_val, sol)

    budget = time_limit - (time.time() - t0) - 0.5
    if budget <= 0.2:
        return
    model.Params.TimeLimit = budget
    model.optimize(_callback)

    if model.SolCount > 0:
        yv = model.getAttr("X", y_list)
        xv = model.getAttr("X", flat_x)
        xr = [xv[i * m:(i + 1) * m] for i in range(n)]
        obj_val, sol = build_solution(
            xr, yv, n, m, demands, fixed_costs, shipping)
        if obj_val < best["obj"] - 1e-9:
            best["obj"] = obj_val
            best["sol"] = sol
            if logger:
                logger.log_solution(obj_val, sol)


def main():
    args = parse_args()
    logger = (SolutionLogger(args.log_path, sense="minimize")
              if args.log_path else None)

    t0 = time.time()

    with open(args.instance_path, "r") as fh:
        data = json.load(fh)

    n = int(data["num_customers"])
    m = int(data["num_plants"])
    demands = [float(v) for v in data["customers"]["demands"]]
    fixed_costs = [float(v) for v in data["plants"]["fixed_costs"]]
    capacities = [float(v) for v in data["plants"]["capacities"]]
    shipping = [[float(v) for v in row] for row in data["shipping_costs"]]

    # Degenerate edge case: no customers or no candidate plants.
    if n == 0 or m == 0:
        sol = {
            "objective_value": 0.0,
            "open_plants": [],
            "assignments": {str(i): {} for i in range(n)},
        }
        if logger:
            logger.log_solution(0.0, sol)
        with open(args.solution_path, "w") as fh:
            json.dump(sol, fh)
        return

    # Guaranteed-feasible greedy solution: fallback and MIP warm start.
    gx, gy = greedy_feasible(n, m, demands, capacities, fixed_costs, shipping)
    g_obj, g_sol = build_solution(gx, gy, n, m, demands, fixed_costs, shipping)
    best = {"obj": g_obj, "sol": g_sol}
    if logger:
        logger.log_solution(g_obj, g_sol)

    # Solve the MIP with the remaining time budget.
    remaining = args.time_limit - (time.time() - t0)
    if remaining > 2.0:
        try:
            run_mip(n, m, demands, fixed_costs, capacities, shipping,
                    gx, gy, args.time_limit, t0, logger, best)
        except Exception:
            pass  # fall back to the best solution recorded so far

    with open(args.solution_path, "w") as fh:
        json.dump(best["sol"], fh)


if __name__ == "__main__":
    main()