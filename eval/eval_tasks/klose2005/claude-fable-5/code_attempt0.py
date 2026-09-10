import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_solution_dict(obj, y_vals, x_vals, n_cust, n_plant, tol=1e-7):
    open_plants = [j for j in range(n_plant) if y_vals[j] > 0.5]
    assignments = {}
    for i in range(n_cust):
        inner = {}
        for j in open_plants:
            v = x_vals[i][j]
            if v > tol:
                inner[str(j)] = float(v)
        # normalize tiny numerical drift
        s = sum(inner.values())
        if s > 0 and abs(s - 1.0) > 1e-9:
            for k in inner:
                inner[k] = inner[k] / s
        assignments[str(i)] = inner
    return {
        "objective_value": float(obj),
        "open_plants": open_plants,
        "assignments": assignments,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n_cust = data["num_customers"]
    n_plant = data["num_plants"]
    demands = data["customers"]["demands"]
    fixed_costs = data["plants"]["fixed_costs"]
    capacities = data["plants"]["capacities"]
    ship = data["shipping_costs"]  # [n_cust][n_plant]

    m = gp.Model("cflp")
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    elapsed = time.time() - start_time
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    m.Params.TimeLimit = remaining

    y = m.addVars(n_plant, vtype=GRB.BINARY, name="y")
    x = m.addVars(n_cust, n_plant, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    # Demand satisfaction
    for i in range(n_cust):
        m.addConstr(gp.quicksum(x[i, j] for j in range(n_plant)) == 1.0)

    # Capacity constraints
    for j in range(n_plant):
        m.addConstr(
            gp.quicksum(demands[i] * x[i, j] for i in range(n_cust))
            <= capacities[j] * y[j]
        )

    # Strong variable upper bounds
    for i in range(n_cust):
        for j in range(n_plant):
            m.addConstr(x[i, j] <= y[j])

    m.setObjective(
        gp.quicksum(fixed_costs[j] * y[j] for j in range(n_plant))
        + gp.quicksum(ship[i][j] * x[i, j] for i in range(n_cust) for j in range(n_plant)),
        GRB.MINIMIZE,
    )

    best = {"sol": None, "obj": float("inf")}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best["obj"] - 1e-9:
                y_vals = model.cbGetSolution([y[j] for j in range(n_plant)])
                x_flat = model.cbGetSolution(
                    [x[i, j] for i in range(n_cust) for j in range(n_plant)]
                )
                x_vals = [
                    x_flat[i * n_plant:(i + 1) * n_plant] for i in range(n_cust)
                ]
                sol = build_solution_dict(obj, y_vals, x_vals, n_cust, n_plant)
                best["obj"] = obj
                best["sol"] = sol
                if logger:
                    logger.log_solution(obj, sol)

    m.optimize(callback)

    # Prefer the final model solution if available (may match or beat callback incumbent)
    if m.SolCount > 0:
        y_vals = [y[j].X for j in range(n_plant)]
        x_vals = [[x[i, j].X for j in range(n_plant)] for i in range(n_cust)]
        sol = build_solution_dict(m.ObjVal, y_vals, x_vals, n_cust, n_plant)
        if m.ObjVal < best["obj"] - 1e-9:
            if logger:
                logger.log_solution(m.ObjVal, sol)
            best["obj"] = m.ObjVal
            best["sol"] = sol
        elif best["sol"] is None:
            best["sol"] = sol

    if best["sol"] is None:
        # Fallback: open all plants, greedy fractional assignment by cheapest shipping
        rem_cap = list(capacities)
        assignments = {}
        total_ship = 0.0
        for i in range(n_cust):
            d = demands[i]
            order = sorted(range(n_plant), key=lambda j: ship[i][j])
            inner = {}
            left = float(d)
            for j in order:
                if left <= 1e-12:
                    break
                take = min(left, rem_cap[j])
                if take > 1e-12:
                    frac = take / d if d > 0 else 0.0
                    inner[str(j)] = frac
                    total_ship += ship[i][j] * frac
                    rem_cap[j] -= take
                    left -= take
            if d == 0 and not inner:
                inner[str(order[0])] = 1.0
            assignments[str(i)] = inner
        open_plants = list(range(n_plant))
        obj = total_ship + sum(fixed_costs)
        best["sol"] = {
            "objective_value": float(obj),
            "open_plants": open_plants,
            "assignments": assignments,
        }
        if logger:
            logger.log_solution(obj, best["sol"])

    with open(args.solution_path, "w") as f:
        json.dump(best["sol"], f, indent=2)


if __name__ == "__main__":
    main()