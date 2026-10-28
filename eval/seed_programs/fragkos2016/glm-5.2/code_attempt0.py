#!/usr/bin/env python3
import argparse
import json
import sys
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    start_time = time.time()

    # Load instance
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    n_items = data["n_items"]
    n_periods = data["n_periods"]
    capacity = data["capacity"]
    initial_inventory = data["initial_inventory"]
    demands = data["demands"]
    setup_costs = data["setup_costs"]
    holding_costs = data["holding_costs"]
    variable_costs = data["variable_costs"]
    setup_times = data["setup_times"]
    variable_times = data["variable_times"]
    big_M = data["big_M"]

    # Build model
    model = gp.Model("CLSP")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # Decision variables
    x = model.addVars(n_items, n_periods, lb=0.0, name="x")
    y = model.addVars(n_items, n_periods, vtype=GRB.BINARY, name="y")
    s = model.addVars(n_items, n_periods + 1, lb=0.0, name="s")

    # Fix initial and final inventory
    for i in range(n_items):
        s[i, 0].lb = initial_inventory[i]
        s[i, 0].ub = initial_inventory[i]
        s[i, n_periods].lb = 0.0
        s[i, n_periods].ub = 0.0

    # Inventory balance constraints
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                s[i, t] + x[i, t] == demands[i][t] + s[i, t + 1],
                name=f"balance_{i}_{t}"
            )

    # Capacity constraints
    for t in range(n_periods):
        model.addConstr(
            gp.quicksum(
                setup_times[i][t] * y[i, t] + variable_times[i][t] * x[i, t]
                for i in range(n_items)
            ) <= capacity[t],
            name=f"capacity_{t}"
        )

    # Linking production to setup
    for i in range(n_items):
        for t in range(n_periods):
            model.addConstr(
                x[i, t] <= big_M[i][t] * y[i, t],
                name=f"link_{i}_{t}"
            )

    # Objective: setup + production + holding cost (on end-of-period inventory)
    model.setObjective(
        gp.quicksum(
            setup_costs[i][t] * y[i, t]
            for i in range(n_items) for t in range(n_periods)
        )
        + gp.quicksum(
            variable_costs[i][t] * x[i, t]
            for i in range(n_items) for t in range(n_periods)
        )
        + gp.quicksum(
            holding_costs[i][t] * s[i, t + 1]
            for i in range(n_items) for t in range(n_periods)
        ),
        GRB.MINIMIZE
    )

    # Callback to log incumbent solutions
    class IncumbentLogger:
        def __init__(self, logger, x_vars, y_vars, s_vars, n_items, n_periods):
            self.logger = logger
            self.n_items = n_items
            self.n_periods = n_periods
            self.last_obj = float('inf')
            self.flat_vars = []
            for i in range(n_items):
                for t in range(n_periods):
                    self.flat_vars.append(x_vars[i, t])
                for t in range(n_periods):
                    self.flat_vars.append(y_vars[i, t])
                for t in range(n_periods + 1):
                    self.flat_vars.append(s_vars[i, t])

        def __call__(self, model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                except Exception:
                    return
                if obj < self.last_obj - 1e-6:
                    self.last_obj = obj
                    vals = model.cbGetSolution(self.flat_vars)
                    idx = 0
                    x_vals = [[0.0] * self.n_periods for _ in range(self.n_items)]
                    y_vals = [[0] * self.n_periods for _ in range(self.n_items)]
                    s_vals = [[0.0] * (self.n_periods + 1) for _ in range(self.n_items)]
                    for i in range(self.n_items):
                        for t in range(self.n_periods):
                            x_vals[i][t] = vals[idx]
                            idx += 1
                        for t in range(self.n_periods):
                            y_vals[i][t] = int(round(vals[idx]))
                            idx += 1
                        for t in range(self.n_periods + 1):
                            s_vals[i][t] = vals[idx]
                            idx += 1
                    sol_dict = {
                        "objective_value": float(obj),
                        "x": x_vals,
                        "s": s_vals,
                        "y": y_vals
                    }
                    if self.logger:
                        self.logger.log_solution(obj, sol_dict)

    incumbent_logger = IncumbentLogger(logger, x, y, s, n_items, n_periods)

    # Adjust time limit for elapsed build time
    elapsed = time.time() - start_time
    remaining = max(1, args.time_limit - elapsed)
    model.Params.TimeLimit = remaining

    # Optimize
    if logger:
        model.optimize(incumbent_logger)
    else:
        model.optimize()

    # Check for feasible solution
    if model.SolCount == 0:
        print("No feasible solution found within the time limit.", file=sys.stderr)
        sys.exit(1)

    obj_val = model.ObjVal
    x_vals = [[x[i, t].X for t in range(n_periods)] for i in range(n_items)]
    y_vals = [[int(round(y[i, t].X)) for t in range(n_periods)] for i in range(n_items)]
    s_vals = [[s[i, t].X for t in range(n_periods + 1)] for i in range(n_items)]

    # Clean tiny negative values due to numerical tolerances
    for i in range(n_items):
        for t in range(n_periods):
            if x_vals[i][t] < 0.0:
                x_vals[i][t] = 0.0
        for t in range(n_periods + 1):
            if s_vals[i][t] < 0.0:
                s_vals[i][t] = 0.0

    solution = {
        "objective_value": float(obj_val),
        "x": x_vals,
        "s": s_vals,
        "y": y_vals
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    # Log final solution if it improved over the last logged incumbent
    if logger and obj_val < incumbent_logger.last_obj - 1e-6:
        logger.log_solution(obj_val, solution)


if __name__ == "__main__":
    main()