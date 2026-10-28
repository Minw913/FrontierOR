import argparse
import json
import math
from typing import Any, Dict, List, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


class IncumbentCallback:
    def __init__(self, logger: Optional[SolutionLogger]):
        self.logger = logger
        self.best = float("inf")

    def __call__(self, model, where):
        if self.logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < self.best - 1e-9:
                self.best = obj
                self.logger.log(float(obj))


def load_instance(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def compute_big_m(
    num_items: int,
    num_periods: int,
    demand: List[List[int]],
    setup_time: List[int],
    variable_time: List[float],
    capacity: List[int],
) -> List[List[float]]:
    # Remaining demand from period t to end
    rem_dem = [[0.0] * num_periods for _ in range(num_items)]
    for i in range(num_items):
        running = 0.0
        for t in range(num_periods - 1, -1, -1):
            running += demand[i][t]
            rem_dem[i][t] = running

    M = [[0.0] * num_periods for _ in range(num_items)]
    for i in range(num_items):
        for t in range(num_periods):
            # Capacity-based upper bound if setup is done
            if capacity[t] < setup_time[i]:
                cap_bound = 0.0
            else:
                if variable_time[i] > 0:
                    cap_bound = (capacity[t] - setup_time[i]) / variable_time[i]
                else:
                    # Degenerate case: zero variable time -> limited only by demand
                    cap_bound = rem_dem[i][t]
            M[i][t] = max(0.0, min(cap_bound, rem_dem[i][t]))
    return M


def build_and_solve(instance: Dict[str, Any], time_limit: int, logger: Optional[SolutionLogger]):
    n = int(instance["num_items"])
    T = int(instance["num_periods"])
    demand = instance["demand"]
    setup_cost = instance["setup_cost"]
    holding_cost = instance["holding_cost"]
    variable_cost = instance["variable_cost"]
    setup_time = instance["setup_time"]
    variable_time = instance["variable_time"]
    capacity = instance["capacity"]
    initial_inventory_cost = instance["initial_inventory_cost"]

    M = compute_big_m(n, T, demand, setup_time, variable_time, capacity)

    model = gp.Model("single_machine_lot_sizing")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(1, int(time_limit))
    model.Params.Threads = 1

    # Variables
    x = model.addVars(n, T, lb=0.0, vtype=GRB.CONTINUOUS, name="prod")
    y = model.addVars(n, T, vtype=GRB.BINARY, name="setup")
    I = model.addVars(n, T + 1, lb=0.0, vtype=GRB.CONTINUOUS, name="inv")

    # Final inventory must be zero
    for i in range(n):
        model.addConstr(I[i, T] == 0.0, name=f"final_inv_{i}")

    # Inventory balance
    for i in range(n):
        for t in range(T):
            model.addConstr(
                I[i, t] + x[i, t] == float(demand[i][t]) + I[i, t + 1],
                name=f"balance_{i}_{t}",
            )

    # Production only if setup and per-item period upper bound
    for i in range(n):
        for t in range(T):
            model.addConstr(x[i, t] <= M[i][t] * y[i, t], name=f"link_{i}_{t}")

    # Capacity constraints per period
    for t in range(T):
        model.addConstr(
            gp.quicksum(setup_time[i] * y[i, t] + variable_time[i] * x[i, t] for i in range(n))
            <= capacity[t],
            name=f"capacity_{t}",
        )

    # Objective: initial inventory + setup + variable production + holding
    obj = gp.quicksum(initial_inventory_cost[i] * I[i, 0] for i in range(n))
    obj += gp.quicksum(setup_cost[i] * y[i, t] for i in range(n) for t in range(T))
    obj += gp.quicksum(variable_cost[i] * x[i, t] for i in range(n) for t in range(T))
    obj += gp.quicksum(holding_cost[i] * I[i, t + 1] for i in range(n) for t in range(T))

    model.setObjective(obj, GRB.MINIMIZE)

    cb = IncumbentCallback(logger)
    model.optimize(cb)

    return model, x, y, I


def fallback_feasible(instance: Dict[str, Any]) -> Dict[str, Any]:
    # Always feasible: satisfy all demand with initial inventory only.
    n = int(instance["num_items"])
    T = int(instance["num_periods"])
    demand = instance["demand"]
    setup_cost = instance["setup_cost"]
    holding_cost = instance["holding_cost"]
    variable_cost = instance["variable_cost"]
    initial_inventory_cost = instance["initial_inventory_cost"]

    production = [[0.0 for _ in range(T)] for _ in range(n)]
    setup = [[0 for _ in range(T)] for _ in range(n)]
    inventory = [[0.0 for _ in range(T + 1)] for _ in range(n)]

    objective = 0.0
    for i in range(n):
        total_dem = float(sum(demand[i]))
        inventory[i][0] = total_dem
        objective += initial_inventory_cost[i] * total_dem
        inv = total_dem
        for t in range(T):
            inv = inv - float(demand[i][t])
            inventory[i][t + 1] = inv
            objective += holding_cost[i] * inv
            # production/setup remain zero

    # setup and variable costs are zero in fallback
    return {
        "objective_value": float(objective),
        "production": production,
        "setup": setup,
        "inventory": inventory,
    }


def extract_solution(model, x, y, I, instance: Dict[str, Any]) -> Dict[str, Any]:
    n = int(instance["num_items"])
    T = int(instance["num_periods"])

    production = [[0.0 for _ in range(T)] for _ in range(n)]
    setup = [[0 for _ in range(T)] for _ in range(n)]
    inventory = [[0.0 for _ in range(T + 1)] for _ in range(n)]

    for i in range(n):
        for t in range(T):
            production[i][t] = float(x[i, t].X)
            setup[i][t] = int(round(y[i, t].X))
        for t in range(T + 1):
            inventory[i][t] = float(I[i, t].X)

    return {
        "objective_value": float(model.ObjVal),
        "production": production,
        "setup": setup,
        "inventory": inventory,
    }


def main():
    parser = argparse.ArgumentParser(description="Solve single-machine multi-item lot sizing with initial inventory.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL path for incumbent logging.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)
    model, x, y, I = build_and_solve(instance, args.time_limit, logger)

    if model.SolCount > 0:
        solution = extract_solution(model, x, y, I, instance)
    else:
        # Extremely unlikely here, but ensure schema-compliant output.
        solution = fallback_feasible(instance)
        if logger:
            logger.log(solution["objective_value"])

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()