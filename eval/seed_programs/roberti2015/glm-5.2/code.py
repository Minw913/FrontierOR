import json
import argparse
import sys
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def build_northwest_corner_solution(supply, demand, capacity, num_sources, num_sinks):
    """
    Construct a feasible solution using the Northwest Corner Rule.
    Returns flow matrix, arcs_used matrix, and objective value (computed later).
    """
    flow = [[0.0] * num_sinks for _ in range(num_sources)]
    supply_rem = list(supply)
    demand_rem = list(demand)
    i, j = 0, 0
    while i < num_sources and j < num_sinks:
        amount = min(supply_rem[i], demand_rem[j])
        # amount is always <= capacity[i][j] because capacity = min(orig_supply, orig_demand)
        flow[i][j] = float(amount)
        supply_rem[i] -= amount
        demand_rem[j] -= amount
        if supply_rem[i] == 0:
            i += 1
        if demand_rem[j] == 0:
            j += 1
    arcs_used = [[1 if flow[i][j] > 1e-6 else 0 for j in range(num_sinks)] for i in range(num_sources)]
    return flow, arcs_used


def compute_objective(flow, arcs_used, fixed_cost, variable_cost, num_sources, num_sinks):
    obj = 0.0
    for i in range(num_sources):
        for j in range(num_sinks):
            if arcs_used[i][j] == 1:
                obj += fixed_cost[i][j] + variable_cost[i][j] * flow[i][j]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Fixed-Charge Transportation Problem Solver")
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Load instance
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    m = instance["num_sources"]
    n = instance["num_sinks"]
    supply = instance["supply"]
    demand = instance["demand"]
    fixed_cost = instance["fixed_cost"]
    variable_cost = instance["variable_cost"]
    capacity = instance["capacity"]

    # Initialize logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Build a feasible solution via Northwest Corner Rule for warm start
    init_flow, init_arcs_used = build_northwest_corner_solution(supply, demand, capacity, m, n)
    init_obj = compute_objective(init_flow, init_arcs_used, fixed_cost, variable_cost, m, n)
    init_solution_dict = {
        "objective_value": init_obj,
        "flow": init_flow,
        "arcs_used": init_arcs_used
    }
    if logger:
        logger.log_solution(init_obj, init_solution_dict)

    # Build Gurobi model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("FCTP", env=env)
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = float(args.time_limit)

    # Variables
    x = model.addVars(m, n, lb=0.0, ub=[capacity[i][j] for i in range(m) for j in range(n)], name="x")
    y = model.addVars(m, n, vtype=GRB.BINARY, name="y")

    # Objective
    model.setObjective(
        gp.quicksum(fixed_cost[i][j] * y[i, j] for i in range(m) for j in range(n)) +
        gp.quicksum(variable_cost[i][j] * x[i, j] for i in range(m) for j in range(n)),
        GRB.MINIMIZE
    )

    # Supply constraints
    for i in range(m):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == supply[i])

    # Demand constraints
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == demand[j])

    # Capacity linking constraints
    for i in range(m):
        for j in range(n):
            model.addConstr(x[i, j] <= capacity[i][j] * y[i, j])

    # Attach variables and data for callback
    model._x = x
    model._y = y
    model._m = m
    model._n = n
    model._fixed_cost = fixed_cost
    model._variable_cost = variable_cost
    model._logger = logger

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_val = model.cbGetSolution(model._x)
            y_val = model.cbGetSolution(model._y)
            m_loc = model._m
            n_loc = model._n
            flow = [[0.0] * n_loc for _ in range(m_loc)]
            arcs_used = [[0] * n_loc for _ in range(m_loc)]
            for i in range(m_loc):
                for j in range(n_loc):
                    fv = x_val[i, j]
                    if fv is not None:
                        flow[i][j] = float(fv)
                        if fv > 1e-6:
                            arcs_used[i][j] = 1
            solution_dict = {
                "objective_value": obj,
                "flow": flow,
                "arcs_used": arcs_used
            }
            if model._logger:
                model._logger.log_solution(obj, solution_dict)

    # Set MIP start from Northwest Corner solution
    for i in range(m):
        for j in range(n):
            x[i, j].Start = init_flow[i][j]
            y[i, j].Start = init_arcs_used[i][j]

    # Optimize with callback
    start_time = time.time()
    model.optimize(callback)
    elapsed = time.time() - start_time

    # Determine best solution
    best_flow = init_flow
    best_arcs = init_arcs_used
    best_obj = init_obj

    if model.SolCount > 0:
        x_vals = model.getAttr("X", x)
        y_vals = model.getAttr("X", y)
        flow = [[0.0] * n for _ in range(m)]
        arcs_used = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                flow[i][j] = float(x_vals[i, j])
                if x_vals[i, j] > 1e-6:
                    arcs_used[i][j] = 1
        obj = model.ObjVal
        if obj < best_obj:
            best_obj = obj
            best_flow = flow
            best_arcs = arcs_used
    else:
        # Fallback: recompute objective from init solution to be safe
        pass

    # Write solution
    solution = {
        "objective_value": float(best_obj),
        "flow": best_flow,
        "arcs_used": best_arcs
    }
    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()