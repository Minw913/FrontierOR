import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    m = data["num_sources"]
    n = data["num_sinks"]
    supply = data["supply"]
    demand = data["demand"]
    fixed_cost = data["fixed_cost"]
    variable_cost = data["variable_cost"]
    capacity = data["capacity"]

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("fctp", env=env)
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    elapsed = time.time() - start_time
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    model.Params.TimeLimit = remaining

    # Variables
    x = model.addVars(m, n, lb=0.0, name="x")
    y = model.addVars(m, n, vtype=GRB.BINARY, name="y")

    # Capacity/linking constraints
    for i in range(m):
        for j in range(n):
            cap = min(capacity[i][j], supply[i], demand[j])
            model.addConstr(x[i, j] <= cap * y[i, j])

    # Supply constraints
    for i in range(m):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == supply[i])

    # Demand constraints
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == demand[j])

    # Objective
    model.setObjective(
        gp.quicksum(
            variable_cost[i][j] * x[i, j] + fixed_cost[i][j] * y[i, j]
            for i in range(m)
            for j in range(n)
        ),
        GRB.MINIMIZE,
    )

    best = {"obj": float("inf"), "flow": None, "used": None}

    def build_solution(obj, flow, used):
        return {
            "objective_value": obj,
            "flow": flow,
            "arcs_used": used,
        }

    def callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best["obj"] - 1e-9:
                xv = mdl.cbGetSolution(x)
                flow = [[max(0.0, float(xv[i, j])) for j in range(n)] for i in range(m)]
                used = [[1 if flow[i][j] > 1e-6 else 0 for j in range(n)] for i in range(m)]
                # recompute exact objective
                true_obj = sum(
                    variable_cost[i][j] * flow[i][j] + fixed_cost[i][j] * used[i][j]
                    for i in range(m)
                    for j in range(n)
                )
                best["obj"] = true_obj
                best["flow"] = flow
                best["used"] = used
                if logger:
                    logger.log_solution(true_obj, build_solution(true_obj, flow, used))

    model.optimize(callback)

    # If model has a solution, take the final one (may be better than last callback)
    if model.SolCount > 0:
        flow = [[max(0.0, float(x[i, j].X)) for j in range(n)] for i in range(m)]
        used = [[1 if flow[i][j] > 1e-6 else 0 for j in range(n)] for i in range(m)]
        true_obj = sum(
            variable_cost[i][j] * flow[i][j] + fixed_cost[i][j] * used[i][j]
            for i in range(m)
            for j in range(n)
        )
        if true_obj < best["obj"] - 1e-9 or best["flow"] is None:
            best["obj"] = true_obj
            best["flow"] = flow
            best["used"] = used
            if logger:
                logger.log_solution(true_obj, build_solution(true_obj, flow, used))

    # Fallback: greedy feasible solution if solver found nothing
    if best["flow"] is None:
        rem_supply = list(supply)
        rem_demand = list(demand)
        flow = [[0.0] * n for _ in range(m)]
        # Sort routes by variable cost + amortized fixed cost heuristic
        routes = sorted(
            ((variable_cost[i][j], i, j) for i in range(m) for j in range(n)),
            key=lambda t: t[0],
        )
        for _, i, j in routes:
            if rem_supply[i] <= 0 or rem_demand[j] <= 0:
                continue
            q = min(rem_supply[i], rem_demand[j], capacity[i][j])
            if q > 0:
                flow[i][j] += q
                rem_supply[i] -= q
                rem_demand[j] -= q
        # If capacities blocked full assignment, do simple fill ignoring extra
        # (capacity = min(supply, demand) so greedy always completes)
        used = [[1 if flow[i][j] > 1e-6 else 0 for j in range(n)] for i in range(m)]
        true_obj = sum(
            variable_cost[i][j] * flow[i][j] + fixed_cost[i][j] * used[i][j]
            for i in range(m)
            for j in range(n)
        )
        best["obj"] = true_obj
        best["flow"] = flow
        best["used"] = used
        if logger:
            logger.log_solution(true_obj, build_solution(true_obj, flow, used))

    solution = build_solution(best["obj"], best["flow"], best["used"])
    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()