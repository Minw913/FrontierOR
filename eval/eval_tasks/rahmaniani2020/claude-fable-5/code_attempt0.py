import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    start_time = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    nF = data["num_facilities"]
    nC = data["num_customers"]
    nS = data["num_scenarios"]

    fixed_costs = data["facilities"]["fixed_costs"]
    capacities = data["facilities"]["capacities"]
    routing_costs = data["routing_costs"]
    probs = data["scenarios"]["probabilities"]
    demands = data["scenarios"]["demands"]

    max_total_demand = data.get("max_total_demand", None)
    if max_total_demand is None:
        max_total_demand = max(sum(demands[s]) for s in range(nS))

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("sfl", env=env)
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    # Decision variables
    y = m.addVars(nF, vtype=GRB.BINARY, name="y")
    # x[i, j, s] flow from facility i to customer j in scenario s
    x = m.addVars(nF, nC, nS, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

    # Objective: fixed costs + expected routing cost
    obj = gp.quicksum(fixed_costs[i] * y[i] for i in range(nF))
    obj += gp.quicksum(
        probs[s] * routing_costs[i][j] * x[i, j, s]
        for i in range(nF) for j in range(nC) for s in range(nS)
    )
    m.setObjective(obj, GRB.MINIMIZE)

    # Demand satisfaction in each scenario
    for s in range(nS):
        for j in range(nC):
            m.addConstr(
                gp.quicksum(x[i, j, s] for i in range(nF)) >= demands[s][j],
                name=f"dem_{s}_{j}",
            )

    # Capacity constraints
    for s in range(nS):
        for i in range(nF):
            m.addConstr(
                gp.quicksum(x[i, j, s] for j in range(nC)) <= capacities[i] * y[i],
                name=f"cap_{s}_{i}",
            )

    # Aggregate capacity must cover max total demand
    m.addConstr(
        gp.quicksum(capacities[i] * y[i] for i in range(nF)) >= max_total_demand,
        name="agg_cap",
    )

    # Time limit (leave a small buffer for writing output)
    elapsed = time.time() - start_time
    remaining = max(1.0, args.time_limit - elapsed - 2.0)
    m.Params.TimeLimit = remaining

    best = {"obj": None, "y": None}

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if best["obj"] is None or obj_val < best["obj"] - 1e-9:
                y_vals = model.cbGetSolution([y[i] for i in range(nF)])
                y_int = [1 if v > 0.5 else 0 for v in y_vals]
                best["obj"] = obj_val
                best["y"] = y_int
                if logger:
                    logger.log_solution(obj_val, {"objective_value": obj_val, "y": y_int})

    m.optimize(callback)

    if m.SolCount > 0:
        obj_val = m.ObjVal
        y_int = [1 if y[i].X > 0.5 else 0 for i in range(nF)]
        if best["obj"] is None or obj_val < best["obj"] - 1e-9:
            best["obj"] = obj_val
            best["y"] = y_int
            if logger:
                logger.log_solution(obj_val, {"objective_value": obj_val, "y": y_int})

    if best["obj"] is None:
        # Fallback: open all facilities and solve LP for flows (should be feasible if instance is)
        for i in range(nF):
            y[i].LB = 1.0
        m.Params.TimeLimit = max(1.0, args.time_limit - (time.time() - start_time) - 1.0)
        m.optimize()
        if m.SolCount > 0:
            best["obj"] = m.ObjVal
            best["y"] = [1] * nF
            if logger:
                logger.log_solution(best["obj"], {"objective_value": best["obj"], "y": best["y"]})
        else:
            best["obj"] = float(sum(fixed_costs))
            best["y"] = [1] * nF

    solution = {
        "objective_value": float(best["obj"]),
        "y": best["y"],
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()