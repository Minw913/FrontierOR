import argparse
import json
import time
from typing import Any, Dict, List

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def load_instance(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_and_solve(instance: Dict[str, Any], time_limit: int, logger: SolutionLogger = None):
    t0 = time.time()

    # Read core dimensions
    n_fac = int(instance["num_facilities"])
    n_cus = int(instance["num_customers"])

    scen_block = instance["scenarios"]
    n_scen = int(scen_block.get("num_scenarios", instance["num_scenarios"]))

    fixed_costs: List[float] = instance["facilities"]["fixed_costs"]
    capacities: List[float] = instance["facilities"]["capacities"]
    routing_costs: List[List[float]] = instance["routing_costs"]
    probabilities: List[float] = scen_block["probabilities"]
    demands: List[List[float]] = scen_block["demands"]

    max_total_demand = instance.get("max_total_demand", max(sum(demands[s]) for s in range(n_scen)))

    model = gp.Model("stochastic_facility_location")

    # Single-core as required
    model.Params.Threads = 1
    model.Params.MIPFocus = 1  # emphasize finding good feasible solutions early
    model.Params.OutputFlag = 0

    # Decision variables
    y = model.addVars(n_fac, vtype=GRB.BINARY, name="y")
    x = model.addVars(n_fac, n_cus, n_scen, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

    # Objective: fixed opening + expected routing costs
    fixed_term = gp.quicksum(fixed_costs[i] * y[i] for i in range(n_fac))
    routing_term = gp.quicksum(
        probabilities[s] * routing_costs[i][j] * x[i, j, s]
        for i in range(n_fac)
        for j in range(n_cus)
        for s in range(n_scen)
    )
    model.setObjective(fixed_term + routing_term, GRB.MINIMIZE)

    # Demand satisfaction per customer and scenario
    for s in range(n_scen):
        for j in range(n_cus):
            model.addConstr(
                gp.quicksum(x[i, j, s] for i in range(n_fac)) >= demands[s][j],
                name=f"demand_s{s}_c{j}"
            )

    # Facility capacity per scenario, activated by opening decision
    for s in range(n_scen):
        for i in range(n_fac):
            model.addConstr(
                gp.quicksum(x[i, j, s] for j in range(n_cus)) <= capacities[i] * y[i],
                name=f"cap_f{i}_s{s}"
            )

    # Global robustness capacity requirement
    model.addConstr(
        gp.quicksum(capacities[i] * y[i] for i in range(n_fac)) >= max_total_demand,
        name="total_capacity_requirement"
    )

    # Simple warm start: open cheapest-capacity-ratio facilities until max demand covered
    order = sorted(
        range(n_fac),
        key=lambda i: (fixed_costs[i] / capacities[i]) if capacities[i] > 0 else float("inf")
    )
    start_y = [0] * n_fac
    cap_acc = 0.0
    for i in order:
        if cap_acc >= max_total_demand:
            break
        if capacities[i] > 0:
            start_y[i] = 1
            cap_acc += capacities[i]
    # If still insufficient (e.g., zero capacities in front), open all with positive capacity
    if cap_acc < max_total_demand:
        for i in range(n_fac):
            if capacities[i] > 0 and start_y[i] == 0:
                start_y[i] = 1
                cap_acc += capacities[i]
                if cap_acc >= max_total_demand:
                    break
    for i in range(n_fac):
        y[i].Start = start_y[i]

    # Remaining runtime budget for solver
    elapsed_build = time.time() - t0
    remaining = max(1.0, float(time_limit) - elapsed_build)
    model.Params.TimeLimit = remaining

    best_logged = [float("inf")]

    def callback(cb_model, where):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged[0]:
                best_logged[0] = float(obj)
                logger.log(float(obj))

    model.optimize(callback)

    if model.SolCount > 0:
        obj_val = float(model.ObjVal)
        y_sol = [int(round(y[i].X)) for i in range(n_fac)]
        if logger is not None and obj_val + 1e-9 < best_logged[0]:
            logger.log(obj_val)
    else:
        # No feasible solution found
        obj_val = 1e30
        y_sol = [0] * n_fac

    return obj_val, y_sol


def main():
    parser = argparse.ArgumentParser(description="Solve stochastic capacitated facility location problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional path to JSONL convergence log.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)
    obj_val, y_sol = build_and_solve(instance, args.time_limit, logger)

    solution = {
        "objective_value": float(obj_val),
        "y": y_sol
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()