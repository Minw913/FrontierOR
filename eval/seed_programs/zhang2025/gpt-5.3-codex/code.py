import argparse
import json
import time
from typing import List, Tuple, Dict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def deduplicate_duties(
    duties: List[List[int]], costs: List[float]
) -> Tuple[List[Tuple[int, ...]], List[float], List[int]]:
    """
    Deduplicate identical duty trip-lists, keeping the cheapest original index.
    Returns:
      unique_duties: list of tuple(trip_ids)
      unique_costs: corresponding minimized costs
      representative_original_idx: original duty index used for output projection
    """
    key_to_idx: Dict[Tuple[int, ...], int] = {}
    unique_duties: List[Tuple[int, ...]] = []
    unique_costs: List[float] = []
    rep_orig_idx: List[int] = []

    for j, d in enumerate(duties):
        key = tuple(int(t) for t in d)
        c = float(costs[j])
        if key not in key_to_idx:
            key_to_idx[key] = len(unique_duties)
            unique_duties.append(key)
            unique_costs.append(c)
            rep_orig_idx.append(j)
        else:
            k = key_to_idx[key]
            if c < unique_costs[k] - 1e-12:
                unique_costs[k] = c
                rep_orig_idx[k] = j

    return unique_duties, unique_costs, rep_orig_idx


def build_trip_to_columns(num_trips: int, duties: List[Tuple[int, ...]]) -> List[List[int]]:
    trip_to_cols = [[] for _ in range(num_trips)]
    for col, duty in enumerate(duties):
        for t in duty:
            if 0 <= t < num_trips:
                trip_to_cols[t].append(col)
    return trip_to_cols


def solve(instance: dict, time_limit: int, logger: SolutionLogger = None):
    start_time = time.time()

    duties_raw = instance["duties"]
    costs_raw = instance["costs"]
    num_trips = int(instance["parameters"]["num_trips_M"])

    if len(duties_raw) != len(costs_raw):
        raise ValueError("Length mismatch: duties vs costs.")

    # Deduplicate exact same duties to reduce model size
    duties, costs, rep_orig_idx = deduplicate_duties(duties_raw, costs_raw)
    n_cols = len(duties)

    trip_to_cols = build_trip_to_columns(num_trips, duties)

    # Quick feasibility check: every trip must appear at least once
    for t in range(num_trips):
        if not trip_to_cols[t]:
            return {
                "objective_value": 1e30,
                "selected_duties": []
            }

    model = gp.Model("driver_duty_set_partitioning")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.MIPFocus = 1  # prioritize finding feasible incumbents
    model.Params.TimeLimit = max(0.0, float(time_limit) - (time.time() - start_time))

    x = model.addVars(n_cols, vtype=GRB.BINARY, obj=costs, name="x")

    # Exact cover: each trip covered by exactly one selected duty
    for t in range(num_trips):
        model.addConstr(gp.quicksum(x[j] for j in trip_to_cols[t]) == 1, name=f"cover_{t}")

    model.ModelSense = GRB.MINIMIZE

    callback_best = {"obj": float("inf")}

    def mip_callback(m, where):
        if logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < callback_best["obj"]:
                callback_best["obj"] = obj
                logger.log(float(obj))

    model.optimize(mip_callback if logger else None)

    if model.SolCount > 0:
        selected_unique = [j for j in range(n_cols) if x[j].X > 0.5]
        selected_original = [rep_orig_idx[j] for j in selected_unique]
        obj = float(model.ObjVal)

        # Ensure final incumbent is logged at least once
        if logger and obj + 1e-9 < callback_best["obj"]:
            logger.log(obj)

        return {
            "objective_value": obj,
            "selected_duties": selected_original
        }

    # No feasible solution found within time limit / infeasible
    return {
        "objective_value": 1e30,
        "selected_duties": []
    }


def main():
    parser = argparse.ArgumentParser(description="Solve bus driver duty set-partitioning problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL log path")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)
    solution = solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()