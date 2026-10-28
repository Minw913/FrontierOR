import argparse
import json
from typing import Any, Dict, List, Optional

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_row_to_columns(num_rows: int, columns: List[List[int]]) -> List[List[int]]:
    row_to_cols = [[] for _ in range(num_rows)]
    for j, covered_rows in enumerate(columns):
        for r in covered_rows:
            if 0 <= r < num_rows:
                row_to_cols[r].append(j)
    return row_to_cols


def solve_instance(instance: Dict[str, Any], time_limit: int, logger: Optional[SolutionLogger]):
    dims = instance["dimensions"]
    num_rows = int(dims["num_rows"])
    num_cols = int(dims["num_cols"])

    costs = instance["cost_vector"]
    columns = instance["constraint_matrix_A"]["columns"]

    if len(costs) != num_cols:
        raise ValueError("cost_vector length does not match num_cols.")
    if len(columns) != num_cols:
        raise ValueError("constraint_matrix_A.columns length does not match num_cols.")

    row_to_cols = build_row_to_columns(num_rows, columns)

    model = gp.Model("airline_crew_rotation_set_partitioning")
    model.Params.TimeLimit = int(time_limit)
    model.Params.Threads = 1
    model.Params.OutputFlag = 0

    x = model.addVars(num_cols, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(float(costs[j]) * x[j] for j in range(num_cols)), GRB.MINIMIZE)

    # Exact cover constraints: each flight leg covered exactly once
    for r in range(num_rows):
        model.addConstr(gp.quicksum(x[j] for j in row_to_cols[r]) == 1, name=f"cover_{r}")

    # Optional base constraints
    if bool(instance.get("has_base_constraints", False)):
        bc = instance.get("base_constraints", {})
        D_rows = bc.get("D_matrix", {}).get("rows", [])
        lower = bc.get("lower_bounds_d1", [])
        upper = bc.get("upper_bounds_d2", [])

        n_base_cons = min(len(D_rows), len(lower), len(upper))
        for b in range(n_base_cons):
            coeffs = D_rows[b]
            if len(coeffs) != num_cols:
                raise ValueError(f"D_matrix row {b} length does not match num_cols.")

            nz_idx = [j for j, v in enumerate(coeffs) if abs(float(v)) > 1e-12]
            expr = gp.quicksum(float(coeffs[j]) * x[j] for j in nz_idx)

            model.addConstr(expr >= float(lower[b]), name=f"base_lb_{b}")
            model.addConstr(expr <= float(upper[b]), name=f"base_ub_{b}")

    # Incumbent logging callback
    best_logged = {"obj": float("inf")}

    def cb(m: gp.Model, where: int):
        if where == GRB.Callback.MIPSOL and logger is not None:
            obj = float(m.cbGet(GRB.Callback.MIPSOL_OBJ))
            if obj + 1e-9 < best_logged["obj"]:
                best_logged["obj"] = obj
                logger.log(obj)

    model.optimize(cb)

    # Extract best feasible solution if available
    variable_values = {str(j): 0.0 for j in range(num_cols)}
    selected_rotations: List[int] = []
    objective_value = 1e100  # fallback if no feasible solution found

    if model.SolCount > 0:
        vals = model.getAttr("X", x)
        for j in range(num_cols):
            v = 1.0 if vals[j] > 0.5 else 0.0
            variable_values[str(j)] = v
            if v > 0.5:
                selected_rotations.append(j)
        objective_value = float(model.ObjVal)

    return {
        "objective_value": objective_value,
        "selected_rotations": selected_rotations,
        "variable_values": variable_values,
    }


def main():
    parser = argparse.ArgumentParser(description="Solve airline crew rotation set partitioning problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to input instance JSON.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Optional path to JSONL convergence log.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance = load_instance(args.instance_path)
    solution = solve_instance(instance, args.time_limit, logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()