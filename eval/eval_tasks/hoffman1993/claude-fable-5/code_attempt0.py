import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


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
        inst = json.load(f)

    num_rows = inst["dimensions"]["num_rows"]
    num_cols = inst["dimensions"]["num_cols"]
    costs = inst["cost_vector"]
    cols = inst["constraint_matrix_A"]["columns"]

    has_base = inst.get("has_base_constraints", False)
    base = inst.get("base_constraints", None) if has_base else None

    model = gp.Model("crew_pairing")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    remaining = max(1.0, args.time_limit - (time.time() - start_time) - 2.0)
    model.Params.TimeLimit = remaining

    x = model.addVars(num_cols, vtype=GRB.BINARY, name="x")
    model.setObjective(
        gp.quicksum(costs[j] * x[j] for j in range(num_cols)), GRB.MINIMIZE
    )

    # Build row -> columns covering it
    row_cols = [[] for _ in range(num_rows)]
    for j in range(num_cols):
        for i in cols[j]:
            row_cols[i].append(j)

    for i in range(num_rows):
        model.addConstr(
            gp.quicksum(x[j] for j in row_cols[i]) == 1, name=f"cover_{i}"
        )

    if base is not None:
        D = base["D_matrix"]["rows"]
        d1 = base["lower_bounds_d1"]
        d2 = base["upper_bounds_d2"]
        nb = base.get("num_bases", len(D))
        for b in range(nb):
            row = D[b]
            expr = gp.quicksum(row[j] * x[j] for j in range(num_cols) if row[j] != 0)
            model.addConstr(expr >= d1[b], name=f"base_lb_{b}")
            model.addConstr(expr <= d2[b], name=f"base_ub_{b}")

    best_holder = {"obj": None, "sol": None}

    def make_solution_dict(obj_val, vals):
        selected = [j for j in range(num_cols) if vals[j] > 0.5]
        var_values = {str(j): (1.0 if vals[j] > 0.5 else 0.0) for j in range(num_cols)}
        return {
            "objective_value": float(obj_val),
            "selected_rotations": selected,
            "variable_values": var_values,
        }

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if best_holder["obj"] is None or obj < best_holder["obj"] - 1e-9:
                vals = m.cbGetSolution([x[j] for j in range(num_cols)])
                sol = make_solution_dict(obj, vals)
                best_holder["obj"] = obj
                best_holder["sol"] = sol
                if logger:
                    logger.log_solution(float(obj), sol)

    model.optimize(callback)

    # Final solution extraction
    sol = None
    if model.SolCount > 0:
        vals = [x[j].X for j in range(num_cols)]
        obj = model.ObjVal
        sol = make_solution_dict(obj, vals)
        if best_holder["obj"] is None or obj < best_holder["obj"] - 1e-9:
            if logger:
                logger.log_solution(float(obj), sol)
    elif best_holder["sol"] is not None:
        sol = best_holder["sol"]

    if sol is None:
        # No feasible solution found; write an empty placeholder
        sol = {
            "objective_value": float("inf"),
            "selected_rotations": [],
            "variable_values": {str(j): 0.0 for j in range(num_cols)},
        }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()