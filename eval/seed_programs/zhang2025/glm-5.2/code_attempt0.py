import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description="Solve the set partitioning problem for driver duties.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to the output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the JSONL file for logging intermediate solutions.")
    
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_trips = instance["parameters"]["num_trips_M"]
    duties = instance["duties"]
    costs = instance["costs"]

    # Deduplicate duties: if multiple duties cover the exact same set of trips,
    # we only need to keep the one with the lowest cost.
    duty_map = {}
    for i, duty in enumerate(duties):
        key = tuple(sorted(duty))
        if key not in duty_map or costs[i] < duty_map[key][0]:
            duty_map[key] = (costs[i], i)

    unique_duties = []
    unique_costs = []
    original_indices = []
    for key, (cost, idx) in duty_map.items():
        unique_duties.append(list(key))
        unique_costs.append(cost)
        original_indices.append(idx)

    num_duties = len(unique_duties)

    # Initialize Gurobi model
    model = gp.Model("set_partitioning")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = float(args.time_limit)

    # Decision variables
    x = model.addVars(num_duties, vtype=GRB.BINARY, obj=unique_costs, name="x")

    # Constraints: each trip must be covered exactly once
    trip_to_duties = [[] for _ in range(num_trips)]
    for j, duty in enumerate(unique_duties):
        for trip in duty:
            if 0 <= trip < num_trips:
                trip_to_duties[trip].append(j)

    for i in range(num_trips):
        model.addConstr(gp.quicksum(x[j] for j in trip_to_duties[i]) == 1, name=f"cover_{i}")

    # Callback to log intermediate incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            x_vals = model.cbGet(GRB.Callback.MIPSOL_X)
            selected = [j for j, v in enumerate(x_vals) if v > 0.5]
            selected_original = sorted([original_indices[j] for j in selected])
            sol_dict = {
                "objective_value": obj,
                "selected_duties": selected_original
            }
            if logger:
                logger.log_solution(obj, sol_dict)

    # Optimize
    model.optimize(callback)

    # Prepare final solution output
    solution_dict = {
        "objective_value": 0.0,
        "selected_duties": []
    }

    if model.SolCount > 0:
        obj_val = float(model.ObjVal)
        selected = [j for j in range(num_duties) if x[j].X > 0.5]
        selected_original = sorted([original_indices[j] for j in selected])
        solution_dict = {
            "objective_value": obj_val,
            "selected_duties": selected_original
        }

    # Write the final solution to the specified output file
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)

if __name__ == "__main__":
    main()