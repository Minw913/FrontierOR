import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve_knapsack(instance_path, solution_path, time_limit, log_path):
    # 1. Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)

    # 2. Setup solution logger
    logger = SolutionLogger(log_path, sense="maximize") if log_path and SolutionLogger else None

    # 3. Create Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)  # Suppress standard output for cleaner logs
    env.start()
    
    model = gp.Model("Knapsack", env=env)
    
    # 4. Enforce requirements
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)  # Constrained to a single CPU core
    model.ModelSense = GRB.MAXIMIZE

    # 5. Extract item features and parameters
    items_data = {item["id"]: item for item in data.get("items", [])}
    ids = list(items_data.keys())

    profits = {i: items_data[i]["profit"] for i in ids}
    weights = {i: items_data[i]["weight"] for i in ids}
    capacity = data.get("capacity", 0)

    # 6. Define decision variables
    # x[i] = 1 if item i is selected, 0 otherwise
    x = model.addVars(ids, vtype=GRB.BINARY, obj=profits, name="x")

    # 7. Add constraints
    # Total accumulated weight of the selected items must not exceed knapsack capacity
    model.addConstr(x.prod(weights) <= capacity, name="CapacityConstraint")

    # 8. Define callback to log incumbent solutions
    def logging_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # 9. Solve the model
    model.optimize(logging_callback)

    # 10. Extract optimal or best feasible solution
    if model.SolCount > 0:
        obj_val = int(round(model.ObjVal))
        # Use threshold to mitigate any minor floating point precision artifacts
        selected_items = [i for i in ids if x[i].X > 0.5]
    else:
        obj_val = 0
        selected_items = []

    # 11. Prepare output strictly following the schema
    solution = {
        "objective_value": obj_val,
        "selected_items": selected_items
    }

    # 12. Write solution JSON
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the 0-1 Knapsack Optimization Problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance data JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where the output solution JSON will be stored.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum execution time limits in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to a JSONL file for logging intermediate solutions.")
    args = parser.parse_args()

    solve_knapsack(args.instance_path, args.solution_path, args.time_limit, args.log_path)