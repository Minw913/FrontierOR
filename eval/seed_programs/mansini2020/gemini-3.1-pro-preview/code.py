import argparse
import json
import sys
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description="Solve Multidimensional Multiple-Choice Knapsack Problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to write the logging JSONL file.")
    args = parser.parse_args()

    # Initialize solution logger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    # Load problem instance
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    capacities = data["capacities"]
    groups = data["groups"]
    m_resources = data["parameters"]["m_resources"]

    # Initialize Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)  # Suppress internal Gurobi standard output for cleanly running environments
    env.start()
    
    model = gp.Model("mmkp", env=env)
    
    # Configure optimizer parameters
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Threads", 1)  # Single core environment requirement

    # For efficient constraint construction
    cap_exprs = [gp.LinExpr() for _ in range(m_resources)]
    x = {}

    # Define variables and efficiently build linear expressions
    for g in groups:
        g_id = g["group_id"]
        choice_expr = gp.LinExpr()
        
        for item in g["items"]:
            i_id = item["item_id"]
            profit = item["profit"]
            weights = item["weights"]
            
            var = model.addVar(vtype=GRB.BINARY, obj=profit, name=f"x_{g_id}_{i_id}")
            x[g_id, i_id] = var
            choice_expr.addTerms(1.0, var)
            
            # Add to capacity expressions if weight is not zero
            for r in range(m_resources):
                w = weights[r]
                if w != 0:
                    cap_exprs[r].addTerms(w, var)
                    
        # Choice constraint: exactly one item selected per group
        model.addConstr(choice_expr == 1, name=f"choice_{g_id}")

    # Capacity constraints: selected items' resource usage must not exceed available capacities
    for r in range(m_resources):
        model.addConstr(cap_exprs[r] <= capacities[r], name=f"cap_{r}")

    # Objective is to maximize sum of profits
    model.ModelSense = GRB.MAXIMIZE

    # Callback mechanism to log every improved feasible solution found during optimization
    def optimization_callback(cb_model, where):
        if where == GRB.Callback.MIPSOL:
            obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Solve the problem
    model.optimize(optimization_callback)

    # Output generation
    if model.SolCount > 0:
        selected_items = []
        total_profit = 0

        for g in groups:
            g_id = g["group_id"]
            for item in g["items"]:
                i_id = item["item_id"]
                # A chosen binary variable will evaluate essentially to 1.0
                if x[g_id, i_id].X > 0.5:
                    selected_items.append({
                        "group_id": g_id,
                        "item_id": i_id,
                        "profit": item["profit"],
                        "weights": item["weights"]
                    })
                    total_profit += item["profit"]
                    break  # Exactly one selection per group
        
        # Format the solution strictly matching the expected schema
        solution = {
            "objective_value": float(total_profit),
            "selected_items": selected_items
        }

        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)

    else:
        # In a case without feasibility in the provided time limit
        print("No feasible solution was found within the time limit.", file=sys.stderr)

if __name__ == "__main__":
    main()