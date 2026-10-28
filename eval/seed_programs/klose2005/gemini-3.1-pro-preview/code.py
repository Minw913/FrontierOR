import json
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Solve Capacitated Facility Location Problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, default=3600, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the log file.")
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_customers = instance["num_customers"]
    num_plants = instance["num_plants"]
    
    demands = instance["customers"]["demands"]
    fixed_costs = instance["plants"]["fixed_costs"]
    capacities = instance["plants"]["capacities"]
    shipping_costs = instance["shipping_costs"]

    # Initialize Logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    # Setup Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    model = gp.Model("CFLP", env=env)
    model.setParam("Threads", 1)  # Execute on a single core
    if args.time_limit:
        model.setParam("TimeLimit", args.time_limit)

    # Variables
    # y[j]: 1 if plant j is open, 0 otherwise
    y = model.addVars(num_plants, vtype=GRB.BINARY, name="y")
    # x[i, j]: fraction of customer i's demand supplied by plant j
    x = model.addVars(num_customers, num_plants, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    # Objective Function
    shipping_dict = {(i, j): shipping_costs[i][j] for i in range(num_customers) for j in range(num_plants)}
    fixed_cost_dict = {j: fixed_costs[j] for j in range(num_plants)}
    model.setObjective(y.prod(fixed_cost_dict) + x.prod(shipping_dict), GRB.MINIMIZE)

    # Constraints
    # 1. Demand completion constraints
    model.addConstrs((x.sum(i, '*') == 1.0 for i in range(num_customers)), name="demand")

    # 2. Capacity constraints for each plant
    model.addConstrs(
        (gp.quicksum(demands[i] * x[i, j] for i in range(num_customers)) <= capacities[j] * y[j] 
         for j in range(num_plants)),
        name="capacity"
    )

    # 3. Strong formulation bounds (tighter LP relaxation for faster convergence)
    model.addConstrs(
        (x[i, j] <= min(1.0, capacities[j] / demands[i] if demands[i] > 0 else 1.0) * y[j]
         for i in range(num_customers) for j in range(num_plants)),
        name="max_frac"
    )

    # Logging callback
    def cb(mod, where):
        if where == GRB.Callback.MIPSOL:
            obj = mod.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Optimize the model
    model.optimize(cb)

    # Extract Solution
    if model.SolCount > 0:
        open_plants = [j for j in range(num_plants) if y[j].X > 0.5]
        assignments = {str(i): {} for i in range(num_customers)}
        
        for i in range(num_customers):
            for j in range(num_plants):
                val = x[i, j].X
                if val > 1e-6:  # Filter out floating-point noise
                    assignments[str(i)][str(j)] = val

        solution = {
            "objective_value": model.ObjVal,
            "open_plants": open_plants,
            "assignments": assignments
        }

        # Write out to the solution path
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        print("No feasible solution found within the time limit.")

if __name__ == "__main__":
    main()