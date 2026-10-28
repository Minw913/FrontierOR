import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve():
    parser = argparse.ArgumentParser(description="Solve stochastic facility location problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where the solution JSON file will be written.")
    parser.add_argument("--time_limit", type=int, required=False, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, required=False, help="Path to a JSONL file for logging intermediate solutions.")
    args = parser.parse_args()

    # Initialize the solution logger if a log path is provided
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    # Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_facilities = data["num_facilities"]
    num_customers = data["num_customers"]
    num_scenarios = data["num_scenarios"]

    fixed_costs = np.array(data["facilities"]["fixed_costs"], dtype=np.float64)
    capacities = np.array(data["facilities"]["capacities"], dtype=np.float64)
    routing_costs = np.array(data["routing_costs"], dtype=np.float64)
    
    probs = np.array(data["scenarios"]["probabilities"], dtype=np.float64)
    demands = np.array(data["scenarios"]["demands"], dtype=np.float64)
    max_total_demand = float(data["max_total_demand"])

    # Precalculate objective function coefficients for continuous variables
    # coeff shape: (num_scenarios, num_facilities, num_customers)
    # coeff[s, i, j] = probability[s] * routing_cost[i, j]
    coeff = probs[:, np.newaxis, np.newaxis] * routing_costs[np.newaxis, :, :]

    # Setup Gurobi Environment
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)  # Restrict to a single CPU core as per requirements
    if args.time_limit is not None:
        env.setParam("TimeLimit", args.time_limit)
    env.start()

    model = gp.Model(env=env)

    # Variables
    # y: 1 if facility is opened, 0 otherwise
    y = model.addMVar(num_facilities, vtype=GRB.BINARY, obj=fixed_costs, name="y")
    
    # x: flow from facility i to customer j under scenario s
    x = model.addMVar((num_scenarios, num_facilities, num_customers), 
                      lb=0.0, vtype=GRB.CONTINUOUS, obj=coeff, name="x")

    # Constraint 1: Meet the demand completely for all customers under each scenario
    # x.sum(axis=1) sums over facilities, yielding shape (num_scenarios, num_customers)
    model.addConstr(x.sum(axis=1) >= demands, name="Demand_Satisfaction")

    # Constraint 2: Cannot exceed facility capacity, only active if opened
    # For a fixed scenario s, x[s].sum(axis=1) yields shape (num_facilities,) which is the total outgoing flow from each facility
    caps_diag = np.diag(capacities)
    for s in range(num_scenarios):
        model.addConstr(x[s].sum(axis=1) <= caps_diag @ y, name=f"Capacity_Limit_s{s}")

    # Constraint 3: The aggregate capacity of all opened facilities must be enough to meet the maximum scenario demand
    model.addConstr(capacities @ y >= max_total_demand, name="Global_Capacity_Sufficient")

    # Optimization callback for logging incumbent solutions
    def opt_callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger is not None:
                logger.log(obj_val)

    # Optimize the model
    model.optimize(opt_callback)

    # Output extraction
    if model.SolCount > 0:
        objective_value = model.ObjVal
        y_vals = [int(round(float(y[i].X))) for i in range(num_facilities)]
        
        solution_output = {
            "objective_value": objective_value,
            "y": y_vals
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(solution_output, f, indent=2)

if __name__ == "__main__":
    solve()