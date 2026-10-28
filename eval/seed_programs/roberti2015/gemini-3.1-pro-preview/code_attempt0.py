import gurobipy as gp
from gurobipy import GRB
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Fixed-Charge Transportation Problem")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the JSONL file for logging intermediate solutions")
    args = parser.parse_args()

    # Read instance
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_sources = data["num_sources"]
    num_sinks = data["num_sinks"]
    supply = data["supply"]
    demand = data["demand"]
    fixed_cost = data["fixed_cost"]
    variable_cost = data["variable_cost"]
    capacity = data["capacity"]

    # Initialize Logger
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    # Gurobi Model
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()
    
    model = gp.Model("FixedChargeTransportation", env=env)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Threads", 1)
    model.setParam("MIPFocus", 1)  # Focus on finding good feasible solutions

    # Prepare dictionaries for rapid constraint/objective generation
    cap_dict = {(i, j): capacity[i][j] for i in range(num_sources) for j in range(num_sinks)}
    vcost = {(i, j): variable_cost[i][j] for i in range(num_sources) for j in range(num_sinks)}
    fcost = {(i, j): fixed_cost[i][j] for i in range(num_sources) for j in range(num_sinks)}

    # Variables
    x = model.addVars(num_sources, num_sinks, lb=0, ub=cap_dict, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVars(num_sources, num_sinks, vtype=GRB.BINARY, name="y")
    
    # Constraints
    model.addConstrs((x.sum(i, '*') == supply[i] for i in range(num_sources)), name="supply")
    model.addConstrs((x.sum('*', j) == demand[j] for j in range(num_sinks)), name="demand")
    model.addConstrs((x[i, j] <= capacity[i][j] * y[i, j] for i in range(num_sources) for j in range(num_sinks)), name="cap")
            
    # Initial feasible solution based on the Northwest Corner Method to guarantee a valid start
    rem_supply = list(supply)
    rem_demand = list(demand)
    
    for i in range(num_sources):
        for j in range(num_sinks):
            x[i, j].Start = 0
            y[i, j].Start = 0

    i, j = 0, 0
    while i < num_sources and j < num_sinks:
        amt = min(rem_supply[i], rem_demand[j])
        if amt > 0:
            x[i, j].Start = amt
            y[i, j].Start = 1
            rem_supply[i] -= amt
            rem_demand[j] -= amt
        if rem_supply[i] == 0:
            i += 1
        if rem_demand[j] == 0:
            j += 1

    # Objective
    model.setObjective(x.prod(vcost) + y.prod(fcost), GRB.MINIMIZE)
    
    # Callback for logging incumbent solutions
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
                
    model.optimize(cb)
    
    # Write output to valid schema JSON
    if model.SolCount > 0:
        obj_val = model.ObjVal
        flow = [[0.0] * num_sinks for _ in range(num_sources)]
        arcs_used = [[0] * num_sinks for _ in range(num_sources)]
        
        for i_src in range(num_sources):
            for j_snk in range(num_sinks):
                flow[i_src][j_snk] = float(x[i_src, j_snk].X)
                arcs_used[i_src][j_snk] = int(round(y[i_src, j_snk].X))
                
        solution = {
            "objective_value": float(obj_val),
            "flow": flow,
            "arcs_used": arcs_used
        }
        
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()