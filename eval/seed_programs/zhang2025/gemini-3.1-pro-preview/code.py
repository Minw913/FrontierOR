import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="Public Transit Driver Duty Selection Optimization")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where the solution JSON file will be written.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to a JSONL file for logging intermediate solutions.")
    
    args = parser.parse_args()
    
    # Initialize the solution logger if path is provided
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Load problem instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    num_trips = data["parameters"]["num_trips_M"]
    num_duties = data["parameters"]["num_duties_N"]
    
    duties = data["duties"]
    costs = data["costs"]
    
    # Configure Gurobi Environment
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)  # Suppress internal prints
    env.setParam("Threads", 1)  # Single core constraint enforcement
    env.start()
    
    model = gp.Model("TransitDriverDuties_SPP", env=env)
    
    # Enforce time limit requirement
    if args.time_limit is not None:
        model.setParam("TimeLimit", args.time_limit)
        
    model.ModelSense = GRB.MINIMIZE
    
    # Use scipy to construct constraints efficiently via matrix algebra if available, fallback otherwise
    try:
        import scipy.sparse as sp
        import numpy as np
        use_matrix = True
    except ImportError:
        use_matrix = False

    if use_matrix:
        row_indices = []
        col_indices = []
        values = []
        
        # Build block matrix structure reflecting which trips are spanned by each duty
        for j, duty_trips in enumerate(duties):
            for trip_id in duty_trips:
                row_indices.append(trip_id)
                col_indices.append(j)
                values.append(1.0)
                
        # Objective & Constraints structure construction
        A = sp.coo_matrix((values, (row_indices, col_indices)), shape=(num_trips, num_duties))
        x = model.addMVar(num_duties, vtype=GRB.BINARY, obj=np.array(costs), name="duty")
        rhs = np.ones(num_trips)
        
        model.addConstr(A @ x == rhs, name="trip_cover")
    else:
        # Pure Python Gurobi API fallback for environments missing `scipy`/`numpy` computations
        x = model.addVars(num_duties, vtype=GRB.BINARY, obj=costs, name="duty")
        col_indices_by_row = [[] for _ in range(num_trips)]
        
        for j, duty_trips in enumerate(duties):
            for trip_id in duty_trips:
                col_indices_by_row[trip_id].append(j)
                
        for i in range(num_trips):
            model.addConstr(gp.quicksum(x[j] for j in col_indices_by_row[i]) == 1, name=f"trip_cover_{i}")
            
    # Custom Callback for Logging Runtime Feasible/Incumbent Solutions
    def mip_logging_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger is not None:
                logger.log(obj)
                
    # Optimize Set Partitioning model
    model.optimize(mip_logging_callback)
    
    objective_value = 0.0
    selected_duties = []
    
    # Output projection handling
    if model.SolCount > 0:
        objective_value = float(model.ObjVal)
        if use_matrix:
            x_vals = x.X
            selected_duties = [j for j, val in enumerate(x_vals) if val > 0.5]
        else:
            selected_duties = [j for j in range(num_duties) if x[j].X > 0.5]
            
    output = {
        "objective_value": objective_value,
        "selected_duties": selected_duties
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    main()