import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_transit_scheduling(instance_path, solution_path, time_limit, log_path):
    # Load the instance
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    num_trips = data["parameters"]["num_trips_M"]
    num_duties = data["parameters"]["num_duties_N"]
    duties = data["duties"]
    costs = data["costs"]
    
    # Create the model
    model = gp.Model("TransitScheduling")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1)  # Enable Gurobi output
    
    # Decision variables: x[i] indicates whether duty i is selected
    x = model.addVars(num_duties, vtype=GRB.BINARY, name="x")
    
    # Constraint: every trip is covered exactly once
    for trip_id in range(num_trips):
        duties_covering_trip = []
        for i in range(num_duties):
            if trip_id in duties[i]:
                duties_covering_trip.append(i)
        
        # Each trip must be covered by exactly one duty
        model.addConstr(gp.quicksum(x[i] for i in duties_covering_trip) == 1, 
                       name=f"CoverTrip_{trip_id}")
    
    # Objective: minimize total cost
    model.setObjective(gp.quicksum(costs[i] * x[i] for i in range(num_duties)), GRB.MINIMIZE)
    
    # Set a callback for logging intermediate solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Optimize
    model.optimize(callback)
    
    # Extract solution
    objective_value = model.ObjVal if model.SolCount > 0 else float('inf')
    selected_duties = [i for i in range(num_duties) if x[i].X > 0.5]
    
    # Prepare and write the solution
    solution = {
        "objective_value": objective_value,
        "selected_duties": selected_duties
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    
    args = parser.parse_args()
    
    solve_transit_scheduling(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )