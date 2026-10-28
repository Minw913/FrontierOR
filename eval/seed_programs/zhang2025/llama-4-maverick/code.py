import json
import argparse
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger
def read_instance_data(instance_path):
    with open(instance_path, 'r') as file:
        data = json.load(file)
    return data

def solve_instance(data, time_limit, logger):
    num_trips = data['parameters']['num_trips_M']
    num_duties = data['parameters']['num_duties_N']
    trips = data['trips']
    duties = data['duties']
    costs = data['costs']

    model = Model("BusTripOptimization")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.Params.LogToConsole = 0  # Disable console logging
    model.Params.TimeLimit = time_limit  # Set time limit

    # Decision variables: Whether to select each duty
    x = model.addVars(num_duties, vtype=GRB.BINARY, name="select_duty")

    # Objective: Minimize total cost of selected duties
    model.setObjective(quicksum(x[i] * costs[i] for i in range(num_duties)), GRB.MINIMIZE)

    # Constraints: Each trip must be covered exactly once
    for trip_id in range(num_trips):
        covering_duties = [i for i, duty in enumerate(duties) if trip_id in duty]
        model.addConstr(quicksum(x[i] for i in covering_duties) == 1, f"cover_trip_{trip_id}")

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        objective_value = model.ObjVal
        selected_duties = [i for i in range(num_duties) if x[i].X > 0.5]
        return objective_value, selected_duties
    else:
        raise Exception(f"Gurobi model failed to find a feasible solution. Status: {model.status}")

def main():
    parser = argparse.ArgumentParser(description='Solve bus trip optimization problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds.')
    parser.add_argument('--log_path', type=str, help='Optional path to log intermediate solutions.')
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    instance_data = read_instance_data(args.instance_path)

    def callback(model, where):
        if where == GRB.Callback.MIP:
            objective_value = model.cbGet(GRB.Callback.MIP_OBJBST)
            if model.cbGet(GRB.Callback.MIP_SOLCNT) > 0 and (not hasattr(callback, 'last_logged') or objective_value < callback.last_logged):
                callback.last_logged = objective_value
                if logger:
                    logger.log(objective_value)

    try:
        model = Model("BusTripOptimization")
        model.setParam("Seed", 0)
        model.setParam("MIPGap", 1e-4)
        model.setParam("NumericFocus", 0)
        model.setParam("Threads", 1)
        model.Params.LogToConsole = 0  
        model.Params.TimeLimit = args.time_limit  

        num_trips = instance_data['parameters']['num_trips_M']
        num_duties = instance_data['parameters']['num_duties_N']
        duties = instance_data['duties']
        costs = instance_data['costs']

        x = model.addVars(num_duties, vtype=GRB.BINARY, name="select_duty")
        model.setObjective(quicksum(x[i] * costs[i] for i in range(num_duties)), GRB.MINIMIZE)

        for trip_id in range(num_trips):
            covering_duties = [i for i, duty in enumerate(duties) if trip_id in duty]
            model.addConstr(quicksum(x[i] for i in covering_duties) == 1, f"cover_trip_{trip_id}")

        model.optimize(callback)

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            objective_value = model.ObjVal
            selected_duties = [i for i in range(num_duties) if x[i].X > 0.5]
            solution = {
                "objective_value": objective_value,
                "selected_duties": selected_duties
            }
            with open(args.solution_path, 'w') as file:
                json.dump(solution, file, indent=4)
        else:
            raise Exception(f"Gurobi model failed to find a feasible solution. Status: {model.status}")
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()