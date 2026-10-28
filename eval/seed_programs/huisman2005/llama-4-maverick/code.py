import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance, time_limit, log_path=None):
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Extract relevant data from instance
    num_trips = instance['num_trips']
    num_depots = instance['num_depots']
    trips = instance['trips']
    depots = instance['depots']
    duty_types = instance['duty_types']
    crew_parameters = instance['crew_parameters']
    vehicle_parameters = instance['vehicle_parameters']
    travel_times = instance['travel_times']
    
    # Initialize Gurobi model
    model = Model("VehicleAndCrewScheduling")
    model.setParam("Threads", 1)
    model.Params.TimeLimit = time_limit
    
    # Decision Variables
    vehicle_used = model.addVars(num_depots, vtype=GRB.BINARY, name="vehicle_used")
    trip_to_depot = model.addVars(num_trips, num_depots, vtype=GRB.BINARY, name="trip_to_depot")
    vehicle_sequence = model.addVars(num_trips, num_trips, num_depots, vtype=GRB.BINARY, name="vehicle_sequence")
    duty_trip = model.addVars(num_trips, num_depots, vtype=GRB.BINARY, name="duty_trip")
    duty_type = model.addVars(num_depots, vtype=GRB.INTEGER, name="duty_type")
    
    # Objective Function
    model.setObjective(quicksum(vehicle_used[d] for d in range(num_depots)) + quicksum(duty_type[d] for d in range(num_depots)), GRB.MINIMIZE)
    
    # Constraints
    # 1. Each trip is assigned to exactly one depot
    for t in range(num_trips):
        model.addConstr(quicksum(trip_to_depot[t, d] for d in range(num_depots)) == 1)
    
    # 2. Vehicle sequence constraints
    for t1 in range(num_trips):
        for d in range(num_depots):
            model.addConstr(quicksum(vehicle_sequence[t1, t2, d] for t2 in range(num_trips)) == trip_to_depot[t1, d])
            model.addConstr(quicksum(vehicle_sequence[t2, t1, d] for t2 in range(num_trips)) == trip_to_depot[t1, d])
    
    # 3. Duty trip constraints
    for t in range(num_trips):
        for d in range(num_depots):
            model.addConstr(duty_trip[t, d] <= trip_to_depot[t, d])
    
    # 4. Duty type constraints (simplified, actual implementation depends on duty type rules)
    for d in range(num_depots):
        model.addConstr(duty_type[d] >= quicksum(duty_trip[t, d] for t in range(num_trips)) / 10)  # Simplified
    
    # 5. Vehicle usage constraint
    for d in range(num_depots):
        model.addConstr(vehicle_used[d] >= quicksum(trip_to_depot[t, d] for t in range(num_trips)) / num_trips)
    
    # Optimize
    model.optimize()
    
    # Log intermediate solutions
    if logger:
        for sol in range(model.SolCount):
            model.Params.SolutionNumber = sol
            objective_value = model.PoolObjVal
            logger.log(objective_value)
    
    # Extract and return solution
    if model.SolCount > 0:
        solution = {
            "objective_value": int(model.ObjVal),
            "vehicle_schedule": {},
            "crew_schedule": []
        }
        
        # Vehicle Schedule
        for d, depot in enumerate(depots):
            depot_name = depot['name']
            solution["vehicle_schedule"][depot_name] = []
            vehicle_trips = []
            for t in range(num_trips):
                if trip_to_depot[t, d].X > 0.5:
                    vehicle_trips.append(trips[t]['trip_id'])
            solution["vehicle_schedule"][depot_name].append(vehicle_trips)
        
        # Crew Schedule
        for d, depot in enumerate(depots):
            depot_name = depot['name']
            for _ in range(int(duty_type[d].X)):
                duty = {
                    "depot": depot_name,
                    "type": list(duty_types.keys())[0],  # Simplified, assuming one duty type
                    "trips": [t for t in range(num_trips) if duty_trip[t, d].X > 0.5]
                }
                solution["crew_schedule"].append(duty)
        
        return solution
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description='Solve Vehicle and Crew Scheduling Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to the log JSONL file')
    args = parser.parse_args()
    
    instance = read_instance(args.instance_path)
    solution = solve(instance, args.time_limit, args.log_path)
    
    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()