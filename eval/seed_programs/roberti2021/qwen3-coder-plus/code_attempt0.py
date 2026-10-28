import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import math
from solution_logger import SolutionLogger


def solve_truck_drone(instance_data):
    # Parse input data
    num_customers = instance_data["num_customers"]
    num_nodes = instance_data["num_nodes"]
    alpha = instance_data["alpha"]
    nodes = instance_data["nodes"]
    truck_times = instance_data["truck_travel_times"]
    drone_times = instance_data["drone_travel_times"]

    # Create a Gurobi model
    model = gp.Model("TruckDroneRouting")
    
    # Set parameters for solver
    model.Params.timelimit = args.time_limit
    model.Params.MIPGap = 0.001  # Stop when gap is 0.1%
    model.Params.OutputFlag = 0  # Turn off solver output during search
    
    # Indices
    locations = list(range(num_nodes))
    customers = list(range(1, num_customers + 1))
    
    # Depot nodes: 0 is source, num_nodes is sink (virtual for depot dest)
    depot_out = 0
    depot_in = num_nodes + 1  # Actually node N+1 in original indexing - virtual node in our formulation
    
    # Corrected indexing since drone times matrix has only 'num_nodes' entries
    # Add one more row/column for virtual depot destination
    drone_times_extended = [row[:] + [0] for row in drone_times]
    drone_times_extended.append([0] * (num_nodes + 1))
    for i in range(len(drone_times)):
        drone_times_extended[i].append(truck_times[i][-1])
    
    # Update num_nodes including virtual depot_in
    extended_nodes = list(range(num_nodes + 1))  # Original 0 to N, plus new virtual node N+1
    extended_customers = list(range(1, num_customers + 1))


    # Variables
    # x[i,j] = 1 if truck goes from i to j
    x = {}
    for i in extended_nodes:
        for j in extended_nodes:
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
                
    # y[i,j] = 1 if drone goes from i to j (using extended drone times)
    y = {}
    for i in extended_nodes:
        for j in extended_nodes:
            if i != j:
                y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    # Customer service type variables
    z_truck = {}  # 1 if customer is served by truck only
    z_drone = {}  # 1 if drone only
    z_combined = {}  # 1 if served by both together
    for c in customers:
        z_truck[c] = model.addVar(vtype=GRB.BINARY, name=f"z_truck_{c}")
        z_drone[c] = model.addVar(vtype=GRB.BINARY, name=f"z_drone_{c}")
        z_combined[c] = model.addVar(vtype=GRB.BINARY, name=f"z_combined_{c}")
    
    # Arrival time variables for each node (including depot_in which should go in answer)
    t = {}
    for i in extended_nodes:
        t[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"t_{i}")
        
    # Objective variable
    makespan = model.addVar(vtype=GRB.CONTINUOUS, name="makespan")
    

    # Constraints
    # Each customer served exactly once
    for c in customers:
        model.addConstr(z_truck[c] + z_drone[c] + z_combined[c] == 1, name=f"serve_once_{c}")

    # Truck must leave and arrive at depot
    # From depot OUT
    model.addConstr(gp.quicksum(x[depot_out, j] for j in extended_nodes if j != depot_out) == 1, name="truck_exit_depot")
    # Arrive at depot IN
    model.addConstr(gp.quicksum(x[i, depot_in] for i in extended_nodes if i != depot_in) == 1, name="truck_return_depot")

    # Drone must leave and arrive at depot
    # From depot OUT
    model.addConstr(gp.quicksum(y[depot_out, j] for j in extended_nodes if j != depot_out) == 1, name="drone_exit_depot")
    # Arrive at depot IN
    model.addConstr(gp.quicksum(y[i, depot_in] for i in extended_nodes if i != depot_in) == 1, name="drone_return_depot")

    # Flow conservation for truck
    for i in extended_customers:
        model.addConstr(gp.quicksum(x[j, i] for j in extended_nodes if j != i) == 
                        gp.quicksum(x[i, j] for j in extended_nodes if j != i), 
                       name=f"truck_flow_conservation_{i}")

    # Flow conservation for drone  
    for i in extended_customers:
        model.addConstr(gp.quicksum(y[j, i] for j in extended_nodes if j != i) == 
                        gp.quicksum(y[i, j] for j in extended_nodes if j != i),
                       name=f"drone_flow_conservation_{i}")


    # Define which nodes have truck exiting (only for trucks and combined)
    for i in customers:
        # Out-degree constraints for truck
        # If drone_only, it does not exit
        model.addConstr(
            gp.quicksum(x[i, j] for j in extended_nodes if j != i) <= 1 - z_drone[i],
            name=f"truck_outdegree_{i}")
        
    for i in customers:
        # Similarly for drone
        model.addConstr(
            gp.quicksum(y[i, j] for j in extended_nodes if j != i) <= 1 - z_truck[i],
            name=f"drone_outdegree_{i}")

    # If combined service, exactly one outgoing truck/drones
    for i in customers:
        # Truck exits customer i exactly once if not drone-only
        model.addConstr(gp.quicksum(x[i, j] for j in extended_nodes if j != i) >= z_truck[i] + z_combined[i],
                        name=f"truck_exit_cond_{i}")
        
        model.addConstr(gp.quicksum(y[i, j] for j in extended_nodes if j != i) >= z_drone[i] + z_combined[i],
                        name=f"drone_exit_cond_{i}")

    # Drone between customers constraint: no arc unless at least one is combined
    for i in customers:
        for j in customers:
            if i != j:
                model.addConstr(y[i, j] + y[j, i] <= z_combined[i] + z_combined[j],
                               name=f"limit_dronetravel_{i}_{j}")

    # Drone singleshop constraint: drone does not do a trip from depot to 1 customer and return
    for i in customers:
        # Prevent: depot_out -> i -> depot_in
        model.addConstr(y[depot_out, i] + y[i, depot_in] <= 1,
                       name=f"no_singleshop_{i}")
    
    # Time constraints: arrival_time >= departure_time + travel time
    bigM = max(max(row) for row in truck_times) * len(locations)  # Big enough but not excessive
    
    # Truck time constraints
    for i in extended_nodes:
        for j in extended_nodes:
            if i != j:
                model.addConstr(t[j] >= t[i] + truck_times[i][min(j, len(truck_times[0])-1)] 
                                - bigM * (1 - x[i, j]),
                               name=f"truck_time_{i}_{j}")

    # Drone time constraints
    for i in extended_nodes:
        for j in extended_nodes:
            if i != j:
                model.addConstr(t[j] >= t[i] + drone_times_extended[i][j]
                                - bigM * (1 - y[i, j]),
                               name=f"drone_time_{i}_{j}")

    # In combined services, the earlier arriver waits for the later to reach sync
    # We handle this by setting the arrival time of customer to max of both
    # Which is handled inherently via the flow-based time constraints above
    # Because if both visit customer i, both need to allow for their arrival time before we can continue.
    
    # Make the makespan be the maximum of arrivals at the depot
    model.addConstr(makespan == t[depot_in])

    # Objective: minimize makespan (last vehicle arrival at depot)
    model.setObjective(makespan, GRB.MINIMIZE)

    # Solve the model
    def log_solution_cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_value)

    # Enable logging callback
    model.optimize(log_solution_cb)
    
    # Extract the solution
    if model.SolCount > 0:
        # Get actual routes
        truck_path = reconstruct_path(x, extended_nodes, depot_out, depot_in)
        drone_path = reconstruct_path(y, extended_nodes, depot_out, depot_in)
        
        # Convert to desired format
        customer_types = {}
        for i in customers:
            if model.getVal(z_truck[i]) > 0.5:
                customer_types[str(i)] = "truck"
            elif model.getVal(z_drone[i]) > 0.5:
                customer_types[str(i)] = "drone"
            elif model.getVal(z_combined[i]) > 0.5:
                customer_types[str(i)] = "combined"
        
        arrival_times = {}
        for i in range(num_nodes):  # Original nodes (no virtual node)
            arrival_times[str(i)] = model.getVal(t[(i)])
        # We should map this correctly to include virtual depot as final time
        arrival_times[str(num_nodes)] = model.getVal(t[depot_in])

        result = {
            "objective_value": model.ObjVal,
            "truck_route": truck_path,
            "drone_route": drone_path,
            "customer_types": customer_types,
            "arrival_times": arrival_times
        }
    else:
        # No solution found - potentially return infeasible state or error
        result = None
    
    return result


def reconstruct_path(vars, nodes, start, end):
    """Reconstruct path from binary variables in order"""
    path = [start]
    current = start
    while current != end:
        next_node_found = False
        for j in nodes:
            if j != current and vars[current, j].X > 0.5:
                path.append(j)
                current = j
                next_node_found = True
                break
        if not next_node_found and current != end:
            break  # Shouldn't happen for valid solution
    return path


def main(args):
    global logger
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    solution = solve_truck_drone(instance_data)
    
    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("No feasible solution found within the time limit.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve Truck Drone Routing Problem')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to input instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, help='Path to logging JSONL file (optional)')
    
    args = parser.parse_args()
    main(args)