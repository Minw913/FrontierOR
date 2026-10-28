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
    model.setParam("Threads", 1)
    
    # Set parameters for solver
    model.Params.timelimit = args.time_limit
    model.Params.MIPGap = 1e-4  # Stop when gap is 0.1%
    model.Params.OutputFlag = 0  # Turn off solver output during search
    
    # Indices: 0 is depot (both origin and destination as per description)
    locations = list(range(num_nodes))
    customers = list(range(1, num_customers + 1))
    
    # Depot nodes: 0 is origin, num_nodes is destination (new index after extending)
    depot_out = 0
    depot_in = num_nodes  # Virtual depot destination node added as next after existing nodes

    # Extend the times matrices to add a virtual depot_in node (index = num_nodes)
    # For virtual depot, distances from/to it mimic those of original depot
    extended_truck_times = []
    for i in range(num_nodes):
        row = truck_times[i][:] + [truck_times[i][0]]  # Distance to new virtual depot
        extended_truck_times.append(row)
    # New row for virtual depot - distance from virtual depot
    extended_truck_times.append(truck_times[0][:] + [0])  # From virtual depot, same as original depot

    extended_drone_times = []
    for i in range(num_nodes):
        row = drone_times[i][:] + [drone_times[i][0]]
        extended_drone_times.append(row)
    extended_drone_times.append(drone_times[0][:] + [0])

    # Extended node list includes the virtual depot destination node
    extended_nodes = list(range(num_nodes + 1))  # 0 to N (N+1 total)
    extended_customers = list(range(1, num_customers + 1))

    # Variables
    # x[i,j] = 1 if truck goes from i to j
    x = {}
    for i in extended_nodes:
        for j in extended_nodes:
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
                
    # y[i,j] = 1 if drone goes from i to j
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
    
    # Arrival time variables for each node
    t = {}
    for i in extended_nodes:
        t[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"t_{i}")
        
    # Objective variable: to minimize the makespan
    makespan = model.addVar(vtype=GRB.CONTINUOUS, name="makespan")


    # Constraints
    # Each customer served exactly once
    for c in customers:
        model.addConstr(z_truck[c] + z_drone[c] + z_combined[c] == 1, name=f"serve_once_{c}")

    # Truck and drone paths: enter and exit the depot
    # Truck must leave depot_out and come back to depot_in
    model.addConstr(gp.quicksum(x[depot_out, j] for j in extended_nodes if j != depot_out) == 1, name="truck_exit_depot")
    model.addConstr(gp.quicksum(x[i, depot_in] for i in extended_nodes if i != depot_in) == 1, name="truck_return_depot")

    # Drone must leave depot_out and come back to depot_in
    model.addConstr(gp.quicksum(y[depot_out, j] for j in extended_nodes if j != depot_out) == 1, name="drone_exit_depot")
    model.addConstr(gp.quicksum(y[i, depot_in] for i in extended_nodes if i != depot_in) == 1, name="drone_return_depot")

    # Flow conservation for truck and drone at customer nodes
    for i in extended_customers:
        # Ensure the in-degree equals out-degree for truck
        model.addConstr(gp.quicksum(x[j, i] for j in extended_nodes if j != i) == 
                        gp.quicksum(x[i, j] for j in extended_nodes if j != i), 
                       name=f"truck_flow_conservation_{i}")

        # Similarly for drone
        model.addConstr(gp.quicksum(y[j, i] for j in extended_nodes if j != i) == 
                        gp.quicksum(y[i, j] for j in extended_nodes if j != i),
                       name=f"drone_flow_conservation_{i}")

    # Truck exits a customer if and only if it serves that customer (truck or combined)
    for i in customers:
        # Outgoing arcs condition
        model.addConstr(gp.quicksum(x[i, j] for j in extended_nodes if j != i) <= z_truck[i] + z_combined[i],
                       name=f"truck_exit_upper_{i}")
        model.addConstr(gp.quicksum(x[i, j] for j in extended_nodes if j != i) >= z_truck[i] + z_combined[i],
                       name=f"truck_exit_lower_{i}")

    # Drone exits a customer if and only if it serves that customer (drone or combined)
    for i in customers:
        model.addConstr(gp.quicksum(y[i, j] for j in extended_nodes if j != i) <= z_drone[i] + z_combined[i],
                       name=f"drone_exit_upper_{i}")
        model.addConstr(gp.quicksum(y[i, j] for j in extended_nodes if j != i) >= z_drone[i] + z_combined[i],
                       name=f"drone_exit_lower_{i}")

    # Drone between customers constraint: no direct arc between customers unless at least one is combined
    for i in customers:
        for j in customers:
            if i != j:
                model.addConstr(y[i, j] <= z_combined[i] + z_combined[j],
                               name=f"limit_drone_between_{i}_{j}_out")
                model.addConstr(y[j, i] <= z_combined[i] + z_combined[j],
                               name=f"limit_drone_between_{j}_{i}_in")

    # Drone single-shop constraint: Cannot visit exactly one customer (depot->cust->depot)
    for i in customers:
        model.addConstr(y[depot_out, i] + y[i, depot_in] <= 1,
                       name=f"no_drone_single_visit_{i}")
        
    # Time propagation constraints
    bigM = sum(sum(row) for row in extended_truck_times) // len(extended_truck_times)
    
    # For truck
    for i in extended_nodes:
        for j in extended_nodes:
            if i != j:
                model.addConstr(t[j] >= t[i] + extended_truck_times[i][j] - bigM * (1 - x[i, j]),
                               name=f"truck_time_link_{i}_{j}")

    # For drone
    for i in extended_nodes:
        for j in extended_nodes:
            if i != j:
                model.addConstr(t[j] >= t[i] + extended_drone_times[i][j] - bigM * (1 - y[i, j]),
                               name=f"drone_time_link_{i}_{j}")

    # Objective: makespan is the arrival time at the final depot
    model.addConstr(makespan == t[depot_in])
    model.setObjective(makespan, GRB.MINIMIZE)

    # Solve the model
    def log_solution_cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_value)

    # Enable logging callback
    if logger is not None:
        model.optimize(log_solution_cb)
    else:
        model.optimize()

    
    # Check if we have a solution and extract it
    if model.SolCount > 0:
        # Reconstruct truck route
        truck_route = [depot_out]
        current = depot_out
        visited = {depot_out}
        while current != depot_in:
            next_node = None
            for j in extended_nodes:
                if j not in visited and x[current, j].X > 0.5:
                    next_node = j
                    truck_route.append(j)
                    current = j
                    visited.add(j)
                    break
            if next_node is None:
                break
        
        drone_route = [depot_out]
        current = depot_out
        visited = {depot_out}
        while current != depot_in:
            next_node = None
            for j in extended_nodes:
                if j not in visited and y[current, j].X > 0.5:
                    next_node = j
                    drone_route.append(j)
                    current = j
                    visited.add(j)
                    break
            if next_node is None:
                break

        customer_types = {}
        for i in customers:
            if z_truck[i].X > 0.5:
                customer_types[str(i)] = "truck"
            elif z_drone[i].X > 0.5:
                customer_types[str(i)] = "drone"
            elif z_combined[i].X > 0.5:
                customer_types[str(i)] = "combined"
                
        arrival_times = {}
        # Include original nodes (0 to N) in output
        for i in range(num_nodes):  # 0 to original depot dest
            arrival_times[str(i)] = t[i].X
        arrival_times[str(num_nodes)] = t[depot_in].X  # Final depot destination (virtual node mapped)

        result = {
            "objective_value": model.ObjVal,
            "truck_route": truck_route,
            "drone_route": drone_route,
            "customer_types": customer_types,
            "arrival_times": arrival_times
        }
    else:
        result = None
    
    return result


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