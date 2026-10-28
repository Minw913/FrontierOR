import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main(instance_path, solution_path, time_limit, log_path):
    # Read instance data
    with open(instance_path) as f:
        data = json.load(f)
    
    # Extract data according to schema
    num_nodes = data["num_nodes"]
    depot_index = data["depot_index"]
    customer_indices = data["customer_indices"]
    truck_travel_times = data["truck_travel_times"]
    drone_battery_consumption = data["drone_battery_consumption"]
    drone_delivery_time = data["drone_delivery_time"]
    parameters = data["parameters"]
    
    num_drones = parameters["num_drones"]
    truck_service_time = parameters["truck_service_time"]
    drone_battery_capacities = parameters["drone_battery_capacities"]
    
    # Initialize solution logger if log_path provided
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Extend nodes with sink (virtual return depot)
    sink = num_nodes
    V = list(range(num_nodes)) + [sink]  # All nodes including sink
    N = list(range(num_nodes))           # Original nodes
    C = customer_indices                 # Customer indices
    D = list(range(num_drones))          # Drone indices
    
    # Create travel time matrix T including sink
    T = [[0] * (num_nodes + 1) for _ in range(num_nodes + 1)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            T[i][j] = truck_travel_times[i][j]
        # Travel to sink uses depot travel time (without service time adjustment)
        T[i][sink] = truck_travel_times[i][depot_index]
    
    # Sink has no outgoing travel
    for j in range(num_nodes + 1):
        T[sink][j] = 0
    
    # Create optimization model
    model = gp.Model("Truck_Drone")
    
    # Decision variables
    x = {}  # Truck routing: x[i,j] = 1 if truck travels from i to j
    for i in V:
        for j in V:
            # Skip self-arcs to prevent staying at same node
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    z = {}  # Drone assignments: z[d,i,k] = 1 if drone d serves customer k from node i
    for d in D:
        for i in N:
            for k in C:
                z[d, i, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{d}_{i}_{k}")
    
    t = {}  # Truck visits: t[k] = 1 if truck visits customer k
    for k in C:
        t[k] = model.addVar(vtype=GRB.BINARY, name=f"t_{k}")
    
    w = {}  # Waiting times: w[i] = waiting time at node i
    for i in N:
        w[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"w_{i}")
    
    # Constraints
    # Each customer served exactly once (by truck or drone)
    for k in C:
        model.addConstr(
            t[k] + gp.quicksum(z[d, i, k] for d in D for i in N) == 1,
            f"serve_{k}"
        )
    
    # Truck starts at depot and ends at sink
    model.addConstr(
        gp.quicksum(x[depot_index, j] for j in V if (depot_index, j) in x) == 1,
        "start"
    )
    model.addConstr(
        gp.quicksum(x[i, sink] for i in V if (i, sink) in x) == 1,
        "end"
    )
    
    # Prevent leaving sink or entering depot
    model.addConstr(
        gp.quicksum(x[sink, j] for j in V if (sink, j) in x) == 0,
        "sink_out"
    )
    model.addConstr(
        gp.quicksum(x[i, depot_index] for i in V if (i, depot_index) in x) == 0,
        "depot_in"
    )
    
    # Flow conservation constraints
    for u in N:
        if u == depot_index:
            # Depot: one outgoing arc, no incoming
            model.addConstr(
                gp.quicksum(x[u, j] for j in V if (u, j) in x) == 1,
                f"out_{u}"
            )
            model.addConstr(
                gp.quicksum(x[i, u] for i in V if (i, u) in x) == 0,
                f"in_{u}"
            )
        else:
            # Other nodes: visit if truck comes in/out
            model.addConstr(
                gp.quicksum(x[i, u] for i in V if (i, u) in x) == t[u],
                f"in_{u}"
            )
            model.addConstr(
                gp.quicksum(x[u, j] for j in V if (u, j) in x) == t[u],
                f"out_{u}"
            )
    
    # Drone dispatch only from visited nodes
    for d in D:
        for i in N:
            # Depot is always visited, no constraint needed
            if i != depot_index:
                for k in C:
                    model.addConstr(
                        z[d, i, k] <= t[i],
                        f"drone_node_{d}_{i}_{k}"
                    )
    
    # Drone battery constraints
    for d in D:
        model.addConstr(
            gp.quicksum(
                drone_battery_consumption[d][i][k] * z[d, i, k]
                for i in N for k in C
            ) <= drone_battery_capacities[d],
            f"battery_{d}"
        )
    
    # Waiting time constraints (max drone time per node)
    for d in D:
        for i in N:
            model.addConstr(
                w[i] >= gp.quicksum(
                    drone_delivery_time[d][i][k] * z[d, i, k] for k in C
                ),
                f"wait_{d}_{i}"
            )
    
    # Objective: total completion time
    truck_travel = gp.quicksum(
        T[i][j] * x[i, j] for (i, j) in x if (i, j) in x
    )
    waiting_time_sum = gp.quicksum(w[i] for i in N)
    model.setObjective(truck_travel + waiting_time_sum, GRB.MINIMIZE)
    
    # Solver parameters
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Callback function for logging intermediate solutions
    def callback_fn(model_param, where):
        if where == GRB.Callback.MIPSOL:
            obj = model_param.cbGet(GRB.Callback.MIPSOL_OBJ)
            
            # Retrieve current solution values
            x_vals = {}
            for (i, j) in x:
                x_vals[i, j] = model_param.cbGetSolution(x[i, j])
            
            z_vals = {}
            for (d, i, k) in z:
                z_vals[d, i, k] = model_param.cbGetSolution(z[d, i, k])
            
            t_vals = {}
            for k in t:
                t_vals[k] = model_param.cbGetSolution(t[k])
            
            w_vals = {}
            for i in w:
                w_vals[i] = model_param.cbGetSolution(w[i])
            
            # Build truck route
            truck_route = []
            for (i, j) in x:
                if x_vals[i, j] > 0.5:
                    truck_route.append([int(i), int(j)])
            
            # Build drone assignments
            drone_assignments = []
            for (d, i, k) in z:
                if z_vals[d, i, k] > 0.5:
                    drone_assignments.append({
                        "drone": int(d),
                        "dispatch_node": int(i),
                        "delivery_node": int(k)
                    })
            
            # Determine visited nodes
            visited_nodes = [depot_index]
            for k in C:
                if t_vals[k] > 0.5:
                    visited_nodes.append(k)
            
            # Build waiting times dictionary
            waiting_times = {}
            for i in visited_nodes:
                waiting_times[str(i)] = float(w_vals[i])
            
            # Prepare solution dictionary
            solution_dict = {
                "objective_value": float(obj),
                "truck_route": truck_route,
                "drone_assignments": drone_assignments,
                "waiting_times": waiting_times
            }
            
            # Log solution if logger available
            if logger:
                logger.log_solution(obj, solution_dict)
    
    # Solve model with callback if logging enabled
    model.optimize(callback_fn if logger else None)
    
    # Process final solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Extract truck route
        truck_route = []
        for (i, j) in x:
            if x[i, j].X > 0.5:
                truck_route.append([int(i), int(j)])
        
        # Extract drone assignments
        drone_assignments = []
        for (d, i, k) in z:
            if z[d, i, k].X > 0.5:
                drone_assignments.append({
                    "drone": int(d),
                    "dispatch_node": int(i),
                    "delivery_node": int(k)
                })
        
        # Determine visited nodes
        visited_nodes = [depot_index]
        for k in C:
            if t[k].X > 0.5:
                visited_nodes.append(k)
        
        # Extract waiting times
        waiting_times = {}
        for i in visited_nodes:
            waiting_times[str(i)] = float(w[i].X)
        
        # Prepare final solution dictionary
        solution_dict = {
            "objective_value": float(model.ObjVal),
            "truck_route": truck_route,
            "drone_assignments": drone_assignments,
            "waiting_times": waiting_times
        }
        
        # Write solution to file
        with open(solution_path, 'w') as f:
            json.dump(solution_dict, f)
    else:
        raise RuntimeError("No feasible solution found within time limit")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to problem instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to write solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Optional path for solution logging')
    args = parser.parse_args()
    
    main(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )