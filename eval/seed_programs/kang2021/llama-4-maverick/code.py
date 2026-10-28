import argparse
import json
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    num_customers = instance['num_customers']
    num_nodes = instance['num_nodes']
    depot_index = instance['depot_index']
    customer_indices = instance['customer_indices']
    truck_travel_times = instance['truck_travel_times']
    drone_battery_consumption = instance['drone_battery_consumption']
    drone_delivery_time = instance['drone_delivery_time']
    drone_battery_capacities = instance['parameters']['drone_battery_capacities']
    num_drones = instance['parameters']['num_drones']

    model = Model("Drone Delivery")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    # Decision Variables
    x = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name='x')  # Truck route arcs
    y = model.addVars(num_drones, num_nodes, customer_indices, vtype=GRB.BINARY, name='y')  # Drone assignments
    w = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, name='w')  # Truck waiting times
    z = model.addVars(customer_indices, vtype=GRB.BINARY, name='z')  # Customer served by truck
    u = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, name='u')  # Node visit order for subtour elimination

    # Objective: Minimize total completion time
    model.setObjective(
        quicksum(truck_travel_times[i][j] * x[i, j] for i in range(num_nodes) for j in range(num_nodes)) +
        quicksum(w[i] for i in range(num_nodes)),
        GRB.MINIMIZE
    )

    # Constraints
    # 1. Truck route starts and ends at depot
    model.addConstr(x.sum(depot_index, '*') == 1, name='leave_depot')
    model.addConstr(x.sum('*', num_nodes - 1) == 1, name='enter_sink')  # Assuming num_nodes - 1 is the virtual sink
    model.addConstr(x.sum('*', depot_index) == 0, name='no_enter_depot')
    model.addConstr(x.sum(num_nodes - 1, '*') == 0, name='no_leave_sink')  # Assuming num_nodes - 1 is the virtual sink

    # 2. Flow conservation for truck route
    for i in range(num_nodes):
        if i != depot_index and i != num_nodes - 1:  # Exclude depot and virtual sink
            model.addConstr(x.sum(i, '*') == x.sum('*', i), name=f'flow_conservation_{i}')

    # 3. Each customer is served exactly once
    for customer in customer_indices:
        model.addConstr(z[customer] + quicksum(y[d, i, customer] for d in range(num_drones) for i in range(num_nodes)) == 1, name=f'serve_customer_{customer}')

    # 4. Drone battery capacity constraints
    for d in range(num_drones):
        model.addConstr(quicksum(drone_battery_consumption[d][i][customer] * y[d, i, customer] for i in range(num_nodes) for customer in customer_indices) <= drone_battery_capacities[d], name=f'drone_battery_{d}')

    # 5. Waiting time at each node must be sufficient for all drones dispatched
    for i in range(num_nodes):
        model.addConstr(w[i] >= quicksum(drone_delivery_time[d][i][customer] * y[d, i, customer] for d in range(num_drones) for customer in customer_indices), name=f'waiting_time_{i}')

    # 6. Subtour elimination
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and i != num_nodes - 1 and j != depot_index:  # Exclude virtual sink and depot
                model.addConstr(u[i] - u[j] + num_nodes * x[i, j] <= num_nodes - 1, name=f'subtour_elimination_{i}_{j}')

    # 7. Node visit order constraints
    for i in range(num_nodes):
        if i != depot_index:
            model.addConstr(u[i] >= 1, name=f'node_order_lower_{i}')
            model.addConstr(u[i] <= num_nodes - 1, name=f'node_order_upper_{i}')

    # Optimize
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Extract solution
        truck_route = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if x[i, j].X > 0.5]
        drone_assignments = [{'drone': d, 'dispatch_node': i, 'delivery_node': customer} for d in range(num_drones) for i in range(num_nodes) for customer in customer_indices if y[d, i, customer].X > 0.5]
        waiting_times = {str(i): w[i].X for i in range(num_nodes) if w[i].X > 0}

        solution = {
            'objective_value': model.ObjVal,
            'truck_route': truck_route,
            'drone_assignments': drone_assignments,
            'waiting_times': waiting_times
        }

        return solution
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model = Model("Drone Delivery")
    model.Params.TimeLimit = args.time_limit

    solution = solve(instance)

    if solution:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

        if logger:
            logger.log_solution(solution['objective_value'], solution)

if __name__ == "__main__":
    main()