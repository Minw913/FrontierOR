import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    num_customers = data["num_customers"]
    num_nodes = data["num_nodes"]
    depot_index = data["depot_index"]
    customer_indices = data["customer_indices"]
    truck_travel_times = data["truck_travel_times"]
    drone_battery_consumption = data["drone_battery_consumption"]
    drone_delivery_time = data["drone_delivery_time"]
    
    params = data["parameters"]
    num_drones = params["num_drones"]
    capacities = params["drone_battery_capacities"]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1)
    env.start()
    
    model = gp.Model("HeterogeneousDroneDelivery", env=env)
    
    # Required Gurobi parameters
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('LazyConstraints', 1)

    # ------------------
    # Variables
    # ------------------
    # x[i, j]: 1 if truck travels from node i to node j
    x = {}
    for i in range(num_nodes):
        for j in range(num_nodes + 1):
            if i == j: 
                continue
            if j == depot_index: 
                continue # Truck must return to virtual sink, not start depot
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            
    # z[i]: 1 if truck visits node i
    z = {}
    for i in range(num_nodes):
        z[i] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}")
        
    # y[c, d, k]: 1 if drone k serves customer c from dispatch node d
    y = {}
    for c in customer_indices:
        for d in range(num_nodes):
            for k in range(num_drones):
                y[c, d, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}_{d}_{k}")
                
    # W[d]: waiting time at visited node d
    W = {}
    for d in range(num_nodes):
        W[d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"W_{d}")

    # ------------------
    # Constraints
    # ------------------
    # 1. Depot is always visited by the truck
    model.addConstr(z[depot_index] == 1, name="depot_visited")

    # 2. Truck flow constraints
    for i in range(num_nodes):
        # Outgoing from i
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_nodes + 1) if (i, j) in x) == z[i], name=f"flow_out_{i}")
        
    for j in range(num_nodes + 1):
        if j == depot_index: 
            continue
        # Incoming to j
        if j == num_nodes: # Virtual sink
            model.addConstr(gp.quicksum(x[i, j] for i in range(num_nodes) if (i, j) in x) == 1, name=f"flow_in_{j}")
        else:
            model.addConstr(gp.quicksum(x[i, j] for i in range(num_nodes) if (i, j) in x) == z[j], name=f"flow_in_{j}")

    # 3. Every customer is served exactly once (by truck or by exactly one drone)
    for c in customer_indices:
        model.addConstr(z[c] + gp.quicksum(y[c, d, k] for d in range(num_nodes) for k in range(num_drones)) == 1, name=f"serve_{c}")
        
    # 4. Drones can only be dispatched from nodes the truck visits
    for c in customer_indices:
        for d in range(num_nodes):
            for k in range(num_drones):
                model.addConstr(y[c, d, k] <= z[d], name=f"dispatch_{c}_{d}_{k}")

    # 5. Drone battery capacity limits over all assigned deliveries
    for k in range(num_drones):
        model.addConstr(
            gp.quicksum(drone_battery_consumption[k][d][c] * y[c, d, k] 
                        for c in customer_indices for d in range(num_nodes)) <= capacities[k],
            name=f"battery_{k}"
        )
        
    # 6. Waiting time at each dispatch node bounds the longest active drone operation
    for d in range(num_nodes):
        for k in range(num_drones):
            model.addConstr(
                W[d] >= gp.quicksum(drone_delivery_time[k][d][c] * y[c, d, k] for c in customer_indices),
                name=f"wait_{d}_{k}"
            )

    # ------------------
    # Objective
    # ------------------
    def get_truck_time(i, j):
        if j == num_nodes: # Virtual sink represents the return depot
            return truck_travel_times[i][depot_index]
        return truck_travel_times[i][j]

    obj = gp.quicksum(get_truck_time(i, j) * x[i, j] for i, j in x.keys()) + gp.quicksum(W[d] for d in range(num_nodes))
    model.setObjective(obj, GRB.MINIMIZE)

    # Context references for the callback
    model._x = x
    model._y = y
    model._z = z
    model._W = W
    model._num_nodes = num_nodes
    model._depot_index = depot_index
    model._logger = logger

    # ------------------
    # Callback
    # ------------------
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                x_vals = model.cbGetSolution(model._x)
                
                edges = [(i, j) for (i, j), val in x_vals.items() if val > 0.5]
                out_node = {i: j for i, j in edges}
                
                visited = set()
                subtours = []
                
                # Subtour detection logic
                for i in out_node:
                    if i not in visited:
                        curr = i
                        path = []
                        while curr not in visited and curr in out_node:
                            visited.add(curr)
                            path.append(curr)
                            curr = out_node[curr]
                            
                        if curr in path:
                            idx = path.index(curr)
                            cycle = path[idx:]
                            subtours.append(cycle)
                            
                if subtours:
                    # Enforce DFJ subtour elimination cuts if any cycles are detected
                    for cycle in subtours:
                        model.cbLazy(gp.quicksum(model._x[u, v] for u in cycle for v in cycle if u != v) <= len(cycle) - 1)
                else:
                    # Solution is completely feasible, so log it
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if model._logger is not None:
                        y_vals = model.cbGetSolution(model._y)
                        W_vals = model.cbGetSolution(model._W)
                        z_vals = model.cbGetSolution(model._z)
                        
                        truck_route = [[i, j] for (i, j) in edges]
                            
                        drone_assignments = []
                        for (c, d, k), val in y_vals.items():
                            if val > 0.5:
                                drone_assignments.append({
                                    "drone": k,
                                    "dispatch_node": d,
                                    "delivery_node": c
                                })
                                
                        waiting_times = {}
                        for d in range(model._num_nodes):
                            if z_vals[d] > 0.5:
                                waiting_times[str(d)] = max(0.0, float(W_vals[d]))
                                
                        sol_dict = {
                            "objective_value": float(obj_val),
                            "truck_route": truck_route,
                            "drone_assignments": drone_assignments,
                            "waiting_times": waiting_times
                        }
                        model._logger.log_solution(float(obj_val), sol_dict)
            except Exception:
                pass

    model.optimize(callback)

    # ------------------
    # Output Best Solution
    # ------------------
    if model.SolCount > 0:
        x_vals = {k: v.X for k, v in x.items()}
        y_vals = {k: v.X for k, v in y.items()}
        W_vals = {k: v.X for k, v in W.items()}
        z_vals = {k: v.X for k, v in z.items()}
        
        edges = [(i, j) for (i, j), val in x_vals.items() if val > 0.5]
        truck_route = [[i, j] for i, j in edges]
        
        drone_assignments = []
        for (c, d, k), val in y_vals.items():
            if val > 0.5:
                drone_assignments.append({
                    "drone": k,
                    "dispatch_node": d,
                    "delivery_node": c
                })
                
        waiting_times = {}
        for d in range(num_nodes):
            if z_vals[d] > 0.5:
                waiting_times[str(d)] = max(0.0, float(W_vals[d]))
                
        sol_dict = {
            "objective_value": float(model.ObjVal),
            "truck_route": truck_route,
            "drone_assignments": drone_assignments,
            "waiting_times": waiting_times
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(sol_dict, f, indent=2)

if __name__ == "__main__":
    solve()