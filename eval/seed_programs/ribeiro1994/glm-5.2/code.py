import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    n = instance['parameters']['n_trips']
    m = instance['parameters']['m_depots']
    vehicle_capacities = instance['vehicle_capacities']

    # Precompute costs and arcs
    depot_to_trip_cost = {}
    for item in instance['depot_to_trip_costs']:
        d = item['depot']
        t = item['trip']
        depot_to_trip_cost[(d, t)] = item['cost']
        
    trip_to_depot_cost = {}
    for item in instance['trip_to_depot_costs']:
        t = item['trip']
        d = item['depot']
        trip_to_depot_cost[(t, d)] = item['cost']
        
    compat_cost = {}
    incoming_arcs = {i: [] for i in range(1, n + 1)}
    outgoing_arcs = {i: [] for i in range(1, n + 1)}
    for item in instance['compatible_arcs']:
        u = item['from_trip']
        v = item['to_trip']
        compat_cost[(u, v)] = item['cost']
        outgoing_arcs[u].append(v)
        incoming_arcs[v].append(u)
        
    model = gp.Model("MDVSP")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit
    
    # Variables
    z = {}
    for i in range(1, n + 1):
        for d in range(1, m + 1):
            z[i, d] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{d}")
            
    x = {}
    for (u, v), c in compat_cost.items():
        for d in range(1, m + 1):
            x[u, v, d] = model.addVar(vtype=GRB.BINARY, obj=c, name=f"x_{u}_{v}_{d}")
            
    y_out = {}
    for d in range(1, m + 1):
        for i in range(1, n + 1):
            c = depot_to_trip_cost.get((d, i), 1000000)
            y_out[d, i] = model.addVar(vtype=GRB.BINARY, obj=c, name=f"y_out_{d}_{i}")
            
    y_in = {}
    for d in range(1, m + 1):
        for i in range(1, n + 1):
            c = trip_to_depot_cost.get((i, d), 1000000)
            y_in[i, d] = model.addVar(vtype=GRB.BINARY, obj=c, name=f"y_in_{i}_{d}")
            
    # Constraints
    # 1. Each trip assigned to exactly one depot
    for i in range(1, n + 1):
        model.addConstr(gp.quicksum(z[i, d] for d in range(1, m + 1)) == 1)
        
    # 2. Flow conservation
    for d in range(1, m + 1):
        for i in range(1, n + 1):
            incoming = gp.quicksum(x[u, i, d] for u in incoming_arcs[i])
            outgoing = gp.quicksum(x[i, v, d] for v in outgoing_arcs[i])
            
            model.addConstr(incoming + y_out[d, i] == z[i, d])
            model.addConstr(outgoing + y_in[i, d] == z[i, d])
            
    # 3. Depot capacities
    for d in range(1, m + 1):
        model.addConstr(gp.quicksum(y_out[d, i] for i in range(1, n + 1)) <= vehicle_capacities[d - 1])
        model.addConstr(gp.quicksum(y_out[d, i] for i in range(1, n + 1)) == gp.quicksum(y_in[i, d] for i in range(1, n + 1)))
        
    # 4. Tighten LP relaxation
    for (u, v) in compat_cost:
        for d in range(1, m + 1):
            model.addConstr(x[u, v, d] <= z[u, d])
            model.addConstr(x[u, v, d] <= z[v, d])

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(callback)

    if model.SolCount == 0:
        solution = {
            "objective_value": 0.0,
            "routes": [],
            "num_vehicles": 0
        }
    else:
        obj_val = model.ObjVal
        routes = []
        for d in range(1, m + 1):
            for i in range(1, n + 1):
                if y_out[d, i].X > 0.5:
                    route_trips = [i]
                    curr = i
                    while True:
                        next_trip = None
                        for v in outgoing_arcs[curr]:
                            if x[curr, v, d].X > 0.5:
                                next_trip = v
                                break
                        if next_trip is None:
                            break
                        route_trips.append(next_trip)
                        curr = next_trip
                    routes.append({"depot": d, "trips": route_trips})
        
        solution = {
            "objective_value": obj_val,
            "routes": routes,
            "num_vehicles": len(routes)
        }

    if logger:
        logger.log_solution(solution["objective_value"], solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()