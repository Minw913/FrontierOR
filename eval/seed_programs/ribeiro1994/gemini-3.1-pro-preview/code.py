import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    args = parser.parse_args()

    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    # Extract dimensions and entities
    m_depots = instance["parameters"]["m_depots"]
    D = list(range(1, m_depots + 1))
    
    # Map vehicle capacities to depot 1-indexed identifiers
    V_list = instance.get("vehicle_capacities", [])
    V = {k: cap for k, cap in zip(D, V_list)}
    
    T = [trip["trip_id"] for trip in instance["trips"]]

    # Build unique compatible arcs and cost maps
    seen_arcs = set()
    arc_list = []
    arc_costs = {}
    out_arcs = {i: [] for i in T}
    
    for arc in instance.get("compatible_arcs", []):
        u = arc["from_trip"]
        v = arc["to_trip"]
        c = arc["cost"]
        if (u, v) not in seen_arcs:
            seen_arcs.add((u, v))
            arc_list.append((u, v))
            arc_costs[(u, v)] = c
            out_arcs[u].append(v)
            
    cd_costs = {}
    for item in instance.get("depot_to_trip_costs", []):
        cd_costs[(item["depot"], item["trip"])] = item["cost"]

    cr_costs = {}
    for item in instance.get("trip_to_depot_costs", []):
        cr_costs[(item["trip"], item["depot"])] = item["cost"]

    # Initialize Gurobi model (Multi-Commodity Flow Formulation)
    model = gp.Model("MDVSP")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.ModelSense = GRB.MINIMIZE
    
    # ------------------ Decisions Variables ------------------ #
    
    # y[k, i] = 1 if a vehicle from depot k goes directly to trip i to start a route
    y_idx = list(cd_costs.keys())
    y = model.addVars(y_idx, vtype=GRB.BINARY, obj=cd_costs, name="y")

    # z[i, k] = 1 if a vehicle goes directly from trip i to return to depot k
    z_idx = list(cr_costs.keys())
    z = model.addVars(z_idx, vtype=GRB.BINARY, obj=cr_costs, name="z")

    # x[k, u, v] = 1 if arc (u, v) is traversed by a vehicle originating from depot k
    x_idx = [(k, u, v) for k in D for u, v in arc_list]
    x_obj = { (k, u, v): arc_costs[(u, v)] for k, u, v in x_idx }
    x = model.addVars(x_idx, vtype=GRB.BINARY, obj=x_obj, name="x")

    # ---------------------- Constraints ---------------------- #

    # 1. Exact Covering: Every trip must be covered by exactly one vehicle across all depots
    for i in T:
        model.addConstr(
            y.sum('*', i) + x.sum('*', '*', i) == 1,
            name=f"cov_{i}"
        )

    # 2. Flow Conservation: What enters a trip bounds to the identical depot it departs from
    for k in D:
        for i in T:
            y_val = y.get((k, i), 0)
            z_val = z.get((i, k), 0)
            
            model.addConstr(
                y_val + x.sum(k, '*', i) == z_val + x.sum(k, i, '*'),
                name=f"flow_{k}_{i}"
            )

    # 3. Depot Capacity: A depot cannot exceed its available quantity of standalone vehicles
    for k in D:
        model.addConstr(
            y.sum(k, '*') <= V.get(k, 0),
            name=f"cap_{k}"
        )

    # ---------------------- Optimization ---------------------- #

    def cb(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    model.optimize(cb)

    # --------------------- Output Parsing --------------------- #
    
    if model.SolCount > 0:
        routes = []
        num_vehicles = 0
        
        # Sequentially map vehicle deployments to chronologically performed trips
        for k in D:
            start_trips = []
            for i in T:
                var = y.get((k, i))
                if var is not None and var.X > 0.5:
                    start_trips.append(i)
            
            for start_trip in start_trips:
                route = [start_trip]
                curr = start_trip
                while True:
                    next_trip = None
                    for j in out_arcs[curr]:
                        var = x.get((k, curr, j))
                        if var is not None and var.X > 0.5:
                            next_trip = j
                            break
                    if next_trip is not None:
                        route.append(next_trip)
                        curr = next_trip
                    else:
                        break
                
                routes.append({
                    "depot": k,
                    "trips": route
                })
                num_vehicles += 1
                
        solution = {
            "objective_value": float(model.ObjVal),
            "routes": routes,
            "num_vehicles": num_vehicles
        }
        
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()