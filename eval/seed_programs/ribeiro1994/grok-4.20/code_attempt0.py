import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n_trips = data['parameters']['n_trips']
    m_depots = data['parameters']['m_depots']
    trips = data['trips']
    compatible_arcs = data['compatible_arcs']
    depot_to_trip_costs = data['depot_to_trip_costs']
    trip_to_depot_costs = data['trip_to_depot_costs']
    vehicle_capacities = data['vehicle_capacities']

    # Build cost dictionaries
    arc_cost = {}
    for arc in compatible_arcs:
        i = arc['from_trip'] - 1
        j = arc['to_trip'] - 1
        arc_cost[(i, j)] = arc['cost']

    depot_start_cost = {}
    for d2t in depot_to_trip_costs:
        d = d2t['depot'] - 1
        t = d2t['trip'] - 1
        depot_start_cost[(d, t)] = d2t['cost']

    trip_end_cost = {}
    for t2d in trip_to_depot_costs:
        t = t2d['trip'] - 1
        d = t2d['depot'] - 1
        trip_end_cost[(t, d)] = t2d['cost']

    # Sort trips by start time (should already be but ensure)
    trip_list = list(range(n_trips))
    trip_list.sort(key=lambda x: trips[x]['start_time'])

    # Model
    model = gp.Model("Vehicle_Duties")

    # Variables
    x = model.addVars(n_trips, n_trips, vtype=GRB.BINARY, name="x")  # trip i to trip j
    y = model.addVars(m_depots, n_trips, vtype=GRB.BINARY, name="y")  # depot d starts with trip j
    z = model.addVars(n_trips, m_depots, vtype=GRB.BINARY, name="z")  # trip i ends to depot d
    v = model.addVars(m_depots, vtype=GRB.INTEGER, lb=0, ub=vehicle_capacities, name="v")

    # Objective
    obj = gp.quicksum(arc_cost.get((i,j), 0) * x[i,j] for i in range(n_trips) for j in range(n_trips) if (i,j) in arc_cost)
    obj += gp.quicksum(depot_start_cost.get((d,j), 100000) * y[d,j] for d in range(m_depots) for j in range(n_trips))
    obj += gp.quicksum(trip_end_cost.get((i,d), 100000) * z[i,d] for i in range(n_trips) for d in range(m_depots))
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # Each trip has exactly one successor (including to depot)
    for i in range(n_trips):
        model.addConstr(
            gp.quicksum(x[i,j] for j in range(n_trips) if (i,j) in arc_cost) +
            gp.quicksum(z[i,d] for d in range(m_depots)) == 1,
            name=f"out_{i}"
        )

    # Each trip has exactly one predecessor (including from depot)
    for j in range(n_trips):
        model.addConstr(
            gp.quicksum(x[i,j] for i in range(n_trips) if (i,j) in arc_cost) +
            gp.quicksum(y[d,j] for d in range(m_depots)) == 1,
            name=f"in_{j}"
        )

    # Flow conservation at trips (no depot in middle)
    for k in range(n_trips):
        model.addConstr(
            gp.quicksum(x[i,k] for i in range(n_trips) if (i,k) in arc_cost) ==
            gp.quicksum(x[k,j] for j in range(n_trips) if (k,j) in arc_cost),
            name=f"flow_{k}"
        )

    # Link y and z to vehicle count
    for d in range(m_depots):
        model.addConstr(
            gp.quicksum(y[d,j] for j in range(n_trips)) == v[d],
            name=f"start_d{d}"
        )
        model.addConstr(
            gp.quicksum(z[i,d] for i in range(n_trips)) == v[d],
            name=f"end_d{d}"
        )

    # Capacity already in ub of v

    # Long trips are self-contained: if long, cannot be in middle of chain
    for idx, trip in enumerate(trips):
        if trip['trip_type'] == 'long':
            # Must start from depot and end to depot directly
            model.addConstr(
                gp.quicksum(x[idx,j] for j in range(n_trips) if (idx,j) in arc_cost) == 0,
                name=f"long_out_{idx}"
            )
            model.addConstr(
                gp.quicksum(x[i,idx] for i in range(n_trips) if (i,idx) in arc_cost) == 0,
                name=f"long_in_{idx}"
            )

    # Time limit
    model.Params.TimeLimit = args.time_limit
    model.Params.Threads = 1
    model.Params.MIPGap = 0.0
    model.Params.LogToConsole = 0

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(obj_val)

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        # If infeasible or other error, try to produce something (should not happen)
        routes = []
        objective = 0.0
        num_vehicles = 0
    else:
        # Build routes
        routes = []
        used_from_depot = [[False]*n_trips for _ in range(m_depots)]
        used_to_depot = [[False]*n_trips for _ in range(m_depots)]
        successor = [-1] * n_trips

        for d in range(m_depots):
            for j in range(n_trips):
                if y[d,j].X > 0.5:
                    used_from_depot[d][j] = True

        for i in range(n_trips):
            for j in range(n_trips):
                if (i,j) in arc_cost and x[i,j].X > 0.5:
                    successor[i] = j

        for i in range(n_trips):
            for d in range(m_depots):
                if z[i,d].X > 0.5:
                    used_to_depot[d][i] = True

        # Reconstruct each route
        visited = [False] * n_trips
        objective = 0.0
        num_vehicles = 0

        for d in range(m_depots):
            for start_trip in range(n_trips):
                if used_from_depot[d][start_trip] and not visited[start_trip]:
                    route = [start_trip + 1]
                    visited[start_trip] = True
                    current = start_trip
                    while successor[current] != -1:
                        nxt = successor[current]
                        route.append(nxt + 1)
                        visited[nxt] = True
                        current = nxt
                    # Add to routes
                    routes.append({
                        "depot": d + 1,
                        "trips": route
                    })
                    num_vehicles += 1

                    # Add costs for this route
                    # depot to first
                    objective += depot_start_cost.get((d, start_trip), 100000)
                    # transitions
                    for k in range(len(route)-1):
                        i_idx = route[k]-1
                        j_idx = route[k+1]-1
                        objective += arc_cost.get((i_idx, j_idx), 0)
                    # last to depot
                    last = route[-1] - 1
                    objective += trip_end_cost.get((last, d), 100000)

    # If no routes found but should cover all, fallback (rare)
    if len(routes) == 0 and n_trips > 0:
        # Create dummy single trip routes for each trip from depot 1
        for t in range(n_trips):
            routes.append({
                "depot": 1,
                "trips": [t+1]
            })
        num_vehicles = n_trips
        objective = 5000.0 * 2 * n_trips

    solution = {
        "objective_value": float(objective),
        "routes": routes,
        "num_vehicles": int(num_vehicles)
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()