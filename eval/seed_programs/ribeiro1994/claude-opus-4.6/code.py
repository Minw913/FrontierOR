import argparse
import json
import math
import time
import sys
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n_trips = instance['parameters']['n_trips']
    m_depots = instance['parameters']['m_depots']
    trips = instance['trips']
    compatible_arcs = instance['compatible_arcs']
    depot_to_trip_costs = instance['depot_to_trip_costs']
    trip_to_depot_costs = instance['trip_to_depot_costs']
    vehicle_capacities = instance['vehicle_capacities']

    trip_by_id = {}
    for t in trips:
        trip_by_id[t['trip_id']] = t

    sorted_trips = sorted(trips, key=lambda t: t['start_time'])
    trip_ids = [t['trip_id'] for t in sorted_trips]

    arc_cost = {}
    arcs_from = defaultdict(list)
    arcs_to = defaultdict(list)
    for arc in compatible_arcs:
        ft = arc['from_trip']
        tt = arc['to_trip']
        c = arc['cost']
        arc_cost[(ft, tt)] = c
        arcs_from[ft].append((tt, c))
        arcs_to[tt].append((ft, c))

    d2t_cost = {}
    d2t_by_trip = defaultdict(list)
    d2t_by_depot = defaultdict(list)
    for dtc in depot_to_trip_costs:
        d = dtc['depot']
        t = dtc['trip']
        c = dtc['cost']
        d2t_cost[(d, t)] = c
        d2t_by_trip[t].append((d, c))
        d2t_by_depot[d].append((t, c))

    t2d_cost = {}
    t2d_by_trip = defaultdict(list)
    t2d_by_depot = defaultdict(list)
    for tdc in trip_to_depot_costs:
        t = tdc['trip']
        d = tdc['depot']
        c = tdc['cost']
        t2d_cost[(t, d)] = c
        t2d_by_trip[t].append((d, c))
        t2d_by_depot[d].append((t, c))

    n_arcs = len(compatible_arcs)
    total_vars_estimate = m_depots * n_arcs + 2 * m_depots * n_trips

    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed

    import gurobipy as gp
    from gurobipy import GRB

    # Always use multi-commodity formulation for correctness
    model = gp.Model("MDVSP")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', max(remaining - 5, 10))
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)

    # Variables
    y = {}
    for d in range(1, m_depots + 1):
        for t_id in trip_ids:
            if (d, t_id) in d2t_cost:
                y[d, t_id] = model.addVar(vtype=GRB.BINARY,
                                           obj=d2t_cost[(d, t_id)],
                                           name=f"y_{d}_{t_id}")

    z = {}
    for t_id in trip_ids:
        for d in range(1, m_depots + 1):
            if (t_id, d) in t2d_cost:
                z[t_id, d] = model.addVar(vtype=GRB.BINARY,
                                           obj=t2d_cost[(t_id, d)],
                                           name=f"z_{t_id}_{d}")

    x = {}
    for arc in compatible_arcs:
        ft = arc['from_trip']
        tt = arc['to_trip']
        c = arc['cost']
        for d in range(1, m_depots + 1):
            x[d, ft, tt] = model.addVar(vtype=GRB.BINARY,
                                          obj=c,
                                          name=f"x_{d}_{ft}_{tt}")

    model.ModelSense = GRB.MINIMIZE
    model.update()

    # Constraints
    # 1. Each trip covered exactly once (incoming)
    for t_id in trip_ids:
        lhs = gp.LinExpr()
        for d in range(1, m_depots + 1):
            if (d, t_id) in y:
                lhs += y[d, t_id]
        for (ft, c) in arcs_to.get(t_id, []):
            for d in range(1, m_depots + 1):
                if (d, ft, t_id) in x:
                    lhs += x[d, ft, t_id]
        model.addConstr(lhs == 1, name=f"cover_in_{t_id}")

    # 2. Each trip covered exactly once (outgoing)
    for t_id in trip_ids:
        lhs = gp.LinExpr()
        for d in range(1, m_depots + 1):
            if (t_id, d) in z:
                lhs += z[t_id, d]
        for (tt, c) in arcs_from.get(t_id, []):
            for d in range(1, m_depots + 1):
                if (d, t_id, tt) in x:
                    lhs += x[d, t_id, tt]
        model.addConstr(lhs == 1, name=f"cover_out_{t_id}")

    # 3. Flow conservation per depot per trip
    for d in range(1, m_depots + 1):
        for t_id in trip_ids:
            incoming = gp.LinExpr()
            outgoing = gp.LinExpr()
            if (d, t_id) in y:
                incoming += y[d, t_id]
            for (ft, c) in arcs_to.get(t_id, []):
                if (d, ft, t_id) in x:
                    incoming += x[d, ft, t_id]
            if (t_id, d) in z:
                outgoing += z[t_id, d]
            for (tt, c) in arcs_from.get(t_id, []):
                if (d, t_id, tt) in x:
                    outgoing += x[d, t_id, tt]
            model.addConstr(incoming == outgoing, name=f"flow_{d}_{t_id}")

    # 4. Depot capacity
    for d in range(1, m_depots + 1):
        lhs = gp.LinExpr()
        for t_id in trip_ids:
            if (d, t_id) in y:
                lhs += y[d, t_id]
        model.addConstr(lhs <= vehicle_capacities[d - 1], name=f"cap_{d}")

    best_obj = [float('inf')]

    def solution_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0]:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)

    model.optimize(solution_callback)

    if model.SolCount > 0:
        routes = []
        obj_val = model.ObjVal

        for d in range(1, m_depots + 1):
            starting_trips = []
            for t_id in trip_ids:
                if (d, t_id) in y and y[d, t_id].X > 0.5:
                    starting_trips.append(t_id)

            for st in starting_trips:
                route = [st]
                current = st
                while True:
                    found_next = False
                    for (tt, c) in arcs_from.get(current, []):
                        if (d, current, tt) in x and x[d, current, tt].X > 0.5:
                            route.append(tt)
                            current = tt
                            found_next = True
                            break
                    if not found_next:
                        break
                routes.append({"depot": d, "trips": route})

        solution = {
            "objective_value": obj_val,
            "routes": routes,
            "num_vehicles": len(routes)
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # Fallback: greedy heuristic - each trip gets its own vehicle from cheapest depot
    routes = []
    total_cost = 0
    depot_usage = [0] * m_depots

    for t_id in trip_ids:
        best_d = None
        best_cost = float('inf')
        for (d, c) in d2t_by_trip.get(t_id, []):
            ret_cost = t2d_cost.get((t_id, d), None)
            if ret_cost is not None and depot_usage[d-1] < vehicle_capacities[d-1]:
                tc = c + ret_cost
                if tc < best_cost:
                    best_cost = tc
                    best_d = d
        if best_d is not None:
            routes.append({"depot": best_d, "trips": [t_id]})
            total_cost += best_cost
            depot_usage[best_d - 1] += 1

    if logger:
        logger.log(total_cost)

    solution = {
        "objective_value": total_cost,
        "routes": routes,
        "num_vehicles": len(routes)
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()