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

    # Build lookup structures
    # Trip IDs are 1-indexed
    trip_by_id = {}
    for t in trips:
        trip_by_id[t['trip_id']] = t

    # Sort trips by start time
    sorted_trips = sorted(trips, key=lambda t: t['start_time'])
    trip_ids = [t['trip_id'] for t in sorted_trips]

    # Compatible arcs: from_trip -> list of (to_trip, cost)
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

    # Depot to trip costs: (depot, trip) -> cost
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

    # Trip to depot costs: (trip, depot) -> cost
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

    # Use Gurobi to solve as a min-cost flow / set partitioning problem
    # Decision variables:
    # x[i,j] = 1 if trip i is followed by trip j on the same vehicle (for compatible arcs)
    # y[d,t] = 1 if a vehicle from depot d starts its duty with trip t
    # z[t,d] = 1 if a vehicle ends its duty at depot d after trip t
    
    # Constraints:
    # Each trip is covered exactly once
    # Flow conservation at each trip node
    # Depot capacity constraints
    # Each vehicle starts and ends at the same depot (handled by flow conservation per depot)

    # We model this as a network flow problem per depot.
    # For each depot d, we have:
    #   - source: depot d
    #   - sink: depot d
    #   - For each trip t: flow conservation
    #     sum of incoming (from depot d or from another trip) = sum of outgoing (to depot d or to another trip)
    #     and each trip is assigned to exactly one depot
    
    # Actually, this is a multi-commodity flow problem (one commodity per depot).
    # Let x_d[i,j] = 1 if on a vehicle from depot d, trip i is followed by trip j
    # Let y_d[t] = 1 if a vehicle from depot d starts with trip t
    # Let z_d[t] = 1 if a vehicle from depot d ends with trip t
    
    # This can be large. Let's try a simpler formulation first.
    # 
    # Alternative: We can use the formulation where we don't track depot identity in trip-to-trip arcs.
    # x[i,j] = 1 if trip i followed by trip j (same vehicle, any depot)
    # y[d,t] = 1 if vehicle from depot d starts with trip t
    # z[t,d] = 1 if vehicle returns to depot d after trip t
    #
    # Flow conservation at trip t:
    #   sum_d y[d,t] + sum_i x[i,t] = 1  (exactly one incoming)
    #   sum_d z[t,d] + sum_j x[t,j] = 1  (exactly one outgoing)
    #
    # For each depot d:
    #   sum_t y[d,t] = sum_t z[t,d]  (vehicles return to same depot)
    #   sum_t y[d,t] <= vehicle_capacities[d-1]
    #
    # But this doesn't guarantee that a vehicle starting from depot d returns to depot d.
    # It only ensures the total number leaving = total returning per depot.
    # For the single-depot case, this is fine. For multi-depot, this is a relaxation.
    # However, in practice, this relaxation is often tight for min-cost flow problems.
    # Actually, this is a well-known result: for the multi-depot vehicle scheduling problem,
    # the LP relaxation of this formulation is NOT always tight, and we may need to use
    # a multi-commodity formulation.
    
    # Let's check the problem size. If n_trips is small enough, we can use multi-commodity.
    # For large instances, we might need the simpler formulation and hope it's integral.
    
    # Let's try the multi-commodity formulation with Gurobi.
    # Variables: for each depot d, for each compatible arc (i,j): x_d[i,j]
    #           for each depot d, for each trip t: y_d[t] (pull-out), z_d[t] (pull-in)
    
    # This could be large: m_depots * n_compatible_arcs + 2 * m_depots * n_trips
    
    n_arcs = len(compatible_arcs)
    total_vars_estimate = m_depots * n_arcs + 2 * m_depots * n_trips
    
    elapsed = time.time() - start_time
    remaining = args.time_limit - elapsed
    
    # For very large instances, use the aggregated formulation
    # For smaller instances, use multi-commodity
    use_multicommodity = total_vars_estimate <= 500000 and n_trips <= 1000
    
    import gurobipy as gp
    from gurobipy import GRB
    
    if use_multicommodity:
        # Multi-commodity flow formulation
        model = gp.Model("MDVSP")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', max(remaining - 5, 10))
        model.setParam('Threads', 1)
        model.setParam('MIPFocus', 1)  # Focus on finding feasible solutions
        
        # Variables
        # y[d,t]: vehicle from depot d starts with trip t
        y = {}
        for d in range(1, m_depots + 1):
            for t_id in trip_ids:
                if (d, t_id) in d2t_cost:
                    y[d, t_id] = model.addVar(vtype=GRB.BINARY, 
                                               cost=d2t_cost[(d, t_id)],
                                               name=f"y_{d}_{t_id}")
        
        # z[t,d]: vehicle returns to depot d after trip t
        z = {}
        for t_id in trip_ids:
            for d in range(1, m_depots + 1):
                if (t_id, d) in t2d_cost:
                    z[t_id, d] = model.addVar(vtype=GRB.BINARY,
                                               cost=t2d_cost[(t_id, d)],
                                               name=f"z_{t_id}_{d}")
        
        # x[d,i,j]: on vehicle from depot d, trip i followed by trip j
        x = {}
        for arc in compatible_arcs:
            ft = arc['from_trip']
            tt = arc['to_trip']
            c = arc['cost']
            for d in range(1, m_depots + 1):
                x[d, ft, tt] = model.addVar(vtype=GRB.BINARY,
                                              cost=c,
                                              name=f"x_{d}_{ft}_{tt}")
        
        model.ModelSense = GRB.MINIMIZE
        model.update()
        
        # Set objective
        obj = gp.LinExpr()
        for (d, t_id), var in y.items():
            obj += d2t_cost[(d, t_id)] * var
        for (t_id, d), var in z.items():
            obj += t2d_cost[(t_id, d)] * var
        for (d, ft, tt), var in x.items():
            obj += arc_cost[(ft, tt)] * var
        model.setObjective(obj, GRB.MINIMIZE)
        
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
        
        # Optimize
        def solution_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj_val)
        
        model.optimize(solution_callback)
        
        if model.SolCount > 0:
            # Extract solution
            routes = []
            obj_val = model.ObjVal
            
            for d in range(1, m_depots + 1):
                # Find all trips that start from this depot
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
    
    # For larger instances or if multi-commodity is too large, use aggregated formulation
    # then do a post-processing to assign depots
    
    model = gp.Model("MDVSP_agg")
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    
    # First try: aggregated formulation (single commodity) + depot assignment
    # x[i,j] = 1 if trip i followed by trip j on some vehicle
    # y[d,t] = 1 if vehicle from depot d starts with trip t  
    # z[t,d] = 1 if vehicle returns to depot d after trip t
    
    x_vars = {}
    for arc in compatible_arcs:
        ft = arc['from_trip']
        tt = arc['to_trip']
        c = arc['cost']
        x_vars[ft, tt] = model.addVar(vtype=GRB.BINARY, obj=c, name=f"x_{ft}_{tt}")
    
    y_vars = {}
    for d in range(1, m_depots + 1):
        for t_id in trip_ids:
            if (d, t_id) in d2t_cost:
                y_vars[d, t_id] = model.addVar(vtype=GRB.BINARY, 
                                                 obj=d2t_cost[(d, t_id)],
                                                 name=f"y_{d}_{t_id}")
    
    z_vars = {}
    for t_id in trip_ids:
        for d in range(1, m_depots + 1):
            if (t_id, d) in t2d_cost:
                z_vars[t_id, d] = model.addVar(vtype=GRB.BINARY,
                                                 obj=t2d_cost[(t_id, d)],
                                                 name=f"z_{t_id}_{d}")
    
    model.ModelSense = GRB.MINIMIZE
    model.update()
    
    # Constraints
    # Each trip incoming = 1
    for t_id in trip_ids:
        lhs = gp.LinExpr()
        for d in range(1, m_depots + 1):
            if (d, t_id) in y_vars:
                lhs += y_vars[d, t_id]
        for (ft, c) in arcs_to.get(t_id, []):
            if (ft, t_id) in x_vars:
                lhs += x_vars[ft, t_id]
        model.addConstr(lhs == 1, name=f"in_{t_id}")
    
    # Each trip outgoing = 1
    for t_id in trip_ids:
        lhs = gp.LinExpr()
        for d in range(1, m_depots + 1):
            if (t_id, d) in z_vars:
                lhs += z_vars[t_id, d]
        for (tt, c) in arcs_from.get(t_id, []):
            if (t_id, tt) in x_vars:
                lhs += x_vars[t_id, tt]
        model.addConstr(lhs == 1, name=f"out_{t_id}")
    
    # Depot balance: vehicles leaving = vehicles returning per depot
    for d in range(1, m_depots + 1):
        lhs_out = gp.LinExpr()
        lhs_in = gp.LinExpr()
        for t_id in trip_ids:
            if (d, t_id) in y_vars:
                lhs_out += y_vars[d, t_id]
            if (t_id, d) in z_vars:
                lhs_in += z_vars[t_id, d]
        model.addConstr(lhs_out == lhs_in, name=f"balance_{d}")
    
    # Depot capacity
    for d in range(1, m_depots + 1):
        lhs = gp.LinExpr()
        for t_id in trip_ids:
            if (d, t_id) in y_vars:
                lhs += y_vars[d, t_id]
        model.addConstr(lhs <= vehicle_capacities[d - 1], name=f"cap_{d}")
    
    # For single depot case, this is exact. For multi-depot, this is a relaxation.
    # We need to add coupling constraints to ensure same depot for start/end of each chain.
    
    # If m_depots == 1, no need for extra constraints
    if m_depots > 1:
        # We need to ensure each chain starts and ends at the same depot.
        # The aggregated formulation doesn't guarantee this.
        # We need multi-commodity flow constraints.
        # Since we already tried multi-commodity above for small instances,
        # here we handle larger instances.
        
        # For larger instances with multiple depots, let's use a different approach:
        # Solve the aggregated LP relaxation, then use it to guide a heuristic.
        # Or, we can add the multi-commodity constraints lazily.
        
        # Actually, let's just solve the multi-commodity formulation but with LP relaxation
        # first, then branch-and-bound. Gurobi should handle this.
        
        # Let's reformulate as multi-commodity for larger instances too.
        # Reset model.
        model.reset()
        model = gp.Model("MDVSP_mc_large")
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)
        elapsed = time.time() - start_time
        remaining = args.time_limit - elapsed
        model.setParam('TimeLimit', max(remaining - 5, 10))
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
        
        for d in range(1, m_depots + 1):
            lhs = gp.LinExpr()
            for t_id in trip_ids:
                if (d, t_id) in y:
                    lhs += y[d, t_id]
            model.addConstr(lhs <= vehicle_capacities[d - 1], name=f"cap_{d}")
        
        def solution_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
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
        else:
            # Fallback: greedy solution
            pass
    else:
        # Single depot case - aggregated formulation is exact
        elapsed = time.time() - start_time
        remaining = args.time_limit - elapsed
        model.setParam('TimeLimit', max(remaining - 5, 10))
        
        def solution_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj_val)
        
        model.optimize(solution_callback)
        
        if model.SolCount > 0:
            routes = []
            obj_val = model.ObjVal
            d = 1  # single depot
            
            starting_trips = []
            for t_id in trip_ids:
                if (d, t_id) in y_vars and y_vars[d, t_id].X > 0.5:
                    starting_trips.append(t_id)
            
            # Build successor map
            successor = {}
            for (ft, tt), var in x_vars.items():
                if var.X > 0.5:
                    successor[ft] = tt
            
            for st in starting_trips:
                route = [st]
                current = st
                while current in successor:
                    current = successor[current]
                    route.append(current)
                routes.append({"depot": d, "trips": route})
            
            solution = {
                "objective_value": obj_val,
                "routes": routes,
                "num_vehicles": len(routes)
            }
            
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
            return
    
    # Fallback: greedy heuristic
    # Each trip gets its own vehicle from the cheapest depot
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