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
        instance = json.load(f)

    network = instance['network']
    time_params = instance['time_parameters']
    demand_data = instance['demand']
    fleet = instance['fleet']

    T = time_params['time_horizon_T']
    nodes = network['nodes']
    links = network['links']

    centroids = [n['id'] for n in nodes if n['type'] == 'centroid']
    junctions = [n['id'] for n in nodes if n['type'] == 'junction']
    all_nodes = [n['id'] for n in nodes]

    # Build graph
    successors = {u: [] for u in all_nodes}
    predecessors = {v: [] for v in all_nodes}
    link_dict = {}
    for link in links:
        u = link['from']
        v = link['to']
        link_id = link['id']
        successors[u].append((v, link_id))
        predecessors[v].append((u, link_id))
        link_dict[(u, v)] = link

    # OD pairs and demand
    od_pairs = []
    demand = {}
    for od in demand_data['od_pairs']:
        o = od['origin']
        d = od['destination']
        num = od['num_travelers']
        deps = od['departure_times']
        od_pairs.append((o, d))
        for t in deps:
            key = (o, d, t)
            demand[key] = demand.get(key, 0) + 1

    # Initial fleet
    initial_parked = fleet['initial_distribution']
    total_vehicles = fleet['total_fleet_size']

    # Create model
    model = gp.Model("SAV_LTM")
    model.setParam('TimeLimit', args.time_limit - 5)
    model.setParam('LogToConsole', 0)
    model.setParam('MIPGap', 0.01)
    model.setParam('Threads', 1)

    # Variables
    # y: turning flows on all links including connectors
    y = {}
    for link in links:
        u = link['from']
        v = link['to']
        fft = link['free_flow_travel_time_steps']
        is_road = link['type'] == 'road'
        for succ_tup in successors.get(v, []):
            w, _ = succ_tup
            for dest in centroids:
                for t in range(T):
                    if is_road and t < fft:
                        y[(u, v, w, dest, t)] = 0.0
                        continue
                    y[(u, v, w, dest, t)] = model.addVar(
                        lb=0.0, ub=GRB.INFINITY, name=f"y_{u}_{v}_{w}_{dest}_{t}"
                    )

    # y_centroid: departures from centroids
    y_cent = {}
    for cent in centroids:
        for succ_tup in successors.get(cent, []):
            junc, _ = succ_tup
            for dest in centroids:
                for t in range(T):
                    y_cent[(cent, junc, dest, t)] = model.addVar(
                        lb=0.0, ub=GRB.INFINITY, name=f"yc_{cent}_{junc}_{dest}_{t}"
                    )

    # N_U, N_D
    N_U = {}
    N_D = {}
    for link in links:
        u = link['from']
        v = link['to']
        for dest in centroids:
            for t in range(T + 1):
                N_U[(u, v, dest, t)] = model.addVar(
                    lb=0.0, ub=GRB.INFINITY, name=f"NU_{u}_{v}_{dest}_{t}"
                )
                N_D[(u, v, dest, t)] = model.addVar(
                    lb=0.0, ub=GRB.INFINITY, name=f"ND_{u}_{v}_{dest}_{t}"
                )

    # p: parked vehicles
    p = {}
    for cent in centroids:
        for t in range(T + 1):
            p[(cent, t)] = model.addVar(
                lb=0.0, ub=GRB.INFINITY, name=f"p_{cent}_{t}"
            )

    # e: pickups
    e = {}
    for o, d in od_pairs:
        for t in range(T):
            e[(o, d, t)] = model.addVar(
                lb=0.0, ub=GRB.INFINITY, name=f"e_{o}_{d}_{t}"
            )

    # omega: waiting travelers
    omega = {}
    for o, d in od_pairs:
        for t in range(T + 1):
            omega[(o, d, t)] = model.addVar(
                lb=0.0, ub=GRB.INFINITY, name=f"om_{o}_{d}_{t}"
            )

    model.update()

    # Objective: total vehicle-time + waiting time
    vehicle_time = gp.LinExpr()
    waiting_time = gp.LinExpr()
    for link in links:
        u = link['from']
        v = link['to']
        if link['type'] != 'road':
            continue
        for dest in centroids:
            for t in range(T):
                vehicle_time.add(N_U[(u, v, dest, t)] - N_D[(u, v, dest, t)])
    for o, d in od_pairs:
        for t in range(T):
            waiting_time.add(omega[(o, d, t)])
    model.setObjective(vehicle_time + waiting_time, GRB.MINIMIZE)

    # Constraints

    # 1. N_U and N_D initialization and dynamics
    for link in links:
        u = link['from']
        v = link['to']
        fft = link['free_flow_travel_time_steps']
        is_road = link['type'] == 'road'
        for dest in centroids:
            # t=0
            model.addConstr(N_U[(u, v, dest, 0)] == 0)
            model.addConstr(N_D[(u, v, dest, 0)] == 0)

            for t in range(T):
                # N_U[t+1] = N_U[t] + sum y from this link to successors at t
                inflow = gp.LinExpr()
                for succ_tup in successors.get(v, []):
                    w, _ = succ_tup
                    key = (u, v, w, dest, t)
                    if key in y:
                        inflow.add(y[key])
                model.addConstr(N_U[(u, v, dest, t + 1)] == N_U[(u, v, dest, t)] + inflow)

                # N_D dynamics
                if not is_road:
                    # centroid connector: downstream = upstream delayed by 1
                    model.addConstr(N_D[(u, v, dest, t + 1)] == N_U[(u, v, dest, t)])
                else:
                    # road link
                    if t + 1 - fft >= 0:
                        arrival = N_U[(u, v, dest, t + 1 - fft)]
                    else:
                        arrival = 0
                    model.addConstr(N_D[(u, v, dest, t + 1)] == N_D[(u, v, dest, t)] + arrival)

    # 2. Sending flow constraints for y (only for roads and outgoing connectors)
    for link in links:
        u = link['from']
        v = link['to']
        fft = link['free_flow_travel_time_steps']
        is_road = link['type'] == 'road'
        cap = link.get('capacity_vph', 100000) * time_params['time_step_duration_seconds'] / 3600.0
        for dest in centroids:
            for t in range(T):
                if t < fft - 1:
                    for succ in successors.get(v, []):
                        w = succ[0]
                        if (u, v, w, dest, t) in y:
                            model.addConstr(y[(u, v, w, dest, t)] == 0)
                    continue
                send = N_U[(u, v, dest, t - fft + 1)] - N_D[(u, v, dest, t)]
                total_out = gp.LinExpr()
                for succ_tup in successors.get(v, []):
                    w, _ = succ_tup
                    key = (u, v, w, dest, t)
                    if key in y:
                        total_out.add(y[key])
                model.addConstr(total_out <= send)

    # 3. Capacity and jam constraints on road links
    for link in links:
        if link['type'] != 'road':
            continue
        u = link['from']
        v = link['to']
        cap = link['capacity_vph'] * time_params['time_step_duration_seconds'] / 3600.0
        jam = link['jam_density_vehicles']
        ctt = link['congested_travel_time_steps']
        for t in range(T):
            # Exit capacity
            total_exit = gp.LinExpr()
            for succ_tup in successors.get(v, []):
                w, _ = succ_tup
                for dest in centroids:
                    key = (u, v, w, dest, t)
                    if key in y:
                        total_exit.add(y[key])
            model.addConstr(total_exit <= cap)

            # Entry capacity
            total_entry = gp.LinExpr()
            for pred_tup in predecessors.get(u, []):
                pred, _ = pred_tup
                for dest in centroids:
                    for w in [v]:
                        key = (pred, u, v, dest, t)
                        if key in y:
                            total_entry.add(y[key])
            model.addConstr(total_entry <= cap)

            # Congested wave constraint
            wave_bound = gp.LinExpr()
            for dest in centroids:
                if t - ctt + 1 >= 0:
                    wave_bound.add(N_U[(u, v, dest, t - ctt + 1)] - N_D[(u, v, dest, t)])
                else:
                    wave_bound.add(-N_D[(u, v, dest, t)])
            wave_bound.add(jam)
            model.addConstr(total_entry <= wave_bound)

    # 4. Centroid connector destination matching
    for cent in centroids:
        for out_link_tup in successors.get(cent, []):
            junc, _ = out_link_tup
            for dest in centroids:
                for t in range(T):
                    if dest != cent:
                        if (cent, junc, dest, t) in y_cent:
                            model.addConstr(y_cent[(cent, junc, dest, t)] == 0)

    # 5. Parked vehicle balance
    for cent in centroids:
        init_p = initial_parked.get(cent, 0.0)
        model.addConstr(p[(cent, 0)] == init_p)
        for t in range(T):
            arrivals = gp.LinExpr()
            for pred_tup in predecessors.get(cent, []):
                pred, link_id = pred_tup
                if link_dict[(pred, cent)]['type'] == 'centroid_connector':
                    arrivals.add(N_U[(pred, cent, cent, t)] - N_D[(pred, cent, cent, t)])
            departures = gp.LinExpr()
            for succ_tup in successors.get(cent, []):
                j, _ = succ_tup
                for d in centroids:
                    key = (cent, j, d, t)
                    if key in y_cent:
                        departures.add(y_cent[key])
            model.addConstr(p[(cent, t + 1)] == p[(cent, t)] + arrivals - departures)
            model.addConstr(departures <= p[(cent, t)])

    # 6. Final parked vehicles conservation
    final_parked_sum = gp.LinExpr()
    for cent in centroids:
        final_parked_sum.add(p[(cent, T)])
    model.addConstr(final_parked_sum == total_vehicles)

    # 7. Traveler waiting dynamics and pickup
    for o, d in od_pairs:
        model.addConstr(omega[(o, d, 0)] == 0)
        for t in range(T):
            dmd = demand.get((o, d, t), 0.0)
            pickup = e[(o, d, t)]
            model.addConstr(omega[(o, d, t + 1)] == omega[(o, d, t)] + dmd - pickup)
            model.addConstr(pickup <= omega[(o, d, t)])
            # link to vehicles
            veh_out = gp.LinExpr()
            for junc_tup in successors.get(o, []):
                j, _ = junc_tup
                key = (o, j, d, t)
                if key in y_cent:
                    veh_out.add(y_cent[key])
            model.addConstr(pickup <= veh_out)
    # Final waiting zero
    for o, d in od_pairs:
        model.addConstr(omega[(o, d, T)] == 0)

    # Callback for logging
    incumbent_obj = None
    def callback(model, where):
        nonlocal incumbent_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if incumbent_obj is None or obj < incumbent_obj - 1e-4:
                incumbent_obj = obj
                if logger:
                    logger.log(float(obj))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        # Fallback: return a feasible but poor solution if needed
        sol = {
            "objective_value": 1e10,
            "y": {},
            "y_centroid": {},
            "N_U": {},
            "N_D": {},
            "p": {},
            "e": {},
            "omega": {}
        }
    else:
        # Build output dicts
        y_out = {}
        for k, var in y.items():
            if var != 0.0:
                val = var.X if hasattr(var, 'X') else 0.0
                if val > 1e-6:
                    key_str = "|".join(map(str, k))
                    y_out[key_str] = float(val)

        y_cent_out = {}
        for k, var in y_cent.items():
            val = var.X if hasattr(var, 'X') else 0.0
            if val > 1e-6:
                key_str = "|".join(map(str, k))
                y_cent_out[key_str] = float(val)

        NU_out = {}
        for k, var in N_U.items():
            val = var.X if hasattr(var, 'X') else 0.0
            if val > 1e-6:
                key_str = "|".join(map(str, k))
                NU_out[key_str] = float(val)

        ND_out = {}
        for k, var in N_D.items():
            val = var.X if hasattr(var, 'X') else 0.0
            if val > 1e-6 or k[3] == 0:  # keep initial zeros if needed but minimal
                key_str = "|".join(map(str, k))
                ND_out[key_str] = float(val)

        p_out = {}
        for k, var in p.items():
            val = var.X if hasattr(var, 'X') else 0.0
            key_str = "|".join(map(str, k))
            p_out[key_str] = float(val)

        e_out = {}
        for k, var in e.items():
            val = var.X if hasattr(var, 'X') else 0.0
            if val > 1e-6:
                key_str = "|".join(map(str, k))
                e_out[key_str] = float(val)

        omega_out = {}
        for k, var in omega.items():
            val = var.X if hasattr(var, 'X') else 0.0
            if val > 1e-6 or k[2] == 0:
                key_str = "|".join(map(str, k))
                omega_out[key_str] = float(val)

        obj_val = model.objVal if model.objVal is not None else 0.0

        sol = {
            "objective_value": float(obj_val),
            "y": y_out,
            "y_centroid": y_cent_out,
            "N_U": NU_out,
            "N_D": ND_out,
            "p": p_out,
            "e": e_out,
            "omega": omega_out
        }

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

    if logger:
        logger.log(sol["objective_value"])

if __name__ == "__main__":
    main()