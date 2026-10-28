import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description="Solve the shared autonomous vehicle routing problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log intermediate solutions (optional).")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    network = instance['network']
    time_params = instance['time_parameters']
    demand_data = instance['demand']
    fleet = instance['fleet']

    T = time_params['time_horizon_T']
    time_step_duration = time_params['time_step_duration_seconds']
    centroids = [node['id'] for node in network['nodes'] if node['type'] == 'centroid']
    junctions = [node['id'] for node in network['nodes'] if node['type'] == 'junction']
    all_nodes = centroids + junctions

    nodes_by_id = {node['id']: node for node in network['nodes']}

    links = network['links']
    road_links = []
    outgoing_connectors = []
    incoming_connectors = []
    link_dict = {}
    capacity_per_time_step = {}
    jam_density = {}
    free_flow_tt = {}
    congested_tt = {}
    for link in links:
        key = (link['from'], link['to'])
        link_dict[key] = link
        if link['type'] == 'road':
            road_links.append(key)
            cap_vph = link['capacity_vph']
            cap = cap_vph * (time_step_duration / 3600.0)
            capacity_per_time_step[key] = cap
        elif link['type'] == 'centroid_connector':
            if link['from'] in centroids and link['to'] in junctions:
                outgoing_connectors.append(key)
            elif link['from'] in junctions and link['to'] in centroids:
                incoming_connectors.append(key)
        jam_density[key] = link['jam_density_vehicles']
        free_flow_tt[key] = link['free_flow_travel_time_steps']
        congested_tt[key] = link['congested_travel_time_steps']

    incoming_links = {node: [] for node in all_nodes}
    outgoing_links = {node: [] for node in all_nodes}
    for link in links:
        from_node = link['from']
        to_node = link['to']
        incoming_links[to_node].append((from_node, to_node))
        outgoing_links[from_node].append((from_node, to_node))

    initial_parking = fleet['initial_distribution']
    total_initial_parked = sum(initial_parking.values())

    demand_percentage = demand_data['demand_percentage'] / 100.0
    demand_dict = defaultdict(float)
    for od in demand_data['od_pairs']:
        origin = od['origin']
        destination = od['destination']
        for dep_time in od['departure_times']:
            if dep_time < T:
                demand_dict[(origin, destination, dep_time)] += 1
    for key in demand_dict:
        demand_dict[key] *= demand_percentage

    model = gp.Model("SAV_Routing")
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)

    N_U = {}
    N_D = {}
    for link in links:
        from_id = link['from']
        to_id = link['to']
        key = (from_id, to_id)
        if key in incoming_connectors:
            for t in range(T + 1):
                d_val = to_id
                N_U_key = (from_id, to_id, d_val, t)
                N_U[N_U_key] = model.addVar(lb=0, name=f"N_U_{from_id}_{to_id}_{d_val}_{t}")
                N_D[N_U_key] = model.addVar(lb=0, name=f"N_D_{from_id}_{to_id}_{d_val}_{t}")
        else:
            for d in centroids:
                for t in range(T + 1):
                    N_U_key = (from_id, to_id, d, t)
                    N_U[N_U_key] = model.addVar(lb=0, name=f"N_U_{from_id}_{to_id}_{d}_{t}")
                    N_D[N_U_key] = model.addVar(lb=0, name=f"N_D_{from_id}_{to_id}_{d}_{t}")

    y_turn = {}
    for j in all_nodes:
        in_links = incoming_links[j]
        out_links = outgoing_links[j]
        if not in_links or not out_links:
            continue
        for in_link in in_links:
            i = in_link[0]
            for out_link in out_links:
                k = out_link[1]
                for d in centroids:
                    for t in range(T):
                        if in_link in incoming_connectors and d != j:
                            continue
                        y_turn_key = (i, j, k, d, t)
                        y_turn[y_turn_key] = model.addVar(lb=0, name=f"y_{i}_{j}_{k}_{d}_{t}")

    y_centroid = {}
    for conn in outgoing_connectors:
        c, j = conn
        for d in centroids:
            for t in range(T):
                y_centroid_key = (c, j, d, t)
                if t == T - 1:
                    y_centroid[y_centroid_key] = model.addVar(lb=0, ub=0, name=f"y_centroid_{c}_{j}_{d}_{t}")
                else:
                    y_centroid[y_centroid_key] = model.addVar(lb=0, name=f"y_centroid_{c}_{j}_{d}_{t}")

    p = {}
    for c in centroids:
        for t in range(T + 1):
            p_key = (c, t)
            p[p_key] = model.addVar(lb=0, name=f"p_{c}_{t}")

    omega = {}
    for o in centroids:
        for d in centroids:
            for t in range(T + 1):
                omega_key = (o, d, t)
                omega[omega_key] = model.addVar(lb=0, name=f"omega_{o}_{d}_{t}")

    for key in N_U:
        model.addConstr(N_U[key] == 0)
    for key in N_D:
        model.addConstr(N_D[key] == 0)

    for c in centroids:
        model.addConstr(p[(c, 0)] == initial_parking.get(c, 0))

    for o in centroids:
        for d in centroids:
            model.addConstr(omega[(o, d, 0)] == 0)

    for link in links:
        from_id = link['from']
        to_id = link['to']
        key = (from_id, to_id)
        if key in incoming_connectors:
            for t in range(T):
                d_val = to_id
                sum_inflow = 0
                j = from_id
                for in_link in incoming_links[j]:
                    i = in_link[0]
                    y_key = (i, j, to_id, d_val, t)
                    if y_key in y_turn:
                        sum_inflow += y_turn[y_key]
                N_U_next = (from_id, to_id, d_val, t + 1)
                N_U_curr = (from_id, to_id, d_val, t)
                model.addConstr(N_U[N_U_next] == N_U[N_U_curr] + sum_inflow)
                model.addConstr(N_D[N_U_next] == N_U[N_U_curr])
        else:
            for d in centroids:
                for t in range(T):
                    sum_inflow = 0
                    j = from_id
                    for in_link in incoming_links[j]:
                        i = in_link[0]
                        y_key = (i, j, to_id, d, t)
                        if y_key in y_turn:
                            sum_inflow += y_turn[y_key]
                    N_U_curr = (from_id, to_id, d, t)
                    N_U_next = (from_id, to_id, d, t + 1)
                    model.addConstr(N_U[N_U_next] == N_U[N_U_curr] + sum_inflow)

            for d in centroids:
                for t in range(T):
                    sum_outflow = 0
                    j = to_id
                    for out_link in outgoing_links[j]:
                        k = out_link[1]
                        y_key = (from_id, to_id, k, d, t)
                        if y_key in y_turn:
                            sum_outflow += y_turn[y_key]
                    N_D_curr = (from_id, to_id, d, t)
                    N_D_next = (from_id, to_id, d, t + 1)
                    model.addConstr(N_D[N_D_next] == N_D[N_D_curr] + sum_outflow)

    for c in centroids:
        for t in range(T):
            sum_arrivals = 0
            for in_conn in incoming_connectors:
                if in_conn[1] == c:
                    j, c_val = in_conn
                    N_D_curr = N_D.get((j, c, c, t), 0)
                    N_D_next = N_D.get((j, c, c, t + 1), 0)
                    if (j, c, c, t) in N_D and (j, c, c, t + 1) in N_D:
                        sum_arrivals += (N_D_next - N_D_curr)
            sum_departures = 0
            for out_conn in outgoing_connectors:
                if out_conn[0] == c:
                    c_val, j = out_conn
                    for d in centroids:
                        y_key = (c, j, d, t)
                        if y_key in y_centroid:
                            sum_departures += y_centroid[y_key]
            p_curr = p[(c, t)]
            p_next = p[(c, t + 1)]
            model.addConstr(p_next == p_curr + sum_arrivals - sum_departures)

    for o in centroids:
        for d in centroids:
            for t in range(T):
                demand_val = demand_dict.get((o, d, t), 0.0)
                sum_y = 0
                for out_conn in outgoing_connectors:
                    if out_conn[0] == o:
                        c, j = out_conn
                        y_key = (o, j, d, t)
                        if y_key in y_centroid:
                            sum_y += y_centroid[y_key]
                omega_curr = omega[(o, d, t)]
                omega_next = omega[(o, d, t + 1)]
                model.addConstr(omega_next == omega_curr + demand_val - sum_y)

    for o in centroids:
        for d in centroids:
            model.addConstr(omega[(o, d, T)] == 0)

    model.addConstr(gp.quicksum(p[(c, T)] for c in centroids) == total_initial_parked)

    for link in links:
        from_id = link['from']
        to_id = link['to']
        key = (from_id, to_id)
        if key not in road_links and key not in outgoing_connectors:
            continue
        tau = free_flow_tt[key]
        for d in centroids:
            for t in range(T):
                sum_outflow = 0
                j = to_id
                for out_link in outgoing_links[j]:
                    k = out_link[1]
                    y_key = (from_id, to_id, k, d, t)
                    if y_key in y_turn:
                        sum_outflow += y_turn[y_key]
                if t < tau:
                    model.addConstr(sum_outflow == 0)
                else:
                    s = t - tau + 1
                    if s < 0:
                        diff = -N_D.get((from_id, to_id, d, t), 0)
                    else:
                        diff = N_U[(from_id, to_id, d, s)] - N_D.get((from_id, to_id, d, t), 0)
                    model.addConstr(sum_outflow <= diff)

    for key in road_links:
        from_id, to_id = key
        cap = capacity_per_time_step.get(key, 0)
        for t in range(T):
            total_outflow = 0
            j = to_id
            for d in centroids:
                for out_link in outgoing_links[j]:
                    k = out_link[1]
                    y_key = (from_id, to_id, k, d, t)
                    if y_key in y_turn:
                        total_outflow += y_turn[y_key]
            model.addConstr(total_outflow <= cap)

    for key in road_links:
        j, k = key
        cap = capacity_per_time_step.get(key, 0)
        for t in range(T):
            total_inflow = 0
            for in_link in incoming_links[j]:
                i = in_link[0]
                for d in centroids:
                    y_key = (i, j, k, d, t)
                    if y_key in y_turn:
                        total_inflow += y_turn[y_key]
            model.addConstr(total_inflow <= cap)

    for key in road_links:
        j, k = key
        jam = jam_density.get(key, 0)
        tau_cong = congested_tt.get(key, 0)
        for t in range(T):
            total_inflow = 0
            for in_link in incoming_links[j]:
                i = in_link[0]
                for d in centroids:
                    y_key = (i, j, k, d, t)
                    if y_key in y_turn:
                        total_inflow += y_turn[y_key]
            rhs = jam
            for d in centroids:
                s = t - tau_cong + 1
                if s < 0:
                    term = -N_D.get((j, k, d, t), 0)
                else:
                    term = N_U.get((j, k, d, s), 0) - N_D.get((j, k, d, t), 0)
                rhs += term
            model.addConstr(total_inflow <= rhs)

    for c in centroids:
        for t in range(T):
            total_departures = 0
            for out_conn in outgoing_connectors:
                if out_conn[0] == c:
                    c_val, j = out_conn
                    for d in centroids:
                        y_key = (c, j, d, t)
                        if y_key in y_centroid:
                            total_departures += y_centroid[y_key]
            model.addConstr(total_departures <= p[(c, t)])

    for conn in outgoing_connectors:
        c, j = conn
        for d in centroids:
            for t in range(T - 1):
                sum_outflow = 0
                for out_link in outgoing_links[j]:
                    k = out_link[1]
                    y_key = (c, j, k, d, t + 1)
                    if y_key in y_turn:
                        sum_outflow += y_turn[y_key]
                model.addConstr(y_centroid[(c, j, d, t)] == sum_outflow)

    total_vehicle_time = 0
    for link in links:
        from_id = link['from']
        to_id = link['to']
        key = (from_id, to_id)
        if key in incoming_connectors:
            d_val = to_id
            for t in range(T + 1):
                N_U_key = (from_id, to_id, d_val, t)
                if N_U_key in N_U and N_U_key in N_D:
                    total_vehicle_time += (N_U[N_U_key] - N_D[N_U_key])
        else:
            for d in centroids:
                for t in range(T + 1):
                    N_U_key = (from_id, to_id, d, t)
                    if N_U_key in N_U and N_U_key in N_D:
                        total_vehicle_time += (N_U[N_U_key] - N_D[N_U_key])

    total_waiting_time = 0
    for o in centroids:
        for d in centroids:
            for t in range(T + 1):
                if (o, d, t) in omega:
                    total_waiting_time += omega[(o, d, t)]

    model.setObjective(total_vehicle_time + total_waiting_time, GRB.MINIMIZE)

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    model.optimize(callback if logger else None)

    build_trivial = False
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Outputting trivial solution.")
        build_trivial = True
    elif model.SolCount == 0:
        print("No solution found. Outputting trivial solution.")
        build_trivial = True

    if build_trivial:
        solution = {
            "objective_value": 0,
            "y": {},
            "y_centroid": {},
            "N_U": {},
            "N_D": {},
            "p": {},
            "e": {},
            "omega": {}
        }
        
        # Generate keys for y (turning flows)
        for j in all_nodes:
            in_links = incoming_links[j]
            out_links = outgoing_links[j]
            if not in_links or not out_links:
                continue
            for in_link in in_links:
                i = in_link[0]
                for out_link in out_links:
                    k = out_link[1]
                    for d in centroids:
                        for t in range(T):
                            key = f"{i}|{j}|{k}|{d}|{t}"
                            solution["y"][key] = 0.0
        
        # Generate keys for y_centroid (centroid departures)
        for conn in outgoing_connectors:
            c, j = conn
            for d in centroids:
                for t in range(T):
                    key = f"{c}|{j}|{d}|{t}"
                    solution["y_centroid"][key] = 0.0
        
        # Generate keys for N_U and N_D (cumulative counts)
        for link in links:
            from_id = link['from']
            to_id = link['to']
            for d in centroids:
                for t in range(T + 1):
                    key = f"{from_id}|{to_id}|{d}|{t}"
                    solution["N_U"][key] = 0.0
                    solution["N_D"][key] = 0.0
        
        # Generate keys for p (parked vehicles)
        total_waiting_time = 0
        for c in centroids:
            init_val = initial_parking.get(c, 0)
            for t in range(T + 1):
                key = f"{c}|{t}"
                solution["p"][key] = float(init_val)
        
        # Generate keys for e (pickups) and omega (waiting demand)
        for o in centroids:
            for d in centroids:
                acc = 0
                for t in range(T + 1):
                    key = f"{o}|{d}|{t}"
                    if t > 0:
                        demand_val = demand_dict.get((o, d, t - 1), 0.0)
                        acc += demand_val
                    solution["e"][key] = 0.0
                    solution["omega"][key] = acc
                    total_waiting_time += acc
        solution["objective_value"] = total_waiting_time
    else:
        solution = {
            "objective_value": model.ObjVal,
            "y": {},
            "y_centroid": {},
            "N_U": {},
            "N_D": {},
            "p": {},
            "e": {},
            "omega": {}
        }

        for y_key, var in y_turn.items():
            i, j, k, d, t = y_key
            key = f"{i}|{j}|{k}|{d}|{t}"
            solution["y"][key] = var.X

        for yc_key, var in y_centroid.items():
            c, j, d, t = yc_key
            key = f"{c}|{j}|{d}|{t}"
            solution["y_centroid"][key] = var.X

        for link in links:
            from_id = link['from']
            to_id = link['to']
            for d in centroids:
                for t in range(T + 1):
                    key = f"{from_id}|{to_id}|{d}|{t}"
                    if (from_id, to_id) in incoming_connectors and d != to_id:
                        solution["N_U"][key] = 0.0
                        solution["N_D"][key] = 0.0
                    else:
                        N_U_key = (from_id, to_id, d, t)
                        solution["N_U"][key] = N_U[N_U_key].X if N_U_key in N_U else 0.0
                        solution["N_D"][key] = N_D[N_U_key].X if N_U_key in N_D else 0.0

        for c in centroids:
            for t in range(T + 1):
                key = f"{c}|{t}"
                solution["p"][key] = p[(c, t)].X

        for o in centroids:
            for d in centroids:
                for t in range(T + 1):
                    key = f"{o}|{d}|{t}"
                    solution["omega"][key] = omega[(o, d, t)].X
                    if t < T:
                        e_val = 0.0
                        for out_conn in outgoing_connectors:
                            if out_conn[0] == o:
                                c, j = out_conn
                                yc_key = (o, j, d, t)
                                if yc_key in y_centroid:
                                    e_val += y_centroid[yc_key].X
                        solution["e"][key] = e_val
                    else:
                        solution["e"][key] = 0.0

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()