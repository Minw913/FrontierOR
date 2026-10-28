import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
from collections import defaultdict


def solve_sav_dynamic_assignment(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    # Preprocessing the data to make indices easier to work with
    node_to_id = {node['id']: i for i, node in enumerate(instance['network']['nodes'])}
    node_id_to_type = {i: node['type'] for i, node in enumerate(instance['network']['nodes'])}

    # Create mappings from link id to link properties
    link_to_id = {}
    id_to_link = {}
    for idx, link in enumerate(instance['network']['links']):
        link_to_id[f"{link['from']}|{link['to']}"] = idx
        id_to_link[idx] = link
    
    n_nodes = len(instance['network']['nodes'])
    n_links = len(instance['network']['links'])
    
    # Extract relevant parameters
    T = instance['time_parameters']['time_horizon_T']
    
    # Get centroids and junctions
    centroids = [i for i in range(n_nodes) if node_id_to_type[i] == 'centroid']
    junctions = [i for i in range(n_nodes) if node_id_to_type[i] == 'junction']
    
    # Build adjacency relationships from the JSON data
    out_links = {i: [] for i in range(n_nodes)}
    in_links = {i: [] for i in range(n_nodes)}
    
    # Also classify links into different groups for convenience
    road_links = []
    centroid_connectors = []
    
    for idx, link in enumerate(instance['network']['links']):
        from_node_id = link['from']
        to_node_id = link['to']
        fr = node_to_id[from_node_id]
        to = node_to_id[to_node_id]
        
        out_links[fr].append(idx)
        in_links[to].append(idx)
        
        if link['type'] == 'road':
            road_links.append(idx)
        elif link['type'] == 'centroid_connector':
            centroid_connectors.append(idx)
    
    # Map link properties to easy-access data structures
    link_data = {}
    for link in instance['network']['links']:
        lid = f"{link['from']}|{link['to']}"
        link_data[lid] = {
            'length': link['length_ft'],
            'ff_time': link['free_flow_travel_time_steps'],
            'capacity': link['capacity_vph'],
            'cong_wave_time': link['congested_travel_time_steps'],
            'jam_density': link['jam_density_vehicles'],
            'type': link['type']
        }

    # Demand processing
    od_pairs = {}
    od_pairs_list = []
    od_departure_times = {}
    for od_pair in instance['demand']['od_pairs']:
        origin_id = node_to_id[od_pair['origin']]
        dest_id = node_to_id[od_pair['destination']]
        key_tuple = (origin_id, dest_id)
        od_pairs[key_tuple] = od_pair['num_travelers']
        od_pairs_list.append(key_tuple)
        od_departure_times[key_tuple] = od_pair['departure_times']

    # Fleet distribution
    initial_parked = [0.0] * n_nodes
    fleet_total = 0
    for centroid_str, count in instance['fleet']['initial_distribution'].items():
        centroid_id = node_to_id[centroid_str]
        initial_parked[centroid_id] = float(count)
        fleet_total += count
    
    # Build model
    m = gp.Model("sav_dynamic_assignment")
    m.setParam('TimeLimit', time_limit)
    m.setParam('LogToConsole', 0)  # Disable Gurobi console output to reduce noise

    # Decision variables
    
    # Link flows y (turning movements)
    # y[(from, via, to, dst, t)] from node "from", go via connector/link "via", 
    # onto next node "to", for destination "dst", at time t
    y = {}      # Internal storage for (link_id, dest, t)
    y_ext = {}  # External schema for output -> (from, via, to, dest, t)
    
    # Variables for turning movements at nodes that are not centroids
    for j in junctions:
        for out_lid in out_links[j]:
            out_link = id_to_link[out_lid]
            out_to_id = node_to_id[out_link['to']]
            for dst in centroids:
                for t in range(T):
                    var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"y_{j}_{out_lid}_{dst}_{t}")
                    y[(out_lid, dst, t)] = var
                    y_ext[f"{instance['network']['nodes'][j]['id']}|{out_link['from']}|{out_link['to']}|{instance['network']['nodes'][dst]['id']}|{t}"] = var

    # Variables for turning from centroids to outgoing connectors (departures)
    y_centroid = {}
    y_centroid_ext = {}
    for c_idx in centroids:
        for out_lid in out_links[c_idx]:
            out_link = id_to_link[out_lid]
            out_to_id = node_to_id[out_link['to']]
            for dst in centroids:
                for t in range(T):
                    var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"y_centroid_{c_idx}_{out_lid}_{dst}_{t}")
                    y[(out_lid, dst, t)] = var
                    y_centroid[(c_idx, out_to_id, dst, t)] = var
                    y_centroid_ext[f"{instance['network']['nodes'][c_idx]['id']}|{instance['network']['nodes'][out_to_id]['id']}|{instance['network']['nodes'][dst]['id']}|{t}"] = var

    # Cumulative upstream counts N_U
    # Indexed by (link_idx, dst, t)
    N_U = {}
    for lid in range(n_links):
        link_info = id_to_link[lid]
        fr = node_to_id[link_info['from']]
        to = node_to_id[link_info['to']]
        for dst in centroids:
            for t in range(T):
                var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"N_U_{fr}_{to}_{dst}_{t}")
                N_U[(lid, dst, t)] = var

    # Cumulative downstream counts N_D
    # Indexed by (link_idx, dst, t)
    N_D = {}
    for lid in range(n_links):
        link_info = id_to_link[lid]
        fr = node_to_id[link_info['from']]
        to = node_to_id[link_info['to']]
        for dst in centroids:
            for t in range(T):
                var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"N_D_{fr}_{to}_{dst}_{t}")
                N_D[(lid, dst, t)] = var

    # Parked vehicles p
    # p[cent_idx][t] - number parked at centroid cent_idx at time t
    p = {}
    for cent_idx in centroids:
        for t in range(T + 1):  # Need T+1 to handle transitions
            var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"p_{cent_idx}_{t}")
            p[(cent_idx, t)] = var

    # Pickups e
    # e[orig][dest][t]
    e = {}
    e_ext = {}
    for (orig, dest), count in od_pairs.items():
        for t in range(T):
            var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"e_{orig}_{dest}_{t}")
            e[(orig, dest, t)] = var
            e_ext[f"{instance['network']['nodes'][orig]['id']}|{instance['network']['nodes'][dest]['id']}|{t}"] = var

    # Waiting travelers omega
    # omega[orig][dest][t]
    omega = {}
    omega_ext = {}
    for (orig, dest), count in od_pairs.items():
        for t in range(T + 1):  # Need T+1 to capture ending condition
            var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"omega_{orig}_{dest}_{t}")
            omega[(orig, dest, t)] = var
            omega_ext[f"{instance['network']['nodes'][orig]['id']}|{instance['network']['nodes'][dest]['id']}|{t}"] = var

    m.update()

    # Constraint: Initial waiting travelers
    for (orig, dest), count in od_pairs.items():
        init_wait_at_t = [0.0] * T
        for dep_time in od_departure_times[(orig, dest)]:
            if dep_time < T:
                init_wait_at_t[dep_time] += 1.0
        m.addConstr(omega[(orig, dest, 0)] == init_wait_at_t[0])
        for t in range(1, T + 1):
            # omega[o,d,t] = omega[o,d,t-1] + new_demand(t) - pickups(o,d,t)
            new_demand_t = init_wait_at_t[t] if t < T else 0.0
            pickup_var = e.get((orig, dest, t - 1), 0.0)
            m.addConstr(omega[(orig, dest, t)] == omega[(orig, dest, t - 1)] + new_demand_t - pickup_var)

    # Constraint: End of horizon – all demand served
    for (orig, dest), count in od_pairs.items():
        m.addConstr(omega[(orig, dest, T)] == 0.0)

    # Constraint: Cumulative N_U definitions for non-centroid connectors
    for lid in range(n_links):
        link = id_to_link[lid]
        ltype = link['type']
        ff_time = link['free_flow_travel_time_steps']
        
        for dst in centroids:
            for t in range(T):
                in_flow_sum = 0.0
                for in_lid in in_links[node_to_id[link['from']]]:
                    in_flow_sum += y.get((in_lid, dst, t), 0.0)
                
                if t >= ff_time - 1:
                    prior_NU = N_U[(lid, dst, t - ff_time)]
                    diff = N_U[(lid, dst, t)] - prior_NU
                else:
                    prior_NU = 0.0
                    diff = N_U[(lid, dst, t)]
                
                m.addConstr(in_flow_sum <= diff)
                
                # For road links: N_U = in_flows_cumulative
                if ltype == 'road':
                    m.addConstr(N_U[(lid, dst, t)] == gp.quicksum(y.get((in_lid, dst, tau), 0.0) for tau in range(t + 1)) if t >= ff_time - 1 else gp.quicksum(y.get((in_lid, dst, tau), 0.0) for in_lid in in_links[node_to_id[link['from']]] for tau in range(min(t + 1, ff_time))))

                # For centroid connectors, we'll handle differently: N_U equals outflow one time step later
                if ltype == 'centroid_connector':
                   pass  # Process further below if needed


    # Correct constraint: N_U definition based on incoming flows delayed by free-flow time
    for lid in range(n_links):
        link = id_to_link[lid]
        fr_node = node_to_id[link['from']]
        to_node = node_to_id[link['to']]
        ltype = link['type']
        ff_time = link['free_flow_travel_time_steps']

        for dst in centroids:
            for t in range(T):
                if t < ff_time:
                    m.addConstr(N_U[(lid, dst, t)] == 0.0)
                else:
                    # This is the actual outflow from previous links at time t - ff_time
                    inflow_term = gp.quicksum(
                        y.get((incoming_lid, dst, t - ff_time), 0.0) 
                        for incoming_lid in in_links[fr_node]
                    )
                    m.addConstr(N_U[(lid, dst, t)] == N_U[(lid, dst, t - 1)] + inflow_term)

    # Constraint: N_D for centroid connectors
    # For incoming centroid connectors, N_D of time t equals N_U of time-1
    incoming_connectors = [lid for lid in centroid_connectors if id_to_link[lid]['to'] in [n_ids for n_ids,j in enumerate(instance['network']['nodes']) if j['type'] =='centroid']]
    outgoing_connectors = [lid for lid in centroid_connectors if id_to_link[lid]['from'] in [n_ids for n_ids,j in enumerate(instance['network']['nodes']) if j['type'] == 'centroid']]

    for lid in incoming_connectors:
        link = id_to_link[lid]
        to_node = node_to_id[link['to']]
        # For all dst != to_node, N_D should remain same as N_U at next time (no flow permitted)
        # For dst == to_node, N_D = N_U previous time
        for dst in centroids:
            for t in range(1, T+1):
                if dst == to_node:
                    if t-1 >= 0:
                        m.addConstr(N_D[(lid, dst, t-1)] == N_U[(lid, dst, t-1)])
                else:
                    if t-1 >= 0:
                        m.addConstr(N_D[(lid, dst, t-1)] == 0.0)


    # Constraint: N_D for regular ROAD links and outgoing centroid connectors
    road_and_outgo_conn = road_links
    for lid in road_and_outgo_conn:
        link = id_to_link[lid]
        fr_node = node_to_id[link['from']]
        to_node = node_to_id[link['to']]
        ltype = link['type']
        ff_time = link['free_flow_travel_time_steps']

        for dst in centroids:
            for t in range(T):
                if t < ff_time:
                    # In first few periods, no outflow possible yet
                    m.addConstr(N_D[(lid, dst, t)] == 0.0)
                else:
                    outflow = gp.quicksum(
                        y.get((lid, dst, t_orig), 0.0)
                        for t_orig in range(max(0, t - ff_time + 1), t + 1)
                    )
                    m.addConstr(N_D[(lid, dst, t)] == outflow)

    # Constraint: Sending flow bounds
    for lid in road_links + outgoing_connectors:
        link = id_to_link[lid]
        capacity = link['capacity_vph'] / 3600.0 * instance['time_parameters']['time_step_duration_seconds']
        ff_time = link['free_flow_travel_time_steps']

        for dst in centroids:
            for t in range(T):
                # Compute sending flow as NU - ND if possible, otherwise max capacity or other limit
                # Since cumulative flow is NU - ND for destination d
                sending_bound = N_U.get((lid, dst, t - ff_time + 1), 0.0) - N_D.get((lid, dst, t), 0.0)
                total_turning_flow = gp.quicksum(
                    y.get((lid, dst, tau), 0.0) for tau in range(t, min(t + ff_time, T))
                ) if ff_time > 0 else (y.get((lid, dst, t), 0.0) if t<T else 0.0)
                
                m.addConstr(total_turning_flow <= sending_bound)


    # Capacity constraints on road links (both entry and exit flow)
    for lid in road_links:
        link = id_to_link[lid]
        capacity = link['capacity_vph'] / 3600.0 * instance['time_parameters']['time_step_duration_seconds']
        ff_time = link['free_flow_travel_time_steps']
        
        # Entry capacity: total incoming turning flows at a given timestep
        for t in range(T):
            in_flow_total = gp.quicksum(
                y.get((in_lid, dst, t), 0.0)
                for in_lid in in_links[node_to_id[link['from']]]
                for dst in centroids
            )
            entry_start_time = max(0, t - ff_time + 1)
            if t >= ff_time:
                m.addConstr(in_flow_total <= capacity)

        # Exit capacity: total outgoing flow from a given link at a given timestep
        for t in range(T):
            out_flow_total = gp.quicksum(
                y.get((lid, dst, t), 0.0)
                for dst in centroids
            )
            if t >= ff_time:  # Only after free-flow time
                m.addConstr(out_flow_total <= capacity)


    # Congested wave constraints on road links
    for lid in road_links:
        link = id_to_link[lid]
        length = link['length_ft']
        cong_wave_time = link['congested_travel_time_steps']
        jam_den = link['jam_density']

        for t in range(T):
            # Sum over all destinations
            total_incoming = gp.quicksum(
                y.get((in_lid, dst, t), 0.0)
                for in_lid in in_links[node_to_id[link['from']]]
                for dst in centroids
            )
            
            upper_bound_for_cong_wave = gp.quicksum(
                N_U.get((in_link_id, dst, max(0, t - cong_wave_time + 1)), 0.0) - 
                N_D.get((in_link_id, dst, t), 0.0)
                for in_link_id in in_links[node_to_id[link['from']]]
                for dst in centroids
            ) + jam_den
            m.addConstr(total_incoming <= upper_bound_for_cong_wave)


    # Vehicle parking dynamics
    for cent in centroids:
        for t in range(1, T + 1):
            incoming_from_road = gp.quicksum(
                N_U.get((in_link_id, cent, t - 1), 0.0) - N_D.get((in_link_id, cent, t - 1), 0.0)
                for in_link_id in in_links[cent]  # Links terminating at this centroid
                if id_to_link[in_link_id]['type'] == 'centroid_connector'
            )

            departing_via_connectors = gp.quicksum(
                y.get((out_link_id, dst, t - 1), 0.0)
                for out_link_id in out_links[cent]
                for dst in centroids
            )

            m.addConstr(p[(cent, t)] == p[(cent, t - 1)] + incoming_from_road - departing_via_connectors)

    # Initial park constraint
    for centr in centroids:
        m.addConstr(p[(centr, 0)] == initial_parked[centr])


    # Final park conservation
    for t in range(T):
        total_parked_change = gp.quicksum(p[(cent, t+1)] - p[(cent, t)] for cent in centroids if (cent, t+1) in p and (cent, t) in p)
    total_final_parked = gp.quicksum(p[(cent, T)] for cent in centroids)
    total_init_parked = sum(initial_parked)
    m.addConstr(total_final_parked == total_init_parked)


    # Pickup limitations
    for (orig, dest) in od_pairs_list:
        for t in range(T):
            # Cannot pick up more than what's waiting
            m.addConstr(e[(orig, dest, t)] <= omega[(orig, dest, t)])
            
            # Cannot pick up more than available departing vehicles in that OD direction
            total_dep_to_dest = gp.quicksum(
                y.get((out_lid, dest, t), 0.0)
                for out_lid in out_links[orig]
            )
            m.addConstr(e[(orig, dest, t)] <= total_dep_to_dest)


    # Setup objective
    # Total system travel time:
    # 1. Vehicle-time on links: sum over all links, destinations, times (N_U - N_D) per link/dest/time
    # 2. Traveler waiting time: sum over all O-D pairs, times of omega[o][d][t]
    
    link_vehicle_time_sum = gp.quicksum(
        (N_U[(lid, dst, t)] - N_D[(lid, dst, t)])
        for lid in range(n_links)
        for dst in centroids
        for t in range(T)
    )

    waiting_time_sum = gp.quicksum(
        omega[(o, d, t)] 
        for o, d in od_pairs_list 
        for t in range(T)
    )

    total_cost = link_vehicle_time_sum + waiting_time_sum

    m.setObjective(total_cost, GRB.MINIMIZE)

    # Set a callback to log solutions if logger exists
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    def log_solution(model, where):
        if where == GRB.CB_MIPSOL and logger:
            logger.log(model.cbGet(GRB.CB_MIPSOL_OBJBST))

    # Optimize
    if logger:
        m.optimize(log_solution)
    else:
        m.optimize()

    # Prepare solution results
    solution_dict = {'objective_value': m.ObjVal, 'y': {}, 'y_centroid': {}, 'N_U': {}, 'N_D': {}, 'p': {}, 'e': {}, 'omega': {}}

    # Fill in the result dictionaries
    for key, var in y_ext.items():
        val = var.X
        if abs(val) > 1e-8:  # Only store if significant
            solution_dict['y'][key] = val

    for key, var in y_centroid_ext.items():
        val = var.X
        if abs(val) > 1e-8:
            solution_dict['y_centroid'][key] = val

    for lid in range(n_links):
        link_info = id_to_link[lid]
        fr_node = node_to_id[link_info['from']]
        to_node = node_to_id[link_info['to']]
        for dst in centroids:
            for t in range(T):
                val = N_U[(lid, dst, t)].X
                key = f"{instance['network']['nodes'][fr_node]['id']}|{instance['network']['nodes'][to_node]['id']}|{instance['network']['nodes'][dst]['id']}|{t}"
                if abs(val) > 1e-8:
                    solution_dict['N_U'][key] = val
                
                val = N_D[(lid, dst, t)].X
                if abs(val) > 1e-8:
                    solution_dict['N_D'][key] = val

    for cent_id in centroids:
        for t in range(T + 1):
            val = p[(cent_id, t)].X
            key = f"{instance['network']['nodes'][cent_id]['id']}|{t}"
            if abs(val) > 1e-8:
                solution_dict['p'][key] = val

    for key, var in e_ext.items():
        val = var.X
        if abs(val) > 1e-8:
            solution_dict['e'][key] = val

    for key, var in omega_ext.items():
        val = var.X
        if abs(val) > 1e-8:
            solution_dict['omega'][key] = val

    # Write solution to file
    with open(solution_path, 'w') as f:
        json.dump(solution_dict, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)

    args = parser.parse_args()

    solve_sav_dynamic_assignment(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == '__main__':
    main()