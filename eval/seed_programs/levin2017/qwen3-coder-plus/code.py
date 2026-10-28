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
    id_to_node = {i: node['id'] for i, node in enumerate(instance['network']['nodes'])}

    # Create mappings from link id to link properties
    link_to_id = {}
    id_to_link = {}
    for idx, link in enumerate(instance['network']['links']):
        link_key = f"{link['from']}|{link['to']}"
        link_to_id[link_key] = idx
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
    for centroid_str, count in instance['fleet']['initial_distribution'].items():
        if centroid_str in node_to_id:
            centroid_id = node_to_id[centroid_str]
            initial_parked[centroid_id] = float(count)
    
    # Build model
    m = gp.Model("sav_dynamic_assignment")
    m.setParam("Threads", 1)
    m.setParam('TimeLimit', time_limit)
    m.setParam('LogToConsole', 0)  # Reduce verbosity

    # Decision variables
    
    # y: Turning movement flows (for junctions)
    y = {}
    for j in junctions:
        for out_lid in out_links[j]:
            out_link = id_to_link[out_lid]
            out_to_id = node_to_id[out_link['to']]
            for dst in centroids:
                for t in range(T):
                    var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"y_{j}_{out_lid}_{dst}_{t}")
                    y[(out_lid, dst, t)] = var

    # y_centroid: Departures from centroids via outgoing connectors
    y_centroid = {}
    for c_idx in centroids:
        for out_lid in out_links[c_idx]:
            out_link = id_to_link[out_lid]
            out_to_node_id = node_to_id[out_link['to']]  # junction/node to which it connects
            for dst in centroids:
                for t in range(T):
                    var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"y_centroid_{c_idx}_{out_lid}_{dst}_{t}")
                    y[(out_lid, dst, t)] = var
                    y_centroid[(c_idx, out_to_node_id, dst, t)] = var

    # Cumulative upstream counts N_U
    N_U = {}
    for lid in range(n_links):
        link_info = id_to_link[lid]
        from_node = node_to_id[link_info['from']]
        to_node = node_to_id[link_info['to']]
        for dst in centroids:
            for t in range(T + 5):  # Extra time steps to avoid index errors during constraints
                var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"N_U_{from_node}_{to_node}_{dst}_{t}")
                N_U[(lid, dst, t)] = var

    # Cumulative downstream counts N_D
    N_D = {}
    for lid in range(n_links):
        link_info = id_to_link[lid]
        from_node = node_to_id[link_info['from']]
        to_node = node_to_id[link_info['to']]
        for dst in centroids:
            for t in range(T + 5):  # Extra time steps to avoid index errors during constraints
                var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"N_D_{from_node}_{to_node}_{dst}_{t}")
                N_D[(lid, dst, t)] = var

    # Parked vehicles p
    p = {}
    for cent_idx in centroids:
        for t in range(T + 1):  # Need T+1 (at end of final timestep)
            var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"p_{cent_idx}_{t}")
            p[(cent_idx, t)] = var

    # Pickups e
    e = {}
    for (orig, dest), count in od_pairs.items():
        for t in range(T):
            var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"e_{orig}_{dest}_{t}")
            e[(orig, dest, t)] = var

    # Waiting travelers omega
    omega = {}
    for (orig, dest), count in od_pairs.items():
        for t in range(T + 1):  # Need T+1 to hold final zero values
            var = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"omega_{orig}_{dest}_{t}")
            omega[(orig, dest, t)] = var

    m.update()

    # Constraint: Initial waiting demand
    for (orig, dest), count in od_pairs.items():
        # Calculate how many people want to travel from orig to dest at each time step
        demand_by_time = [0.0] * T
        for dep_time in od_departure_times[(orig, dest)]:
            if 0 <= dep_time < T:
                demand_by_time[dep_time] += 1.0
        
        # Initial waiting travelers
        m.addConstr(omega[(orig, dest, 0)] == demand_by_time[0])
        
        # Evolution of waiting queue
        for t in range(1, T + 1):
            new_demand = demand_by_time[t - 1] if t - 1 < len(demand_by_time) else 0.0  # arrivals in period [t-1, t]
            m.addConstr(omega[(orig, dest, t)] == omega[(orig, dest, t - 1)] + new_demand - e[(orig, dest, t - 1)])

    # Constraint: Final waiting demand must be 0
    for (orig, dest), count in od_pairs.items():
        m.addConstr(omega[(orig, dest, T)] == 0.0)

    # Constraint: Initial parked vehicles
    for cent_idx in centroids:
        m.addConstr(p[(cent_idx, 0)] == initial_parked[cent_idx])

    # Parking evolution: p[cent, t+1] = p[cent, t] + incoming - departing
    for cent in centroids:
        for t in range(T):
            # Incoming: from incoming centroid connectors
            incoming_vehicles = gp.quicksum(
                N_U[(in_lid, cent, t)] - N_D[(in_lid, cent, t)]
                for in_lid in in_links[cent]
                if id_to_link[in_lid]['type'] == 'centroid_connector' and node_to_id[id_to_link[in_lid]['from']] == cent
            )  # Only for incoming connectors to cent
            
            # Departing: sum of all y_centroid from this centroid
            departing_vehicles = gp.quicksum(
                y_centroid.get((cent, node_to_id[id_to_link[out_lid]['to']], dst, t), 0.0)
                for out_lid in out_links[cent]
                for dst in centroids
            )
            m.addConstr(p[(cent, t + 1)] == p[(cent, t)] + incoming_vehicles - departing_vehicles)

    # Final park conservation (total parked vehicles remains same) - only if needed
    # We don't enforce this as it follows from other constraints.

    # For each link, update N_U and N_D according to dynamics
    for lid in range(n_links):
        link_info = id_to_link[lid]
        ltype = link_info['type']
        from_node = node_to_id[link_info['from']]
        to_node = node_to_id[link_info['to']]
        ff_time = link_info['free_flow_travel_time_steps']
        
        for dst in centroids:
            # Initialize early timesteps for N_U and N_D to 0 for all links
            for t in range(max(T+1, ff_time+1)):  # Initialize for at least up to free flow time
                if t == 0:
                    m.addConstr(N_U[(lid, dst, t)] == 0.0)
                    m.addConstr(N_D[(lid, dst, t)] == 0.0)
                elif t < ff_time:
                    pass  # We can add specific constraints if needed; we'll calculate normally

            for t in range(ff_time, T):
                # N_U computation: what enters a link during [t-1, t] period
                if ltype == 'road' or (ltype == 'centroid_connector' and from_node in centroids):
                    # For each type, the flow of vehicles entering during [t-1, t] is
                    # a combination of turning movements from upstream links
                    inflow_sum = 0.0
                    if ltype == 'centroid_connector' and from_node in centroids:
                        # This is a centroid-to-network link. So inflow is from departures
                        # from centroid during [t-ff_time, t-ff_time+1] -> affects N_U at t
                        inflow_sum = y.get((lid, dst, t - ff_time), 0.0)
                    else:
                        # For regular roads, inflow comes from incoming link(s)
                        inflow_sum = gp.quicksum(
                            y.get((in_link_lid, dst, t - ff_time), 0.0)
                            for in_link_lid in in_links[from_node]
                        )

                    m.addConstr(N_U[(lid, dst, t)] == N_U[(lid, dst, t - 1)] + inflow_sum)
                
                # N_D computation: what exits the link during [t-1, t] period
                # This happens with delay equal to free flow time
                if ltype == 'road' or (ltype == 'centroid_connector' and to_node in centroids):
                    # Vehicles that entered at t-ff_time exit at time t (with some flow limits later)
                    m.addConstr(N_D[(lid, dst, t)] == N_U[(lid, dst, t - ff_time)])


    # Flow conservation and capacity constraints
    for r_id in road_links:
        link = id_to_link[r_id]
        cap = link['capacity_vph'] / 3600.0 * instance['time_parameters']['time_step_duration_seconds']
        ff_time = link['free_flow_travel_time_steps']

        for t in range(ff_time, T):
            # Total flow that can exit the link during [t-1, t]
            total_out_flow = gp.quicksum(y.get((r_id, dst, t), 0.0) for dst in centroids)
            m.addConstr(total_out_flow <= cap)

            # Sending flow constraint: amount that can turn is bounded by how many
            # are actually available at upstream end
            for dst in centroids:
                sending_flow = N_U[(r_id, dst, t - ff_time)] - N_D[(r_id, dst, t - 1)]
                m.addConstr(y[(r_id, dst, t)] <= sending_flow + 1e-10)  # Add small epsilon to allow numeric tolerance

    # For each link that's not a road, process flow updates correctly
    # For incoming centroid connectors (ending at centroids)
    for cid in centroid_connectors:
        if id_to_link[cid]['to'] in [id_to_node[c] for c in centroids]:  # incoming connector ending at centroid c
            link = id_to_link[cid]
            to_cent = node_to_id[link['to']]
            from_node = node_to_id[link['from']]
            ff_time = link['free_flow_travel_time_steps']

            for dst in centroids:
                for t in range(ff_time, T):
                    # For these connectors, if dst == to_cent, then they can move through
                    # Otherwise if dest != to_cent, this flow should be 0
                    if dst != to_cent:
                        m.addConstr(y.get((cid, dst, t), 0.0) == 0.0)
                    else:
                        # These are handled similarly to road flows
                        sending_flow = N_U[(cid, dst, t - ff_time)] - N_D[(cid, dst, t - 1)]
                        m.addConstr(y.get((cid, dst, t), 0.0) <= sending_flow + 1e-9) 

    # Pick-up constraints: can't pick up more than current waiting or available departing vehicles
    for (o, d) in od_pairs_list:
        for t in range(T):
            # Can't pick up more than currently waiting
            m.addConstr(e[(o, d, t)] <= omega[(o, d, t)])
            
            # Can't pick up more than departing to that destination from origin centroid
            total_departing_o_to_d = gp.quicksum(
                y_centroid.get((o, node_to_id[id_to_link[out_lid]['to']], d, t), 0.0)
                for out_lid in out_links[o]  # all out connectors from origin
            )
            m.addConstr(e[(o, d, t)] <= total_departing_o_to_d)

    # Departing vehicles can't exceed currently parked
    for c_idx in centroids:
        for t in range(T):
            total_departing_from_c = gp.quicksum(
                y_centroid.get((c_idx, node_to_id[id_to_link[out_lid]['to']], dst, t), 0.0)
                for out_lid in out_links[c_idx]
                for dst in centroids
            )
            m.addConstr(total_departing_from_c <= p[(c_idx, t)])

    # OBJECTIVE
    # 1. Vehicle hours on links: sum_t (NU - ND) for all dest per each link
    vehicle_time_on_links = gp.quicksum(
        (N_U[(lid, dst, t)] - N_D[(lid, dst, t)])  # vehicles on link at time t
        for lid in road_links  # only count for road links
        for dst in centroids
        for t in range(T)
    )
    
    # Add contributions from centroid connectors if meaningful
    total_wait_time = gp.quicksum(
        omega[(o, d, t)]
        for o, d in od_pairs_list
        for t in range(T)
    )
    
    obj = vehicle_time_on_links + total_wait_time
    
    m.setObjective(obj, GRB.MINIMIZE)

    # Set up logging callback
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL and logger:
            objval = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            logger.log(objval)

    # Solve
    m.optimize(mycallback if logger else None)

    # Exit if model status is infeasible/unbounded
    if m.status in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
        print(f"Model infeasible or unbounded, status: {m.status}")
        return

    # Collect results
    result = {
        'objective_value': m.ObjVal if hasattr(m, 'ObjVal') else float('inf'),
        'y': {},
        'y_centroid': {},
        'N_U': {},
        'N_D': {},
        'p': {},
        'e': {},
        'omega': {}
    }

    # Extract solution values, skipping values that are too small
    EPS = 1e-6
    
    # y variables for junction links
    for (out_lid, dst, t), var in y.items():
        if (out_lid, dst, t) in var.getAttr('X'):
            val = var.X
        else:
            val = 0.0
        if val > EPS:
            link_info = id_to_link[out_lid]
            from_node = node_to_id[link_info['from']]
            to_node = node_to_id[link_info['to']]
            key = f"{id_to_node[from_node]}|{id_to_node[to_node]}|{id_to_node[dst]}|{t}"
            result['y'][key] = val

    # y_centroid
    for (c_idx, to_node_id, dst, t), var in y_centroid.items():
        val = var.X
        if val > EPS:
            key = f"{id_to_node[c_idx]}|{id_to_node[to_node_id]}|{id_to_node[dst]}|{t}"
            result['y_centroid'][key] = val

    for lid in range(n_links):
        link_info = id_to_link[lid]
        from_node = node_to_id[link_info['from']]
        to_node = node_to_id[link_info['to']]
        for dst in centroids:
            # Only store values up to T as per schema
            for t in range(T):
                val = N_U[(lid, dst, t)].X
                if val > EPS:
                    key = f"{id_to_node[from_node]}|{id_to_node[to_node]}|{id_to_node[dst]}|{t}"
                    result['N_U'][key] = val
                
                val = N_D[(lid, dst, t)].X
                if val > EPS:
                    key = f"{id_to_node[from_node]}|{id_to_node[to_node]}|{id_to_node[dst]}|{t}"
                    result['N_D'][key] = val

    for cent_idx in centroids:
        for t in range(T + 1):
            val = p[(cent_idx, t)].X
            if val > EPS:
                key = f"{id_to_node[cent_idx]}|{t}"
                result['p'][key] = val

    for (orig, dest, t), var in e.items():
        val = var.X
        if val > EPS:
            key = f"{id_to_node[orig]}|{id_to_node[dest]}|{t}"
            result['e'][key] = val

    for (orig, dest, t), var in omega.items():
        val = var.X
        if val > EPS:
            key = f"{id_to_node[orig]}|{id_to_node[dest]}|{t}"
            result['omega'][key] = val

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(result, f)


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